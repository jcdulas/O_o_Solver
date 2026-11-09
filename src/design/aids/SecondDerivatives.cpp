#include "oos/design/aids/SecondDerivatives.hpp"

#include "../sensitivity/SensitivityComputation.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

// The stencil sides of one direction: +h, −h, +h/2, −h/2.
constexpr std::array<double, 4> SIDE_SCALES{1.0, -1.0, 0.5, -0.5};
// An element item is linear in u (affine under a thermal strain) when r(2u) − 2·r(u) + r(0) vanishes to this relative
// size: the rounding of three recoveries, far below the curvature of a von Mises stress or a strain energy.
constexpr double AFFINE_TOLERANCE = 1.0e-8;
// The relative size of the displacement step ε·du of ∇_w L·du (exact for an item linear in u, whatever the step): as
// the large step of the element items' weights (detail::LINEAR_STEP_RELATIVE).
constexpr double DIRECTION_STEP_RELATIVE = 1.0e-3;
// The element matrices of a design point grow with the reached elements: their memory is taken from the budget by
// chunks of at least this size before they are copied (one reservation per element would lock the budget per element).
constexpr std::uint64_t POINT_RESERVATION_CHUNK = std::uint64_t{1} << 20;
// The working vectors of the g set a step makes besides those it keeps (a load and a solve's result).
constexpr std::uint64_t WORKING_VECTORS = 2;

// The step of x_j: SECOND_DERIVATIVE_STEP_RELATIVE of |x_j|, FIELD_STEP_AT_ZERO at zero.
double stepOf(double value) noexcept
{
    return value != 0.0 ? SECOND_DERIVATIVE_STEP_RELATIVE * std::abs(value) : FIELD_STEP_AT_ZERO;
}

double largestMagnitudeOf(std::span<const double> values) noexcept
{
    double largest = 0.0;
    for (const double value : values)
        largest = std::max(largest, std::abs(value));
    return largest;
}

// What an item is to the Lagrangian: its kind and its index in the lists of the linearisation.
struct ItemRole
{
    enum class Kind : std::uint8_t
    {
        Displacement, // r = u_dof: no explicit dependence on x
        Element,      // an element item (a job and its item)
        SpcForce,
        Mass,
        SelfAdjoint,
        Deflection
    };
    Kind kind = Kind::Displacement;
    std::size_t index = 0; // the job, SPC item, mass item, self-adjoint item or point item
    std::size_t entry = 0; // the item within its job
};

// One requested item: its key, subcase and adjoint vector.
struct HessianItem
{
    ItemKey key;
    std::size_t subcase = 0; // the index of its subcase in the linearisation (none for a global item)
    bool isGlobal = false;   // WEIGHT, VOLUME, FRMASS: no displacement, no adjoint
    ItemRole role;
    std::vector<double> adjoint; // λ on the g set: bounded by the requested items
};

// The element computations of one design point x' (the stiffness of the elements reached, the loads that change).
struct DesignPointState
{
    std::shared_ptr<const input::ModelSnapshot> snapshot;
    std::unique_ptr<elements::SnapshotIndex> index;
    std::unique_ptr<assembly::ElementSet> set;
    std::vector<std::vector<base::DofIndex>> dofs; // per element reached
    std::vector<std::vector<double>> matrices;
    std::vector<std::vector<double>> loads; // per subcase: the loads that change (dense g), empty when none
    std::unique_ptr<LinearityIndicator> linearity;
    std::map<std::size_t, double> massValues;          // per mass item
    std::vector<base::MemoryReservation> reservations; // the memory of the matrices and loads above
    std::uint64_t reservedBytes = 0;
    std::uint64_t usedBytes = 0;
};

class SecondComputation : public detail::SensitivityComputation
{
public:
    SecondComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                      const DesignProblem& problem, base::MemoryBudget& budget)
        : detail::SensitivityComputation(inputs, options, budget), m_problem(problem)
    {
    }

    // The linearisation, the items of the responses asked for with their adjoint vectors, the elements reached.
    void prepare(const std::vector<std::int32_t>& responseIds)
    {
        if (hasExternalUnits())
            throw base::OptimisationError(
                "second derivatives: the analysis has an external unit, whose second derivatives its plugin does not "
                "give in this version; ask for them without the external unit");
        for (std::size_t r = 0; r < m_in.model->relations().size(); ++r)
            if (const DesignRelation& relation = m_in.model->relations()[r]; relation.form == RelationForm::Supplied)
                throw base::OptimisationError(base::concatenate(
                    {"second derivatives: the relation ", relation.label(),
                     " is given by a function of the caller, whose values at the design points of the differences this "
                     "version does not ask for; give it by terms= or expression="}));
        linearise();
        for (const detail::SubcaseState& state : m_subcases)
            if (state.isCombination())
                throw base::OptimisationError(base::concatenate(
                    {"second derivatives of SubcaseCombination ", std::to_string(state.subcase.label),
                     ": this aid forms the loads of each subcase at the moved points, which a "
                     "SubcaseCombination has none of "
                     "(its loads are its terms'); ask for the product with weights, hessian_vector(x, v, "
                     "weights=...), which takes a SubcaseCombination"}));
        rolesOfItems();
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
        {
            const ResponseValues& values = valuesAt(p);
            if (!responseIds.empty() &&
                std::find(responseIds.begin(), responseIds.end(), values.id) == responseIds.end())
                continue;
            if (m_requested.count(values.id) == 0)
                continue;
            for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
                m_items.push_back(itemOf({p, i}));
        }
        std::uint64_t adjointCount = 0;
        for (const HessianItem& item : m_items)
            adjointCount += m_linearised.at(item.key).readsDisplacements() ? 1 : 0;
        reserveVectors(adjointCount + WORKING_VECTORS, "second derivatives adjoint vectors");
        adjoints();
        reachedElements();
    }

    SecondDerivativesResult hessians(const std::vector<std::size_t>& variables)
    {
        const std::size_t n = variables.size();
        const std::vector<double> x0 = m_in.design == nullptr ? std::vector<double>{} : xOf();
        reserveVectors(n * m_subcases.size() + WORKING_VECTORS, "second derivatives displacement derivatives");
        directSolves(variables);
        const std::size_t count = m_items.size();
        // The values of the Lagrangian at the points: center, the diagonal points (with ∇_w L·du_k), the pairs.
        DesignPointState center = makePoint(x0);
        std::vector<double> atCenter(count);
        for (std::size_t i = 0; i < count; ++i)
            atCenter[i] = lagrangian(center, m_items[i], displacementsOf(m_items[i]));
        center = {};
        std::vector<double> steps(n);
        std::vector<std::array<double, 2>> widths(n); // the stored half-widths at h and h/2
        // [j][side][i] and [j][side][k][i]
        std::vector<std::array<std::vector<double>, 4>> diagonal(n);
        std::vector<std::array<std::vector<std::vector<double>>, 4>> directional(n);
        for (std::size_t a = 0; a < n; ++a)
        {
            const std::size_t j = variables[a];
            steps[a] = stepOf(x0[j]);
            std::array<double, 4> moved{};
            for (std::size_t side = 0; side < 4; ++side)
            {
                std::vector<double> x = x0;
                x[j] = x0[j] + SIDE_SCALES[side] * steps[a];
                moved[side] = x[j];
                DesignPointState point = makePoint(x);
                diagonal[a][side].resize(count);
                directional[a][side].assign(n, std::vector<double>(count));
                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::span<const double> u = displacementsOf(m_items[i]);
                    diagonal[a][side][i] = lagrangian(point, m_items[i], u);
                    if (m_items[i].isGlobal)
                        continue; // no displacement: ∇_w L = 0
                    for (std::size_t b = 0; b < n; ++b)
                        directional[a][side][b][i] = alongDirection(point, m_items[i], u, diagonal[a][side][i],
                                                                    m_directions[b][m_items[i].subcase]);
                }
            }
            widths[a] = {(moved[0] - moved[1]) / 2.0, (moved[2] - moved[3]) / 2.0};
        }
        SecondDerivativesResult result;
        result.variables = variables;
        std::vector<double> hessian(count * n * n, 0.0);
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t i = 0; i < count; ++i)
            {
                const double coarse =
                    (diagonal[a][0][i] - 2.0 * atCenter[i] + diagonal[a][1][i]) / (widths[a][0] * widths[a][0]);
                const double fine =
                    (diagonal[a][2][i] - 2.0 * atCenter[i] + diagonal[a][3][i]) / (widths[a][1] * widths[a][1]);
                hessian[(i * n + a) * n + a] += richardson(coarse, fine);
            }
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = a + 1; b < n; ++b)
            {
                // L(++) − L(+−) − L(−+) + L(−−) over 4·h_a·h_b, at h and h/2: per item, [scale][corner].
                std::vector<std::array<std::array<double, 4>, 2>> values(count);
                std::array<std::array<double, 2>, 2> pairWidths{};
                for (std::size_t scale = 0; scale < 2; ++scale)
                {
                    const double factor = scale == 0 ? 1.0 : 0.5;
                    std::array<double, 2> movedA{};
                    std::array<double, 2> movedB{};
                    for (std::size_t corner = 0; corner < 4; ++corner)
                    {
                        const double signA = corner < 2 ? 1.0 : -1.0;
                        const double signB = corner % 2 == 0 ? 1.0 : -1.0;
                        std::vector<double> x = x0;
                        x[variables[a]] += signA * factor * steps[a];
                        x[variables[b]] += signB * factor * steps[b];
                        movedA[corner < 2 ? 0 : 1] = x[variables[a]];
                        movedB[corner % 2 == 0 ? 0 : 1] = x[variables[b]];
                        DesignPointState point = makePoint(x);
                        for (std::size_t i = 0; i < count; ++i)
                            values[i][scale][corner] = lagrangian(point, m_items[i], displacementsOf(m_items[i]));
                    }
                    pairWidths[scale] = {(movedA[0] - movedA[1]) / 2.0, (movedB[0] - movedB[1]) / 2.0};
                }
                for (std::size_t i = 0; i < count; ++i)
                {
                    std::array<double, 2> differences{};
                    for (std::size_t scale = 0; scale < 2; ++scale)
                    {
                        const std::array<double, 4>& c = values[i][scale];
                        differences[scale] =
                            (c[0] - c[1] - c[2] + c[3]) / (4.0 * pairWidths[scale][0] * pairWidths[scale][1]);
                    }
                    const double mixed = richardson(differences[0], differences[1]);
                    hessian[(i * n + a) * n + b] += mixed;
                    hessian[(i * n + b) * n + a] += mixed;
                }
            }
        // ∂_j(∇_w L·du_k) + ∂_k(∇_w L·du_j).
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = 0; b < n; ++b)
                for (std::size_t i = 0; i < count; ++i)
                {
                    const double coarse = (directional[a][0][b][i] - directional[a][1][b][i]) / (2.0 * widths[a][0]);
                    const double fine = (directional[a][2][b][i] - directional[a][3][b][i]) / (2.0 * widths[a][1]);
                    const double rate = richardson(coarse, fine); // ∂_a(∇_w L·du_b)
                    hessian[(i * n + a) * n + b] += rate;
                    hessian[(i * n + b) * n + a] += rate;
                }
        // Per response, in the order of the evaluation.
        std::size_t first = 0;
        while (first < count)
        {
            const std::size_t p = m_items[first].key.response;
            std::size_t last = first;
            while (last < count && m_items[last].key.response == p)
                ++last;
            ResponseHessian response;
            response.id = valuesAt(p).id;
            response.itemCount = last - first;
            response.values.assign(hessian.begin() + static_cast<std::ptrdiff_t>(first * n * n),
                                   hessian.begin() + static_cast<std::ptrdiff_t>(last * n * n));
            result.responses.push_back(std::move(response));
            first = last;
        }
        result.solves = m_solves;
        result.designPoints = m_designPoints;
        return result;
    }

    HessianVectorResult hessianVector(std::span<const double> direction)
    {
        const std::size_t n = m_variableCount;
        if (direction.size() != n)
            throw std::invalid_argument(base::concatenate(
                {"hessian_vector: expected a direction of ", std::to_string(n),
                 " values (one per independent variable), found ", std::to_string(direction.size())}));
        const std::vector<double> x0 = xOf();
        const std::size_t count = m_items.size();
        HessianVectorResult result;
        result.itemCount = count;
        result.values.assign(count * n, 0.0);
        if (count == 0)
            return result;
        result.id = valuesAt(m_items.front().key.response).id;
        // du_v per subcase and the weights of the Lagrangian at the four points along v, per item.
        reserveVectors(m_subcases.size() + 4 * count + WORKING_VECTORS, "second derivatives Hessian-vector vectors");
        // du_v = K⁻¹·Σ_k v_k·b_k per subcase (one solve each).
        const std::vector<std::vector<double>> alongV = directSolveOf(direction);
        // The step along v: the relative step of the largest variable, over |v|.
        const double vNorm = largestMagnitudeOf(direction);
        if (vNorm == 0.0)
            return result;
        double xScale = largestMagnitudeOf(x0);
        const double t = (xScale != 0.0 ? SECOND_DERIVATIVE_STEP_RELATIVE * xScale : FIELD_STEP_AT_ZERO) / vNorm;
        const auto pointAt = [&](std::size_t j, double sj, double sv)
        {
            std::vector<double> x = x0;
            for (std::size_t k = 0; k < n; ++k)
                x[k] += sv * direction[k];
            if (j < n)
                x[j] += sj;
            return x;
        };
        for (std::size_t j = 0; j < n; ++j)
        {
            const double h = stepOf(x0[j]);
            // Σ_k ∂²L/∂x_j∂x_k·v_k: the mixed stencil of e_j and v, at (h, t) and (h/2, t/2).
            std::vector<std::array<double, 2>> perItem(count);
            for (std::size_t scale = 0; scale < 2; ++scale)
            {
                const double factor = scale == 0 ? 1.0 : 0.5;
                std::array<std::vector<double>, 4> corners;
                for (std::size_t corner = 0; corner < 4; ++corner)
                {
                    const double signJ = corner < 2 ? 1.0 : -1.0;
                    const double signV = corner % 2 == 0 ? 1.0 : -1.0;
                    DesignPointState point = makePoint(pointAt(j, signJ * factor * h, signV * factor * t));
                    corners[corner].resize(count);
                    for (std::size_t i = 0; i < count; ++i)
                        corners[corner][i] = lagrangian(point, m_items[i], displacementsOf(m_items[i]));
                }
                for (std::size_t i = 0; i < count; ++i)
                    perItem[i][scale] = (corners[0][i] - corners[1][i] - corners[2][i] + corners[3][i]) /
                                        (4.0 * factor * h * factor * t);
            }
            // ∂_j(∇_w L·du_v).
            std::array<std::vector<double>, 4> along;
            std::array<double, 4> moved{};
            for (std::size_t side = 0; side < 4; ++side)
            {
                const std::vector<double> x = pointAt(j, SIDE_SCALES[side] * h, 0.0);
                moved[side] = x[j];
                DesignPointState point = makePoint(x);
                along[side].assign(count, 0.0);
                for (std::size_t i = 0; i < count; ++i)
                {
                    if (m_items[i].isGlobal)
                        continue;
                    const std::span<const double> u = displacementsOf(m_items[i]);
                    const double base = lagrangian(point, m_items[i], u);
                    along[side][i] = alongDirection(point, m_items[i], u, base, alongV[m_items[i].subcase]);
                }
            }
            const double halfWidth = (moved[0] - moved[1]) / 2.0;
            const double halfWidthFine = (moved[2] - moved[3]) / 2.0;
            for (std::size_t i = 0; i < count; ++i)
            {
                const double mixed = richardson(perItem[i][0], perItem[i][1]);
                const double rate = richardson((along[0][i] - along[1][i]) / (2.0 * halfWidth),
                                               (along[2][i] - along[3][i]) / (2.0 * halfWidthFine));
                result.values[i * n + j] += mixed + rate;
            }
        }
        // μᵀ·b_j, μ = K⁻¹·∂_v(∇_w L): the weights of the Lagrangian at x ± t·v, differenced.
        std::array<std::vector<std::vector<double>>, 4> weights; // [side][item] on the g set
        for (std::size_t side = 0; side < 4; ++side)
        {
            DesignPointState point = makePoint(pointAt(n, 0.0, SIDE_SCALES[side] * t));
            for (std::size_t i = 0; i < count; ++i)
                weights[side].push_back(lagrangianWeights(point, m_items[i]));
        }
        std::vector<double> products(n);
        for (std::size_t i = 0; i < count; ++i)
        {
            if (m_items[i].isGlobal)
                continue; // no displacement: μ = 0
            std::vector<double> z(gSize());
            for (std::size_t g = 0; g < gSize(); ++g)
                z[g] = richardson((weights[0][i][g] - weights[1][i][g]) / (2.0 * t),
                                  (weights[2][i][g] - weights[3][i][g]) / t);
            if (largestMagnitudeOf(z) == 0.0)
                continue;
            const std::vector<double> mu = solveOne(m_items[i].subcase, z);
            m_pseudoLoads->dotAll(m_items[i].subcase, mu, products);
            for (std::size_t j = 0; j < n; ++j)
                result.values[i * n + j] += products[j];
        }
        result.solves = m_solves;
        result.designPoints = m_designPoints;
        return result;
    }

private:
    [[nodiscard]] static double richardson(double coarse, double fine) noexcept { return (4.0 * fine - coarse) / 3.0; }

    // Takes `count` vectors of the g set from the memory budget before they are made: they grow with the model.
    void reserveVectors(std::uint64_t count, const std::string& label)
    {
        m_reservations.push_back(m_budget.reserve(count * static_cast<std::uint64_t>(gSize()) * sizeof(double), label));
    }

    // Takes `bytes` of a design point's matrices and loads from the budget, by chunks (POINT_RESERVATION_CHUNK).
    void reserveForPoint(DesignPointState& state, std::uint64_t bytes)
    {
        if (state.usedBytes + bytes > state.reservedBytes)
        {
            const std::uint64_t chunk = std::max(bytes, POINT_RESERVATION_CHUNK);
            state.reservations.push_back(m_budget.reserve(chunk, "second derivatives design point"));
            state.reservedBytes += chunk;
        }
        state.usedBytes += bytes;
    }

    [[nodiscard]] std::vector<double> xOf() const
    {
        // x of the design point: the variables of the design values at the independent indices.
        std::vector<double> x;
        for (const std::size_t v : m_in.model->independentIndices())
            x.push_back(m_in.design->variables.values[v]);
        return x;
    }

    void rolesOfItems()
    {
        for (std::size_t j = 0; j < m_jobs.size(); ++j)
            for (std::size_t k = 0; k < m_jobs[j].keys.size(); ++k)
                m_roles[m_jobs[j].keys[k]] = {ItemRole::Kind::Element, j, k};
        for (std::size_t s = 0; s < m_spcItems.size(); ++s)
            m_roles[m_spcItems[s].key] = {ItemRole::Kind::SpcForce, s, 0};
        for (std::size_t m = 0; m < m_massItems.size(); ++m)
            m_roles[m_massItems[m].key] = {ItemRole::Kind::Mass, m, 0};
        for (std::size_t a = 0; a < m_selfAdjointItems.size(); ++a)
            m_roles[m_selfAdjointItems[a].key] = {ItemRole::Kind::SelfAdjoint, a, 0};
        for (std::size_t q = 0; q < m_pointItems.size(); ++q)
            m_roles[m_pointItems[q].key] = {ItemRole::Kind::Deflection, q, 0};
    }

    [[nodiscard]] HessianItem itemOf(const ItemKey& key)
    {
        const ResponseValues& values = valuesAt(key.response);
        const ItemLinearisation& item = m_linearised.at(key);
        const auto refuse = [&](const std::string& why)
        {
            throw base::OptimisationError(
                base::concatenate({"second derivatives of response ", std::to_string(values.id), " (", values.label,
                                   "), item ", std::to_string(key.item), ": ", why,
                                   "; leave it out of responses= of DesignProblem.second_derivatives"}));
        };
        if (item.isEquation)
            refuse(
                "an equation response (DesignResp2, DesignResp3): its second derivatives are not formed in this "
                "version");
        HessianItem result;
        result.key = key;
        result.isGlobal = item.subcase == 0;
        result.subcase = result.isGlobal ? 0 : m_subcaseIndex.at(item.subcase);
        if (const auto role = m_roles.find(key); role != m_roles.end())
            result.role = role->second;
        else if (const DesignResp1Response* response = m_in.responses->findDesignResp1(values.id);
                 response == nullptr || response->definition().type != ResponseType::Displacement)
            refuse(response != nullptr && response->definition().type == ResponseType::Rotation
                       ? "a rotation magnitude is not linear in the displacements"
                       : "its kind has no second derivatives in this version");
        if (result.role.kind == ItemRole::Kind::Element)
            checkAffine(result.role, refuse);
        if (result.role.kind == ItemRole::Kind::SelfAdjoint &&
            !m_selfAdjointItems[result.role.index].forceWeights.empty())
            for (const std::size_t q : changingQuantities())
                if (!m_pseudoLoads->appliedLoadPartOf(q, result.subcase).empty())
                    refuse("a mean displacement whose applied forces change with the design");
        return result;
    }

    template <typename Refuse>
    void checkAffine(const ItemRole& role, const Refuse& refuse)
    {
        const detail::ElementJob& job = m_jobs[role.index];
        const detail::SubcaseState& state = m_subcases[job.subcase];
        const elements::ElementBatchView& view = m_set.batches()[job.element.batch].view;
        std::vector<double> zero(gSize(), 0.0);
        std::vector<double> twice(gSize());
        for (std::size_t g = 0; g < gSize(); ++g)
            twice[g] = 2.0 * state.displacements[g];
        const double atZero = job.function->values(view, m_in.overlaid, nodalOf(zero, state), m_budget)[role.entry];
        const double atU = job.baseValues[role.entry];
        const double atTwice = job.function->values(view, m_in.overlaid, nodalOf(twice, state), m_budget)[role.entry];
        const double scale = std::max({std::abs(atZero), std::abs(atU), std::abs(atTwice)});
        if (std::abs(atTwice - 2.0 * atU + atZero) > AFFINE_TOLERANCE * scale)
            refuse(
                "it is not linear in the displacements (von Mises, principal stresses, a strain energy, a margin): "
                "its second derivative with respect to u is not formed in this version");
    }

    [[nodiscard]] std::vector<std::size_t> changingQuantities() const
    {
        std::vector<std::size_t> quantities;
        for (std::size_t q = 0; q < m_quantities.size(); ++q)
            if (!m_quantities[q].gradient.empty())
                quantities.push_back(q);
        return quantities;
    }

    [[nodiscard]] elements::NodalDisplacements nodalOf(std::span<const double> w,
                                                       const detail::SubcaseState& state) const
    {
        const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap.gridCount());
        return {w.first(gridValues), w.subspan(gridValues),
                state.temperatures ? &state.temperatures->temperatures() : nullptr};
    }

    [[nodiscard]] std::span<const double> displacementsOf(const HessianItem& item) const
    {
        return m_subcases.empty() ? std::span<const double>{} : m_subcases[item.subcase].displacements.span();
    }

    // λ of each item: scale·u for a self-adjoint one, solved for the others (one solve each, by blocks).
    void adjoints()
    {
        for (HessianItem& item : m_items)
        {
            const ItemLinearisation& linear = m_linearised.at(item.key);
            if (!linear.readsDisplacements())
                continue;
            if (item.role.kind == ItemRole::Kind::SelfAdjoint)
            {
                const double scale = m_selfAdjointItems[item.role.index].scale;
                const std::span<const double> u = m_subcases[item.subcase].displacements.span();
                item.adjoint.assign(u.begin(), u.end());
                for (double& value : item.adjoint)
                    value *= scale;
                continue;
            }
            std::vector<double> weights(gSize(), 0.0);
            for (const auto& [dof, weight] : linear.weights)
                weights[static_cast<std::size_t>(dof)] = weight;
            item.adjoint = solveOne(item.subcase, weights);
        }
    }

    // K⁻¹·v on the g set, through the l set of the subcase's system (one solve).
    [[nodiscard]] std::vector<double> solveOne(std::size_t subcase, std::span<const double> vectorG)
    {
        const SensitivitySystem& system = systemOf(m_subcases[subcase]);
        const auto equations = static_cast<std::size_t>(system.equationCount());
        linalg::DenseMatrix rightHandSide(m_budget, equations, 1, "second derivatives right-hand side");
        std::vector<double> column(equations);
        system.toEquations(vectorG, column);
        for (std::size_t e = 0; e < equations; ++e)
            rightHandSide(e, 0) = column[e];
        if (equations > 0)
            system.solve(rightHandSide.view());
        ++m_solves;
        for (std::size_t e = 0; e < equations; ++e)
            column[e] = rightHandSide(e, 0);
        std::vector<double> result(gSize());
        system.fromEquations(column, result);
        return result;
    }

    // du_k = K⁻¹·b_k of each variable asked for, per subcase (the direct method: one solve per variable and subcase).
    void directSolves(const std::vector<std::size_t>& variables)
    {
        std::set<std::size_t> subcases;
        for (const HessianItem& item : m_items)
            if (m_linearised.at(item.key).subcase != 0)
                subcases.insert(item.subcase);
        m_directions.assign(variables.size(), std::vector<std::vector<double>>(m_subcases.size()));
        for (std::size_t b = 0; b < variables.size(); ++b)
            for (const std::size_t s : subcases)
            {
                std::vector<double> load(gSize());
                m_pseudoLoads->assemble(variables[b], s, load);
                m_directions[b][s] =
                    largestMagnitudeOf(load) == 0.0 ? std::vector<double>(gSize(), 0.0) : solveOne(s, load);
            }
    }

    // du_v = K⁻¹·Σ_k v_k·b_k per subcase of the items.
    [[nodiscard]] std::vector<std::vector<double>> directSolveOf(std::span<const double> direction)
    {
        std::set<std::size_t> subcases;
        for (const HessianItem& item : m_items)
            if (m_linearised.at(item.key).subcase != 0)
                subcases.insert(item.subcase);
        std::vector<std::vector<double>> result(m_subcases.size());
        for (const std::size_t s : subcases)
        {
            std::vector<double> load(gSize(), 0.0);
            std::vector<double> one(gSize());
            for (std::size_t k = 0; k < direction.size(); ++k)
            {
                if (direction[k] == 0.0)
                    continue;
                m_pseudoLoads->assemble(k, s, one);
                for (std::size_t g = 0; g < gSize(); ++g)
                    load[g] += direction[k] * one[g];
            }
            result[s] = largestMagnitudeOf(load) == 0.0 ? std::vector<double>(gSize(), 0.0) : solveOne(s, load);
        }
        return result;
    }

    // The elements any designed quantity reaches (the fields with a gradient, the shapes): only they change between
    // the design points.
    void reachedElements()
    {
        std::set<ElementRef> reached;
        const AffectedElements affected(*m_in.model, m_set);
        for (std::size_t r = 0; r < m_in.design->fields.size(); ++r)
            if (!m_in.design->fields[r].gradient.empty())
                reached.insert(affected.of(r).begin(), affected.of(r).end());
        if (!m_in.design->shapes.empty())
        {
            const ShapeElements shapes(*m_in.model, *m_in.targets, m_set);
            for (std::size_t b = 0; b < m_in.design->shapes.size(); ++b)
                if (!m_in.design->shapes[b].gradient.empty())
                    reached.insert(shapes.of(b).begin(), shapes.of(b).end());
        }
        m_reached.assign(reached.begin(), reached.end());
    }

    [[nodiscard]] DesignPointState makePoint(const std::vector<double>& x)
    {
        DesignPointState state;
        const DesignPoint point = m_problem.designPoint(x);
        state.snapshot = m_problem.overlaySnapshot(point);
        state.index = std::make_unique<elements::SnapshotIndex>(*state.snapshot, m_budget);
        state.set = std::make_unique<assembly::ElementSet>(*state.index, m_registry);
        ++m_designPoints;
        elements::ElementMatrixBlock block;
        for (const ElementRef& element : m_reached)
        {
            const assembly::AssemblyBatch& batch = state.set->batches()[element.batch];
            std::vector<base::DofIndex> dofs;
            std::vector<double> matrix;
            if (batch.family->facets().has(elements::Facet::Stiffness))
            {
                block.clear();
                batch.family->computeStiffness(batch.view, {element.row, element.row + 1}, block);
                if (!block.empty())
                {
                    reserveForPoint(
                        state, block.dofs(0).size() * sizeof(base::DofIndex) + block.matrix(0).size() * sizeof(double));
                    for (const elements::ElementDof& dof : block.dofs(0))
                        dofs.push_back(m_dofMap.dofOf(dof));
                    matrix.assign(block.matrix(0).begin(), block.matrix(0).end());
                }
            }
            state.dofs.push_back(std::move(dofs));
            state.matrices.push_back(std::move(matrix));
        }
        state.loads.resize(m_subcases.size());
        const input::ParamTable* params =
            state.snapshot->hasTable(input::TableRole::Params) ? &state.snapshot->params() : nullptr;
        // The concentrated loads change with x when a shape turns one (DerivedGeometry).
        bool turnsLoads = false;
        for (std::size_t b = 0; b < m_in.design->shapes.size(); ++b)
            turnsLoads = turnsLoads || m_in.targets->derived().changesConcentratedLoads(b);
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const detail::SubcaseState& subcase = m_subcases[s];
            const bool hasThermal = subcase.temperatures != nullptr;
            const bool hasTurnedLoads = turnsLoads && subcase.hasConcentratedLoads;
            if (!hasThermal && !subcase.hasDistributedLoads && !subcase.hasInertialLoads && !hasTurnedLoads)
                continue;
            std::vector<double>& loads = state.loads[s];
            reserveForPoint(state, static_cast<std::uint64_t>(gSize()) * sizeof(double));
            loads.assign(gSize(), 0.0);
            if (hasThermal)
                for (const ElementRef& element : m_reached)
                {
                    const assembly::AssemblyBatch& batch = state.set->batches()[element.batch];
                    if (batch.family->isThermallyInert())
                        continue;
                    elements::ElementVectorBlock vectors;
                    batch.family->computeThermalLoads(batch.view, {element.row, element.row + 1},
                                                      subcase.temperatures->temperatures(), vectors);
                    for (std::size_t v = 0; v < vectors.size(); ++v)
                        for (std::size_t d = 0; d < vectors.dofs(v).size(); ++d)
                            loads[static_cast<std::size_t>(m_dofMap.dofOf(vectors.dofs(v)[d]))] += vectors.values(v)[d];
                }
            base::DiagnosticSink sink;
            // The point's own positions (the radius of an RotationalForceLoad, the location of an ACCEL) and tables.
            const assembly::DofMap pointMap(state.snapshot->nodes(), state.index->points());
            const assembly::LoadVectorBuilder builder(*state.set, pointMap, m_budget);
            if (subcase.hasDistributedLoads)
                builder.addFamilyLoads(*subcase.subcase.load, assembly::LoadFamily::Distributed, nullptr, loads,
                                       m_budget, sink);
            if (hasTurnedLoads)
                builder.addFamilyLoads(*subcase.subcase.load, assembly::LoadFamily::Concentrated, nullptr, loads,
                                       m_budget, sink);
            if (subcase.hasInertialLoads)
            {
                const assembly::MassAssemblyOptions options =
                    assembly::MassAssemblyOptions::fromParams(params, subcase.subcase.label, sink);
                linalg::BlockSparseMatrix mass(systemOf(subcase).pattern(), m_budget, "second derivatives mass");
                const assembly::NonStructuralMassShares shares(*state.set);
                assembly::MassAssembler(*state.set, m_dofMap, shares, options).assemble(mass, m_in.pool);
                builder.addFamilyLoads(*subcase.subcase.load, assembly::LoadFamily::Inertial, &mass, loads, m_budget,
                                       sink);
            }
        }
        return state;
    }

    // Σ_e aᵀ·K_e(x')·b over the elements reached.
    [[nodiscard]] static double bilinear(const DesignPointState& point, std::span<const double> a,
                                         std::span<const double> b)
    {
        double sum = 0.0;
        for (std::size_t e = 0; e < point.dofs.size(); ++e)
        {
            const std::vector<base::DofIndex>& dofs = point.dofs[e];
            const std::vector<double>& matrix = point.matrices[e];
            const std::size_t n = dofs.size();
            for (std::size_t r = 0; r < n; ++r)
            {
                const double left = a[static_cast<std::size_t>(dofs[r])];
                if (left == 0.0)
                    continue;
                double row = 0.0;
                for (std::size_t c = 0; c < n; ++c)
                    row += matrix[r * n + c] * b[static_cast<std::size_t>(dofs[c])];
                sum += left * row;
            }
        }
        return sum;
    }

    // Σ_e K_e(x')·b over the elements reached, into `out` (added).
    static void stiffnessProduct(const DesignPointState& point, std::span<const double> b, std::span<double> out)
    {
        for (std::size_t e = 0; e < point.dofs.size(); ++e)
        {
            const std::vector<base::DofIndex>& dofs = point.dofs[e];
            const std::vector<double>& matrix = point.matrices[e];
            const std::size_t n = dofs.size();
            for (std::size_t r = 0; r < n; ++r)
            {
                double row = 0.0;
                for (std::size_t c = 0; c < n; ++c)
                    row += matrix[r * n + c] * b[static_cast<std::size_t>(dofs[c])];
                out[static_cast<std::size_t>(dofs[r])] += row;
            }
        }
    }

    [[nodiscard]] const LinearityIndicator& linearityOf(DesignPointState& point)
    {
        if (!point.linearity)
            point.linearity = std::make_unique<LinearityIndicator>(*point.snapshot, point.snapshot, m_budget);
        return *point.linearity;
    }

    // The part of r(w, x') that depends on x' (the rest cancels in every difference).
    [[nodiscard]] double itemValue(DesignPointState& point, const HessianItem& item, std::span<const double> w)
    {
        const detail::SubcaseState* state = m_subcases.empty() ? nullptr : &m_subcases[item.subcase];
        switch (item.role.kind)
        {
        case ItemRole::Kind::Displacement:
            return 0.0;
        case ItemRole::Kind::Element:
        {
            detail::ElementJob& job = m_jobs[item.role.index];
            const elements::ElementBatchView& view = point.set->batches()[job.element.batch].view;
            return job.function->values(view, point.snapshot, nodalOf(w, *state), m_budget)[item.role.entry];
        }
        case ItemRole::Kind::SpcForce:
        {
            // wᵀ·(K_R(x')·u − P_R(x')): the SPC weights against the stiffness and loads that change.
            const detail::SpcItem& spc = m_spcItems[item.role.index];
            std::vector<double> weights(gSize(), 0.0);
            for (const auto& [dof, value] : spc.w)
                weights[static_cast<std::size_t>(dof)] = value;
            double value = bilinear(point, weights, w);
            if (!point.loads[item.subcase].empty())
                value -= dotSparse(spc.w, point.loads[item.subcase]);
            return value;
        }
        case ItemRole::Kind::Mass:
        {
            const auto found = point.massValues.find(item.role.index);
            if (found != point.massValues.end())
                return found->second;
            const double value = massValue(*point.snapshot, m_massItems[item.role.index]);
            point.massValues[item.role.index] = value;
            return value;
        }
        case ItemRole::Kind::SelfAdjoint:
        {
            const std::vector<double>& loads = point.loads[item.subcase];
            if (loads.empty())
                return 0.0;
            double sum = 0.0;
            for (std::size_t g = 0; g < gSize(); ++g)
                sum += loads[g] * w[g];
            return m_selfAdjointItems[item.role.index].scale * sum;
        }
        case ItemRole::Kind::Deflection:
        {
            const detail::PointItem& entry = m_pointItems[item.role.index];
            const ShellPointGeometry* geometry = linearityOf(point).findShellPoint(entry.grid);
            std::array<double, 3> translations{};
            for (int c = 0; c < 3; ++c)
                translations[static_cast<std::size_t>(c)] =
                    w[static_cast<std::size_t>(*m_dofMap.findDof(entry.grid, c + 1))];
            return LinearityIndicator::deflectionRatio(*geometry, translations);
        }
        }
        return 0.0;
    }

    [[nodiscard]] double massValue(const input::ModelSnapshot& snapshot, const detail::MassItem& item)
    {
        const ResponseType type = item.response->definition().type;
        if (type == ResponseType::Weight)
        {
            const auto [row, column] = item.response->weightEntry();
            return MassProperties::rigidBodyMass(snapshot, DesignResp1Response::weightReference(snapshot),
                                                 m_in.massFormulation, {}, m_budget)(
                static_cast<std::size_t>(row - 1), static_cast<std::size_t>(column - 1));
        }
        if (type == ResponseType::Volume)
            return MassProperties::volume(snapshot, m_in.overlaid, {}, m_budget);
        // FRMASS = m/m0, m0 = m / value at the design point.
        const double value = valuesAt(item.key.response).values[item.key.item];
        const double current = MassProperties::rigidBodyMass(*m_in.overlaid, {0.0, 0.0, 0.0}, m_in.massFormulation,
                                                             item.response->massFilter(), m_budget)(0, 0);
        const double mass = MassProperties::rigidBodyMass(snapshot, {0.0, 0.0, 0.0}, m_in.massFormulation,
                                                          item.response->massFilter(), m_budget)(0, 0);
        return current != 0.0 ? mass * value / current : 0.0;
    }

    // L(x', w) = r(w, x') − λᵀ·(K_R(x')·w − P_R(x')), the parts that depend on x'.
    [[nodiscard]] double lagrangian(DesignPointState& point, const HessianItem& item, std::span<const double> w)
    {
        double value = itemValue(point, item, w);
        if (!item.adjoint.empty())
        {
            value -= bilinear(point, item.adjoint, w);
            const std::vector<double>& loads = point.loads[item.subcase];
            if (!loads.empty())
                for (std::size_t g = 0; g < gSize(); ++g)
                    value += item.adjoint[g] * loads[g];
        }
        return value;
    }

    // ∇_w L·d at x': [L(x', u + ε·d) − L(x', u)]/ε, exact for an item linear in u.
    [[nodiscard]] double alongDirection(DesignPointState& point, const HessianItem& item, std::span<const double> u,
                                        double atU, const std::vector<double>& direction)
    {
        if (direction.empty())
            return 0.0;
        const double size = largestMagnitudeOf(direction);
        if (size == 0.0)
            return 0.0;
        const double reference = largestMagnitudeOf(u);
        const double epsilon = DIRECTION_STEP_RELATIVE * (reference > 0.0 ? reference : 1.0) / size;
        std::vector<double> moved(u.begin(), u.end());
        for (std::size_t g = 0; g < moved.size(); ++g)
            moved[g] += epsilon * direction[g];
        return (lagrangian(point, item, moved) - atU) / epsilon;
    }

    // ∇_w L at x' on the g set (the parts that depend on x'): the item's weights less K_R(x')·λ.
    [[nodiscard]] std::vector<double> lagrangianWeights(DesignPointState& point, const HessianItem& item)
    {
        std::vector<double> weights(gSize(), 0.0);
        const std::span<const double> u = displacementsOf(item);
        switch (item.role.kind)
        {
        case ItemRole::Kind::Displacement:
        case ItemRole::Kind::Mass:
            break;
        case ItemRole::Kind::Element:
        {
            // The weights over the element's DOFs, by central differences in u (exact: the item is linear in u).
            detail::ElementJob& job = m_jobs[item.role.index];
            const elements::ElementBatchView& view = point.set->batches()[job.element.batch].view;
            const double reference = largestMagnitudeOf(u);
            const double delta = DIRECTION_STEP_RELATIVE * (reference > 0.0 ? reference : 1.0);
            std::vector<double> moved(u.begin(), u.end());
            for (const elements::ElementDof& dof : job.function->dofs(m_set.batches()[job.element.batch].view))
            {
                const auto g = static_cast<std::size_t>(m_dofMap.dofOf(dof));
                moved[g] = u[g] + delta;
                const double up = job.function->values(view, point.snapshot, nodalOf(moved, m_subcases[item.subcase]),
                                                       m_budget)[item.role.entry];
                moved[g] = u[g] - delta;
                const double down = job.function->values(view, point.snapshot, nodalOf(moved, m_subcases[item.subcase]),
                                                         m_budget)[item.role.entry];
                moved[g] = u[g];
                weights[g] += (up - down) / (2.0 * delta);
            }
            break;
        }
        case ItemRole::Kind::SpcForce:
        {
            std::vector<double> spcWeights(gSize(), 0.0);
            for (const auto& [dof, value] : m_spcItems[item.role.index].w)
                spcWeights[static_cast<std::size_t>(dof)] = value;
            stiffnessProduct(point, spcWeights, weights);
            break;
        }
        case ItemRole::Kind::SelfAdjoint:
        {
            const std::vector<double>& loads = point.loads[item.subcase];
            const double scale = m_selfAdjointItems[item.role.index].scale;
            for (std::size_t g = 0; g < loads.size(); ++g)
                weights[g] += scale * loads[g];
            break;
        }
        case ItemRole::Kind::Deflection:
        {
            const detail::PointItem& entry = m_pointItems[item.role.index];
            const ShellPointGeometry* geometry = linearityOf(point).findShellPoint(entry.grid);
            const std::array<double, 3> local = LinearityIndicator::deflectionWeights(*geometry, entry.translations);
            for (int c = 0; c < 3; ++c)
                weights[static_cast<std::size_t>(*m_dofMap.findDof(entry.grid, c + 1))] +=
                    local[static_cast<std::size_t>(c)];
            break;
        }
        }
        if (!item.adjoint.empty())
        {
            std::vector<double> product(gSize(), 0.0);
            stiffnessProduct(point, item.adjoint, product);
            for (std::size_t g = 0; g < gSize(); ++g)
                weights[g] -= product[g];
        }
        return weights;
    }

    const DesignProblem& m_problem;
    std::map<ItemKey, ItemRole> m_roles;
    std::vector<HessianItem> m_items;
    std::vector<ElementRef> m_reached;
    std::vector<std::vector<std::vector<double>>> m_directions; // [variable asked][subcase] du on the g set
    std::vector<base::MemoryReservation> m_reservations;        // the memory of the vectors of the g set kept
    std::int64_t m_solves = 0;
    std::int64_t m_designPoints = 0;
};

GradientOptions optionsOf(const std::vector<std::int32_t>& responseIds)
{
    GradientOptions options;
    options.responseIds = responseIds;
    options.usesSelfAdjoint = false; // the computation takes λ = scale·u itself (its weights are kept)
    return options;
}

} // namespace

SecondDerivativesResult SecondDerivatives::compute(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                   const SecondDerivativeRequest& request,
                                                   const CallableRegistry& callables, base::MemoryBudget& budget,
                                                   const solutions::StaticOptions& analysisOptions,
                                                   std::size_t blockItems)
{
    const ResponseSubcases subcases = problem.responseSubcases(evaluation);
    const StaticSensitivityInputs inputs =
        problem.sensitivityInputs(evaluation, subcases, callables, analysisOptions, blockItems);
    const GradientOptions options = optionsOf(request.responseIds);
    SecondComputation computation(inputs, options, problem, budget);
    computation.prepare(request.responseIds);
    std::vector<std::size_t> variables = request.variables;
    if (variables.empty())
        for (std::size_t j = 0; j < problem.model().independentCount(); ++j)
            variables.push_back(j);
    for (const std::size_t j : variables)
        if (j >= problem.model().independentCount())
            throw std::invalid_argument(base::concatenate(
                {"second derivatives: expected variables below ", std::to_string(problem.model().independentCount()),
                 " (the independent design variables), found ", std::to_string(j)}));
    return computation.hessians(variables);
}

HessianVectorResult SecondDerivatives::hessianVector(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                     std::int32_t responseId, std::span<const double> direction,
                                                     const CallableRegistry& callables, base::MemoryBudget& budget,
                                                     const solutions::StaticOptions& analysisOptions,
                                                     std::size_t blockItems)
{
    const ResponseSubcases subcases = problem.responseSubcases(evaluation);
    const StaticSensitivityInputs inputs =
        problem.sensitivityInputs(evaluation, subcases, callables, analysisOptions, blockItems);
    const std::vector<std::int32_t> ids{responseId};
    const GradientOptions options = optionsOf(ids);
    SecondComputation computation(inputs, options, problem, budget);
    computation.prepare(ids);
    return computation.hessianVector(direction);
}

} // namespace oos::design
