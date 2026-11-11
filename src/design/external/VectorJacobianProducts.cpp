#include "oos/design/external/VectorJacobianProducts.hpp"

#include "../sensitivity/SensitivityComputation.hpp"
#include "HessianBlocks.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/design/responses/ResultLookup.hpp"
#include "oos/design/sensitivity/AdjointColumns.hpp"
#include "oos/design/sensitivity/SolutionReads.hpp"
#include "oos/elements/shells/LaminateRate.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace oos::design
{

namespace
{

// The weights of a vjp by linearised response (its position in the linearisation), dense or sparse: the
// non-zero weights of a row, items increasing, whatever the form.
class WeightRows
{
public:
    template <typename ValuesAt>
    WeightRows(const ProductWeights& weights, const LinearisedItems& linearised, ValuesAt valuesAt)
        : m_sources(linearised.items.size())
    {
        for (std::size_t p = 0; p < linearised.items.size(); ++p)
        {
            const ResponseValues& values = valuesAt(p);
            Source& source = m_sources[p];
            source.itemCount = values.itemCount();
            const auto dense = weights.weights.find(values.id);
            const auto sparse = weights.sparseWeights.find(values.id);
            if (dense != weights.weights.end() && sparse != weights.sparseWeights.end())
                throw std::invalid_argument(base::concatenate(
                    {"vjp: expected the weights of response ", std::to_string(values.id), " (", values.label,
                     ") in one form, found them dense and sparse; give one of the two"}));
            if (dense != weights.weights.end())
            {
                if (dense->second.size() != weights.rows * source.itemCount)
                    throw std::invalid_argument(base::concatenate(
                        {"vjp: expected the weights of response ", std::to_string(values.id), " (", values.label,
                         ") of shape (", std::to_string(weights.rows), ", ", std::to_string(source.itemCount),
                         "), found ", std::to_string(dense->second.size()), " values"}));
                source.dense = dense->second.data();
            }
            else if (sparse != weights.sparseWeights.end())
            {
                check(sparse->second, weights.rows, values);
                source.sparse = &sparse->second;
            }
        }
    }

    // Calls f(item, weight) for the non-zero weights of row `row` on the response at `p`, items increasing; returns the
    // weights read.
    template <typename Function>
    std::int64_t forEach(std::size_t p, std::size_t row, Function&& f) const
    {
        const Source& source = m_sources[p];
        if (source.dense != nullptr)
        {
            const double* weights = source.dense + row * source.itemCount;
            for (std::size_t i = 0; i < source.itemCount; ++i)
                if (weights[i] != 0.0)
                    f(i, weights[i]);
            return static_cast<std::int64_t>(source.itemCount);
        }
        if (source.sparse == nullptr)
            return 0;
        const SparseWeights& sparse = *source.sparse;
        for (std::size_t k = sparse.rowStart[row]; k < sparse.rowStart[row + 1]; ++k)
            if (sparse.values[k] != 0.0)
                f(sparse.items[k], sparse.values[k]);
        return static_cast<std::int64_t>(sparse.rowStart[row + 1] - sparse.rowStart[row]);
    }

private:
    struct Source
    {
        std::size_t itemCount = 0;
        const double* dense = nullptr;
        const SparseWeights* sparse = nullptr;
    };

    static void check(const SparseWeights& sparse, std::size_t rows, const ResponseValues& values)
    {
        const auto fail = [&](const std::string& found)
        {
            throw std::invalid_argument(
                base::concatenate({"vjp: expected the sparse weights of response ", std::to_string(values.id), " (",
                                   values.label, ") in compressed rows of shape (", std::to_string(rows), ", ",
                                   std::to_string(values.itemCount()), "), found ", found}));
        };
        if (sparse.rowStart.size() != rows + 1)
            fail(base::concatenate({std::to_string(sparse.rowStart.size()), " row starts"}));
        if (sparse.items.size() != sparse.values.size())
            fail(base::concatenate(
                {std::to_string(sparse.items.size()), " items for ", std::to_string(sparse.values.size()), " values"}));
        if (sparse.rowStart.front() != 0 || sparse.rowStart.back() != sparse.items.size())
            fail("row starts that do not run from 0 to the entries");
        for (std::size_t r = 0; r < rows; ++r)
        {
            if (sparse.rowStart[r + 1] < sparse.rowStart[r])
                fail(base::concatenate({"decreasing row starts at row ", std::to_string(r)}));
            for (std::size_t k = sparse.rowStart[r]; k < sparse.rowStart[r + 1]; ++k)
            {
                if (sparse.items[k] >= values.itemCount())
                    fail(base::concatenate(
                        {"the item ", std::to_string(sparse.items[k]), " in row ", std::to_string(r)}));
                if (k > sparse.rowStart[r] && sparse.items[k] <= sparse.items[k - 1])
                    fail(base::concatenate({"items not increasing in row ", std::to_string(r),
                                            " (sum the duplicates and sort the indices)"}));
            }
        }
    }

    std::vector<Source> m_sources;
};

// The weighted sums of a linearisation that the Hessian-vector products difference: with the weights y of
// one row, Σᵢ yᵢ·∂rᵢ/∂x (the explicit parts), Σᵢ yᵢ·cᵢ per subcase (the items' weights on the g set), Σ_s λ_sᵀ·b_js
// (the pseudo-loads against given adjoint vectors) and K·w on given elements.
class WeightedTerms : public detail::SensitivityComputation
{
public:
    using detail::SensitivityComputation::SensitivityComputation;

    // Σᵢ yᵢ·∂rᵢ/∂x over the variables, the rows' items in the order of the responses and of their items.
    [[nodiscard]] std::vector<double> weightedExplicit(const ProductWeights& weights) const
    {
        const WeightRows rows = rowsOf(weights);
        std::vector<double> values(m_variableCount, 0.0); // bounded by the variables
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            static_cast<void>(rows.forEach(p, 0,
                                           [&](std::size_t i, double w)
                                           {
                                               const ItemLinearisation& item = m_linearised.items[p][i];
                                               if (!item.isWanted)
                                                   return;
                                               for (const auto& [column, value] : item.explicitPart)
                                                   values[column] += w * value;
                                           }));
        return values;
    }

    // Σᵢ yᵢ·cᵢ on the g set over the items of subcase `label`.
    [[nodiscard]] std::vector<double> weightedLoad(const ProductWeights& weights, std::int32_t label) const
    {
        const WeightRows rows = rowsOf(weights);
        std::vector<double> load(gSize(), 0.0); // a g-set vector
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            static_cast<void>(rows.forEach(p, 0,
                                           [&](std::size_t i, double w)
                                           {
                                               const ItemLinearisation& item = m_linearised.items[p][i];
                                               if (!item.isWanted || item.subcase != label)
                                                   return;
                                               for (const auto& [dof, value] : item.weights)
                                                   load[static_cast<std::size_t>(dof)] += w * value;
                                           }));
        return load;
    }

    // Σ_s λ_sᵀ·b_js per variable j, λ_s by subcase label (the subcases without one skipped).
    [[nodiscard]] std::vector<double> adjointProducts(const std::map<std::int32_t, std::vector<double>>& adjoints) const
    {
        std::vector<double> values(m_variableCount, 0.0); // bounded by the variables
        std::vector<double> products(m_variableCount);    // bounded by the variables
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const auto adjoint = adjoints.find(m_subcases[s].subcase.label);
            if (adjoint == adjoints.end() || m_variableCount == 0)
                continue;
            m_pseudoLoads->dotAll(s, adjoint->second, products);
            for (std::size_t j = 0; j < m_variableCount; ++j)
                values[j] += products[j];
        }
        return values;
    }

    // K·w on the g set, K assembled from the elements `elements` alone (those the direction moves: the others' K is
    // the same at every point along it).
    [[nodiscard]] std::vector<double> stiffnessTimes(const std::vector<ElementRef>& elements,
                                                     std::span<const double> w) const
    {
        std::vector<double> product(gSize(), 0.0); // a g-set vector
        elements::ElementMatrixBlock block;
        for (const ElementRef& element : elements)
        {
            const assembly::AssemblyBatch& batch = m_set.batches()[element.batch];
            if (!batch.family->facets().has(elements::Facet::Stiffness))
                continue;
            block.clear(); // computeStiffness appends
            batch.family->computeStiffness(batch.view, {element.row, element.row + 1}, block);
            if (block.empty())
                continue;
            const std::span<const elements::ElementDof> dofs = block.dofs(0);
            const std::span<const double> matrix = block.matrix(0);
            std::vector<std::size_t> rows(dofs.size()); // bounded by the element's DOFs
            for (std::size_t r = 0; r < dofs.size(); ++r)
                rows[r] = static_cast<std::size_t>(m_dofMap.dofOf(dofs[r]));
            for (std::size_t r = 0; r < dofs.size(); ++r)
            {
                double sum = 0.0;
                for (std::size_t c = 0; c < dofs.size(); ++c)
                    sum += matrix[r * dofs.size() + c] * w[rows[c]];
                product[rows[r]] += sum;
            }
        }
        return product;
    }

    // Σ y·w of the weighted SPC force items of subcase `label` on the g set (empty: none), r = wᵀ·(Kgg·u − P): their
    // weights c = Kgg·w move with K, so the adjoint the stiffness terms see is λ − Σ y·w.
    [[nodiscard]] std::vector<double> weightedSpcRows(const ProductWeights& weights, std::int32_t label) const
    {
        std::map<ItemKey, double> weightOf;
        std::set<std::size_t> responses;
        for (const detail::SpcItem& item : m_spcItems)
            if (m_subcases[item.subcase].subcase.label == label)
            {
                weightOf.emplace(item.key, 0.0);
                responses.insert(item.key.response);
            }
        if (weightOf.empty())
            return {};
        const WeightRows rows = rowsOf(weights);
        for (const std::size_t p : responses)
            static_cast<void>(rows.forEach(p, 0,
                                           [&](std::size_t i, double w)
                                           {
                                               if (const auto found = weightOf.find({p, i}); found != weightOf.end())
                                                   found->second = w;
                                           }));
        std::vector<double> sum; // a g-set vector, when some weight is not zero
        for (const detail::SpcItem& item : m_spcItems)
        {
            const auto found = weightOf.find(item.key);
            if (found == weightOf.end() || found->second == 0.0)
                continue;
            if (sum.empty())
                sum.assign(gSize(), 0.0);
            for (const auto& [dof, value] : item.w)
                sum[static_cast<std::size_t>(dof)] += found->second * value;
        }
        return sum;
    }

    // The labels of the subcases with a non-zero weight on a wanted item (the subcases a product solves).
    [[nodiscard]] std::set<std::int32_t> weightedSubcases(const ProductWeights& weights) const
    {
        std::set<std::int32_t> labels;
        const WeightRows rows = rowsOf(weights);
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            static_cast<void>(rows.forEach(p, 0,
                                           [&](std::size_t i, double)
                                           {
                                               const ItemLinearisation& item = m_linearised.items[p][i];
                                               if (item.isWanted && item.subcase != 0)
                                                   labels.insert(item.subcase);
                                           }));
        return labels;
    }

    // The relief of the moved points of the products: this point's element set and the reliefs of its
    // constraint sets; null without inertia relief.
    [[nodiscard]] std::optional<detail::ReliefPoint> reliefPointOf(std::vector<ElementRef> moved) const
    {
        if (std::none_of(m_subcases.begin(), m_subcases.end(),
                         [](const detail::SubcaseState& state) { return state.relief != nullptr; }))
            return std::nullopt;
        detail::ReliefPoint point;
        point.set = &m_set;
        point.moved = std::move(moved);
        for (const std::unique_ptr<SensitivitySystem>& system : m_systems)
            point.reliefs.push_back(system->relief());
        return point;
    }

    // The labels of the subcases linearised, in order.
    [[nodiscard]] std::vector<std::int32_t> subcaseLabels() const
    {
        std::vector<std::int32_t> labels; // bounded by the subcases
        for (const detail::SubcaseState& state : m_subcases)
            labels.push_back(state.subcase.label);
        return labels;
    }

protected:
    [[nodiscard]] WeightRows rowsOf(const ProductWeights& weights) const
    {
        return WeightRows(weights, m_linearised,
                          [this](std::size_t p) -> const ResponseValues& { return valuesAt(p); });
    }
};

// The linearisation at a perturbed state (x + σ·v, u + σ·δu) of the Hessian-vector products: the
// evaluation's inputs with the design point x' (its snapshot and field values), the displacements of each subcase
// shifted by σ·δu, no value check (the evaluation's values are those of x).
class PerturbedComputation : public WeightedTerms
{
public:
    // `reliefPoint`: under inertia relief, the point's element set, the elements the direction moves and the point's
    // reliefs, from which the relief of the moved point is made; null without relief.
    PerturbedComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                         base::MemoryBudget& budget, std::map<std::int32_t, std::vector<double>> shifts,
                         const detail::ReliefPoint* reliefPoint = nullptr)
        : WeightedTerms(inputs, options, budget)
    {
        m_checksValues = false;
        m_displacementShifts = std::move(shifts);
        if (reliefPoint != nullptr)
            m_reliefPoint = *reliefPoint;
        linearise();
    }
};

// The agreement asked of T built from the l DOFs and Gm with the expansion of the solver (fromEquations), relative to
// the expansion: the rounding of a Gm row's sum, far below a wrong term (a coefficient of order 1).
constexpr double TRANSFORM_CHECK_TOLERANCE = 1.0e-12;

// The components of a grid point on the g set (DofMap: 6 per grid point).
constexpr int MAX_GRID_COMPONENTS = 6;

// The pieces per thread of the right-hand sides and of the products of a vjp's solve, and the fewest
// columns of a piece: a piece allocates a g-set vector of its own, so a few per thread (the columns cost about the
// same).
constexpr std::size_t PRODUCT_PIECES_PER_THREAD = 2;
constexpr std::size_t PRODUCT_COLUMNS_PER_PIECE = 4;
// The columns the right-hand sides of a vjp's solves are allocated by (the shared route's solves vary by a few
// columns: a buffer made again for each larger one was zeroed again each time).
constexpr std::size_t RIGHT_HAND_SIDE_SLACK = 64;

// The seconds from `from` to now, and `from` moved to now (the phases of a vjp's solves, ProductReport).
double lap(std::chrono::steady_clock::time_point& from)
{
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - from).count();
    from = now;
    return seconds;
}

// The rows of a jvp (ProductSelection): the positions in the evaluation of the responses returned, in its
// order, and per position whether each of its items is selected (empty: every item). Unrestricted: every response.
struct JvpRows
{
    bool isRestricted = false;
    std::vector<std::size_t> positions;
    std::vector<std::vector<std::uint8_t>> isSelected; // per entry of `positions`
    std::vector<std::int32_t> ids;                     // per entry of `positions`, its response's ID
};

// The rows of a selection on the evaluation's responses; throws std::invalid_argument for a response the evaluation
// does not hold, a mask of the wrong size or on a response the selection leaves out, a subcase not analysed.
JvpRows jvpRowsOf(const DesignEvaluation& evaluation, const ProductSelection& selection)
{
    JvpRows rows;
    const std::vector<ResponseValues>& values = evaluation.responses;
    if (selection.isEmpty())
    {
        rows.positions.resize(values.size());
        std::iota(rows.positions.begin(), rows.positions.end(), std::size_t{0});
        rows.isSelected.resize(values.size());
        return rows;
    }
    rows.isRestricted = true;
    std::map<std::int32_t, std::size_t> positionOf;
    for (std::size_t p = 0; p < values.size(); ++p)
        positionOf.emplace(values[p].id, p);
    const auto require = [&](std::int32_t id, const char* what)
    {
        if (positionOf.count(id) == 0)
            throw std::invalid_argument(base::concatenate(
                {"jvp: expected ", what, " of an evaluated response, found response ", std::to_string(id),
                 ", which the evaluation does not hold (not defined, or evaluated in no subcase); select the "
                 "responses the evaluation holds"}));
    };
    std::set<std::int32_t> selected(selection.responseIds.begin(), selection.responseIds.end());
    for (const std::int32_t id : selected)
        require(id, "a selection");
    std::map<std::int32_t, const ItemMask*> maskOf;
    for (const ItemMask& mask : selection.itemMasks)
    {
        require(mask.responseId, "an item mask");
        if (!selection.responseIds.empty() && selected.count(mask.responseId) == 0)
            throw std::invalid_argument(base::concatenate(
                {"jvp: expected item masks on the selected responses, found one on response ",
                 std::to_string(mask.responseId), ", which the selection leaves out; select it, or drop its mask"}));
        if (!maskOf.emplace(mask.responseId, &mask).second)
            throw std::invalid_argument(
                base::concatenate({"jvp: expected one item mask per response, found two on "
                                   "response ",
                                   std::to_string(mask.responseId), "; give one"}));
        const std::size_t itemCount = values[positionOf.at(mask.responseId)].itemCount();
        if (mask.isRetained.size() != itemCount)
            throw std::invalid_argument(
                base::concatenate({"jvp: expected an item mask of ", std::to_string(itemCount), " flags on response ",
                                   std::to_string(mask.responseId), " (its items), found ",
                                   std::to_string(mask.isRetained.size()), "; give one flag per item"}));
    }
    if (selection.responseIds.empty())
        for (const auto& [id, mask] : maskOf)
            selected.insert(id);
    if (selected.empty())
        for (const ResponseValues& value : values)
            selected.insert(value.id);
    const std::set<std::int32_t> subcases(selection.subcases.begin(), selection.subcases.end());
    for (const std::int32_t label : subcases)
        if (std::find(evaluation.subcases.begin(), evaluation.subcases.end(), label) == evaluation.subcases.end())
            throw std::invalid_argument(
                base::concatenate({"jvp: expected subcases of the analysis, found subcase ", std::to_string(label),
                                   ", which it did not analyse; select analysed subcases"}));
    for (std::size_t p = 0; p < values.size(); ++p)
    {
        if (selected.count(values[p].id) == 0)
            continue;
        const auto mask = maskOf.find(values[p].id);
        std::vector<std::uint8_t> flags(values[p].itemCount(), 1); // bounded by the response's items
        for (std::size_t i = 0; i < flags.size(); ++i)
        {
            const bool isMasked = mask != maskOf.end() && mask->second->isRetained[i] == 0;
            const std::int32_t subcase = values[p].items[i].subcase;
            const bool isOtherSubcase = !subcases.empty() && subcase != 0 && subcases.count(subcase) == 0;
            flags[i] = static_cast<std::uint8_t>(!isMasked && !isOtherSubcase);
        }
        rows.positions.push_back(p);
        rows.isSelected.push_back(std::move(flags));
        rows.ids.push_back(values[p].id);
    }
    return rows;
}

// The linearisation a restricted jvp makes for itself: the selected items (an item mask per selected
// response) and the fields of the variables its directions move.
GradientOptions restrictedOptionsOf(const JvpRows& rows, std::span<const double> directions, std::size_t count,
                                    std::size_t variableCount)
{
    GradientOptions options;
    options.responseIds = rows.ids;
    for (std::size_t k = 0; k < rows.positions.size(); ++k)
        options.itemMasks.push_back({rows.ids[k], rows.isSelected[k]});
    options.isVariableWanted.assign(variableCount, 0);
    if (count > 0 && directions.size() == variableCount * count)
        for (std::size_t j = 0; j < variableCount; ++j)
            for (std::size_t d = 0; d < count; ++d)
                if (directions[j * count + d] != 0.0)
                {
                    options.isVariableWanted[j] = 1;
                    break;
                }
    return options;
}

// The linearisation of every item, then the products on the factors of the subcases.
class ProductComputation : public WeightedTerms
{
public:
    ProductComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                       base::MemoryBudget& budget)
        : WeightedTerms(inputs, options, budget)
    {
    }

    [[nodiscard]] const StaticSensitivityInputs& inputs() const noexcept { return m_in; }

    // What a solve of the Hessian-vector products reads of its solution on the condensed path: the
    // pseudo-loads' DOFs (λ and δλ: dotted with b and moved by ∂K/∂v), and for δu the items' too (the linearisation at
    // the moved states reads the displacements of the weighted items' elements).
    enum class Reads : std::uint8_t
    {
        PseudoLoads,
        Direction
    };

    // K⁻¹·w per subcase label (one solve each, on the subcase's factor), w and the result on the g set.
    [[nodiscard]] std::map<std::int32_t, std::vector<double>> solve(
        const std::map<std::int32_t, std::vector<double>>& loads, std::int64_t& solves,
        Reads what = Reads::PseudoLoads) const
    {
        std::map<std::int32_t, std::vector<double>> result;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const auto load = loads.find(m_subcases[s].subcase.label);
            if (load == loads.end())
                continue;
            const SensitivitySystem& system = systemOf(m_subcases[s]);
            const auto equations = static_cast<std::size_t>(system.equationCount());
            linalg::DenseMatrix rightHandSide(m_budget, equations, 1, "Hessian-vector solve");
            std::vector<double> column(equations);
            system.toEquations(load->second, column);
            for (std::size_t e = 0; e < equations; ++e)
                rightHandSide(e, 0) = column[e];
            base::BudgetedBuffer<std::uint8_t> reads =
                SolutionReads::ofPseudoLoads(system, *m_pseudoLoads, s, m_budget);
            if (what == Reads::Direction && !reads.empty())
            {
                std::vector<ItemKey> keys; // bounded by the items
                for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                    for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
                        if (m_linearised.items[p][i].isWanted && m_linearised.items[p][i].subcase == load->first)
                            keys.push_back({p, i});
                const base::BudgetedBuffer<std::uint8_t> items =
                    SolutionReads::ofWeights(system, m_linearised, keys, m_budget);
                for (std::size_t g = 0; g < items.size(); ++g)
                    reads[g] = static_cast<std::uint8_t>(reads[g] | items[g]);
            }
            if (equations > 0)
                system.solve(rightHandSide.view(), reads.span());
            ++solves;
            for (std::size_t e = 0; e < equations; ++e)
                column[e] = rightHandSide(e, 0);
            std::vector<double> solution(gSize()); // a g-set vector
            system.fromEquations(column, solution);
            result.emplace(m_subcases[s].subcase.label, std::move(solution));
        }
        return result;
    }

    // Σ_j v_j·b_js per subcase label (the right-hand side of δu = K⁻¹·b_v), for the subcases of `labels` (every one
    // when null).
    [[nodiscard]] std::map<std::int32_t, std::vector<double>> directionLoads(
        std::span<const double> v, const std::set<std::int32_t>* labels = nullptr) const
    {
        std::map<std::int32_t, std::vector<double>> result;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            if (labels != nullptr && labels->count(m_subcases[s].subcase.label) == 0)
                continue;
            std::vector<double> sum(gSize()); // a g-set vector
            m_pseudoLoads->combine(s, v, sum);
            result.emplace(m_subcases[s].subcase.label, std::move(sum));
        }
        return result;
    }

    // The elements whose computations a move of x along v changes: those of the fields some variable of v drives.
    [[nodiscard]] std::vector<ElementRef> elementsMovedBy(std::span<const double> v) const
    {
        const AffectedElements affected(*m_in.model, m_set);
        std::set<ElementRef> moved;
        const std::vector<FieldValue>& fields = m_in.design->fields;
        for (std::size_t r = 0; r < fields.size(); ++r)
            if (std::any_of(fields[r].gradient.begin(), fields[r].gradient.end(),
                            [&](const auto& term) { return term.second != 0.0 && v[term.first] != 0.0; }))
                moved.insert(affected.of(r).begin(), affected.of(r).end());
        return {moved.begin(), moved.end()};
    }

    // Refuses what the blocks of a design point do not cover (OptimisationError naming it).
    void requireBlocks() const
    {
        const auto refuse = [](const std::string& what)
        {
            throw base::OptimisationError(
                base::concatenate({"blocks: ", what,
                                   "; this version gives the blocks of an analysis on the full path: "
                                   "analyse the point on the full path, without it"}));
        };
        if (hasExternalUnits())
            refuse("the analysis has external units, whose K* is no element's");
        if (!m_in.design->shapes.empty())
            refuse("the design has shape variables");
        for (const detail::SubcaseState& state : m_subcases)
        {
            if (state.isAeroelastic)
                refuse(base::concatenate({"subcase ", std::to_string(state.subcase.label), " is a SAERO subcase"}));
            if (state.relief != nullptr)
                refuse(base::concatenate({"subcase ", std::to_string(state.subcase.label), " has inertia relief"}));
            if (systemOf(state).isCondensed())
                refuse("the analysis is on the condensed path");
        }
    }

    // The blocks of the design point (DesignBlocks), on the linearisation of every response, with their
    // structural supports; with `hessian` (the element Hessian blocks of (x, y)) and the
    // adjoints λ_s of y by subcase label, also the second derivatives of the Lagrangian.
    void blocks(DesignBlocks& result, const detail::HessianBlocks* hessian = nullptr,
                const std::map<std::int32_t, std::vector<double>>* adjoints = nullptr) const
    {
        requireBlocks();
        const std::size_t g = gSize();
        const std::size_t n = m_variableCount;
        // The constraint sets of the subcases, in the order of their first subcase.
        std::map<const SensitivitySystem*, std::size_t> setOf;
        for (const detail::SubcaseState& state : m_subcases)
        {
            const SensitivitySystem& system = systemOf(state);
            if (setOf.emplace(&system, result.sets.size()).second)
                result.sets.push_back(setBlocksOf(system));
        }
        // The structural supports: per variable the DOFs of the nodes of the elements its fields reach
        // (the support of its pseudo-loads, whatever the point), per element the variables of its fields (the support
        // of its items' explicit parts). The elements read (the fields' and the element items') each in a slot, their
        // node DOFs formed on the pool, each in its own slot: the same supports whatever the threads.
        const std::vector<FieldValue>& fields = m_in.design->fields;
        const AffectedElements affected(*m_in.model, m_set);
        const std::span<const assembly::AssemblyBatch> batches = m_set.batches();
        std::vector<std::size_t> batchStart(batches.size() + 1, 0); // bounded by the batches
        for (std::size_t b = 0; b < batches.size(); ++b)
            batchStart[b + 1] = batchStart[b] + static_cast<std::size_t>(batches[b].view.rowCount());
        std::vector<std::int64_t> slotOf(batchStart.back(), -1); // one per element of the set
        std::vector<ElementRef> slots;                           // bounded by the elements read
        const auto slotFor = [&](const ElementRef& element)
        {
            std::int64_t& slot = slotOf[batchStart[element.batch] + static_cast<std::size_t>(element.row)];
            if (slot < 0)
            {
                slot = static_cast<std::int64_t>(slots.size());
                slots.push_back(element);
            }
            return static_cast<std::size_t>(slot);
        };
        for (std::size_t r = 0; r < fields.size(); ++r)
            if (!fields[r].gradient.empty())
                for (const ElementRef& element : affected.of(r))
                    static_cast<void>(slotFor(element));
        for (const detail::ElementJob& job : m_jobs)
            static_cast<void>(slotFor(job.element));
        std::vector<std::vector<base::DofIndex>> nodeDofs(slots.size()); // per slot
        base::parallelFor(m_in.pool, slots.size(), detail::ELEMENT_JOBS_PER_PIECE,
                          [&](std::size_t begin, std::size_t end)
                          {
                              for (std::size_t k = begin; k < end; ++k)
                                  nodeDofs[k] = elementNodeDofs(slots[k]);
                          });
        std::vector<std::vector<base::DofIndex>> variableDofs(n);         // bounded by the variables' elements
        std::vector<std::vector<std::int64_t>> variablesOf(slots.size()); // per slot
        for (std::size_t r = 0; r < fields.size(); ++r)
        {
            if (fields[r].gradient.empty())
                continue;
            for (const ElementRef& element : affected.of(r))
            {
                const std::size_t slot = slotFor(element);
                const std::vector<base::DofIndex>& dofs = nodeDofs[slot];
                for (const auto& [column, derivative] : fields[r].gradient)
                {
                    variableDofs[column].insert(variableDofs[column].end(), dofs.begin(), dofs.end());
                    variablesOf[slot].push_back(static_cast<std::int64_t>(column));
                }
            }
        }
        for (std::vector<base::DofIndex>& dofs : variableDofs)
            sortUnique(dofs);
        for (std::vector<std::int64_t>& variables : variablesOf)
            sortUnique(variables);
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            SubcaseBlocks subcase;
            subcase.subcase = m_subcases[s].subcase.label;
            subcase.set = setOf.at(&systemOf(m_subcases[s]));
            subcase.pseudoLoads.rows = subcase.pseudoLoadSupport.rows = g;
            subcase.pseudoLoads.columns = subcase.pseudoLoadSupport.columns = n;
            // The mixed terms Σ_e A_up·∂p/∂x of the subcase by variable, in the blocks' order.
            std::vector<std::vector<std::pair<base::DofIndex, double>>> mixedOf(hessian != nullptr ? n : 0);
            if (hessian != nullptr)
                for (const detail::HessianBlocks::Block& block : hessian->blocks())
                {
                    if (block.subcase != s || block.field == detail::HessianBlocks::NO_FIELD || block.mixed.empty())
                        continue;
                    for (std::size_t i = 0; i < block.dofs.size(); ++i)
                        for (const auto& [column, derivative] : fields[block.field].gradient)
                            mixedOf[column].emplace_back(block.dofs[i], block.mixed[i] * derivative);
                }
            std::vector<double> load(g);         // a g-set vector
            std::vector<base::DofIndex> support; // one variable's, bounded by its elements' DOFs
            for (std::size_t j = 0; j < n; ++j)
            {
                support = variableDofs[j];
                if (m_pseudoLoads->hasLoad(j, s))
                {
                    m_pseudoLoads->assemble(j, s, load);
                    for (std::size_t dof = 0; dof < g; ++dof)
                        if (load[dof] != 0.0)
                        {
                            subcase.pseudoLoads.add(static_cast<std::int64_t>(dof), static_cast<std::int64_t>(j),
                                                    load[dof]);
                            support.push_back(static_cast<base::DofIndex>(dof));
                        }
                }
                if (hessian != nullptr)
                    for (const auto& [dof, value] : mixedOf[j])
                        support.push_back(dof);
                sortUnique(support);
                for (const base::DofIndex dof : support)
                    subcase.pseudoLoadSupport.add(static_cast<std::int64_t>(dof), static_cast<std::int64_t>(j));
                if (hessian == nullptr)
                    continue;
                const std::size_t first = subcase.hessianMixed.size();
                subcase.hessianMixed.resize(first + support.size(), 0.0);
                for (const auto& [dof, value] : mixedOf[j])
                    subcase.hessianMixed[first + static_cast<std::size_t>(
                                                     std::lower_bound(support.begin(), support.end(), dof) -
                                                     support.begin())] += value;
            }
            if (hessian != nullptr)
            {
                const std::size_t set = subcase.set;
                subcase.hessianDisplacement =
                    displacementHessian(systemOf(m_subcases[s]), result.sets[set].stiffnessSupport.row.size(), *hessian,
                                        s, result.hessianAsymmetry);
                if (const auto found = adjoints->find(subcase.subcase); found != adjoints->end())
                    subcase.adjoint = found->second;
            }
            result.subcases.push_back(std::move(subcase));
        }
        std::size_t itemCount = 0;
        for (const std::vector<ItemLinearisation>& items : m_linearised.items)
            itemCount += items.size();
        result.itemWeights.rows = result.explicitParts.rows = itemCount;
        result.itemWeights.columns = g;
        result.explicitParts.columns = n;
        result.itemWeightSupport.rows = result.explicitPartSupport.rows = itemCount;
        result.itemWeightSupport.columns = g;
        result.explicitPartSupport.columns = n;
        // The elements of the element items (their slots) and the SPC force items, for their supports.
        std::vector<std::vector<std::int64_t>> slotOfItem(m_linearised.items.size()); // per response, its items
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            slotOfItem[p].assign(m_linearised.items[p].size(), -1);
        for (const detail::ElementJob& job : m_jobs)
        {
            const auto slot = static_cast<std::int64_t>(slotFor(job.element));
            for (const ItemKey& key : job.keys)
                slotOfItem[key.response][key.item] = slot;
        }
        std::map<ItemKey, const detail::SpcItem*> spcOfItem;
        for (const detail::SpcItem& item : m_spcItems)
            spcOfItem.emplace(item.key, &item);
        std::vector<std::vector<std::int32_t>> neighbours; // of the node graph, made for the first SPC force item
        std::vector<base::DofIndex> dofs;                  // one item's support, bounded by its element's DOFs
        std::vector<std::int64_t> columns;                 // the same
        std::int64_t row = 0;
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
            {
                const ItemLinearisation& item = m_linearised.items[p][i];
                result.itemSubcases.push_back(item.subcase);
                if (item.isWanted)
                {
                    dofs.clear();
                    columns.clear();
                    for (const auto& [dof, value] : item.weights)
                    {
                        result.itemWeights.add(row, static_cast<std::int64_t>(dof), value);
                        dofs.push_back(dof);
                    }
                    for (const auto& [column, value] : item.explicitPart)
                    {
                        result.explicitParts.add(row, static_cast<std::int64_t>(column), value);
                        columns.push_back(static_cast<std::int64_t>(column));
                    }
                    if (const std::int64_t slot = slotOfItem[p][i]; slot >= 0)
                    {
                        const std::vector<base::DofIndex>& own = nodeDofs[static_cast<std::size_t>(slot)];
                        dofs.insert(dofs.end(), own.begin(), own.end());
                        const std::vector<std::int64_t>& variables = variablesOf[static_cast<std::size_t>(slot)];
                        columns.insert(columns.end(), variables.begin(), variables.end());
                    }
                    else if (const auto spc = spcOfItem.find({p, i}); spc != spcOfItem.end())
                        addNeighbourDofs(*spc->second, neighbours, dofs);
                    sortUnique(dofs);
                    sortUnique(columns);
                    for (const base::DofIndex dof : dofs)
                        result.itemWeightSupport.add(row, static_cast<std::int64_t>(dof));
                    for (const std::int64_t column : columns)
                        result.explicitPartSupport.add(row, column);
                }
                ++row;
            }
        if (hessian != nullptr)
            designHessian(*hessian, result);
        result.dofPoints.resize(g);
        result.dofComponents.resize(g);
        for (std::size_t dof = 0; dof < g; ++dof)
        {
            const assembly::DofLabel label = m_dofMap.labelOf(static_cast<base::DofIndex>(dof));
            result.dofPoints[dof] = m_dofMap.pointIdOf(static_cast<base::DofIndex>(dof));
            result.dofComponents[dof] =
                static_cast<std::int8_t>(label.kind == input::PointKind::Scalar ? 0 : label.component);
        }
    }

    // Kgg's lower triangle, T and the equations' DOFs of one constraint set.
    [[nodiscard]] ConstraintSetBlocks setBlocksOf(const SensitivitySystem& system) const
    {
        ConstraintSetBlocks set;
        const std::size_t g = gSize();
        const auto equations = static_cast<std::size_t>(system.equationCount());
        // The lower triangle of Kgg, block by block (BlockSparseMatrix: the lower blocks, column-major values, the
        // diagonal blocks complete), on the g-set DOFs of the pattern (node-major: the g set's order).
        const linalg::BlockSparseMatrix& kgg = system.kgg();
        const linalg::SparsePattern& pattern = kgg.pattern();
        set.stiffness.rows = set.stiffness.columns = g;
        for (std::int32_t column = 0; column < pattern.nodeCount(); ++column)
            for (std::int64_t block = pattern.columnBegin(column); block < pattern.columnBegin(column + 1); ++block)
            {
                const std::int32_t blockRow = pattern.blockRow(block);
                const int rows = pattern.nodeDofCount(blockRow);
                const int columns = pattern.nodeDofCount(column);
                const std::span<const double> values = kgg.blockValues(block);
                for (int j = 0; j < columns; ++j)
                    for (int i = blockRow == column ? j : 0; i < rows; ++i)
                    {
                        set.stiffnessSupport.add(pattern.firstDof(blockRow) + i, pattern.firstDof(column) + j);
                        if (const double value = values[static_cast<std::size_t>(i + j * rows)]; value != 0.0)
                            set.stiffness.add(pattern.firstDof(blockRow) + i, pattern.firstDof(column) + j, value);
                    }
            }
        set.stiffnessSupport.rows = set.stiffnessSupport.columns = g;
        // T: the equation of each l DOF from one expansion (fromEquations of e + 1 on equation e: exact integers on
        // the l rows), Gm on the m rows; checked against fromEquations, else made column by column.
        set.transform.rows = g;
        set.transform.columns = equations;
        set.equationDofs.assign(equations, -1);
        std::vector<double> probe(equations);
        std::vector<double> expanded(g); // a g-set vector
        for (std::size_t e = 0; e < equations; ++e)
            probe[e] = static_cast<double>(e + 1);
        system.fromEquations(probe, expanded);
        const assembly::RigidElimination& gm = system.gm();
        std::vector<std::int64_t> equationOf(g, -1);
        for (std::size_t dof = 0; dof < g; ++dof)
        {
            const auto index = static_cast<base::DofIndex>(dof);
            if (system.isConstrained(index) || gm.isDependent(index))
                continue;
            const double value = expanded[dof];
            if (value >= 1.0 && value <= static_cast<double>(equations) && value == std::round(value))
            {
                equationOf[dof] = static_cast<std::int64_t>(value) - 1;
                set.equationDofs[static_cast<std::size_t>(equationOf[dof])] = static_cast<std::int64_t>(dof);
            }
        }
        const bool isMapped =
            std::none_of(set.equationDofs.begin(), set.equationDofs.end(), [](std::int64_t dof) { return dof < 0; });
        if (isMapped)
        {
            for (std::size_t dof = 0; dof < g; ++dof)
                if (equationOf[dof] >= 0)
                    set.transform.add(static_cast<std::int64_t>(dof), equationOf[dof], 1.0);
            for (std::size_t i = 0; i < gm.rowCount(); ++i)
                for (const assembly::ConstraintTerm& term : gm.row(i))
                    if (equationOf[static_cast<std::size_t>(term.dof)] >= 0)
                        set.transform.add(gm.mDofs()[i], equationOf[static_cast<std::size_t>(term.dof)],
                                          term.coefficient);
        }
        set.isTransformFromGm = isMapped && expandsAs(system, set.transform);
        if (!set.isTransformFromGm)
        {
            // Column by column (O(equations · g)): T·e_k is the expansion of the unit equation k.
            set.transform.row.clear();
            set.transform.column.clear();
            set.transform.value.clear();
            std::fill(probe.begin(), probe.end(), 0.0);
            for (std::size_t e = 0; e < equations; ++e)
            {
                probe[e] = 1.0;
                system.fromEquations(probe, expanded);
                probe[e] = 0.0;
                for (std::size_t dof = 0; dof < g; ++dof)
                    if (expanded[dof] != 0.0)
                    {
                        set.transform.add(static_cast<std::int64_t>(dof), static_cast<std::int64_t>(e), expanded[dof]);
                        if (expanded[dof] == 1.0 && set.equationDofs[e] < 0 &&
                            !gm.isDependent(static_cast<base::DofIndex>(dof)))
                            set.equationDofs[e] = static_cast<std::int64_t>(dof);
                    }
            }
        }
        return set;
    }

    // A vector sorted, each value once.
    template <typename T>
    static void sortUnique(std::vector<T>& values)
    {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }

    // The DOFs of the nodes of an element (the structural supports), sorted: every component of each grid
    // point (6) and the scalar points of its stiffness' DOFs, or of its mass' for an element without stiffness (a mass
    // element a field designs); none for an element with neither. Every quantity of the element (its matrices, its
    // loads, its recoveries' slopes) lives on them.
    [[nodiscard]] std::vector<base::DofIndex> elementNodeDofs(const ElementRef& element) const
    {
        const assembly::AssemblyBatch& batch = m_set.batches()[element.batch];
        const elements::ElementFamily& family = *batch.family;
        const elements::ElementRange range{element.row, element.row + 1};
        elements::ElementMatrixBlock block;
        if (family.facets().has(elements::Facet::Stiffness))
            family.computeStiffness(batch.view, range, block);
        if (block.empty() && family.facets().has(elements::Facet::Mass))
            family.computeMass(batch.view, range, elements::MassOptions{}, block);
        std::vector<base::DofIndex> dofs; // bounded by the element's nodes
        if (block.empty())
            return dofs;
        for (const elements::ElementDof& dof : block.dofs(0))
        {
            if (dof.kind == input::PointKind::Scalar)
                dofs.push_back(m_dofMap.scalarDof(dof.row));
            else
                for (int component = 1; component <= MAX_GRID_COMPONENTS; ++component)
                    dofs.push_back(m_dofMap.gridDof(dof.row, component));
        }
        sortUnique(dofs);
        return dofs;
    }

    // The node of the node-block pattern of a g-set DOF (grid points, then scalar points: DofMap's numbering).
    [[nodiscard]] std::int32_t nodeOfDof(base::DofIndex dof) const noexcept
    {
        const base::DofIndex gridDofs = static_cast<base::DofIndex>(MAX_GRID_COMPONENTS) * m_dofMap.gridCount();
        return dof < gridDofs ? static_cast<std::int32_t>(dof / MAX_GRID_COMPONENTS)
                              : m_dofMap.gridCount() + static_cast<std::int32_t>(dof - gridDofs);
    }

    // Adds to `dofs` the support of an SPC force item's weights K·w: every DOF of the nodes coupled in the
    // node graph to the nodes of w (theirs included). `neighbours` the graph's adjacency, made at the first call.
    void addNeighbourDofs(const detail::SpcItem& item, std::vector<std::vector<std::int32_t>>& neighbours,
                          std::vector<base::DofIndex>& dofs) const
    {
        const linalg::SparsePattern& pattern = systemOf(m_subcases[item.subcase]).kgg().pattern();
        if (neighbours.empty())
        {
            neighbours.resize(static_cast<std::size_t>(pattern.nodeCount()));
            for (std::int32_t column = 0; column < pattern.nodeCount(); ++column)
                for (std::int64_t block = pattern.columnBegin(column); block < pattern.columnBegin(column + 1); ++block)
                {
                    const std::int32_t blockRow = pattern.blockRow(block);
                    neighbours[static_cast<std::size_t>(column)].push_back(blockRow);
                    if (blockRow != column)
                        neighbours[static_cast<std::size_t>(blockRow)].push_back(column);
                }
        }
        for (const auto& [dof, value] : item.w)
            for (const std::int32_t node : neighbours[static_cast<std::size_t>(nodeOfDof(dof))])
                for (int k = 0; k < pattern.nodeDofCount(node); ++k)
                    dofs.push_back(static_cast<base::DofIndex>(pattern.firstDof(node) + k));
    }

    // Σ_e A_uu of subcase s on the lower blocks of Kgg: one value per entry of the set's
    // stiffnessSupport (`entries`), in its order (setBlocksOf: the columns of nodes, their blocks, each block by
    // columns, the diagonal blocks' lower halves), each pair (a, b) of the element blocks at its lower entry a ≥ b.
    // `asymmetry` gets the largest |A_uu(d, e) − A_uu(e, d)| of the blocks relative to their largest |A_uu|.
    [[nodiscard]] std::vector<double> displacementHessian(const SensitivitySystem& system, std::size_t entries,
                                                          const detail::HessianBlocks& hessian, std::size_t s,
                                                          double& asymmetry) const
    {
        const linalg::SparsePattern& pattern = system.kgg().pattern();
        std::vector<std::int64_t> offsets(static_cast<std::size_t>(pattern.blockCount())); // bounded by Kgg's blocks
        std::int64_t offset = 0;
        for (std::int32_t column = 0; column < pattern.nodeCount(); ++column)
            for (std::int64_t block = pattern.columnBegin(column); block < pattern.columnBegin(column + 1); ++block)
            {
                offsets[static_cast<std::size_t>(block)] = offset;
                const std::int64_t rows = pattern.nodeDofCount(pattern.blockRow(block));
                const std::int64_t columns = pattern.nodeDofCount(column);
                offset += pattern.blockRow(block) == column ? columns * (columns + 1) / 2 : rows * columns;
            }
        if (static_cast<std::size_t>(offset) != entries)
            throw std::logic_error(
                base::concatenate({"blocks: expected the lower blocks of Kgg to hold ", std::to_string(entries),
                                   " entries, found ", std::to_string(offset), " (a bug of O_o_Solver: report it)"}));
        std::vector<double> values(entries, 0.0); // one per entry of Kgg's lower blocks
        double largest = 0.0;
        double difference = 0.0;
        for (const detail::HessianBlocks::Block& block : hessian.blocks())
        {
            if (block.subcase != s || block.displacement.empty())
                continue;
            const std::size_t n = block.dofs.size();
            for (std::size_t d = 0; d < n; ++d)
                for (std::size_t e = 0; e < n; ++e)
                {
                    const double value = block.displacement[d * n + e];
                    largest = std::max(largest, std::abs(value));
                    difference = std::max(difference, std::abs(value - block.displacement[e * n + d]));
                    const base::DofIndex a = block.dofs[d];
                    const base::DofIndex b = block.dofs[e];
                    if (a < b || value == 0.0)
                        continue;
                    const std::int32_t rowNode = nodeOfDof(a);
                    const std::int32_t columnNode = nodeOfDof(b);
                    const std::int64_t found = pattern.findBlock(rowNode, columnNode);
                    if (found < 0)
                        throw std::logic_error(base::concatenate(
                            {"blocks: expected the DOFs ", std::to_string(a), " and ", std::to_string(b),
                             " of an element's Hessian block to be coupled in Kgg, found them not (a bug of "
                             "O_o_Solver: report it)"}));
                    const std::int64_t i = a - pattern.firstDof(rowNode);
                    const std::int64_t j = b - pattern.firstDof(columnNode);
                    const std::int64_t rows = pattern.nodeDofCount(rowNode);
                    const std::int64_t place =
                        rowNode == columnNode ? j * rows - j * (j - 1) / 2 + (i - j) : j * rows + i;
                    values[static_cast<std::size_t>(offsets[static_cast<std::size_t>(found)] + place)] += value;
                }
        }
        if (largest > 0.0)
            asymmetry = std::max(asymmetry, difference / largest);
        return values;
    }

    // Σ_e A_pp·(∂p/∂x)ᵀ·(∂p/∂x) on the pairs of variables of each field: the fields' terms summed in
    // the blocks' order, then each field's pairs in the order of its gradient.
    void designHessian(const detail::HessianBlocks& hessian, DesignBlocks& result) const
    {
        const std::vector<FieldValue>& fields = m_in.design->fields;
        std::map<std::pair<std::size_t, std::size_t>, double> fieldTerms;
        for (const detail::HessianBlocks::Block& block : hessian.blocks())
            if (block.field != detail::HessianBlocks::NO_FIELD)
            {
                const std::size_t other = block.otherField == detail::HessianBlocks::NO_FIELD ? block.field
                                                                                           : block.otherField;
                fieldTerms[{block.field, other}] += block.fieldTerm;
                if (other != block.field)
                    fieldTerms[{other, block.field}] += block.fieldTerm;
            }
        std::vector<std::pair<std::int64_t, std::int64_t>> pairs; // bounded by the relations' terms squared
        for (const FieldValue& field : fields)
            for (const auto& [first, a] : field.gradient)
                for (const auto& [second, b] : field.gradient)
                    pairs.emplace_back(static_cast<std::int64_t>(first), static_cast<std::int64_t>(second));
        for (const auto& [key, value] : fieldTerms)
            for (const auto& [first, a] : fields[key.first].gradient)
                for (const auto& [second, b] : fields[key.second].gradient)
                    pairs.emplace_back(static_cast<std::int64_t>(first), static_cast<std::int64_t>(second));
        sortUnique(pairs);
        result.designSupport.rows = result.designSupport.columns = m_variableCount;
        for (const auto& [first, second] : pairs)
            result.designSupport.add(first, second);
        result.hessianDesign.assign(pairs.size(), 0.0);
        for (const auto& [key, value] : fieldTerms)
        {
            if (value == 0.0)
                continue;
            for (const auto& [first, a] : fields[key.first].gradient)
                for (const auto& [second, b] : fields[key.second].gradient)
                {
                    const auto place = std::lower_bound(
                        pairs.begin(), pairs.end(),
                        std::make_pair(static_cast<std::int64_t>(first), static_cast<std::int64_t>(second)));
                    result.hessianDesign[static_cast<std::size_t>(place - pairs.begin())] += value * a * b;
                }
        }
    }

    // Whether T·q equals fromEquations(q) for a q with no two equal entries (to the rounding of the Gm sums).
    [[nodiscard]] bool expandsAs(const SensitivitySystem& system, const SparseTriplets& transform) const
    {
        const auto equations = static_cast<std::size_t>(system.equationCount());
        std::vector<double> q(equations);
        for (std::size_t e = 0; e < equations; ++e)
            q[e] = std::sin(static_cast<double>(e) + 1.0);
        std::vector<double> expected(gSize());
        system.fromEquations(q, expected);
        std::vector<double> product(gSize(), 0.0);
        for (std::size_t k = 0; k < transform.value.size(); ++k)
            product[static_cast<std::size_t>(transform.row[k])] +=
                transform.value[k] * q[static_cast<std::size_t>(transform.column[k])];
        double scale = 0.0;
        double difference = 0.0;
        for (std::size_t dof = 0; dof < expected.size(); ++dof)
        {
            scale = std::max(scale, std::abs(expected[dof]));
            difference = std::max(difference, std::abs(expected[dof] - product[dof]));
        }
        return difference <= TRANSFORM_CHECK_TOLERANCE * std::max(scale, 1.0);
    }

    // The element Hessian blocks of the weights `weights` (one row) and the adjoints λ_s of `adjoints` (by subcase
    // label) in closed form (HessianBlocks), or the reason the closed forms do not cover them: the caller
    // then differences the linearisation at the moved states. Covered: the element items of a Linear or plane
    // von Mises column, the displacements, the SPC forces (their weights w join the adjoint, λ̃ = λ − Σ y·w), the
    // relations linear in x (DVxREL1, through linear links) of one cell, whose family gives the closed
    // second derivatives (ElementFamily::stiffnessFieldSecondDerivative ...; the mass's under inertial loads), no
    // thermal load or pressure the field moves. The elements are formed on the pool, each block its own: the same
    // blocks whatever the threads.
    struct ClosedBlocks
    {
        std::unique_ptr<detail::HessianBlocks> blocks;
        std::string reason; // why there are none
    };
    [[nodiscard]] ClosedBlocks hessianBlocks(const ProductWeights& weights,
                                             const std::map<std::int32_t, std::vector<double>>& adjoints,
                                             base::MemoryBudget& budget)
    {
        ClosedBlocks result;
        if (m_modal && !m_modal->isEmpty())
        {
            result.reason = "modal field Hessians require the modal adjoint derivative, outside static element blocks";
            return result;
        }
        if (m_options.elementDerivatives != ElementDerivativeMethod::Analytic)
        {
            result.reason = "the element derivatives are differenced (GradientOptions::elementDerivatives)";
            return result;
        }
        const std::size_t g = gSize();
        // The weighted items, by key.
        std::map<ItemKey, double> weightOf;
        const WeightRows rows = rowsOf(weights);
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            static_cast<void>(rows.forEach(p, 0,
                                           [&](std::size_t i, double w)
                                           {
                                               if (m_linearised.items[p][i].isWanted)
                                                   weightOf[{p, i}] = w;
                                           }));
        std::set<ItemKey> covered;
        for (const detail::ElementJob& job : m_jobs)
            covered.insert(job.keys.begin(), job.keys.end());
        for (const detail::SpcItem& item : m_spcItems)
            covered.insert(item.key);
        // The section loads: a section's rows are weighted sums of its members' FORCE items with
        // coefficients of the geometry (output::SectionLoads::memberCoefficients: w·T/W, constant in the sizing
        // fields), so their second derivatives are those of the members' force items, each member's weights over its
        // force columns Σ_k y_k·C_m(k, l) (sectionMemberWeights). A cut's loads (grid point forces) stay differenced.
        std::map<ItemKey, const detail::SectionItem*> sectionItemOf;
        for (const detail::SectionItem& item : m_sectionItems)
            if (m_sectionRows.at(item.response).sections)
                sectionItemOf.emplace(item.key, &item);
        for (const auto& [key, weight] : weightOf)
            if (sectionItemOf.count(key) != 0)
                covered.insert(key);
        for (const auto& [key, weight] : weightOf)
        {
            if (covered.count(key) != 0)
                continue;
            const DesignResp1Response* response = m_in.responses->findDesignResp1(valuesAt(key.response).id);
            if (response == nullptr || response->definition().type != ResponseType::Displacement)
            {
                result.reason =
                    base::concatenate({"response ", std::to_string(valuesAt(key.response).id), " (",
                                       response == nullptr ? std::string("not a DesignResp1")
                                                           : std::string(responseTypeName(response->definition().type)),
                                       ") has no closed second derivatives in this version"});
                return result;
            }
        }
        // The adjoints of the stiffness terms, λ̃_s = λ_s − Σ y·w of the SPC force items of s (empty: zero).
        std::vector<std::vector<double>> effective(m_subcases.size());
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (const auto found = adjoints.find(m_subcases[s].subcase.label); found != adjoints.end())
                effective[s] = found->second;
        for (const detail::SpcItem& item : m_spcItems)
        {
            const auto found = weightOf.find(item.key);
            if (found == weightOf.end())
                continue;
            std::vector<double>& adjoint = effective[item.subcase];
            if (adjoint.empty())
                adjoint.assign(g, 0.0);
            for (const auto& [dof, value] : item.w)
                adjoint[static_cast<std::size_t>(dof)] -= found->second * value;
        }
        const auto isNonZero = [](const std::vector<double>& values)
        { return std::any_of(values.begin(), values.end(), [](double value) { return value != 0.0; }); };
        // Under inertia relief the relief of the moved point (Q and the accelerations move with the mass) is in the
        // differences of the linearisation, not in the closed forms.
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (m_subcases[s].relief != nullptr && isNonZero(effective[s]))
            {
                result.reason = base::concatenate(
                    {"subcase ", std::to_string(m_subcases[s].subcase.label),
                     " has inertia relief, whose relief at the moved points the closed forms do not hold"});
                return result;
            }
        // The fields reaching each element. Diagonal blocks keep one field, while pair blocks keep cross terms.
        const std::vector<FieldValue>& fields = m_in.design->fields;
        const AffectedElements affected(*m_in.model, m_set);
        std::map<ElementRef, std::vector<std::size_t>> fieldsOf;
        for (std::size_t r = 0; r < fields.size(); ++r)
        {
            if (fields[r].gradient.empty() || affected.of(r).empty())
                continue;
            const DesignRelation& relation = m_in.model->relations()[r];
            const bool isDerived = m_in.targets->derivedProperties().groupOf(r).has_value() ||
                                   m_in.targets->hasDerivedMaterial(r);
            const bool isPlainCell =
                isDirectFieldTarget(relation, (*m_in.targets)[r]) && !isDerived;
            if (relation.form != RelationForm::Linear)
            {
                result.reason = base::concatenate({"the relation ", relation.label(),
                                                   " is not linear in x (its second derivatives in x are not formed)"});
                return result;
            }
            // A derived card or scaled target needs its own second-order chain rule. A direct field, including
            // a material's summary copy, is passed to the family for its closed second derivatives.
            if (!isPlainCell)
            {
                const elements::DesignedField designed{relation.targetCard, relation.field, relation.targetId,
                                                       relation.fieldItem};
                const char* what = elements::isAnisoMat2DStiffnessTerm(designed) ? "a AnisoMat2D stiffness term"
                                   : elements::isPlyThickness(designed)          ? "a ply thickness"
                                   : isDerived ? "a section dimension"
                                               : "a field written in several cells or scaled";
                result.reason = base::concatenate(
                    {"the relation ", relation.label(), " drives ", what, " (", relation.targetCard, " ",
                     std::to_string(relation.targetId), " ", relation.field,
                     relation.fieldItem >= 0 ? "[" + std::to_string(relation.fieldItem) + "]" : std::string(),
                     "), whose element second derivatives are not formed in closed form: the "
                     "products difference the linearisation"});
                return result;
            }
            for (const ElementRef& element : affected.of(r))
                fieldsOf[element].push_back(r);
        }
        // The elements and subcases with a block: the weighted items of a job, a designed field with a stiffness
        // term (a non-zero adjoint).
        struct Entry
        {
            ElementRef element;
            std::size_t subcase = 0;
            std::vector<std::size_t> jobs; // with weighted items
            std::size_t field = detail::HessianBlocks::NO_FIELD;
            bool keepsDisplacement = true;
        };
        struct PairEntry
        {
            ElementRef element;
            std::size_t subcase;
            std::vector<std::size_t> jobs;
            std::size_t first;
            std::size_t second;
        };
        std::map<std::pair<ElementRef, std::size_t>, std::vector<std::size_t>> jobsOf;
        for (std::size_t j = 0; j < m_jobs.size(); ++j)
        {
            const detail::ElementJob& job = m_jobs[j];
            const bool isWeighted = std::any_of(job.keys.begin(), job.keys.end(),
                                                [&](const ItemKey& key) { return weightOf.count(key) != 0; });
            if (isWeighted)
                jobsOf[{job.element, job.subcase}].push_back(j);
        }
        const SectionMemberWeights sectionWeights = sectionMemberWeights(weightOf, sectionItemOf);
        std::set<std::pair<ElementRef, std::size_t>> needed;
        for (const auto& [key, jobs] : jobsOf)
            needed.insert(key);
        for (const auto& [key, members] : sectionWeights)
            needed.insert(key);
        for (const auto& [element, elementFields] : fieldsOf)
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
                if (isNonZero(effective[s]))
                    needed.insert({element, s});
        std::vector<Entry> entries;
        std::vector<PairEntry> pairEntries;
        entries.reserve(needed.size());
        for (const auto& [element, s] : needed)
        {
            const auto found = jobsOf.find({element, s});
            const auto jobs = found == jobsOf.end() ? std::vector<std::size_t>{} : found->second;
            const auto fieldFound = fieldsOf.find(element);
            if (fieldFound == fieldsOf.end())
                entries.push_back({element, s, jobs});
            else
            {
                const auto& elementFields = fieldFound->second;
                for (std::size_t a = 0; a < elementFields.size(); ++a)
                {
                    entries.push_back({element, s, jobs, elementFields[a], a == 0});
                    for (std::size_t b = a + 1; b < elementFields.size(); ++b)
                        pairEntries.push_back({element, s, jobs, elementFields[a], elementFields[b]});
                }
            }
        }
        // Keep one field's subcases together so their displacement-independent slopes are built once.
        std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b)
                         {
                             if (a.element != b.element)
                                 return a.element < b.element;
                             return a.field != b.field ? a.field < b.field : a.subcase < b.subcase;
                         });
        // The accelerations of the inertial subcases, made before the parallel loop (inertialRateOf is lazy).
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (m_subcases[s].hasInertialLoads && m_subcases[s].subcase.load)
                static_cast<void>(inertialRateOf(s));

        std::vector<detail::HessianBlocks::Block> blocks(entries.size() + pairEntries.size());
        std::vector<std::string> reasons(blocks.size());
        // PERFORMANCE: the entries of an element (its subcases, consecutive) share what does not depend on the subcase
        // (ElementParts: the slopes of its items' recoveries and of their rates, its stiffness rates in the field),
        // formed once instead of once per subcase; the same values, so the same blocks.
        // Measured: on the Hessian-vector products of the thread benchmarks.
        base::parallelFor(m_in.pool, entries.size(), detail::ELEMENT_JOBS_PER_PIECE,
                          [&](std::size_t begin, std::size_t end)
                          {
                              std::optional<base::BudgetedBuffer<double>> copy;
                              ElementParts parts;
                              for (std::size_t k = begin; k < end; ++k)
                              {
                                  if (k == begin || entries[k].element != entries[k - 1].element ||
                                      entries[k].field != entries[k - 1].field)
                                      parts = ElementParts{};
                                  const auto members = sectionWeights.find({entries[k].element, entries[k].subcase});
                                  reasons[k] =
                                      hessianBlockOf(entries[k].element, entries[k].subcase, entries[k].jobs,
                                                     entries[k].field, weightOf, effective[entries[k].subcase],
                                                     members == sectionWeights.end() ? nullptr : &members->second, copy,
                                                     parts, blocks[k]);
                                  if (!entries[k].keepsDisplacement)
                                      blocks[k].displacement.clear();
                              }
                          });
        base::parallelFor(m_in.pool, pairEntries.size(), detail::ELEMENT_JOBS_PER_PIECE,
                          [&](std::size_t begin, std::size_t end)
                          {
                              for (std::size_t k = begin; k < end; ++k)
                              {
                                  const auto& entry = pairEntries[k];
                                  const auto members = sectionWeights.find({entry.element, entry.subcase});
                                  reasons[entries.size() + k] = hessianFieldCouplingOf(
                                      entry.element, entry.subcase, entry.jobs, entry.first, entry.second, weightOf,
                                      effective[entry.subcase],
                                      members == sectionWeights.end() ? nullptr : &members->second,
                                      blocks[entries.size() + k]);
                              }
                          });
        for (std::size_t k = 0; k < reasons.size(); ++k)
            if (!reasons[k].empty())
            {
                result.reason = reasons[k];
                return result;
            }
        // Reserved once made, as the kept linearisation (heldBytes): blocks that do not fit are dropped, and the
        // products difference the linearisation instead.
        try
        {
            result.blocks = std::make_unique<detail::HessianBlocks>(std::move(blocks), budget);
        }
        catch (const base::MemoryBudgetError& error)
        {
            result.reason = base::concatenate({"the Hessian blocks do not fit the memory budget (", error.what(), ")"});
        }
        return result;
    }

    // The equation responses of the second order among the weights: their weights y moved onto
    // the items of their response arguments, μ_k = Σ_i y_i·∂f_i/∂L_k (the items' `arguments`), added to the weights of
    // the other responses (`effective`: what the second-order adjoint's own terms take), the weighted equations and the
    // subcases of their arguments (where δu is solved: δL reads it).
    struct EquationWeights
    {
        ProductWeights effective;
        std::vector<std::size_t> equations; // positions of the weighted equation responses
        std::set<std::int32_t> subcases;
    };
    [[nodiscard]] EquationWeights equationWeights(const ProductWeights& weights) const
    {
        EquationWeights result;
        const WeightRows rows = rowsOf(weights);
        std::map<ItemKey, double> moved; // μ on the argument items
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
        {
            const ResponseValues& values = valuesAt(p);
            const EquationResponse* equation = equationOf(values.id);
            if (equation == nullptr)
                continue;
            const bool isWeighted =
                weights.weights.count(values.id) != 0 || weights.sparseWeights.count(values.id) != 0;
            if (!isWeighted)
                continue;
            requireSecondOrderCallable(*equation);
            result.equations.push_back(p);
            static_cast<void>(rows.forEach(
                p, 0,
                [&](std::size_t i, double y)
                {
                    const ItemLinearisation& item = m_linearised.items[p][i];
                    if (!item.isWanted)
                        return;
                    for (const auto& [argument, derivative] : item.arguments)
                    {
                        const ItemLinearisation& of = m_linearised.at(argument);
                        if (of.isEquation)
                            throw base::OptimisationError(base::concatenate(
                                {"hessian_vector: response ", std::to_string(values.id), " (", values.label,
                                 ") reads another equation response, whose second "
                                 "derivatives are not chained in this version; give the "
                                 "equation the responses themselves"}));
                        moved[argument] += y * derivative;
                        if (of.subcase != 0)
                            result.subcases.insert(of.subcase);
                    }
                }));
        }
        if (result.equations.empty())
            return result;
        // The other responses' weights as given; the responses μ reaches dense (their given weights included).
        result.effective.rows = 1;
        std::set<std::size_t> reached;
        for (const auto& [key, value] : moved)
            reached.insert(key.response);
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
        {
            const ResponseValues& values = valuesAt(p);
            if (std::find(result.equations.begin(), result.equations.end(), p) != result.equations.end())
                continue;
            if (reached.count(p) == 0)
            {
                if (const auto dense = weights.weights.find(values.id); dense != weights.weights.end())
                    result.effective.weights.emplace(values.id, dense->second);
                if (const auto sparse = weights.sparseWeights.find(values.id); sparse != weights.sparseWeights.end())
                    result.effective.sparseWeights.emplace(values.id, sparse->second);
                continue;
            }
            std::vector<double> row(values.itemCount(), 0.0); // bounded by the response's items
            static_cast<void>(rows.forEach(p, 0, [&](std::size_t i, double w) { row[i] = w; }));
            for (const auto& [key, value] : moved)
                if (key.response == p)
                    row[key.item] += value;
            result.effective.weights.emplace(values.id, std::move(row));
        }
        return result;
    }

    // What the callables of the weighted equations add to a product: with δu per subcase
    // (`shifts`), δL_k = ∂L_k/∂x·v + c_k·δu of every argument value, the callables' (w_x, w_L), then the terms
    // w_x·∂var/∂x + Σ_k w_Lk·∂L_k/∂x (`values`) and the right-hand sides Σ_k w_Lk·c_k of δλ per subcase (`rates`).
    struct EquationTerms
    {
        std::vector<double> values;
        std::map<std::int32_t, std::vector<double>> rates;
    };
    [[nodiscard]] EquationTerms equationTerms(const ProductWeights& weights, const EquationWeights& equations,
                                              std::span<const double> v,
                                              const std::map<std::int32_t, std::vector<double>>& shifts) const
    {
        EquationTerms result;
        result.values.assign(m_variableCount, 0.0);
        KnownResponses known;
        for (const ResponseValues& values : *m_in.values)
            known[values.id] = &values;
        ResultLookup lookup(m_in.results->arrays);
        ResponseContext context;
        context.snapshot = m_in.overlaid.get();
        context.snapshotOwner = m_in.overlaid;
        context.results = &lookup;
        context.model = m_in.model;
        context.design = m_in.design;
        context.massFormulation = m_in.massFormulation;
        context.budget = &m_budget;
        const auto dotRow = [&](const auto& row)
        {
            double sum = 0.0;
            for (const auto& [column, value] : row)
                sum += value * v[column];
            return sum;
        };
        const auto directionOf = [&](const ArgumentSource& source) -> double
        {
            switch (source.kind)
            {
            case ArgumentSource::Kind::Response:
            {
                const ItemLinearisation& item = m_linearised.at({m_positionOf.at(source.responseId), source.index});
                double move = dotRow(item.explicitPart);
                if (item.subcase != 0)
                    if (const auto found = shifts.find(item.subcase); found != shifts.end())
                        move += dotSparse(item.weights, found->second);
                return move;
            }
            case ArgumentSource::Kind::Variable:
                return dotRow(m_in.design->variables.derivatives[source.index]);
            case ArgumentSource::Kind::Field:
                return dotRow(m_in.design->fields[source.index].gradient);
            case ArgumentSource::Kind::Coordinate: // shapes are refused (requireSecondOrder)
            case ArgumentSource::Kind::Constant:
                return 0.0;
            }
            return 0.0;
        };
        const WeightRows rows = rowsOf(weights);
        std::map<ItemKey, double> loadProducts; // w_L by argument item
        for (const std::size_t p : equations.equations)
        {
            const ResponseValues& values = valuesAt(p);
            std::vector<double> itemWeights(values.itemCount(), 0.0); // bounded by the response's items
            static_cast<void>(rows.forEach(p, 0,
                                           [&](std::size_t i, double y)
                                           {
                                               if (m_linearised.items[p][i].isWanted)
                                                   itemWeights[i] = y;
                                           }));
            const EquationResponse& equation = *equationOf(values.id);
            for (const auto& [source, w] :
                 equation.secondOrderTerms(m_in.responseSubcases->at(values.id), known, context, *m_in.callables,
                                           m_in.blockItems, itemWeights, directionOf))
                switch (source.kind)
                {
                case ArgumentSource::Kind::Response:
                    loadProducts[{m_positionOf.at(source.responseId), source.index}] += w;
                    break;
                case ArgumentSource::Kind::Variable:
                    for (const auto& [column, derivative] : m_in.design->variables.derivatives[source.index])
                        result.values[column] += w * derivative;
                    break;
                case ArgumentSource::Kind::Field:
                    for (const auto& [column, derivative] : m_in.design->fields[source.index].gradient)
                        result.values[column] += w * derivative;
                    break;
                case ArgumentSource::Kind::Coordinate:
                case ArgumentSource::Kind::Constant:
                    break;
                }
        }
        for (const auto& [key, w] : loadProducts)
        {
            const ItemLinearisation& item = m_linearised.at(key);
            for (const auto& [column, value] : item.explicitPart)
                result.values[column] += w * value;
            if (item.subcase == 0 || item.weights.empty())
                continue;
            std::vector<double>& rate = result.rates[item.subcase];
            if (rate.empty())
                rate.assign(gSize(), 0.0);
            for (const auto& [dof, value] : item.weights)
                rate[static_cast<std::size_t>(dof)] += w * value;
        }
        return result;
    }

    // The Hessian-vector product along v from the element blocks: δu_s = K⁻¹·b_v, δp = ∂p/∂x·v, the loads
    // and terms of the blocks, δλ_s = K⁻¹·loads_s, (H·v)_j = Σ_p ∂p/∂x_j·terms_p + Σ_s δλ_sᵀ·b_js. Two solves per
    // subcase; the rest is linear in the blocks' size. Same bits for the same blocks and v (one thread, fixed order).
    // `equations`: the terms of the equations of the second order from δu, added to δλ's right-hand
    // sides and to the product (null: none).
    using EquationHook = std::function<EquationTerms(const std::map<std::int32_t, std::vector<double>>&)>;
    void blockProduct(const detail::HessianBlocks& blocks, std::span<const double> v, std::vector<double>& values,
                      ProductReport& report, const EquationHook* equations = nullptr) const
    {
        std::map<std::int32_t, std::vector<double>> shifts = solve(directionLoads(v), report.solves, Reads::Direction);
        std::optional<EquationTerms> equationTerms;
        if (equations != nullptr)
            equationTerms = (*equations)(shifts);
        const std::vector<FieldValue>& fields = m_in.design->fields;
        std::vector<double> fieldMoves(fields.size(), 0.0); // bounded by the relations
        for (std::size_t r = 0; r < fields.size(); ++r)
            for (const auto& [column, derivative] : fields[r].gradient)
                fieldMoves[r] += derivative * v[column];
        std::vector<std::vector<double>> moves(m_subcases.size()); // bounded by the subcases
        std::vector<std::vector<double>> loads(m_subcases.size()); // bounded by the subcases
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            if (const auto found = shifts.find(m_subcases[s].subcase.label); found != shifts.end())
                moves[s] = std::move(found->second);
            loads[s].assign(gSize(), 0.0); // a g-set vector
        }
        std::vector<double> terms(fields.size(), 0.0); // bounded by the relations
        blocks.apply(moves, fieldMoves, loads, terms);
        if (equationTerms)
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
                if (const auto found = equationTerms->rates.find(m_subcases[s].subcase.label);
                    found != equationTerms->rates.end())
                    for (std::size_t g = 0; g < gSize(); ++g)
                        loads[s][g] += found->second[g];
        std::map<std::int32_t, std::vector<double>> rates;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (std::any_of(loads[s].begin(), loads[s].end(), [](double value) { return value != 0.0; }))
                rates.emplace(m_subcases[s].subcase.label, std::move(loads[s]));
        values = adjointProducts(solve(rates, report.solves));
        for (std::size_t r = 0; r < fields.size(); ++r)
            if (terms[r] != 0.0)
                for (const auto& [column, derivative] : fields[r].gradient)
                    values[column] += derivative * terms[r];
        if (equationTerms)
            for (std::size_t j = 0; j < values.size(); ++j)
                values[j] += equationTerms->values[j];
    }

    // The equation response of ID `id` (null: another kind).
    [[nodiscard]] const EquationResponse* equationOf(std::int32_t id) const
    {
        for (const EquationResponse& equation : m_in.responses->equations())
            if (equation.definition().id == id)
                return &equation;
        return nullptr;
    }

    // Refuses an equation response without the second-order callable (before any solve).
    void requireSecondOrderCallable(const EquationResponse& equation) const
    {
        const EquationResponseDefinition& definition = equation.definition();
        const auto found = m_in.callables->find(definition.callable);
        if (definition.form != EquationForm::Callable || found == m_in.callables->end() || found->second == nullptr ||
            !found->second->hasHessianVector())
            throw base::OptimisationError(base::concatenate(
                {"hessian_vector: response ", std::to_string(definition.id), " (", definition.label,
                 ") is an equation response without hessian_vector= (its second derivatives in "
                 "its variables and loads are the callable's); declare it with add_equation_response(..., function=, "
                 "gradient=, hessian_vector=), or give weights on the responses it reads"}));
    }

    // The weights of the section loads on their members' force columns (hessianBlocks): per member element
    // and subcase, its section family and Σ_k y_k·C_m(k, l) over its force columns l (output::SectionLoads::
    // memberCoefficients), the weighted items in their order.
    struct SectionMember
    {
        output::SectionFamily family = output::SectionFamily::Shell;
        std::vector<double> weights; // over the family's components
    };
    using SectionMemberWeights = std::map<std::pair<ElementRef, std::size_t>, SectionMember>;
    [[nodiscard]] SectionMemberWeights sectionMemberWeights(
        const std::map<ItemKey, double>& weightOf,
        const std::map<ItemKey, const detail::SectionItem*>& sectionItemOf) const
    {
        SectionMemberWeights result;
        std::map<std::pair<std::int32_t, std::size_t>, std::vector<output::SectionLoads::Coefficients>> coefficientsOf;
        for (const auto& [key, weight] : weightOf)
        {
            const auto found = sectionItemOf.find(key);
            if (found == sectionItemOf.end() || weight == 0.0)
                continue;
            const detail::SectionItem& item = *found->second;
            const output::SectionLoads& loads = *m_sectionRows.at(item.response).sections;
            const output::SectionLoads::Section& section = loads.sections()[item.index];
            auto coefficients = coefficientsOf.find({item.response, item.index});
            if (coefficients == coefficientsOf.end())
                coefficients =
                    coefficientsOf
                        .emplace(std::make_pair(item.response, item.index), loads.memberCoefficients(item.index, m_set))
                        .first;
            const std::size_t count = loads.componentCount(item.index);
            for (std::size_t m = 0; m < section.members.size(); ++m)
            {
                const output::SectionLoads::Member& member = section.members[m];
                SectionMember& entry = result[{ElementRef{member.batch, member.row}, item.subcase}];
                entry.family = section.family;
                entry.weights.resize(count, 0.0);
                for (std::size_t l = 0; l < count; ++l)
                    entry.weights[l] +=
                        weight * coefficients->second[m][item.component * output::MAX_SECTION_COMPONENTS + l];
            }
        }
        return result;
    }

    // What the blocks of one element share between its subcases (hessianBlocks): the slopes of its items (for every
    // job of the element) and its stiffness rates in its designed field, formed by the first block that needs them.
    struct ElementParts
    {
        ItemSlopeCache slopes;
        bool hasStiffnessRates = false;
        bool areStiffnessRatesClosed = false;
        elements::ElementMatrixBlock firstStiffnessRate;
        elements::ElementMatrixBlock secondStiffnessRate;
    };

    // The applied-load contribution to the element's Lagrangian curvature: lambda^T P_pq.
    // Fixed geometry makes P independent of displacement, so there is no extra displacement/field block.
    [[nodiscard]] std::optional<double> loadCurvatureOf(const ElementRef& element, const detail::SubcaseState& state,
                                                       const elements::DesignedField& first,
                                                       const elements::DesignedField& second,
                                                       std::span<const double> adjoint) const
    {
        const auto& batch = m_set.batches()[element.batch];
        const auto& family = *batch.family;
        const auto a = family.designFieldUse(first), b = family.designFieldUse(second);
        elements::ElementLoadContext context;
        if ((a.changesLoads || b.changesLoads) && !family.isThermallyInert() && state.temperatures)
            context.temperatures = &state.temperatures->temperatures();
        if (a.changesDistributedLoads || b.changesDistributedLoads)
            if (const auto found = state.elementLoads.find(element); found != state.elementLoads.end())
            {
                context.pressures = found->second.first;
                context.lineLoads = found->second.second;
            }
        if (!context.temperatures && context.pressures.empty() && context.lineLoads.empty())
            return 0.0;
        if (state.isCombination())
            return std::nullopt; // Combined applied-load curvature needs the weighted loads of its source cases.
        elements::ElementVectorBlock rate;
        if (!family.loadFieldMixedDerivative(batch.view, element.row, first, second, context, rate))
            return std::nullopt;
        double result = 0.0;
        for (std::size_t k = 0; k < rate.size(); ++k)
            for (std::size_t i = 0; i < rate.dofs(k).size(); ++i)
                result += rate.values(k)[i] * adjoint[static_cast<std::size_t>(m_dofMap.dofOf(rate.dofs(k)[i]))];
        return result;
    }

    // The block of one element in one subcase (hessianBlocks), or the reason the closed forms do not give it. `copy`
    // the piece's copy of the displacements for the recoveries at unit displacements of a family without slopes;
    // `parts` what the element's blocks of its other subcases formed already.
    [[nodiscard]] std::string hessianBlockOf(const ElementRef& element, std::size_t subcase,
                                             const std::vector<std::size_t>& jobs,
                                             std::size_t field,
                                             const std::map<ItemKey, double>& weightOf,
                                             const std::vector<double>& adjoint, const SectionMember* section,
                                             std::optional<base::BudgetedBuffer<double>>& copy, ElementParts& parts,
                                             detail::HessianBlocks::Block& block) const
    {
        const assembly::AssemblyBatch& batch = m_set.batches()[element.batch];
        const elements::ElementFamily& family = *batch.family;
        const elements::ElementBatchView& view = batch.view;
        const detail::SubcaseState& state = m_subcases[subcase];
        const auto name = [&]
        { return base::concatenate({view.card(), " ", std::to_string(view.elementId(element.row))}); };
        block.subcase = subcase;
        std::vector<elements::DesignedField> designed; // at most one
        if (field != detail::HessianBlocks::NO_FIELD)
        {
            block.field = field;
            const DesignRelation& relation = m_in.model->relations()[block.field];
            designed.push_back({relation.targetCard, relation.field, relation.targetId, relation.fieldItem});
        }
        const auto inField = [&]
        {
            return designed.empty() ? std::string()
                                    : base::concatenate({" in ", designed.front().card, " ", designed.front().column});
        };
        const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap.gridCount());
        const auto scratch = [&]
        {
            if (!copy)
                copy.emplace(m_budget, gSize(), "Hessian blocks scratch displacements");
            std::copy(state.displacements.data(), state.displacements.data() + gSize(), copy->data());
            return ScratchDisplacements{copy->span().first(gridValues), copy->span().subspan(gridValues)};
        };
        // The DOFs of the block: the union of those of its parts, in the order met (the element's own order).
        std::vector<elements::ElementDof> dofs; // bounded by the element's DOFs
        const auto placeOf = [&](const elements::ElementDof& dof) -> std::size_t
        {
            const auto found = std::find(dofs.begin(), dofs.end(), dof);
            if (found != dofs.end())
                return static_cast<std::size_t>(found - dofs.begin());
            const std::size_t n = dofs.size();
            dofs.push_back(dof);
            if (!designed.empty())
                block.mixed.push_back(0.0);
            if (!block.displacement.empty())
            {
                std::vector<double> grown((n + 1) * (n + 1), 0.0); // bounded by the element's DOFs
                for (std::size_t r = 0; r < n; ++r)
                    std::copy_n(block.displacement.begin() + static_cast<std::ptrdiff_t>(r * n), n,
                                grown.begin() + static_cast<std::ptrdiff_t>(r * (n + 1)));
                block.displacement = std::move(grown);
            }
            return n;
        };
        // The second derivatives of weighted items of the element added to the block.
        const auto addItems = [&](const ItemSecondOrder& items)
        {
            const std::size_t n = items.dofs.size();
            std::vector<std::size_t> at(n); // bounded by the element's DOFs
            for (std::size_t d = 0; d < n; ++d)
                at[d] = placeOf(items.dofs[d]);
            if (!items.displacementBlock.empty())
            {
                if (block.displacement.empty())
                    block.displacement.assign(dofs.size() * dofs.size(), 0.0);
                for (std::size_t d = 0; d < n; ++d)
                    for (std::size_t e = 0; e < n; ++e)
                        block.displacement[at[d] * dofs.size() + at[e]] += items.displacementBlock[d * n + e];
            }
            if (!designed.empty())
            {
                for (std::size_t d = 0; d < n; ++d)
                    block.mixed[at[d]] += items.mixedBlock[d];
                block.fieldTerm += items.fieldBlock[0];
            }
        };
        // The force items of the element as a member of the weighted section loads.
        if (section != nullptr)
        {
            std::vector<ElementItemSpec> specs; // bounded by the family's force columns
            std::vector<double> sectionWeights; // the same
            for (std::size_t l = 0; l < section->weights.size(); ++l)
            {
                if (section->weights[l] == 0.0)
                    continue;
                const std::int32_t code = sectionForceItemCode(section->family, view.card(), l);
                const auto located = code == 0 ? std::variant<ItemLocation, std::string>(std::string())
                                               : locateItem(ResponseType::Force, view.card(), code);
                const ItemLocation* location = std::get_if<ItemLocation>(&located);
                if (location == nullptr)
                    return base::concatenate({"the section loads of ", name(),
                                              " read a force column that is no FORCE "
                                              "item, whose closed second derivatives are not formed"});
                specs.push_back({ResponseType::Force, code, *location, 0});
                sectionWeights.push_back(section->weights[l]);
            }
            if (!specs.empty())
            {
                const ElementItemFunction forces(family, element.row, std::move(specs));
                const std::optional<ItemSecondOrder> items =
                    forces.secondOrder(view, state.nodal(m_dofMap), sectionWeights, designed, scratch, &parts.slopes);
                if (!items)
                    return base::concatenate({"the section loads of ", name(), " have no closed second derivatives",
                                              inField(), " in this version"});
                addItems(*items);
            }
        }
        // The items of the element's jobs.
        for (const std::size_t j : jobs)
        {
            const detail::ElementJob& job = m_jobs[j];
            std::vector<double> itemWeights(job.keys.size(), 0.0);
            for (std::size_t k = 0; k < job.keys.size(); ++k)
                if (const auto found = weightOf.find(job.keys[k]); found != weightOf.end())
                    itemWeights[k] = found->second;
            const std::optional<ItemSecondOrder> items =
                job.function->secondOrder(view, state.nodal(m_dofMap, job.element), itemWeights,
                                           designed, scratch, &parts.slopes);
            if (!items)
                return base::concatenate({"the weighted items of ", name(), " have no closed second derivatives",
                                          inField(), " in this version"});
            addItems(*items);
        }
        // The stiffness and load terms of the field with the adjoint: −(∂K/∂p)·λ̃ and λ̃ᵀ·(∂²P/∂p² − (∂²K/∂p²)·u).
        if (!designed.empty() && !adjoint.empty())
        {
            const elements::FieldUse use = family.designFieldUse(designed.front());
            const auto loadTerm = loadCurvatureOf(element, state, designed.front(), designed.front(), adjoint);
            if (!loadTerm)
                return base::concatenate({name(), " has no closed applied-load curvature", inField()});
            block.fieldTerm += *loadTerm;
            const auto valueAt = [](const std::vector<double>& vector, base::DofIndex dof)
            { return vector[static_cast<std::size_t>(dof)]; };
            if (use.changesStiffness && family.facets().has(elements::Facet::Stiffness))
            {
                if (!parts.hasStiffnessRates)
                {
                    parts.areStiffnessRatesClosed = family.stiffnessFieldDerivative(view, element.row, designed.front(),
                                                                                    parts.firstStiffnessRate) &&
                                                    family.stiffnessFieldSecondDerivative(
                                                        view, element.row, designed.front(), parts.secondStiffnessRate);
                    parts.hasStiffnessRates = true;
                }
                if (!parts.areStiffnessRatesClosed)
                    return base::concatenate({name(), " has no closed second derivative of its stiffness in ",
                                              designed.front().card, " ", designed.front().column});
                const elements::ElementMatrixBlock& first = parts.firstStiffnessRate;
                const elements::ElementMatrixBlock& second = parts.secondStiffnessRate;
                if (!first.empty())
                {
                    const std::span<const elements::ElementDof> partDofs = first.dofs(0);
                    const std::span<const double> matrix = first.matrix(0);
                    const std::size_t n = partDofs.size();
                    for (std::size_t r = 0; r < n; ++r)
                    {
                        double sum = 0.0;
                        for (std::size_t c = 0; c < n; ++c)
                            sum += matrix[r * n + c] * valueAt(adjoint, m_dofMap.dofOf(partDofs[c]));
                        block.mixed[placeOf(partDofs[r])] -= sum;
                    }
                }
                if (!second.empty())
                {
                    const std::span<const elements::ElementDof> partDofs = second.dofs(0);
                    const std::span<const double> matrix = second.matrix(0);
                    const std::size_t n = partDofs.size();
                    double product = 0.0;
                    for (std::size_t r = 0; r < n; ++r)
                    {
                        double sum = 0.0;
                        for (std::size_t c = 0; c < n; ++c)
                            sum += matrix[r * n + c] *
                                   state.displacements[static_cast<std::size_t>(m_dofMap.dofOf(partDofs[c]))];
                        product += valueAt(adjoint, m_dofMap.dofOf(partDofs[r])) * sum;
                    }
                    block.fieldTerm -= product;
                }
            }
            if (use.changesMass && family.facets().has(elements::Facet::Mass) && state.hasInertialLoads &&
                state.subcase.load)
            {
                // P = WTMASS·M·a (InertialLoads, addInertialLoadRate): ∂²P/∂p² = WTMASS·(∂²M/∂p²)·a.
                const InertialRate& rate = *m_inertialRates[subcase];
                elements::MassOptions options;
                options.formulation = rate.options.formulation;
                elements::ElementMatrixBlock second;
                if (!family.massFieldSecondDerivative(view, element.row, designed.front(), options, second))
                    return base::concatenate({name(), " has no closed second derivative of its mass in ",
                                              designed.front().card, " ", designed.front().column});
                if (!second.empty())
                {
                    const std::span<const elements::ElementDof> partDofs = second.dofs(0);
                    const std::span<const double> matrix = second.matrix(0);
                    const std::size_t n = partDofs.size();
                    double product = 0.0;
                    for (std::size_t r = 0; r < n; ++r)
                    {
                        double sum = 0.0;
                        for (std::size_t c = 0; c < n; ++c)
                            sum += matrix[r * n + c] *
                                   rate.accelerations[static_cast<std::size_t>(m_dofMap.dofOf(partDofs[c]))];
                        product += valueAt(adjoint, m_dofMap.dofOf(partDofs[r])) * sum;
                    }
                    block.fieldTerm += rate.options.weightToMass * product;
                }
            }
        }
        block.dofs.resize(dofs.size());
        for (std::size_t d = 0; d < dofs.size(); ++d)
            block.dofs[d] = m_dofMap.dofOf(dofs[d]);
        return {};
    }

    // The off-diagonal field term of one element's Lagrangian. The displacement and field/displacement terms
    // are already in its diagonal blocks; this block stores only A_pq = A_qp.
    [[nodiscard]] std::string hessianFieldCouplingOf(const ElementRef& element, std::size_t subcase,
                                                     const std::vector<std::size_t>& jobs,
                                                     std::size_t first, std::size_t second,
                                                     const std::map<ItemKey, double>& weightOf,
                                                     const std::vector<double>& adjoint, const SectionMember* section,
                                                     detail::HessianBlocks::Block& block) const
    {
        const auto& batch = m_set.batches()[element.batch];
        const auto& family = *batch.family;
        const auto& view = batch.view;
        const auto& state = m_subcases[subcase];
        const auto fieldAt = [&](std::size_t index)
        {
            const DesignRelation& relation = m_in.model->relations()[index];
            return elements::DesignedField{relation.targetCard, relation.field, relation.targetId, relation.fieldItem};
        };
        const auto a = fieldAt(first), b = fieldAt(second);
        const auto unavailable = [&]
        {
            return base::concatenate({view.card(), " ", std::to_string(view.elementId(element.row)),
                                       " has no closed mixed derivative in ", a.card, " ", a.column, " and ",
                                       b.card, " ", b.column});
        };
        block.subcase = subcase;
        block.field = first;
        block.otherField = second;
        for (const std::size_t j : jobs)
        {
            const auto& job = m_jobs[j];
            std::vector<double> weights(job.keys.size(), 0.0);
            for (std::size_t k = 0; k < job.keys.size(); ++k)
                if (const auto found = weightOf.find(job.keys[k]); found != weightOf.end())
                    weights[k] = found->second;
            const auto rate = job.function->mixedFieldRate(view, state.nodal(m_dofMap, job.element), weights, a, b);
            if (!rate)
                return unavailable();
            block.fieldTerm += *rate;
        }
        if (section != nullptr)
        {
            std::vector<ElementItemSpec> specs;
            std::vector<double> weights;
            for (std::size_t component = 0; component < section->weights.size(); ++component)
            {
                if (section->weights[component] == 0.0)
                    continue;
                const std::int32_t code = sectionForceItemCode(section->family, view.card(), component);
                const auto location = locateItem(ResponseType::Force, view.card(), code);
                const auto* item = std::get_if<ItemLocation>(&location);
                if (item == nullptr)
                    return unavailable();
                specs.push_back({ResponseType::Force, code, *item, 0});
                weights.push_back(section->weights[component]);
            }
            const ElementItemFunction function(family, element.row, std::move(specs));
            const auto rate = function.mixedFieldRate(view, state.nodal(m_dofMap), weights, a, b);
            if (!rate)
                return unavailable();
            block.fieldTerm += *rate;
        }
        if (adjoint.empty())
            return {};
        const auto loadTerm = loadCurvatureOf(element, state, a, b, adjoint);
        if (!loadTerm)
            return unavailable();
        block.fieldTerm += *loadTerm;
        const auto product = [&](const elements::ElementMatrixBlock& matrices, std::span<const double> right)
        {
            double value = 0.0;
            for (std::size_t k = 0; k < matrices.size(); ++k)
            {
                const auto dofs = matrices.dofs(k);
                const auto matrix = matrices.matrix(k);
                for (std::size_t i = 0; i < dofs.size(); ++i)
                    for (std::size_t j = 0; j < dofs.size(); ++j)
                        value += adjoint[static_cast<std::size_t>(m_dofMap.dofOf(dofs[i]))] *
                                 matrix[i * dofs.size() + j] * right[static_cast<std::size_t>(m_dofMap.dofOf(dofs[j]))];
            }
            return value;
        };
        const auto useA = family.designFieldUse(a), useB = family.designFieldUse(b);
        if (family.facets().has(elements::Facet::Stiffness) && useA.changesStiffness && useB.changesStiffness)
        {
            elements::ElementMatrixBlock rate;
            if (!family.stiffnessFieldMixedDerivative(view, element.row, a, b, rate))
                return unavailable();
            block.fieldTerm -= product(rate, state.displacements.span());
        }
        if (family.facets().has(elements::Facet::Mass) && useA.changesMass && useB.changesMass &&
            state.hasInertialLoads && state.subcase.load)
        {
            const InertialRate& inertial = *m_inertialRates[subcase];
            elements::MassOptions options;
            options.formulation = inertial.options.formulation;
            elements::ElementMatrixBlock rate;
            if (!family.massFieldMixedDerivative(view, element.row, a, b, options, rate))
                return unavailable();
            block.fieldTerm += inertial.options.weightToMass * product(rate, inertial.accelerations.span());
        }
        return {};
    }

    // Refuses what the Hessian-vector products do not differentiate (OptimisationError naming it): the responses
    // whose second derivatives the linearisation at a perturbed state does not give (equations, external, contour,
    // section and cut loads, self-adjoint and modal responses: their terms read more than the point and the
    // displacements), the shape variables, the condensed path and inertia relief.
    void requireSecondOrder(const ProductWeights& weights, bool asksSupplied) const
    {
        const auto refuse = [](const std::string& what)
        {
            throw base::OptimisationError(base::concatenate(
                {"hessian_vector: ", what,
                 "; this version forms the Hessian-vector products of the displacement, element, SPC force, weight, "
                 "volume, section and cut load responses and of the equations with hessian_vector=: give weights on "
                 "those only, or difference vjp instead"}));
        };
        const auto isSupported = [](ResponseType type)
        {
            switch (type)
            {
            case ResponseType::Weight:
            case ResponseType::Volume:
            case ResponseType::Displacement:
            case ResponseType::Stress:
            case ResponseType::Strain:
            case ResponseType::Force:
            case ResponseType::SpcForce:
            case ResponseType::StrainEnergy:
            case ResponseType::CompositeStress:
            case ResponseType::CompositeStrain:
            case ResponseType::CompositeFailure:
            case ResponseType::CompositeStrengthRatio:
            case ResponseType::SectionLoad: // differenced at the moved states
            case ResponseType::CutLoad:
                return true;
            default:
                return false;
            }
        };
        std::set<std::int32_t> weighted;
        for (const auto& [id, values] : weights.weights)
            weighted.insert(id);
        for (const auto& [id, values] : weights.sparseWeights)
            weighted.insert(id);
        for (const std::int32_t id : weighted)
        {
            const DesignResp1Response* response = m_in.responses->findDesignResp1(id);
            if (response == nullptr)
                refuse(base::concatenate({"response ", std::to_string(id),
                                          " is an external response (or an equation that reads one), whose second "
                                          "derivatives are not formed"}));
            if (!isSupported(response->definition().type))
                refuse(base::concatenate({"response ", std::to_string(id), " is of type ",
                                          std::string(responseTypeName(response->definition().type))}));
        }
        if (!m_in.design->shapes.empty())
            refuse("the design has shape variables, whose second derivatives are not formed");
        for (std::size_t r = 0; r < m_in.model->relations().size() && !asksSupplied; ++r)
            if (const DesignRelation& relation = m_in.model->relations()[r]; relation.form == RelationForm::Supplied)
                refuse(base::concatenate({"the relation ", relation.label(),
                                          " is given by a function of the caller, whose values at the moved points "
                                          "are not asked for (ProductContext::suppliedAt)"}));
        for (const detail::SubcaseState& state : m_subcases)
        {
            if (state.isAeroelastic)
                refuse(base::concatenate({"subcase ", std::to_string(state.subcase.label), " is a SAERO subcase"}));
        }
    }

    // The linearisation of every item the options ask for (once: the products then only read it).
    void prepare() { linearise(); }

    // The factorisations made so far for the sensitivities (the constraint sets the analysis did not keep).
    [[nodiscard]] std::int32_t factorisationCount() const
    {
        std::int32_t count = 0;
        for (const std::unique_ptr<SensitivitySystem>& system : m_systems)
            count += system->factorisationCount();
        return count;
    }

    // The memory the linearisation holds beside what it reserved in the budget already (the displacements of the
    // subcases; the factors are the analysis'): the items' weights and explicit parts and the pseudo-loads, the
    // vectors that grow with the model, estimated from their sizes.
    [[nodiscard]] std::uint64_t heldBytes() const
    {
        constexpr std::uint64_t ENTRY_BYTES = sizeof(std::pair<base::DofIndex, double>);
        std::uint64_t entries = 0;
        std::uint64_t items = 0;
        for (const std::vector<ItemLinearisation>& response : m_linearised.items)
        {
            items += response.size();
            for (const ItemLinearisation& item : response)
                entries +=
                    item.weights.size() + item.explicitPart.size() + item.ownExplicit.size() + item.arguments.size();
        }
        for (std::size_t q = 0; q < m_quantities.size(); ++q)
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
                entries += m_pseudoLoads->of(q, s).size() + m_pseudoLoads->loadPartOf(q, s).size() +
                           m_pseudoLoads->appliedLoadPartOf(q, s).size();
        return entries * ENTRY_BYTES + items * sizeof(ItemLinearisation);
    }

    // A right-hand side of the products of a vjp: the row `row` of the block in the subcase of index `subcase`.
    struct ProductColumn
    {
        std::size_t subcase = 0;
        std::size_t row = 0;
    };

    // The reads of every full-path system of the subcases (none on the condensed path), by system (AdjointColumns):
    // every DOF when `readsEvery` (the adjoint vectors kept whole).
    [[nodiscard]] std::map<const SensitivitySystem*, std::unique_ptr<AdjointColumns>> productReads(
        bool readsEvery) const
    {
        std::map<const SensitivitySystem*, std::vector<std::size_t>> subcasesOf;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (const SensitivitySystem& system = systemOf(m_subcases[s]); !system.isCondensed())
                subcasesOf[&system].push_back(s);
        std::map<const SensitivitySystem*, std::unique_ptr<AdjointColumns>> reads;
        for (const auto& [system, subcases] : subcasesOf)
            reads.emplace(system, std::make_unique<AdjointColumns>(*system, *m_pseudoLoads, subcases, readsEvery,
                                                                   m_budget, m_in.pool));
        return reads;
    }

    // The right-hand side of a column: Tᵀ·(Σ w·c) of the items its row weights in its subcase, written to `equations`.
    // On the full path from the items' entries alone: `scratch`, a g-set vector of zeros, receives them,
    // and AdjointColumns::reduceToEquations gives the values of toEquations() at every equation, in O(the items'
    // entries + Gm) instead of O(g set). Otherwise (the condensed path) toEquations().
    void columnLoad(const WeightRows& rows, std::size_t row, std::int32_t label, const SensitivitySystem& system,
                    const AdjointColumns* reads, std::vector<double>& scratch, std::vector<base::DofIndex>& touched,
                    std::span<double> equations) const
    {
        const auto add = [&](std::size_t p, std::size_t i, double w)
        {
            const ItemLinearisation& item = m_linearised.items[p][i];
            if (!item.isWanted || item.subcase != label || !item.readsDisplacements())
                return;
            for (const auto& [dof, value] : item.weights)
            {
                scratch[static_cast<std::size_t>(dof)] += w * value;
                touched.push_back(dof);
            }
        };
        if (reads == nullptr)
        {
            std::fill(scratch.begin(), scratch.end(), 0.0);
            for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                rows.forEach(p, row, [&](std::size_t i, double w) { add(p, i, w); });
            system.toEquationsInPlace(scratch, equations);
            std::fill(scratch.begin(), scratch.end(), 0.0);
            touched.clear();
            return;
        }
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            rows.forEach(p, row, [&](std::size_t i, double w) { add(p, i, w); });
        reads->reduceToEquations(scratch, touched, equations);
    }

    // `buffer` holds at least `columns` right-hand sides of `equations` rows: made again only for another row count or
    // more columns, then with RIGHT_HAND_SIDE_SLACK columns of slack and zeroed on the pool (every column a solve uses
    // is written before it is solved).
    void reserveRightHandSides(linalg::DenseMatrix& buffer, std::size_t equations, std::size_t columns) const
    {
        if (buffer.rows() == equations && buffer.columns() >= columns)
            return;
        buffer = linalg::DenseMatrix(); // the memory back first
        buffer = linalg::DenseMatrix(m_budget, equations,
                                     base::pieceCount(columns, RIGHT_HAND_SIDE_SLACK) * RIGHT_HAND_SIDE_SLACK,
                                     "vjp adjoint loads", m_in.pool);
    }

    // The columns of a block of a vjp solved on `system` (their subcases on it), and their products added to `values`
    // (the block's rows × variables) in column order. The right-hand sides and the products are made on the pool,
    // column by column (a column's values do not depend on the others).
    // PERFORMANCE: the right-hand sides (a g-set vector reduced to the equations per column) and the products (the
    // column expanded to the g set, then dotted with every pseudo-load) took about half the time of the products on
    // the calling thread alone, O(g set) per column (on the 1/16 model of the gradient benchmarks: 3.3 s and 2.5 s
    // against 6.6 s of solves); on the pool and from the items' entries and at the read DOFs only (columnLoad,
    // AdjointColumns).
    // Measured: at 1/16, the right-hand sides 3.3 -> 0.39 s, the products 2.5 -> 0.71 s; the products by tiles of the
    // g set followed (AdjointColumns::addProducts).
    // `buffer`: the right-hand sides of the call's solves, kept from one solve to the next (allocated again only for
    // another row count or more columns: the first touch of its pages is paid once).
    void solveColumns(const WeightRows& rows, std::size_t first, std::span<const ProductColumn> columns,
                      const SensitivitySystem& system, std::map<std::int32_t, std::vector<double>>* adjoints,
                      AdjointColumns* reads, linalg::DenseMatrix& buffer, std::vector<double>& values,
                      ProductReport& report) const
    {
        const std::size_t n = m_variableCount;
        const auto equations = static_cast<std::size_t>(system.equationCount());
        auto clock = std::chrono::steady_clock::now();
        reserveRightHandSides(buffer, equations, columns.size());
        const linalg::DenseMatrixView view = buffer.view().block(0, 0, equations, columns.size());
        const std::size_t threads = m_in.pool != nullptr ? m_in.pool->threadCount() : 1;
        const std::size_t piece =
            std::max(PRODUCT_COLUMNS_PER_PIECE, base::pieceCount(columns.size(), PRODUCT_PIECES_PER_THREAD * threads));
        // On the full path the columns placed in the solve by the first equation of their items (solveSlots).
        std::vector<std::int32_t> firstEquations(reads != nullptr ? columns.size() : 0); // bounded by a solve
        for (std::size_t c = 0; c < firstEquations.size(); ++c)
        {
            firstEquations[c] = system.equationCount();
            const std::int32_t label = m_subcases[columns[c].subcase].subcase.label;
            for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                rows.forEach(p, first + columns[c].row,
                             [&](std::size_t i, double)
                             {
                                 const ItemLinearisation& item = m_linearised.items[p][i];
                                 if (item.isWanted && item.subcase == label && item.readsDisplacements())
                                     firstEquations[c] =
                                         std::min(firstEquations[c], reads->firstEquation(item.weights));
                             });
        }
        const std::vector<std::size_t> slot = AdjointColumns::solveSlots(firstEquations);
        const auto columnOf = [&](std::size_t c)
        {
            const std::size_t position = slot.empty() ? c : slot[c];
            return std::span<double>(view.data + position * view.leadingDimension, equations);
        };
        base::parallelFor(m_in.pool, columns.size(), piece,
                          [&](std::size_t begin, std::size_t end)
                          {
                              std::vector<double> scratch(gSize(), 0.0); // one per piece
                              std::vector<base::DofIndex> touched;       // the entries of a column's items
                              for (std::size_t c = begin; c < end; ++c)
                                  columnLoad(rows, first + columns[c].row, m_subcases[columns[c].subcase].subcase.label,
                                             system, reads, scratch, touched, columnOf(c));
                          });
        report.rightHandSideSeconds += lap(clock);
        if (equations > 0)
        {
            if (system.isCondensed())
            {
                // The condensed path recovers the adjoint vectors in the driven units only; one subcase.
                const base::BudgetedBuffer<std::uint8_t> unitReads =
                    SolutionReads::ofPseudoLoads(system, *m_pseudoLoads, columns.front().subcase, m_budget);
                system.solve(view, unitReads.span());
            }
            else
                system.solve(view);
        }
        report.solveSeconds += lap(clock);
        report.solves += static_cast<std::int64_t>(columns.size());
        if (reads != nullptr && !reads->readsEvery() && adjoints == nullptr)
        {
            const std::span<const base::DofIndex> readDofs = reads->readDofs();
            addProducts(*reads, columns, values,
                        [&](std::size_t firstRead, std::size_t endRead, base::DofIndex firstDof,
                            std::span<const std::size_t> pieceColumns, double* block, std::size_t stride)
                        {
                            for (std::size_t k = 0; k < pieceColumns.size(); ++k)
                            {
                                const std::span<const double> solved = columnOf(pieceColumns[k]);
                                for (std::size_t r = firstRead; r < endRead; ++r)
                                    block[static_cast<std::size_t>(readDofs[r] - firstDof) * stride + k] =
                                        reads->valueAtRead(solved, r);
                            }
                        });
            report.dotSeconds += lap(clock);
            return;
        }
        base::BudgetedBuffer<double> products(m_budget, columns.size() * n, "vjp products of a solve");
        std::vector<std::vector<double>> kept(adjoints != nullptr ? columns.size() : 0); // one per subcase
        base::parallelFor(m_in.pool, columns.size(), piece,
                          [&](std::size_t begin, std::size_t end)
                          {
                              std::vector<double> load(gSize()); // one per piece
                              for (std::size_t c = begin; c < end; ++c)
                              {
                                  if (reads != nullptr)
                                      reads->expand(columnOf(c), load);
                                  else
                                      system.fromEquations(columnOf(c), load);
                                  if (adjoints != nullptr)
                                      kept[c] = load;
                                  if (n > 0)
                                      m_pseudoLoads->dotAll(columns[c].subcase, load,
                                                            products.span().subspan(c * n, n));
                              }
                          });
        for (std::size_t c = 0; c < columns.size(); ++c)
        {
            if (adjoints != nullptr)
                (*adjoints)[m_subcases[columns[c].subcase].subcase.label] = std::move(kept[c]);
            const std::size_t row = columns[c].row;
            for (std::size_t j = 0; j < n; ++j)
                values[row * n + j] += products[c * n + j];
        }
        report.dotSeconds += lap(clock);
    }

    // The products of the solved columns `columns` with their subcases' pseudo-loads added to their rows of `values` in
    // column order, by tiles of the g set (AdjointColumns::addProducts; `fill` gives the columns' values at a tile's
    // read DOFs): the bits of dotAll per column, then the additions in column order.
    void addProducts(AdjointColumns& reads, std::span<const ProductColumn> columns, std::vector<double>& values,
                     const AdjointColumns::TileFill& fill) const
    {
        const std::size_t n = m_variableCount;
        std::vector<std::size_t> subcases(columns.size()); // bounded by a solve's columns
        std::vector<double*> destinations(columns.size()); // the same
        for (std::size_t c = 0; c < columns.size(); ++c)
        {
            subcases[c] = columns[c].subcase;
            destinations[c] = values.data() + columns[c].row * n;
        }
        reads.addProducts(subcases, destinations, fill);
    }

    // The items whose derivatives with respect to the displacements are the same vector (the same DOFs, the same bits
    // of the values), numbered by class (shared item adjoints): per response the class of each item, −1
    // for an item that reads no displacement or is not wanted; per class its first item (response, item).
    struct ItemClasses
    {
        std::vector<std::vector<std::int32_t>> classOf;
        std::vector<std::pair<std::size_t, std::size_t>> first;
    };

    [[nodiscard]] ItemClasses itemClasses() const
    {
        ItemClasses classes;
        classes.classOf.resize(m_linearised.items.size());
        std::unordered_map<std::uint64_t, std::vector<std::int32_t>> byHash; // bounded by the items
        const auto isSame = [](const SparseVector& a, const SparseVector& b)
        {
            return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
                                                      [](const auto& x, const auto& y)
                                                      {
                                                          return x.first == y.first &&
                                                                 std::bit_cast<std::uint64_t>(x.second) ==
                                                                     std::bit_cast<std::uint64_t>(y.second);
                                                      });
        };
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
        {
            classes.classOf[p].assign(m_linearised.items[p].size(), -1);
            for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
            {
                const ItemLinearisation& item = m_linearised.items[p][i];
                if (!item.isWanted || !item.readsDisplacements())
                    continue;
                // FNV-1a over the entries' bits.
                std::uint64_t hash = 14695981039346656037ULL;
                for (const auto& [dof, value] : item.weights)
                    for (const std::uint64_t word :
                         {static_cast<std::uint64_t>(dof), std::bit_cast<std::uint64_t>(value)})
                        hash = (hash ^ word) * 1099511628211ULL;
                std::vector<std::int32_t>& candidates = byHash[hash];
                std::int32_t found = -1;
                for (const std::int32_t k : candidates)
                {
                    const auto [q, j] = classes.first[static_cast<std::size_t>(k)];
                    if (isSame(m_linearised.items[q][j].weights, item.weights))
                    {
                        found = k;
                        break;
                    }
                }
                if (found < 0)
                {
                    found = static_cast<std::int32_t>(classes.first.size());
                    classes.first.emplace_back(p, i);
                    candidates.push_back(found);
                }
                classes.classOf[p][i] = found;
            }
        }
        return classes;
    }

    // The items of the columns of a solve by class (solveSharedColumns): per column its (class in `classes`, weight)
    // terms at [start[c], start[c + 1]), in the order the row's weights are read; `classes`: the distinct classes, in
    // the order of their first term.
    struct ColumnTerms
    {
        std::vector<std::size_t> start;
        std::vector<std::int32_t> localClass;
        std::vector<double> weight;
        std::vector<std::int32_t> classes;
    };

    [[nodiscard]] ColumnTerms columnTerms(const WeightRows& rows, std::size_t first,
                                          std::span<const ProductColumn> columns, const ItemClasses& classes) const
    {
        ColumnTerms terms;
        std::unordered_map<std::int32_t, std::int32_t> localOf; // bounded by the columns' items
        terms.start.push_back(0);
        for (const ProductColumn& column : columns)
        {
            const std::int32_t label = m_subcases[column.subcase].subcase.label;
            for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                rows.forEach(p, first + column.row,
                             [&](std::size_t i, double w)
                             {
                                 const ItemLinearisation& item = m_linearised.items[p][i];
                                 if (!item.isWanted || item.subcase != label || !item.readsDisplacements())
                                     return;
                                 const std::int32_t k = classes.classOf[p][i];
                                 const auto [at, isNew] =
                                     localOf.try_emplace(k, static_cast<std::int32_t>(terms.classes.size()));
                                 if (isNew)
                                     terms.classes.push_back(k);
                                 terms.localClass.push_back(at->second);
                                 terms.weight.push_back(w);
                             });
            terms.start.push_back(terms.localClass.size());
        }
        return terms;
    }

    // The columns of a block of a vjp from the adjoint vectors of their items' classes (shared item adjoints): the
    // classes' derivative vectors solved (one column each), expanded at the read DOFs, a column's adjoint
    // vector there the sum of its terms (λ = Σ w·λ_class, in the order of `terms`), dotted with its subcase's
    // pseudo-loads; the products added to `values` in column order. The full path, `reads` not every DOF.
    // PERFORMANCE: the combination and the dots by tiles of the g set (AdjointColumns::addProducts): a
    // piece's columns combined at a tile's read DOFs from the classes' vectors there, then dotted with the pseudo-loads
    // in the tile, so that a pseudo-load is read once per piece and tile (each column streamed its subcase's before,
    // some 6 MB a column on the mixed-fidelity 1/16 model) and a class's vector once per piece. A column's sums are
    // those of before, in the same order.
    // Measured: on the mixed-fidelity 1/16 model of the gradient benchmarks, 50 load cases.
    // `classValues`: the classes' vectors at the read DOFs, kept from one solve to the next as `buffer` is (each value
    // written before it is read).
    void solveSharedColumns(std::span<const ProductColumn> columns, const SensitivitySystem& system,
                            AdjointColumns& reads, const ItemClasses& classes, const ColumnTerms& terms,
                            linalg::DenseMatrix& buffer, base::BudgetedBuffer<double>& classValues,
                            std::vector<double>& values, ProductReport& report) const
    {
        const std::vector<std::size_t> slot = solveClasses(terms.classes, system, reads, classes, buffer, report);
        addSharedProducts(columns, system, reads, terms, slot, buffer, classValues, values, report);
    }

    [[nodiscard]] std::size_t pieceOf(std::size_t total) const
    {
        const std::size_t threads = m_in.pool != nullptr ? m_in.pool->threadCount() : 1;
        return std::max(PRODUCT_COLUMNS_PER_PIECE, base::pieceCount(total, PRODUCT_PIECES_PER_THREAD * threads));
    }

    // The derivative vectors of the classes `solved` (ItemClasses' numbering) solved on `system`, one column each of
    // `buffer`, placed by the first equation of their vectors (solveSlots): the column of solved[k] is the returned
    // slot[k]. A column's bits do not depend on the others (linalg::FactorSolve).
    [[nodiscard]] std::vector<std::size_t> solveClasses(std::span<const std::int32_t> solved,
                                                        const SensitivitySystem& system, AdjointColumns& reads,
                                                        const ItemClasses& classes, linalg::DenseMatrix& buffer,
                                                        ProductReport& report) const
    {
        const std::size_t count = solved.size();
        const auto equations = static_cast<std::size_t>(system.equationCount());
        auto clock = std::chrono::steady_clock::now();
        reserveRightHandSides(buffer, equations, count);
        const linalg::DenseMatrixView view = buffer.view().block(0, 0, equations, count);
        std::vector<std::int32_t> firstEquations(count); // bounded by a solve
        for (std::size_t k = 0; k < count; ++k)
        {
            const auto [p, i] = classes.first[static_cast<std::size_t>(solved[k])];
            firstEquations[k] = reads.firstEquation(m_linearised.items[p][i].weights);
        }
        std::vector<std::size_t> slot = AdjointColumns::solveSlots(firstEquations);
        base::parallelFor(m_in.pool, count, pieceOf(count),
                          [&](std::size_t begin, std::size_t end)
                          {
                              std::vector<double> scratch(gSize(), 0.0); // one per piece
                              std::vector<base::DofIndex> touched;       // the entries of a class's vector
                              for (std::size_t k = begin; k < end; ++k)
                              {
                                  const auto [p, i] = classes.first[static_cast<std::size_t>(solved[k])];
                                  for (const auto& [dof, value] : m_linearised.items[p][i].weights)
                                  {
                                      scratch[static_cast<std::size_t>(dof)] += value;
                                      touched.push_back(dof);
                                  }
                                  reads.reduceToEquations(
                                      scratch, touched,
                                      std::span<double>(view.data + slot[k] * view.leadingDimension, equations));
                              }
                          });
        report.rightHandSideSeconds += lap(clock);
        if (equations > 0)
            system.solve(view);
        report.solveSeconds += lap(clock);
        report.solves += static_cast<std::int64_t>(count);
        return slot;
    }

    // The products of a block's columns from the solved columns of their classes (`slot[k]`: the column of `buffer`
    // that holds terms.classes[k]): the classes' vectors at the read DOFs, then the combination and the dots by tiles
    // (the PERFORMANCE comment of solveSharedColumns), added to `values`.
    void addSharedProducts(std::span<const ProductColumn> columns, const SensitivitySystem& system,
                           AdjointColumns& reads, const ColumnTerms& terms, std::span<const std::size_t> slot,
                           const linalg::DenseMatrix& buffer, base::BudgetedBuffer<double>& classValues,
                           std::vector<double>& values, ProductReport& report) const
    {
        const std::size_t count = terms.classes.size();
        const auto equations = static_cast<std::size_t>(system.equationCount());
        const std::span<const base::DofIndex> readDofs = reads.readDofs();
        const std::size_t readCount = readDofs.size();
        auto clock = std::chrono::steady_clock::now();
        const double* const solvedData = buffer.data();
        const std::size_t leading = buffer.leadingDimension();
        const auto columnOf = [&](std::size_t k)
        { return std::span<const double>(solvedData + slot[k] * leading, equations); };
        // PERFORMANCE: kept over the call: zeroing a buffer of the classes × the read DOFs made again at each solve
        // (some 200 MB on the mixed-fidelity 1/16 model) was about a second of the products there.
        if (classValues.size() < count * readCount)
        {
            classValues = base::BudgetedBuffer<double>(); // the memory back first
            classValues = base::BudgetedBuffer<double>(m_budget, count * readCount, "vjp item adjoints at the reads",
                                                       base::FOR_OVERWRITE);
        }
        double* const adjointsAtReads = classValues.data();
        base::parallelFor(m_in.pool, count, pieceOf(count),
                          [&](std::size_t begin, std::size_t end)
                          {
                              for (std::size_t k = begin; k < end; ++k)
                                  for (std::size_t r = 0; r < readCount; ++r)
                                      adjointsAtReads[k * readCount + r] = reads.valueAtRead(columnOf(k), r);
                          });
        report.combineSeconds += lap(clock);
        addProducts(
            reads, columns, values,
            [&](std::size_t firstRead, std::size_t endRead, base::DofIndex firstDof,
                std::span<const std::size_t> pieceColumns, double* block, std::size_t stride)
            {
                // A tile's read DOFs are at most its DOFs.
                std::array<double, AdjointColumns::DOT_TILE_DOFS> sum{};
                const std::size_t width = endRead - firstRead;
                for (std::size_t k = 0; k < pieceColumns.size(); ++k)
                {
                    const std::size_t c = pieceColumns[k];
                    std::fill_n(sum.begin(), width, 0.0);
                    for (std::size_t t = terms.start[c]; t < terms.start[c + 1]; ++t)
                    {
                        const double* adjoint =
                            adjointsAtReads + static_cast<std::size_t>(terms.localClass[t]) * readCount + firstRead;
                        const double w = terms.weight[t];
                        for (std::size_t r = 0; r < width; ++r)
                            sum[r] += w * adjoint[r];
                    }
                    for (std::size_t r = 0; r < width; ++r)
                        block[static_cast<std::size_t>(readDofs[firstRead + r] - firstDof) * stride + k] = sum[r];
                }
            });
        report.dotSeconds += lap(clock);
    }

    // The products of a vjp on the linearisation (prepare() made). `adjoints` (one row of weights): the adjoint vectors
    // λ_s of the row by subcase label, which a Hessian-vector product with the same weights takes.
    // `sharesItemAdjoints`: ProductContext::sharesItemAdjoints.
    void vjp(const ProductWeights& weights, std::size_t block, ProductSink& sink, ProductReport& report,
             std::map<std::int32_t, std::vector<double>>* adjoints = nullptr, bool sharesItemAdjoints = false) const
    {
        const WeightRows rows(weights, m_linearised,
                              [this](std::size_t p) -> const ResponseValues& { return valuesAt(p); });
        const std::size_t n = m_variableCount;
        // A Hessian-vector product reads the adjoint vectors of its one row whole; otherwise the full path solves them
        // at the DOFs the products read only.
        const bool keepsAdjoints = adjoints != nullptr && weights.rows == 1;
        const std::map<const SensitivitySystem*, std::unique_ptr<AdjointColumns>> reads = productReads(keepsAdjoints);
        const auto readsOf = [&](const SensitivitySystem& system) -> AdjointColumns*
        {
            const auto found = reads.find(&system);
            return found == reads.end() ? nullptr : found->second.get();
        };
        linalg::DenseMatrix buffer;               // the right-hand sides of the solves
        base::BudgetedBuffer<double> classValues; // the shared route's class vectors at the read DOFs
        const bool shares = sharesItemAdjoints && !keepsAdjoints;
        const ItemClasses classes = shares ? itemClasses() : ItemClasses{};
        // The rows gathered: consecutive blocks whose only solve is one of the
        // shared route on one factor wait, their classes gathered into one solve of at most `block` columns; then
        // each block's products from the solved columns, its values to the sink in block order.
        // PERFORMANCE: a block of 1 024 rows weights some 330 classes on the 1/16 model: gathered, a solve holds some
        // 1 000 columns, eight tiles in the flow of FactorSolve instead of three, a third of the solves' tails and
        // fixed costs. Each block's route is decided as before, a class's column has the bits it has alone, and a
        // waiting block's values receive the same products in the same order: the bits of every product.
        // Measured: on the 1/16 model of the gradient benchmarks, 50 load cases.
        struct WaitingBlock
        {
            std::size_t first = 0;
            std::size_t count = 0;
            std::vector<double> values;
            std::vector<ProductColumn> columns;
            ColumnTerms terms;
        };
        std::vector<WaitingBlock> waiting;                           // their classes within `block` columns
        std::vector<std::int32_t> gatheredClasses;                   // the waiting blocks' classes, first seen first
        std::unordered_map<std::int32_t, std::size_t> gatheredIndex; // a class's place in gatheredClasses
        const SensitivitySystem* gatheredSystem = nullptr;
        const auto finishWaiting = [&]
        {
            if (waiting.empty())
                return;
            AdjointColumns& systemReads = *readsOf(*gatheredSystem);
            const std::vector<std::size_t> gatheredSlot =
                solveClasses(gatheredClasses, *gatheredSystem, systemReads, classes, buffer, report);
            for (WaitingBlock& waited : waiting)
            {
                std::vector<std::size_t> slot(waited.terms.classes.size()); // bounded by a solve
                for (std::size_t k = 0; k < slot.size(); ++k)
                    slot[k] = gatheredSlot[gatheredIndex.at(waited.terms.classes[k])];
                addSharedProducts(waited.columns, *gatheredSystem, systemReads, waited.terms, slot, buffer, classValues,
                                  waited.values, report);
                sink.consume({waited.first, waited.count, waited.count, n, waited.values});
                ++report.blocks;
            }
            waiting.clear();
            gatheredClasses.clear();
            gatheredIndex.clear();
            gatheredSystem = nullptr;
        };
        for (std::size_t first = 0; first < weights.rows; first += block)
        {
            const std::size_t count = std::min(block, weights.rows - first);
            std::vector<double> values(count * n, 0.0); // bounded by a block
            // The explicit parts.
            for (std::size_t r = 0; r < count; ++r)
                for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                    report.weightsRead += rows.forEach(p, first + r,
                                                       [&](std::size_t i, double w)
                                                       {
                                                           const ItemLinearisation& item = m_linearised.items[p][i];
                                                           if (!item.isWanted)
                                                               return;
                                                           for (const auto& [column, value] : item.explicitPart)
                                                               values[r * n + column] += w * value;
                                                       });
            // λ_rᵀ·b per subcase, on the rows that weight items of the subcase: the columns (subcase, row) in subcase
            // then row order, solved together while their subcases are on one factor of the full path, at most
            // `block` of them (the block of rows over several subcases one solve, instead of
            // a solve per subcase of its share of the block). A column is solved alone as among others (the bits of a
            // column do not depend on the others: linalg::FactorSolve), and the products of a row are added in
            // subcase order, as before.
            std::vector<ProductColumn> pending; // bounded by `block`
            const SensitivitySystem* pendingSystem = nullptr;
            // The block's first solve when it is of the shared route: held until the block has another solve (then
            // it is solved first, the products added in the order of before) or ends (then the block may wait).
            std::vector<WaitingBlock> held; // at most one
            const SensitivitySystem* heldSystem = nullptr;
            bool hasSolved = false;
            // With shared item adjoints, the columns of each factor of the full path, the factors in the order of their
            // first subcase, each in subcase then row order.
            std::vector<std::pair<const SensitivitySystem*, std::vector<ProductColumn>>> groups;
            const auto flush = [&]
            {
                if (pending.empty())
                    return;
                if (!held.empty())
                {
                    solveSharedColumns(held.front().columns, *heldSystem, *readsOf(*heldSystem), classes,
                                       held.front().terms, buffer, classValues, values, report);
                    held.clear();
                    hasSolved = true;
                }
                AdjointColumns* systemReads = readsOf(*pendingSystem);
                if (shares && systemReads != nullptr && !systemReads->readsEvery())
                {
                    ColumnTerms terms = columnTerms(rows, first, pending, classes);
                    if (terms.classes.size() < pending.size())
                    {
                        if (!hasSolved)
                        {
                            held.push_back({first, count, {}, pending, std::move(terms)});
                            heldSystem = pendingSystem;
                        }
                        else
                            solveSharedColumns(pending, *pendingSystem, *systemReads, classes, terms, buffer,
                                               classValues, values, report);
                        hasSolved = true;
                        pending.clear();
                        return;
                    }
                }
                solveColumns(rows, first, pending, *pendingSystem, keepsAdjoints ? adjoints : nullptr, systemReads,
                             buffer, values, report);
                hasSolved = true;
                pending.clear();
            };
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                const std::int32_t label = m_subcases[s].subcase.label;
                const SensitivitySystem& system = systemOf(m_subcases[s]);
                const auto readsInSubcase = [&](std::size_t p, std::size_t i)
                {
                    const ItemLinearisation& item = m_linearised.items[p][i];
                    return item.isWanted && item.subcase == label && item.readsDisplacements();
                };
                std::vector<std::size_t> used; // bounded by a block
                for (std::size_t r = 0; r < count; ++r)
                {
                    bool isUsed = false;
                    for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                        report.weightsRead += rows.forEach(
                            p, first + r, [&](std::size_t i, double) { isUsed = isUsed || readsInSubcase(p, i); });
                    if (isUsed)
                        used.push_back(r);
                }
                if (used.empty())
                    continue;
                if (shares && !system.isCondensed())
                {
                    // Shared item adjoints: the factor's columns over the block's subcases together (a subcase's
                    // items shared with the subcases of its factor, whichever subcases lie between).
                    auto group = std::find_if(groups.begin(), groups.end(),
                                              [&](const auto& entry) { return entry.first == &system; });
                    if (group == groups.end())
                        group = groups.insert(groups.end(), {&system, {}});
                    for (const std::size_t r : used)
                        group->second.push_back({s, r});
                    continue;
                }
                // The condensed path solves a subcase alone (its reads are the subcase's units).
                if (&system != pendingSystem || system.isCondensed())
                {
                    flush();
                    pendingSystem = &system;
                }
                for (const std::size_t r : used)
                {
                    if (pending.size() == block)
                        flush();
                    pending.push_back({s, r});
                }
            }
            flush();
            for (const auto& [system, columns] : groups)
                for (std::size_t from = 0; from < columns.size(); from += block)
                {
                    pendingSystem = system;
                    pending.assign(
                        columns.begin() + static_cast<std::ptrdiff_t>(from),
                        columns.begin() + static_cast<std::ptrdiff_t>(std::min(columns.size(), from + block)));
                    flush();
                }
            if (held.empty())
            {
                // Its solves made: the waiting blocks (earlier) go first.
                finishWaiting();
                sink.consume({first, count, count, n, values});
                ++report.blocks;
                continue;
            }
            // The block's only solve is of the shared route: it waits with the others when their classes and its own
            // fit one solve on one factor.
            std::size_t added = 0;
            for (const std::int32_t k : held.front().terms.classes)
                added += gatheredIndex.count(k) == 0 ? 1 : 0;
            if (!waiting.empty() && (heldSystem != gatheredSystem || gatheredClasses.size() + added > block))
                finishWaiting();
            for (const std::int32_t k : held.front().terms.classes)
                if (gatheredIndex.try_emplace(k, gatheredClasses.size()).second)
                    gatheredClasses.push_back(k);
            gatheredSystem = heldSystem;
            held.front().values = std::move(values);
            waiting.push_back(std::move(held.front()));
        }
        finishWaiting();
    }

    // The products of a jvp on the linearisation (prepare() made, holding the responses of `rows`): the rows of the
    // items of `rows` (a restricted call's selected items, the others NaN, and only the subcases they are
    // in solved).
    void jvp(std::span<const double> directions, std::size_t count, std::size_t block, ProductSink& sink,
             ProductReport& report, const JvpRows& rows) const
    {
        const std::size_t n = m_variableCount;
        if (directions.size() != n * count)
            throw std::invalid_argument(
                base::concatenate({"jvp: expected directions of shape (", std::to_string(n), ", ",
                                   std::to_string(count), "), found ", std::to_string(directions.size()), " values"}));
        // The items computed (linearised and selected) in the order of the rows, and those of each subcase.
        struct ComputedItem
        {
            std::size_t row;
            ItemKey key;
        };
        std::vector<ComputedItem> computed; // bounded by the items
        std::vector<std::uint8_t> isRowComputed;
        std::size_t itemCount = 0;
        for (std::size_t k = 0; k < rows.positions.size(); ++k)
        {
            const std::size_t p = rows.positions[k];
            const std::vector<ItemLinearisation>& items = m_linearised.items[p];
            for (std::size_t i = 0; i < items.size(); ++i, ++itemCount)
            {
                const bool isComputed = items[i].isWanted && (rows.isSelected[k].empty() || rows.isSelected[k][i] != 0);
                isRowComputed.push_back(static_cast<std::uint8_t>(isComputed));
                if (isComputed)
                    computed.push_back({itemCount, {p, i}});
            }
        }
        std::vector<std::vector<std::size_t>> computedOf(m_subcases.size()); // per subcase, into `computed`
        {
            std::map<std::int32_t, std::size_t> subcaseOf;
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
                subcaseOf.emplace(m_subcases[s].subcase.label, s);
            for (std::size_t c = 0; c < computed.size(); ++c)
            {
                const auto found = subcaseOf.find(m_linearised.at(computed[c].key).subcase);
                if (found != subcaseOf.end())
                    computedOf[found->second].push_back(c);
            }
        }
        for (std::size_t first = 0; first < count; first += block)
        {
            const std::size_t width = std::min(block, count - first);
            std::vector<double> values(itemCount * width, std::numeric_limits<double>::quiet_NaN()); // a block
            // The explicit parts (NaN for the items with no gradient or not selected).
            for (const ComputedItem& item : computed)
            {
                const ItemLinearisation& linear = m_linearised.at(item.key);
                for (std::size_t d = 0; d < width; ++d)
                {
                    double sum = 0.0;
                    for (const auto& [column, value] : linear.explicitPart)
                        sum += value * directions[column * count + first + d];
                    values[item.row * width + d] = sum;
                }
            }
            // The variables each direction moves, in increasing order (the order of the sum of its pseudo-loads).
            std::vector<std::vector<std::size_t>> moved(width); // bounded by a block
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t d = 0; d < width; ++d)
                    if (directions[j * count + first + d] != 0.0)
                        moved[d].push_back(j);
            // δu = K⁻¹·Σ_j v_j·b_j per subcase, then c_i·δu (not in a SUBCOM's term that no item reads; in a restricted
            // call, only in the subcases of its computed items).
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                if (m_subcases[s].isTermOnly || (rows.isRestricted && computedOf[s].empty()))
                    continue;
                const SensitivitySystem& system = systemOf(m_subcases[s]);
                const auto equations = static_cast<std::size_t>(system.equationCount());
                linalg::DenseMatrix rightHandSides(m_budget, equations, width, "jvp direct loads");
                std::vector<double> load(gSize());
                std::vector<double> sum(gSize());
                std::vector<double> column(equations);
                for (std::size_t d = 0; d < width; ++d)
                {
                    std::fill(sum.begin(), sum.end(), 0.0);
                    for (const std::size_t j : moved[d])
                    {
                        if (!m_pseudoLoads->hasLoad(j, s))
                            continue;
                        const double v = directions[j * count + first + d];
                        m_pseudoLoads->assemble(j, s, load);
                        for (std::size_t g = 0; g < gSize(); ++g)
                            sum[g] += v * load[g];
                    }
                    system.toEquations(sum, column);
                    for (std::size_t e = 0; e < equations; ++e)
                        rightHandSides(e, d) = column[e];
                }
                // The condensed path recovers the units the items of the subcase read only.
                std::vector<ItemKey> keys; // bounded by the items
                keys.reserve(computedOf[s].size());
                for (const std::size_t c : computedOf[s])
                    keys.push_back(computed[c].key);
                const base::BudgetedBuffer<std::uint8_t> reads =
                    SolutionReads::ofWeights(system, m_linearised, keys, m_budget);
                if (equations > 0)
                    system.solve(rightHandSides.view(), reads.span());
                report.solves += static_cast<std::int64_t>(width);
                for (std::size_t d = 0; d < width; ++d)
                {
                    for (std::size_t e = 0; e < equations; ++e)
                        column[e] = rightHandSides(e, d);
                    system.fromEquations(column, load);
                    for (const std::size_t c : computedOf[s])
                        values[computed[c].row * width + d] +=
                            dotSparse(m_linearised.at(computed[c].key).weights, load);
                }
            }
            sink.consume({first, width, itemCount, width, values});
            ++report.blocks;
        }
    }
};

const CallableRegistry& noCallables()
{
    static const CallableRegistry NONE;
    return NONE;
}

StaticSensitivityInputs inputsOf(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                 const ResponseSubcases& subcases, const ProductContext& context)
{
    return problem.sensitivityInputs(evaluation, subcases,
                                     context.callables != nullptr ? *context.callables : noCallables(),
                                     context.analysisOptions, context.blockItems, context.externals);
}

void requireBlock(std::size_t block, const char* function)
{
    if (block == 0)
        throw std::invalid_argument(std::string(function).append(
            ": expected a block of at least one row or direction, found 0; give block= a positive number"));
}

// The responses a product reads: those of the weights (a vjp), or every one (a jvp).
struct ReadResponses
{
    bool isEvery = false;
    std::set<std::int32_t> ids;

    // Whether these include every response of `other`.
    [[nodiscard]] bool includes(const ReadResponses& other) const
    {
        return isEvery || (!other.isEvery && std::includes(ids.begin(), ids.end(), other.ids.begin(), other.ids.end()));
    }
};

} // namespace

// The linearisation and what it reads (its inputs point into the evaluation, the subcases and the options here). The
// members are destroyed in reverse order: the computation first, the reservation last.
struct ProductLinearisation::State
{
    base::MemoryReservation reservation;
    const DesignProblem* problem = nullptr;
    const DesignEvaluation* evaluation = nullptr;
    std::size_t blockItems = 0;
    ReadResponses responses;
    ResponseSubcases subcases;
    StaticSensitivityInputs inputs;
    GradientOptions options;
    std::unique_ptr<ProductComputation> computation;
    std::int32_t factorisationsReported = 0;
    // The adjoint vectors of Σᵢ yᵢ·rᵢ of the last Hessian-vector product, per subcase label, and the
    // weights y they are of (each weighted response's non-zero weights, items increasing), with their memory.
    std::vector<std::pair<std::int32_t, std::vector<std::pair<std::size_t, double>>>> adjointWeights;
    std::map<std::int32_t, std::vector<double>> adjoints;
    base::MemoryReservation adjointReservation;
    // The element Hessian blocks of these weights and adjoints (null: not made, or not closed: then
    // `blocksReason` says why and the products difference the linearisation).
    bool hasBlocksOfAdjoint = false;
    std::unique_ptr<detail::HessianBlocks> hessianBlocks;
    std::string blocksReason;

    // Whether this linearisation is that of the evaluation `at` of `on`, for this batch size.
    [[nodiscard]] bool isOf(const DesignProblem& on, const DesignEvaluation& at, const ProductContext& context) const
    {
        return problem == &on && evaluation == &at && blockItems == context.blockItems;
    }
};

ProductLinearisation::ProductLinearisation() = default;
ProductLinearisation::ProductLinearisation(ProductLinearisation&&) noexcept = default;
ProductLinearisation& ProductLinearisation::operator=(ProductLinearisation&&) noexcept = default;
ProductLinearisation::~ProductLinearisation() = default;

void ProductLinearisation::clear() noexcept
{
    m_state.reset();
}

std::uint64_t ProductLinearisation::bytes() const noexcept
{
    if (!m_state)
        return 0;
    return m_state->reservation.bytes() + m_state->adjointReservation.bytes() +
           (m_state->hessianBlocks ? m_state->hessianBlocks->bytes() : 0);
}

namespace
{

// The linearisation of the responses `read` at the evaluation (with `restricted`, of what those options select).
std::unique_ptr<ProductLinearisation::State> linearise(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                       const ReadResponses& read, base::MemoryBudget& budget,
                                                       const ProductContext& context,
                                                       const GradientOptions* restricted = nullptr)
{
    auto state = std::make_unique<ProductLinearisation::State>();
    state->problem = &problem;
    state->evaluation = &evaluation;
    state->blockItems = context.blockItems;
    state->responses = read;
    state->subcases = problem.responseSubcases(evaluation);
    state->inputs = inputsOf(problem, evaluation, state->subcases, context);
    if (restricted != nullptr)
        state->options = *restricted;
    else if (!read.isEvery)
        state->options.responseIds.assign(read.ids.begin(), read.ids.end());
    state->computation = std::make_unique<ProductComputation>(state->inputs, state->options, budget);
    state->computation->prepare();
    return state;
}

} // namespace

// The products of a call on a linearisation: the kept one when it is the evaluation's and covers the responses `read`,
// else a new one (of the responses the kept one read and of the call's), kept in `kept` when given and when the
// budget holds it, else dropped after the call. With `restricted` (a restricted jvp), the new one is of
// what those options select, for this call alone: the kept one stays.
template <typename Products>
ProductReport VectorJacobianProducts::products(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                               bool readsEvery, const std::set<std::int32_t>& read,
                                               base::MemoryBudget& budget, const ProductContext& context,
                                               ProductLinearisation* kept, const Products& compute,
                                               const GradientOptions* restricted)
{
    ProductReport report;
    const auto started = std::chrono::steady_clock::now();
    const ReadResponses asked{readsEvery, read};
    ProductLinearisation::State* state = kept != nullptr ? kept->m_state.get() : nullptr;
    const bool isSameEvaluation = state != nullptr && state->isOf(problem, evaluation, context);
    std::unique_ptr<ProductLinearisation::State> made;
    if (isSameEvaluation && state->responses.includes(asked))
    {
        report.isLinearisationReused = true;
        // This call's callables, evaluators and pool (the products read none of them: the inputs stay valid).
        state->inputs.callables = context.callables != nullptr ? context.callables : &noCallables();
        state->inputs.externals = context.externals;
        state->inputs.pool = context.analysisOptions.pool;
    }
    else if (restricted != nullptr)
    {
        // PERFORMANCE: a restricted jvp linearises its selected items and the fields of its directions' variables
        // only, and keeps nothing (a partial linearisation would serve no later call along other directions): the
        // fields' pseudo-loads of every variable are most of a linearisation, and a bay's directions move a few.
        // Measured: on a short demo of a stiffened panel model, its 16 jvp of the members' bays.
        made = linearise(problem, evaluation, asked, budget, context, restricted);
        state = made.get();
        report.lineariseItemSeconds = made->computation->report().itemSeconds;
        report.lineariseFieldSeconds = made->computation->report().fieldSeconds;
    }
    else
    {
        ReadResponses responses = asked;
        // PERFORMANCE: a call that reads responses the kept linearisation of the point does not hold linearises every
        // response of the problem (as a jvp does), not the union of the two sets: a caller that asks its rows group by
        // group, each group on other responses (fused rows, a zone's reserve factors on its zone's
        // loads), linearised the union again at every group, its fields' pseudo-loads included (the same for every
        // set), so that the groups cost groups × a linearisation and their items' work grew as groups². At most two
        // linearisations per point then; every product the same bits (each item is linearised on its own, the
        // pseudo-loads are the same). The union stays when every response cannot be
        // linearised (a response products refuse, an equation's callable failing, which a vjp that does not weight it
        // never reads).
        // Measured: the wing of the metrics campaign at 1/128, 56 fused rows in 56 groups of zones: more than 58 s
        // (cancelled) against about 1.4 s.
        const bool takesEvery = isSameEvaluation && !responses.isEvery;
        if (isSameEvaluation && !responses.isEvery)
        {
            responses.isEvery = state->responses.isEvery;
            responses.ids.insert(state->responses.ids.begin(), state->responses.ids.end());
        }
        if (kept != nullptr)
            kept->clear(); // the earlier linearisation's memory goes back first
        if (takesEvery)
        {
            try
            {
                made = linearise(problem, evaluation, ReadResponses{true, {}}, budget, context);
            }
            catch (const std::exception&)
            {
                // A response the products do not linearise (refused, its callable failing): the union of the two sets
                // below, as before.
                made.reset();
            }
        }
        if (!made)
            made = linearise(problem, evaluation, responses, budget, context);
        state = made.get();
        report.lineariseItemSeconds = made->computation->report().itemSeconds;
        report.lineariseFieldSeconds = made->computation->report().fieldSeconds;
        if (kept != nullptr)
        {
            try
            {
                made->reservation =
                    budget.reserve(made->computation->heldBytes(), "kept linearisation of the products");
                kept->m_state = std::move(made);
            }
            catch (const base::MemoryBudgetError&)
            {
                // Not kept: the products are made all the same, and the next call linearises again.
            }
        }
    }
    const auto linearised = std::chrono::steady_clock::now();
    compute(*state, report);
    const std::int32_t factorisations = state->computation->factorisationCount();
    report.factorisations = factorisations - state->factorisationsReported;
    state->factorisationsReported = factorisations;
    report.linearisationBytes = kept != nullptr ? kept->bytes() : 0;
    report.lineariseSeconds = std::chrono::duration<double>(linearised - started).count();
    report.productSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - linearised).count();
    return report;
}

namespace
{

std::vector<std::pair<std::int32_t, std::vector<std::pair<std::size_t, double>>>> weightsKeyOf(
    const ProductWeights& weights);

// The adjoint vectors of one row of weights kept with the linearisation, replacing those of other
// weights; not kept when the budget does not hold them.
void keepAdjoints(ProductLinearisation::State& state, const ProductWeights& weights,
                  std::map<std::int32_t, std::vector<double>> adjoints, base::MemoryBudget& budget)
{
    auto key = weightsKeyOf(weights);
    if (state.adjointWeights == key && !state.adjoints.empty())
        return;
    state.adjoints.clear();
    state.adjointReservation.release();
    state.hessianBlocks.reset();
    state.hasBlocksOfAdjoint = false;
    state.blocksReason.clear();
    state.adjointWeights.clear();
    const std::uint64_t bytes = static_cast<std::uint64_t>(adjoints.size()) *
                                (adjoints.empty() ? 0 : adjoints.begin()->second.size()) * sizeof(double);
    try
    {
        state.adjointReservation = budget.reserve(bytes, "Hessian-vector adjoint vectors (kept from a vjp)");
    }
    catch (const base::MemoryBudgetError&)
    {
        return;
    }
    state.adjoints = std::move(adjoints);
    state.adjointWeights = std::move(key);
}

} // namespace

ProductReport VectorJacobianProducts::vjp(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                          const ProductWeights& weights, std::size_t block, ProductSink& sink,
                                          base::MemoryBudget& budget, const ProductContext& context,
                                          ProductLinearisation* kept)
{
    requireBlock(block, "vjp");
    std::set<std::int32_t> weighted; // bounded by the responses
    for (const auto& [id, values] : weights.weights)
        weighted.insert(id);
    for (const auto& [id, values] : weights.sparseWeights)
        weighted.insert(id);
    for (const std::int32_t id : weighted)
        if (std::none_of(evaluation.responses.begin(), evaluation.responses.end(),
                         [id](const ResponseValues& response) { return response.id == id; }))
            throw std::invalid_argument(
                base::concatenate({"vjp: expected weights on evaluated responses, found "
                                   "response ",
                                   std::to_string(id), ", which the evaluation does not hold"}));
    return products(problem, evaluation, false, weighted, budget, context, kept,
                    [&](ProductLinearisation::State& made, ProductReport& report)
                    {
                        if (weights.rows != 1)
                        {
                            made.computation->vjp(weights, block, sink, report, nullptr, context.sharesItemAdjoints);
                            return;
                        }
                        // One row: its adjoint vectors kept for a Hessian-vector product at the same (x, y) (λ kept
                        // from the gradient), when the budget holds them.
                        std::map<std::int32_t, std::vector<double>> adjoints;
                        made.computation->vjp(weights, block, sink, report, &adjoints);
                        keepAdjoints(made, weights, std::move(adjoints), budget);
                    });
}

namespace
{

// The step of the differences along v: HESSIAN_STEP_RELATIVE of the largest |x_j|, over the largest |v_j|,
// one central difference. The terms differenced are the linearisation's at the moved states, smooth in x and u, whose
// own rounding is some 1e-12 of their size: measured on the small caisson (von Mises items, MID3) against the central
// differences of vjp (themselves good to 7e-8), the products agree to 6e-6 at a step of 1e-3 (truncation), 9e-8 at
// 1e-4 and 3e-8 at 1e-5, and to 3e-8 at 1e-3 extrapolated from t and t/2 (twice the linearisations): 1e-4, central.
constexpr double HESSIAN_STEP_RELATIVE = 1.0e-4;
// Under inertia relief the linearisation at a moved state carries a round-off of some 1e-9 of its
// terms (the relieved loads' projection on the support motions, the free barrels in millimetres: the vjp of the free
// mixed tiny barrel moves by 4e-8 of itself between points 1e-7 apart, against 1e-12 clamped), which the step 1e-4
// makes 6e-4 of a small entry (a CompProp ply's); at 1e-3 the products agree with the extrapolated differences of vjp
// to their own noise (1e-5 of that entry) and their truncation stays under 3e-7 of the largest entry (the 1D free
// barrel, every variable of the loop). That step's truncation is not small along a section dimension: along the
// stringer's BarShapeProp dimensions of the free mixed tiny barrel the product is 4.8e-5 of its largest entry off the
// converged differences of vjp at 1e-3, 1.2e-5 at 5e-4, 1.9e-6 at 2e-4 (the law t²; the same curvature clamped, 2.8e-5
// at 1e-3, which 1e-4 makes 2.5e-7), and a smaller step brings the relief's round-off back (a ply's entry moves by 1e-4
// of itself at 2e-4). So under relief the central differences at t and t/2 are extrapolated (Richardson, (4·D(t/2) −
// D(t)) / 3, truncation O(t⁴)): the stringer's product then 6e-8 of the converged differences at every t from 2e-3 to
// 1e-5, the ply's as at 1e-3 (1e-5 of its entry); four linearisations at the moved states instead of two (the
// product 1.3 to 2.2 times its time before on the tiny barrels), the same solves (the rates combined before δλ's
// solve).
constexpr double RELIEVED_HESSIAN_STEP_RELATIVE = 1.0e-3;
constexpr std::array<double, 2> HESSIAN_SIDES{1.0, -1.0};

// The rates along σ of a product's differenced terms at one step: the explicit part Σ yᵢ·∂rᵢ/∂x_j, the adjoint part
// Σ_s λ_sᵀ·b_js, and the right-hand sides Σ yᵢ·cᵢ − K·λ_s per subcase.
struct StepRates
{
    std::vector<double> explicitRates;
    std::vector<double> adjointRates;
    std::map<std::int32_t, std::vector<double>> loadRates;
};

// Richardson's extrapolation of central differences at t and t/2, (4·D(t/2) − D(t)) / 3, in place in `half`.
void extrapolateInto(std::vector<double>& half, const std::vector<double>& full)
{
    for (std::size_t k = 0; k < half.size(); ++k)
        half[k] = (4.0 * half[k] - full[k]) / 3.0;
}

// The non-zero weights of the one row of each weighted response, items increasing: the key of the kept adjoint.
std::vector<std::pair<std::int32_t, std::vector<std::pair<std::size_t, double>>>> weightsKeyOf(
    const ProductWeights& weights)
{
    std::map<std::int32_t, std::vector<std::pair<std::size_t, double>>> rows;
    for (const auto& [id, values] : weights.weights)
        for (std::size_t i = 0; i < values.size(); ++i)
            if (values[i] != 0.0)
                rows[id].emplace_back(i, values[i]);
    for (const auto& [id, sparse] : weights.sparseWeights)
        for (std::size_t k = sparse.rowStart[0]; k < sparse.rowStart[1]; ++k)
            if (sparse.values[k] != 0.0)
                rows[id].emplace_back(sparse.items[k], sparse.values[k]);
    return {rows.begin(), rows.end()};
}

// The central difference of the values at σ = t and −t.
double rateOf(const std::array<double, 2>& values, double t)
{
    return (values[0] - values[1]) / (2.0 * t);
}

} // namespace

namespace
{

void ensureAdjoints(ProductLinearisation::State& state, const ProductWeights& weights, base::MemoryBudget& budget,
                    ProductReport& report);
void ensureHessianBlocks(ProductLinearisation::State& state, const ProductWeights& effective,
                         base::MemoryBudget& budget, ProductReport& report);

} // namespace

ProductReport VectorJacobianProducts::hessianVector(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                    const ProductWeights& weights, std::span<const double> direction,
                                                    std::vector<double>& values, base::MemoryBudget& budget,
                                                    const ProductContext& context, ProductLinearisation* kept,
                                                    std::size_t count)
{
    if (weights.rows != 1)
        throw std::invalid_argument(
            base::concatenate({"hessian_vector: expected one row of weights (the multipliers y of Σ yᵢ·rᵢ), found ",
                               std::to_string(weights.rows), " rows; give one row per response"}));
    std::set<std::int32_t> weighted; // bounded by the responses
    for (const auto& [id, rows] : weights.weights)
        weighted.insert(id);
    for (const auto& [id, rows] : weights.sparseWeights)
        weighted.insert(id);
    const std::vector<double> x0 = evaluation.analysis.point.x();
    if (count == 0)
        throw std::invalid_argument("hessian_vector: expected one direction or more, found none");
    if (count == 1 && direction.size() != x0.size())
        throw std::invalid_argument(
            base::concatenate({"hessian_vector: expected a direction of ", std::to_string(x0.size()),
                               " values (one per independent variable), found ", std::to_string(direction.size())}));
    if (direction.size() != x0.size() * count)
        throw std::invalid_argument(
            base::concatenate({"hessian_vector: expected directions of shape (", std::to_string(x0.size()), ", ",
                               std::to_string(count), "): ", std::to_string(x0.size() * count),
                               " values, one column per direction, found ", std::to_string(direction.size())}));
    return products(
        problem, evaluation, false, weighted, budget, context, kept,
        [&](ProductLinearisation::State& state, ProductReport& report)
        {
            ProductComputation& computation = *state.computation;
            // The equation responses of the second order: their weights y moved onto their arguments
            // (μ), their callables' terms added from δu; the product's own terms on the effective weights.
            const ProductComputation::EquationWeights equations = computation.equationWeights(weights);
            const bool hasEquations = !equations.equations.empty();
            const ProductWeights& effective = hasEquations ? equations.effective : weights;
            computation.requireSecondOrder(effective, static_cast<bool>(context.suppliedAt));
            const std::size_t n = x0.size();
            values.assign(n * count, 0.0);
            ensureAdjoints(state, weights, budget, report);
            // H·v along one direction v into `out` (each column of several directions as alone).
            bool hasAskedBlocks = false; // the blocks made or found for the first direction that moves
            const auto along = [&](std::span<const double> v, std::vector<double>& out)
            {
                out.assign(n, 0.0);
                const double vScale = std::accumulate(v.begin(), v.end(), 0.0, [](double largest, double value)
                                                      { return std::max(largest, std::abs(value)); });
                if (vScale == 0.0)
                    return;
                // The element Hessian blocks of (x, y), made by the first product and kept for the next.
                if (!context.usesHessianBlocks)
                    report.hessianBlocksReason = "the products difference the linearisation (usesHessianBlocks false)";
                else if (!hasAskedBlocks)
                    ensureHessianBlocks(state, effective, budget, report);
                hasAskedBlocks = true;
                if (context.usesHessianBlocks)
                    report.hessianBlocksReason = state.blocksReason;
                if (context.usesHessianBlocks && state.hessianBlocks)
                {
                    report.hasHessianBlocks = true;
                    report.hessianBlockBytes = state.hessianBlocks->bytes();
                    report.hessianBlockCount = static_cast<std::int64_t>(state.hessianBlocks->size());
                    const ProductComputation::EquationHook hook =
                        [&](const std::map<std::int32_t, std::vector<double>>& shifts)
                    { return computation.equationTerms(weights, equations, v, shifts); };
                    computation.blockProduct(*state.hessianBlocks, v, out, report, hasEquations ? &hook : nullptr);
                    return;
                }
                // δu_s = K⁻¹·Σ_j v_j·b_js in the subcases a weight reads (the others enter no term), read on the
                // condensed path where the pseudo-loads and the weighted items are.
                std::set<std::int32_t> weightedLabels = computation.weightedSubcases(effective);
                weightedLabels.insert(equations.subcases.begin(), equations.subcases.end());
                const std::map<std::int32_t, std::vector<double>> shifts =
                    computation.solve(computation.directionLoads(v, &weightedLabels), report.solves,
                                      ProductComputation::Reads::Direction);
                const std::vector<ElementRef> moved = computation.elementsMovedBy(v);
                const std::optional<detail::ReliefPoint> reliefPoint = computation.reliefPointOf(moved);
                // The callables' terms at (v, δL), δL from δu (one call per batch of the weighted items).
                std::optional<ProductComputation::EquationTerms> equationTerms;
                if (hasEquations)
                    equationTerms = computation.equationTerms(weights, equations, v, shifts);
                const double xScale = std::accumulate(
                    x0.begin(), x0.end(), 0.0, [](double largest, double x) { return std::max(largest, std::abs(x)); });
                const double relative = reliefPoint ? RELIEVED_HESSIAN_STEP_RELATIVE : HESSIAN_STEP_RELATIVE;
                const double t = (xScale > 0.0 ? relative * xScale : FIELD_STEP_AT_ZERO) / vScale;
                // The steps: t, and under inertia relief t/2 too, extrapolated (RELIEVED_HESSIAN_STEP_RELATIVE).
                std::vector<double> steps{t}; // one or two
                if (reliefPoint)
                    steps.push_back(0.5 * t);
                GradientOptions options;
                // The responses the product's own terms read (an equation's arguments in its place: its callable gives
                // its curvature, the moved states never call it).
                std::set<std::int32_t> readIds;
                for (const auto& [id, row] : effective.weights)
                    readIds.insert(id);
                for (const auto& [id, row] : effective.sparseWeights)
                    readIds.insert(id);
                options.responseIds.assign(readIds.begin(), readIds.end());
                // The finite element sections of the moved states on the meshes of the point (O_o_Reader's
                // reference=): their properties smooth in the dimensions there.
                std::vector<double> reference; // bounded by the relations
                for (const FieldValue& field : computation.inputs().design->fields)
                    reference.push_back(field.value);
                std::vector<StepRates> rateOfSteps; // one per step
                for (const double step : steps)
                {
                    // The terms at the two states (x ± step·v, u ± step·δu).
                    std::array<std::vector<double>, 2> explicitParts;
                    std::array<std::vector<double>, 2> adjointParts;
                    std::array<std::map<std::int32_t, std::vector<double>>, 2> loads; // Σ yᵢ·cᵢ − K(x')·λ per subcase
                    for (std::size_t side = 0; side < HESSIAN_SIDES.size(); ++side)
                    {
                        const double sigma = HESSIAN_SIDES[side] * step;
                        std::vector<double> x = x0;
                        for (std::size_t j = 0; j < n; ++j)
                            x[j] += sigma * v[j];
                        // The relations given by functions at the moved point, asked of their callables.
                        const std::vector<SuppliedField> supplied =
                            context.suppliedAt ? context.suppliedAt(x) : std::vector<SuppliedField>{};
                        const DesignPoint point = problem.designPoint(x, supplied, reference);
                        StaticSensitivityInputs inputs = computation.inputs();
                        inputs.overlaid = problem.overlaySnapshot(point);
                        inputs.design = &point.values();
                        inputs.sectionReference = reference;
                        std::map<std::int32_t, std::vector<double>> displacementShifts;
                        for (const auto& [label, du] : shifts)
                        {
                            std::vector<double> shift(du.size()); // a g-set vector
                            for (std::size_t g = 0; g < du.size(); ++g)
                                shift[g] = sigma * du[g];
                            displacementShifts.emplace(label, std::move(shift));
                        }
                        const PerturbedComputation perturbed(inputs, options, budget, std::move(displacementShifts),
                                                             reliefPoint ? &*reliefPoint : nullptr);
                        ++report.perturbedLinearisations;
                        explicitParts[side] = perturbed.weightedExplicit(effective);
                        adjointParts[side] = perturbed.adjointProducts(state.adjoints);
                        for (const auto& [label, adjoint] : state.adjoints)
                        {
                            // The weights Kgg·w of the SPC force items are those of the analysis' Kgg in the perturbed
                            // linearisation: their move enters as −(∂K/∂v)·(−Σ y·w), the stiffness terms taking λ − Σ
                            // y·w.
                            std::vector<double> load = perturbed.weightedLoad(effective, label);
                            std::vector<double> stiffnessAdjoint = adjoint;
                            const std::vector<double> spcRows = computation.weightedSpcRows(effective, label);
                            for (std::size_t g = 0; g < spcRows.size(); ++g)
                                stiffnessAdjoint[g] -= spcRows[g];
                            const std::vector<double> stiffness = perturbed.stiffnessTimes(moved, stiffnessAdjoint);
                            for (std::size_t g = 0; g < load.size(); ++g)
                                load[g] -= stiffness[g];
                            loads[side].emplace(label, std::move(load));
                        }
                    }
                    StepRates made;
                    made.explicitRates.resize(n);
                    made.adjointRates.resize(n);
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        made.explicitRates[j] = rateOf({explicitParts[0][j], explicitParts[1][j]}, step);
                        made.adjointRates[j] = rateOf({adjointParts[0][j], adjointParts[1][j]}, step);
                    }
                    for (const auto& [label, adjoint] : state.adjoints)
                    {
                        std::vector<double> rate(adjoint.size()); // a g-set vector
                        for (std::size_t g = 0; g < rate.size(); ++g)
                            rate[g] = rateOf({loads[0].at(label)[g], loads[1].at(label)[g]}, step);
                        made.loadRates.emplace(label, std::move(rate));
                    }
                    rateOfSteps.push_back(std::move(made));
                }
                StepRates stepRates = std::move(rateOfSteps.back());
                if (rateOfSteps.size() == 2)
                {
                    const StepRates& full = rateOfSteps.front();
                    extrapolateInto(stepRates.explicitRates, full.explicitRates);
                    extrapolateInto(stepRates.adjointRates, full.adjointRates);
                    for (auto& [label, rate] : stepRates.loadRates)
                        extrapolateInto(rate, full.loadRates.at(label));
                }
                // δλ_s = K⁻¹·d/dσ (Σ yᵢ·cᵢ − K·λ_s): the derivative of the weights along (v, δu) less (∂K/∂v)·λ_s.
                std::map<std::int32_t, std::vector<double>> rates = std::move(stepRates.loadRates);
                // The equations' Σ w_L·∂L/∂u on the right-hand sides of δλ (as ∂²g/∂u² of a response not linear in u).
                if (equationTerms)
                    for (const auto& [label, rate] : equationTerms->rates)
                    {
                        std::vector<double>& sum = rates[label];
                        if (sum.empty())
                            sum.assign(rate.size(), 0.0);
                        for (std::size_t g = 0; g < rate.size(); ++g)
                            sum[g] += rate[g];
                    }
                const std::map<std::int32_t, std::vector<double>> adjointRates =
                    computation.solve(rates, report.solves);
                // (H·v)_j = d/dσ (Σ yᵢ·∂rᵢ/∂x_j + Σ_s λ_sᵀ·b_js) + Σ_s δλ_sᵀ·b_js.
                const std::vector<double> rateProducts = computation.adjointProducts(adjointRates);
                for (std::size_t j = 0; j < n; ++j)
                    out[j] = stepRates.explicitRates[j] + stepRates.adjointRates[j] + rateProducts[j];
                // The equations' w_x·∂var/∂x + Σ w_L·∂L/∂x.
                if (equationTerms)
                    for (std::size_t j = 0; j < n; ++j)
                        out[j] += equationTerms->values[j];
            };
            std::vector<double> column(n); // one direction
            std::vector<double> product;   // its H·v
            for (std::size_t c = 0; c < count; ++c)
            {
                for (std::size_t j = 0; j < n; ++j)
                    column[j] = direction[j * count + c];
                along(column, product);
                for (std::size_t j = 0; j < n; ++j)
                    values[j * count + c] = product[j];
            }
        });
}

namespace
{

// λ_s = K⁻¹·Σᵢ yᵢ·cᵢ per subcase of the weights (one row), kept with the linearisation for the next products and blocks
// at the same weights; new weights drop the adjoints and blocks of the former ones.
void ensureAdjoints(ProductLinearisation::State& state, const ProductWeights& weights, base::MemoryBudget& budget,
                    ProductReport& report)
{
    ProductComputation& computation = *state.computation;
    const auto key = weightsKeyOf(weights);
    if (state.adjointWeights == key && !state.adjoints.empty())
    {
        report.isAdjointReused = true;
        return;
    }
    state.adjoints.clear();
    state.adjointReservation.release();
    state.hessianBlocks.reset();
    state.hasBlocksOfAdjoint = false;
    state.blocksReason.clear();
    std::map<std::int32_t, std::vector<double>> loads;
    for (const std::int32_t label : computation.subcaseLabels())
    {
        std::vector<double> load = computation.weightedLoad(weights, label);
        if (std::any_of(load.begin(), load.end(), [](double value) { return value != 0.0; }))
            loads.emplace(label, std::move(load));
    }
    std::int64_t solves = 0;
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(loads.size()) * (loads.empty() ? 0 : loads.begin()->second.size()) * sizeof(double);
    state.adjointReservation = budget.reserve(bytes, "Hessian-vector adjoint vectors");
    state.adjoints = computation.solve(loads, solves);
    state.adjointWeights = key;
    report.adjointSolves = solves;
}

// The element Hessian blocks of (x, y) of the kept adjoints: made by the first call at these weights
// (a product or the blocks) and kept for the next; null with `blocksReason` where the closed forms stop.
void ensureHessianBlocks(ProductLinearisation::State& state, const ProductWeights& effective,
                         base::MemoryBudget& budget, ProductReport& report)
{
    if (state.hasBlocksOfAdjoint)
    {
        report.areHessianBlocksReused = state.hessianBlocks != nullptr;
        return;
    }
    const auto started = std::chrono::steady_clock::now();
    ProductComputation::ClosedBlocks made = state.computation->hessianBlocks(effective, state.adjoints, budget);
    state.hessianBlocks = std::move(made.blocks);
    state.blocksReason = std::move(made.reason);
    state.hasBlocksOfAdjoint = true;
    report.hessianBlockSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

} // namespace

DesignBlocks VectorJacobianProducts::blocks(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                            base::MemoryBudget& budget, const ProductContext& context,
                                            ProductLinearisation* kept, const ProductWeights* weights)
{
    if (weights != nullptr && weights->rows != 1)
        throw std::invalid_argument(
            base::concatenate({"blocks: expected one row of weights (the multipliers y of Σ yᵢ·rᵢ), found ",
                               std::to_string(weights->rows), " rows; give one row per response"}));
    DesignBlocks result;
    const ProductReport report =
        products(problem, evaluation, true, {}, budget, context, kept,
                 [&](ProductLinearisation::State& made, ProductReport& report)
                 {
                     ProductComputation& computation = *made.computation;
                     if (weights == nullptr)
                     {
                         computation.blocks(result);
                         return;
                     }
                     // The second derivatives of the Lagrangian of (x, y): the adjoint and the element Hessian
                     // blocks of hessianVector, made or kept.
                     computation.requireBlocks();
                     const auto refuse = [](const std::string& what)
                     {
                         throw base::OptimisationError(base::concatenate(
                             {"blocks with weights: ", what,
                              "; the blocks of the Lagrangian are those of the element Hessian blocks in closed form, "
                              "where hessian_vector(x, v, weights=) differences the linearisation instead (products, "
                              "not blocks): give weights on the items the closed forms cover, or take the products of "
                              "hessian_vector"}));
                     };
                     if (const ProductComputation::EquationWeights equations = computation.equationWeights(*weights);
                         !equations.equations.empty())
                         refuse(
                             "the weights reach an equation response of the second order, whose callable gives "
                             "Hessian-vector products only");
                     try
                     {
                         computation.requireSecondOrder(*weights, static_cast<bool>(context.suppliedAt));
                     }
                     catch (const base::OptimisationError& error)
                     {
                         refuse(error.what());
                     }
                     ensureAdjoints(made, *weights, budget, report);
                     if (!context.usesHessianBlocks)
                         refuse("the closed forms are not asked for (ProductContext::usesHessianBlocks false)");
                     ensureHessianBlocks(made, *weights, budget, report);
                     if (!made.hessianBlocks)
                         refuse(made.blocksReason);
                     report.hasHessianBlocks = true;
                     report.hessianBlocksReason.clear();
                     report.hessianBlockBytes = made.hessianBlocks->bytes();
                     report.hessianBlockCount = static_cast<std::int64_t>(made.hessianBlocks->size());
                     computation.blocks(result, made.hessianBlocks.get(), &made.adjoints);
                     result.hasHessian = true;
                 });
    if (result.hasHessian)
        result.hessianReport = report;
    return result;
}

ProductReport VectorJacobianProducts::jvp(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                          std::span<const double> directions, std::size_t count, std::size_t block,
                                          ProductSink& sink, base::MemoryBudget& budget, const ProductContext& context,
                                          ProductLinearisation* kept, const ProductSelection& selection)
{
    requireBlock(block, "jvp");
    const JvpRows rows = jvpRowsOf(evaluation, selection);
    const auto compute = [&](const ProductLinearisation::State& made, ProductReport& report)
    { made.computation->jvp(directions, count, block, sink, report, rows); };
    if (!rows.isRestricted)
        return products(problem, evaluation, true, {}, budget, context, kept, compute);
    const GradientOptions options = restrictedOptionsOf(rows, directions, count, problem.model().independentCount());
    return products(problem, evaluation, false, std::set<std::int32_t>(rows.ids.begin(), rows.ids.end()), budget,
                    context, kept, compute, &options);
}

} // namespace oos::design
