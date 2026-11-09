#include "oos/design/aids/ApproximationHints.hpp"

#include "../sensitivity/SensitivityComputation.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

// A power below this (in absolute value) is no dependence: the variable gets the direct approximation.
constexpr double NO_EXPONENT = 0.5;

// The coefficients a_p of the cubic Σ a_p·s^p through the values at HINT_SCALES (Gaussian elimination with partial
// pivoting on the 4 × 4 Vandermonde system: its condition at these scales is about 1e3, far from the rounding).
std::array<double, HINT_POWER_COUNT> fitCubic(const std::array<double, HINT_POWER_COUNT>& values)
{
    constexpr std::size_t N = HINT_POWER_COUNT;
    std::array<std::array<double, N + 1>, N> system{};
    for (std::size_t k = 0; k < N; ++k)
    {
        double power = 1.0;
        for (std::size_t p = 0; p < N; ++p)
        {
            system[k][p] = power;
            power *= HINT_SCALES[k];
        }
        system[k][N] = values[k];
    }
    for (std::size_t column = 0; column < N; ++column)
    {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < N; ++row)
            if (std::abs(system[row][column]) > std::abs(system[pivot][column]))
                pivot = row;
        std::swap(system[column], system[pivot]);
        for (std::size_t row = column + 1; row < N; ++row)
        {
            const double factor = system[row][column] / system[column][column];
            for (std::size_t c = column; c <= N; ++c)
                system[row][c] -= factor * system[column][c];
        }
    }
    std::array<double, N> coefficients{};
    for (std::size_t row = N; row-- > 0;)
    {
        double sum = system[row][N];
        for (std::size_t c = row + 1; c < N; ++c)
            sum -= system[row][c] * coefficients[c];
        coefficients[row] = sum / system[row][row];
    }
    return coefficients;
}

// The shares of each power and the exponent Σ p·a_p / Σ a_p of a quantity sampled at HINT_SCALES (zero shares for a
// quantity that does not change).
void sharesOf(const std::array<double, HINT_POWER_COUNT>& samples, std::array<double, HINT_POWER_COUNT>& shares,
              double& exponent, double& atDesign)
{
    atDesign = samples[1]; // HINT_SCALES[1] = 1: the design point
    shares = {};
    exponent = 0.0;
    if (!(std::abs(atDesign) > 0.0))
        return;
    const std::array<double, HINT_POWER_COUNT> coefficients = fitCubic(samples);
    for (std::size_t p = 0; p < HINT_POWER_COUNT; ++p)
    {
        shares[p] = coefficients[p] / atDesign;
        exponent += static_cast<double>(p) * shares[p];
    }
}

int roundedPower(double exponent)
{
    const int power = static_cast<int>(std::lround(exponent));
    return std::clamp(power, -HINT_LARGEST_POWER, HINT_LARGEST_POWER);
}

bool isDisplacementType(std::string_view type)
{
    return type == "Displacement" || type == "Compliance" || type == "MeanDisplacement" ||
           type == "DeflectionToThickness" || type == "Rotation";
}

bool isMassType(std::string_view type)
{
    return type == "Weight" || type == "Volume" || type == "FractionalMass";
}

// The energies and masses of the relations, on the snapshots with one field scaled: the linearisation of
// StaticSensitivity gives the subcases with their displacements, the element set and the DOF map.
class HintsComputation : public detail::SensitivityComputation
{
public:
    HintsComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options, base::MemoryBudget& budget)
        : detail::SensitivityComputation(inputs, options, budget)
    {
        m_preparesEverySubcase = true;
    }

    std::vector<RelationHint> run()
    {
        prepareSubcases();
        const DesignModel& model = *m_in.model;
        const AffectedElements affected(model, m_set);
        std::vector<double> fields;
        for (const FieldValue& field : m_in.design->fields)
            fields.push_back(field.value);
        std::vector<double> shapes;
        for (const FieldValue& shape : m_in.design->shapes)
            shapes.push_back(shape.value);
        std::vector<RelationHint> hints;
        for (std::size_t r = 0; r < model.relations().size(); ++r)
        {
            RelationHint hint;
            hint.relation = r;
            hint.label = model.relations()[r].label();
            const std::vector<ElementRef>& elements = affected.of(r);
            if (elements.empty() || fields[r] == 0.0)
            {
                hints.push_back(std::move(hint));
                continue;
            }
            std::set<std::pair<std::string, std::int32_t>> reached;
            for (const ElementRef& element : elements)
                reached.emplace(m_set.batches()[element.batch].view.card(),
                                m_set.batches()[element.batch].view.elementId(element.row));
            std::array<double, HINT_POWER_COUNT> energies{};
            std::array<double, HINT_POWER_COUNT> masses{};
            for (std::size_t k = 0; k < HINT_POWER_COUNT; ++k)
            {
                std::vector<double> scaled = fields;
                scaled[r] *= HINT_SCALES[k];
                const input::ColumnOverlay overlay = m_in.targets->overlayOf(model.shapes(), scaled, shapes);
                const auto snapshot =
                    std::make_shared<const input::ModelSnapshot>(overlay.applyTo(*m_in.base, m_in.base));
                const elements::SnapshotIndex index(*snapshot, m_budget);
                const assembly::ElementSet set(index, elementRegistry());
                energies[k] = energyOf(set, elements);
                const ElementFilter isReached = [&reached](std::string_view card, std::int32_t elementId, std::int32_t)
                { return reached.count({std::string(card), elementId}) != 0; };
                masses[k] = MassProperties::rigidBodyMass(*snapshot, {0.0, 0.0, 0.0}, m_in.massFormulation, isReached,
                                                          m_budget)(0, 0);
            }
            sharesOf(energies, hint.stiffnessShares, hint.stiffnessExponent, hint.energy);
            sharesOf(masses, hint.massShares, hint.massExponent, hint.mass);
            hints.push_back(std::move(hint));
        }
        return hints;
    }

private:
    // Σ over the analysed subcases and the elements of u_eᵀ·K_e·u_e on an element set.
    [[nodiscard]] double energyOf(const assembly::ElementSet& set, const std::vector<ElementRef>& elements) const
    {
        double energy = 0.0;
        elements::ElementMatrixBlock block;
        for (const ElementRef& element : elements)
        {
            const assembly::AssemblyBatch& batch = set.batches()[element.batch];
            if (!batch.family->facets().has(elements::Facet::Stiffness))
                continue;
            block.clear();
            batch.family->computeStiffness(batch.view, {element.row, element.row + 1}, block);
            if (block.empty())
                continue;
            const std::span<const elements::ElementDof> dofs = block.dofs(0);
            const std::span<const double> matrix = block.matrix(0);
            const std::size_t n = dofs.size();
            for (const detail::SubcaseState& state : m_subcases)
                for (std::size_t i = 0; i < n; ++i)
                {
                    const double ui = state.displacements[static_cast<std::size_t>(m_dofMap.dofOf(dofs[i]))];
                    if (ui == 0.0)
                        continue;
                    for (std::size_t j = 0; j < n; ++j)
                        energy += ui * matrix[i * n + j] *
                                  state.displacements[static_cast<std::size_t>(m_dofMap.dofOf(dofs[j]))];
                }
        }
        return energy;
    }
};

} // namespace

const ResponseHint* ApproximationHintsResult::find(std::int32_t responseId) const noexcept
{
    for (const ResponseHint& hint : responses)
        if (hint.responseId == responseId)
            return &hint;
    return nullptr;
}

std::string ApproximationHints::formOf(int power)
{
    if (power == 1)
        return "x";
    if (power == -1)
        return "1/x";
    if (power > 0)
        return std::string("x^").append(std::to_string(power));
    return std::string("1/x^").append(std::to_string(-power));
}

ApproximationHintsResult ApproximationHints::compute(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                     base::MemoryBudget& budget,
                                                     const solutions::StaticOptions& analysisOptions)
{
    const ResponseSubcases subcases = problem.responseSubcases(evaluation);
    const CallableRegistry noCallables;
    const StaticSensitivityInputs inputs =
        problem.sensitivityInputs(evaluation, subcases, noCallables, analysisOptions, EQUATION_BATCH_ITEMS);
    const GradientOptions options;
    HintsComputation computation(inputs, options, budget);
    ApproximationHintsResult result;
    result.relations = computation.run();

    // The variables: the exponents of their relations, weighted by the energies (masses) and the elasticities.
    const DesignModel& model = problem.model();
    const DesignValues& values = evaluation.analysis.point.values();
    const std::vector<double>& x = evaluation.analysis.point.x();
    const std::vector<std::size_t>& independent = model.independentIndices();
    for (std::size_t column = 0; column < independent.size(); ++column)
    {
        VariableHint hint;
        hint.column = column;
        hint.label = model.variables()[independent[column]].label;
        double energy = 0.0;
        double mass = 0.0;
        for (const RelationHint& relation : result.relations)
        {
            const FieldValue& field = values.fields[relation.relation];
            double rate = 0.0;
            for (const auto& [variable, derivative] : field.gradient)
                if (variable == column)
                    rate += derivative;
            if (rate == 0.0 || field.value == 0.0)
                continue;
            const double elasticity = x[column] * rate / field.value; // d ln v / d ln x
            hint.stiffnessExponent += relation.energy * relation.stiffnessExponent * elasticity;
            energy += relation.energy;
            hint.massExponent += relation.mass * relation.massExponent * elasticity;
            mass += relation.mass;
        }
        hint.stiffnessExponent = energy != 0.0 ? hint.stiffnessExponent / energy : 0.0;
        hint.massExponent = mass != 0.0 ? hint.massExponent / mass : 0.0;
        result.variables.push_back(std::move(hint));
    }

    // The responses, by their DesignConstrScreen type.
    for (const ResponseValues& response : evaluation.responses)
    {
        ResponseHint hint;
        hint.responseId = response.id;
        hint.type = problem.responses().screeningType(response.id);
        for (const VariableHint& variable : result.variables)
        {
            int power = 1;
            if (isDisplacementType(hint.type) && std::abs(variable.stiffnessExponent) >= NO_EXPONENT)
                power = -roundedPower(variable.stiffnessExponent);
            else if (isMassType(hint.type) && std::abs(variable.massExponent) >= NO_EXPONENT)
                power = roundedPower(variable.massExponent);
            else if (!isMassType(hint.type) && hint.type != "DesignResp2" && hint.type != "DesignResp3" &&
                     std::abs(variable.stiffnessExponent) >= NO_EXPONENT)
                power = -1;
            if (power == 0)
                power = 1;
            hint.exponents.push_back(power);
            hint.forms.push_back(formOf(power));
        }
        result.responses.push_back(std::move(hint));
    }
    return result;
}

} // namespace oos::design
