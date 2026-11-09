#include "oos/solutions/sol144/TrimLinearisation.hpp"

#include "TrimStructure.hpp"

#include <algorithm>
#include <stdexcept>

namespace oos::solutions
{

using namespace trim;

namespace
{

void requireGSize(std::span<const double> values, std::size_t gSize, const char* what)
{
    if (values.size() != gSize)
        throw std::invalid_argument(base::concatenate({"TrimLinearisation: expected ", std::to_string(gSize), " ", what,
                                                       " (the g set), found ", std::to_string(values.size())}));
}

// The g-set loads of the box forces q·Δcp_j·A_j through the splines (the trimmed loads of TrimSolution::run).
std::vector<double> boxLoadsOf(const TrimLinearisation::State& state, std::span<const double> pressures, double q)
{
    const assembly::DofMap& dofMap = state.model->dofMap();
    const std::size_t n = state.boxes->size();
    std::vector<double> forces(n, 0.0); // bounded by the boxes
    for (std::size_t j = 0; j < n; ++j)
        forces[j] = q * pressures[j] * (*state.boxes)[j].area;
    std::vector<double> loads(static_cast<std::size_t>(dofMap.gSetSize()), 0.0);
    addPointLoads(state.splines->structuralLoads(forces), dofMap, loads);
    return loads;
}

} // namespace

TrimLinearisation::TrimLinearisation(std::unique_ptr<State> state) : m_state(std::move(state))
{
    if (!m_state || !m_state->model)
        throw std::invalid_argument("TrimLinearisation: expected the parts of a trim run, found none");
}

TrimLinearisation::~TrimLinearisation() = default;

std::vector<base::EntityId> TrimLinearisation::subcases() const
{
    std::vector<base::EntityId> labels;
    for (const TrimSubcaseState& subcase : m_state->subcases)
        labels.push_back(subcase.label);
    return labels;
}

std::size_t TrimLinearisation::gSetSize() const noexcept
{
    return static_cast<std::size_t>(m_state->model->dofMap().gSetSize());
}

std::vector<std::string> TrimLinearisation::variableLabels(std::size_t subcase) const
{
    std::vector<std::string> labels;
    for (const aero::TrimSetting& setting : m_state->subcases.at(subcase).trimCase.settings)
        labels.push_back(setting.label);
    return labels;
}

bool TrimLinearisation::hasSupports(std::size_t subcase) const
{
    return m_state->groups[m_state->subcases.at(subcase).group]->structure->supportCount() > 0;
}

std::int64_t TrimLinearisation::condensedUnitOf(base::EntityId elementId) const
{
    if (!m_state->plan)
        return 0;
    for (const reduction::UnitPlan& unit : m_state->plan->units())
        if (unit.decision == reduction::UnitDecision::Condensed &&
            std::binary_search(unit.elementIds.begin(), unit.elementIds.end(), elementId))
            return unit.unitId;
    return 0;
}

TrimLinearisation::Directions TrimLinearisation::trimmedState(std::size_t subcase) const
{
    const TrimSubcaseState& state = m_state->subcases.at(subcase);
    return {state.displacements, state.supportMotion};
}

namespace
{

// δΔcp(Q) = F⁻¹·AEQR·S·K⁻¹·Q of a structural load Q = load − M·D_r·accelerations (g set; `accelerations` of the
// supports, empty for none), F the factor of the pressures of the restrained structure (D − AEQR·q·E) or of the free
// one (D − AEQR·q·E_R): the pressures the structure's deformation under Q adds (TrimLinearisation class comment). The
// inertial part goes to the structure as accelerations, not as loads: the condensed path forms it from the units'
// inertial patterns, its loads on the interior points of the condensed units being refused.
std::vector<double> particularPressures(const TrimLinearisation::State& state, TrimGroup& group,
                                        const TrimSubcaseState& subcase, std::span<const double> load,
                                        std::span<const double> accelerations, bool isUnrestrained,
                                        base::MemoryBudget& budget)
{
    const assembly::DofMap& dofMap = state.model->dofMap();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const std::size_t n = state.boxes->size();
    linalg::DenseMatrix loads(budget, gSize, 1, "trim derivative load");
    std::copy(load.begin(), load.end(), &loads(0, 0));
    const std::size_t r = group.structure->supportCount();
    linalg::DenseMatrix motions(budget, r, accelerations.empty() ? 0 : 1, "trim derivative accelerations");
    for (std::size_t k = 0; k < motions.rows() && !accelerations.empty(); ++k)
        motions(k, 0) = accelerations[k];
    linalg::DenseMatrix displacements(budget, gSize, 1, "trim derivative deformation");
    group.structure->solve(loads.view(), motions.view(), displacements.view(), false, budget);
    linalg::DenseMatrix pressures(budget, n, 1, "trim derivative pressures");
    slopesOf(*state.splines, dofMap, std::span<const double>(&displacements(0, 0), gSize), pressures.view(), 0);
    for (std::size_t j = 0; j < n; ++j)
        pressures(j, 0) *= subcase.feedback;
    (isUnrestrained ? subcase.freeFactor : subcase.restrainedFactor)->solve(pressures.view());
    return std::vector<double>(&pressures(0, 0), &pressures(0, 0) + n);
}

} // namespace

TrimLinearisation::StateDerivative TrimLinearisation::stateDerivative(std::size_t subcase,
                                                                      std::span<const double> stiffnessLoad,
                                                                      std::span<const double> inertialLoad,
                                                                      bool withDisplacements,
                                                                      base::MemoryBudget& budget)
{
    TrimSubcaseState& state = m_state->subcases.at(subcase);
    TrimGroup& group = *m_state->groups[state.group];
    const std::size_t gSize = gSetSize();
    requireGSize(stiffnessLoad, gSize, "stiffness pseudo-loads");
    requireGSize(inertialLoad, gSize, "inertial pseudo-loads");
    const std::size_t n = m_state->boxes->size();
    const std::size_t r = group.structure->supportCount();
    const std::size_t m = state.trimCase.settings.size();
    const double q = state.dynamicPressure;

    // Q = −∂K·u − ∂M·D_r·ü_r, its pressures δΔcp(Q), and the constant of the trim equations: q·D_rᵀ·T·δΔcp(Q) +
    // D_rᵀ·(−∂M·D_r·ü_r).
    std::vector<double> load(gSize, 0.0);
    for (std::size_t dof = 0; dof < gSize; ++dof)
        load[dof] = stiffnessLoad[dof] + inertialLoad[dof];
    const std::vector<double> particular = particularPressures(*m_state, group, state, load, {}, false, budget);
    std::vector<double> constant(r, 0.0);
    const linalg::DenseMatrix& modes = group.structure->supportModes();
    for (std::size_t k = 0; k < r; ++k)
    {
        double aerodynamic = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            aerodynamic += group.flexibility.supportLoads(k, j) * particular[j];
        double inertial = 0.0;
        for (std::size_t dof = 0; dof < gSize; ++dof)
            inertial += modes(dof, k) * inertialLoad[dof];
        constant[k] = q * aerodynamic + inertial;
    }
    // The free variables of the same trim system, the fixed ones held (their values do not move with the design).
    aero::TrimCase held = state.trimCase;
    for (aero::TrimSetting& setting : held.settings)
        if (setting.state == aero::TrimVariableState::Fixed)
            setting.value = 0.0;

    StateDerivative derivative;
    derivative.variables = aero::TrimSystem::solve(held, state.equations, constant, budget);
    derivative.supportAccelerations.assign(r, 0.0);
    derivative.pressures = particular;
    for (std::size_t x = 0; x < m; ++x)
    {
        for (std::size_t k = 0; k < r; ++k)
            derivative.supportAccelerations[k] += state.accelerations(k, x) * derivative.variables[x];
        for (std::size_t j = 0; j < n; ++j)
            derivative.pressures[j] += state.restrained(j, x) * derivative.variables[x];
    }
    if (!withDisplacements)
        return derivative;

    // du = K⁻¹·(q·T·dΔcp − M·D_r·dü_r + Q), every DOF the outputs read (the condensed path recovers its units).
    const std::vector<double> boxLoads = boxLoadsOf(*m_state, derivative.pressures, q);
    linalg::DenseMatrix loads(budget, gSize, 1, "trim derivative loads");
    for (std::size_t dof = 0; dof < gSize; ++dof)
        loads(dof, 0) = boxLoads[dof] + load[dof];
    linalg::DenseMatrix accelerations(budget, r, r > 0 ? 1 : 0, "trim derivative accelerations");
    for (std::size_t k = 0; k < r; ++k)
        accelerations(k, 0) = derivative.supportAccelerations[k];
    linalg::DenseMatrix displacements(budget, gSize, 1, "trim derivative displacements");
    group.structure->solve(loads.view(), accelerations.view(), displacements.view(), true, budget);
    derivative.displacements.assign(&displacements(0, 0), &displacements(0, 0) + gSize);
    return derivative;
}

TrimLinearisation::Directions TrimLinearisation::coefficientState(std::size_t subcase, std::size_t variable,
                                                                  bool isUnrestrained, base::MemoryBudget& budget)
{
    TrimSubcaseState& state = m_state->subcases.at(subcase);
    TrimGroup& group = *m_state->groups[state.group];
    if (variable >= state.trimCase.settings.size())
        throw std::invalid_argument(base::concatenate({"TrimLinearisation: expected a variable of the trim case (",
                                                       std::to_string(state.trimCase.settings.size()),
                                                       "), found index ", std::to_string(variable)}));
    const std::size_t r = group.structure->supportCount();
    if (isUnrestrained && r == 0)
        throw std::invalid_argument(
            "TrimLinearisation: the unrestrained coefficients need FreeBodySupport DOFs; the subcase has "
            "none");
    const auto key = std::make_pair(variable, isUnrestrained);
    if (const auto found = state.coefficientStates.find(key); found != state.coefficientStates.end())
        return {found->second.first, found->second.second};

    const std::size_t gSize = gSetSize();
    const std::size_t n = m_state->boxes->size();
    const double q = state.dynamicPressure;
    linalg::DenseMatrix& column = isUnrestrained ? state.unrestrained : state.restrained;
    const std::vector<double> boxLoads = boxLoadsOf(*m_state, std::span<const double>(&column(0, variable), n), q);
    // ü_r(x) restrained; a_x = m_rr⁻¹·D_rᵀ·q·T·Δcp_x unrestrained (the inertia of the free body).
    linalg::DenseMatrix accelerations(budget, r, r > 0 ? 1 : 0, "coefficient accelerations");
    for (std::size_t k = 0; k < r; ++k)
    {
        if (!isUnrestrained)
        {
            accelerations(k, 0) = state.accelerations(k, variable);
            continue;
        }
        double work = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            work += group.flexibility.supportLoads(k, j) * column(j, variable);
        accelerations(k, 0) = q * work;
    }
    if (isUnrestrained)
        group.supportMass->solve(accelerations.view());
    linalg::DenseMatrix loads(budget, gSize, 1, "coefficient loads");
    std::copy(boxLoads.begin(), boxLoads.end(), &loads(0, 0));
    linalg::DenseMatrix displacements(budget, gSize, 1, "coefficient displacements");
    group.structure->solve(loads.view(), accelerations.view(), displacements.view(), false, budget);
    std::vector<double> motion(gSize, 0.0);
    const linalg::DenseMatrix& modes = group.structure->supportModes();
    for (std::size_t k = 0; k < r; ++k)
        for (std::size_t dof = 0; dof < gSize; ++dof)
            motion[dof] += modes(dof, k) * accelerations(k, 0);
    auto& entry = state.coefficientStates[key];
    entry.first.assign(&displacements(0, 0), &displacements(0, 0) + gSize);
    entry.second = std::move(motion);
    return {entry.first, entry.second};
}

std::array<double, 6> TrimLinearisation::coefficientDerivative(std::size_t subcase, std::size_t variable,
                                                               bool isUnrestrained,
                                                               std::span<const double> stiffnessLoad,
                                                               std::span<const double> inertialLoad,
                                                               base::MemoryBudget& budget)
{
    TrimSubcaseState& state = m_state->subcases.at(subcase);
    TrimGroup& group = *m_state->groups[state.group];
    const std::size_t gSize = gSetSize();
    requireGSize(stiffnessLoad, gSize, "stiffness pseudo-loads");
    requireGSize(inertialLoad, gSize, "inertial pseudo-loads");
    (void)coefficientState(subcase, variable, isUnrestrained, budget); // checks the variable and the supports
    const std::size_t r = group.structure->supportCount();
    std::vector<double> relief;
    if (isUnrestrained)
    {
        // R·y = y − M·D_r·m_rr⁻¹·D_rᵀ·y: the derivative of the relief of the free body, its inertia M·D_r·p that of
        // the support accelerations p = m_rr⁻¹·D_rᵀ·y.
        const linalg::DenseMatrix& modes = group.structure->supportModes();
        linalg::DenseMatrix projection(budget, r, 1, "relieved inertia");
        for (std::size_t k = 0; k < r; ++k)
        {
            double work = 0.0;
            for (std::size_t dof = 0; dof < gSize; ++dof)
                work += modes(dof, k) * inertialLoad[dof];
            projection(k, 0) = work;
        }
        group.supportMass->solve(projection.view());
        relief.assign(&projection(0, 0), &projection(0, 0) + r);
    }
    std::vector<double> load(gSize, 0.0);
    for (std::size_t dof = 0; dof < gSize; ++dof)
        load[dof] = inertialLoad[dof] + stiffnessLoad[dof];
    const std::vector<double> pressures =
        particularPressures(*m_state, group, state, load, relief, isUnrestrained, budget);
    return aero::ReferenceCoefficients::of(m_state->reference,
                                           combine(group.flexibility.structuralResultant, pressures));
}

} // namespace oos::solutions
