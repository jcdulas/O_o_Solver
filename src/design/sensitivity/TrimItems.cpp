#include "TrimItems.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <string>

namespace oos::design::detail
{

TrimItems::TrimItems(const Inputs& inputs) : m_in(inputs)
{
    if (m_in.trim == nullptr || !m_in.trim->linearisation)
        throw base::OptimisationError(
            "the sensitivities of the SAERO subcases solve on the trim the analysis kept as a function of the "
            "structure, and this analysis kept none; evaluate the point with its factors kept (keep_analysis=True, "
            "the default)");
    m_linearisation = m_in.trim->linearisation.get();
    m_subcases = m_linearisation->subcases();
    base::DiagnosticSink ignored; // the analysis reported the problems of its PARAMs
    for (const base::EntityId label : m_subcases)
    {
        const assembly::MassAssemblyOptions options =
            assembly::MassAssemblyOptions::fromParams(m_in.params, label, ignored);
        elements::MassOptions massOptions;
        massOptions.formulation = options.formulation;
        m_massOptions.push_back(massOptions);
        m_weightToMass.push_back(options.weightToMass);
        m_appliedAccelerations.push_back(m_linearisation->appliedAcceleration(m_appliedAccelerations.size(),
                                                                              *m_in.budget));
    }
}

bool TrimItems::isTrimSubcase(std::int32_t subcase) const
{
    return std::find(m_subcases.begin(), m_subcases.end(), subcase) != m_subcases.end();
}

std::vector<std::int32_t> TrimItems::subcases() const
{
    return {m_subcases.begin(), m_subcases.end()};
}

std::size_t TrimItems::subcaseIndex(std::int32_t label) const
{
    const auto found = std::find(m_subcases.begin(), m_subcases.end(), label);
    if (found == m_subcases.end())
        throw base::OptimisationError(base::concatenate({"the sensitivities: subcase ", std::to_string(label),
                                                         " is not a SAERO subcase of the analysis; analyse it as a "
                                                         "SAERO subcase (AnalysisKind = SAERO)"}));
    return static_cast<std::size_t>(found - m_subcases.begin());
}

std::span<const double> TrimItems::displacementsOf(std::int32_t subcase) const
{
    return m_linearisation->trimmedState(subcaseIndex(subcase)).displacements;
}

std::size_t TrimItems::directionOf(std::size_t subcase, bool isState, std::size_t variable, bool isUnrestrained)
{
    for (std::size_t d = 0; d < m_directions.size(); ++d)
    {
        const Direction& direction = m_directions[d];
        if (direction.subcase == subcase && direction.isState == isState &&
            (isState || (direction.variable == variable && direction.isUnrestrained == isUnrestrained)))
            return d;
    }
    Direction direction;
    direction.subcase = subcase;
    direction.isState = isState;
    direction.variable = variable;
    direction.isUnrestrained = isUnrestrained;
    m_directions.push_back(std::move(direction));
    return m_directions.size() - 1;
}

void TrimItems::add(const ItemKey& key, ResponseType type, const ResponseItem& label)
{
    Item item;
    item.key = key;
    item.type = type;
    item.subcase = subcaseIndex(label.subcase);
    const solutions::TrimSubcaseResult& result = m_in.trim->subcases.at(item.subcase);
    const auto found =
        std::find_if(result.variables.begin(), result.variables.end(),
                     [&](const solutions::TrimVariableValue& value) { return value.id == label.entity; });
    if (found == result.variables.end())
        throw base::OptimisationError(base::concatenate(
            {"the sensitivity of a ", responseTypeName(type), " item: AeroRigidTrimVariable or AeroControlSurface ",
             std::to_string(label.entity), " is not a variable of the AeroTrim of subcase ",
             std::to_string(label.subcase), "; name a variable of that AeroTrim"}));
    item.variable = static_cast<std::size_t>(found - result.variables.begin());
    if (type == ResponseType::StabilityDerivative)
    {
        item.component = static_cast<std::size_t>(label.component - 1);
        item.isUnrestrained = label.layer == 0;
        item.direction = directionOf(item.subcase, false, item.variable, item.isUnrestrained);
    }
    else
        item.direction = directionOf(item.subcase, true, 0, false);
    m_items.push_back(item);
}

void TrimItems::addStaticItem(const ItemKey& key, std::int32_t subcase)
{
    const std::size_t index = subcaseIndex(subcase);
    (void)directionOf(index, true, 0, false);
    m_staticItems.push_back({key, index});
}

void TrimItems::prepare()
{
    const std::size_t gSize = m_linearisation->gSetSize();
    for (Direction& direction : m_directions)
    {
        const solutions::TrimLinearisation::Directions vectors =
            direction.isState ? m_linearisation->trimmedState(direction.subcase)
                              : m_linearisation->coefficientState(direction.subcase, direction.variable,
                                                                  direction.isUnrestrained, *m_in.budget);
        if (!direction.isState)
            ++m_solves;
        direction.displacements = vectors.displacements;
        direction.motions = vectors.motions;
        direction.stiffnessLoad = base::BudgetedBuffer<double>(*m_in.budget, gSize, "trim stiffness pseudo-loads");
        direction.inertialLoad = base::BudgetedBuffer<double>(*m_in.budget, gSize, "trim inertial pseudo-loads");
    }
}

void TrimItems::addElement(std::size_t /*quantity*/, const elements::ElementFamily& family, std::int64_t row,
                           base::EntityId elementId, const ElementDerivatives& derivatives,
                           const elements::FieldUse& use, const ElementMatrixDerivative& stiffness)
{
    if (isEmpty())
        return;
    // The pseudo-loads of the element are solved on the interface problem of the condensed path, which knows the
    // interiors of its condensed units by their condensed load patterns only (TrimLinearisation::condensedUnitOf):
    // the static sensitivities condense such loads through the interior factors (BoundaryReducedSensitivity), which
    // the trim does not keep. The units the design drives stay full, as the automatic condensation keeps them.
    if (const std::int64_t unit = m_linearisation->condensedUnitOf(elementId);
        unit != 0 && (!stiffness.isEmpty() || use.changesMass))
        throw base::OptimisationError(base::concatenate(
            {"the gradients of the SAERO subcases on the condensed path: the design moves ", family.card(), " ",
             std::to_string(elementId), ", inside condensed unit ", std::to_string(unit),
             ", and the derivatives of the trim solve on the interface problem only (expected the units the design "
             "drives kept full); keep them full: condensation=\"auto\", or Solver.plan(full=problem.driven_units(), "
             "retain=solver.splineNodes())"}));
    const assembly::DofMap& dofMap = *m_in.dofMap;
    // −dK_e·u_e on the element's DOFs, for every direction.
    if (!stiffness.isEmpty())
    {
        const std::size_t n = stiffness.dofs.size();
        std::vector<base::DofIndex> dofs; // bounded by the element's DOFs
        for (const elements::ElementDof& dof : stiffness.dofs)
            dofs.push_back(dofMap.dofOf(dof));
        for (Direction& direction : m_directions)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                double product = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                    product += stiffness.values[i * n + j] * direction.displacements[static_cast<std::size_t>(dofs[j])];
                direction.stiffnessLoad[static_cast<std::size_t>(dofs[i])] -= product;
            }
            direction.isTouched = true;
        }
    }
    // −dM_e·w_e, with the mass of each subcase as the trim assembled it (PARAM COUPMASS, WTMASS).
    if (use.changesMass && family.facets().has(elements::Facet::Mass))
    {
        // The directions of one subcase share its mass: the derivative is taken once per formulation and WTMASS.
        std::map<std::pair<int, double>, ElementMatrixDerivative> byOptions;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const auto key = std::make_pair(static_cast<int>(m_massOptions[s].formulation), m_weightToMass[s]);
            if (byOptions.count(key) == 0)
                byOptions.emplace(key, derivatives.mass(row, m_massOptions[s]));
        }
        for (Direction& direction : m_directions)
        {
            const std::size_t s = direction.subcase;
            const ElementMatrixDerivative& mass =
                byOptions.at(std::make_pair(static_cast<int>(m_massOptions[s].formulation), m_weightToMass[s]));
            if (mass.isEmpty())
                continue;
            const std::size_t n = mass.dofs.size();
            std::vector<base::DofIndex> dofs; // bounded by the element's DOFs
            for (const elements::ElementDof& dof : mass.dofs)
                dofs.push_back(dofMap.dofOf(dof));
            for (std::size_t i = 0; i < n; ++i)
            {
                double product = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                {
                    const auto dof = static_cast<std::size_t>(dofs[j]);
                    double motion = direction.motions[dof];
                    if (direction.isState && !m_appliedAccelerations[s].empty())
                        motion -= m_appliedAccelerations[s][dof];
                    product += mass.values[i * n + j] * motion;
                }
                direction.inertialLoad[static_cast<std::size_t>(dofs[i])] -= m_weightToMass[s] * product;
            }
            direction.isTouched = true;
        }
    }
}

void TrimItems::finishQuantity(std::size_t quantity, const LinearisedItems& linearised)
{
    for (std::size_t d = 0; d < m_directions.size(); ++d)
    {
        Direction& direction = m_directions[d];
        if (!direction.isTouched)
            continue;
        if (direction.isState)
        {
            bool hasStatic = false;
            for (const StaticItem& item : m_staticItems)
                hasStatic = hasStatic || item.subcase == direction.subcase;
            const solutions::TrimLinearisation::StateDerivative derivative =
                m_linearisation->stateDerivative(direction.subcase, direction.stiffnessLoad.span(),
                                                 direction.inertialLoad.span(), hasStatic, *m_in.budget);
            m_solves += hasStatic ? 2 : 1;
            for (std::size_t k = 0; k < m_items.size(); ++k)
                if (m_items[k].direction == d)
                    m_byQuantity[{k, quantity}] += derivative.variables[m_items[k].variable];
            for (std::size_t k = 0; k < m_staticItems.size(); ++k)
            {
                if (m_staticItems[k].subcase != direction.subcase)
                    continue;
                double value = 0.0;
                for (const auto& [dof, weight] : linearised.at(m_staticItems[k].key).weights)
                    value += weight * derivative.displacements[static_cast<std::size_t>(dof)];
                m_byQuantity[{m_items.size() + k, quantity}] += value;
            }
        }
        else
        {
            const std::array<double, 6> coefficients = m_linearisation->coefficientDerivative(
                direction.subcase, direction.variable, direction.isUnrestrained, direction.stiffnessLoad.span(),
                direction.inertialLoad.span(), *m_in.budget);
            ++m_solves;
            for (std::size_t k = 0; k < m_items.size(); ++k)
                if (m_items[k].direction == d)
                    m_byQuantity[{k, quantity}] += coefficients[m_items[k].component];
        }
        std::fill(direction.stiffnessLoad.data(), direction.stiffnessLoad.data() + direction.stiffnessLoad.size(), 0.0);
        std::fill(direction.inertialLoad.data(), direction.inertialLoad.data() + direction.inertialLoad.size(), 0.0);
        direction.isTouched = false;
    }
}

void TrimItems::addExplicitParts(const std::vector<FieldValue>& quantities, LinearisedItems& linearised) const
{
    for (const auto& [entry, derivative] : m_byQuantity)
    {
        const auto [item, quantity] = entry;
        const ItemKey& key = item < m_items.size() ? m_items[item].key : m_staticItems[item - m_items.size()].key;
        VariableRow& row = linearised.at(key).explicitPart;
        for (const auto& [column, gradient] : quantities[quantity].gradient)
            row.emplace_back(column, derivative * gradient);
    }
    // The static items of the SAERO subcases have their whole derivative: no solve of their own, and an equation that
    // reads them chains their explicit part only.
    for (const StaticItem& item : m_staticItems)
        linearised.at(item.key).weights = SparseVector{};
}

} // namespace oos::design::detail
