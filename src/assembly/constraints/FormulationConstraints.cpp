#include "oos/assembly/constraints/FormulationConstraints.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <string>
#include <string_view>

namespace oos::assembly
{

namespace
{

// The concentrated load cards of loadsTable: each acts on the components of one grid point.
constexpr std::array<std::string_view, 6> CONCENTRATED_LOAD_CARDS{
    "PointForceLoad", "PointForceLoad1", "PointForceLoad2", "PointMomentLoad", "PointMomentLoad1", "PointMomentLoad2"};

// The enforced value of a held DOF (0 when it has none).
double enforcedValueOf(const SpcSet& spc, base::DofIndex dof)
{
    const std::span<const EnforcedDisplacement> values = spc.enforcedDisplacements();
    const auto found =
        std::lower_bound(values.begin(), values.end(), dof,
                         [](const EnforcedDisplacement& value, base::DofIndex key) { return value.dof < key; });
    return found != values.end() && found->dof == dof ? found->value : 0.0;
}

bool isHeld(const SpcSet& spc, base::DofIndex dof)
{
    return std::binary_search(spc.sbDofs().begin(), spc.sbDofs().end(), dof) ||
           std::binary_search(spc.sgDofs().begin(), spc.sgDofs().end(), dof);
}

} // namespace

std::vector<elements::HeldGridComponent> heldGridComponentsOf(const SpcSet& spc, const DofMap& dofMap)
{
    std::vector<elements::HeldGridComponent> held; // bounded by the s set
    const auto add = [&](std::span<const base::DofIndex> dofs)
    {
        for (const base::DofIndex dof : dofs)
        {
            const DofLabel label = dofMap.labelOf(dof);
            if (label.kind == input::PointKind::Grid)
                held.push_back({label.row, static_cast<int>(label.component), enforcedValueOf(spc, dof)});
        }
    };
    add(spc.sgDofs());
    add(spc.sbDofs());
    // A component in both sets is held once (by sg, with sb's value: an SPC value on a permanent constraint is 0).
    std::sort(held.begin(), held.end(), [](const elements::HeldGridComponent& a, const elements::HeldGridComponent& b)
              { return a.gridRow != b.gridRow ? a.gridRow < b.gridRow : a.component < b.component; });
    held.erase(std::unique(held.begin(), held.end(),
                           [](const elements::HeldGridComponent& a, const elements::HeldGridComponent& b)
                           { return a.gridRow == b.gridRow && a.component == b.component; }),
               held.end());
    return held;
}

SpcSet withFormulationSupports(const SpcSet& spc, const DofMap& dofMap, const elements::FormulationModel* model,
                               elements::EntityLayout::Kind kind, base::MemoryBudget& budget,
                               base::DiagnosticSink& sink)
{
    if (model == nullptr)
        return spc;
    std::vector<elements::DerivedSupport> supports;
    model->deriveSupports(kind, heldGridComponentsOf(spc, dofMap), supports, sink);
    std::vector<DerivedSupportDof> derived; // bounded by the entity DOFs of the held edges and faces
    derived.reserve(supports.size());
    for (const elements::DerivedSupport& support : supports)
    {
        DerivedSupportDof dof{dofMap.dofOf(support.dof), support.value, {}};
        for (const elements::FormulationRelationTerm& term : support.carriedTo)
            if (term.coefficient != 0.0)
                dof.carriedTo.push_back({dofMap.dofOf(term.dof), term.coefficient});
        derived.push_back(std::move(dof));
    }
    std::sort(derived.begin(), derived.end(),
              [](const DerivedSupportDof& a, const DerivedSupportDof& b) { return a.dof < b.dof; });
    derived.erase(std::unique(derived.begin(), derived.end(),
                              [](const DerivedSupportDof& a, const DerivedSupportDof& b) { return a.dof == b.dof; }),
                  derived.end());
    return spc.withDerivedSupports(std::move(derived), budget);
}

void UsedElsewhere::markConcentratedLoads(const input::ModelSnapshot& snapshot, const DofMap& dofMap)
{
    const input::LoadTable* loads = snapshot.findLoads();
    if (loads == nullptr)
        return;
    const base::ContractReader& table = loads->loads();
    const base::DictionaryView cards = loads->loadCards();
    const std::span<const std::int32_t> targets = table.values<std::int32_t>("targetId", base::NullPolicy::Allow);
    const base::ValidityView validTargets = table.validity("targetId");
    for (std::int64_t row = 0; row < loads->loadRowCount(); ++row)
    {
        const std::string_view card = cards[row];
        if (std::find(CONCENTRATED_LOAD_CARDS.begin(), CONCENTRATED_LOAD_CARDS.end(), card) ==
                CONCENTRATED_LOAD_CARDS.end() ||
            !validTargets.isValid(row))
            continue;
        const std::optional<std::int32_t> grid = dofMap.points().findGrid(targets[static_cast<std::size_t>(row)]);
        if (!grid)
            continue;
        if (loaded.size() != flags.size())
            loaded.assign(flags.size(), 0);
        const int first = card.starts_with("PointMomentLoad") ? 4 : 1;
        for (int component = first; component < first + 3; ++component)
        {
            const auto dof = static_cast<std::size_t>(dofMap.gridDof(*grid, component));
            flags[dof] = 1;
            loaded[dof] = 1;
        }
    }
}

void UsedElsewhere::markStiffened(std::span<const std::uint8_t> stiffenedComponents, const DofMap& dofMap)
{
    for (std::int32_t node = 0; node < dofMap.gridCount(); ++node)
    {
        const std::uint8_t bits = stiffenedComponents[static_cast<std::size_t>(node)];
        for (int c = 0; c < 6; ++c)
            if ((bits & (1U << static_cast<unsigned>(c))) != 0)
                flags[static_cast<std::size_t>(dofMap.gridDof(node, c + 1))] = 1;
    }
}

void UsedElsewhere::markIndependentTerms(const RigidElimination& gm)
{
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
        for (const ConstraintTerm& term : gm.relationTerms(i))
            flags[static_cast<std::size_t>(term.dof)] = 1;
}

FormulationConstraints::FormulationConstraints(const elements::FormulationModel& model,
                                               elements::EntityLayout::Kind kind, const DofMap& dofMap,
                                               const SpcSet& spc, const UsedElsewhere& usedElsewhere,
                                               const RigidElimination& userRelations, base::DiagnosticSink& sink)
    : m_model(&model), m_kind(kind), m_dofMap(&dofMap), m_relations(dofMap)
{
    const std::vector<elements::HeldGridComponent> held = heldGridComponentsOf(spc, dofMap);

    // The ties (section 3.4).
    std::vector<elements::ElementDof> dependents;
    std::vector<std::size_t> offsets{0};
    std::vector<elements::FormulationRelationTerm> terms;
    model.ties(kind, held, dependents, offsets, terms, sink);
    std::vector<FormulationTerm> row; // bounded by one relation
    for (std::size_t i = 0; i < dependents.size(); ++i)
    {
        row.clear();
        for (std::size_t t = offsets[i]; t < offsets[i + 1]; ++t)
            row.push_back({terms[t].dof, terms[t].coefficient});
        m_relations.add(dependents[i], row);
    }

    // The determined grid components (section 5.4): held, the relation's own, observed, or excluded.
    std::vector<elements::HeldGridComponent> determined;
    model.determinedGridComponents(kind, determined);
    std::set<std::int32_t> notTransmitted;
    std::set<std::int32_t> loadedZero;
    std::set<std::int32_t> heldZero;
    std::vector<elements::FormulationRelationTerm> determinedTerms;
    for (const elements::HeldGridComponent& component : determined)
    {
        const base::DofIndex dof = dofMap.gridDof(component.gridRow, component.component);
        if (isHeld(spc, dof))
            continue;
        if (userRelations.isDependent(dof))
        {
            notTransmitted.insert(component.gridRow);
            continue;
        }
        if (usedElsewhere.flags[static_cast<std::size_t>(dof)] == 0)
        {
            m_excluded.push_back(dof);
            continue;
        }
        determinedTerms.clear();
        model.determinedRow(kind, component.gridRow, component.component, determinedTerms);
        if (determinedTerms.empty())
        {
            // A component the formulation holds at zero (the rotation of a grid point of solids alone): a load on it
            // acts on nothing (spec section 7: refused), an element on it acts against a support.
            const bool isLoaded =
                !usedElsewhere.loaded.empty() && usedElsewhere.loaded[static_cast<std::size_t>(dof)] != 0;
            if (isLoaded)
                loadedZero.insert(component.gridRow);
            else
                heldZero.insert(component.gridRow);
        }
        row.clear();
        for (const elements::FormulationRelationTerm& term : determinedTerms)
            row.push_back({term.dof, term.coefficient});
        m_relations.add(elements::ElementDof::grid(component.gridRow, component.component), row);
    }
    for (const std::int32_t gridRow : notTransmitted)
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "rotation-not-transmitted";
        diagnostic.card = "Node";
        diagnostic.id = dofMap.gridId(gridRow);
        diagnostic.message = base::concatenate(
            {"grid point ", std::to_string(dofMap.gridId(gridRow)),
             " is a grid point of shells of the mixed hybrid formulation whose rotation a relation of the model makes "
             "dependent: the relation moves that rotation, but the shells, which have no rotation of their own at the "
             "grid point, are not tied to it; tie the relation to the translations of the shells' grid points "
             "instead"});
        sink.report(std::move(diagnostic));
    }
    for (const std::int32_t gridRow : loadedZero)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "load-unsupported";
        diagnostic.card = "Node";
        diagnostic.id = dofMap.gridId(gridRow);
        diagnostic.message = base::concatenate(
            {"expected no moment at grid point ", std::to_string(dofMap.gridId(gridRow)),
             ", a grid point of solids alone in the mixed hybrid formulation: the solids have no rotation there, so "
             "the moment would act on nothing; apply forces at the grid points of the solids instead"});
        sink.report(std::move(diagnostic));
    }
    for (const std::int32_t gridRow : heldZero)
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "rotation-held";
        diagnostic.card = "Node";
        diagnostic.id = dofMap.gridId(gridRow);
        diagnostic.message = base::concatenate(
            {"grid point ", std::to_string(dofMap.gridId(gridRow)),
             " is a grid point of solids alone in the mixed hybrid formulation, which have no rotation there: an "
             "element or a relation that uses its rotation acts against a rotation held at zero"});
        sink.report(std::move(diagnostic));
    }
    std::sort(m_excluded.begin(), m_excluded.end());
}

void FormulationConstraints::fillExcluded(std::span<double> displacementsG) const
{
    if (m_excluded.empty())
        return;
    const elements::NodalDisplacements displacements =
        elements::NodalDisplacements::ofGSet(displacementsG, static_cast<std::size_t>(m_dofMap->gridCount()),
                                             static_cast<std::size_t>(m_dofMap->scalarCount()), &m_dofMap->entities());
    // The values are gathered first: the formulation reads the vector while they are computed.
    std::vector<std::pair<base::DofIndex, double>> values; // bounded by the determined components
    m_model->fillDeterminedComponents(m_kind, displacements,
                                      [&](std::int32_t gridRow, int component, double value)
                                      {
                                          const base::DofIndex dof = m_dofMap->gridDof(gridRow, component);
                                          if (std::binary_search(m_excluded.begin(), m_excluded.end(), dof))
                                              values.emplace_back(dof, value);
                                      });
    for (const auto& [dof, value] : values)
        displacementsG[static_cast<std::size_t>(dof)] = value;
}

} // namespace oos::assembly
