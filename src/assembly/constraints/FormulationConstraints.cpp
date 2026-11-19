#include "oos/assembly/constraints/FormulationConstraints.hpp"

#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/assembly/dofs/EntityForceCarry.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
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

// Whether `dof` is a support the formulation derived from the supports of the set (withFormulationSupports).
bool isDerived(const SpcSet& spc, base::DofIndex dof)
{
    const std::span<const DerivedSupportDof> derived = spc.derivedSupports();
    const auto found = std::lower_bound(derived.begin(), derived.end(), dof,
                                        [](const DerivedSupportDof& support, base::DofIndex key)
                                        { return support.dof < key; });
    return found != derived.end() && found->dof == dof;
}

bool isBefore(const elements::HeldGridComponent& a, const elements::HeldGridComponent& b)
{
    return a.gridRow != b.gridRow ? a.gridRow < b.gridRow : a.component < b.component;
}

elements::ElementDof elementDofOf(const DofLabel& label)
{
    if (label.kind == input::PointKind::Grid)
        return elements::ElementDof::grid(label.row, static_cast<int>(label.component));
    if (label.kind == input::PointKind::Scalar)
        return elements::ElementDof::scalar(label.row);
    return elements::ElementDof::entity(label.row, static_cast<int>(label.component));
}

// A tied row is written on an unknown whose coefficient is at least this times the largest of the row.
constexpr double PIVOT_RATIO = 1.0e-3;
// An axis lies in the plane of the shells of a grid point when its part along their normal is below this.
constexpr double IN_PLANE = 1.0e-8;

// The directions (in the CD system of a grid point of shells) about which a support of the rotation components
// `held` holds the mean rotation of the shells, `normal` their common normal there when they have one: the axes held
// that lie in the plane of the shells. The rotation about the normal is not held: it is the rotation of the membrane
// field in its plane, which the same support leaves free on a shell without drilling stiffness (a support of the
// rotation about the normal at every grid point of a plate, written to remove that singularity, must not stiffen
// it). Each held axis is a direction at a fold, where every rotation bends a shell, and at a grid point that holds
// its rotations alone (`isAlone`: no grid point joined to it by an edge holds a rotation), the clamp that holds a
// free body at one point.
std::vector<std::array<double, 3>> heldRotationDirections(const std::array<bool, 3>& held,
                                                          const std::optional<base::Vec3>& normal, bool isAlone)
{
    std::vector<std::array<double, 3>> directions;
    const auto axis = [](std::size_t s)
    {
        std::array<double, 3> e{};
        e[s] = 1.0;
        return e;
    };
    const auto count = static_cast<int>(held[0]) + static_cast<int>(held[1]) + static_cast<int>(held[2]);
    if (count == 0)
        return directions;
    if (!normal || isAlone)
    {
        for (std::size_t s = 0; s < 3; ++s)
            if (held[s])
                directions.push_back(axis(s));
        return directions;
    }
    const base::Vec3& n = *normal;
    if (count == 1)
    {
        const std::size_t s = held[0] ? 0 : (held[1] ? 1 : 2);
        if (std::fabs(n[s]) <= IN_PLANE)
            directions.push_back(axis(s));
        return directions;
    }
    if (count == 2)
    {
        // The held axes span the plane normal to the free axis f; it meets the plane of the shells along e_f × n.
        const std::size_t f = !held[0] ? 0 : (!held[1] ? 1 : 2);
        const std::size_t a = (f + 1) % 3;
        const std::size_t b = (f + 2) % 3;
        std::array<double, 3> d{};
        d[a] = -n[b]; // (e_f × n)_a = −n_b, (e_f × n)_b = n_a for (f, a, b) cyclic
        d[b] = n[a];
        const double length = std::hypot(d[a], d[b]);
        if (length <= IN_PLANE)
        {
            directions.push_back(axis(a)); // the held plane is the plane of the shells
            directions.push_back(axis(b));
            return directions;
        }
        d[a] /= length;
        d[b] /= length;
        directions.push_back(d);
        return directions;
    }
    // The three axes held: the two rotations about axes of the plane of the shells, t1 the CD axis farthest from the
    // normal projected on that plane, t2 = n × t1.
    std::size_t s0 = 0;
    for (std::size_t s = 1; s < 3; ++s)
        if (std::fabs(n[s]) < std::fabs(n[s0]))
            s0 = s;
    std::array<double, 3> t1 = axis(s0);
    for (std::size_t s = 0; s < 3; ++s)
        t1[s] -= n[s0] * n[s];
    const double length = std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    for (double& value : t1)
        value /= length;
    directions.push_back(t1);
    directions.push_back({n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]});
    return directions;
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
                held.push_back({label.row, static_cast<int>(label.component), enforcedValueOf(spc, dof), false});
        }
    };
    add(spc.sgDofs());
    add(spc.sbDofs());
    // A component in both sets is held once (by sg, with sb's value: an SPC value on a permanent constraint is 0).
    std::sort(held.begin(), held.end(), isBefore);
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
    // The reactions are carried to the grid points as every force on an entity DOF is.
    const EntityForceCarry carry(dofMap);
    for (const elements::DerivedSupport& support : supports)
    {
        DerivedSupportDof dof{dofMap.dofOf(support.dof), support.value, {}, {}};
        dof.reactionTo = carry.termsOf(dof.dof);
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
                                               const RigidElimination& userRelations, base::DiagnosticSink& sink,
                                               const GlueConstraints* glue)
    : m_model(&model), m_kind(kind), m_dofMap(&dofMap), m_relations(dofMap), m_held(heldGridComponentsOf(spc, dofMap))
{
    // The DOFs the glue makes dependent, increasing: each follows its own point of a face, so no edge is carried
    // between two of them.
    std::vector<base::DofIndex> glued; // bounded by the rows of the glue
    if (glue != nullptr)
        for (std::size_t i = 0; i < glue->rowCount(); ++i)
            glued.push_back(glue->dependentDof(i));
    std::sort(glued.begin(), glued.end());
    // The grid components the formulation reads along its edges and faces: those the set holds and those a relation
    // of the model makes dependent (a rigid element, a multipoint constraint).
    m_followed = m_held;
    bool hasDependentGrids = false;
    for (const base::DofIndex dof : userRelations.mDofs())
    {
        const DofLabel label = dofMap.labelOf(dof);
        if (label.kind != input::PointKind::Grid || std::binary_search(glued.begin(), glued.end(), dof))
            continue;
        m_followed.push_back({label.row, static_cast<int>(label.component), 0.0, true});
        hasDependentGrids = true;
    }
    std::sort(m_followed.begin(), m_followed.end(), isBefore);
    // The rows of those relations (resolved on their independent DOFs), which the relations of the formulation take
    // in place of the dependent DOFs they name (addRelation).
    m_userOffsets.push_back(0);
    for (std::size_t i = 0; i < userRelations.rowCount(); ++i)
    {
        if (dofMap.labelOf(userRelations.mDofs()[i]).kind != input::PointKind::Grid)
            continue;
        m_userDofs.push_back(userRelations.mDofs()[i]);
        const std::span<const ConstraintTerm> terms = userRelations.row(i);
        m_userTerms.insert(m_userTerms.end(), terms.begin(), terms.end());
        m_userOffsets.push_back(m_userTerms.size());
    }

    // The ties (section 3.4).
    std::vector<elements::ElementDof> dependents;
    std::vector<std::size_t> offsets{0};
    std::vector<elements::FormulationRelationTerm> terms;
    model.ties(kind, m_followed, dependents, offsets, terms, sink);
    std::vector<FormulationTerm> row; // bounded by one relation
    for (std::size_t i = 0; i < dependents.size(); ++i)
    {
        row.clear();
        for (std::size_t t = offsets[i]; t < offsets[i + 1]; ++t)
            row.push_back({terms[t].dof, terms[t].coefficient});
        addRelation(m_relations, dependents[i], row);
    }
    m_tieCount = m_relations.rowCount();

    // The edges and faces that follow a relation of the model (section 5.6): the entity DOFs that the grid components
    // held or dependent hold together and the supports alone do not, each the image of its grid components by the map
    // that carries the values of a support along the edge. `isCarried`: the grid components an edge or a face reads.
    std::vector<std::uint8_t> isCarried(static_cast<std::size_t>(dofMap.gSetSize()), 0);
    if (hasDependentGrids)
    {
        base::DiagnosticSink quiet; // the notes of the supports are those of withFormulationSupports
        std::vector<elements::DerivedSupport> supports;
        model.deriveSupports(kind, m_followed, supports, quiet);
        std::set<base::DofIndex> taken;
        for (const elements::DerivedSupport& support : supports)
        {
            for (const elements::FormulationRelationTerm& term : support.carriedTo)
                isCarried[static_cast<std::size_t>(dofMap.dofOf(term.dof))] = 1;
            const base::DofIndex dof = dofMap.dofOf(support.dof);
            if (isDerived(spc, dof) || !taken.insert(dof).second)
                continue;
            row.clear();
            for (const elements::FormulationRelationTerm& term : support.carriedTo)
                row.push_back({term.dof, term.coefficient});
            addRelation(m_relations, support.dof, row);
            m_followedDofs.push_back(dof);
        }
    }
    else
    {
        for (const DerivedSupportDof& derived : spc.derivedSupports())
            for (const ConstraintTerm& term : derived.carriedTo)
                isCarried[static_cast<std::size_t>(term.dof)] = 1;
    }

    // The determined grid components (section 5.4): read along an edge, tied where no edge reads them, observed, or
    // excluded.
    std::vector<elements::HeldGridComponent> determined;
    model.determinedGridComponents(kind, determined);
    std::set<std::int32_t> loadedZero;
    std::set<std::int32_t> heldZero;
    // Per grid point, its rotation components held and dependent that no edge reads.
    struct IsolatedRotations
    {
        std::array<bool, 3> held{};
        std::array<bool, 3> dependent{};
    };
    std::map<std::int32_t, IsolatedRotations> isolated;
    std::vector<elements::FormulationRelationTerm> determinedTerms;
    for (const elements::HeldGridComponent& component : determined)
    {
        const base::DofIndex dof = dofMap.gridDof(component.gridRow, component.component);
        const bool isHeldBySet = isHeld(spc, dof);
        if (isHeldBySet || userRelations.isDependent(dof))
        {
            if (isCarried[static_cast<std::size_t>(dof)] != 0)
                continue;
            const auto s = static_cast<std::size_t>((component.component - 1) % 3);
            if (component.component >= 4)
                (isHeldBySet ? isolated[component.gridRow].held : isolated[component.gridRow].dependent)[s] = true;
            else
            {
                TiedRow tied{component.gridRow, 1, {}};
                tied.direction[s] = 1.0;
                m_tiedRows.push_back(tied);
            }
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
        addRelation(m_relations, elements::ElementDof::grid(component.gridRow, component.component), row);
    }
    m_firstTiedRow = m_relations.rowCount();

    // The rotations held or dependent at a grid point where no edge reads them (section 5.6): the mean rotation of the
    // shells there is tied to them. A dependent one in each of its components; a held one about the axes of the plane
    // of the shells only.
    // The grid points joined by an edge of shells to a grid point that holds a rotation.
    std::vector<std::uint8_t> hasHeldNeighbour;
    if (!isolated.empty())
    {
        const auto holdsRotation = [&](std::int32_t gridRow)
        {
            const base::DofIndex first = dofMap.gridDof(gridRow, 4);
            return isHeld(spc, first) || isHeld(spc, first + 1) || isHeld(spc, first + 2);
        };
        hasHeldNeighbour.assign(static_cast<std::size_t>(dofMap.gridCount()), 0);
        std::vector<std::array<std::int32_t, 2>> edges;
        model.shellEdges(edges);
        for (const std::array<std::int32_t, 2>& edge : edges)
        {
            if (holdsRotation(edge[0]))
                hasHeldNeighbour[static_cast<std::size_t>(edge[1])] = 1;
            if (holdsRotation(edge[1]))
                hasHeldNeighbour[static_cast<std::size_t>(edge[0])] = 1;
        }
    }
    for (const auto& [gridRow, rotations] : isolated)
    {
        for (std::size_t s = 0; s < 3; ++s)
            if (rotations.dependent[s])
            {
                TiedRow tied{gridRow, 4, {}};
                tied.direction[s] = 1.0;
                m_tiedRows.push_back(tied);
            }
        const bool isAlone = hasHeldNeighbour[static_cast<std::size_t>(gridRow)] == 0;
        for (const std::array<double, 3>& direction :
             heldRotationDirections(rotations.held, model.commonShellNormal(gridRow), isAlone))
            m_tiedRows.push_back({gridRow, 4, direction});
    }
    // The rows eliminated one after the other: what the supports hold along directions that are no component of an
    // edge or a face (section 5.6), then the tied rotations.
    std::set<std::int32_t> notTransmitted;
    std::vector<ConstraintRow> rows = constraintRowsOn(model);
    m_supportRowCount = rows.size() - m_tiedRows.size();
    m_rowPivots.assign(rows.size(), -1);
    if (!rows.empty())
    {
        // An unknown takes a row when nothing else determines it; a row that reaches grid components the model's
        // relations move (a tied rotation, a row of a dependent grid point) also when no relation written so far reads
        // it: an observer relation reached through those components would close a loop.
        std::vector<std::uint8_t> isFree(static_cast<std::size_t>(dofMap.gSetSize()), 1);
        std::vector<std::uint8_t> isRead(static_cast<std::size_t>(dofMap.gSetSize()), 0);
        for (const DerivedSupportDof& derived : spc.derivedSupports())
            isFree[static_cast<std::size_t>(derived.dof)] = 0;
        for (std::size_t i = 0; i < m_relations.rowCount(); ++i)
        {
            isFree[static_cast<std::size_t>(m_relations.dependentDof(i))] = 0;
            for (const ConstraintTerm& term : m_relations.row(i))
                isRead[static_cast<std::size_t>(term.dof)] = 1;
        }
        const PivotChoice choosePivot = [&](const ConstraintRow& row, std::size_t index)
        {
            double largest = 0.0;
            double best = 0.0;
            base::DofIndex pivot = -1;
            for (const auto& [dof, coefficient] : row.coefficients)
            {
                if (dofMap.labelOf(dof).kind != input::PointKind::Entity)
                    continue;
                largest = std::max(largest, std::fabs(coefficient));
                const bool isEligible = isFree[static_cast<std::size_t>(dof)] != 0 &&
                                        (!row.chains || isRead[static_cast<std::size_t>(dof)] == 0);
                if (isEligible && std::fabs(coefficient) > best)
                {
                    best = std::fabs(coefficient);
                    pivot = dof;
                }
            }
            if (pivot < 0 || best < PIVOT_RATIO * largest)
            {
                if (index < m_supportRowCount)
                    return base::DofIndex{-1};
                // The rotation of shells that a relation moves and no unknown of the shells can follow.
                const TiedRow& tied = m_tiedRows[index - m_supportRowCount];
                const base::DofIndex first = dofMap.gridDof(tied.gridRow, tied.firstComponent);
                if (largest > 0.0 && tied.firstComponent == 4 &&
                    (userRelations.isDependent(first) || userRelations.isDependent(first + 1) ||
                     userRelations.isDependent(first + 2)))
                    notTransmitted.insert(tied.gridRow);
                return base::DofIndex{-1};
            }
            isFree[static_cast<std::size_t>(pivot)] = 0;
            return pivot;
        };
        eliminateRows(rows, m_rowPivots, m_relations, &choosePivot);
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
             "dependent: the relation moves that rotation, but no unknown of the shells there is free to follow it "
             "(other relations and supports determine them all), so the shells are not tied to it; tie the relation "
             "to the translations of the shells' grid points instead"});
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

std::span<const ConstraintTerm> FormulationConstraints::userRowOf(base::DofIndex dof) const noexcept
{
    const auto found = std::lower_bound(m_userDofs.begin(), m_userDofs.end(), dof);
    if (found == m_userDofs.end() || *found != dof)
        return {};
    const auto i = static_cast<std::size_t>(found - m_userDofs.begin());
    return {m_userTerms.data() + m_userOffsets[i], m_userOffsets[i + 1] - m_userOffsets[i]};
}

bool FormulationConstraints::isUserDependent(base::DofIndex dof) const noexcept
{
    return std::binary_search(m_userDofs.begin(), m_userDofs.end(), dof);
}

void FormulationConstraints::substituteUserRows(SparseRow& coefficients) const
{
    // The rows are resolved: one pass replaces every dependent DOF.
    std::vector<std::pair<base::DofIndex, double>> dependents; // bounded by one row
    for (const auto& [dof, coefficient] : coefficients)
        if (isUserDependent(dof))
            dependents.emplace_back(dof, coefficient);
    for (const auto& [dof, coefficient] : dependents)
    {
        coefficients.erase(dof);
        for (const ConstraintTerm& term : userRowOf(dof))
            coefficients[term.dof] += coefficient * term.coefficient;
    }
}

void FormulationConstraints::addRelation(FormulationRelations& relations, const elements::ElementDof& dependent,
                                         std::span<const FormulationTerm> row) const
{
    const bool chains = std::any_of(row.begin(), row.end(), [&](const FormulationTerm& term)
                                    { return isUserDependent(m_dofMap->dofOf(term.dof)); });
    if (!chains)
    {
        relations.add(dependent, row);
        return;
    }
    SparseRow coefficients;
    for (const FormulationTerm& term : row)
        coefficients[m_dofMap->dofOf(term.dof)] += term.coefficient;
    substituteUserRows(coefficients);
    std::vector<FormulationTerm> resolved; // bounded by one relation
    resolved.reserve(coefficients.size());
    for (const auto& [dof, coefficient] : coefficients)
        resolved.push_back({elementDofOf(m_dofMap->labelOf(dof)), coefficient});
    relations.add(dependent, resolved);
}

std::vector<FormulationConstraints::ConstraintRow> FormulationConstraints::constraintRowsOn(
    const elements::FormulationModel& model) const
{
    std::vector<ConstraintRow> rows; // bounded by the edges of the held directions and the tied rotations
    // What the supports hold along directions that are no component of an edge or a face.
    std::vector<std::vector<elements::FormulationRelationTerm>> supportRows;
    model.supportRows(m_kind, m_followed, supportRows);
    for (const std::vector<elements::FormulationRelationTerm>& terms : supportRows)
    {
        ConstraintRow row;
        for (const elements::FormulationRelationTerm& term : terms)
        {
            const base::DofIndex dof = m_dofMap->dofOf(term.dof);
            row.coefficients[dof] += term.coefficient;
            row.chains = row.chains || isUserDependent(dof);
        }
        substituteUserRows(row.coefficients);
        rows.push_back(std::move(row));
    }
    // The tied rows: Σ_s d_s·(W_s·u − u_g,s) = 0 with W_s the row of the determined component s of the grid point.
    std::vector<elements::FormulationRelationTerm> terms;
    for (const TiedRow& tied : m_tiedRows)
    {
        ConstraintRow row;
        row.chains = true;
        for (int s = 0; s < 3; ++s)
        {
            const double d = tied.direction[static_cast<std::size_t>(s)];
            if (d == 0.0)
                continue;
            terms.clear();
            model.determinedRow(m_kind, tied.gridRow, tied.firstComponent + s, terms);
            for (const elements::FormulationRelationTerm& term : terms)
                row.coefficients[m_dofMap->dofOf(term.dof)] += d * term.coefficient;
            row.coefficients[m_dofMap->gridDof(tied.gridRow, tied.firstComponent + s)] -= d;
        }
        substituteUserRows(row.coefficients);
        rows.push_back(std::move(row));
    }
    return rows;
}

void FormulationConstraints::eliminateRows(std::vector<ConstraintRow>& rows, std::vector<base::DofIndex>& pivots,
                                           FormulationRelations& relations, const PivotChoice* choosePivot) const
{
    // Each row Σ_j a_j·u_j = 0 is written on one unknown p: u_p = −Σ_{j≠p} (a_j / a_p)·u_j. The unknowns of the earlier
    // rows are replaced in it first (an elimination row by row), so that no row names the unknown of another before
    // it. A row without an unknown (pivot −1) is left out.
    std::vector<std::pair<base::DofIndex, SparseRow>> eliminated; // bounded by the rows
    std::vector<FormulationTerm> relation;
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        SparseRow& coefficients = rows[index].coefficients;
        for (const auto& [pivot, substitute] : eliminated)
        {
            const auto found = coefficients.find(pivot);
            if (found == coefficients.end())
                continue;
            const double factor = found->second;
            coefficients.erase(found);
            for (const auto& [dof, coefficient] : substitute)
                coefficients[dof] += factor * coefficient;
        }
        if (choosePivot != nullptr)
            pivots[index] = (*choosePivot)(rows[index], index);
        if (pivots[index] < 0)
            continue;
        const auto own = coefficients.find(pivots[index]);
        if (own == coefficients.end() || own->second == 0.0)
            throw std::logic_error("FormulationConstraints::eliminateRows: a row without its unknown (another "
                                   "layout)");
        SparseRow substitute;
        relation.clear();
        for (const auto& [dof, coefficient] : coefficients)
        {
            if (dof == pivots[index] || coefficient == 0.0)
                continue;
            substitute[dof] = -coefficient / own->second;
            relation.push_back({elementDofOf(m_dofMap->labelOf(dof)), -coefficient / own->second});
        }
        relations.add(elementDofOf(m_dofMap->labelOf(pivots[index])), relation);
        eliminated.emplace_back(pivots[index], std::move(substitute));
    }
}

void FormulationConstraints::fillExcluded(std::span<double> displacementsG) const
{
    fillExcludedOn(*m_model, displacementsG);
}

FormulationRelations FormulationConstraints::relationsOn(const elements::FormulationModel& model,
                                                         base::DiagnosticSink& sink) const
{
    FormulationRelations relations(*m_dofMap);
    std::vector<elements::ElementDof> dependents;
    std::vector<std::size_t> offsets{0};
    std::vector<elements::FormulationRelationTerm> terms;
    model.ties(m_kind, m_followed, dependents, offsets, terms, sink);
    if (dependents.size() != m_tieCount)
        throw std::logic_error(base::concatenate({"FormulationConstraints::relationsOn: expected ",
                                                  std::to_string(m_tieCount), " ties, found ",
                                                  std::to_string(dependents.size()), " (another layout)"}));
    std::vector<FormulationTerm> row; // bounded by one relation
    for (std::size_t i = 0; i < dependents.size(); ++i)
    {
        if (m_dofMap->dofOf(dependents[i]) != m_relations.dependentDof(i))
            throw std::logic_error("FormulationConstraints::relationsOn: a tie of another dependent DOF (another "
                                   "layout)");
        row.clear();
        for (std::size_t t = offsets[i]; t < offsets[i + 1]; ++t)
            row.push_back({terms[t].dof, terms[t].coefficient});
        addRelation(relations, dependents[i], row);
    }
    // The edges and faces that follow a relation of the model, in this set's order.
    if (!m_followedDofs.empty())
    {
        base::DiagnosticSink quiet;
        std::vector<elements::DerivedSupport> supports;
        model.deriveSupports(m_kind, m_followed, supports, quiet);
        std::map<base::DofIndex, const elements::DerivedSupport*> byDof;
        for (const elements::DerivedSupport& support : supports)
            byDof.emplace(m_dofMap->dofOf(support.dof), &support);
        for (const base::DofIndex dof : m_followedDofs)
        {
            const auto found = byDof.find(dof);
            if (found == byDof.end())
                throw std::logic_error("FormulationConstraints::relationsOn: an edge that follows a relation here and "
                                       "not on the other model (another layout)");
            row.clear();
            for (const elements::FormulationRelationTerm& term : found->second->carriedTo)
                row.push_back({term.dof, term.coefficient});
            addRelation(relations, found->second->dof, row);
        }
    }
    // The observer relations.
    std::vector<elements::FormulationRelationTerm> determinedTerms;
    for (std::size_t i = m_tieCount + m_followedDofs.size(); i < m_firstTiedRow; ++i)
    {
        const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(m_relations.dependentDof(i));
        determinedTerms.clear();
        model.determinedRow(m_kind, nodeDof.node, nodeDof.component + 1, determinedTerms);
        row.clear();
        for (const elements::FormulationRelationTerm& term : determinedTerms)
            row.push_back({term.dof, term.coefficient});
        addRelation(relations, elements::ElementDof::grid(nodeDof.node, nodeDof.component + 1), row);
    }
    // The rows of the supports' directions and the tied rows, on this set's unknowns.
    std::vector<ConstraintRow> rows = constraintRowsOn(model);
    if (rows.size() != m_rowPivots.size())
        throw std::logic_error("FormulationConstraints::relationsOn: other rows to eliminate (another layout)");
    std::vector<base::DofIndex> pivots = m_rowPivots;
    eliminateRows(rows, pivots, relations, nullptr);
    return relations;
}

void FormulationConstraints::fillExcludedOn(const elements::FormulationModel& model,
                                            std::span<double> displacementsG) const
{
    if (m_excluded.empty())
        return;
    const elements::NodalDisplacements displacements =
        elements::NodalDisplacements::ofGSet(displacementsG, static_cast<std::size_t>(m_dofMap->gridCount()),
                                             static_cast<std::size_t>(m_dofMap->scalarCount()), &m_dofMap->entities());
    // The values are gathered first: the formulation reads the vector while they are computed.
    std::vector<std::pair<base::DofIndex, double>> values; // bounded by the determined components
    model.fillDeterminedComponents(m_kind, displacements,
                                      [&](std::int32_t gridRow, int component, double value)
                                      {
                                          const base::DofIndex dof = m_dofMap->gridDof(gridRow, component);
                                          if (std::binary_search(m_excluded.begin(), m_excluded.end(), dof))
                                              values.emplace_back(dof, value);
                                      });
    for (const auto& [dof, value] : values)
        displacementsG[static_cast<std::size_t>(dof)] = value;
}

void FormulationConstraints::moveExcludedWeights(std::span<double> weightsG) const
{
    std::vector<elements::FormulationRelationTerm> terms; // bounded by one row
    for (const base::DofIndex dof : m_excluded)
    {
        const double weight = weightsG[static_cast<std::size_t>(dof)];
        if (weight == 0.0)
            continue;
        weightsG[static_cast<std::size_t>(dof)] = 0.0;
        const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(dof);
        terms.clear();
        m_model->determinedRow(m_kind, nodeDof.node, nodeDof.component + 1, terms);
        for (const elements::FormulationRelationTerm& term : terms)
            weightsG[static_cast<std::size_t>(m_dofMap->dofOf(term.dof))] += weight * term.coefficient;
    }
}

} // namespace oos::assembly
