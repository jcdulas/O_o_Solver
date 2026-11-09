#include "oos/solutions/sol101/BasisFixedRows.hpp"

#include "oos/base/math/Vec3.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace oos::solutions
{

namespace
{

// The DofMap node of a point ID, or −1 when the model has no such point.
std::int32_t nodeOfPoint(const assembly::DofMap& dofMap, base::EntityId id)
{
    const std::optional<input::PointRef> point = dofMap.points().find(id);
    if (!point)
        return -1;
    return point->kind == input::PointKind::Scalar ? dofMap.gridCount() + point->row : point->row;
}

// The g-set DOFs of a node (six for a grid point, one for a scalar point), into `dofs`; their count.
int dofsOfNode(const assembly::DofMap& dofMap, std::int32_t node, std::array<base::DofIndex, 6>& dofs) noexcept
{
    if (node < dofMap.gridCount())
    {
        for (int c = 0; c < 6; ++c)
            dofs[static_cast<std::size_t>(c)] = dofMap.gridDof(node, c + 1);
        return 6;
    }
    dofs[0] = dofMap.scalarDof(node - dofMap.gridCount());
    return 1;
}

// The side of a grid point to a plane (point, normal): behind it (the dot product of its offset with the normal below
// zero: O_o_Reader's side of a plane cut), or not; a scalar point has none.
enum class Side : std::uint8_t
{
    None,
    Behind,
    Ahead
};

Side sideOf(const assembly::DofMap& dofMap, std::int32_t node, const std::array<double, 6>& plane)
{
    if (node >= dofMap.gridCount())
        return Side::None;
    const base::Vec3 x = dofMap.positionInBasic(node);
    const double along = (x[0] - plane[0]) * plane[3] + (x[1] - plane[1]) * plane[4] + (x[2] - plane[2]) * plane[5];
    return along < 0.0 ? Side::Behind : Side::Ahead;
}

// Adds to `taken` (increasing rows of Gm) the rows chained to them, transitively: the rows their own terms make
// dependent (children) and the rows whose own terms are on their m DOFs (parents).
void closeOverChains(const assembly::RigidElimination& gm, std::vector<std::size_t>& taken)
{
    std::vector<std::vector<std::size_t>> parentsOf(gm.rowCount()); // bounded by the rows of Gm
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
        for (const assembly::ConstraintTerm& term : gm.relationTerms(i))
            if (const std::optional<std::size_t> child = gm.findRow(term.dof))
                parentsOf[*child].push_back(i);
    std::vector<std::uint8_t> isTaken(gm.rowCount(), 0);
    std::vector<std::size_t> pending(taken.begin(), taken.end());
    for (const std::size_t i : taken)
        isTaken[i] = 1;
    while (!pending.empty())
    {
        const std::size_t i = pending.back();
        pending.pop_back();
        const auto reach = [&](std::size_t other)
        {
            if (isTaken[other] != 0)
                return;
            isTaken[other] = 1;
            pending.push_back(other);
        };
        for (const assembly::ConstraintTerm& term : gm.relationTerms(i))
            if (const std::optional<std::size_t> child = gm.findRow(term.dof))
                reach(*child);
        for (const std::size_t parent : parentsOf[i])
            reach(parent);
    }
    taken.clear();
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
        if (isTaken[i] != 0)
            taken.push_back(i);
}

// The relations of the kept rows, with chains their own terms and the kept rows of the chains, parents first.
void keepRelations(const assembly::RigidElimination& gm, std::span<const std::size_t> taken, base::MemoryBudget& budget,
                   KeptFixedRows& rows)
{
    rows.isMpcSplit = true;
    rows.relations = base::BudgetedBuffer<std::int32_t>(budget, taken.size(), "fixed rows relation IDs");
    for (std::size_t r = 0; r < taken.size(); ++r)
        rows.relations[r] = gm.relationOf(taken[r]);
    if (!gm.hasChains())
        return;
    std::size_t termCount = 0;
    for (const std::size_t i : taken)
        termCount += gm.relationTerms(i).size();
    rows.ownOffsets = base::BudgetedBuffer<std::int64_t>(budget, taken.size() + 1, "fixed rows own terms");
    rows.ownTermDofs = base::BudgetedBuffer<base::DofIndex>(budget, termCount, "fixed rows own terms");
    rows.ownTermCoefficients = base::BudgetedBuffer<double>(budget, termCount, "fixed rows own terms");
    std::size_t next = 0;
    for (std::size_t r = 0; r < taken.size(); ++r)
    {
        rows.ownOffsets[r] = static_cast<std::int64_t>(next);
        for (const assembly::ConstraintTerm& term : gm.relationTerms(taken[r]))
        {
            rows.ownTermDofs[next] = term.dof;
            rows.ownTermCoefficients[next] = term.coefficient;
            ++next;
        }
    }
    rows.ownOffsets[taken.size()] = static_cast<std::int64_t>(next);
    std::vector<std::int64_t> order; // bounded by the kept rows
    for (const std::int64_t i : gm.chainOrder())
        if (const auto found = std::lower_bound(taken.begin(), taken.end(), static_cast<std::size_t>(i));
            found != taken.end() && *found == static_cast<std::size_t>(i))
            order.push_back(found - taken.begin());
    rows.chainOrder = base::BudgetedBuffer<std::int64_t>(budget, order.size(), "fixed rows chain order");
    std::copy(order.begin(), order.end(), rows.chainOrder.data());
}

// The kept row whose m DOF is `dof` (mDofs increasing), or nothing.
std::optional<std::size_t> keptRowOf(const KeptFixedRows& rows, base::DofIndex dof)
{
    const base::DofIndex* end = rows.mDofs.data() + rows.mDofs.size();
    const base::DofIndex* found = std::lower_bound(rows.mDofs.data(), end, dof);
    if (found == end || *found != dof)
        return std::nullopt;
    return static_cast<std::size_t>(found - rows.mDofs.data());
}

// The multiPointConstrForce rows of the nodes of `made` split by relation (RigidElimination::forEachRelationForce on
// the kept rows): λ_r = q_m(r) the kept MPC force at the row's m DOF, through the chains λ_child += c·λ_parent, then
// each row's force λ_r·(e_m − Σ c·e_t) on its own DOFs, summed per node and relation in the rows' order.
void splitByRelation(const KeptFixedRows& rows, const assembly::DofMap& dofMap,
                     std::vector<output::StiffnessRecovery::PointFixedRows>& made, std::span<const double> mpc)
{
    const std::size_t rowCount = rows.mDofs.size();
    std::vector<double> multipliers(rowCount, 0.0); // bounded by the kept rows
    for (std::size_t r = 0; r < rowCount; ++r)
        if (const std::optional<std::size_t> place = rows.placeOf(rows.mDofs[r]))
            multipliers[r] = mpc[*place];
    const bool hasOwn = rows.ownOffsets.size() != 0;
    const auto termsOf = [&](std::size_t r)
    {
        const base::BudgetedBuffer<std::int64_t>& offsets = hasOwn ? rows.ownOffsets : rows.offsets;
        return std::make_pair(static_cast<std::size_t>(offsets[r]), static_cast<std::size_t>(offsets[r + 1]));
    };
    const auto termDof = [&](std::size_t t) { return hasOwn ? rows.ownTermDofs[t] : rows.termDofs[t]; };
    const auto termCoefficient = [&](std::size_t t)
    { return hasOwn ? rows.ownTermCoefficients[t] : rows.termCoefficients[t]; };
    for (std::size_t k = 0; k < rows.chainOrder.size(); ++k)
    {
        const auto parent = static_cast<std::size_t>(rows.chainOrder[k]);
        const auto [first, end] = termsOf(parent);
        for (std::size_t t = first; t < end; ++t)
            if (const std::optional<std::size_t> child = keptRowOf(rows, termDof(t)))
                multipliers[*child] += termCoefficient(t) * multipliers[parent];
    }
    for (output::StiffnessRecovery::PointFixedRows& fixed : made)
        fixed.isMpcSplit = true;
    const auto add = [&](std::size_t r, base::DofIndex dof, double force)
    {
        const linalg::NodeDof nodeDof = dofMap.nodeDofOf(dof);
        const auto found = std::lower_bound(made.begin(), made.end(), nodeDof.node,
                                            [](const output::StiffnessRecovery::PointFixedRows& entry,
                                               std::int32_t node) { return entry.node < node; });
        if (found == made.end() || found->node != nodeDof.node)
            return;
        std::vector<output::StiffnessRecovery::RelationRow>& relations = found->mpcByRelation;
        const std::int32_t relation = rows.relations[r];
        auto place = std::lower_bound(relations.begin(), relations.end(), relation,
                                      [](const output::StiffnessRecovery::RelationRow& row, std::int32_t id)
                                      { return row.relation < id; });
        if (place == relations.end() || place->relation != relation)
            place = relations.insert(place, output::StiffnessRecovery::RelationRow{relation, {}});
        place->values[static_cast<std::size_t>(nodeDof.component)] += force;
    };
    for (std::size_t r = 0; r < rowCount; ++r)
    {
        add(r, rows.mDofs[r], multipliers[r]);
        const auto [first, end] = termsOf(r);
        for (std::size_t t = first; t < end; ++t)
            add(r, termDof(t), -termCoefficient(t) * multipliers[r]);
    }
}

} // namespace

std::uint64_t BasisFixedRows::estimateBytes(const BasisFixedRowRequest& request, std::uint64_t gSize,
                                            std::uint64_t units) noexcept
{
    if (!request.isRequested)
        return 0;
    constexpr std::uint64_t DOFS_PER_POINT = 6;
    const std::uint64_t dofs = request.isAll ? gSize : DOFS_PER_POINT * request.points.size();
    return dofs * (units * sizeof(double) + sizeof(base::DofIndex) + 1);
}

std::shared_ptr<KeptFixedRows> BasisFixedRows::plan(const BasisFixedRowRequest& request, const assembly::DofMap& dofMap,
                                                    const assembly::RigidElimination& gm,
                                                    std::span<const std::uint8_t> isConstrained, std::size_t units,
                                                    base::MemoryBudget& budget)
{
    if (!request.isRequested)
        return nullptr;
    auto rows = std::make_shared<KeptFixedRows>();
    rows->isAll = request.isAll;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const std::size_t nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    // The rows of Gm taken: every one, or those that reach a point asked or straddle a plane.
    std::vector<std::size_t> taken; // bounded by the rows of Gm
    base::BudgetedBuffer<std::uint8_t> isPoint(budget, request.isAll ? 0 : nodeCount, "fixed rows points");
    if (request.isAll)
    {
        for (std::size_t i = 0; i < gm.rowCount(); ++i)
            taken.push_back(i);
    }
    else
    {
        for (const base::EntityId id : request.points)
            if (const std::int32_t node = nodeOfPoint(dofMap, id); node >= 0)
                isPoint[static_cast<std::size_t>(node)] = 1;
        std::vector<std::int32_t> rowNodes; // bounded by a row of Gm
        for (std::size_t i = 0; i < gm.rowCount(); ++i)
        {
            rowNodes.clear();
            rowNodes.push_back(dofMap.nodeDofOf(gm.mDofs()[i]).node);
            for (const assembly::ConstraintTerm& term : gm.row(i))
                rowNodes.push_back(dofMap.nodeDofOf(term.dof).node);
            bool isTaken = std::any_of(rowNodes.begin(), rowNodes.end(),
                                       [&](std::int32_t node) { return isPoint[static_cast<std::size_t>(node)] != 0; });
            for (std::size_t p = 0; p < request.planes.size() && !isTaken; ++p)
            {
                bool isBehind = false;
                bool isAhead = false;
                for (const std::int32_t node : rowNodes)
                {
                    const Side side = sideOf(dofMap, node, request.planes[p]);
                    isBehind = isBehind || side == Side::Behind;
                    isAhead = isAhead || side == Side::Ahead;
                }
                isTaken = isBehind && isAhead;
            }
            if (isTaken)
                taken.push_back(i);
        }
        // Split rows through a chain need every row of the chain (the multipliers of the parents reach the children).
        if (request.isMpcSplit && gm.hasChains())
            closeOverChains(gm, taken);
        // Every point of the rows taken has its rows given (the multiPointConstrForce rows of a relation's points on a
        // side).
        for (const std::size_t i : taken)
        {
            isPoint[static_cast<std::size_t>(dofMap.nodeDofOf(gm.mDofs()[i]).node)] = 1;
            for (const assembly::ConstraintTerm& term : gm.row(i))
                isPoint[static_cast<std::size_t>(dofMap.nodeDofOf(term.dof).node)] = 1;
            if (request.isMpcSplit)
                for (const assembly::ConstraintTerm& term : gm.relationTerms(i))
                    isPoint[static_cast<std::size_t>(dofMap.nodeDofOf(term.dof).node)] = 1;
        }
        std::size_t pointCount = 0;
        std::size_t dofCount = 0;
        std::array<base::DofIndex, 6> dofs{};
        for (std::size_t node = 0; node < nodeCount; ++node)
            if (isPoint[node] != 0)
            {
                ++pointCount;
                dofCount += static_cast<std::size_t>(dofsOfNode(dofMap, static_cast<std::int32_t>(node), dofs));
            }
        rows->nodes = base::BudgetedBuffer<std::int32_t>(budget, pointCount, "fixed rows nodes");
        rows->dofs = base::BudgetedBuffer<base::DofIndex>(budget, dofCount, "fixed rows DOFs");
        std::size_t nextNode = 0;
        std::size_t nextDof = 0;
        for (std::size_t node = 0; node < nodeCount; ++node)
        {
            if (isPoint[node] == 0)
                continue;
            rows->nodes[nextNode++] = static_cast<std::int32_t>(node);
            const int count = dofsOfNode(dofMap, static_cast<std::int32_t>(node), dofs);
            for (int c = 0; c < count; ++c)
                rows->dofs[nextDof++] = dofs[static_cast<std::size_t>(c)];
        }
        // A grid point's DOFs come before the scalar points': increasing either way (DofMap's numbering).
    }
    const std::size_t kept = request.isAll ? gSize : rows->dofs.size();
    rows->isHeld = base::BudgetedBuffer<std::uint8_t>(budget, kept, "fixed rows held DOFs");
    for (std::size_t i = 0; i < kept; ++i)
    {
        const auto dof = static_cast<std::size_t>(request.isAll ? static_cast<base::DofIndex>(i) : rows->dofs[i]);
        rows->isHeld[i] = dof < isConstrained.size() ? isConstrained[dof] : 0;
    }
    rows->loads = linalg::DenseMatrix(budget, kept, units, "fixed rows unit loads");
    // The rows taken, in Gm's order (that of computeMpcForces' sums).
    std::size_t termCount = 0;
    for (const std::size_t i : taken)
        termCount += gm.row(i).size();
    rows->mDofs = base::BudgetedBuffer<base::DofIndex>(budget, taken.size(), "fixed rows relations");
    rows->offsets = base::BudgetedBuffer<std::int64_t>(budget, taken.size() + 1, "fixed rows relation offsets");
    rows->termDofs = base::BudgetedBuffer<base::DofIndex>(budget, termCount, "fixed rows relation terms");
    rows->termCoefficients = base::BudgetedBuffer<double>(budget, termCount, "fixed rows relation coefficients");
    std::size_t nextTerm = 0;
    std::vector<std::int32_t> mNodes; // bounded by the rows taken
    for (std::size_t r = 0; r < taken.size(); ++r)
    {
        const std::size_t i = taken[r];
        rows->mDofs[r] = gm.mDofs()[i];
        rows->offsets[r] = static_cast<std::int64_t>(nextTerm);
        for (const assembly::ConstraintTerm& term : gm.row(i))
        {
            rows->termDofs[nextTerm] = term.dof;
            rows->termCoefficients[nextTerm] = term.coefficient;
            ++nextTerm;
        }
        mNodes.push_back(dofMap.nodeDofOf(gm.mDofs()[i]).node);
    }
    rows->offsets[taken.size()] = static_cast<std::int64_t>(nextTerm);
    if (request.isMpcSplit)
        keepRelations(gm, taken, budget, *rows);
    std::sort(mNodes.begin(), mNodes.end());
    mNodes.erase(std::unique(mNodes.begin(), mNodes.end()), mNodes.end());
    rows->mNodes = base::BudgetedBuffer<std::int32_t>(budget, mNodes.size(), "fixed rows dependent nodes");
    std::copy(mNodes.begin(), mNodes.end(), rows->mNodes.data());
    return rows;
}

void BasisFixedRows::take(KeptFixedRows& rows, std::size_t unit, std::span<const double> loadsG) noexcept
{
    const std::size_t kept = rows.loads.rows();
    for (std::size_t i = 0; i < kept; ++i)
        rows.loads(i, unit) = loadsG[rows.isAll ? i : static_cast<std::size_t>(rows.dofs[i])];
}

std::vector<output::StiffnessRecovery::PointFixedRows> BasisFixedRows::rowsOf(
    const KeptFixedRows& rows, std::size_t unit, std::span<const std::int32_t> rowNodes,
    const output::StiffnessRecovery::PointForces& forces, std::size_t column, std::span<const double> addedLoadsG,
    const assembly::DofMap& dofMap, base::MemoryBudget& budget, Work& work)
{
    const std::size_t kept = rows.loads.rows();
    if (work.stiffness.size() != kept)
    {
        work.stiffness = base::BudgetedBuffer<double>(budget, kept, "fixed rows K u");
        work.loads = base::BudgetedBuffer<double>(budget, kept, "fixed rows P");
        work.mpc = base::BudgetedBuffer<double>(budget, kept, "fixed rows MPC forces");
    }
    std::fill(work.stiffness.data(), work.stiffness.data() + kept, 0.0);
    std::fill(work.mpc.data(), work.mpc.data() + kept, 0.0);
    for (std::size_t i = 0; i < kept; ++i)
    {
        const auto dof = rows.isAll ? i : static_cast<std::size_t>(rows.dofs[i]);
        work.loads[i] = rows.loads(i, unit) + (addedLoadsG.empty() ? 0.0 : addedLoadsG[dof]);
    }
    // K·u at the kept DOFs: the contributions are −K_e·u_e, summed in their order (the element order at a node).
    std::array<base::DofIndex, 6> dofs{};
    for (const output::StiffnessRecovery::PointContribution& contribution : forces.columns[column])
    {
        const int count = dofsOfNode(dofMap, contribution.node, dofs);
        for (int c = 0; c < count; ++c)
            if (const std::optional<std::size_t> place = rows.placeOf(dofs[static_cast<std::size_t>(c)]))
                work.stiffness[*place] -= contribution.values[static_cast<std::size_t>(c)];
    }
    // The MPC forces (RigidElimination::computeMpcForces, its rows in its order).
    for (std::size_t r = 0; r < rows.mDofs.size(); ++r)
    {
        const std::optional<std::size_t> dependent = rows.placeOf(rows.mDofs[r]);
        if (!dependent)
            continue;
        const double force = work.stiffness[*dependent] - work.loads[*dependent];
        work.mpc[*dependent] = force;
        for (auto t = static_cast<std::size_t>(rows.offsets[r]); t < static_cast<std::size_t>(rows.offsets[r + 1]); ++t)
            if (const std::optional<std::size_t> place = rows.placeOf(rows.termDofs[t]))
                work.mpc[*place] -= rows.termCoefficients[t] * force;
    }
    // The rows of the nodes asked that keep them.
    std::vector<output::StiffnessRecovery::PointFixedRows> made; // bounded by the nodes asked
    for (const std::int32_t node : rowNodes)
    {
        if (!rows.isAll && !std::binary_search(rows.nodes.data(), rows.nodes.data() + rows.nodes.size(), node))
            continue;
        output::StiffnessRecovery::PointFixedRows fixed;
        fixed.node = node;
        const int count = dofsOfNode(dofMap, node, dofs);
        for (int c = 0; c < count; ++c)
        {
            const std::optional<std::size_t> place = rows.placeOf(dofs[static_cast<std::size_t>(c)]);
            if (!place)
                continue;
            const auto at = static_cast<std::size_t>(c);
            fixed.applied[at] = work.loads[*place];
            fixed.mpc[at] = work.mpc[*place];
            if (rows.isHeld[*place] != 0)
            {
                fixed.isHeld = true;
                fixed.spc[at] = work.stiffness[*place] - work.loads[*place] - work.mpc[*place];
            }
        }
        made.push_back(fixed);
    }
    if (rows.isMpcSplit)
        splitByRelation(rows, dofMap, made, work.mpc.span());
    return made;
}

} // namespace oos::solutions
