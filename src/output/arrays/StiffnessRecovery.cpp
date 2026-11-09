#include "oos/output/arrays/StiffnessRecovery.hpp"

#include "oos/assembly/assemble/ElementVolumes.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace oos::output
{

namespace
{

// The fixed rows of a grid point in the table, in this order, before the element rows.
constexpr std::int32_t APPLIED_LOAD_NAME = 0;
constexpr std::int32_t SPC_FORCE_NAME = 1;
constexpr std::int32_t MPC_FORCE_NAME = 2;
constexpr std::int32_t TOTAL_NAME = 3;
// The element ID of the subtotal row of a strain energy table, above any valid element ID.
constexpr std::int32_t SUBTOTAL_ELEMENT_ID = 100000000;

// The multiPointConstrForce rows split by relation of the nodes `isNodeTaken` flags: per node, by increasing relation,
// its force (RigidElimination::forEachRelationForce, summed in its order). Sized by the relations' points (budgeted).
struct SplitMpcRows
{
    std::vector<std::int32_t> nodes;     // increasing, a node once per relation at it
    std::vector<std::int32_t> relations; // per row
    std::vector<std::array<double, 6>> values;
    base::MemoryReservation charge;
};

SplitMpcRows splitMpcRowsOf(std::span<const assembly::RelationForce> forces, const assembly::DofMap& dofMap,
                            std::span<const std::uint8_t> isNodeTaken, base::MemoryBudget& budget)
{
    SplitMpcRows split;
    std::vector<std::pair<std::int32_t, std::int32_t>> keys; // (node, relation), bounded by the forces
    split.charge = budget.reserve(forces.size() * (sizeof(std::pair<std::int32_t, std::int32_t>) + 7 * sizeof(double)),
                                  "split MPC force rows");
    keys.reserve(forces.size());
    for (const assembly::RelationForce& force : forces)
    {
        const std::int32_t node = dofMap.nodeDofOf(force.dof).node;
        if (isNodeTaken[static_cast<std::size_t>(node)] != 0)
            keys.emplace_back(node, force.relation);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    split.nodes.reserve(keys.size());
    split.relations.reserve(keys.size());
    for (const auto& [node, relation] : keys)
    {
        split.nodes.push_back(node);
        split.relations.push_back(relation);
    }
    split.values.assign(keys.size(), std::array<double, 6>{});
    for (const assembly::RelationForce& force : forces)
    {
        const linalg::NodeDof nodeDof = dofMap.nodeDofOf(force.dof);
        if (isNodeTaken[static_cast<std::size_t>(nodeDof.node)] == 0)
            continue;
        const auto found = std::lower_bound(keys.begin(), keys.end(), std::make_pair(nodeDof.node, force.relation));
        split.values[static_cast<std::size_t>(found - keys.begin())][static_cast<std::size_t>(nodeDof.component)] +=
            force.force;
    }
    return split;
}

bool isNonzeroRow(const std::array<double, 6>& values)
{
    return std::any_of(values.begin(), values.end(), [](double value) { return value != 0.0; });
}

// One element's force on one grid point.
struct ElementContribution
{
    std::int32_t pointRow = 0; // node index of DofMap: grid rows, then scalar points
    std::int32_t elementId = 0;
    std::int32_t name = 0;
    std::array<double, 6> values{};
};

void requireGSize(std::span<const double> values, std::size_t size, const char* what)
{
    if (!values.empty() && values.size() != size)
        throw std::invalid_argument(base::concatenate({"StiffnessRecovery: expected ", std::to_string(size), " ", what,
                                                       " (the g set), found ", std::to_string(values.size())}));
}

// The element matrices of one range and the forces f = K_e·u_e of its elements (minus their thermal loads when
// `temperatures` is given), in row order: one element's forces at forces[offsets[i], offsets[i + 1]).
struct RangeForces
{
    elements::ElementMatrixBlock block;
    std::vector<double> forces;       // bounded by the range
    std::vector<std::size_t> offsets; // one per element of the block, and the end
};

void computeRangeForces(const elements::ElementFamily& family, const elements::ElementBatchView& view,
                        elements::ElementRange range, const assembly::DofMap& dofMap,
                        std::span<const double> displacements, const elements::ElementTemperatures* temperatures,
                        RangeForces& out)
{
    out.block.clear();
    family.computeStiffness(view, range, out.block);
    elements::ElementVectorBlock thermal;
    if (temperatures != nullptr && !family.isThermallyInert())
        family.computeThermalLoads(view, range, *temperatures, thermal);
    out.forces.clear();
    out.offsets.assign(1, 0);
    std::size_t nextThermal = 0;
    for (std::size_t i = 0; i < out.block.size(); ++i)
    {
        const std::span<const elements::ElementDof> dofs = out.block.dofs(i);
        const std::span<const double> k = out.block.matrix(i);
        const std::size_t n = dofs.size();
        const std::size_t at = out.forces.size();
        out.forces.resize(at + n, 0.0);
        const std::span<double> force(out.forces.data() + at, n);
        for (std::size_t r = 0; r < n; ++r)
        {
            double sum = 0.0;
            for (std::size_t c = 0; c < n; ++c)
                sum += k[r * n + c] * displacements[static_cast<std::size_t>(dofMap.dofOf(dofs[c]))];
            force[r] = sum;
        }
        while (nextThermal < thermal.size() && thermal.elementRow(nextThermal) < out.block.elementRow(i))
            ++nextThermal;
        while (nextThermal < thermal.size() && thermal.elementRow(nextThermal) == out.block.elementRow(i))
        {
            const std::span<const elements::ElementDof> thermalDofs = thermal.dofs(nextThermal);
            const std::span<const double> values = thermal.values(nextThermal);
            for (std::size_t t = 0; t < thermalDofs.size(); ++t)
                for (std::size_t r = 0; r < n; ++r)
                    if (dofs[r] == thermalDofs[t])
                    {
                        force[r] -= values[t];
                        break;
                    }
            ++nextThermal;
        }
        out.offsets.push_back(out.forces.size());
    }
}

// Calls visit(batchIndex, row, dofs, f) for each element with a stiffness, f = K_e·u_e (minus its thermal load when
// `temperatures` is given), in batch and row order.
// PERFORMANCE: the element matrices and their forces computed a wave of ranges at a time on the pool (each element's
// forces depend on its own matrix only), then visited in order: the grid point forces and strain energies of a deck
// recomputed every element's stiffness on one thread.
// Measured: on a mid-size static deck and a wing box deck.
template <typename Visit>
void forEachElementForce(const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                         std::span<const double> displacements, const elements::ElementTemperatures* temperatures,
                         const base::CancellationToken* cancellation, base::ThreadPool* pool, Visit&& visit)
{
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    std::vector<RangeForces> wave(threads); // bounded by the threads
    for (std::size_t b = 0; b < set.batches().size(); ++b)
    {
        const assembly::AssemblyBatch& batch = set.batches()[b];
        const elements::ElementFamily& family = *batch.family;
        if (!family.facets().has(elements::Facet::Stiffness))
            continue;
        const std::int64_t rows = batch.view.rowCount();
        std::int64_t length = StiffnessRecovery::RANGE_ELEMENTS;
        if (threads > 1)
        {
            const auto tasks = static_cast<std::int64_t>(threads) * StiffnessRecovery::RANGES_PER_THREAD;
            length = std::min(length, std::max(StiffnessRecovery::MIN_RANGE_ELEMENTS, (rows + tasks - 1) / tasks));
        }
        for (std::int64_t first = 0; first < rows; first += length * static_cast<std::int64_t>(threads))
        {
            if (cancellation != nullptr)
                cancellation->throwIfCancelled();
            const std::size_t ranges = static_cast<std::size_t>(
                std::min<std::int64_t>(static_cast<std::int64_t>(threads), (rows - first + length - 1) / length));
            const auto compute = [&](std::size_t r)
            {
                const std::int64_t begin = first + static_cast<std::int64_t>(r) * length;
                computeRangeForces(family, batch.view, {begin, std::min(rows, begin + length)}, dofMap, displacements,
                                   temperatures, wave[r]);
            };
            if (pool != nullptr && ranges > 1)
                pool->runBatch(ranges, compute);
            else
                for (std::size_t r = 0; r < ranges; ++r)
                    compute(r);
            for (std::size_t r = 0; r < ranges; ++r)
            {
                const RangeForces& range = wave[r];
                for (std::size_t i = 0; i < range.block.size(); ++i)
                    visit(b, range.block.elementRow(i), range.block.dofs(i),
                          std::span<const double>(range.forces.data() + range.offsets[i],
                                                  range.offsets[i + 1] - range.offsets[i]));
            }
        }
    }
}

// The element matrices of one range and, for each element, its strain energies under every column of a block of
// vectors (element-major: energies[i·columns + j]).
struct RangeEnergies
{
    elements::ElementMatrixBlock block;
    std::vector<double> energies;  // bounded by the range × the columns of a group
    std::vector<double> force;     // one element's K_e·u_e
    std::vector<std::size_t> dofs; // one element's g-set DOFs
};

void computeRangeEnergies(const elements::ElementFamily& family, const elements::ElementBatchView& view,
                          elements::ElementRange range, const assembly::DofMap& dofMap,
                          linalg::ConstDenseMatrixView vectors, RangeEnergies& out)
{
    out.block.clear();
    family.computeStiffness(view, range, out.block);
    const std::size_t columns = vectors.columns;
    out.energies.assign(out.block.size() * columns, 0.0);
    for (std::size_t i = 0; i < out.block.size(); ++i)
    {
        const std::span<const elements::ElementDof> dofs = out.block.dofs(i);
        const std::span<const double> k = out.block.matrix(i);
        const std::size_t n = dofs.size();
        out.dofs.resize(n);
        for (std::size_t c = 0; c < n; ++c)
            out.dofs[c] = static_cast<std::size_t>(dofMap.dofOf(dofs[c]));
        out.force.resize(n);
        for (std::size_t j = 0; j < columns; ++j)
        {
            const double* u = vectors.pointer(0, j);
            // The chains of computeRangeForces and strainEnergy(): f_r = Σ_c K_rc·u_c, E = ½·Σ_k u_k·f_k.
            for (std::size_t r = 0; r < n; ++r)
            {
                double sum = 0.0;
                for (std::size_t c = 0; c < n; ++c)
                    sum += k[r * n + c] * u[out.dofs[c]];
                out.force[r] = sum;
            }
            double energy = 0.0;
            for (std::size_t r = 0; r < n; ++r)
                energy += u[out.dofs[r]] * out.force[r];
            out.energies[i * columns + j] = 0.5 * energy;
        }
    }
}

// Calls visit(batchIndex, row, energies) for each element with a stiffness, in batch and row order, energies[j] its
// strain energy ½·u_jᵀ·K_e·u_j under column j of `vectors` (g × columns): each element's stiffness computed once for
// every column, the ranges of a wave on the pool (as forEachElementForce), visited in order.
template <typename Visit>
void forEachElementEnergies(const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                            linalg::ConstDenseMatrixView vectors, const base::CancellationToken* cancellation,
                            base::ThreadPool* pool, Visit&& visit)
{
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    const std::size_t columns = vectors.columns;
    std::vector<RangeEnergies> wave(threads); // bounded by the threads
    for (std::size_t b = 0; b < set.batches().size(); ++b)
    {
        const assembly::AssemblyBatch& batch = set.batches()[b];
        const elements::ElementFamily& family = *batch.family;
        if (!family.facets().has(elements::Facet::Stiffness))
            continue;
        const std::int64_t rows = batch.view.rowCount();
        std::int64_t length = StiffnessRecovery::RANGE_ELEMENTS;
        if (threads > 1)
        {
            const auto tasks = static_cast<std::int64_t>(threads) * StiffnessRecovery::RANGES_PER_THREAD;
            length = std::min(length, std::max(StiffnessRecovery::MIN_RANGE_ELEMENTS, (rows + tasks - 1) / tasks));
        }
        for (std::int64_t first = 0; first < rows; first += length * static_cast<std::int64_t>(threads))
        {
            if (cancellation != nullptr)
                cancellation->throwIfCancelled();
            const std::size_t ranges = static_cast<std::size_t>(
                std::min<std::int64_t>(static_cast<std::int64_t>(threads), (rows - first + length - 1) / length));
            const auto compute = [&](std::size_t r)
            {
                const std::int64_t begin = first + static_cast<std::int64_t>(r) * length;
                computeRangeEnergies(family, batch.view, {begin, std::min(rows, begin + length)}, dofMap, vectors,
                                     wave[r]);
            };
            if (pool != nullptr && ranges > 1)
                pool->runBatch(ranges, compute);
            else
                for (std::size_t r = 0; r < ranges; ++r)
                    compute(r);
            for (std::size_t r = 0; r < ranges; ++r)
            {
                const RangeEnergies& range = wave[r];
                for (std::size_t i = 0; i < range.block.size(); ++i)
                    visit(b, range.block.elementRow(i),
                          std::span<const double>(range.energies.data() + i * columns, columns));
            }
        }
    }
}

} // namespace

GridPointForceTable StiffnessRecovery::gridPointForces(std::int32_t subcase, const NodalState& state,
                                                       const OutputRequestPlan& request, base::MemoryBudget& budget,
                                                       const base::CancellationToken* cancellation) const
{
    return gridPointForces(subcase, state, request, CondensedUnits{}, budget, cancellation);
}

GridPointForceTable StiffnessRecovery::gridPointForces(std::int32_t subcase, const NodalState& state,
                                                       const OutputRequestPlan& request, const CondensedUnits& units,
                                                       base::MemoryBudget& budget,
                                                       const base::CancellationToken* cancellation) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    requireGSize(state.displacements, gSize, "displacements");
    requireGSize(state.appliedLoads, gSize, "applied loads");
    requireGSize(state.spcForces, gSize, "SPC forces");
    requireGSize(state.mpcForces, gSize, "MPC forces");
    const std::int32_t nodeCount = m_dofMap->nodeCount();
    const auto pointIdOf = [&](std::int32_t node)
    {
        return node < m_dofMap->gridCount() ? m_dofMap->gridId(node) : m_dofMap->scalarId(node - m_dofMap->gridCount());
    };
    const auto dofsOf = [&](std::int32_t node, std::array<base::DofIndex, 6>& dofs) -> int
    {
        if (node < m_dofMap->gridCount())
        {
            for (int c = 0; c < 6; ++c)
                dofs[static_cast<std::size_t>(c)] = m_dofMap->gridDof(node, c + 1);
            return 6;
        }
        dofs[0] = m_dofMap->scalarDof(node - m_dofMap->gridCount());
        return 1;
    };

    GridPointForceTable table;
    table.subcase = subcase;
    table.names.assign(GridPointForceTable::FIXED_ROW_NAMES.begin(), GridPointForceTable::FIXED_ROW_NAMES.end());
    // The element contributions, grouped by grid point (a stable sort keeps the element order within a point).
    std::vector<ElementContribution> contributions; // bounded by the element-grid pairs of the request
    base::MemoryReservation contributionCharge;
    std::vector<std::int32_t> nameOfBatch(m_set->batches().size(), -1); // bounded by the element types
    forEachElementForce(
        *m_set, *m_dofMap, state.displacements, state.temperatures, cancellation, m_pool,
        [&](std::size_t batchIndex, std::int64_t row, std::span<const elements::ElementDof> dofs,
            std::span<const double> force)
        {
            const elements::ElementBatchView& view = m_set->batches()[batchIndex].view;
            if (std::binary_search(units.excludedElements.begin(), units.excludedElements.end(), view.elementId(row)))
                return;
            if (nameOfBatch[batchIndex] < 0)
            {
                nameOfBatch[batchIndex] = static_cast<std::int32_t>(table.names.size());
                table.names.push_back(m_set->batches()[batchIndex].family->resultElementType(false));
            }
            const std::size_t firstOfElement = contributions.size();
            for (std::size_t k = 0; k < dofs.size(); ++k)
            {
                const base::DofIndex dof = m_dofMap->dofOf(dofs[k]);
                const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(dof);
                if (!request.takes(pointIdOf(nodeDof.node)))
                    continue;
                ElementContribution* entry = nullptr;
                for (std::size_t e = firstOfElement; e < contributions.size(); ++e)
                    if (contributions[e].pointRow == nodeDof.node)
                        entry = &contributions[e];
                if (entry == nullptr)
                {
                    contributions.push_back({nodeDof.node, view.elementId(row), nameOfBatch[batchIndex], {}});
                    entry = &contributions.back();
                }
                entry->values[static_cast<std::size_t>(nodeDof.component)] -= force[k];
            }
            if (contributions.size() * sizeof(ElementContribution) > contributionCharge.bytes())
                contributionCharge = budget.reserve(2 * contributions.size() * sizeof(ElementContribution),
                                                    "grid point force contributions");
        });
    std::stable_sort(contributions.begin(), contributions.end(),
                     [](const ElementContribution& a, const ElementContribution& b)
                     { return a.pointRow < b.pointRow; });

    // The rows of each grid point of the request.
    const auto isNonzero = [&](std::span<const double> values, const std::array<base::DofIndex, 6>& dofs, int count)
    {
        if (values.empty())
            return false;
        for (int c = 0; c < count; ++c)
            if (values[static_cast<std::size_t>(dofs[static_cast<std::size_t>(c)])] != 0.0)
                return true;
        return false;
    };
    // The condensed units' rows: those of the requested points, under one name.
    std::int32_t unitName = -1;
    std::size_t unitRowCount = 0;
    for (const UnitRow& unitRow : units.unitRows)
        if (request.takes(pointIdOf(unitRow.node)))
            ++unitRowCount;
    if (unitRowCount > 0)
    {
        unitName = static_cast<std::int32_t>(table.names.size());
        table.names.emplace_back(UNIT_ROW_NAME);
    }
    // The multiPointConstrForce rows by relation, when the reader takes them.
    const bool isMpcSplit = state.isMpcSplit && !state.mpcForces.empty();
    SplitMpcRows split;
    if (isMpcSplit)
    {
        base::BudgetedBuffer<std::uint8_t> isNodeTaken(budget, static_cast<std::size_t>(nodeCount),
                                                       "split MPC force nodes");
        for (std::int32_t node = 0; node < nodeCount; ++node)
            isNodeTaken[static_cast<std::size_t>(node)] = request.takes(pointIdOf(node)) ? 1 : 0;
        split = splitMpcRowsOf(state.mpcRelationForces, *m_dofMap, isNodeTaken.span(), budget);
    }
    const auto splitRowsWritten = [&](std::size_t first, std::size_t end)
    {
        std::size_t written = 0;
        for (std::size_t r = first; r < end; ++r)
            written += split.relations[r] != 0 || isNonzeroRow(split.values[r]) ? 1 : 0;
        return written;
    };
    std::size_t rowCount = contributions.size() + unitRowCount;
    std::array<base::DofIndex, 6> dofs{};
    std::size_t nextSplit = 0;
    for (std::int32_t node = 0; node < nodeCount; ++node)
    {
        if (!request.takes(pointIdOf(node)))
            continue;
        const int count = dofsOf(node, dofs);
        bool isConstrained = false;
        for (int c = 0; c < count; ++c)
            isConstrained =
                isConstrained || state.isConstrained[static_cast<std::size_t>(dofs[static_cast<std::size_t>(c)])] != 0;
        rowCount += 2 + (isConstrained ? 1 : 0);
        if (!isMpcSplit)
        {
            rowCount += isNonzero(state.mpcForces, dofs, count) ? 1 : 0;
            continue;
        }
        while (nextSplit < split.nodes.size() && split.nodes[nextSplit] < node)
            ++nextSplit;
        std::size_t end = nextSplit;
        while (end < split.nodes.size() && split.nodes[end] == node)
            ++end;
        rowCount += splitRowsWritten(nextSplit, end);
        nextSplit = end;
    }
    nextSplit = 0;
    table.nodeIds = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "grid point force grid points");
    table.elementIds = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "grid point force elements");
    table.nameIndices = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "grid point force names");
    table.values = base::BudgetedBuffer<double>(budget, rowCount * 6, "grid point force values");
    std::size_t next = 0;
    std::size_t nextContribution = 0;
    std::size_t nextUnitRow = 0;
    for (std::int32_t node = 0; node < nodeCount; ++node)
    {
        if (!request.takes(pointIdOf(node)))
            continue;
        const int count = dofsOf(node, dofs);
        const std::int32_t id = pointIdOf(node);
        std::array<double, 6> total{};
        const auto append = [&](std::int32_t elementId, std::int32_t name, const std::array<double, 6>& values)
        {
            table.nodeIds[next] = id;
            table.elementIds[next] = elementId;
            table.nameIndices[next] = name;
            for (std::size_t c = 0; c < 6; ++c)
                table.values[next * 6 + c] = values[c];
            ++next;
        };
        const auto pointValues = [&](std::span<const double> valuesG)
        {
            std::array<double, 6> values{};
            for (int c = 0; c < count; ++c)
                values[static_cast<std::size_t>(c)] =
                    valuesG[static_cast<std::size_t>(dofs[static_cast<std::size_t>(c)])];
            for (std::size_t c = 0; c < 6; ++c)
                total[c] += values[c];
            return values;
        };
        append(0, APPLIED_LOAD_NAME, pointValues(state.appliedLoads));
        bool isConstrained = false;
        for (int c = 0; c < count; ++c)
            isConstrained =
                isConstrained || state.isConstrained[static_cast<std::size_t>(dofs[static_cast<std::size_t>(c)])] != 0;
        if (isConstrained)
            append(0, SPC_FORCE_NAME, pointValues(state.spcForces));
        if (!isMpcSplit && isNonzero(state.mpcForces, dofs, count))
            append(0, MPC_FORCE_NAME, pointValues(state.mpcForces));
        if (isMpcSplit)
        {
            // total adds the point's MPC force as before (the same bits); the rows are its relations'.
            if (isNonzero(state.mpcForces, dofs, count))
                static_cast<void>(pointValues(state.mpcForces));
            while (nextSplit < split.nodes.size() && split.nodes[nextSplit] < node)
                ++nextSplit;
            for (; nextSplit < split.nodes.size() && split.nodes[nextSplit] == node; ++nextSplit)
                if (split.relations[nextSplit] != 0 || isNonzeroRow(split.values[nextSplit]))
                    append(split.relations[nextSplit], MPC_FORCE_NAME, split.values[nextSplit]);
        }
        while (nextContribution < contributions.size() && contributions[nextContribution].pointRow < node)
            ++nextContribution;
        while (nextContribution < contributions.size() && contributions[nextContribution].pointRow == node)
        {
            const ElementContribution& contribution = contributions[nextContribution++];
            for (std::size_t c = 0; c < 6; ++c)
                total[c] += contribution.values[c];
            append(contribution.elementId, contribution.name, contribution.values);
        }
        while (nextUnitRow < units.unitRows.size() && units.unitRows[nextUnitRow].node < node)
            ++nextUnitRow;
        while (nextUnitRow < units.unitRows.size() && units.unitRows[nextUnitRow].node == node)
        {
            const UnitRow& unitRow = units.unitRows[nextUnitRow++];
            for (std::size_t c = 0; c < 6; ++c)
                total[c] += unitRow.values[c];
            append(unitRow.unitId, unitName, unitRow.values);
        }
        append(0, TOTAL_NAME, total);
    }
    return table;
}

StiffnessRecovery::PointForces StiffnessRecovery::pointForces(
    std::span<const std::uint8_t> isNodeTaken, std::span<const std::vector<elements::ElementRange>> rowsOfBatch,
    linalg::ConstDenseMatrixView displacements, base::MemoryBudget& budget, const base::CancellationToken* cancellation,
    const RecoveryWindow* window) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    const std::size_t columns = displacements.columns;
    const std::size_t rowCount = window != nullptr ? 6 * window->gridSlotCount + window->scalarSlotCount : gSize;
    // The row of a g-set DOF in `displacements`: the DOF itself, or its slot's in the window.
    const auto rowOf = [&](base::DofIndex dof) -> std::size_t
    {
        if (window == nullptr)
            return static_cast<std::size_t>(dof);
        const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(dof);
        const bool isGrid = nodeDof.node < m_dofMap->gridCount();
        const std::int32_t slot =
            isGrid ? window->slots.gridSlots[static_cast<std::size_t>(nodeDof.node)]
                   : window->slots.scalarSlots[static_cast<std::size_t>(nodeDof.node - m_dofMap->gridCount())];
        if (slot < 0)
            throw std::invalid_argument(base::concatenate(
                {"StiffnessRecovery::pointForces: grid point ", std::to_string(m_dofMap->pointIdOf(dof)),
                 " of an element given holds no slot of the window (a bug of O_o_Solver: report it)"}));
        return isGrid ? 6 * static_cast<std::size_t>(slot) + static_cast<std::size_t>(nodeDof.component)
                      : 6 * window->gridSlotCount + static_cast<std::size_t>(slot);
    };
    if (columns > 0 && displacements.rows != rowCount)
        throw std::invalid_argument(
            base::concatenate({"StiffnessRecovery::pointForces: expected displacements of ", std::to_string(gSize),
                               " rows (the g set), found ", std::to_string(displacements.rows)}));
    if (isNodeTaken.size() != static_cast<std::size_t>(m_dofMap->nodeCount()) ||
        rowsOfBatch.size() != m_set->batches().size())
        throw std::invalid_argument(base::concatenate(
            {"StiffnessRecovery::pointForces: expected a flag per node (", std::to_string(m_dofMap->nodeCount()),
             ") and a list of rows per batch (", std::to_string(m_set->batches().size()), "), found ",
             std::to_string(isNodeTaken.size()), " and ", std::to_string(rowsOfBatch.size())}));
    PointForces out;
    out.names.assign(GridPointForceTable::FIXED_ROW_NAMES.begin(), GridPointForceTable::FIXED_ROW_NAMES.end());
    out.columns.resize(columns);
    if (columns == 0)
        return out;
    const std::size_t threads = m_pool != nullptr ? m_pool->threadCount() : 1;
    std::vector<elements::ElementMatrixBlock> wave(threads); // bounded by the threads
    std::vector<double> force;                               // bounded by an element's DOFs
    std::vector<std::size_t> dofIndex;                       // bounded by an element's DOFs
    std::vector<std::size_t> rowIndex;                       // bounded by an element's DOFs
    std::vector<std::size_t> firstOfElement(columns);        // bounded by the columns
    std::uint64_t contributionCount = 0;
    base::MemoryReservation contributionCharge;
    for (std::size_t b = 0; b < m_set->batches().size(); ++b)
    {
        const assembly::AssemblyBatch& batch = m_set->batches()[b];
        const elements::ElementFamily& family = *batch.family;
        if (!family.facets().has(elements::Facet::Stiffness) || rowsOfBatch[b].empty())
            continue;
        // The ranges given, cut to RANGE_ELEMENTS each, computed a wave of one per thread at a time.
        std::vector<elements::ElementRange> pieces; // bounded by the batch's rows
        for (const elements::ElementRange& range : rowsOfBatch[b])
            for (std::int64_t first = range.first; first < range.last; first += RANGE_ELEMENTS)
                pieces.push_back({first, std::min(range.last, first + RANGE_ELEMENTS)});
        std::int32_t name = -1;
        for (std::size_t p = 0; p < pieces.size(); p += threads)
        {
            if (cancellation != nullptr)
                cancellation->throwIfCancelled();
            const std::size_t count = std::min(threads, pieces.size() - p);
            const auto compute = [&](std::size_t r)
            {
                wave[r].clear();
                family.computeStiffness(batch.view, pieces[p + r], wave[r]);
            };
            if (m_pool != nullptr && count > 1)
                m_pool->runBatch(count, compute);
            else
                for (std::size_t r = 0; r < count; ++r)
                    compute(r);
            for (std::size_t r = 0; r < count; ++r)
            {
                const elements::ElementMatrixBlock& block = wave[r];
                for (std::size_t i = 0; i < block.size(); ++i)
                {
                    const std::span<const elements::ElementDof> dofs = block.dofs(i);
                    const std::span<const double> k = block.matrix(i);
                    const std::size_t n = dofs.size();
                    dofIndex.resize(n);
                    bool isTaken = false;
                    for (std::size_t c = 0; c < n; ++c)
                    {
                        dofIndex[c] = static_cast<std::size_t>(m_dofMap->dofOf(dofs[c]));
                        isTaken =
                            isTaken || isNodeTaken[static_cast<std::size_t>(
                                           m_dofMap->nodeDofOf(static_cast<base::DofIndex>(dofIndex[c])).node)] != 0;
                    }
                    if (!isTaken)
                        continue;
                    rowIndex.resize(n);
                    for (std::size_t c = 0; c < n; ++c)
                        rowIndex[c] = rowOf(static_cast<base::DofIndex>(dofIndex[c]));
                    if (name < 0)
                    {
                        name = static_cast<std::int32_t>(out.names.size());
                        out.names.push_back(batch.family->resultElementType(false));
                    }
                    const std::int32_t elementId = batch.view.elementId(block.elementRow(i));
                    force.resize(n);
                    for (std::size_t j = 0; j < columns; ++j)
                    {
                        const double* u = displacements.pointer(0, j);
                        // The chain of computeRangeForces: f_r = Σ_c K_rc·u_c.
                        for (std::size_t row = 0; row < n; ++row)
                        {
                            double sum = 0.0;
                            for (std::size_t c = 0; c < n; ++c)
                                sum += k[row * n + c] * u[rowIndex[c]];
                            force[row] = sum;
                        }
                        std::vector<PointContribution>& made = out.columns[j];
                        firstOfElement[j] = made.size();
                        for (std::size_t c = 0; c < n; ++c)
                        {
                            const linalg::NodeDof nodeDof =
                                m_dofMap->nodeDofOf(static_cast<base::DofIndex>(dofIndex[c]));
                            if (isNodeTaken[static_cast<std::size_t>(nodeDof.node)] == 0)
                                continue;
                            PointContribution* entry = nullptr;
                            for (std::size_t e = firstOfElement[j]; e < made.size(); ++e)
                                if (made[e].node == nodeDof.node)
                                    entry = &made[e];
                            if (entry == nullptr)
                            {
                                made.push_back({nodeDof.node, elementId, name, {}});
                                entry = &made.back();
                                ++contributionCount;
                            }
                            entry->values[static_cast<std::size_t>(nodeDof.component)] -= force[c];
                        }
                    }
                    if (contributionCount * sizeof(PointContribution) > contributionCharge.bytes())
                        contributionCharge = budget.reserve(2 * contributionCount * sizeof(PointContribution),
                                                            "grid point force contributions");
                }
            }
        }
    }
    for (std::vector<PointContribution>& made : out.columns)
        std::stable_sort(made.begin(), made.end(),
                         [](const PointContribution& a, const PointContribution& b) { return a.node < b.node; });
    return out;
}

GridPointForceTable StiffnessRecovery::pointForceTable(std::int32_t subcase, const PointForces& forces,
                                                       std::size_t column, std::span<const std::int32_t> rowNodes,
                                                       std::span<const PointFixedRows> fixedRows,
                                                       base::MemoryBudget& budget) const
{
    if (column >= forces.columns.size())
        throw std::invalid_argument(
            base::concatenate({"StiffnessRecovery::pointForceTable: expected a column below ",
                               std::to_string(forces.columns.size()), ", found ", std::to_string(column)}));
    const std::vector<PointContribution>& contributions = forces.columns[column];
    const auto pointIdOf = [&](std::int32_t node)
    {
        return node < m_dofMap->gridCount() ? m_dofMap->gridId(node) : m_dofMap->scalarId(node - m_dofMap->gridCount());
    };
    const auto isNonzero = [](const std::array<double, 6>& values) { return isNonzeroRow(values); };
    const auto mpcRowsOf = [&](const PointFixedRows& fixed) -> std::size_t
    {
        if (!fixed.isMpcSplit)
            return isNonzero(fixed.mpc) ? 1 : 0;
        return static_cast<std::size_t>(std::count_if(fixed.mpcByRelation.begin(), fixed.mpcByRelation.end(),
                                                      [&](const RelationRow& row)
                                                      { return row.relation != 0 || isNonzero(row.values); }));
    };
    // The rows: per node its contributions, and with fixed rows appliedLoad, singlePointConstrForce,
    // multiPointConstrForce and total.
    std::size_t rowCount = 0;
    {
        std::size_t nextContribution = 0;
        std::size_t nextFixed = 0;
        for (const std::int32_t node : rowNodes)
        {
            while (nextContribution < contributions.size() && contributions[nextContribution].node < node)
                ++nextContribution;
            while (nextContribution < contributions.size() && contributions[nextContribution].node == node)
            {
                ++rowCount;
                ++nextContribution;
            }
            while (nextFixed < fixedRows.size() && fixedRows[nextFixed].node < node)
                ++nextFixed;
            if (nextFixed < fixedRows.size() && fixedRows[nextFixed].node == node)
            {
                const PointFixedRows& fixed = fixedRows[nextFixed];
                rowCount += 2 + (fixed.isHeld ? 1 : 0) + mpcRowsOf(fixed);
            }
        }
    }
    GridPointForceTable table;
    table.subcase = subcase;
    table.names = forces.names;
    table.nodeIds = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "grid point force grid points");
    table.elementIds = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "grid point force elements");
    table.nameIndices = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "grid point force names");
    table.values = base::BudgetedBuffer<double>(budget, rowCount * GridPointForceTable::COMPONENT_COUNT,
                                                "grid point force values");
    std::size_t next = 0;
    std::size_t nextContribution = 0;
    std::size_t nextFixed = 0;
    for (const std::int32_t node : rowNodes)
    {
        const std::int32_t id = pointIdOf(node);
        std::array<double, 6> total{};
        const auto append =
            [&](std::int32_t elementId, std::int32_t name, const std::array<double, 6>& values, bool isInTotal = true)
        {
            table.nodeIds[next] = id;
            table.elementIds[next] = elementId;
            table.nameIndices[next] = name;
            for (std::size_t c = 0; c < 6; ++c)
            {
                table.values[next * 6 + c] = values[c];
                if (isInTotal)
                    total[c] += values[c];
            }
            ++next;
        };
        while (nextFixed < fixedRows.size() && fixedRows[nextFixed].node < node)
            ++nextFixed;
        const PointFixedRows* fixed =
            nextFixed < fixedRows.size() && fixedRows[nextFixed].node == node ? &fixedRows[nextFixed] : nullptr;
        if (fixed != nullptr)
        {
            append(0, APPLIED_LOAD_NAME, fixed->applied);
            if (fixed->isHeld)
                append(0, SPC_FORCE_NAME, fixed->spc);
            if (!fixed->isMpcSplit && isNonzero(fixed->mpc))
                append(0, MPC_FORCE_NAME, fixed->mpc);
            if (fixed->isMpcSplit)
            {
                // total adds the point's MPC force as before (the same bits); the rows are its relations'.
                if (isNonzero(fixed->mpc))
                    for (std::size_t c = 0; c < 6; ++c)
                        total[c] += fixed->mpc[c];
                for (const RelationRow& row : fixed->mpcByRelation)
                    if (row.relation != 0 || isNonzero(row.values))
                        append(row.relation, MPC_FORCE_NAME, row.values, false);
            }
        }
        while (nextContribution < contributions.size() && contributions[nextContribution].node < node)
            ++nextContribution;
        while (nextContribution < contributions.size() && contributions[nextContribution].node == node)
        {
            const PointContribution& contribution = contributions[nextContribution++];
            append(contribution.elementId, contribution.name, contribution.values);
        }
        if (fixed != nullptr)
        {
            const std::array<double, 6> sum = total;
            append(0, TOTAL_NAME, sum);
        }
    }
    return table;
}

void StiffnessRecovery::strainEnergy(std::int32_t subcase, std::span<const double> displacements,
                                     const OutputRequestPlan& request, base::MemoryBudget& budget,
                                     std::vector<ElementResultTable>& tables,
                                     const base::CancellationToken* cancellation) const
{
    requireGSize(displacements, static_cast<std::size_t>(m_dofMap->gSetSize()), "displacements");
    std::vector<std::vector<ElementResultTable>> ofVector;
    strainEnergies(subcase,
                   {displacements.data(), displacements.size(), 1, std::max<std::size_t>(1, displacements.size())},
                   request, budget, ofVector, cancellation);
    for (ElementResultTable& table : ofVector.front())
        tables.push_back(std::move(table));
}

void StiffnessRecovery::strainEnergies(std::int32_t subcase, linalg::ConstDenseMatrixView vectors,
                                       const OutputRequestPlan& request, base::MemoryBudget& budget,
                                       std::vector<std::vector<ElementResultTable>>& tables,
                                       const base::CancellationToken* cancellation, std::size_t largestGroup) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (vectors.columns > 0 && vectors.rows != gSize)
        throw std::invalid_argument(
            base::concatenate({"StiffnessRecovery::strainEnergies: expected vectors of ", std::to_string(gSize),
                               " rows (the g set), found ", std::to_string(vectors.rows)}));
    tables.resize(vectors.columns);
    if (vectors.columns == 0)
        return;
    // The energies of every element (the percentages are of the total of the model), then the tables of the request.
    std::size_t elementCount = 0;
    for (const assembly::AssemblyBatch& batch : m_set->batches())
        elementCount += static_cast<std::size_t>(batch.view.rowCount());
    std::vector<std::size_t> firstOfBatch(m_set->batches().size() + 1, 0); // bounded by the element types
    for (std::size_t b = 0; b < m_set->batches().size(); ++b)
        firstOfBatch[b + 1] = firstOfBatch[b] + static_cast<std::size_t>(m_set->batches()[b].view.rowCount());
    // PERFORMANCE: the vectors in groups (the modes of an eigenvalue solution), each element's stiffness computed once
    // for a group on the pool, its energies the chains of the vector alone (forEachElementEnergies), so the same bits;
    // a group's energies held within STRAIN_ENERGY_MEMORY_DIVISOR-th of the memory left. The tables of every vector
    // are made from them as they were made from one.
    // Measured: 20 modes of a 242 406 DOF plate with every element's strain energy.
    const std::uint64_t perVector = std::max<std::uint64_t>(1, elementCount) * sizeof(double);
    const std::uint64_t share = budget.available() / STRAIN_ENERGY_MEMORY_DIVISOR;
    const std::uint64_t most = largestGroup > 0 ? std::min(largestGroup, vectors.columns) : vectors.columns;
    const auto group = static_cast<std::size_t>(std::clamp<std::uint64_t>(share / perVector, 1, most));
    std::vector<std::uint8_t> hasStiffness(m_set->batches().size(), 0); // bounded by the element types
    // The batches with a requested element, whose volumes the density needs (the same for every vector: made once).
    std::vector<std::size_t> requestedRows(m_set->batches().size(), 0); // bounded by the element types
    std::vector<std::uint8_t> isRequested(m_set->batches().size(), 0);  // bounded by the element types
    std::optional<assembly::ElementVolumes> volumes;
    for (std::size_t firstVector = 0; firstVector < vectors.columns; firstVector += group)
    {
        const std::size_t width = std::min(group, vectors.columns - firstVector);
        base::BudgetedBuffer<double> energies(budget, elementCount * width, "element strain energies");
        std::vector<double> totals(width, 0.0); // bounded by the group
        forEachElementEnergies(
            *m_set, *m_dofMap, vectors.block(0, firstVector, vectors.rows, width), cancellation, m_pool,
            [&](std::size_t batchIndex, std::int64_t row, std::span<const double> energy)
            {
                const std::size_t at = (firstOfBatch[batchIndex] + static_cast<std::size_t>(row)) * width;
                for (std::size_t j = 0; j < width; ++j)
                {
                    energies[at + j] = energy[j];
                    totals[j] += energy[j];
                }
                hasStiffness[batchIndex] = 1;
            });
        if (!volumes)
        {
            for (std::size_t b = 0; b < m_set->batches().size(); ++b)
            {
                if (hasStiffness[b] == 0)
                    continue;
                const elements::ElementBatchView& view = m_set->batches()[b].view;
                for (std::int64_t row = 0; row < view.rowCount(); ++row)
                    requestedRows[b] += request.takes(view.elementId(row)) ? 1 : 0;
                isRequested[b] = requestedRows[b] > 0 ? 1 : 0;
            }
            volumes.emplace(*m_set, isRequested, budget, m_pool);
        }
        for (std::size_t j = 0; j < width; ++j)
        {
            const double total = totals[j];
            for (std::size_t b = 0; b < m_set->batches().size(); ++b)
            {
                if (isRequested[b] == 0)
                    continue;
                const elements::ElementBatchView& view = m_set->batches()[b].view;
                const std::span<const double> volumesOfBatch = volumes->volumesOf(b);
                std::size_t rows = requestedRows[b];
                ElementResultTable table;
                table.kind = "strainEnergy";
                table.elementType = m_set->batches()[b].family->resultElementType(false);
                table.subcase = subcase;
                table.components = {"strainEnergy", "percent", "strainEnergyDensity"};
                // One row per element, then the subtotal of the element type (element ID SUBTOTAL_ELEMENT_ID).
                ++rows;
                table.ids = base::BudgetedBuffer<std::int32_t>(budget, rows, "strain energy ids");
                table.nodeIds = base::BudgetedBuffer<std::int32_t>(budget, rows, "strain energy grid points");
                table.layers = base::BudgetedBuffer<std::int32_t>(budget, rows, "strain energy layers");
                table.values = base::BudgetedBuffer<double>(budget, rows * 3, "strain energy values");
                std::size_t next = 0;
                double subtotal = 0.0;
                for (std::int64_t row = 0; row < view.rowCount(); ++row)
                {
                    if (!request.takes(view.elementId(row)))
                        continue;
                    const double energy = energies[(firstOfBatch[b] + static_cast<std::size_t>(row)) * width + j];
                    table.ids[next] = view.elementId(row);
                    table.layers[next] = 1;
                    table.values[next * 3] = energy;
                    table.values[next * 3 + 1] = total != 0.0 ? 100.0 * energy / total : 0.0;
                    // The density is the energy over the element volume; none for an element without a volume (a
                    // spring).
                    const double volume = volumesOfBatch.empty() ? 0.0 : volumesOfBatch[static_cast<std::size_t>(row)];
                    table.values[next * 3 + 2] =
                        volume > 0.0 ? energy / volume : std::numeric_limits<double>::quiet_NaN();
                    subtotal += energy;
                    ++next;
                }
                table.ids[next] = SUBTOTAL_ELEMENT_ID;
                table.layers[next] = 1;
                table.values[next * 3] = subtotal;
                table.values[next * 3 + 1] = total != 0.0 ? 100.0 * subtotal / total : 0.0;
                // No density on the subtotal row (NaN): the densities of elements of different volumes do not add.
                table.values[next * 3 + 2] = std::numeric_limits<double>::quiet_NaN();
                tables[firstVector + j].push_back(std::move(table));
            }
        }
    }
}

} // namespace oos::output
