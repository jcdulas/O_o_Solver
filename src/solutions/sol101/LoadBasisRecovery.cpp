#include "oos/solutions/sol101/LoadBasisRecovery.hpp"

#include "oos/assembly/assemble/ElementConnectivity.hpp"
#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/output/arrays/NodalResultBuilder.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/solutions/sol101/BasisFixedRows.hpp"
#include "oos/solutions/sol101/CondensedLoadBasis.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::solutions
{

namespace
{

// The refusal of what a thermal unit cannot give: the thermal loads of its elements are not in its grid point forces,
// declared quantities and section and cut loads.
[[noreturn]] void refuseThermalUnits(const char* what)
{
    throw base::SolverError(base::concatenate(
        {"the ", what,
         " of the thermal units of a load basis are not made (their elements' thermal loads are not in "
         "them); ask them of the mechanical units only, or solve the thermal subcases in a "
         "static analysis"}));
}

} // namespace

LoadBasisRecovery::LoadBasisRecovery(const input::ModelSnapshot& snapshot, std::shared_ptr<const KeptLoadBasis> basis)
    : m_snapshot(&snapshot), m_basis(std::move(basis)), m_registry(elements::ElementRegistry::forModel(snapshot))
{
    if (!m_basis || !m_basis->index || !m_basis->dofMap)
        throw std::invalid_argument(
            "LoadBasisRecovery: expected a kept load basis with its index and DOF map, found "
            "none (a bug of O_o_Solver: report it)");
    m_set = std::make_unique<assembly::ElementSet>(*m_basis->index, m_registry);
    for (const KeptLoadBasisSet& set : m_basis->sets)
        m_unitCount += static_cast<std::int32_t>(set.units.size());
}

LoadBasisRecovery::UnitPlace LoadBasisRecovery::placeOf(std::int32_t unit) const
{
    if (unit < 1 || unit > m_unitCount)
        throw std::invalid_argument(
            base::concatenate({"load basis: expected a unit from 1 to ", std::to_string(m_unitCount), ", found ",
                               std::to_string(unit), "; take the units from SolverBasis.unit_ids"}));
    std::int32_t first = 1;
    for (std::size_t s = 0; s < m_basis->sets.size(); ++s)
    {
        const auto count = static_cast<std::int32_t>(m_basis->sets[s].units.size());
        if (unit < first + count)
            return {s, static_cast<std::size_t>(unit - first)};
        first += count;
    }
    throw std::invalid_argument("load basis: the units of the sets do not add up (a bug of O_o_Solver: report it)");
}

std::size_t LoadBasisRecovery::condensedUnitCount() const noexcept
{
    for (const KeptLoadBasisSet& set : m_basis->sets)
        if (set.condensed)
            return set.condensed->set->units().size();
    return 0;
}

std::vector<std::uint8_t> LoadBasisRecovery::condensedUnitsRead(bool isEveryUnit, std::span<const std::int32_t> points,
                                                                std::span<const std::int32_t> elements,
                                                                std::span<const std::int32_t> touchedPoints) const
{
    const std::size_t unitCount = condensedUnitCount();
    std::vector<std::uint8_t> isRead(unitCount, isEveryUnit ? 1 : 0); // bounded by the condensed units
    if (isEveryUnit || unitCount == 0)
        return isRead;
    const std::vector<KeptLoadBasis::CondensedElement>& owners = m_basis->condensedElements;
    for (const std::int32_t elementId : elements)
    {
        const auto found = std::lower_bound(owners.begin(), owners.end(), elementId,
                                            [](const KeptLoadBasis::CondensedElement& entry, std::int32_t id)
                                            { return entry.elementId < id; });
        if (found != owners.end() && found->elementId == elementId)
            isRead[found->unit] = 1;
    }
    const KeptCondensedUnits* units = nullptr;
    for (const KeptLoadBasisSet& set : m_basis->sets)
        if (set.condensed)
            units = &set.condensed->set->units();
    // The units holding one of `asked` (sorted): as an interior point, or as any point of theirs.
    const auto markHolders = [&](std::vector<std::int32_t> asked, bool isInteriorOnly)
    {
        if (asked.empty())
            return;
        std::sort(asked.begin(), asked.end());
        for (std::size_t u = 0; u < unitCount && units != nullptr; ++u)
        {
            const KeptCondensedUnit& unit = (*units)[u];
            for (std::size_t local = 0; local < unit.modelDofOfLocal.size() && isRead[u] == 0; ++local)
                if ((!isInteriorOnly || unit.isInteriorLocal[local] != 0) && unit.modelDofOfLocal[local] >= 0 &&
                    std::binary_search(asked.begin(), asked.end(),
                                       m_basis->dofMap->pointIdOf(unit.modelDofOfLocal[local])))
                    isRead[u] = 1;
        }
    };
    markHolders({points.begin(), points.end()}, true);
    markHolders({touchedPoints.begin(), touchedPoints.end()}, false);
    return isRead;
}

namespace
{

// The units whose element tables are recovered together (LoadBasisRecovery::recover): as many as their gathered
// displacements fit in 1/UNIT_BLOCK_SHARE of the memory left, at least one and at most UNIT_BLOCK_MAX (beyond, the
// slopes of an element are a small part of its rows); the bits do not depend on the block.
constexpr std::uint64_t UNIT_BLOCK_SHARE = 16;
constexpr std::size_t UNIT_BLOCK_MAX = 256;

std::size_t unitBlockOf(std::size_t gSize, std::size_t units, const base::MemoryBudget& budget)
{
    const std::uint64_t perUnit = static_cast<std::uint64_t>(std::max<std::size_t>(gSize, 1)) * sizeof(double);
    const std::uint64_t fits = budget.available() / UNIT_BLOCK_SHARE / perUnit;
    return static_cast<std::size_t>(
        std::clamp<std::uint64_t>(fits, 1, std::min<std::uint64_t>(UNIT_BLOCK_MAX, std::max<std::size_t>(units, 1))));
}

// The units whose grid point forces are made together (one StiffnessRecovery::pointForces): as many as their element
// contributions fit in 1/UNIT_BLOCK_SHARE of the memory left, at least one.
constexpr std::uint64_t CONTRIBUTION_BYTES = sizeof(output::StiffnessRecovery::PointContribution);

std::size_t forceBlockOf(std::uint64_t pairs, std::size_t units, const base::MemoryBudget& budget)
{
    const std::uint64_t perUnit = std::max<std::uint64_t>(pairs, 1) * CONTRIBUTION_BYTES;
    const std::uint64_t fits = budget.available() / UNIT_BLOCK_SHARE / perUnit;
    return static_cast<std::size_t>(std::clamp<std::uint64_t>(fits, 1, std::max<std::size_t>(units, 1)));
}

} // namespace

// The g-set displacements of the units asked, one after the other: a column of the kept displacements on the full
// path; on the condensed path, the interface's displacements and the interiors of the condensed units read, recovered
// for the units asked of the set together (CondensedBasisColumns).
class LoadBasisRecovery::UnitDisplacements
{
public:
    UnitDisplacements(const LoadBasisRecovery& recovery, std::span<const std::int32_t> units,
                      std::span<const std::uint8_t> isUnitRead, base::MemoryBudget& budget, base::ThreadPool* pool)
        : m_basis(recovery.m_basis.get())
    {
        const std::vector<KeptLoadBasisSet>& sets = m_basis->sets;
        m_columns.resize(sets.size());
        for (const std::int32_t unit : units)
        {
            const UnitPlace place = recovery.placeOf(unit);
            std::vector<std::size_t>& asked = m_columns[place.set];
            const auto found = std::find(asked.begin(), asked.end(), place.column);
            m_places.push_back({place.set, static_cast<std::size_t>(found - asked.begin())});
            if (found == asked.end())
                asked.push_back(place.column);
        }
        m_condensed.resize(sets.size());
        std::unique_lock<std::mutex> lock(recovery.m_condensedRecovery, std::defer_lock);
        for (std::size_t s = 0; s < sets.size(); ++s)
        {
            if (!sets[s].condensed || m_columns[s].empty())
                continue;
            if (!lock.owns_lock())
                lock.lock();
            const linalg::DenseKernels kernels(sets[s].condensed->set->kernel(), linalg::DenseKernelOptions{pool});
            m_condensed[s].emplace(*sets[s].condensed, m_columns[s], isUnitRead, kernels, budget);
            if (m_buffer.size() == 0)
                m_buffer = base::BudgetedBuffer<double>(budget, static_cast<std::size_t>(m_basis->dofMap->gSetSize()),
                                                        "load basis unit displacements");
        }
    }

    // The g-set displacements of the i-th unit asked (valid until the next call).
    [[nodiscard]] std::span<const double> of(std::size_t i)
    {
        const Place& place = m_places[i];
        if (m_condensed[place.set])
        {
            m_condensed[place.set]->fill(place.asked, m_buffer.span());
            return m_buffer.span();
        }
        const KeptLoadBasisSet& set = m_basis->sets[place.set];
        const std::size_t column = m_columns[place.set][place.asked];
        return {set.displacements.data() + column * set.displacements.leadingDimension(), set.displacements.rows()};
    }

    // Adds the relief loads of the units recovered for the i-th unit asked (the condensed path with inertia relief:
    // CondensedBasisColumns::addUnitReliefLoads), nothing on the full path.
    void addUnitRelief(std::size_t i, std::span<double> loadsG) const
    {
        const Place& place = m_places[i];
        if (m_condensed[place.set])
            m_condensed[place.set]->addUnitReliefLoads(place.asked, loadsG);
    }

private:
    struct Place
    {
        std::size_t set = 0;
        std::size_t asked = 0; // among the columns asked of the set
    };
    const KeptLoadBasis* m_basis;
    std::vector<Place> m_places;                                   // per unit asked
    std::vector<std::vector<std::size_t>> m_columns;               // per set: the columns asked, in order
    std::vector<std::optional<CondensedBasisColumns>> m_condensed; // per set
    base::BudgetedBuffer<double> m_buffer;
};

output::ResultArrays LoadBasisRecovery::recover(std::span<const std::int32_t> units, bool withDisplacements,
                                                bool isAllPoints, std::span<const std::int32_t> points,
                                                const output::ElementOutputSelection& selection,
                                                base::MemoryBudget& budget, base::ThreadPool* pool,
                                                const output::OutputRequestPlan* gridPointForces) const
{
    output::ResultArrays arrays;
    if (isStreamed())
        return recoverStreamed(units, withDisplacements, selection, gridPointForces, budget);
    const output::NodalResultBuilder nodal(*m_basis->dofMap);
    const output::ElementResultBuilder elements(*m_set, *m_basis->dofMap);
    const bool hasElementTables =
        selection.stress.isRequested || selection.strain.isRequested || selection.forces.isRequested;
    // The condensed units whose interiors the tables read: those of the points and of the elements asked.
    bool isEveryUnit = withDisplacements && isAllPoints;
    std::vector<std::int32_t> elementIds; // bounded by the elements asked
    for (const output::ElementKindSelection* kind : {&selection.stress, &selection.strain, &selection.forces})
    {
        if (!kind->isRequested)
            continue;
        isEveryUnit = isEveryUnit || kind->isAll;
        elementIds.insert(elementIds.end(), kind->members.begin(), kind->members.end());
    }
    const bool hasForces = gridPointForces != nullptr && gridPointForces->isRequested;
    if (hasForces && hasThermalUnit(units))
        refuseThermalUnits("grid point forces");
    isEveryUnit = isEveryUnit || (hasForces && gridPointForces->isAll);
    // The initial strains of the thermal units asked, one field per key of temperature and deformation sets, which
    // their element tables are recovered with. Its problems were reported by the analysis, which read the same sets.
    std::map<InitialStrainKey, std::unique_ptr<assembly::TemperatureField>> temperatureFields;
    std::vector<const elements::ElementTemperatures*> temperaturesOf(units.size(), nullptr); // per unit asked
    if (hasElementTables)
        for (std::size_t j = 0; j < units.size(); ++j)
        {
            const UnitPlace place = placeOf(units[j]);
            const LoadBasisUnit& unit = m_basis->sets[place.set].units[place.column];
            if (!unit.isThermal())
                continue;
            InitialStrainKey key;
            if (unit.temperatureLoad != 0)
            {
                key.temperatureLoad = unit.temperatureLoad;
                key.temperatureInitial = unit.temperatureInitial;
            }
            if (unit.deformation != 0)
                key.deformation = unit.deformation;
            auto [found, isNew] = temperatureFields.try_emplace(key);
            if (isNew)
            {
                base::DiagnosticSink reportedByTheAnalysis;
                found->second =
                    initialStrainsOf(*m_snapshot, m_basis->dofMap->points(), key, budget, reportedByTheAnalysis);
            }
            temperaturesOf[j] = &found->second->temperatures();
        }
    // The grid point forces: the rows of the points asked; the element forces at those points and at the
    // dependent points of the relations the units' sets keep (their multiPointConstrForce: q_m = (K·u − P)_m).
    const assembly::DofMap& dofMap = *m_basis->dofMap;
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    std::vector<std::int32_t> rowNodes; // bounded by the points asked
    base::BudgetedBuffer<std::uint8_t> isForceNode(budget, hasForces ? nodeCount : 0, "load basis force nodes");
    std::vector<std::int32_t> touchedPoints; // bounded by the points asked
    if (hasForces)
    {
        const input::PointIndex& pointIndex = dofMap.points();
        const auto nodeOf = [&](base::EntityId id) -> std::int32_t
        {
            const std::optional<input::PointRef> point = pointIndex.find(id);
            if (!point)
                return -1;
            return point->kind == input::PointKind::Scalar ? dofMap.gridCount() + point->row : point->row;
        };
        if (gridPointForces->isAll)
            for (std::size_t node = 0; node < nodeCount; ++node)
                rowNodes.push_back(static_cast<std::int32_t>(node));
        else
            for (const std::int32_t id : gridPointForces->members)
                if (const std::int32_t node = nodeOf(id); node >= 0)
                    rowNodes.push_back(node);
        std::sort(rowNodes.begin(), rowNodes.end());
        rowNodes.erase(std::unique(rowNodes.begin(), rowNodes.end()), rowNodes.end());
        for (const std::int32_t node : rowNodes)
            isForceNode[static_cast<std::size_t>(node)] = 1;
        for (const std::int32_t unit : units)
            if (const std::shared_ptr<const KeptFixedRows>& rows = m_basis->sets[placeOf(unit).set].fixedRows)
                for (const std::int32_t node : rows->mNodes.span())
                    isForceNode[static_cast<std::size_t>(node)] = 1;
        for (std::size_t node = 0; node < nodeCount && !isEveryUnit; ++node)
            if (isForceNode[node] != 0)
                touchedPoints.push_back(node < static_cast<std::size_t>(dofMap.gridCount())
                                            ? dofMap.gridId(static_cast<std::int32_t>(node))
                                            : dofMap.scalarId(static_cast<std::int32_t>(node) - dofMap.gridCount()));
    }
    const std::vector<std::uint8_t> isUnitRead = condensedUnitsRead(
        isEveryUnit, withDisplacements ? points : std::span<const std::int32_t>{}, elementIds, touchedPoints);
    UnitDisplacements displacementsOf(*this, units, isUnitRead, budget, pool);
    const output::StiffnessRecovery stiffness(*m_set, dofMap, pool);
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    // The elements with a grid point among the force nodes, and their element-node pairs there (the size of a block's
    // contributions).
    std::uint64_t pairs = 0;
    const std::vector<std::vector<elements::ElementRange>> forceRows =
        hasForces ? rowsAtNodes(isForceNode.span(), budget, pool, pairs)
                  : std::vector<std::vector<elements::ElementRange>>{};
    // The condensed path with inertia relief: the relief loads of the units recovered, their part of appliedLoad.
    bool hasUnitRelief = false;
    for (const KeptLoadBasisSet& set : m_basis->sets)
        hasUnitRelief =
            hasUnitRelief || (set.fixedRows && set.condensed && set.condensed->reliefAmplitudes.columns() > 0);
    base::BudgetedBuffer<double> unitRelief(budget, hasForces && hasUnitRelief ? gSize : 0,
                                            "load basis unit relief loads");
    BasisFixedRows::Work fixedWork;
    // PERFORMANCE: the element tables of the units asked recovered by blocks (ElementResultBuilder::buildCases: the
    // slopes of an element made once for the units of a block), the units' displacements gathered block by block; a
    // unit has no line load (refused in a kept basis) and no temperature (thermal cases are refused or left out), so
    // every unit is a block case. Whether a family's kinds are vectorised depends on the units asked, not on the
    // blocks, so a block is as large as the memory allows. The grid point forces of a block likewise
    // (StiffnessRecovery::pointForces: each element's matrix made once for the block's units, only the elements with
    // a grid point asked).
    const bool gathers = hasElementTables || hasForces;
    const std::size_t block = gathers ? unitBlockOf(gSize, units.size(), budget) : 1;
    linalg::DenseMatrix gathered(budget, gSize, gathers ? block : 0, "load basis unit displacements block");
    for (std::size_t first = 0; first < units.size(); first += block)
    {
        const std::size_t count = std::min(block, units.size() - first);
        std::vector<std::vector<output::ElementResultTable>> blockTables(count); // bounded by the block
        if (gathers)
            for (std::size_t j = 0; j < count; ++j)
            {
                const std::span<const double> displacements = displacementsOf.of(first + j);
                std::copy(displacements.begin(), displacements.end(), &gathered(0, j));
            }
        if (hasElementTables)
        {
            // The thermal units of the block with their temperatures; a block without one gives no loads, as before.
            std::vector<output::CaseLoads> loads; // bounded by the block
            if (std::any_of(temperaturesOf.begin() + static_cast<std::ptrdiff_t>(first),
                            temperaturesOf.begin() + static_cast<std::ptrdiff_t>(first + count),
                            [](const elements::ElementTemperatures* field) { return field != nullptr; }))
            {
                loads.resize(count);
                for (std::size_t j = 0; j < count; ++j)
                    loads[j].temperatures = temperaturesOf[first + j];
            }
            elements.buildCases(units.subspan(first, count), gathered.view().block(0, 0, gSize, count), units.size(),
                                budget, blockTables, pool, nullptr, &selection, loads);
        }
        std::vector<output::GridPointForceTable> forceTables(hasForces ? count : 0); // bounded by the block
        if (hasForces)
        {
            const std::size_t forceBlock = forceBlockOf(pairs, count, budget);
            for (std::size_t from = 0; from < count; from += forceBlock)
            {
                const std::size_t columns = std::min(forceBlock, count - from);
                const output::StiffnessRecovery::PointForces forces = stiffness.pointForces(
                    isForceNode.span(), forceRows, gathered.view().block(0, from, gSize, columns), budget);
                for (std::size_t c = 0; c < columns; ++c)
                {
                    const std::size_t j = from + c;
                    const std::int32_t unit = units[first + j];
                    const UnitPlace place = placeOf(unit);
                    const std::shared_ptr<const KeptFixedRows>& kept = m_basis->sets[place.set].fixedRows;
                    std::vector<output::StiffnessRecovery::PointFixedRows> fixed; // bounded by the points asked
                    if (kept)
                    {
                        std::span<const double> added;
                        if (unitRelief.size() > 0)
                        {
                            std::fill(unitRelief.data(), unitRelief.data() + gSize, 0.0);
                            displacementsOf.addUnitRelief(first + j, unitRelief.span());
                            added = unitRelief.span();
                        }
                        fixed = BasisFixedRows::rowsOf(*kept, place.column, rowNodes, forces, c, added, dofMap, budget,
                                                       fixedWork);
                    }
                    forceTables[j] = stiffness.pointForceTable(unit, forces, c, rowNodes, fixed, budget);
                }
            }
        }
        for (std::size_t j = 0; j < count; ++j)
        {
            const std::int32_t unit = units[first + j];
            // The gathered column when there is one (on the condensed path the units' interiors are filled anew at
            // each call of `of`).
            const std::span<const double> displacements =
                gathers ? std::span<const double>(&gathered(0, j), gSize) : displacementsOf.of(first + j);
            if (withDisplacements)
                arrays.nodalTables.push_back(
                    nodal.build("displacements", unit, displacements, budget, isAllPoints, points));
            for (output::ElementResultTable& table : blockTables[j])
                arrays.elementTables.push_back(std::move(table));
            if (hasForces)
                arrays.gridPointForceTables.push_back(std::move(forceTables[j]));
        }
    }
    return arrays;
}

const LoadBasisRecovery::ElementsOfNodes& LoadBasisRecovery::elementsOfNodes(base::MemoryBudget& budget,
                                                                             base::ThreadPool* pool) const
{
    const std::lock_guard<std::mutex> lock(m_elementsOfNodesMade);
    if (m_elementsOfNodes)
        return *m_elementsOfNodes;
    const assembly::DofMap& dofMap = *m_basis->dofMap;
    const assembly::ElementConnectivity connectivity(*m_set, dofMap, budget, pool);
    auto made = std::make_unique<ElementsOfNodes>();
    // The batch and row of each element ID (the elements' IDs are one space).
    const std::span<const assembly::AssemblyBatch> batches = m_set->batches();
    made->batchFirst.assign(1, 0);
    for (const assembly::AssemblyBatch& batch : batches)
        made->batchFirst.push_back(made->batchFirst.back() + batch.view.rowCount());
    const auto elementCount = static_cast<std::size_t>(made->batchFirst.back());
    struct Located
    {
        base::EntityId elementId = 0;
        std::int64_t element = 0;
    };
    base::BudgetedBuffer<Located> located(budget, elementCount, "load basis element IDs");
    for (std::size_t b = 0; b < batches.size(); ++b)
        for (std::int64_t row = 0; row < batches[b].view.rowCount(); ++row)
            located[static_cast<std::size_t>(made->batchFirst[b] + row)] = {batches[b].view.elementId(row),
                                                                            made->batchFirst[b] + row};
    std::sort(located.data(), located.data() + elementCount,
              [](const Located& a, const Located& c) { return a.elementId < c.elementId; });
    const auto elementOf = [&](base::EntityId elementId) -> std::int64_t
    {
        const Located* found =
            std::lower_bound(located.data(), located.data() + elementCount, elementId,
                             [](const Located& entry, base::EntityId id) { return entry.elementId < id; });
        return found != located.data() + elementCount && found->elementId == elementId ? found->element : -1;
    };
    // Node → elements, CSR: the groups of the connectivity (two nodes or more) and the elements of one node.
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    made->offsets = base::BudgetedBuffer<std::int64_t>(budget, nodeCount + 1, "load basis node elements offsets");
    const linalg::NodeGroupList groups = connectivity.groups();
    const std::span<const std::int32_t> groupIds = connectivity.groupElementIds();
    const std::size_t groupCount = groupIds.size();
    for (std::size_t g = 0; g < groupCount; ++g)
        for (const std::int32_t node : groups.group(g))
            ++made->offsets[static_cast<std::size_t>(node) + 1];
    for (const assembly::ElementConnectivity::PointElement& point : connectivity.pointElements())
        ++made->offsets[static_cast<std::size_t>(point.node) + 1];
    for (std::size_t node = 0; node < nodeCount; ++node)
        made->offsets[node + 1] += made->offsets[node];
    made->elements = base::BudgetedBuffer<std::int64_t>(budget, static_cast<std::size_t>(made->offsets[nodeCount]),
                                                        "load basis node elements");
    base::BudgetedBuffer<std::int64_t> next(budget, nodeCount, "load basis node elements cursor");
    std::copy(made->offsets.data(), made->offsets.data() + nodeCount, next.data());
    for (std::size_t g = 0; g < groupCount; ++g)
    {
        const std::int64_t element = elementOf(groupIds[g]);
        for (const std::int32_t node : groups.group(g))
            made->elements[static_cast<std::size_t>(next[static_cast<std::size_t>(node)]++)] = element;
    }
    for (const assembly::ElementConnectivity::PointElement& point : connectivity.pointElements())
        made->elements[static_cast<std::size_t>(next[static_cast<std::size_t>(point.node)]++)] =
            elementOf(point.elementId);
    m_elementsOfNodes = std::move(made);
    return *m_elementsOfNodes;
}

std::vector<std::vector<elements::ElementRange>> LoadBasisRecovery::rowsAtNodes(
    std::span<const std::uint8_t> isNodeTaken, base::MemoryBudget& budget, base::ThreadPool* pool,
    std::uint64_t& pairs) const
{
    const ElementsOfNodes& index = elementsOfNodes(budget, pool);
    const auto elementCount = static_cast<std::size_t>(index.batchFirst.back());
    base::BudgetedBuffer<std::uint8_t> isTaken(budget, elementCount, "load basis elements at the points");
    pairs = 0;
    for (std::size_t node = 0; node < isNodeTaken.size(); ++node)
    {
        if (isNodeTaken[node] == 0)
            continue;
        for (auto k = static_cast<std::size_t>(index.offsets[node]);
             k < static_cast<std::size_t>(index.offsets[node + 1]); ++k)
            if (const std::int64_t element = index.elements[k]; element >= 0)
            {
                isTaken[static_cast<std::size_t>(element)] = 1;
                ++pairs;
            }
    }
    std::vector<std::vector<elements::ElementRange>> rows(index.batchFirst.size() - 1); // per batch
    for (std::size_t b = 0; b + 1 < index.batchFirst.size(); ++b)
    {
        const std::int64_t first = index.batchFirst[b];
        const std::int64_t count = index.batchFirst[b + 1] - first;
        for (std::int64_t row = 0; row < count; ++row)
        {
            if (isTaken[static_cast<std::size_t>(first + row)] == 0)
                continue;
            if (!rows[b].empty() && rows[b].back().last == row)
                rows[b].back().last = row + 1;
            else
                rows[b].push_back({row, row + 1});
        }
    }
    return rows;
}

bool LoadBasisRecovery::hasThermalUnit(std::span<const std::int32_t> units) const
{
    return std::any_of(units.begin(), units.end(),
                       [&](std::int32_t unit)
                       {
                           const UnitPlace place = placeOf(unit);
                           return m_basis->sets[place.set].units[place.column].isThermal();
                       });
}

bool LoadBasisRecovery::isStreamed() const noexcept
{
    return std::any_of(m_basis->sets.begin(), m_basis->sets.end(),
                       [](const KeptLoadBasisSet& set) { return set.streamed != nullptr; });
}

output::ResultArrays LoadBasisRecovery::recoverStreamed(std::span<const std::int32_t> units, bool withDisplacements,
                                                        const output::ElementOutputSelection& selection,
                                                        const output::OutputRequestPlan* gridPointForces,
                                                        base::MemoryBudget& budget) const
{
    const bool asksTables = withDisplacements || selection.stress.isRequested || selection.strain.isRequested ||
                            selection.forces.isRequested;
    if (asksTables || gridPointForces == nullptr || !gridPointForces->isRequested)
        throw base::SolverError(
            "the load basis of the full path streamed keeps its declared quantities alone (its regions are gone after "
            "the solve): no unit's displacements nor element tables; declare what is read before the solve "
            "(the quantities of the load basis request), or make the basis on the full or condensed path");
    output::ResultArrays arrays;
    for (const std::int32_t unit : units)
    {
        const UnitPlace place = placeOf(unit);
        const std::shared_ptr<const KeptStreamedBasis>& kept = m_basis->sets[place.set].streamed;
        if (!kept || place.column >= kept->gridPointForces.size())
            throw base::SolverError(
                "the load basis of the full path streamed keeps the grid point forces of the declared cuts' points "
                "alone, and none were declared; declare the cuts (BasisQuantities(cuts=...))");
        const output::GridPointForceTable& from = kept->gridPointForces[place.column];
        // The rows of the points asked (every kept point when the request takes all).
        std::vector<std::size_t> rows; // bounded by the kept rows
        for (std::size_t r = 0; r < from.rowCount(); ++r)
            if (gridPointForces->isAll ||
                std::binary_search(gridPointForces->members.begin(), gridPointForces->members.end(), from.nodeIds[r]))
                rows.push_back(r);
        output::GridPointForceTable table;
        table.subcase = unit;
        table.analysis = from.analysis;
        table.names = from.names;
        table.nodeIds =
            base::BudgetedBuffer<std::int32_t>(budget, rows.size(), "load basis grid point force grid points");
        table.elementIds =
            base::BudgetedBuffer<std::int32_t>(budget, rows.size(), "load basis grid point force elements");
        table.nameIndices =
            base::BudgetedBuffer<std::int32_t>(budget, rows.size(), "load basis grid point force names");
        table.values = base::BudgetedBuffer<double>(budget, rows.size() * output::GridPointForceTable::COMPONENT_COUNT,
                                                    "load basis grid point force values");
        for (std::size_t k = 0; k < rows.size(); ++k)
        {
            table.nodeIds[k] = from.nodeIds[rows[k]];
            table.elementIds[k] = from.elementIds[rows[k]];
            table.nameIndices[k] = from.nameIndices[rows[k]];
            std::copy_n(from.values.data() + rows[k] * output::GridPointForceTable::COMPONENT_COUNT,
                        output::GridPointForceTable::COMPONENT_COUNT,
                        table.values.data() + k * output::GridPointForceTable::COMPONENT_COUNT);
        }
        arrays.gridPointForceTables.push_back(std::move(table));
    }
    return arrays;
}

base::BudgetedBuffer<double> LoadBasisRecovery::quantities(std::span<const std::int32_t> units,
                                                           const output::QuantityMap& map,
                                                           const output::ElementOutputSelection& selection,
                                                           base::MemoryBudget& budget, base::ThreadPool* pool) const
{
    if (hasThermalUnit(units))
        refuseThermalUnits("declared quantities");
    const std::size_t count = map.quantityCount();
    base::BudgetedBuffer<double> values(budget, units.size() * count, "load basis declared quantities");
    if (isStreamed())
    {
        // The full path streamed made the quantities of the declared map in its downward pass: the same map only.
        for (std::size_t j = 0; j < units.size(); ++j)
        {
            const UnitPlace place = placeOf(units[j]);
            const std::shared_ptr<const KeptStreamedBasis>& kept = m_basis->sets[place.set].streamed;
            if (!kept || kept->quantityCount != count || kept->termCount != map.termCount())
                throw base::SolverError(
                    "the load basis of the full path streamed keeps the quantities declared with it alone (its regions "
                    "are gone after the solve); declare these with the load basis request");
            std::copy_n(kept->quantities.data() + place.column * count, count, values.data() + j * count);
        }
        return values;
    }
    if (map.elements().empty())
        return values;
    // The forces of the map's elements alone (their IDs sorted: QuantityMap::elements).
    output::ElementOutputSelection forces = selection;
    forces.stress.isRequested = false;
    forces.strain.isRequested = false;
    forces.forces.isRequested = true;
    forces.forces.isAll = false;
    forces.forces.members = map.elements();
    // The units recovered together: their tables (a few rows of at most QUANTITY_TABLE_BYTES an element) within a
    // UNIT_BLOCK_SHARE-th of the memory left, as recover()'s own blocks.
    constexpr std::uint64_t QUANTITY_TABLE_BYTES = 1024;
    const std::uint64_t perUnit = static_cast<std::uint64_t>(map.elements().size()) * QUANTITY_TABLE_BYTES;
    const auto block = static_cast<std::size_t>(
        std::clamp<std::uint64_t>(budget.available() / UNIT_BLOCK_SHARE / std::max<std::uint64_t>(perUnit, 1), 1,
                                  std::min<std::uint64_t>(UNIT_BLOCK_MAX, std::max<std::size_t>(units.size(), 1))));
    for (std::size_t first = 0; first < units.size(); first += block)
    {
        const std::size_t taken = std::min(block, units.size() - first);
        const std::span<const std::int32_t> blockUnits = units.subspan(first, taken);
        const output::ResultArrays arrays = recover(blockUnits, false, false, {}, forces, budget, pool);
        // The tables come unit by unit, in the order of the block's units (recover()).
        std::size_t next = 0;
        for (std::size_t j = 0; j < taken; ++j)
        {
            std::size_t end = next;
            while (end < arrays.elementTables.size() && arrays.elementTables[end].subcase == blockUnits[j])
                ++end;
            map.reduce(std::span<const output::ElementResultTable>(arrays.elementTables.data() + next, end - next),
                       std::span<double>(values.data() + (first + j) * count, count));
            next = end;
        }
    }
    return values;
}

std::optional<output::SectionLoadTable> LoadBasisRecovery::sectionLoads(const output::SectionLoadRequest& request,
                                                                        bool isCut, std::span<const std::int32_t> units,
                                                                        base::MemoryBudget& budget,
                                                                        base::DiagnosticSink& sink) const
{
    if (hasThermalUnit(units))
        refuseThermalUnits(isCut ? "cut loads" : "section loads");
    output::SectionLoadRun run(request, *m_snapshot, *m_set, *m_basis->dofMap, units.size(), budget, sink);
    if (sink.hasErrors())
        return std::nullopt;
    std::vector<std::int32_t> elementIds; // bounded by the elements of the sections and cuts
    for (const output::SectionDefinition& section : request.sections)
        for (const output::SectionMember& member : section.members)
            elementIds.push_back(member.elementId);
    for (const output::CutDefinition& cut : request.cuts)
        elementIds.insert(elementIds.end(), cut.elements.begin(), cut.elements.end());
    const std::vector<std::uint8_t> isUnitRead = condensedUnitsRead(false, {}, elementIds);
    UnitDisplacements displacementsOf(*this, units, isUnitRead, budget, nullptr);
    for (std::size_t i = 0; i < units.size(); ++i)
        run.offerCase(units[i], displacementsOf.of(i), nullptr, budget);
    return isCut ? run.takeCutTable() : run.takeSectionTable();
}

} // namespace oos::solutions
