#include "oos/solutions/common/StructureCache.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <utility>

namespace oos::solutions
{

namespace
{

// What a graph entry is keyed by: the graph and its node DOF counts, copied into the entry's budget.
struct GraphKey
{
    linalg::NodeGraph graph;
    base::BudgetedBuffer<std::uint8_t> nodeDofCounts;
};

// What an ordering is keyed by besides its graph: the elimination flags of the solved set (one per g-set DOF) and the
// positions of the grid points (three per grid point, in the basic system).
struct OrderingKey
{
    base::BudgetedBuffer<std::uint8_t> eliminated;
    base::BudgetedBuffer<double> positions;
};

template <typename T>
bool isEqual(std::span<const T> a, std::span<const T> b) noexcept
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}

bool isSameGraph(const GraphKey& key, const linalg::NodeGraph& graph, std::span<const std::uint8_t> counts) noexcept
{
    return key.graph.nodeCount() == graph.nodeCount() && isEqual(key.nodeDofCounts.span(), counts) &&
           isEqual(key.graph.offsets(), graph.offsets()) && isEqual(key.graph.adjacency(), graph.adjacency());
}

// The positions the key of an ordering holds: every grid point's, in the basic system.
base::BudgetedBuffer<double> positionsOf(const assembly::DofMap& dofMap, base::MemoryBudget& budget)
{
    const auto grids = static_cast<std::size_t>(dofMap.gridCount());
    base::BudgetedBuffer<double> positions(budget, 3 * grids, "structure key: positions", base::FOR_OVERWRITE);
    for (std::size_t g = 0; g < grids; ++g)
    {
        const base::Vec3 position = dofMap.positionInBasic(static_cast<std::int32_t>(g));
        for (std::size_t c = 0; c < 3; ++c)
            positions[3 * g + c] = position[c];
    }
    return positions;
}

// Moves the entry at `at` to the front of `entries`.
template <typename Entry>
void moveToFront(std::vector<std::shared_ptr<Entry>>& entries, std::size_t at)
{
    std::rotate(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(at),
                entries.begin() + static_cast<std::ptrdiff_t>(at) + 1);
}

} // namespace

struct StructureCache::OrderingEntry
{
    assembly::DofSet keptSet = assembly::DofSet::L;
    linalg::OrderingMethod method = linalg::OrderingMethod::Automatic;
    std::shared_ptr<const base::KeptObject<OrderingKey>> key;
    std::shared_ptr<KeptOrdering> kept;

    [[nodiscard]] bool matches(assembly::DofSet set, linalg::OrderingMethod orderingMethod,
                               const OrderingKey& other) const noexcept
    {
        return keptSet == set && method == orderingMethod &&
               isEqual(key->object().eliminated.span(), other.eliminated.span()) &&
               isEqual(key->object().positions.span(), other.positions.span());
    }
};

struct StructureCache::GraphEntry
{
    std::shared_ptr<const base::KeptObject<GraphKey>> key;
    std::shared_ptr<const base::KeptObject<linalg::SparsePattern>> pattern; // null until asked for
    std::vector<std::shared_ptr<OrderingEntry>> orderings;                  // the most recently used first
};

linalg::FactorStorage FactorStorageSlot::take(const linalg::SymbolicFactor& symbolic)
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    for (auto held = m_storages.begin(); held != m_storages.end(); ++held)
        if (held->fits(symbolic))
        {
            linalg::FactorStorage taken = std::move(*held);
            m_storages.erase(held);
            return taken;
        }
    return {};
}

void FactorStorageSlot::offer(linalg::FactorStorage storage) noexcept
{
    if (storage.empty())
        return;
    linalg::FactorStorage oldest;
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        try
        {
            m_storages.push_back(std::move(storage));
        }
        catch (const std::bad_alloc&)
        {
            return; // no room to list it: `storage` is freed (keeping it only saves time)
        }
        if (m_storages.size() > KEPT_STORAGE_COUNT)
        {
            oldest = std::move(m_storages.front());
            m_storages.erase(m_storages.begin());
        }
    }
    // `oldest` is freed outside the lock.
}

void FactorStorageSlot::clear() noexcept
{
    std::vector<linalg::FactorStorage> held;
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        held.swap(m_storages);
    }
}

std::uint64_t FactorStorageSlot::bytes() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    std::uint64_t total = 0;
    for (const linalg::FactorStorage& storage : m_storages)
        total += storage.bytes();
    return total;
}

std::shared_ptr<const base::KeptObject<linalg::SolvePlan>> KeptOrdering::plan() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_plan;
}

void KeptOrdering::offerPlan(std::shared_ptr<const base::KeptObject<linalg::SolvePlan>> plan)
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    if (m_plan == nullptr)
        m_plan = std::move(plan);
}

std::uint64_t KeptOrdering::bytes() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_ordering->bytes() + (m_plan != nullptr ? m_plan->bytes() : 0);
}

std::shared_ptr<StructureCache::GraphEntry> StructureCache::findGraph(const linalg::NodeGraph& graph,
                                                                      std::span<const std::uint8_t> nodeDofCounts)
{
    for (std::size_t k = 0; k < m_graphs.size(); ++k)
        if (isSameGraph(m_graphs[k]->key->object(), graph, nodeDofCounts))
        {
            moveToFront(m_graphs, k);
            return m_graphs.front();
        }
    return nullptr;
}

std::shared_ptr<StructureCache::GraphEntry> StructureCache::graphEntry(const linalg::NodeGraph& graph,
                                                                       std::span<const std::uint8_t> nodeDofCounts)
{
    if (std::shared_ptr<GraphEntry> found = findGraph(graph, nodeDofCounts))
        return found;
    auto entry = std::make_shared<GraphEntry>();
    entry->key = base::KeptObject<GraphKey>::build(
        [&](base::MemoryBudget& budget)
        {
            GraphKey key{
                linalg::NodeGraph::fromAdjacency(graph.nodeCount(), graph.offsets(), graph.adjacency(), budget),
                base::BudgetedBuffer<std::uint8_t>(budget, nodeDofCounts.size(), "structure key: DOF counts",
                                                   base::FOR_OVERWRITE)};
            std::copy(nodeDofCounts.begin(), nodeDofCounts.end(), key.nodeDofCounts.data());
            return key;
        });
    m_graphs.insert(m_graphs.begin(), entry);
    if (m_graphs.size() > GRAPH_CAPACITY)
        m_graphs.pop_back();
    return entry;
}

StructurePattern StructureCache::pattern(const linalg::NodeGraph& graph, std::span<const std::uint8_t> nodeDofCounts)
{
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if (const std::shared_ptr<GraphEntry> entry = findGraph(graph, nodeDofCounts); entry && entry->pattern)
        {
            ++m_counts.patternsReused;
            return {entry->pattern, true};
        }
    }
    // Built outside the lock: the analyses of other design points go on meanwhile.
    std::shared_ptr<const base::KeptObject<linalg::SparsePattern>> built =
        base::KeptObject<linalg::SparsePattern>::build([&](base::MemoryBudget& budget)
                                                       { return linalg::SparsePattern(graph, nodeDofCounts, budget); });
    const std::lock_guard<std::mutex> lock(m_mutex);
    const std::shared_ptr<GraphEntry> entry = graphEntry(graph, nodeDofCounts);
    if (entry->pattern != nullptr)
    {
        // Another analysis built the same pattern meanwhile: the first one is kept (the same structure).
        ++m_counts.patternsReused;
        return {entry->pattern, true};
    }
    entry->pattern = std::move(built);
    ++m_counts.patternsBuilt;
    return {entry->pattern, false};
}

StructureOrdering StructureCache::ordering(const linalg::NodeGraph& graph, const assembly::DofMap& dofMap,
                                           const assembly::DofSets& sets, assembly::DofSet keptSet,
                                           linalg::OrderingMethod method, base::ThreadPool* pool,
                                           const base::CancellationToken* cancellation)
{
    // The key of the analysis, in a budget of its own: kept with the ordering when it is built.
    std::shared_ptr<const base::KeptObject<OrderingKey>> key = base::KeptObject<OrderingKey>::build(
        [&](base::MemoryBudget& budget)
        { return OrderingKey{sets.eliminationFlags(keptSet, budget), positionsOf(dofMap, budget)}; });
    base::MemoryBudget countsBudget(base::MemoryBudget::UNLIMITED);
    const base::BudgetedBuffer<std::uint8_t> nodeDofCounts = dofMap.nodeDofCounts(countsBudget);
    const auto findOrdering = [&](GraphEntry& entry) -> std::shared_ptr<KeptOrdering>
    {
        for (std::size_t k = 0; k < entry.orderings.size(); ++k)
            if (entry.orderings[k]->matches(keptSet, method, key->object()))
            {
                moveToFront(entry.orderings, k);
                return entry.orderings.front()->kept;
            }
        return nullptr;
    };
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if (const std::shared_ptr<GraphEntry> entry = findGraph(graph, nodeDofCounts.span()))
            if (std::shared_ptr<KeptOrdering> found = findOrdering(*entry))
            {
                ++m_counts.orderingsReused;
                return {std::move(found), true};
            }
    }
    // Built outside the lock.
    auto built = std::make_shared<KeptOrdering>(
        base::KeptObject<FsetOrdering>::build(
            [&](base::MemoryBudget& budget)
            { return FsetOrdering(graph, dofMap, sets, keptSet, method, budget, pool, cancellation); }),
        m_storageSlot);
    const std::lock_guard<std::mutex> lock(m_mutex);
    const std::shared_ptr<GraphEntry> entry = graphEntry(graph, nodeDofCounts.span());
    if (std::shared_ptr<KeptOrdering> found = findOrdering(*entry))
    {
        // Another analysis built the same ordering meanwhile: the first one is kept (the same structure).
        ++m_counts.orderingsReused;
        return {std::move(found), true};
    }
    auto ordering = std::make_shared<OrderingEntry>();
    ordering->keptSet = keptSet;
    ordering->method = method;
    ordering->key = std::move(key);
    ordering->kept = built;
    entry->orderings.insert(entry->orderings.begin(), std::move(ordering));
    if (entry->orderings.size() > ORDERING_CAPACITY)
        entry->orderings.pop_back();
    ++m_counts.orderingsBuilt;
    // A new ordering: the structure changed, the panels' storages held would serve no factorisation soon.
    m_storageSlot->clear();
    return {std::move(built), false};
}

void StructureCache::countPlan(bool isReused)
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    ++(isReused ? m_counts.plansReused : m_counts.plansBuilt);
}

void StructureCache::countStorage(bool isReused)
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    ++(isReused ? m_counts.storagesReused : m_counts.storagesAllocated);
}

std::shared_ptr<const base::KeptObject<reduction::UnitBoundaryData>> StructureCache::unitBoundary(
    const reduction::CacheKey& key)
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    const auto found = m_unitBoundaries.find(key.hash());
    if (found == m_unitBoundaries.end() || !key.matches(found->second.key))
        return nullptr;
    found->second.lastUse = ++m_unitBoundaryUses;
    ++m_counts.unitBoundariesReused;
    return found->second.data;
}

void StructureCache::keepUnitBoundary(const reduction::CacheKey& key,
                                      std::shared_ptr<const base::KeptObject<reduction::UnitBoundaryData>> data)
{
    if (data == nullptr)
        return;
    const std::lock_guard<std::mutex> lock(m_mutex);
    const std::uint64_t hash = key.hash();
    if (const auto found = m_unitBoundaries.find(hash); found != m_unitBoundaries.end())
    {
        m_unitBoundaryBytes -= found->second.data->bytes();
        m_unitBoundaries.erase(found);
    }
    // The least recently used dropped first (a linear scan per drop: rare, the cache holding what one model's units
    // need).
    while (!m_unitBoundaries.empty() && (m_unitBoundaries.size() >= UNIT_BOUNDARY_CAPACITY ||
                                         m_unitBoundaryBytes + data->bytes() > UNIT_BOUNDARY_MAXIMUM_BYTES))
    {
        auto oldest = m_unitBoundaries.begin();
        for (auto entry = m_unitBoundaries.begin(); entry != m_unitBoundaries.end(); ++entry)
            if (entry->second.lastUse < oldest->second.lastUse)
                oldest = entry;
        m_unitBoundaryBytes -= oldest->second.data->bytes();
        m_unitBoundaries.erase(oldest);
    }
    if (data->bytes() > UNIT_BOUNDARY_MAXIMUM_BYTES)
        return;
    m_unitBoundaryBytes += data->bytes();
    m_unitBoundaries[hash] = UnitBoundaryEntry{key, std::move(data), ++m_unitBoundaryUses};
    ++m_counts.unitBoundariesKept;
}

std::uint64_t StructureCache::bytes() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    std::uint64_t total = m_storageSlot->bytes() + m_unitBoundaryBytes;
    for (const std::shared_ptr<GraphEntry>& graph : m_graphs)
    {
        total += graph->key->bytes() + (graph->pattern != nullptr ? graph->pattern->bytes() : 0);
        for (const std::shared_ptr<OrderingEntry>& ordering : graph->orderings)
            total += ordering->key->bytes() + ordering->kept->bytes();
    }
    return total;
}

StructureCacheCounts StructureCache::counts() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_counts;
}

std::shared_ptr<const FsetOrdering> orderingOf(StructureCache* structures, const linalg::NodeGraph& graph,
                                               const assembly::DofMap& dofMap, const assembly::DofSets& sets,
                                               assembly::DofSet keptSet, linalg::OrderingMethod method,
                                               base::MemoryBudget& budget, base::ThreadPool* pool,
                                               const base::CancellationToken* cancellation, bool* isReused)
{
    if (structures == nullptr)
    {
        if (isReused != nullptr)
            *isReused = false;
        return std::make_shared<const FsetOrdering>(graph, dofMap, sets, keptSet, method, budget, pool, cancellation);
    }
    const StructureOrdering taken = structures->ordering(graph, dofMap, sets, keptSet, method, pool, cancellation);
    if (isReused != nullptr)
        *isReused = taken.isReused;
    return taken.kept->ordering().lend(budget, "ordering and symbolic factor (kept)");
}

void StructureCache::clear()
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_graphs.clear();
    m_storageSlot->clear();
    m_unitBoundaries.clear();
    m_unitBoundaryBytes = 0;
    m_counts = StructureCacheCounts{};
}

} // namespace oos::solutions
