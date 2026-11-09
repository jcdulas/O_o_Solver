#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/base/memory/KeptObject.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/reduction/cache/CacheKey.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace oos::solutions
{

// What the analyses took from a StructureCache and built for it, counted since it was made (or cleared).
struct StructureCacheCounts
{
    std::int64_t patternsReused = 0;
    std::int64_t patternsBuilt = 0;
    std::int64_t orderingsReused = 0;
    std::int64_t orderingsBuilt = 0;
    std::int64_t plansReused = 0;
    std::int64_t plansBuilt = 0;
    // The factorisations that took the panels' storage of an earlier factor or allocated their own.
    std::int64_t storagesReused = 0;
    std::int64_t storagesAllocated = 0;
    // The condensed units built without their equations from the boundary data kept for their cache key, and the
    // boundary data kept.
    std::int64_t unitBoundariesReused = 0;
    std::int64_t unitBoundariesKept = 0;
};

// The panels' storage of the factors of a StructureCache's orderings that were freed, for the next factorisations of
// the same shapes: what an analysis's factors gave back when it went, so that the next analysis of the same structure
// (a design loop's next point) writes its panels in memory already mapped. It holds at most KEPT_STORAGE_COUNT storages
// (the oldest freed first), and none of a structure that changed: the cache frees them when it builds a new ordering
// (StructureCache::ordering), and on clear(). Shared by the cache and its kept orderings (an analysis may outlive the
// cache's clearing).
//
// Thread-safety: every member is thread-safe (a mutex). Ownership: owns the storages it holds.
class FactorStorageSlot
{
public:
    // A run's constraint sets, each factorised once (the orderings a cache keeps per graph).
    static constexpr std::size_t KEPT_STORAGE_COUNT = 8;

    // A storage held that fits a factorisation of `symbolic` (taken: no longer held), or an empty one.
    [[nodiscard]] linalg::FactorStorage take(const linalg::SymbolicFactor& symbolic);
    // Holds `storage` (beyond KEPT_STORAGE_COUNT the oldest is freed).
    void offer(linalg::FactorStorage storage) noexcept;
    // Frees the storages held.
    void clear() noexcept;
    [[nodiscard]] std::uint64_t bytes() const;

private:
    mutable std::mutex m_mutex;
    std::vector<linalg::FactorStorage> m_storages; // guarded by m_mutex; the oldest first
};

// The ordering of a solved set kept by a StructureCache: its FsetOrdering (the numbering of the equations and the
// symbolic factor), the plan of the solves of its first positive definite factorisation, which every later
// factorisation of the same symbolic factor takes (FactorOptions::keptSolvePlan), and the cache's slot of a freed
// factor's panels (FactorStorageSlot), which a later factorisation of the same shape takes (FactorOptions::storage).
//
// Thread-safety: the ordering is immutable; the plan is offered once, under a mutex, the first offer kept; the slot is
// thread-safe. Ownership: shares its kept objects and the slot.
class KeptOrdering
{
public:
    KeptOrdering(std::shared_ptr<const base::KeptObject<FsetOrdering>> ordering,
                 std::shared_ptr<FactorStorageSlot> storageSlot)
        : m_ordering(std::move(ordering)), m_storageSlot(std::move(storageSlot))
    {
    }

    [[nodiscard]] const base::KeptObject<FsetOrdering>& ordering() const noexcept { return *m_ordering; }
    // The plan of the solves, or null until a factorisation offered one.
    [[nodiscard]] std::shared_ptr<const base::KeptObject<linalg::SolvePlan>> plan() const;
    // Keeps `plan` unless one was offered before.
    void offerPlan(std::shared_ptr<const base::KeptObject<linalg::SolvePlan>> plan);
    // The cache's slot of a freed factor's panels.
    [[nodiscard]] FactorStorageSlot& storageSlot() const noexcept { return *m_storageSlot; }
    // The bytes the ordering and the plan hold.
    [[nodiscard]] std::uint64_t bytes() const;

private:
    std::shared_ptr<const base::KeptObject<FsetOrdering>> m_ordering;
    std::shared_ptr<FactorStorageSlot> m_storageSlot;
    mutable std::mutex m_mutex;
    std::shared_ptr<const base::KeptObject<linalg::SolvePlan>> m_plan; // guarded by m_mutex
};

// An ordering taken from a StructureCache, and whether an earlier analysis had built it.
struct StructureOrdering
{
    std::shared_ptr<KeptOrdering> kept;
    bool isReused = false;
};

// A sparse pattern taken from a StructureCache, and whether an earlier analysis had built it.
struct StructurePattern
{
    std::shared_ptr<const base::KeptObject<linalg::SparsePattern>> kept;
    bool isReused = false;
};

// The structure of the analyses of one model kept between them (reuse across design points): what depends only on the
// node graph and the solved sets of a constraint set, so that a new design point that changes no topology (a sizing
// change, an analysis run again after a change of values) is factorised numerically only, with the bits of a fresh
// analysis:
// - the sparse pattern of a node graph and its node DOF counts (linalg::SparsePattern), keyed by the graph and the
//   counts;
// - the ordering and the symbolic factor of a solved set (FsetOrdering) and the plan of the solves (SolvePlan), keyed
//   besides by the solved set (its elimination flags: the SPCs, the m set, AUTOSPC and the r set), the positions of
//   the grid points (the ordering's geometric dissection reads them: a shape variable builds it again) and the ordering
//   method.
// - the panels' storage of the factors freed (FactorStorageSlot): a factorisation of the same shape writes
//   its panels in that memory, mapped already, instead of a fresh allocation's pages;
// - the boundary data of the condensed units (reduction::UnitBoundaryData), keyed by their cache key
//   (reduction::CacheKey, compared in full: CacheKey::matches): what a unit taken from the condensation cache needs
//   beside its equations, so that a later analysis builds none for it (the condensation cache keeps only the reduced
//   units; this is the analysis' structure, in memory only).
// Every key is compared exactly with what the new analysis built (the graph, the flags and the positions as they are,
// bit for bit): any change of connectivity, constraints, singular DOFs or positions builds the structure again; the
// ordering, the symbolic analysis and the solve plan are deterministic functions of their key (the orderings depend
// on the graph, the weights and the coordinates only, whatever the threads: ConstrainedOrdering), so what is kept is
// what a fresh analysis would build.
//
// Memory: each kept object lives in a budget of its own (base::KeptObject); an analysis that takes one reserves its
// bytes in its own budget for as long as it holds it (KeptObject::lend), as if it had built it; bytes() is what the
// cache holds, the panels' storage of the slot included (a factorisation that takes it reserves its bytes in its own
// budget, as a new allocation's). At most GRAPH_CAPACITY graphs and ORDERING_CAPACITY orderings per graph are kept, the
// least recently used dropped first (what an analysis still holds lives on with it).
//
// Thread-safety: every member is thread-safe (the analyses of a batch of design points share one cache); a structure
// is built outside the lock, and two analyses that build the same one at once keep the first. Ownership: owns the
// kept structures (shared with the analyses that hold them).
//
// PERFORMANCE: the ordering, the symbolic factor, the pattern and the solve plan built once per key, compared exactly
// (the comparison is linear in the graph, milliseconds at 1 M DOF). Not kept: the map of Kgg's blocks to the CSC values
// (220 to 330 MB at 1 M DOF for the 0.17 s of the conversion, which runs on the pool instead).
// Measured: the ordering and the symbolic factor of a new point of the 1 M DOF plate 0.15 -> 0.003 s, 97 MB kept; of
// the 250 000 DOF assembly 0.034 -> 0.001 s, 23 MB kept.
class StructureCache
{
public:
    // The graphs kept: the constraint sets of a run share one graph but for the m-set couplings of their MPC sets.
    static constexpr std::size_t GRAPH_CAPACITY = 4;
    // The orderings kept per graph: the solved sets of a run (its SPC sets, the f set of its eigenvalue analyses).
    static constexpr std::size_t ORDERING_CAPACITY = 8;
    // The units' boundary data kept: the units of a large model (10 000) and their bytes (a few kB a unit
    // of a few thousand DOFs; the singularity rows of its interior points the most).
    static constexpr std::size_t UNIT_BOUNDARY_CAPACITY = 20'000;
    static constexpr std::uint64_t UNIT_BOUNDARY_MAXIMUM_BYTES = std::uint64_t{256} << 20U;

    StructureCache() = default;
    StructureCache(const StructureCache&) = delete;
    StructureCache& operator=(const StructureCache&) = delete;
    StructureCache(StructureCache&&) = delete;
    StructureCache& operator=(StructureCache&&) = delete;
    ~StructureCache() = default;

    // The sparse pattern of `graph` with `nodeDofCounts` (one per node of the graph): the kept one built from an equal
    // graph and counts, else built in a budget of its own and kept. Throws as SparsePattern.
    [[nodiscard]] StructurePattern pattern(const linalg::NodeGraph& graph, std::span<const std::uint8_t> nodeDofCounts);

    // The ordering of the set `keptSet` of `sets` on `graph` (FsetOrdering's arguments): the kept one whose key is
    // equal, else built in a budget of its own and kept. Throws as FsetOrdering.
    [[nodiscard]] StructureOrdering ordering(const linalg::NodeGraph& graph, const assembly::DofMap& dofMap,
                                             const assembly::DofSets& sets, assembly::DofSet keptSet,
                                             linalg::OrderingMethod method, base::ThreadPool* pool = nullptr,
                                             const base::CancellationToken* cancellation = nullptr);

    // Counts a factorisation that took the kept plan of its ordering (isReused) or built it (FsetSolver).
    void countPlan(bool isReused);
    // Counts a factorisation that took the panels' storage of the slot (isReused) or allocated its own (FsetSolver).
    void countStorage(bool isReused);

    // The boundary data kept for a unit's cache key (counted as reused), or null.
    [[nodiscard]] std::shared_ptr<const base::KeptObject<reduction::UnitBoundaryData>> unitBoundary(
        const reduction::CacheKey& key);
    // Keeps the boundary data of a unit's cache key (an earlier one of the key replaced), the least recently used
    // dropped beyond UNIT_BOUNDARY_CAPACITY entries or UNIT_BOUNDARY_MAXIMUM_BYTES.
    void keepUnitBoundary(const reduction::CacheKey& key,
                          std::shared_ptr<const base::KeptObject<reduction::UnitBoundaryData>> data);

    // The bytes the kept structures hold.
    [[nodiscard]] std::uint64_t bytes() const;
    [[nodiscard]] StructureCacheCounts counts() const;
    // Drops every kept structure (what an analysis still holds lives on with it) and the counts.
    void clear();

private:
    struct GraphEntry;
    struct OrderingEntry;

    // The entry of an equal graph and counts, moved to the front, or null. Requires the lock.
    [[nodiscard]] std::shared_ptr<GraphEntry> findGraph(const linalg::NodeGraph& graph,
                                                        std::span<const std::uint8_t> nodeDofCounts);
    // The entry of `graph` and its counts, made when absent. Requires the lock.
    [[nodiscard]] std::shared_ptr<GraphEntry> graphEntry(const linalg::NodeGraph& graph,
                                                         std::span<const std::uint8_t> nodeDofCounts);

    mutable std::mutex m_mutex;
    std::vector<std::shared_ptr<GraphEntry>> m_graphs; // the most recently used first
    StructureCacheCounts m_counts;
    // The units' boundary data by the hash of their cache key, with the key and the use counter of their last use.
    struct UnitBoundaryEntry
    {
        reduction::CacheKey key;
        std::shared_ptr<const base::KeptObject<reduction::UnitBoundaryData>> data;
        std::uint64_t lastUse = 0;
    };
    std::map<std::uint64_t, UnitBoundaryEntry> m_unitBoundaries;
    std::uint64_t m_unitBoundaryBytes = 0;
    std::uint64_t m_unitBoundaryUses = 0;
    std::shared_ptr<FactorStorageSlot> m_storageSlot = std::make_shared<FactorStorageSlot>();
};

// The ordering of `keptSet` (FsetOrdering's arguments): taken from `structures` when given, lent with its bytes
// reserved in `budget` while the pointer lives (and `isReused` set when an earlier analysis built it), else built in
// `budget`. Throws as FsetOrdering.
[[nodiscard]] std::shared_ptr<const FsetOrdering> orderingOf(
    StructureCache* structures, const linalg::NodeGraph& graph, const assembly::DofMap& dofMap,
    const assembly::DofSets& sets, assembly::DofSet keptSet, linalg::OrderingMethod method, base::MemoryBudget& budget,
    base::ThreadPool* pool, const base::CancellationToken* cancellation, bool* isReused = nullptr);

} // namespace oos::solutions
