#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/supernodal/PartialFactor.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace oos::reduction
{

// The interior factor of a unit kept beyond the analysis that made it (between the analyses of a process), with a
// copy of the equations it was made from, so that
// a later analysis takes it only for the same equations (KeptInteriorFactors::take): the unit (the hash of its element
// IDs), its boundary labels (the rows of K*), its interior count and its condensed matrix (UnitSystem::matrix(), in the
// order of its equations: offsets, rows and values).
//
// Memory: the factor's buffers are reserved in the budget of the analysis that factorised it, which factorMemoryOwner
// keeps alive (declared first: given back after the factor); the copies of the equations in a budget of their own
// (declared second). bytes is what it holds: the factor (StaticCondensation::interiorFactorBytes) and the copies.
//
// Thread-safety: a value. Ownership: owns the factor and the copies; movable, not copyable. Its move assignment frees
// what it held first, each buffer before the budget it is reserved in (a member-wise assignment would free the budgets
// first: they are declared before the buffers).
struct KeptInteriorFactor
{
    KeptInteriorFactor() = default;
    KeptInteriorFactor(KeptInteriorFactor&&) noexcept = default;
    KeptInteriorFactor& operator=(KeptInteriorFactor&& other) noexcept;
    KeptInteriorFactor(const KeptInteriorFactor&) = delete;
    KeptInteriorFactor& operator=(const KeptInteriorFactor&) = delete;
    ~KeptInteriorFactor() = default;

    std::shared_ptr<base::MemoryBudget> factorMemoryOwner;
    std::unique_ptr<base::MemoryBudget> equationsBudget;
    std::uint64_t unitIdentity = 0;
    std::int64_t unitId = 0; // the plan's ID of the analysis that kept it (the reports only)
    base::BudgetedBuffer<UnitDofLabel> boundaryLabels;
    std::int32_t interiorDofCount = 0;
    base::BudgetedBuffer<std::int64_t> columnOffsets;
    base::BudgetedBuffer<std::int32_t> rowIndices;
    base::BudgetedBuffer<double> values;
    // The shape of the symbolic factor of the equations (a guard beside the matrix: the ordering is a function of it).
    std::int32_t supernodeCount = 0;
    std::int64_t panelValueCount = 0;
    linalg::PartialFactor factor;
    std::uint64_t bytes = 0;
};

// The factor of `system`'s condensation kept for later analyses: `factor` (made from system.matrix(), its buffers
// reserved in *factorMemoryOwner) with a copy of the equations, `factorBytes` its bytes. Throws std::invalid_argument
// when the system has no equations or the factor is not kept, MemoryBudgetError never (the copies' budget is
// unlimited).
[[nodiscard]] KeptInteriorFactor keepInteriorFactor(std::uint64_t unitIdentity, std::int64_t unitId,
                                                    const UnitSystem& system, linalg::PartialFactor factor,
                                                    std::uint64_t factorBytes,
                                                    std::shared_ptr<base::MemoryBudget> factorMemoryOwner);

// What a KeptInteriorFactors did since it was made (or cleared).
struct KeptInteriorFactorCounts
{
    std::int64_t offered = 0;    // factors offered
    std::int64_t taken = 0;      // taken by a unit of the same equations
    std::int64_t replaced = 0;   // dropped for a factor of the same unit offered after it
    std::int64_t mismatched = 0; // dropped when its unit came back with other equations
    std::int64_t evicted = 0;    // dropped for the limit, the earliest offered first
    std::int64_t refused = 0;    // offered larger than the limit (none kept)
};

// The interior factors of the condensed units kept between the analyses of a process: a static analysis offers the
// factors it still holds at its end; a
// later analysis (the same Solver, or a new one built from a deck whose loads or masses only changed) takes the factor
// of a unit whose equations are those it was made from, compared in full (take): the unit, its boundary labels, its
// interior count, its condensed matrix to the bit, the symbolic shape of the factor, its dense kernel family and pivot
// policy. The factorisation is deterministic (linalg::PartialFactor), so the factor taken is the one the unit would
// factorise again, and every result has the bits of a cold run.
//
// Not the condensation cache: memory of the process, never written to disk. It holds at most limitBytes()
// (an eighth of the physical memory by default), the earliest offered dropped first; one factor per unit (a later
// offer of the same unit replaces it); a factor taken leaves it (an analysis holds it, and offers it again at its end).
//
// Thread-safety: every member is thread-safe (a mutex; the comparison of take() runs under it). Ownership: owns the
// factors it holds.
class KeptInteriorFactors
{
public:
    // The default limit: this fraction of the installed physical memory (base::MemoryInfo::totalBytes), so that a few
    // worker processes of a loop each keep their units' factors without taking the machine.
    static constexpr std::uint64_t DEFAULT_LIMIT_DIVISOR = 8;

    // limitBytes 0: none is kept.
    explicit KeptInteriorFactors(std::uint64_t limitBytes) : m_limitBytes(limitBytes) {}
    KeptInteriorFactors(const KeptInteriorFactors&) = delete;
    KeptInteriorFactors& operator=(const KeptInteriorFactors&) = delete;
    KeptInteriorFactors(KeptInteriorFactors&&) = delete;
    KeptInteriorFactors& operator=(KeptInteriorFactors&&) = delete;
    ~KeptInteriorFactors() = default;

    // DEFAULT_LIMIT_DIVISOR's share of this machine's memory (0 when the memory is unknown).
    [[nodiscard]] static std::uint64_t defaultLimitBytes();

    // Keeps `factor` (an earlier one of its unit replaced; the earliest offered dropped beyond the limit; refused when
    // it alone exceeds the limit).
    void offer(KeptInteriorFactor factor);
    // The factor of the unit `unitIdentity` when it was made from `system`'s equations with the kernel family `kernel`
    // and the pivot policy `policy`, taken (no longer held); none otherwise (a factor of that unit with other equations
    // is dropped: its unit changed). O(the unit's matrix) for the comparison.
    [[nodiscard]] std::optional<KeptInteriorFactor> take(std::uint64_t unitIdentity, const UnitSystem& system,
                                                         linalg::KernelKind kernel, const linalg::PivotPolicy& policy);

    [[nodiscard]] std::uint64_t limitBytes() const;
    // A new limit (the earliest offered dropped down to it).
    void setLimitBytes(std::uint64_t limitBytes);
    [[nodiscard]] std::uint64_t bytes() const;
    [[nodiscard]] std::size_t count() const;
    [[nodiscard]] KeptInteriorFactorCounts counts() const;
    // Drops every factor held (and the counts).
    void clear();

private:
    // Drops the earliest offered until the bytes held are at most the limit. Requires the lock.
    void evictBeyondLimit();

    mutable std::mutex m_mutex;
    std::uint64_t m_limitBytes = 0;         // guarded by m_mutex
    std::vector<KeptInteriorFactor> m_kept; // guarded by m_mutex; the earliest offered first, one per unit
    std::uint64_t m_bytes = 0;              // guarded by m_mutex
    KeptInteriorFactorCounts m_counts;      // guarded by m_mutex
};

} // namespace oos::reduction
