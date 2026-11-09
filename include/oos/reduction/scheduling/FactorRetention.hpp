#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace oos::reduction
{

// What happened to the interior factor of a unit after its condensation.
enum class RetentionDecision : std::uint8_t
{
    Kept,              // kept in memory for its recovery
    ReleasedUnneeded,  // released: the unit is not recovered, nothing reads it
    ReleasedForBudget, // released: the budget needs its memory for the rest of the analysis; computed again at recovery
};

[[nodiscard]] const char* nameOf(RetentionDecision decision) noexcept;

// The decision on one unit's interior factor, as the report names it (a release is always named in the report).
struct RetentionRecord
{
    std::int64_t unitId = 0;
    std::uint64_t factorBytes = 0;
    RetentionDecision decision = RetentionDecision::Kept;
    std::string reason;
};

// Which interior factors stay in memory after their unit's condensation (kept while the memory budget allows it, it
// then serves recovery directly; released otherwise, the release being named in the report; this never changes a
// result):
// - a unit that is not recovered releases its factor (no recovery reads it);
// - a recovered unit keeps it when the budget's free bytes, its factor still held, cover the reserve of the rest of the
//   analysis (the units not yet condensed, the interface, the recoveries, the cases and the results:
//   UnitScheduler::reserve()); otherwise it is released, and its recovery factorises the interior again, to the same
//   bits (PartialFactor is deterministic; the condensation checks the K* of the new factor against its own).
// The decisions are taken in a fixed order (the units of a wave in the scheduler's order, after the wave), from the
// budget's accounting at that moment, so that they are the same in every run with the same budget; a kept factor stays
// to the end of the analysis.
//
// Thread-safety: not for concurrent use. Ownership: owns its records.
class FactorRetention
{
public:
    // Decides for a unit whose factor of `factorBytes` is held now: `availableBytes` the budget's free bytes now,
    // `reserveBytes` what the rest of the analysis needs beyond what is held now.
    RetentionDecision decide(std::int64_t unitId, std::uint64_t factorBytes, bool isRecovered,
                             std::uint64_t availableBytes, std::uint64_t reserveBytes);

    [[nodiscard]] const std::vector<RetentionRecord>& records() const noexcept { return m_records; }
    [[nodiscard]] std::uint64_t keptBytes() const noexcept { return m_keptBytes; }
    [[nodiscard]] std::int64_t keptCount() const noexcept { return m_keptCount; }
    // The recovered units whose factor was released for the budget (the report names them).
    [[nodiscard]] std::vector<std::int64_t> releasedForBudget() const;

private:
    std::vector<RetentionRecord> m_records; // one per unit factorised, bounded by the units
    std::uint64_t m_keptBytes = 0;
    std::int64_t m_keptCount = 0;
};

} // namespace oos::reduction
