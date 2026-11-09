#include "oos/reduction/scheduling/FactorRetention.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

namespace oos::reduction
{

const char* nameOf(RetentionDecision decision) noexcept
{
    switch (decision)
    {
    case RetentionDecision::Kept:
        return "kept";
    case RetentionDecision::ReleasedUnneeded:
        return "released (not recovered)";
    case RetentionDecision::ReleasedForBudget:
        return "released (memory budget)";
    }
    return "";
}

RetentionDecision FactorRetention::decide(std::int64_t unitId, std::uint64_t factorBytes, bool isRecovered,
                                          std::uint64_t availableBytes, std::uint64_t reserveBytes)
{
    RetentionRecord record;
    record.unitId = unitId;
    record.factorBytes = factorBytes;
    if (!isRecovered)
    {
        record.decision = RetentionDecision::ReleasedUnneeded;
        record.reason = "the unit is not recovered: no recovery reads its factor";
    }
    else if (availableBytes >= reserveBytes)
    {
        record.decision = RetentionDecision::Kept;
        record.reason = base::concatenate({"kept for its recovery: ", base::formatBytes(availableBytes),
                                           " free with it held cover the ", base::formatBytes(reserveBytes),
                                           " the rest of the analysis needs"});
        m_keptBytes += factorBytes;
        ++m_keptCount;
    }
    else
    {
        record.decision = RetentionDecision::ReleasedForBudget;
        record.reason = base::concatenate(
            {"released: the rest of the analysis needs ", base::formatBytes(reserveBytes), " and ",
             base::formatBytes(availableBytes),
             " are free with it held; its recovery factorises the interior again, to the same bits (raise memory= to "
             "keep it)"});
    }
    m_records.push_back(record);
    return record.decision;
}

std::vector<std::int64_t> FactorRetention::releasedForBudget() const
{
    std::vector<std::int64_t> units;
    for (const RetentionRecord& record : m_records)
        if (record.decision == RetentionDecision::ReleasedForBudget)
            units.push_back(record.unitId);
    return units;
}

} // namespace oos::reduction
