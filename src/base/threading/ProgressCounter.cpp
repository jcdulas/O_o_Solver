#include "oos/base/threading/ProgressCounter.hpp"

namespace oos::base
{

bool ProgressCounter::isReportDue(std::chrono::steady_clock::time_point now)
{
    const bool isEnd = done() >= m_total;
    if (isEnd && !m_hasReportedEnd)
    {
        m_hasReportedEnd = true;
        m_hasReported = true;
        m_lastReport = now;
        return true;
    }
    if (!m_hasReported || now - m_lastReport >= MIN_REPORT_INTERVAL)
    {
        if (isEnd)
            return false;
        m_hasReported = true;
        m_lastReport = now;
        return true;
    }
    return false;
}

} // namespace oos::base
