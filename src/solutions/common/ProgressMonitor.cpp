#include "oos/solutions/common/ProgressMonitor.hpp"

#include <utility>

namespace oos::solutions
{

base::ProgressCounter& ProgressMonitor::beginStage(std::string stage, std::uint64_t total)
{
    auto counter = std::make_shared<base::ProgressCounter>(std::move(stage), total);
    base::ProgressCounter& started = *counter;
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_previous = std::move(m_current);
        m_current = std::move(counter);
    }
    if (m_stageListener)
        m_stageListener(started.stage());
    return started;
}

ProgressSnapshot ProgressMonitor::poll() const
{
    std::shared_ptr<base::ProgressCounter> current;
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        current = m_current;
    }
    if (!current)
        return {};
    return {current->stage(), current->done(), current->total()};
}

} // namespace oos::solutions
