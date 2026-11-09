#include "oos/base/threading/TaskGraph.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::base
{

TaskGraph::TaskId TaskGraph::add(std::function<void()> task, const std::vector<TaskId>& dependencies)
{
    std::size_t wave = 0;
    for (const TaskId dependency : dependencies)
    {
        if (dependency >= m_tasks.size())
            throw std::invalid_argument(
                concatenate({"TaskGraph::add: expected dependencies among the ", std::to_string(m_tasks.size()),
                             " tasks added before, found the task ID ", std::to_string(dependency)}));
        wave = std::max(wave, m_wave[dependency] + 1);
    }
    m_tasks.push_back(std::move(task));
    m_wave.push_back(wave);
    return m_tasks.size() - 1;
}

void TaskGraph::run(ThreadPool* pool, const CancellationToken* cancellation) const
{
    if (m_tasks.empty())
        return;
    const std::size_t waveCount = *std::max_element(m_wave.begin(), m_wave.end()) + 1;
    std::vector<std::vector<TaskId>> waves(waveCount);
    for (TaskId id = 0; id < m_tasks.size(); ++id)
        waves[m_wave[id]].push_back(id);
    for (const std::vector<TaskId>& wave : waves)
    {
        // One task per piece: the tasks of a wave are independent, and each is one piece of work.
        parallelFor(
            pool, wave.size(), 1,
            [&](std::size_t first, std::size_t last)
            {
                for (std::size_t index = first; index < last; ++index)
                    m_tasks[wave[index]]();
            },
            cancellation);
    }
}

} // namespace oos::base
