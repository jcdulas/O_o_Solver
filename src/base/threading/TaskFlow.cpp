#include "oos/base/threading/TaskFlow.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OperationCancelledError.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::base
{
namespace
{

using Clock = std::chrono::steady_clock;

// The task the calling thread runs, for the earliest start of the tasks it adds.
struct RunningTask
{
    const TaskFlow* flow = nullptr;
    double readyAt = 0.0;
    Clock::time_point started;
};
thread_local RunningTask runningTask;

} // namespace

TaskFlow::TaskId TaskFlow::add(Body body, std::span<const TaskId> dependencies, std::int64_t priority)
{
    const std::lock_guard lock(m_mutex);
    const TaskId id = m_tasks.size();
    for (const TaskId dependency : dependencies)
        if (dependency >= id)
            throw std::invalid_argument(
                concatenate({"TaskFlow::add: expected dependencies among the ", std::to_string(id),
                             " tasks added before, found the task ID ", std::to_string(dependency)}));
    Task& task = m_tasks.emplace_back();
    task.body = std::move(body);
    task.priority = priority;
    if (runningTask.flow == this)
        task.readyAt = runningTask.readyAt + std::chrono::duration<double>(Clock::now() - runningTask.started).count();
    bool isSkipped = false;
    for (const TaskId dependency : dependencies)
    {
        Task& before = m_tasks[dependency];
        if (before.state == State::Succeeded)
        {
            task.readyAt = std::max(task.readyAt, before.finishAt);
            continue;
        }
        if (before.state == State::Failed || before.state == State::Skipped)
        {
            isSkipped = true;
            continue;
        }
        before.successors.push_back(id);
        ++task.remaining;
    }
    if (isSkipped)
    {
        // Its dependencies that are still to run keep it as a successor: being skipped, it is never made ready.
        task.state = State::Skipped;
        task.body = nullptr;
    }
    else if (task.remaining == 0)
        makeReady(id);
    return id;
}

void TaskFlow::makeReady(TaskId task)
{
    m_tasks[task].state = State::Ready;
    m_ready.push({m_tasks[task].priority, task});
    m_changed.notify_one();
}

void TaskFlow::skipDependents(TaskId task)
{
    std::vector<TaskId> pending(m_tasks[task].successors); // bounded by the flow
    while (!pending.empty())
    {
        const TaskId next = pending.back();
        pending.pop_back();
        Task& dependent = m_tasks[next];
        if (dependent.state == State::Skipped)
            continue;
        dependent.state = State::Skipped;
        dependent.body = nullptr;
        pending.insert(pending.end(), dependent.successors.begin(), dependent.successors.end());
    }
}

void TaskFlow::finish(TaskId task, bool hasSucceeded)
{
    Task& done = m_tasks[task];
    done.state = hasSucceeded ? State::Succeeded : State::Failed;
    done.body = nullptr;
    if (!hasSucceeded)
        skipDependents(task);
    else
        for (const TaskId next : done.successors)
        {
            Task& successor = m_tasks[next];
            successor.readyAt = std::max(successor.readyAt, done.finishAt);
            --successor.remaining;
            if (successor.remaining == 0 && successor.state == State::Waiting)
                makeReady(next);
        }
    done.successors.clear();
}

void TaskFlow::work(const CancellationToken* cancellation)
{
    std::unique_lock lock(m_mutex);
    while (true)
    {
        if (!m_isStopping && cancellation != nullptr && cancellation->isCancelled())
            m_isStopping = true;
        if (!m_isStopping && !m_ready.empty())
        {
            const TaskId id = m_ready.top().id;
            m_ready.pop();
            Task& task = m_tasks[id];
            task.state = State::Running;
            Body body = std::move(task.body);
            ++m_running;
            const RunningTask outer = runningTask;
            runningTask = {this, task.readyAt, Clock::now()};
            lock.unlock();
            bool hasSucceeded = false;
            std::exception_ptr failure;
            try
            {
                hasSucceeded = body();
            }
            catch (...)
            {
                failure = std::current_exception();
            }
            body = nullptr; // what the body captured goes before the flow moves on
            const double seconds = std::chrono::duration<double>(Clock::now() - runningTask.started).count();
            runningTask = outer;
            lock.lock();
            --m_running;
            Task& ran = m_tasks[id]; // the deque may have grown: its elements stay where they are
            ran.finishAt = ran.readyAt + seconds;
            m_workSeconds += seconds;
            m_criticalPathSeconds = std::max(m_criticalPathSeconds, ran.finishAt);
            if (failure)
            {
                m_isStopping = true;
                if (!m_failure || id < m_failedTask)
                {
                    m_failure = failure;
                    m_failedTask = id;
                }
            }
            finish(id, hasSucceeded && !failure);
            m_changed.notify_all();
            continue;
        }
        // Nothing to take: done when nothing runs (only a running task adds or releases tasks).
        if (m_running == 0)
        {
            m_changed.notify_all();
            return;
        }
        m_changed.wait(lock);
    }
}

void TaskFlow::run(ThreadPool* pool, const CancellationToken* cancellation)
{
    const std::size_t threads = pool == nullptr ? 1 : pool->threadCount();
    if (threads <= 1)
        work(cancellation);
    else
        pool->runBatch(threads, [&](std::size_t) { work(cancellation); });
    if (m_failure)
        std::rethrow_exception(m_failure);
    if (cancellation != nullptr)
        cancellation->throwIfCancelled();
}

bool TaskFlow::hasSucceeded(TaskId task) const
{
    const std::lock_guard lock(m_mutex);
    return task < m_tasks.size() && m_tasks[task].state == State::Succeeded;
}

std::size_t TaskFlow::size() const
{
    const std::lock_guard lock(m_mutex);
    return m_tasks.size();
}

double TaskFlow::workSeconds() const
{
    const std::lock_guard lock(m_mutex);
    return m_workSeconds;
}

double TaskFlow::criticalPathSeconds() const
{
    const std::lock_guard lock(m_mutex);
    return m_criticalPathSeconds;
}

} // namespace oos::base
