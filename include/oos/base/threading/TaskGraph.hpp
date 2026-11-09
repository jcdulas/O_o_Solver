#pragma once

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace oos::base
{

// A set of tasks with dependencies (the supernodes of an elimination tree, the units of an assembly): a task runs only
// after every task it depends on has finished. Tasks are added with their dependencies, which must already be in the
// graph, so the graph has no cycle by construction.
//
// Execution runs the graph in waves: wave k holds the tasks whose dependencies are all in earlier waves, and the tasks
// of a wave run as one batch of the pool, in the order they were added. What a task computes must not depend on which
// other tasks run at the same time (it writes only what it owns, and reads only what its dependencies wrote); then the
// result is the same for any number of threads.
// TODO: waves are simple and deterministic but leave threads idle at the end of a wide wave; a ready-queue
// scheduler may replace them if a measurement shows a gain worth it (acceptable now: the supernodal factorisation runs
// its subtrees in one wave and its top part, where the kernels are threaded, in a few).
//
// Thread-safety: add() from one thread; run() may be called once at a time. Ownership: owns the task callables.
class TaskGraph
{
public:
    using TaskId = std::size_t;

    // Adds a task and returns its ID; every dependency must be the ID of a task added before (std::invalid_argument
    // otherwise).
    TaskId add(std::function<void()> task, const std::vector<TaskId>& dependencies = {});

    [[nodiscard]] std::size_t size() const noexcept { return m_tasks.size(); }
    // The wave of a task: 0 without dependencies, otherwise 1 + the largest wave of its dependencies.
    [[nodiscard]] std::size_t waveOf(TaskId task) const { return m_wave.at(task); }

    // Runs every task (on the pool, or serially without one). Rethrows the failure of the lowest task of the earliest
    // wave that fails; the later waves do not run. With a token, the tasks not started when it is cancelled are
    // skipped and OperationCancelledError is thrown.
    void run(ThreadPool* pool, const CancellationToken* cancellation = nullptr) const;

private:
    std::vector<std::function<void()>> m_tasks;
    std::vector<std::size_t> m_wave;
};

} // namespace oos::base
