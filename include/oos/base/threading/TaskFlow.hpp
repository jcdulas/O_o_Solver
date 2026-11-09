#pragma once

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <queue>
#include <span>
#include <vector>

namespace oos::base
{

// A set of tasks with dependencies and priorities, run as soon as their dependencies are done: the tile tasks of a
// large front and the supernodes of the top of an elimination tree in one flow, so
// that the threads do not wait at the end of a phase as they do between the batches of ParallelFor or the waves of
// TaskGraph.
//
// Scheduling: a ready queue, the highest priority first, ties the task added first. The flow runs on the threads of a
// ThreadPool (the calling thread included), each thread taking ready tasks until none is ready nor running. A task runs
// on one thread: it must not run a nested batch on the pool (the flow's threads wait on the flow, not on the pool's
// batches). A running task may add tasks; their dependencies may be done already.
//
// Failures: a task's body returns false when it failed (a zero pivot, say): every task that depends on it, directly or
// not, is skipped, and the others run. A body that throws stops the flow: no task starts any more, the running ones
// finish, and run() rethrows the exception of the lowest task that threw. With a cancellation token, the flow stops the
// same way once it is cancelled and run() throws OperationCancelledError.
//
// Determinism: the flow decides only which thread runs a task and when, never what it computes. A task writes only what
// it owns and reads only what its dependencies wrote (directly or not); then the results do not depend on the number of
// threads or the timing. Which tasks are skipped after a failure is fixed by the dependencies alone.
//
// Work and critical path (the model of the flow on more threads): each task's wall time is measured, and
// its earliest finish on unlimited threads is its time after the latest earliest finish of its dependencies (a task
// added by a running task depends on what that task had done when it added it): workSeconds() is the sum of the times,
// criticalPathSeconds() the latest earliest finish. On p threads a greedy schedule takes at least
// max(work / p, critical path) and at most work / p + critical path (Brent 1974; Graham 1969).
//
// Thread-safety: add() from the thread that builds the flow before run(), or from the flow's tasks while it runs; run()
// once. Ownership: owns the task bodies until they have run.
class TaskFlow
{
public:
    using TaskId = std::size_t;
    // Returns false when the task failed (its dependents are skipped).
    using Body = std::function<bool()>;

    TaskFlow() = default;
    TaskFlow(const TaskFlow&) = delete;
    TaskFlow& operator=(const TaskFlow&) = delete;

    // Adds a task that runs after every task of `dependencies` has succeeded (IDs returned by add() before:
    // std::invalid_argument otherwise), before the ready tasks of lower priority.
    TaskId add(Body body, std::span<const TaskId> dependencies = {}, std::int64_t priority = 0);
    TaskId add(Body body, std::initializer_list<TaskId> dependencies, std::int64_t priority = 0)
    {
        return add(std::move(body), std::span<const TaskId>(dependencies.begin(), dependencies.size()), priority);
    }

    // Runs the tasks on the pool (on the calling thread alone without one or with one thread) until every task has run
    // or been skipped. Rethrows the first failure (above); OperationCancelledError when the token is cancelled.
    void run(ThreadPool* pool, const CancellationToken* cancellation = nullptr);

    // After run(): whether the task ran and returned true.
    [[nodiscard]] bool hasSucceeded(TaskId task) const;
    [[nodiscard]] std::size_t size() const;
    // After run(): the tasks' wall time summed, and the longest chain of dependencies in it (class comment).
    [[nodiscard]] double workSeconds() const;
    [[nodiscard]] double criticalPathSeconds() const;

private:
    enum class State : std::uint8_t
    {
        Waiting,
        Ready,
        Running,
        Succeeded,
        Failed,
        Skipped
    };
    struct Task
    {
        Body body;
        std::vector<TaskId> successors;
        std::size_t remaining = 0; // dependencies not yet succeeded
        std::int64_t priority = 0;
        State state = State::Waiting;
        double readyAt = 0.0;  // its earliest start on unlimited threads, seconds from the flow's start
        double finishAt = 0.0; // its earliest finish
    };
    struct ReadyEntry
    {
        std::int64_t priority = 0;
        TaskId id = 0;
        // The queue's top is the largest: the highest priority, then the lowest ID.
        bool operator<(const ReadyEntry& other) const noexcept
        {
            return priority != other.priority ? priority < other.priority : id > other.id;
        }
    };

    void work(const CancellationToken* cancellation);
    // Under the mutex: the task is done; its successors are released (succeeded) or skipped (failed).
    void finish(TaskId task, bool hasSucceeded);
    void skipDependents(TaskId task);
    void makeReady(TaskId task);

    mutable std::mutex m_mutex;
    std::condition_variable m_changed;
    std::deque<Task> m_tasks; // stable references while tasks are added
    std::priority_queue<ReadyEntry> m_ready;
    std::size_t m_running = 0;
    bool m_isStopping = false;
    double m_workSeconds = 0.0;
    double m_criticalPathSeconds = 0.0;
    TaskId m_failedTask = 0;
    std::exception_ptr m_failure;
};

} // namespace oos::base
