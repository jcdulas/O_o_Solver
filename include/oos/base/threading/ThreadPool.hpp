#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace oos::base
{

// A fixed set of worker threads that run batches of indexed tasks. A batch is "call
// task(i) for every i in [0, count)"; the calling thread works on its own batch too, so that a task may itself run a
// batch (nested parallelism) without deadlock.
//
// Scheduling: every batch gets a sequence number when it is pushed. A thread waiting for its batch helps only that
// batch and the batches pushed after it (the nested ones), newest first; idle workers also take the newest batch
// first. So a waiting caller never picks up an older, outer task (which may be long and delay its own batch: a
// priority inversion), and the tasks stacked on one thread's call stack belong to batches of increasing sequence: the
// depth of nested help is bounded by the number of batches alive (the nesting depth times the number of threads
// that push batches), never by the number of tasks.
//
// Determinism: the pool decides only *which thread* runs a task, never *what* a task computes. Results that combine
// several tasks must be combined by the caller in index order (ParallelFor does it), so that they do not depend on the
// number of threads or on the timing. A task runs in the subnormal mode of the thread that pushed its
// batch (SubnormalModeScope), whichever thread runs it: a run whose calling thread flushes subnormal numbers
// flushes them in every task, and the workers return to IEEE arithmetic between tasks.
//
// Errors: when tasks throw, the batch finishes the tasks already started, skips the others, and rethrows the exception
// of the task with the lowest index. Thread-safe; the pool must outlive the batches it runs.
class ThreadPool
{
public:
    // threadCount includes the calling thread: 1 runs everything on the caller, with no worker thread.
    explicit ThreadPool(std::size_t threadCount);
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool();

    [[nodiscard]] std::size_t threadCount() const noexcept { return m_workers.size() + 1; }

    // Runs task(0), ..., task(count - 1) and returns when all have finished.
    void runBatch(std::size_t count, const std::function<void(std::size_t)>& task);

private:
    struct Batch;

    void workerLoop(std::stop_token stopToken);
    // Claims and runs one task of the newest batch whose sequence is at least minimumSequence; false when there was
    // none. Batches whose tasks are all claimed leave the list on the way.
    bool runOneTask(std::unique_lock<std::mutex>& lock, std::uint64_t minimumSequence);
    // True when a batch of sequence at least minimumSequence may have a task to claim (under the mutex).
    [[nodiscard]] bool hasWorkFrom(std::uint64_t minimumSequence) const noexcept;
    static void runTask(Batch& batch, std::size_t index);

    std::mutex m_mutex;
    std::condition_variable_any m_workAvailable;
    std::condition_variable m_batchFinished;
    std::vector<std::shared_ptr<Batch>> m_batches; // the batches with tasks to claim, by increasing sequence
    std::uint64_t m_nextSequence = 0;
    std::vector<std::jthread> m_workers;
};

} // namespace oos::base
