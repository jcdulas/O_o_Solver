#include "oos/base/threading/ThreadPool.hpp"

#include "oos/base/platform/SubnormalModeScope.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <stop_token>

namespace oos::base
{

struct ThreadPool::Batch
{
    const std::function<void(std::size_t)>* task = nullptr;
    std::size_t count = 0;
    std::uint64_t sequence = 0;         // the order of the pushes
    std::size_t next = 0;               // the next index to claim (under the pool's mutex)
    std::size_t finished = 0;           // tasks finished or skipped (under the pool's mutex)
    std::atomic<bool> hasFailed{false}; // once set, the tasks not yet started are skipped
    std::mutex failureMutex;            // guards failedIndex and failure
    std::size_t failedIndex = std::numeric_limits<std::size_t>::max();
    std::exception_ptr failure;
    SubnormalMode subnormals = SubnormalMode::Ieee; // the mode of the thread that pushed the batch
};

ThreadPool::ThreadPool(std::size_t threadCount)
{
#if defined(OOS_WASM)
    // WebAssembly runs on one thread (the module is built without pthreads): every batch runs on the caller.
    (void)threadCount;
    const std::size_t workerCount = 0;
#else
    const std::size_t workerCount = threadCount > 1 ? threadCount - 1 : 0;
#endif
    m_workers.reserve(workerCount);
    for (std::size_t i = 0; i < workerCount; ++i)
        m_workers.emplace_back([this](std::stop_token stopToken) { workerLoop(stopToken); });
}

ThreadPool::~ThreadPool()
{
    for (auto& worker : m_workers)
        worker.request_stop();
    m_workAvailable.notify_all();
    m_workers.clear(); // joins
}

void ThreadPool::runBatch(std::size_t count, const std::function<void(std::size_t)>& task)
{
    if (count == 0)
        return;
    auto batch = std::make_shared<Batch>();
    batch->task = &task;
    batch->count = count;
    batch->subnormals = subnormalModeOfThisThread();

    std::unique_lock lock(m_mutex);
    batch->sequence = m_nextSequence++;
    const std::uint64_t ownSequence = batch->sequence;
    m_batches.push_back(batch);
    m_workAvailable.notify_all();
    m_batchFinished.notify_all(); // callers waiting for an older batch may help with this one
    // The caller works on its own batch and on the batches pushed after it, newest first, until its batch is
    // complete; it never takes a task of an older batch.
    while (batch->finished < batch->count)
    {
        if (!runOneTask(lock, ownSequence))
            m_batchFinished.wait(lock, [&] { return batch->finished == batch->count || hasWorkFrom(ownSequence); });
    }
    if (batch->failure)
        std::rethrow_exception(batch->failure);
}

bool ThreadPool::hasWorkFrom(std::uint64_t minimumSequence) const noexcept
{
    // A batch leaves the list when its last task is claimed, so a batch in the list has a task to claim.
    return !m_batches.empty() && m_batches.back()->sequence >= minimumSequence;
}

bool ThreadPool::runOneTask(std::unique_lock<std::mutex>& lock, std::uint64_t minimumSequence)
{
    if (!hasWorkFrom(minimumSequence))
        return false;
    // The newest batch: the list is ordered by sequence, and a batch in it has an unclaimed task.
    const std::shared_ptr<Batch> batch = m_batches.back();
    const std::size_t index = batch->next++;
    if (batch->next >= batch->count)
        m_batches.pop_back(); // every task claimed; the batch completes when the running ones finish
    const bool isSkipped = batch->hasFailed;
    lock.unlock();
    if (!isSkipped)
        runTask(*batch, index);
    lock.lock();
    ++batch->finished;
    if (batch->finished == batch->count)
        m_batchFinished.notify_all();
    return true;
}

void ThreadPool::runTask(Batch& batch, std::size_t index)
{
    // The arithmetic of the thread that pushed the batch, whichever thread runs the task (class comment): a scope of
    // the mode the thread is in writes nothing.
    const SubnormalModeScope subnormals(batch.subnormals);
    try
    {
        (*batch.task)(index);
    }
    catch (...)
    {
        // The failure of the lowest index is kept, whatever the order in which the threads fail; it is read only after
        // the batch has completed.
        std::lock_guard failureLock(batch.failureMutex);
        batch.hasFailed.store(true);
        if (index < batch.failedIndex)
        {
            batch.failedIndex = index;
            batch.failure = std::current_exception();
        }
    }
}

void ThreadPool::workerLoop(std::stop_token stopToken)
{
    std::unique_lock lock(m_mutex);
    while (!stopToken.stop_requested())
    {
        if (!runOneTask(lock, 0))
            m_workAvailable.wait(lock, stopToken, [this] { return !m_batches.empty(); });
    }
}

} // namespace oos::base
