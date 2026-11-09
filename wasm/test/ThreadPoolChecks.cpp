// Exercise real workers, nested batches and exception propagation in the compiled runtime.
#include "oos/api/Threading.hpp"

#include <emscripten/bind.h>

#include <atomic>
#include <barrier>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

int checkThreads(int count)
{
    auto pool = oos::api::acquireThreadPool(static_cast<std::size_t>(count));
    if (!pool)
        return 1;
    std::vector<std::thread::id> ids(static_cast<std::size_t>(count));
    std::barrier ready(count);
    pool->runBatch(static_cast<std::size_t>(count),
                   [&](std::size_t i)
                   {
                       ids[i] = std::this_thread::get_id();
                       ready.arrive_and_wait();
                   });
    const std::set<std::thread::id> unique(ids.begin(), ids.end());
    if (unique.size() != static_cast<std::size_t>(count))
        throw std::runtime_error("tasks did not run on distinct threads");
    std::atomic<int> completed{0};
    pool->runBatch(static_cast<std::size_t>(count * 2),
                   [&](std::size_t) { pool->runBatch(17, [&](std::size_t) { completed.fetch_add(1); }); });
    if (completed != count * 2 * 17)
        throw std::runtime_error("nested batches lost work");
    bool caught = false;
    try
    {
        pool->runBatch(19,
                       [](std::size_t i)
                       {
                           if (i == 2)
                               throw std::runtime_error("expected task failure");
                       });
    }
    catch (const std::runtime_error& error)
    {
        caught = std::string(error.what()) == "expected task failure";
    }
    if (!caught)
        throw std::runtime_error("a task exception did not reach its caller");
    pool->runBatch(3, [](std::size_t) {}); // the pool remains usable after an exception
    return static_cast<int>(unique.size());
}

EMSCRIPTEN_BINDINGS(oos_thread_pool_checks)
{
    emscripten::function("checkThreads", &checkThreads);
}
