#include "oos/api/Threading.hpp"

#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

namespace oos::api
{
namespace
{
std::mutex poolMutex;
std::map<std::size_t, std::weak_ptr<base::ThreadPool>> pools;

ThreadingInfo buildInfo()
{
#if defined(OOS_WASM) && !defined(__EMSCRIPTEN_PTHREADS__)
    return {};
#elif defined(OOS_WASM)
    return {true, OOS_WASM_MAX_THREADS, 0};
#else
    return {true, std::numeric_limits<std::size_t>::max(), 0};
#endif
}

std::size_t activeWorkers()
{
    std::size_t result = 0;
    for (auto it = pools.begin(); it != pools.end();)
    {
        if (const auto pool = it->second.lock())
        {
            result += pool->threadCount() - 1;
            ++it;
        }
        else
            it = pools.erase(it);
    }
    return result;
}
} // namespace

ThreadingInfo threadingInfo()
{
    std::lock_guard lock(poolMutex);
    auto result = buildInfo();
    result.activeWorkers = activeWorkers();
    return result;
}

std::shared_ptr<base::ThreadPool> acquireThreadPool(std::size_t threads)
{
    const auto info = buildInfo();
    if (threads == 0 || threads > info.maximumThreads)
        throw std::invalid_argument("threads must be between 1 and the module's maximumThreads");
    if (threads == 1)
        return {};
    std::lock_guard lock(poolMutex);
    if (const auto pool = pools[threads].lock())
        return pool;
    const auto used = activeWorkers();
    if (threads - 1 > info.maximumThreads - 1 - used)
        throw std::invalid_argument(
            "the preloaded worker capacity is in use; release evaluations with another "
            "thread count or reuse their thread count");
    auto pool = std::make_shared<base::ThreadPool>(threads);
    pools[threads] = pool;
    return pool;
}
} // namespace oos::api
