#pragma once

#include "oos/base/threading/ThreadPool.hpp"

#include <cstddef>
#include <memory>

namespace oos::api
{
struct ThreadingInfo
{
    bool enabled = false;
    std::size_t maximumThreads = 1;
    std::size_t activeWorkers = 0;
};

[[nodiscard]] ThreadingInfo threadingInfo();
// One thread means no pool. Owners requesting the same size share a pool; the module's preloaded worker
// capacity is checked before any worker is created. Retained analyses must retain this owner too.
[[nodiscard]] std::shared_ptr<base::ThreadPool> acquireThreadPool(std::size_t threads);
} // namespace oos::api
