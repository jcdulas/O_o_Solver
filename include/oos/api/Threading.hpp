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

// The thread count that lets the solver choose (AnalysisSettings::threads, OptimizationSettings::threads): see
// automaticThreadCount.
inline constexpr std::size_t AUTOMATIC_THREADS = 0;

[[nodiscard]] ThreadingInfo threadingInfo();
// The threads of a run that lets the solver choose: every logical core the machine reports natively
// (std::thread::hardware_concurrency, 1 when it reports none), within the module's maximumThreads in WebAssembly (1 in
// the build without threads). The kernels give the same bits with any thread count (src/linalg/README.md, the
// arithmetic contract), so the choice changes the time of a run and never its results.
[[nodiscard]] std::size_t automaticThreadCount();
// One thread means no pool; AUTOMATIC_THREADS asks for automaticThreadCount(). Owners requesting the same size share a
// pool; the module's preloaded worker capacity is checked before any worker is created. Retained analyses must retain
// this owner too.
[[nodiscard]] std::shared_ptr<base::ThreadPool> acquireThreadPool(std::size_t threads);
} // namespace oos::api
