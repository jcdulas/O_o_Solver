#include "oos/design/problem/DesignBatch.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ProgressCounter.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/solutions/common/ProgressMonitor.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

namespace oos::design
{

namespace
{

// One point's analysis on its share of the threads and of the memory, its failure captured as an exception pointer
// (classified after the batch, on the calling thread, so that the order of the points decides which one is raised).
struct PointRun
{
    BatchOutcome outcome;
    std::exception_ptr error;
};

void runPoint(const DesignProblem& problem, const BatchPoint& point, const solutions::StaticOptions& options,
              std::size_t threads, std::uint64_t memoryBytes, PointRun& run) noexcept
{
    try
    {
        run.outcome.budget = std::make_unique<base::MemoryBudget>(memoryBytes);
        const std::unique_ptr<base::ThreadPool> pool =
            threads > 1 ? std::make_unique<base::ThreadPool>(threads) : nullptr;
        solutions::StaticOptions pointOptions = options;
        pointOptions.pool = pool.get();
        pointOptions.monitor = nullptr; // the batch reports points, not the stages of each analysis
        run.outcome.analysis.emplace(
            problem.evaluateStatics(point.x, pointOptions, *run.outcome.budget, point.supplied, false));
    }
    catch (...)
    {
        run.outcome.analysis.reset();
        run.error = std::current_exception();
    }
}

} // namespace

std::vector<BatchOutcome> DesignBatch::analyse(const DesignProblem& problem, std::span<const BatchPoint> points,
                                               const solutions::StaticOptions& options, const BatchSettings& settings)
{
    if (settings.parallelPoints == 0 || settings.threads == 0)
        throw std::invalid_argument(base::concatenate(
            {"evaluate_batch: expected at least one parallel point and one thread, found parallel_points = ",
             std::to_string(settings.parallelPoints), " and threads = ", std::to_string(settings.threads)}));
    const std::size_t parallel = std::min(settings.parallelPoints, std::max<std::size_t>(points.size(), 1));
    const std::size_t shareThreads = std::max<std::size_t>(settings.threads / parallel, 1);
    const std::uint64_t shareMemory = settings.memoryBytes == base::MemoryBudget::UNLIMITED
                                          ? base::MemoryBudget::UNLIMITED
                                          : settings.memoryBytes / parallel;
    base::ProgressCounter* counter =
        options.monitor != nullptr ? &options.monitor->beginStage("design points", points.size()) : nullptr;

    std::vector<PointRun> runs(points.size());
    // The points in groups of `parallel`, each group run at once and finished before the next starts: the memory held
    // at any time is at most `parallel` analyses (one share each).
    const std::unique_ptr<base::ThreadPool> outer =
        parallel > 1 ? std::make_unique<base::ThreadPool>(parallel) : nullptr;
    for (std::size_t first = 0; first < points.size(); first += parallel)
    {
        const std::size_t count = std::min(parallel, points.size() - first);
        const auto task = [&](std::size_t k)
        {
            runPoint(problem, points[first + k], options, shareThreads, shareMemory, runs[first + k]);
            if (counter != nullptr)
                counter->advance();
        };
        if (outer)
            outer->runBatch(count, task);
        else
            for (std::size_t k = 0; k < count; ++k)
                task(k);
        if (options.cancellation != nullptr && options.cancellation->isCancelled())
            break; // the points not started stay without outcome; the cancellation is raised below
    }

    std::vector<BatchOutcome> outcomes;
    outcomes.reserve(points.size());
    for (PointRun& run : runs)
    {
        if (run.error)
        {
            std::optional<PointFailure> failure = classifyFailure(run.error);
            if (!failure || !settings.capturesFailures)
                std::rethrow_exception(run.error);
            run.outcome.failure = std::move(failure);
        }
        else if (!run.outcome.analysis)
        {
            // Not started: the batch was cancelled before it.
            if (options.cancellation != nullptr)
                options.cancellation->throwIfCancelled();
        }
        outcomes.push_back(std::move(run.outcome));
    }
    return outcomes;
}

} // namespace oos::design
