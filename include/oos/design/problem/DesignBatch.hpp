#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/problem/EvaluationStatus.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::design
{

// One point of a batch: x and the values of the relations given by the caller's functions at x (computed by the
// caller on the calling thread before the batch starts).
struct BatchPoint
{
    std::vector<double> x;
    std::vector<SuppliedField> supplied;
};

// What one point of a batch gives: its analysis with the budget the analysis is reserved in, or the failure that
// stopped it (with failures captured). The budget is declared first so that it outlives the analysis.
struct BatchOutcome
{
    std::unique_ptr<base::MemoryBudget> budget;
    std::optional<StaticDesignEvaluation> analysis;
    std::optional<PointFailure> failure;
};

// How a batch runs.
struct BatchSettings
{
    std::size_t parallelPoints = 1; // the analyses run at the same time
    std::size_t threads = 1;        // the threads of the whole batch, shared out between the parallel points
    std::uint64_t memoryBytes = base::MemoryBudget::UNLIMITED; // the memory of the whole batch, shared out likewise
    bool capturesFailures = false; // a failure of a point is its outcome, not an exception
};

// The analyses of several design points at once: the points are taken in their order, `parallelPoints` at a time, each
// analysis on its share of the threads (threads / parallelPoints, at least one) and of the memory budget (memoryBytes /
// parallelPoints), every analysis sharing the problem's base snapshot (the symbolic analyses are made per point: the
// design point changes the values, not the pattern). The responses are evaluated afterwards on the calling thread
// (DesignProblem::evaluateResponses), point by point in order, so that the caller's functions are called from it.
//
// Determinism: a static analysis gives the same bits whatever its number of threads, and no point reads
// another's results, so the outcomes are those of the serial run whatever `parallelPoints`. A share of the memory too
// small for a point gives that point a Budget failure the serial run would not have: the share is the documented price
// of the parallel points.
//
// Thread-safety: analyse() runs the points on worker threads it owns and returns when all are done; the problem must
// not change meanwhile (it is immutable). Ownership: the outcomes own their analyses and budgets.
class DesignBatch
{
public:
    // The analyses of `points` with `options` (its pool and monitor are replaced: each point gets its own pool, and
    // `options.monitor`, when set, gets one stage "design points" advanced once per point). Throws, when failures are
    // not captured, the failure of the first failing point in the order of the points (after every started point has
    // finished); always throws a failure classifyFailure() does not classify (a cancellation, a misuse).
    // std::invalid_argument for parallelPoints or threads of 0.
    [[nodiscard]] static std::vector<BatchOutcome> analyse(const DesignProblem& problem,
                                                           std::span<const BatchPoint> points,
                                                           const solutions::StaticOptions& options,
                                                           const BatchSettings& settings);
};

} // namespace oos::design
