#pragma once

#include "oos/solutions/common/AnalysisPath.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace oos::solutions
{

// The size above which the size rule of the automatic path choice takes the condensed path: where the condensed path
// became faster than the full one on the reference PC, 6 threads, on the synthetic assemblies of plates joined by
// RigidInterpolation-Bushing-RigidInterpolation fasteners with one load case per component and the displacements of
// one component asked for: condensed / full time 1.14 at 43 000 DOFs, 0.90 at 98 000, 0.83 at 146 000, 0.91 at 175 000,
// 0.41 at 843 000. The costs of both paths decide (choosePath); the size decides only a tie of their estimated times,
// and stands in when an estimate is missing.
inline constexpr std::int64_t FULL_PATH_MAX_DOFS = 100'000;
// The share of the budget the full path's factor may take under the size rule (no estimate of the whole run): the other
// half holds the matrices, the load vectors and the result tables of the run (SOL 103 and SOL 105 keep this rule).
inline constexpr double FULL_PATH_BUDGET_SHARE = 0.5;
// The ratio of two estimated times beyond which the faster path is taken: the time estimates of both paths (the
// pre-flight estimates) are within PATH_TIME_MARGIN of the measured times on the calibration models, so a smaller ratio
// is no evidence that one path is faster; within it the size rule decides.
inline constexpr double PATH_TIME_MARGIN = 1.5;

// The analysis whose path is chosen: each has its own rule (choosePath).
enum class PathAnalysis : std::uint8_t
{
    Statics,     // SOL 101
    Modes,       // SOL 103
    Buckling,    // SOL 105
    Trim,        // SOL 144
    Optimisation // SOL 200
};

[[nodiscard]] std::string_view nameOf(PathAnalysis analysis) noexcept;

// The estimated cost of one path, as the choice reads it (the pre-flight estimates).
struct PathCost
{
    // The smallest memory budget the path runs in: the full path's factorisation, matrices, vectors and output tables;
    // the condensed path's memory plan in the memory mode of the run. Nothing when not estimated: the condensed path's
    // memory plan is made only when the choice depends on it (PathChoice::needsCondensedMemory).
    std::optional<std::uint64_t> memoryBytes;
    double seconds = 0.0; // its estimated time on the run's threads
};

// What the choice reads: the size of the model, the estimate of its full factorisation, the costs of both paths, the
// budget.
struct PathChoiceInputs
{
    PathAnalysis analysis = PathAnalysis::Statics;
    std::int64_t dofCount = 0; // the g set
    // The memory of the full factorisation (factor, update stack and indices: SymbolicFactor::memoryPeakBytes of the
    // whole model, StaticPreflight); nothing when not estimated (a model above maxFullPathDofs needs no estimate).
    std::optional<std::uint64_t> fullFactorBytes;
    std::uint64_t budgetBytes = 0;
    // PARAM INREL = −2 (the supports chosen by the solver), which the condensed path refuses: such a static model stays
    // on the full path, said so.
    bool hasAutomaticInertiaRelief = false;
    // FULL_PATH_MAX_DOFS unless the caller tests the rule with another size.
    std::int64_t maxFullPathDofs = FULL_PATH_MAX_DOFS;
    // The estimated costs of both paths: with both, the costs decide a static analysis; with either missing, and for
    // the other analyses, the size rule.
    std::optional<PathCost> fullCost;
    std::optional<PathCost> condensedCost;
    // PATH_TIME_MARGIN, but in the tests of the rule.
    double timeMargin = PATH_TIME_MARGIN;
};

// The path the automatic choice takes and why, in a sentence for the run record and the check report.
struct PathChoice
{
    AnalysisPath path = AnalysisPath::Full;
    std::string reason;
    bool isAboveSizeLimit = false;   // dofCount > maxFullPathDofs
    bool isAboveBudgetShare = false; // fullFactorBytes > FULL_PATH_BUDGET_SHARE × budgetBytes
    bool isCostBased = false;        // the costs of both paths decided (not the size rule)
    // The choice depends on the condensed path's memory, which the inputs do not give: the path is the condensed one if
    // it fits the budget and the full one otherwise; the caller makes the condensed path's memory plan and asks again.
    bool needsCondensedMemory = false;
};

// The automatic path choice:
// - statics with the costs of both paths: the memory first, then the time. The full path when it fits the budget and
//   the condensed path does not; the condensed path when the full path does not fit (if the condensed path does not fit
//   either, its memory plan refuses the analysis before any work, with its estimate); when both fit, the path estimated
//   faster by more than timeMargin times, and within it the size rule below; a condensed path whose memory is not given
//   is taken as fitting, the choice saying when it depends on it (needsCondensedMemory);
// - statics without the costs (the size rule): the full path when the model has at most maxFullPathDofs DOFs and its
//   estimated full factorisation takes at most FULL_PATH_BUDGET_SHARE of the budget; the condensed path otherwise (the
//   caller then makes the plan of every unit, those the plan recommends keeping full excepted; a model of one unit
//   stays on the full path, which the plan says); the full path whatever its size with PARAM INREL = −2, which the
//   condensed path refuses;
// - modes: the rule of statics, the reason of the condensed path saying that its modes are approximate (Craig-Bampton),
//   each unit keeping its fixed-interface modes up to reduction::CB_CUTOFF_FACTOR times the highest frequency
//   requested;
// - buckling (SOL 105 never condenses while the full path fits): the condensed path only when the full factorisation is
//   above the share of the budget, whatever the size, the reason saying that the local buckling of the condensed units
//   is not checked;
// - optimisation (SOL 200 on statics): the size rule of statics, the caller's plan keeping full the units the design
//   variables drive;
// - trim (SOL 144): the size rule of statics, the caller's plan keeping the spline nodes on the boundaries of their
//   units (trimRetainedPoints).
// Throws std::invalid_argument for a negative size or time, a zero budget or a margin below 1.
[[nodiscard]] PathChoice choosePath(const PathChoiceInputs& inputs);

} // namespace oos::solutions
