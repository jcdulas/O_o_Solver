#include "oos/solutions/common/PathChoice.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/reduction/condense/CraigBampton.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace oos::solutions
{

namespace
{

// CB_CUTOFF_FACTOR as the report prints it.
std::string cutoffFactorText()
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%g", reduction::CB_CUTOFF_FACTOR);
    return std::string(text.data());
}

// A time in seconds, three significant digits.
std::string secondsText(double seconds)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.3g s", seconds);
    return std::string(text.data());
}

// The size rule's clause: the model's DOFs against the limit.
std::string sizeText(const PathChoiceInputs& inputs, bool isAbove)
{
    return base::concatenate({"the model has ", std::to_string(inputs.dofCount), " DOFs, ",
                              isAbove ? "more than" : "at most", " FULL_PATH_MAX_DOFS (",
                              std::to_string(inputs.maxFullPathDofs), ")"});
}

// The choice of a static analysis by the costs of both paths (choosePath).
PathChoice chooseByCost(const PathChoiceInputs& inputs, PathChoice choice)
{
    const PathCost& full = *inputs.fullCost;
    const PathCost& condensed = *inputs.condensedCost;
    const std::string budget = base::formatBytes(inputs.budgetBytes);
    const std::string fullMemory = base::formatBytes(*full.memoryBytes);
    const bool isFullFitting = *full.memoryBytes <= inputs.budgetBytes;
    const bool isCondensedKnown = condensed.memoryBytes.has_value();
    const bool isCondensedFitting = !isCondensedKnown || *condensed.memoryBytes <= inputs.budgetBytes;
    const std::string condensedMemory = isCondensedKnown ? base::formatBytes(*condensed.memoryBytes) : std::string();
    choice.isCostBased = true;
    if (!isFullFitting)
    {
        choice.path = AnalysisPath::Condensed;
        const std::string fullClause = base::concatenate(
            {"the full path's estimated memory (", fullMemory, ") is above the memory budget (", budget, ")"});
        if (!isCondensedKnown)
            choice.reason = base::concatenate(
                {"the condensed path: ", fullClause, "; the condensed analysis checks its own before any work"});
        else if (isCondensedFitting)
            choice.reason = base::concatenate(
                {"the condensed path: ", fullClause, ", the condensed path's (", condensedMemory, ") fits"});
        else
            choice.reason = base::concatenate(
                {"the condensed path: neither path fits the memory budget (", budget, "): the full path needs about ",
                 fullMemory, ", the condensed path ", condensedMemory,
                 "; its memory plan refuses the analysis before any work, with the remedies"});
        return choice;
    }
    if (!isCondensedFitting)
    {
        choice.path = AnalysisPath::Full;
        choice.reason =
            base::concatenate({"the full path: its estimated memory (", fullMemory, ") fits the memory budget (",
                               budget, "), the condensed path's (", condensedMemory, ") does not"});
        return choice;
    }
    const std::string memories =
        isCondensedKnown ? base::concatenate({"both paths fit the memory budget (", budget, ": full ", fullMemory,
                                              ", condensed ", condensedMemory, ")"})
                         : base::concatenate({"the full path fits the memory budget (", budget, ": ", fullMemory, ")"});
    std::array<char, 32> margin{};
    std::snprintf(margin.data(), margin.size(), "%g", inputs.timeMargin);
    const std::string times =
        base::concatenate({"full ", secondsText(full.seconds), ", condensed ", secondsText(condensed.seconds)});
    AnalysisPath faster = AnalysisPath::Full;
    std::string why;
    if (full.seconds * inputs.timeMargin <= condensed.seconds)
        why = base::concatenate({memories, " and the full path is estimated faster by more than PATH_TIME_MARGIN = ",
                                 margin.data(), " times (", times, ")"});
    else if (condensed.seconds * inputs.timeMargin <= full.seconds)
    {
        faster = AnalysisPath::Condensed;
        why = base::concatenate(
            {memories, " and the condensed path is estimated faster by more than PATH_TIME_MARGIN = ", margin.data(),
             " times (", times, ")"});
    }
    else
    {
        // A tie: the size rule.
        faster = choice.isAboveSizeLimit ? AnalysisPath::Condensed : AnalysisPath::Full;
        why = base::concatenate({memories, " and their estimated times are within PATH_TIME_MARGIN = ", margin.data(),
                                 " times of each other (", times, "): the size decides, ",
                                 sizeText(inputs, choice.isAboveSizeLimit)});
    }
    choice.path = faster;
    choice.needsCondensedMemory = faster == AnalysisPath::Condensed && !isCondensedKnown;
    choice.reason = base::concatenate({faster == AnalysisPath::Full ? "the full path: " : "the condensed path: ", why,
                                       choice.needsCondensedMemory ? "; if its memory plan fits the budget" : ""});
    return choice;
}

// The comparison of the factor estimate with the share of the budget, as a clause (fullFactorBytes is given).
std::string factorText(const PathChoiceInputs& inputs, bool isAboveShare)
{
    return base::concatenate({"the estimated factor of its full factorisation (",
                              base::formatBytes(*inputs.fullFactorBytes), ") is ",
                              isAboveShare ? "more than" : "at most", " half of the memory budget (",
                              base::formatBytes(inputs.budgetBytes), ")"});
}

} // namespace

std::string_view nameOf(PathAnalysis analysis) noexcept
{
    switch (analysis)
    {
    case PathAnalysis::Statics:
        return "SOL 101";
    case PathAnalysis::Modes:
        return "SOL 103";
    case PathAnalysis::Buckling:
        return "SOL 105";
    case PathAnalysis::Trim:
        return "SOL 144";
    case PathAnalysis::Optimisation:
        return "SOL 200";
    }
    return "";
}

PathChoice choosePath(const PathChoiceInputs& inputs)
{
    if (inputs.dofCount < 0 || inputs.maxFullPathDofs < 0)
        throw std::invalid_argument(
            base::concatenate({"choosePath: expected non-negative sizes, found ", std::to_string(inputs.dofCount),
                               " DOFs and a limit of ", std::to_string(inputs.maxFullPathDofs)}));
    if (inputs.budgetBytes == 0)
        throw std::invalid_argument("choosePath: expected a memory budget of at least one byte, found 0");
    if (!(inputs.timeMargin >= 1.0))
        throw std::invalid_argument(base::concatenate(
            {"choosePath: expected a time margin of at least 1, found ", std::to_string(inputs.timeMargin)}));
    for (const std::optional<PathCost>* cost : {&inputs.fullCost, &inputs.condensedCost})
        if (*cost && !((*cost)->seconds >= 0.0))
            throw std::invalid_argument(base::concatenate(
                {"choosePath: expected non-negative estimated times, found ", std::to_string((*cost)->seconds)}));
    if (inputs.fullCost && !inputs.fullCost->memoryBytes)
        throw std::invalid_argument("choosePath: the full path's cost needs its estimated memory");
    PathChoice choice;
    const std::string dofs = std::to_string(inputs.dofCount);
    const std::string limit = std::to_string(inputs.maxFullPathDofs);
    choice.isAboveSizeLimit = inputs.dofCount > inputs.maxFullPathDofs;
    choice.isAboveBudgetShare =
        inputs.fullFactorBytes &&
        static_cast<double>(*inputs.fullFactorBytes) > FULL_PATH_BUDGET_SHARE * static_cast<double>(inputs.budgetBytes);

    if (inputs.analysis == PathAnalysis::Buckling)
    {
        // SOL 105 never condenses while the full path fits; when it does not, it condenses and the report lists, at its
        // top, the units whose local buckling is not checked. The size alone never condenses it.
        if (choice.isAboveBudgetShare)
        {
            choice.path = AnalysisPath::Condensed;
            choice.reason = base::concatenate(
                {"the condensed path (Craig-Bampton): ", factorText(inputs, true),
                 ", so the full path of SOL 105 does not fit; the local buckling of the condensed units is not "
                 "checked (they are listed at the top of the report)"});
            return choice;
        }
        choice.path = AnalysisPath::Full;
        choice.reason =
            inputs.fullFactorBytes
                ? base::concatenate({"the full path: SOL 105 condenses only when the full path does not fit, and ",
                                     factorText(inputs, false)})
                : std::string(
                      "the full path: SOL 105 condenses only when the full path does not fit; its factorisation "
                      "was not estimated");
        return choice;
    }
    // An optimisation on statics (SOL 200) and a trim (SOL 144) follow the static rule; the caller's plan keeps full
    // the units the design variables drive (every unit condensed except those driven by a design variable), or keeps
    // the spline nodes on the boundaries of their units (SOL 144: static condensation, spline nodes retained).
    // SOL 103 may condense, by the rule of statics, and then says that its modes are approximate (Craig-Bampton), with
    // the cut-off.
    const bool isModes = inputs.analysis == PathAnalysis::Modes;
    const std::string craigBampton =
        isModes
            ? base::concatenate({" (Craig-Bampton: the modes are approximate, each unit keeping its fixed-interface "
                                 "modes up to CB_CUTOFF_FACTOR = ",
                                 cutoffFactorText(), " times the highest frequency requested)"})
            : std::string();
    if (isModes)
    {
        if (choice.isAboveSizeLimit)
        {
            choice.path = AnalysisPath::Condensed;
            choice.reason = base::concatenate({"the condensed path", craigBampton, ": the model has ", dofs,
                                               " DOFs, more than FULL_PATH_MAX_DOFS (", limit, ")"});
            return choice;
        }
        if (choice.isAboveBudgetShare)
        {
            choice.path = AnalysisPath::Condensed;
            choice.reason = base::concatenate({"the condensed path", craigBampton, ": ", factorText(inputs, true),
                                               ", so the full path does not fit comfortably"});
            return choice;
        }
        choice.path = AnalysisPath::Full;
        choice.reason =
            base::concatenate({"the full path: the model has ", dofs, " DOFs, at most FULL_PATH_MAX_DOFS (", limit, ")",
                               inputs.fullFactorBytes ? base::concatenate({", and ", factorText(inputs, false)})
                                                      : std::string("; its factorisation was not estimated")});
        return choice;
    }

    if (inputs.hasAutomaticInertiaRelief)
    {
        choice.path = AnalysisPath::Full;
        choice.reason = "the full path: PARAM InertiaRelief = -2 is not supported on the condensed path";
        if (choice.isAboveSizeLimit || choice.isAboveBudgetShare)
            choice.reason.append(
                "; the model is large enough for the condensed path: give the supports with FreeBodySupport "
                "and PARAM InertiaRelief = -1 to use it");
        return choice;
    }
    // The costs decide a static analysis; a trim and an optimisation keep the size rule (their condensed
    // path's cost over the trim's iterations or the design cycles is not that of one static analysis).
    if (inputs.analysis == PathAnalysis::Statics && inputs.fullCost && inputs.condensedCost)
        return chooseByCost(inputs, choice);
    if (choice.isAboveSizeLimit)
    {
        choice.path = AnalysisPath::Condensed;
        choice.reason =
            base::concatenate({"the condensed path: the model has ", dofs, " DOFs, more than FULL_PATH_MAX_DOFS (",
                               limit, "), above which the condensed path is the faster one"});
        return choice;
    }
    if (!inputs.fullFactorBytes)
    {
        choice.path = AnalysisPath::Full;
        choice.reason = base::concatenate({"the full path: the model has ", dofs, " DOFs, at most FULL_PATH_MAX_DOFS (",
                                           limit, "); its factorisation was not estimated"});
        return choice;
    }
    if (choice.isAboveBudgetShare)
    {
        choice.path = AnalysisPath::Condensed;
        choice.reason = base::concatenate(
            {"the condensed path: ", factorText(inputs, true), ", so the full path does not fit comfortably"});
        return choice;
    }
    choice.path = AnalysisPath::Full;
    choice.reason = base::concatenate({"the full path: the model has ", dofs, " DOFs, at most FULL_PATH_MAX_DOFS (",
                                       limit, "), and ", factorText(inputs, false)});
    return choice;
}

} // namespace oos::solutions
