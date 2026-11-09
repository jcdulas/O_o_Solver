#include "oos/solutions/common/RunChecks.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"

#include <cstdio>
#include <string>
#include <utility>

namespace oos::solutions
{

void stopOnErrors(const base::DiagnosticSink& sink, const std::string& stage)
{
    if (!sink.hasErrors())
        return;
    std::string message;
    std::size_t listed = 0;
    std::size_t errors = 0;
    for (const base::Diagnostic& diagnostic : sink.diagnostics())
    {
        if (diagnostic.severity != base::Severity::Error)
            continue;
        ++errors;
        if (listed < LISTED_PROBLEM_COUNT)
        {
            message.append("\n  ").append(base::formatDiagnostic(diagnostic));
            ++listed;
        }
    }
    std::string text = base::concatenate({"the model has ", std::to_string(errors), " problem", errors == 1 ? "" : "s",
                                          " that stop the analysis before ", stage, ":"});
    text.append(message);
    if (errors > listed)
        text.append("\n  and ").append(std::to_string(errors - listed)).append(" more");
    text.append("\ncorrect them in the model and run again");
    throw base::SolverError(text);
}

void reportSupport(const input::ModelSnapshot& snapshot, const elements::ElementRegistry& registry,
                   AnalysisKind analysis, base::DiagnosticSink& sink)
{
    const base::SupportReport report = checkSupport(snapshot, registry, analysis);
    for (const base::Diagnostic& diagnostic : report.ignored)
        sink.report(diagnostic);
    if (report.isComplete())
        return;
    std::string message = base::concatenate(
        {"the model uses ", std::to_string(report.refusals.size()), " card", report.refusals.size() == 1 ? "" : "s",
         " or field", report.refusals.size() == 1 ? "" : "s", " that this version of O_o_Solver does not support:"});
    for (std::size_t i = 0; i < report.refusals.size() && i < LISTED_PROBLEM_COUNT; ++i)
        message.append("\n  ").append(base::formatDiagnostic(report.refusals[i]));
    if (report.refusals.size() > LISTED_PROBLEM_COUNT)
        message.append("\n  and ")
            .append(std::to_string(report.refusals.size() - LISTED_PROBLEM_COUNT))
            .append(" more");
    throw base::UnsupportedCardError(message);
}

void reportKeptPivots(const linalg::PivotDiagnostics& pivots, const linalg::EquationNamer& namer,
                      base::EntityId subcase, base::DiagnosticSink& sink)
{
    if (!pivots.hasIssues())
        return;
    // MAXRATIO as the SingularityError of a stopping policy writes it (PivotDiagnostics::enforce).
    constexpr std::size_t RATIO_TEXT_SIZE = 32;
    char ratio[RATIO_TEXT_SIZE];
    std::snprintf(ratio, sizeof(ratio), "%.3e", pivots.maxRatio());
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Warning;
    diagnostic.code = "max-diagonal-ratio-exceeded";
    diagnostic.card = "SubcaseStart";
    diagnostic.id = subcase;
    diagnostic.field = "SingularityBailOut";
    diagnostic.message = base::concatenate(
        {std::to_string(pivots.issueCount()),
         " pivots of the stiffness fail the factor diagonal ratio test (above MaxDiagonalRatio = ", ratio,
         ", or negative) and PARAM SingularityBailOut -1 keeps the factor, so the results near these DOFs may be "
         "inaccurate:\n",
         pivots.describe(namer),
         "constrain the mechanisms at these DOFs (SinglePointConstr, AutoSingularityConstr), or check the "
         "connections and the properties there"});
    sink.report(std::move(diagnostic));
}

} // namespace oos::solutions
