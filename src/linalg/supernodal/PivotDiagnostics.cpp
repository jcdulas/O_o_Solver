#include "oos/linalg/supernodal/PivotDiagnostics.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace oos::linalg
{
namespace
{

std::string formatRatio(double value)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
}

} // namespace

std::string describeEquation(const EquationNamer& namer, std::int32_t equation)
{
    if (!namer)
        return std::string("equation ").append(std::to_string(equation));
    const base::GridComponent dof = namer(equation);
    if (dof.component == 0)
        return std::string("scalar point ").append(std::to_string(dof.grid));
    return std::string("grid ")
        .append(std::to_string(dof.grid))
        .append(" component ")
        .append(std::to_string(dof.component));
}

base::GridComponent gridComponentOf(const EquationNamer& namer, std::int32_t equation)
{
    return namer ? namer(equation) : base::GridComponent{equation, 0};
}

PivotDiagnostics::PivotDiagnostics(std::vector<PivotIssue> issues, std::int64_t issueCount, double largestRatio,
                                   std::int32_t equationOfLargestRatio, double maxRatio)
    : m_issues(std::move(issues)),
      m_issueCount(std::max(issueCount, static_cast<std::int64_t>(m_issues.size()))),
      m_largestRatio(largestRatio),
      m_equationOfLargestRatio(equationOfLargestRatio),
      m_maxRatio(maxRatio)
{
}

std::string PivotDiagnostics::describe(const EquationNamer& namer) const
{
    std::string text;
    const std::size_t listed = std::min(m_issues.size(), LISTED_ISSUE_COUNT);
    for (std::size_t k = 0; k < listed; ++k)
    {
        const PivotIssue& issue = m_issues[k];
        text.append("  ")
            .append(describeEquation(namer, issue.equation))
            .append(": matrix diagonal ")
            .append(formatRatio(issue.matrixDiagonal))
            .append(", factor diagonal ")
            .append(formatRatio(issue.factorDiagonal))
            .append(", ratio ")
            .append(formatRatio(issue.ratio))
            .append(issue.kind == PivotIssueKind::NegativePivot ? " (negative: not positive definite)\n"
                                                                : " (above the ratio limit: near mechanism)\n");
    }
    const auto listedCount = static_cast<std::int64_t>(listed);
    if (m_issueCount > listedCount)
        text.append("  and ").append(std::to_string(m_issueCount - listedCount)).append(" more\n");
    return text;
}

void PivotDiagnostics::enforce(const PivotPolicy& policy, const EquationNamer& namer) const
{
    if (m_issueCount == 0 || !policy.doesStopOnIssues())
        return;
    std::vector<base::GridComponent> dofs;
    dofs.reserve(m_issues.size());
    for (const PivotIssue& issue : m_issues)
        dofs.push_back(gridComponentOf(namer, issue.equation));
    throw base::SingularityError(
        base::concatenate(
            {"the stiffness matrix is nearly singular or not positive definite: ", std::to_string(m_issueCount),
             " pivots fail the factor diagonal ratio test (ratio of the matrix diagonal to the factor "
             "diagonal above the ratio limit ",
             formatRatio(m_maxRatio), ", or negative):\n", describe(namer),
             "constrain the mechanisms at these DOFs (single-point constraints), check the connections and the "
             "properties there, or let the pivot policy keep the factor (the singularity bail-out) to continue "
             "with it"}),
        std::move(dofs));
}

} // namespace oos::linalg
