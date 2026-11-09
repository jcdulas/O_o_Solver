#include "oos/design/problem/ShapeDistortion.hpp"

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/model/DesignCodes.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"

#include <set>
#include <string>
#include <tuple>

namespace oos::design
{

namespace
{

using ProblemKey = std::tuple<std::string, std::string, std::int64_t, std::string>;

[[nodiscard]] ProblemKey keyOf(const base::Diagnostic& diagnostic)
{
    return {diagnostic.code, diagnostic.card, diagnostic.id, diagnostic.field};
}

} // namespace

std::vector<base::Diagnostic> ShapeDistortion::elementProblems(const input::ModelSnapshot& snapshot,
                                                               base::MemoryBudget& budget)
{
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    const elements::SnapshotIndex index(snapshot, budget);
    const assembly::ElementSet set(index, registry);
    base::DiagnosticSink sink;
    set.check(sink);
    return sink.diagnostics();
}

std::vector<base::Diagnostic> ShapeDistortion::distortions(const std::vector<base::Diagnostic>& baseProblems,
                                                           const std::vector<base::Diagnostic>& point)
{
    std::set<ProblemKey> known;
    for (const base::Diagnostic& problem : baseProblems)
        known.insert(keyOf(problem));
    std::vector<base::Diagnostic> distortions;
    for (const base::Diagnostic& problem : point)
    {
        if (problem.severity == base::Severity::Information || known.count(keyOf(problem)) != 0)
            continue;
        base::Diagnostic distortion = problem;
        distortion.code = design_code::SHAPE_DISTORTION;
        distortion.message = base::concatenate(
            {"the design point moves its grid points (DesignNodeMove) so that the element check finds [", problem.code,
             "] ", problem.message,
             "; the base design does not have it: bound the shape variables (DesignVar DesignVar.lowerBound, "
             "DesignVar.upperBound, or the move limits) "
             "so that the mesh stays valid"});
        distortions.push_back(std::move(distortion));
    }
    return distortions;
}

} // namespace oos::design
