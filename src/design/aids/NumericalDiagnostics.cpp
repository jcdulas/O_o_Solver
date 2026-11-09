#include "oos/design/aids/NumericalDiagnostics.hpp"

#include "oos/solutions/common/FactorisedSet.hpp"

#include <algorithm>

namespace oos::design
{

NumericalDiagnosticsResult NumericalDiagnostics::of(const StaticDesignEvaluation& analysis)
{
    const solutions::StaticResults& results = analysis.results;
    NumericalDiagnosticsResult result;
    result.largestRatio = results.largestPivotRatio;
    for (const std::shared_ptr<const solutions::FactorisedSet>& set : results.systems)
    {
        FactorisationDiagnostics factorisation;
        factorisation.subcases.assign(set->subcases().begin(), set->subcases().end());
        const linalg::PivotDiagnostics& pivots = set->pivotDiagnostics();
        factorisation.largestRatio = pivots.largestRatio();
        factorisation.maxRatio = pivots.maxRatio();
        factorisation.pivotIssues = pivots.issueCount();
        if (pivots.equationOfLargestRatio() >= 0)
        {
            const base::GridComponent where = set->nameOfEquation(pivots.equationOfLargestRatio());
            factorisation.ratioGrid = where.grid;
            factorisation.ratioComponent = where.component;
        }
        const assembly::DofSets& sets = set->solveSets();
        const assembly::DofMap& dofMap = set->dofMap();
        for (base::DofIndex dof = 0; dof < sets.gSetSize(); ++dof)
            if (sets.isAutoSpc(dof))
            {
                const assembly::DofLabel label = dofMap.labelOf(dof);
                factorisation.autoSpc.emplace_back(dofMap.pointIdOf(dof),
                                                   label.kind == input::PointKind::Scalar ? 0 : label.component);
            }
        result.largestRatio = std::max(result.largestRatio, factorisation.largestRatio);
        result.factorisations.push_back(std::move(factorisation));
    }
    for (const output::SubcaseCheck& check : results.checks)
        if (!check.isCombination)
            result.subcases.push_back({check.subcase, check.epsilon, check.externalWork});
    result.singularityTables = results.singularityTables;
    for (const base::Diagnostic& diagnostic : results.diagnostics)
        if (diagnostic.severity == base::Severity::Warning)
            result.warnings.push_back(base::formatDiagnostic(diagnostic));
    return result;
}

} // namespace oos::design
