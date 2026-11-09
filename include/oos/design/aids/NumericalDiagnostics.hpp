#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/problem/DesignProblem.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

// The numerical state of one factorisation (one constraint set) of an analysis.
struct FactorisationDiagnostics
{
    std::vector<std::int32_t> subcases;
    double largestRatio = 0.0;       // the largest a_jj/d_j of the factorisation (PARAM MAXRATIO's ratio)
    std::int32_t ratioGrid = 0;      // where: the point and component of its equation (0: unknown)
    std::int32_t ratioComponent = 0; //
    double maxRatio = 0.0;           // the MAXRATIO the analysis tested against
    std::int64_t pivotIssues = 0;    // the pivots above MAXRATIO
    std::vector<std::pair<std::int32_t, std::int32_t>> autoSpc; // the (grid, component) AUTOSPC constrained
};

// The checks of one subcase: EPSILON and the external work of the static solution.
struct SubcaseDiagnostics
{
    std::int32_t subcase = 0;
    double epsilon = 0.0;
    double externalWork = 0.0;
};

struct NumericalDiagnosticsResult
{
    double largestRatio = 0.0; // over the factorisations
    std::vector<FactorisationDiagnostics> factorisations;
    std::vector<SubcaseDiagnostics> subcases;
    std::vector<std::string> singularityTables; // the table of the singular components of each constraint set
    std::vector<std::string> warnings;          // the warnings of the analysis
};

// The numerical diagnostics of an analysis of a design point (an optimisation aid beyond gradients: per analysis,
// the largest factor diagonal ratio (MAXRATIO) and where, the AUTOSPC applied, EPSILON, the external work; a
// gradient computed on an analysis with a warning carries the warning). No computation
// beyond reading the analysis: the pivot diagnostics of the factorisations the analysis kept (StaticResults::systems;
// without them the largest ratio only, with no place), the AUTOSPC DOFs of their solved sets, the checks of the
// subcases and the warnings.
//
// Thread-safety: stateless. Ownership: none kept.
class NumericalDiagnostics
{
public:
    [[nodiscard]] static NumericalDiagnosticsResult of(const StaticDesignEvaluation& analysis);
};

} // namespace oos::design
