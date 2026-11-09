#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/solutions/common/RunCatalog.hpp"

#include <cstddef>
#include <string>

namespace oos::solutions
{

// How many problems an error message of a run lists before "and N more".
inline constexpr std::size_t LISTED_PROBLEM_COUNT = 20;

// Throws SolverError listing the error diagnostics of the sink, when there are any: the problems of the model that stop
// the analysis before `stage` ("the assembly", "the solve"), each named, and what to do.
void stopOnErrors(const base::DiagnosticSink& sink, const std::string& stage);

// The support check of a run (checkSupport): the ignored cards become warnings of the run (reported to `sink`), and a
// refusal stops it with UnsupportedCardError listing every unsupported card or field.
void reportSupport(const input::ModelSnapshot& snapshot, const elements::ElementRegistry& registry,
                   AnalysisKind analysis, base::DiagnosticSink& sink);

// A factor kept with failed pivots (PARAM BAILOUT -1; a stopping policy throws SingularityError instead): the run goes
// on, and a warning of `subcase` names the ill-conditioned DOFs (through `namer`), so that an inaccurate result is
// never silent (a singular or ill-conditioned system is reported with its location). Nothing when the ratio test
// passed.
void reportKeptPivots(const linalg::PivotDiagnostics& pivots, const linalg::EquationNamer& namer,
                      base::EntityId subcase, base::DiagnosticSink& sink);

} // namespace oos::solutions
