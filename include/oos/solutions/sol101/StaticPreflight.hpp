#pragma once

#include "oos/assembly/constraints/GluedContactPairs.hpp"
#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/diagnostics/SupportReport.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace oos::solutions
{

// A glued pair of contact bodies of one subcase, as the pre-flight check lists it.
struct SubcaseGluedPair
{
    base::EntityId subcase = 1;
    assembly::GluedPair pair;
};

// What the pre-flight check of a static run found: the support report, the problems of the model,
// the sizes, and the estimates of the full path.
struct StaticPreflightReport
{
    // The cards and field values of the run's support catalog (StaticSolution step 1): refusals and supported items.
    base::SupportReport support;
    // The problems the element check, the subcases and the output plans find (errors stop a run; warnings do not).
    std::vector<base::Diagnostic> problems;
    // The glued pairs of contact bodies of each subcase (the touching ones are problems).
    std::vector<SubcaseGluedPair> gluedPairs;
    // false when a refusal or an error leaves the estimates out (they need the elements and the subcases).
    bool hasEstimates = false;
    std::int64_t gSetSize = 0;
    std::int64_t elementCount = 0;
    std::int64_t subcaseCount = 0; // the subcases and SUBCOMs a run solves and recovers
    // The volume of the requested outputs of every case (rows × components × 8 bytes), and per table kind.
    std::uint64_t outputBytes = 0;
    std::vector<std::pair<std::string, std::uint64_t>> outputBytesByKind;
    // The factorisation of the full path, estimated on the g set without its constraints (an upper bound): the
    // ordering and symbolic analysis of the element connectivity with the couplings the elimination of the rigid
    // elements, the fasteners' piercing points and the MPC equations of the first solved subcase adds.
    std::int64_t factorEquationCount = 0;
    std::int64_t factorEntryCount = 0;
    double factorFlopCount = 0.0;
    std::uint64_t factorMemoryBytes = 0; // panels, update stack and indices (SymbolicFactor::memoryPeakBytes)
    std::uint64_t stiffnessBytes = 0;    // the values of Kgg, Knn and the scalar copy the factorisation reads
};

// The pre-flight check of a static run: the steps of StaticSolution that come
// before any computation (the support check, the element check, the subcases and their output plans, the output
// volume), each problem kept instead of stopping at the first, and the estimates of the full path: the ordering and
// the symbolic analysis of the stiffness pattern give the factor's size, flops and memory exactly for the
// unconstrained g set. Nothing is assembled or factorised.
//
// The caller adds what the core does not see (cards O_o_Reader keeps outside the snapshot, unit heuristics,
// parameters, element statuses) and gathers everything in its own check report.
//
// Thread-safety: run() is const; it uses the pool of the options (ordering). Ownership: borrows the snapshot.
class StaticPreflight
{
public:
    // isFactorEstimated: false leaves out the ordering and the symbolic analysis (the factor estimates stay 0; the
    // cheap checks and the output volume only).
    StaticPreflight(const input::ModelSnapshot& snapshot, StaticOptions options, bool isFactorEstimated = true)
        : m_snapshot(&snapshot), m_options(std::move(options)), m_isFactorEstimated(isFactorEstimated)
    {
    }

    // Never throws for a problem of the model (it is a diagnostic of the report); throws MemoryBudgetError,
    // OperationCancelledError, std::invalid_argument (a subcase asked for that the Case Control does not have).
    [[nodiscard]] StaticPreflightReport run(base::MemoryBudget& budget) const;

private:
    const input::ModelSnapshot* m_snapshot;
    StaticOptions m_options;
    bool m_isFactorEstimated;
};

} // namespace oos::solutions
