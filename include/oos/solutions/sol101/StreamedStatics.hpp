#pragma once

#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/loads/LoadBasis.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/ElementResultTable.hpp"
#include "oos/output/arrays/NodalResultBuilder.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/QuantityMap.hpp"
#include "oos/output/envelopes/EnvelopeAccumulator.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/sol101/KeptLoadBasis.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::solutions
{

// How the full path streamed orders the model (the free cut or a dissection aligned on the components): Free, the full
// path's ordering (the nested dissection of the node graph, every relation's couplings as edges); Components, the
// components never cut, the separators the connections between them, only a component above the bound cut inside
// (StreamedOptions::componentNodes); Hybrid, the connections cut first, a component cut inside when it is above the
// bound or when a cut through it is much smaller than one through its connections.
enum class StreamedDissection : std::uint8_t
{
    Free,
    Components,
    Hybrid
};

[[nodiscard]] std::string_view nameOf(StreamedDissection dissection) noexcept;

// What the full path streamed keeps of its factor (the partial factor retention):
// Auto, the regions most expensive to recompute kept from the upward pass within the memory the plan leaves
// (reduction::MemoryMode Auto and Speed); None, none (MemoryMode::LowMemory); Given, within
// StreamedOptions::retainedBytes and the memory left.
enum class StreamedRetention : std::uint8_t
{
    Auto,
    None,
    Given
};

[[nodiscard]] std::string_view nameOf(StreamedRetention retention) noexcept;

// The options of the full path streamed (StaticOptions::streamed).
struct StreamedOptions
{
    // The most bytes a region keeps (its panels and the y of its right-hand sides); nothing: the largest bound the
    // memory plan fits.
    std::optional<std::uint64_t> regionBytes;
    StreamedDissection dissection = StreamedDissection::Free;
    // The component-aligned orders: a component of more nodes than this is cut inside (0: the default, a sixteenth of
    // the model's nodes).
    std::int64_t largestUncutNodes = 0;
    // Stop after the memory plan (the pre-flight check): the record says what the solve would hold at the
    // budget (its bound and retention; the least, above the budget, when nothing fits), nothing is factorised, nothing
    // is refused.
    bool isMemoryPlanOnly = false;
    // The most values a chunk of the outputs holds (the displacements of its nodes for every basis vector); 0: the
    // default (DEFAULT_CHUNK_BYTES).
    std::int64_t chunkValues = 0;
    // The partial factor retention: its rule, and with Given the most bytes it retains.
    StreamedRetention retention = StreamedRetention::Auto;
    std::uint64_t retainedBytes = 0;
};

// The levels of the tree whose separators a record gives (StreamedSetRecord::separatorLevelDofs).
inline constexpr std::size_t SEPARATOR_LEVELS = 4;
// The steps of StreamedSetRecord::regionLadder.
inline constexpr std::size_t REGION_LADDER_STEPS = 8;
// The consumers of StreamedSetRecord::heldConsumers.
inline constexpr std::size_t HELD_CONSUMERS = 12;
// The steps of StreamedSetRecord::retentionLadder: 0, 1/8, 1/4, 1/2 and all of the values below the top.
inline constexpr std::size_t RETENTION_LADDER_STEPS = 5;

// What the full path streamed did for one constraint set (the run report's `streamed` entry, per set).
struct StreamedSetRecord
{
    std::int32_t constraintSet = 0;
    std::string dissection;
    // The components of the model (StreamedDissection::Components, Hybrid; 0 for Free), and the DOFs of the separators
    // of each of the top SEPARATOR_LEVELS levels of the tree (level 0 the root's).
    std::int32_t componentCount = 0;
    std::vector<std::int64_t> separatorLevelDofs;
    // The cost of smaller regions, from the symbolic factor alone: for the bounds of 1/2, 1/4, ... 1/256 of the
    // factor's values (REGION_LADDER_STEPS), the regions factorised again and their flops in the chosen order (what a
    // smaller budget costs in time).
    struct RegionStep
    {
        std::int64_t boundValues = 0;
        std::int64_t keptValues = 0;
        std::int64_t runs = 0;
        double recomputedFlops = 0.0;
    };
    std::vector<RegionStep> regionLadder;
    // The partial factor retention: at the chosen bound and order, for retention budgets of 0, 1/8, 1/4,
    // 1/2 and all of the values below the top (RETENTION_LADDER_STEPS), the values and regions retained and the flops
    // recomputed after them (what more memory saves).
    struct RetentionStep
    {
        std::int64_t budgetValues = 0;
        std::int64_t retainedValues = 0;
        std::int64_t retainedRegions = 0;
        double recomputedFlops = 0.0;
        double recomputedWork = 0.0; // with the sweeps of the right-hand sides (RegionSweepCost::recomputedWork)
    };
    std::vector<RetentionStep> retentionLadder;
    // The largest consumers of the budget when the memory plan is made (what it holds before the solve).
    std::vector<std::pair<std::string, std::uint64_t>> heldConsumers;
    std::int64_t equations = 0;
    std::int64_t supernodes = 0;
    std::int64_t factorValues = 0; // the panels of the whole factor (never held)
    double flops = 0.0;            // one factorisation
    std::int64_t largestFrontOrder = 0;
    std::int64_t rootSeparatorDofs = 0; // the columns of the root supernode's chain (the top separator)
    std::int64_t regionValues = 0;      // the bound taken
    std::int64_t regionRuns = 0;        // the regions factorised again (those not retained)
    double recomputedFlops = 0.0;       // their flops
    std::int64_t keptValues = 0;
    // The partial factor retention: its rule ("auto", "none" or "given"), the budget it had (the memory the plan left,
    // capped by StreamedOptions::retainedBytes), the values and regions retained and the flops their runs would take.
    std::string retention;
    std::uint64_t retentionBudgetBytes = 0;
    std::int64_t retainedValues = 0;
    std::int64_t retainedRegions = 0;
    double retainedFlops = 0.0;
    // The work of the runs recomputed and avoided: their flops and the sweeps of their right-hand sides.
    double recomputedWork = 0.0;
    double retainedWork = 0.0;
    std::int64_t belowTopValues = 0; // the values of every region below the top (all retained: the factor kept)
    std::int64_t basisVectors = 0;
    std::int64_t cases = 0;
    std::int64_t elementsPlaced = 0;
    std::int64_t pressureRows = 0;
    std::int64_t chunks = 0;
    std::int64_t largestChunkValues = 0;
    // The memory plan: what the budget held before the solve, what the solve reserves beside it (its parts), and
    // their sum (the estimate of the peak); the budget's measured peak.
    std::uint64_t heldBeforeBytes = 0;
    std::uint64_t workspaceBytes = 0;
    std::uint64_t keptBytes = 0;
    std::uint64_t retainedBytes = 0;
    std::uint64_t solutionBytes = 0;
    std::uint64_t arrayBytes = 0;
    std::uint64_t outputBytes = 0;
    std::uint64_t estimatedBytes = 0; // the parts reserved by the solve
    std::uint64_t estimatedPeakBytes = 0;
    std::uint64_t leastPeakBytes = 0;         // the least: one postorder, one panel a region
    std::uint64_t leastParallelPeakBytes = 0; // the least with the subtrees on the threads
    std::uint64_t leastWorkspaceBytes = 0;    // the least's fronts and updates
    // At the least's peak: the update matrices waiting on its stack, the supernode then made (its update, panel and
    // y_c) and that supernode's columns and rows.
    std::uint64_t leastPeakStackBytes = 0;
    std::uint64_t leastPeakMadeBytes = 0;
    std::int64_t leastPeakColumns = 0;
    std::int64_t leastPeakRows = 0;
    std::string order; // "parallel" or "postorder"
    std::uint64_t measuredPeakBytes = 0;
    bool isRelieved = false;
    // The refinement of the relieved solution (linalg::IterativeRefinement): whether made, the
    // largest relative correction of each step (the first an estimate of the error of a single solve).
    bool isRefined = false;
    std::vector<double> refinementCorrections;
    double largestPivotRatio = 0.0;
    std::int64_t pivotIssueCount = 0;
    std::int64_t autoSpcCount = 0;
    // Seconds: the constraints and singularities (the element pass), the ordering and symbolic analysis, the placement
    // and the loads' descriptions, the relief's sums, the memory plan (the estimates of its candidate bounds), the
    // solve (both passes with the outputs).
    double constraintSeconds = 0.0;
    double orderingSeconds = 0.0;
    double placementSeconds = 0.0;
    double reliefSeconds = 0.0;
    double planSeconds = 0.0;
    double solveSeconds = 0.0;
    // Of the solve (the first solve of a refined one): the upward pass, the region runs, the downward pass
    // with the outputs beside each region (linalg::RegionSweepCost's seconds).
    double upwardSeconds = 0.0;
    double regionSeconds = 0.0;
    double downwardSeconds = 0.0;
    // Of the working storage: the packed panels of the fronts made by tasks (linalg::TaskedFront).
    std::uint64_t packingBytes = 0;
    // The plan's time: the work of the upward pass, the part of the upward and region work made in
    // one postorder on the calling thread, and the estimated time in units of work on the threads.
    double upwardWork = 0.0;
    double serialWork = 0.0;
    double estimatedWork = 0.0;
    // The upward pass's measured tasks and their model on more threads (the first solve's).
    linalg::TaskModel taskModel;
};

// The static analysis of one constraint set on the full path streamed: the
// constraints and the singularities of the full path without its matrices (ConstrainedSystemOptions::diagonalBlocks:
// StreamedFront::nodeDiagonals), its ordering (StreamedDissection), the load basis of the full path
// (StaticLoadCases), the elements and loads placed at their fronts (StreamedElements, StreamedLoads, StreamedFront),
// inertia relief on rigid support motions (a free body on its conditioned supports, its solution refined by
// further streamed solves when it can be held whole), the solve without the factor
// (linalg::StreamedSolve on the front source) and the outputs region by region (StreamedOutputs).
//
// Refused before any numeric work, each with the path that gives it: glued contact, PARAM INREL = −2, a free body
// that a constraint or a grounded element holds besides its six supports, cases of one set with different output
// requests, outputs other than the displacements and the element tables (SPC and MPC forces, applied loads, grid point
// forces, strain energies), and the loads StreamedLoads refuses.
//
// Thread-safety: stateless. Ownership: none.
class StreamedStatics
{
public:
    // The default bound of an output chunk: 512 MB with its cases' displacements and tables
    // (StreamedOutputs::chunkValuesFor).
    static constexpr std::uint64_t DEFAULT_CHUNK_BYTES = std::uint64_t{512} << 20;

    struct GroupInputs
    {
        const AnalysisModel* model = nullptr;
        const assembly::LoadVectorBuilder* loads = nullptr;
        const output::ElementResultBuilder* elementTables = nullptr;
        const output::NodalResultBuilder* nodalTables = nullptr;
        std::span<const StaticSubcase> subcases;   // the analysis's
        std::span<const std::size_t> cases;        // the set's, into subcases
        std::span<const output::OutputPlan> plans; // per subcase
        const assembly::SpcSet* spc = nullptr;
        std::span<const base::DofIndex> supports; // the FreeBodySupport DOFs
        assembly::InertiaReliefMode relief = assembly::InertiaReliefMode::None;
        ConstraintSelection selection;
        linalg::PivotPolicy pivotPolicy;
        output::EnvelopeAccumulator* envelopes = nullptr;
        bool keepsCaseTables = true;
        std::uint64_t resultBytes = 0; // the result tables kept to the end (the output volume estimate)
        std::int32_t constraintSet = 0;
        const linalg::DenseKernels* kernels = nullptr;
        base::ThreadPool* pool = nullptr;
        const base::CancellationToken* cancellation = nullptr;
        assembly::LoadBasisTolerance tolerance = assembly::LoadBasisTolerance::Roundoff;
        linalg::KernelKind kernel = linalg::KernelKind::Reference;
        // The load basis with declared quantities (StaticOptions::keepsLoadBasis): no case is
        // recovered; the vectors' declared quantities (`basisQuantities`, the force tables of its elements with the
        // describers of `basisSelection`) and their grid point forces at the points of `basisFixedRows` with their
        // fixed rows are made in the downward pass (StreamedOutputs' basis mode); `firstUnit` numbers the set's first
        // unit (the units numbered over the sets, as their subcases).
        bool keepsLoadBasis = false;
        const output::QuantityMap* basisQuantities = nullptr;
        output::ElementOutputSelection basisSelection;
        BasisFixedRowRequest basisFixedRows;
        std::int32_t firstUnit = 1;
    };
    struct GroupResult
    {
        std::vector<std::vector<output::NodalResultTable>> nodal; // per case of the set
        std::vector<std::vector<output::ElementResultTable>> element;
        // With keepsLoadBasis: the set's load basis (its units and coefficients) and what the vectors gave.
        std::shared_ptr<const StaticLoadCases> loadCases;
        std::shared_ptr<KeptStreamedBasis> basis;
        std::shared_ptr<const KeptFixedRows> fixedRows; // the points, relations and loads of its fixed rows
        std::string singularityTable;
        linalg::PivotDiagnostics diagnostics;
        StreamedSetRecord record;
    };

    // Throws SolverError naming what the path refuses, MemoryBudgetError before any numeric work (the estimate, the
    // least and the largest consumers), SingularityError, OperationCancelledError.
    [[nodiscard]] static GroupResult solveGroup(const GroupInputs& inputs, const StreamedOptions& options,
                                                base::MemoryBudget& budget, base::DiagnosticSink& sink);
};

} // namespace oos::solutions
