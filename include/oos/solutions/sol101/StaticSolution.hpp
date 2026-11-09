#pragma once

#include "oos/assembly/loads/LoadBasis.hpp"
#include "oos/assembly/relief/GridPointWeight.hpp"
#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/QuantityMap.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/output/diagnostics/SubcaseChecks.hpp"
#include "oos/output/envelopes/EnvelopeAccumulator.hpp"
#include "oos/output/monitors/MonitorPoint3.hpp"
#include "oos/output/sections/SectionLoadRun.hpp"
#include "oos/output/sections/SectionLoadTable.hpp"
#include "oos/output/sections/ZoneRun.hpp"
#include "oos/output/sections/ZoneTable.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"
#include "oos/reduction/scheduling/KeptInteriorFactors.hpp"
#include "oos/solutions/common/AnalysisPath.hpp"
#include "oos/solutions/common/AnalysisTimings.hpp"
#include "oos/solutions/common/FactorisedSet.hpp"
#include "oos/solutions/common/ProgressMonitor.hpp"
#include "oos/solutions/common/RunCatalog.hpp"
#include "oos/solutions/common/StructureCache.hpp"
#include "oos/solutions/sol101/CondensedStatics.hpp"
#include "oos/solutions/sol101/ExternalUnitPath.hpp"
#include "oos/solutions/sol101/KeptLoadBasis.hpp"
#include "oos/solutions/sol101/StreamedStatics.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace oos::solutions
{

// What a static analysis runs and with what.
struct StaticOptions
{
    // The subcases to solve, in this order; empty: every subcase of the Case Control (SUBCASE rows of subcasesTable,
    // or the global section when the deck has no SUBCASE).
    std::vector<base::EntityId> subcases;
    base::ThreadPool* pool = nullptr; // null: one thread; not owned
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    const base::CancellationToken* cancellation = nullptr; // not owned, may be null
    ProgressMonitor* monitor = nullptr;                    // not owned, may be null
    // The precision of the load basis: the round-off (default) or the deck precision of each value.
    assembly::LoadBasisTolerance loadBasisTolerance = assembly::LoadBasisTolerance::Roundoff;
    // The envelopes computed over the load cases.
    std::vector<output::EnvelopeSpec> envelopes;
    // Whether the per-case tables of a kind that has an envelope are kept too (false: only the envelope).
    bool keepsCaseTables = true;
    // Whether the subcases recover the outputs their Case Control asks for (true, the default) or only requiredOutputs
    // and knownOutputs, with the Case Control's describers where it asks for the same kind (false: a DesignProblem
    // that reads its responses only; the tables of the deck are not made).
    bool usesCaseControlOutputs = true;
    // The memory the result tables may take; nothing: half of the memory budget left when the analysis starts.
    std::optional<std::uint64_t> outputLimitBytes;
    // Whether the element tables of every case are recovered one case at a time (output::ElementResultBuilder::build),
    // the reference of the recovery of many cases at once from the recovery slopes (buildCases), which the
    // tests compare with; false (the default): the cases of a constraint set recovered by blocks where they are enough.
    bool isRecoveredCaseByCase = false;
    // A destination of the result tables written as they come (streaming to files), used when the tables would not fit
    // outputLimitBytes: each subcase's tables are handed to it, on the analysis thread, as soon as the subcase is
    // recovered (in the order of the recovery: the constraint sets one after the other, then the SUBCOMs), and none is
    // kept in StaticResults::arrays, so that the memory they take does not grow with the number of load cases. Tables
    // that fit are kept in StaticResults::arrays and never handed over. Empty: tables that would not fit are refused
    // before the analysis (OutputVolumeError).
    std::function<void(output::ResultArrays&&)> resultSink;
    // With a resultSink, what the destination can take (the free space of the disk of a file): a volume above it is
    // refused before the analysis (OutputVolumeError); nothing: no limit.
    std::optional<std::uint64_t> sinkLimitBytes;
    // Outputs added to the requests of every solved subcase, whatever its Case Control asks (the outputs the design
    // responses of a DesignProblem read: a DesignResp1 is evaluated whether or not the deck requests its output). A
    // kind requested here and by the Case Control takes the union of their points or elements, with the describers of
    // the Case Control; a kind the Case Control does not request takes the describers given here.
    std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> requiredOutputs{};
    // Outputs added as requiredOutputs are, but that take on the condensed path only what the analysis knows without
    // recovering a unit for them (the interface's points and the units recovered for another reason): the
    // displacements of every point that the linearity check of a design point reads. On the full path the
    // same as requiredOutputs.
    std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> knownOutputs{};
    // The elements of one task of the element recovery (output::ElementResultBuilder): its RANGE_ELEMENTS but in the
    // tests of the threaded recovery, which lower it so that a small model is shared between the threads.
    std::int64_t recoveryRangeElements = output::ElementResultBuilder::RANGE_ELEMENTS;
    // The subcases whose g-set displacements the results keep (StaticResults::keptDisplacements): the preload subcases
    // of a buckling solution, whose differential stiffness is computed from them.
    std::vector<base::EntityId> keptDisplacements;
    // The analysis whose cards the support check accepts: a static run, or the preload of a buckling run (its
    // LanczosEigenMethod and BucklingEigenMethod cards are then read, not ignored).
    AnalysisKind analysis = AnalysisKind::Statics;
    // Whether the results keep each constraint set with its factor (StaticResults::systems; on the condensed path
    // StaticResults::condensedSystems, with the interface factor and the units): the static sensitivities of
    // a design point solve on them instead of factorising again (DesignProblem::evaluateStatics asks for it when
    // gradients are wanted). The kept sets stay in the memory budget until the results are dropped.
    bool keepsSystems = false;
    // The load basis handed over to the caller: the run keeps
    // the units, the coefficients and the displacements of the basis of each constraint set (StaticResults::loadBasis)
    // and recovers no case: no table, no check, no SUBCOM is made. Full path only, without sections, zones, envelopes,
    // MonitorNodeForceSum3 or external units (std::invalid_argument otherwise). A subcase with a thermal load is
    // refused (an error naming it), or with excludesThermalCases left out and listed (KeptLoadBasis::excludedCases),
    // and so is a SUBCOM with one or that combines such a subcase; a subcase loaded by line loads (PLOAD1) is refused:
    // its bar forces are in the station layout, not the superposition of its units.
    bool keepsLoadBasis = false;
    bool excludesThermalCases = false;
    // With keepsLoadBasis: a subcase with a thermal load is kept, its thermal
    // part the unit of its pair of temperature sets (the thermal family's basis: one vector per pair, coefficient 1
    // in each case of the pair), recovered with those temperatures (LoadBasisUnit::temperatureLoad). A SUBCOM with a
    // thermal load or that combines a thermal subcase is still refused (a SUBCOM's thermal strains are its own, not the
    // combination of its subcases'); the full path only, not streamed (std::invalid_argument otherwise).
    bool keepsThermalUnits = false;
    // With keepsLoadBasis: the fixed rows of the units' grid point forces kept with the basis (KeptFixedRows): none by
    // default (the element rows alone).
    BasisFixedRowRequest basisFixedRows;
    // With keepsLoadBasis on the full path streamed: the map of the declared quantities, made
    // in the downward pass (StreamedStatics), and the describers of its elements' force tables; the run keeps no
    // displacement. Null: the streamed path keeps no load basis (refused, unless fixed rows are asked).
    std::shared_ptr<const output::QuantityMap> basisQuantities;
    output::ElementOutputSelection basisSelection;
    // The condensed path (AnalysisPath::Condensed; every analysis has a full and a condensed path): the plan
    // of the units, made on the same snapshot (solutions::CondensationPlanner); null (the default) for the full path. A
    // plan that condenses no unit is the full path.
    std::shared_ptr<const reduction::CondensationPlan> condensation;
    // With a plan: whether the condensed units a design point drives (a non-empty design stamp, designStamps) are
    // recovered whatever the outputs ask (their pseudo-loads read the displacements of their elements), and
    // condensed units recovered beside those the outputs ask for (their IDs, as a DesignProblem asks for them).
    bool recoversDrivenUnits = false;
    std::vector<std::int64_t> alsoRecoveredUnits;
    // With a plan: the condensed units to recover (their IDs); nothing: those whose points or elements an output of a
    // subcase asks for (only the requested units are recovered).
    std::optional<std::vector<std::int64_t>> recoveredUnits;
    // With a plan: the condensation cache (null: every unit condensed, nothing kept; not owned), the name of the model
    // in the entries it stores, and the design stamps of the units a design point drives (part of the cache key; a
    // design point applied as an overlay of the snapshot).
    reduction::CondensationCache* cache = nullptr;
    std::string cacheModelTag;
    std::vector<UnitDesignStamp> designStamps;
    // With a plan and a cache: every unit taken from the cache built with its equations too, to check its
    // key (CondensedStatics::Inputs::verifiesCacheHits: base::InternalError when they differ from those of its entry);
    // a check meant for testing.
    bool verifiesCacheHits = false;
    // With a plan: the memory mode of the condensed path (reduction::MemoryMode): Auto keeps everything when it fits,
    // else recovers the units in sequence, else releases the equations of as few units as needed; Speed keeps
    // everything (refused when it does not fit); LowMemory recovers in sequence and releases every unit's equations.
    // The same bits in every mode. Speed by default.
    reduction::MemoryMode memoryMode = reduction::MemoryMode::Speed;
    // The section and cut loads of the analysis (sizing tools; output/sections/SectionLoadRun): checked
    // with the model before any factorisation (SolverError naming every problem of a definition), computed on the load
    // basis and combined for every load case, with their envelopes, the sink of the batches of cases and the values of
    // every case when kept (StaticResults::sectionLoads, cutLoads). Empty (the default): none. On the condensed path,
    // the units that hold an element of a section or a cut are recovered.
    output::SectionLoadRequest sectionLoads;
    // The zones of local models whose contour data the analysis computes (external local models;
    // output/sections/ZoneRun): resolved with the model before any factorisation (SolverError naming every problem),
    // computed on the load basis and combined for every load case, their data on the basis kept on request
    // (StaticResults::zones). Empty (the default): none. On the condensed path, the units that hold a member of a zone
    // are recovered.
    output::ZoneRequest zones;
    // The MonitorNodeForceSum3 of the model, their GRIDSET and ELEMSET expanded (output::MonitorPoint3;
    // MonitorNodeForceSum3 is supported in SOL 101 too): their loads in every case
    // (StaticResults::monitorPoints3Loads). On the condensed path, the units that hold an element of an ELEMSET are
    // recovered.
    std::vector<output::MonitorPoint3Definition> monitorPoints3;
    // With a plan: the work above which a unit is condensed alone with every thread (reduction::SchedulerOptions), but
    // in the tests of the scheduler, which lower it so that a small unit takes the threaded path.
    double largeUnitMinimumFlops = reduction::LARGE_UNIT_MINIMUM_FLOPS;
    // With a plan that condenses a unit: how far the run goes (CondensedStage). MemoryPlan (the pre-flight check): the
    // estimate of the condensed path in StaticResults::condensed->memory, nothing condensed, no budget or
    // output limit enforced; Condensation: the units condensed, their reduced units in
    // StaticResults::reducedUnits, no unit recovered, no table, no check. Both return before any constraint set is
    // solved; std::invalid_argument with a plan that condenses no unit.
    CondensedStage condensedStage = CondensedStage::Complete;
    // With the Interface stage (CondensedStage::Interface): each
    // constraint set's interface problem handed over (StaticResults::interfaces), its boundary basis solved too when
    // true (its solutions kept), no unit recovered, no table.
    bool solvesInterfaceExport = false;
    // With the Condensation stage: the units condensed with their mass
    // (their reduced boundary mass M* = Ψ̃ᵀ·M·Ψ̃ in ReducedUnitResult::reduction), and with a mode request (a count, or
    // a cut-off eigenvalue) their Craig-Bampton modal part (ReducedUnitResult::modal; CondensedStatics::reduceModes).
    bool condensesWithMass = false;
    reduction::ModeRequest unitModes;
    // The interface's solve of the constraint sets replaced by the caller's values (the
    // analysis otherwise that of a complete run: every table and check made from them). Per constraint set (its index
    // among the run's sets, in the order they are solved): GivenEquations, the solutions of its equations per column
    // of its boundary basis or per case (perCase); GivenLocal, the boundary displacements of the condensed unit
    // `unitId` on its K* rows, one column per subcase of caseLabels, every set taking its own (its other local DOFs at
    // zero: the outputs are then
    // restricted to that unit, CondensedStatics::unitOnlyOutputPlanOf, and its checks mean nothing). Empty (the
    // default): the interface solves. A set without an entry, or values of other sizes, is refused
    // (std::invalid_argument).
    struct GivenBoundary
    {
        std::int32_t constraintSet = 0;
        reduction::BoundarySolveMode mode = reduction::BoundarySolveMode::GivenEquations;
        bool isPerCase = false;
        std::int64_t unitId = 0;
        linalg::ConstDenseMatrixView values;    // not owned: the caller's, valid for the run
        std::vector<base::EntityId> caseLabels; // GivenLocal: the subcase of each column (bounded by the subcases)
    };
    std::vector<GivenBoundary> givenBoundaries;
    // The external units: each replaces the elements of its zone by the
    // matrices of a plugin on the contour DOFs (ExternalUnitPath), checked before any factorisation (every problem a
    // SolverError naming its zone). On the full path a dense block of Kgg (kept in the factorised sets with
    // keepsSystems); on the condensed path a unit of the interface problem, the units holding its zone kept
    // full by the plan (a condensed unit holding a replaced element is refused, naming both). Empty (the default):
    // none.
    std::vector<ExternalUnitRequest> externalUnits;
    // The structure kept between the analyses of the model (reused across design points):
    // on the full path, the sparse pattern, the ordering, the symbolic factor and the plan of the solves of each
    // constraint set taken from it when an earlier analysis built them from the same node graph, solved set and
    // positions (the results have the bits of a run without it), else built and kept there. Null (the default): built
    // for this run alone. Not owned; it outlives the run, and what the results keep of it lives on with them.
    StructureCache* structures = nullptr;
    // The interior factors kept between the analyses of the process (null: none). On the
    // condensed path a unit to condense, or to factorise again for its recovery, takes the factor of its equations
    // from it (CondensedStatics::Inputs::keptFactors), and a complete static run offers it the factors its units still
    // hold at its end when sharedBudget is the budget given to run() (the kept factors keep it alive). Not owned.
    reduction::KeptInteriorFactors* keptFactors = nullptr;
    std::shared_ptr<base::MemoryBudget> sharedBudget;
    // The full path streamed (StreamedStatics): the full
    // path's equations solved without the factor, no matrix of the model assembled, the outputs made region by
    // region. Refused (std::invalid_argument) with a condensation plan, keepsSystems, keepsLoadBasis, sections, zones,
    // MonitorNodeForceSum3, external units, kept displacements or SUBCOMs; what it does not give is refused or warned
    // by StreamedStatics.
    bool isStreamed = false;
    StreamedOptions streamed;
};

// The g-set displacements of one subcase (one value per DOF of assembly::DofMap), kept on request.
struct KeptDisplacements
{
    base::EntityId subcase = 0;
    base::BudgetedBuffer<double> values;
};

// The seconds spent in each stage of a static analysis (summed over the constraint sets).
struct StaticTimings
{
    double setup = 0.0;         // support check, element check, DOF sets, connectivity, node graph, pattern
    double assembly = 0.0;      // stiffness, m-set elimination, singularities, mass, loads
    double factorisation = 0.0; // ordering, symbolic analysis, numeric factorisation
    double solve = 0.0;
    double recovery = 0.0;  // displacements, SPC and MPC forces, result tables
    double loadBasis = 0.0; // the compact load matrices and the choice of the bases
    // The parts of setup, assembly and factorisation on the full path (the condensed path reports its units apart).
    AnalysisPartTimings parts;
};

// The load basis of one family for one constraint set (the report gives, per load family, the basis taken and its
// size), with the subcases it serves.
struct LoadBasisRecord
{
    std::int32_t constraintSet = 0;       // the index of the constraint set (the factorisation) in the run
    std::vector<base::EntityId> subcases; // the solved subcases of that set
    assembly::LoadBasisReport report;
};

// The loads of the MonitorNodeForceSum3 of a case (StaticResults::monitorPoints3Loads): per point of
// StaticOptions::monitorPoints3, in that order, the six loads along and about the axes of its CD.
struct MonitorPoint3Loads
{
    base::EntityId subcase = 0;
    std::vector<std::array<double, 6>> loads; // bounded by the monitor points
};

// With LoadBasisTolerance::WrittenPrecision, a worst case solved again on its own: its relative load residual
// and the deviation of its displacements from the combination, max |u_combined − u_alone| / max |u_alone|.
struct DeckDeviation
{
    base::EntityId subcase = 0;
    assembly::LoadFamily family = assembly::LoadFamily::Distributed;
    double loadResidual = 0.0;
    double displacementDeviation = 0.0;
};

// The results of a static analysis and its report.
// The inertia relief of one constraint set on the full path: whether its support motions
// are the rigid-body motions (a body free apart from six FreeBodySupport DOFs: the largest reaction of the rigid motion
// within assembly::RIGID_SUPPORT_TOLERANCE of the largest stiffness of the model) or solved from the stiffness (D_l =
// −K_ll⁻¹·K_lr), that reaction ratio (-1 when not computed: other than six independent supports), the supports the set
// was solved on when they are not its FreeBodySupport DOFs (assembly::chooseConditionedSupports; empty otherwise), by
// point ID and component, and the iterative refinement of the relieved solution on those supports: the largest
// correction of each step relative to the largest displacement of its basis vector (the first one an estimate of the
// error of a single solve: the condition number of the free body times the round-off; empty when not refined).
struct ReliefRecord
{
    struct Support
    {
        base::EntityId point = 0;
        std::int32_t component = 0;
    };
    std::int32_t constraintSet = 0;
    bool isRigid = false;
    double reaction = -1.0;
    std::vector<Support> solveSupports;
    std::vector<double> refinementCorrections;
};

struct StaticResults
{
    output::ResultArrays arrays;    // empty when the tables were handed to StaticOptions::resultSink
    bool areTablesStreamed = false; // the tables went to StaticOptions::resultSink (they would not fit in memory)
    std::vector<base::Diagnostic> diagnostics;  // the warnings and information of the run (errors stop it)
    std::vector<std::string> singularityTables; // the table of the singular components of each constraint set, as text
    std::vector<std::string> usetTables;        // with PARAM USETPRT >= 0, the USET table of each constraint set
    std::int64_t gSetSize = 0;
    std::int64_t largestFsetSize = 0;
    double largestPivotRatio = 0.0;      // the largest MAXRATIO ratio of the factorisations
    std::int32_t constraintSetCount = 0; // the factorisations made
    std::string kernel;
    StaticTimings timings;
    std::vector<output::SubcaseCheck> checks;                 // per subcase and SUBCOM, in the order of the results
    std::optional<assembly::GridPointWeight> gridPointWeight; // with PARAM GRDPNT
    std::vector<output::EnvelopeTable> envelopes;
    std::vector<LoadBasisRecord> loadBases;
    std::vector<ReliefRecord> reliefs; // per constraint set with inertia relief (the full path)
    std::vector<DeckDeviation> deckDeviations;
    std::int64_t solvedVectorCount = 0; // the right-hand sides solved: the basis vectors of every constraint set
    std::int64_t loadCaseCount = 0;     // the subcases solved
    std::vector<KeptDisplacements> keptDisplacements; // those of StaticOptions::keptDisplacements, in solve order
    // With StaticOptions::keepsSystems, each constraint set with its factor, in the order of the factorisations.
    std::vector<std::shared_ptr<const FactorisedSet>> systems;
    // With StaticOptions::keepsLoadBasis, the load basis of every constraint set (null otherwise).
    std::shared_ptr<KeptLoadBasis> loadBasis;
    // The same on the condensed path: each constraint set with its interface factor and the units.
    std::vector<std::shared_ptr<const CondensedFactorisedSet>> condensedSystems;
    // The path taken, and on the condensed path its report (units condensed and recovered, their load bases and
    // recoveries, the interface problem of each constraint set).
    AnalysisPath path = AnalysisPath::Full;
    std::optional<CondensedRunRecord> condensed;
    // On the full path streamed: what each constraint set did (StreamedSetRecord).
    std::vector<StreamedSetRecord> streamed;
    // The section and cut loads of StaticOptions::sectionLoads (nothing when none was asked for).
    std::optional<output::SectionLoadTable> sectionLoads;
    std::optional<output::SectionLoadTable> cutLoads;
    // The contour data of the zones of StaticOptions::zones, in their order (empty when none was asked for).
    std::vector<output::ZoneTable> zones;
    // The MonitorNodeForceSum3 of StaticOptions::monitorPoints3, and their loads in each case (MonitorPoint3Loads, in
    // the order of the checks: the subcases, then the SUBCOMs).
    std::vector<output::MonitorPoint3Definition> monitorPoints3;
    std::vector<MonitorPoint3Loads> monitorPoints3Loads;
    // With CondensedStage::Condensation: each condensed unit's reduced unit and the coefficients of the load cases on
    // its patterns, by plan order, and the labels of those cases (the columns of the coefficients).
    std::vector<ReducedUnitResult> reducedUnits;
    std::vector<base::EntityId> reducedCaseLabels;
    // With CondensedStage::Interface: the interface problem of each constraint set, in solve order.
    std::vector<InterfaceExport> interfaces;
    // The external units of StaticOptions::externalUnits, in their order: their placement, checks, contour data per
    // case and load bases (ExternalUnitRecord).
    std::vector<ExternalUnitRecord> externalUnits;
    // With StaticOptions::structures: what the constraint sets of the run took from the cache and built for it, and
    // the bytes the cache holds after the run.
    bool isStructureCacheUsed = false;
    StructureCacheCounts structureReuse;
    std::uint64_t structureCacheBytes = 0;
};

// The linear static solution on the full model (SOL 101): for each subcase,
// Kgg·ug = Pg with the single-point constraints, the multipoint constraints and rigid elements eliminated, the grid
// point singularities constrained by AUTOSPC, inertia relief when PARAM INREL asks for it, the load cases solved on
// their load basis, and the outputs the Case Control requests with the equilibrium checks (OLOAD and SPC force
// resultants, EPSILON).
//
// The steps (src/assembly/README.md, "How a static solve uses them"; src/solutions/README.md):
//  1. the support check of the run (RunCatalog), the element check, the subcases (StaticSubcases: SUBCASE and SUBCOM;
//     SYM, SYMCOM, REPCASE refused) and their output plans (OutputPlan): an unsupported card or a problem of the model
//     stops the run before any computation, with every problem named (UnsupportedCardError, SolverError);
//  2. the volume of the requested outputs is estimated (OutputVolumeEstimate) and an output that would not fit is
//     refused before the analysis with its remedies (OutputVolumeError);
//  3. the solved subcases are grouped by constraint set (the same MPC set, the same constrained DOFs with
//  EnforcedMotionConstr, the
//     same FreeBodySupport DOFs and PARAM INREL): one assembly and one factorisation of K_ll per group;
//  4. per group (ConstrainedSystem, shared with the eigenvalue solutions as AnalysisModel is for step 1): Gm, the DOF
//     sets (the r set of FreeBodySupport, or the supports PARAM INREL = −2 chooses), the pattern, Kgg, Knn, the grid
//     point singularity processor, the l-set factorisation (FsetSolver; kept with StaticOptions::keepsSystems), Mgg
//     when an inertial load, inertia relief or PARAM GRDPNT needs it (the grid point weight generator is computed from
//     the first one), the support modes of inertia relief (D_l = −K_ll⁻¹·K_lr) and InertiaRelief;
//  5. the load cases of the group and their load basis per family (StaticLoadCases, LoadBasis): the basis vectors are
//     made (inertia relief applied), reduced to the l set with their enforced displacements, solved at once; their
//     displacements, residual (EPSILON), SPC and MPC forces computed;
//  6. the cases are combined from the basis batch by batch (ResultCombination: displacements, applied loads, SPC and
//     MPC forces, reduced loads and residuals), then each is recovered: its nodal tables on their SETs, its element
//     tables (ElementResultBuilder with the case's temperatures and line loads; the non-linear columns therefore
//     exact), GPFORCE and ESE (StiffnessRecovery), its checks (OLOAD and SPC force resultants, EPSILON, external work,
//     the support accelerations), and the envelopes (EnvelopeAccumulator);
//  7. with LoadBasisTolerance::WrittenPrecision, the worst cases of each family are solved on their own and their
//  deviation
//     reported;
//  8. the SUBCOMs combine the results of their subcases (displacements, applied loads, SPC and MPC forces) and are
//     recovered with their own temperatures and the scaled line loads of their terms.
//
// Determinism: every sum is in a fixed order (assembly by element order, serial elimination, loads by table rows, the
// load basis serial or column by column, the supernodal factorisation and solves and the combination products
// deterministic for any number of threads), so the results have the same bits whatever the pool.
//
// Thread-safety: run() is const; it uses the pool of the options. Ownership: borrows the snapshot, which must outlive
// the call.
class StaticSolution
{
public:
    StaticSolution(const input::ModelSnapshot& snapshot, StaticOptions options)
        : m_snapshot(&snapshot), m_options(std::move(options))
    {
    }

    // Throws UnsupportedCardError (cards not supported), SolverError (problems of the model, listed),
    // SingularityError (a mechanism, with its DOFs, and its zero-energy shapes on the full path; on the condensed path
    // the message says how to get them), MemoryBudgetError, OperationCancelledError, std::invalid_argument (a subcase
    // asked for that the Case Control does not have).
    [[nodiscard]] StaticResults run(base::MemoryBudget& budget) const;

    // The subcases a run solves: those of the options, or every analysis subcase of the snapshot.
    [[nodiscard]] std::vector<base::EntityId> subcasesToSolve() const;

private:
    // run() on the path of the options.
    [[nodiscard]] StaticResults runOnPath(base::MemoryBudget& budget) const;

    const input::ModelSnapshot* m_snapshot;
    StaticOptions m_options;
};

} // namespace oos::solutions
