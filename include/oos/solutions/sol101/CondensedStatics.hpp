#pragma once

#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/loads/LoadBasis.hpp"
#include "oos/assembly/loads/LoadFamily.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/reduction/cache/CacheEntry.hpp"
#include "oos/reduction/cache/CondensationCache.hpp"
#include "oos/reduction/condense/CraigBampton.hpp"
#include "oos/reduction/condense/ReducedUnit.hpp"
#include "oos/reduction/interface/BoundaryBasis.hpp"
#include "oos/reduction/interface/InterfaceProblem.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"
#include "oos/reduction/recovery/UnitRecovery.hpp"
#include "oos/reduction/scheduling/KeptInteriorFactors.hpp"
#include "oos/reduction/scheduling/MemoryMode.hpp"
#include "oos/reduction/scheduling/UnitScheduler.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/StructureCache.hpp"
#include "oos/solutions/sol101/CondensedFactorisedSet.hpp"
#include "oos/solutions/sol101/ExternalUnitPath.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::solutions
{

// What one condensed unit did in a run (StaticResults::condensed).
struct CondensedUnitRecord
{
    std::int64_t unitId = 0;
    std::int64_t boundaryDofCount = 0;
    std::int64_t interiorDofCount = 0;
    std::int64_t spiderCount = 0;  // absorbed by change of basis
    std::int64_t patternCount = 0; // its load basis (and the six inertial patterns of inertia relief)
    bool isRecovered = false;
    std::vector<assembly::LoadBasisReport> loadBases;      // per family that loads it
    std::vector<reduction::RecoveryStatistics> recoveries; // per constraint set, when recovered
    // The condensation cache: whether the unit was condensed in this run (its interior factorised for K*) or taken from
    // the cache, where from, why a lookup missed, and why a new condensation was not stored.
    bool isCondensedNow = true;
    std::string cacheOutcome; // "memory_hit", "disk_hit", "miss"; empty without a cache
    std::string cacheCause;   // on a miss
    bool isCacheStored = false;
    std::string cacheNote; // why it was not stored
    // A unit taken from the cache and recovered: its interior factorised for the recovery (its K* checked against the
    // cached one, to the bit).
    bool isRefactorisedForRecovery = false;
    // A unit taken from the cache built without its equations (the analysis reads none of them), from the
    // boundary data an earlier build of its key left in the structure cache (isBoundaryDataReused), or from its
    // stiffness alone.
    bool isWithoutEquations = false;
    bool isBoundaryDataReused = false;
    // The memory budget: the wave of the scheduler it was condensed in (reduction::UnitScheduler, from 0), the threads
    // of its factorisation (0 when taken from the cache), what became of its interior factor
    // (reduction::nameOf(RetentionDecision), and why; empty when taken from the cache), the bytes that factor held, and
    // how many times its recovery factorised its interior again (a released factor, or one never made: the cache).
    std::int32_t wave = 0;
    std::int32_t threadCount = 0;
    std::string factorRetention;
    std::string factorRetentionReason;
    std::uint64_t factorBytes = 0;
    std::int32_t refactorisationCount = 0;
    // Its estimated footprint (reduction::UnitFootprint): what it keeps to the end, the peak of its condensation.
    std::uint64_t estimatedKeptBytes = 0;
    std::uint64_t estimatedCondensationBytes = 0;
    // The memory mode: its equations released after its condensation (its factor with them), and how many times they
    // were built again for its recovery.
    bool isEquationReleased = false;
    std::int32_t rebuildCount = 0;
    // Condensed without its interior factor, and recovered by regions of its elimination tree (its factor above the
    // strategy's bound), with the count of those recoveries.
    bool isCondensedWithoutFactor = false;
    bool isRecoveredByRegions = false;
    // Its interior factor taken from those an earlier analysis of the process kept (Inputs::keptFactors), for its
    // condensation (no factorisation: its patterns condensed by its forward sweep) or for its recovery (in place of a
    // factorisation again).
    bool isInteriorFactorReused = false;
    std::int64_t regionRecoveryCount = 0;
    // Craig-Bampton (CondensedStatics::reduceModes): the fixed-interface modes it keeps and the highest of
    // their eigenvalues; where its modal part (K_qq, M_aq, M_qq) came from ("computed", "memory_hit", "disk_hit"); what
    // became of its modes, the recovery data its recovery or a buckling analysis needs ("computed" in this run,
    // "memory_hit" or "disk_hit" from the cache, "recomputed" when the cache did not have them: evicted under
    // cache_recovery_max_size or never stored, computed again to the same bits; empty when not needed), with the
    // cause; and the factorisations of its Lanczos solves.
    std::int64_t modeCount = 0;
    double highestModeEigenvalue = 0.0;
    std::string modalOutcome;
    std::string modesOutcome;
    std::string modesNote;
    std::int64_t modeFactorisationCount = 0;
    bool isModesStored = false; // its modes were stored in the cache in this run
};

// The Craig-Bampton request of an eigenvalue analysis on the condensed path (CondensedStatics::reduceModes): the
// fixed-interface modes every unit keeps, as a count or a cut-off, and where the cut-off comes from.
struct CraigBamptonRecord
{
    std::int32_t modeCount = 0;       // the plan's count per unit; 0: the cut-off below
    double cutoffEigenvalue = 0.0;    // λ_c = CB_CUTOFF_FACTOR² × referenceEigenvalue
    double cutoffFrequency = 0.0;     // √λ_c / 2π (cycles)
    double cutoffFactor = 0.0;        // CB_CUTOFF_FACTOR
    double referenceEigenvalue = 0.0; // the highest eigenvalue requested
    double referenceFrequency = 0.0;
    // How the highest frequency requested was found: "V2" (the upper bound of the request), "statically condensed"
    // (the highest root of the request on the statically condensed model: an upper bound of the true one), "count"
    // (a count per unit: no cut-off).
    std::string reference;
};

// What the memory budget did on the condensed path.
struct CondensedMemoryRecord
{
    std::uint64_t limitBytes = 0;
    std::uint64_t baselineBytes = 0;         // held when the condensation started (the model, its sets and loads)
    std::uint64_t estimatedMinimumBytes = 0; // the smallest budget the analysis fits (reduction::CondensedMemoryPlan)
    std::uint64_t estimatedCondensationBytes = 0; // its condensation phase, one unit at a time, beyond the baseline
    std::uint64_t estimatedLaterBytes = 0;        // its interface and recovery phase beyond what the units keep
    std::uint64_t measuredPeakBytes = 0;          // the budget's peak at the end of the last constraint set
    std::int32_t waveCount = 0;
    std::int32_t largestWave = 0; // the most units condensed at once
    std::int32_t threadCount = 0; // the threads of the run
    std::uint64_t keptFactorBytes = 0;
    std::int64_t keptFactorCount = 0;
    std::vector<std::int64_t> releasedForBudget;        // the recovered units whose factor the budget released
    std::vector<base::MemoryConsumer> largestConsumers; // of the estimate, largest first
    // The memory mode (reduction::MemoryMode, CondensedMemoryPlan): the mode the analysis ran in, its
    // recovery ("all at once", "in sequence"), the units whose equations it released, why (what "auto" switched), the
    // smallest budget of each strategy (every recovery at once; in sequence; in sequence with every unit's equations
    // released: 0 when the analysis does not allow it), the builds of units' equations again at their recovery and the
    // waves of the recoveries in sequence (every constraint set and batch).
    std::string mode;
    std::string recovery;
    std::int64_t releasedEquationCount = 0;
    std::string modeReason;
    std::uint64_t estimatedAllAtOnceBytes = 0;
    std::uint64_t estimatedSequentialBytes = 0;
    std::uint64_t estimatedReleasedBytes = 0;
    // The units condensed without their factor, the region bound of their recoveries (0: their interiors
    // factorised again whole), and the smallest budgets without the factors and with the least regions.
    bool isWithoutFactor = false;
    std::uint64_t regionBytes = 0;
    std::uint64_t estimatedWithoutFactorBytes = 0;
    std::uint64_t estimatedRegionsBytes = 0;
    // The interface problem solved without its factor, its region bound (its factor's bytes or more: every panel kept),
    // its factor's bytes, the smallest budget with it (beside the units' least strategy), and the flops its regions are
    // estimated to factorise again.
    bool isInterfaceWithoutFactor = false;
    std::uint64_t interfaceRegionBytes = 0;
    std::uint64_t interfaceFactorBytes = 0;
    std::uint64_t estimatedInterfaceRegionsBytes = 0;
    double estimatedInterfaceRecomputedFlops = 0.0;
    std::int64_t rebuiltUnitCount = 0;
    std::int64_t recoveryWaveCount = 0;
};

// The design stamp of a unit of the analysis (part of its cache key: the hash of the values a design point
// gives to the fields of the unit and to the positions of its grid points; design::computeDesignStamps), by the ID of
// the unit O_o_Reader gives (a component; FieldTargets::unitIds).
struct UnitDesignStamp
{
    std::int64_t unitId = 0;
    std::uint64_t stamp = 0;
};

// What the interface problem of one constraint set did.
struct CondensedSetRecord
{
    std::int32_t constraintSet = 0;
    std::int64_t interfaceDofCount = 0; // its local DOFs
    std::int64_t equationCount = 0;
    std::int64_t distinctRightHandSides = 0; // Q
    std::int64_t solvedColumnCount = 0;      // Q, or the cases when Q exceeds them
    bool isCaseBasis = false;
    // The memory budget: the threads of the interface factorisation (one when the threaded working storage did not
    // fit), the cases recovered together and why (lowered from the largest batch when the budget left does not hold
    // it).
    std::int32_t interfaceThreadCount = 1;
    std::int64_t caseBatchSize = 0;
    std::string caseBatchNote;
    // The waves of units of its recoveries and cases (consecutive units, bounded by the budget left; every batch), and
    // the units whose equations were built again for their recovery.
    std::int64_t recoveryWaveCount = 0;
    std::int64_t rebuiltUnitCount = 0;
    // The seconds of its phases: the relations and the interface problem (assembled, its AUTOSPC, factorised); the
    // boundary basis (its solves); the recoveries (the interior factors made again and the units' basis solutions); the
    // cases (their batches: the combinations, the recovered interiors, the g-set vectors and the caller's tables).
    double interfaceSeconds = 0.0;
    double basisSeconds = 0.0;
    double recoverySeconds = 0.0;
    double caseSeconds = 0.0;
    // Inertia relief: the share of the support motions that is not rigid (their misfit to the rigid fit, over the
    // largest motion) and the reactions of the rigid fit over the largest force of a unit support motion (refused above
    // the tolerance); -1 without relief.
    double reliefMisfit = -1.0;
    double reliefReaction = -1.0;
    // The supports the interface was solved on in place of the FreeBodySupport DOFs (six translations of the farthest
    // boundary points; empty: the FreeBodySupport DOFs), by point ID and component.
    struct Support
    {
        base::EntityId point = 0;
        std::int32_t component = 0;
    };
    std::vector<Support> reliefSolveSupports;
    // The units recovered unit by unit (GroupInputs::visitUnit): each recovered unit's recovery made once for every
    // case of the set, not once per batch of cases.
    bool isUnitByUnit = false;
    // The interface problem's factor (its panels' values; without it, those it would hold) and whether it was kept;
    // without it, its solves (each a factorisation streamed), the columns they took, the subtrees their regions
    // factorised again and the flops of those, the most panel values a region kept.
    std::int64_t interfaceFactorValues = 0;
    bool isInterfaceFactorKept = true;
    std::int64_t interfaceSolveCount = 0;
    std::int64_t interfaceSolvedColumns = 0;
    std::int64_t interfaceRegionRuns = 0;
    double interfaceRecomputedFlops = 0.0;
    std::int64_t interfaceKeptPanelValues = 0;
};

// How far a condensed static run goes (StaticOptions::condensedStage).
enum class CondensedStage : std::uint8_t
{
    Complete,     // every phase: the condensation, the interface problem, the recovery, the outputs
    MemoryPlan,   // the light setup and the memory plan only (a check: CondensedRunRecord::memory is the estimate of
                  // the condensed path; no unit built, looked up in the cache or factorised, the budget not enforced)
    Condensation, // the units condensed (and stored in the cache), nothing after: no interface problem, no recovery, no
                  // table (the caller reads CondensedStatics::reducedUnits)
    Interface     // the units condensed, then per constraint set its interface problem assembled and the right-hand
                  // sides of its boundary basis (solved when asked), handed over; no unit recovered, no case, no table
                  // (the caller reads CondensedStatics::takeInterfaceExports)
};

// The interface problem of one constraint set as the analysis assembles it (CondensedStage::Interface): the K* of the
// condensed units and of the external units, the elements of the units kept full and of the connection system, its
// relations eliminated and its held DOFs removed as the analysis does (reduction::InterfaceProblem), with the
// right-hand sides of its boundary basis (reduction::BoundaryBasis), so that another program can solve it and give the
// solution back (GroupInputs::boundaryMode).
//
// The local DOFs are the interface's points × their DOFs; the equations the local DOFs neither held nor dependent, in
// the order of the factor. A solution x of the equations (one column per column of the basis) gives the local
// displacements u: u at the equations' DOFs x, at the held DOFs their enforced values, at the dependent DOFs Σ c·u over
// their relation's terms (each term on an equation or a held DOF), in the order of the relations.
struct InterfaceExport
{
    std::int32_t constraintSet = 0;
    std::vector<base::EntityId> subcases; // the labels of its cases, in their order (bounded by the subcases)
    // Per local DOF: its label (a grid or scalar point, or the Fastener or Weld and side of a generated point;
    // a grid point's components in its displacement system, a generated point's in the basic system).
    base::BudgetedBuffer<reduction::UnitDofLabel> localLabels;
    base::BudgetedBuffer<std::int64_t> localOfEquation; // per equation
    base::BudgetedBuffer<std::int64_t> heldLocals;      // increasing
    // The relations: per row its dependent local DOF, its terms [relationOffsets[r], relationOffsets[r + 1]).
    base::BudgetedBuffer<std::int64_t> relationDependents;
    base::BudgetedBuffer<std::int64_t> relationOffsets;
    base::BudgetedBuffer<std::int64_t> relationTermDofs;
    base::BudgetedBuffer<double> relationTermCoefficients;
    // K_ll on the equations: its lower triangle in compressed sparse columns (linalg::SymmetricCsc's arrays).
    base::BudgetedBuffer<std::int64_t> matrixOffsets;
    base::BudgetedBuffer<std::int32_t> matrixRows;
    base::BudgetedBuffer<double> matrixValues;
    // The boundary basis: the right-hand sides F (equations × columns), the enforced values of the held DOFs (local ×
    // columns; no column: none), the coefficients β of every case (columns × cases: a case's right-hand side is F·β),
    // what each column is (empty in the case basis, whose columns are the cases), and when solved the solutions
    // (equations × columns; empty otherwise).
    linalg::DenseMatrix rightHandSides;
    linalg::DenseMatrix enforced;
    linalg::DenseMatrix coefficients;
    bool isCaseBasis = false;
    struct Column
    {
        std::int64_t unitId = 0;   // the condensed unit whose pattern it is, or 0
        std::int64_t pattern = -1; // the unit's pattern, or -1
        std::int64_t block = -1;   // the block of the interface's own loads (0: its load basis), or -1
        std::int64_t vector = -1;  // the block's vector, or -1
    };
    std::vector<Column> columns; // bounded by the basis
    linalg::DenseMatrix solutions;
    // Per condensed unit, in the order of the units: its ID and the local DOFs of its K* rows (ReducedUnit labels'
    // order); per external unit: its zone and the local DOFs of its contour.
    std::vector<std::int64_t> unitIds;                     // bounded by the units
    std::vector<std::vector<std::int64_t>> unitLocals;     // bounded by the units' boundaries
    std::vector<std::string> externalZoneNames;            // bounded by the external units
    std::vector<std::vector<std::int64_t>> externalLocals; // bounded by their contours
};

// A condensed unit as a condensation-only run gives it (CondensedStage::Condensation): K*, f* per load basis vector,
// DOF labels.
struct ReducedUnitResult
{
    std::int64_t unitId = 0;
    // Its reduced unit (labels, K*, f*, the inertial patterns, the rigid-body mass), a copy in the budget of the run
    // (reducedUnits): an entry of the condensation cache is budgeted in the cache's budget, which a caller holding the
    // results may outlive (a unit taken from the cache, its solver released first, would give its reservation back to
    // a budget gone).
    std::shared_ptr<const reduction::ReducedUnit> reduced;
    // The unit's load basis: the first loadPatternCount columns of reduced->patterns() (the six inertial patterns of
    // inertia relief follow when a subcase has it), the family of each vector, and the coefficients α of every solved
    // load case on them (loadPatternCount × cases, CondensedStatics::caseLabels order): the condensed loads of a case
    // are f*·α. The loads at the unit's boundary points are not in f* (they are the interface's, so each acts once).
    std::size_t loadPatternCount = 0;
    std::vector<assembly::LoadFamily> families; // bounded by the unit's basis
    linalg::DenseMatrix coefficients;
    std::int64_t interiorDofCount = 0;
    // What the cache keeps of the unit beside its reduced unit, copied as it is: its reduced boundary mass M* =
    // Ψ̃ᵀ·M·Ψ̃ when the run condensed the units with their mass (Inputs::isCraigBampton; empty otherwise), its
    // Craig-Bampton modal part when the run asked for fixed-interface modes (reduceModes; null otherwise), and the
    // position in the basic system of each point generated on a Fastener or Weld side among its labels.
    linalg::DenseMatrix boundaryMass;
    std::shared_ptr<const reduction::ModalReduction> modal;
    struct GeneratedPoint
    {
        base::EntityId fastener = 0;
        reduction::FastenerSide side = reduction::FastenerSide::None;
        base::Vec3 position{};
    };
    std::vector<GeneratedPoint> generatedPoints; // bounded by the unit's boundary
};

// The report of a condensed static analysis.
struct CondensedRunRecord
{
    std::vector<CondensedUnitRecord> units; // the condensed units, by increasing ID
    std::vector<std::int64_t> fullUnits;    // the units kept full
    std::vector<CondensedSetRecord> constraintSets;
    std::vector<assembly::LoadBasisReport> interfaceLoadBases; // the interface's own loads, per family
    CondensedMemoryRecord memory;
    // An eigenvalue analysis: the request of its last reduction (reduceModes).
    std::optional<CraigBamptonRecord> craigBampton;
    // The seconds of the condensation's phases: the preparation (steps 1 to 4: what the subcases share, the units'
    // relations and descriptions, the load cases and the units' load bases, the memory plan), the condensation (step 5:
    // the waves of units) and the interface's own loads with the outputs of the units not recovered (steps 6 and 7).
    double preparationSeconds = 0.0;
    double condensationSeconds = 0.0;
    double interfaceLoadsSeconds = 0.0;
    // The units whose boundary data the run gave the structure cache.
    std::int64_t keptUnitBoundaryCount = 0;
};

// The condensed path of the linear static solution (AnalysisPath::Condensed of the SOL 101 driver): the units of a plan
// condensed once for every subcase, then, per constraint set, the interface problem (reduction::InterfaceProblem), the
// boundary basis of its cases (reduction::BoundaryBasis), the recovery of the units asked for
// (reduction::UnitRecovery), and each case's g-set vectors for the outputs and checks of StaticSolution, the same as
// the full path's to round-off.
//
// Built once per run (the condensation, in this order):
// 1. what every subcase shares (CommonConstraints): the DOFs held in every subcase, the MPC cards common to all;
// 2. the relations eliminated inside the units (reduction::PlannedUnits::unitRowFilter) and the description of each
//    condensed unit;
// 3. the load cases of every solved subcase (StaticLoadCases with its family descriptions) and, per unit, the load
//    basis chosen on the rows that load it (reduction::UnitLoadBasis: its interior points, its elements, the
//    DOFs of its points held in every subcase), and the units to recover;
// 4. the memory plan: each unit's footprint from the plan's estimates
//    and its patterns (reduction::UnitFootprint), the interface phase (reduction::InterfaceFootprint), one case of a
//    batch and the result tables (Inputs::resultBytes): a budget the analysis cannot fit one unit at a time is refused
//    here, before any unit is condensed, with the estimate and the largest consumers (base::MemoryBudgetError);
// 5. the units condensed in waves (reduction::UnitScheduler: by decreasing work, several units on one thread each while
//    their estimated peaks fit the budget left, a large unit or the only one the budget admits alone with every thread
//    the memory of its factorisation allows): per wave, their equations (reduction::UnitSystem: its interior AUTOSPC),
//    then in order their patterns, every vector's g-set loads split by rows (the interior rows of a unit are its
//    patterns, the rows of the interface points the interface's loads, so that each load acts once; the enforced
//    displacements of a DOF held in every subcase are a pattern of every unit holding it and of the interface) and
//    their cache lookups, then the condensation of those not in the cache (reduction::StaticCondensation, with the six
//    inertial patterns −M·Φ_r when some subcase has inertia relief, and the work of the patterns G = F_oᵀ·K_oo⁻¹·F_o
//    for the external work of the checks), then in order their stores and the decision on their interior factors
//    (reduction::FactorRetention: kept for the recovery while the budget allows, released otherwise and named);
// 6. the interface's own load basis, on the rows of the interface points and the elements touching them.
// Each condensation is the same whatever the wave, the threads or the units beside it: the results have the same bits
// with any budget that fits and any number of threads.
// Per constraint set (solveGroup): the full path's Gm and DOF sets of the subcases (their errors reported as the full
// path reports them: a cycle of relations, a DOF dependent twice, an SPC on a dependent DOF), the relations eliminated
// at the interface (PlannedUnits::interfaceRowFilter), the interface problem with its AUTOSPC (its factorisation on one
// thread when the threaded one does not fit the budget), inertia relief from the units' rigid-body mass properties and
// inertial patterns (no unit recovered), the boundary basis, the recovered units one after the other (a released or
// never made interior factor computed again, to the bits of the condensation, and released again once the unit's
// basis solutions are made), and the cases by batches whose size the budget left decides (lowered from the largest
// batch when needed, named in the set's record): the displacements, applied loads, SPC forces, MPC forces and check
// values of each.
//
// The memory mode (Inputs::memoryMode, reduction::CondensedMemoryPlan) decides, in step 4, what the analysis keeps:
// every recovery made before the cases ("all at once", as above), or the units recovered in sequence: per batch of
// cases, the units in waves of consecutive units bounded by the budget left (one thread each, or every thread for a
// large one), each unit's recovery made in its wave and released once its part of the batch is in the case vectors;
// and, when the strategy says so, a unit's equations released after its condensation (its factor with them; the system
// without equations of a cache hit kept) and built again from the snapshot at each of its recoveries, its K* checked to
// the bit. The case vectors of a batch are filled by the interface, then by each unit on its own DOFs in the order of
// the units (the same order of every sum of a case as "all at once"), never filled whole per case. Every mode, wave and
// batch gives the same bits. In sequence, a set whose caller takes the recovered units' results unit by unit
// (GroupInputs::visitUnit) makes each recovered unit's recovery once for all its cases, before the batches, rather than
// in every batch.
//
// The outputs of a unit that is not recovered (the requested outputs of such a unit are absent and listed in a
// diagnostic): its interior points and its elements are left out of every output plan (outputPlanOf) and listed in one
// diagnostic per unit ("output-not-recovered"); at its boundary points the displacements, the applied loads and the SPC
// forces of the interface's held DOFs are those of the full path, the unit's force on them is the grid point force row
// K*·u_a − f*·α (output::StiffnessRecovery::CondensedUnits), and the SPC forces of the DOFs it holds there with a
// stiffness, and the MPC forces its relations make there, are left out and listed. The checks need no recovery: the
// OLOAD resultant from the loads, the SPC force resultant from the interface's reactions and each such unit's
// equilibrium, EPSILON from the residual of the interface and of the interiors of the units recovered (r_o = f_o −
// K_o·u on their equations), the external work from the interface and G.
//
// The condensation cache, when Inputs::cache gives one: after its equations and patterns, each unit's key
// (reduction::makeUnitKey: its stamps, its design stamp, the MATRIX_PARAMETERS, the version and the kernel, the
// checksums of its equations and patterns) is looked up; a hit takes K*, f*, the inertial patterns, the rigid-body mass
// and the pattern work from the cache and factorises nothing (a unit recovered then factorises its interior for the
// recovery, its K* checked against the cached one to the bit); a miss condenses the unit and stores the result. The
// entries the run uses stay pinned (never evicted) until it ends. The results are the same bits either way: an
// incremental run gives the same bits as a complete run.
//
// Thread-safety: not for concurrent use (one run). Ownership: owns the units' equations, condensations and patterns
// (budgeted); borrows the snapshot, the analysis model, the plan, the load builder and the cache, which must outlive
// it.
class CondensedStatics
{
public:
    // The small units of a wave of the condensation: at most this many per thread (reduction::SchedulerOptions).
    static constexpr std::size_t WAVE_UNITS_PER_THREAD = 4;

    struct Inputs
    {
        const input::ModelSnapshot* snapshot = nullptr;
        const AnalysisModel* model = nullptr;
        const reduction::CondensationPlan* plan = nullptr;
        const assembly::LoadVectorBuilder* loads = nullptr;
        // Every subcase of the run with its SPC set (SUBCOMs included), the indices of those solved.
        std::span<const StaticSubcase> subcases;
        std::span<const assembly::SpcSet> spcSets;
        std::span<const std::size_t> solved;
        // The output plans of the subcases (the units whose outputs are asked for are recovered).
        std::span<const output::OutputPlan> plans;
        // The condensed units to recover (by ID); nothing: those whose points or elements an output asks for.
        std::optional<std::vector<std::int64_t>> recoveredUnits;
        // Elements whose units are recovered in any case (sorted IDs): those of the section and cut loads of the run,
        // whose forces need the displacements of the elements (StaticOptions::sectionLoads).
        std::vector<std::int32_t> elementsToRecover;
        assembly::LoadBasisTolerance tolerance = assembly::LoadBasisTolerance::Roundoff;
        // Whether some constraint set has inertia relief: the units then condense their six inertial patterns.
        bool hasInertiaRelief = false;
        // The condensation cache (null: every unit condensed, nothing stored), the name of the model in the entries it
        // stores, and the design stamps of the units a design point drives (the others' stamp is empty).
        reduction::CondensationCache* cache = nullptr;
        std::string cacheModelTag;
        std::span<const UnitDesignStamp> designStamps;
        const linalg::DenseKernels* kernels = nullptr;
        base::ThreadPool* pool = nullptr;
        const base::CancellationToken* cancellation = nullptr;
        // The memory the results take until the end of the run (the result tables kept in memory, the vectors of the
        // subcases a SUBCOM combines): part of the memory plan, reserved before the case batches.
        std::uint64_t resultBytes = 0;
        // The work above which a unit is condensed alone with every thread (reduction::SchedulerOptions); the tests
        // lower it so that the shared path runs on small units.
        double largeUnitMinimumFlops = reduction::LARGE_UNIT_MINIMUM_FLOPS;
        // How far the run goes: with MemoryPlan the constructor stops after step 4 (the plan recorded, never
        // enforced); with Condensation after step 5, the memory plan counting the condensation phase only (no
        // interface, recovery, case or result follows) and no unit recovered.
        CondensedStage stage = CondensedStage::Complete;
        // Whether each constraint set is kept with its interface factor and the units (takeKeptSets(): the
        // sensitivities of a design point solve on them), and whether the units a design point drives (a non-empty
        // design stamp) are recovered whatever the outputs ask (their pseudo-loads need their displacements).
        bool keepsSystems = false;
        bool recoversDrivenUnits = false;
        // Condensed units recovered beside those the outputs (or recoveredUnits) ask for, by ID.
        std::vector<std::int64_t> alsoRecoveredUnits;
        // The external units of the run (null: none): each a unit of the interface problem, its K* on its
        // contour DOFs, its zone's interior DOFs held, the loads of its zone given to its plugin per vector of the
        // interface's load basis. Their zones must lie in units kept full (a condensed unit holding a replaced element
        // is refused, naming both); inertia relief with them is refused (their rigid-body mass is not in the
        // interface's). Not owned.
        ExternalUnitPath* externalUnits = nullptr;
        // Craig-Bampton (with the Condensation stage): the units keep their mass and add their reduced
        // boundary mass M_aa = Ψ̃ᵀ·M·Ψ̃ to their condensation (CachedReduction::boundaryMass, under a Craig-Bampton key
        // with no mode); then reduceModes() gives each its modal part and solveEigenGroup() solves the eigenproblem of
        // a constraint set. The units to recover are decided as for statics (recoveredUnits, or the output plans).
        bool isCraigBampton = false;
        // The load basis handed over (StaticOptions::keepsLoadBasis; with keepsSystems): the interior factors of the
        // units that some case loads (a load pattern of their own) are kept as those of recovered units (while the
        // budget allows: FactorRetention), since the basis vectors are condensed with them after the condensation.
        // Every unit is then a unit its blocks may read, loadBasisColumnCount columns at once (the largest constraint
        // set's basis), and the memory mode applies: all at once (every interior factor kept once made, for the
        // blocks), or in sequence (no factor kept beyond its use: the basis vectors' loads condensed, each block's
        // recovery), without the factors, with the units' equations released (kept sets that build them again:
        // KeptCondensedUnit::rebuild) and by regions, as for statics.
        bool handsOverLoadBasis = false;
        std::int64_t loadBasisColumnCount = 0;
        // The structure cache of the analyses (null: none), which keeps the units' boundary data between them; whether
        // the caller reads no equation of a unit beside what CondensedStatics reads (the static solution: a unit taken
        // from the cache and not recovered is then built without them, when the analysis has no Craig-Bampton modes, no
        // inertia relief and keeps no system); whether every hit is built with its equations too, to check its key (the
        // equations' checksum, the boundary data kept: InternalError when they differ; the tests run with it); and
        // whether the stores of the cache are finished by the caller (finishCacheStores, before record()) rather than
        // at the end of the condensation.
        StructureCache* structures = nullptr;
        bool takesHitsWithoutEquations = false;
        bool verifiesCacheHits = false;
        bool defersCacheStores = false;
        // The memory mode (reduction::MemoryMode, CondensedMemoryPlan): what the analysis keeps. The recovery in
        // sequence and the release of units' equations apply to a complete static run that keeps no system (no
        // Craig-Bampton, no kept sets for the sensitivities) and to the load basis handed over (handsOverLoadBasis);
        // another run keeps everything, as Speed.
        reduction::MemoryMode memoryMode = reduction::MemoryMode::Speed;
        // The interior factors kept between the analyses of the process (null: none taken). A unit to condense (not in
        // the cache, or no cache) or to factorise again for its recovery takes the factor of its equations when one is
        // kept (KeptInteriorFactors::take), its bytes reserved in the budget; a static run that keeps no system offers
        // the factors it holds at its end (offerInteriorFactors). Not for Craig-Bampton units (their mass is in their
        // reduction) nor a strategy without the factors.
        reduction::KeptInteriorFactors* keptFactors = nullptr;
        // Inertia relief on six FreeBodySupport DOFs of a free body solved on six translations of the farthest boundary
        // points (assembly::chooseConditionedSupports), the displacements brought back to the gauge of the
        // FreeBodySupport DOFs; false (a run whose interface factor serves the sensitivities): held by the
        // FreeBodySupport DOFs.
        bool conditionsReliefSupports = false;
    };

    // Throws SolverError for the problems of the model (the relations, the loads), std::invalid_argument for a plan
    // that does not fit the snapshot, SingularityError for a unit whose interior is a mechanism (not held by its
    // boundary), MemoryBudgetError, OperationCancelledError.
    CondensedStatics(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    CondensedStatics(const CondensedStatics&) = delete;
    CondensedStatics& operator=(const CondensedStatics&) = delete;
    CondensedStatics(CondensedStatics&&) = delete;
    CondensedStatics& operator=(CondensedStatics&&) = delete;
    ~CondensedStatics();

    // The output plan of a subcase on the condensed path: the points and elements of the units not recovered left
    // out (the strain energy too when one is not recovered: its percentages need every element).
    [[nodiscard]] output::OutputPlan outputPlanOf(const output::OutputPlan& plan) const;

    // ---- The recovery unit by unit (GroupInputs::visitUnit) ----

    // Whether the memory plan recovers the units in sequence and some unit is recovered: a constraint set may then
    // take its recovered units' results unit by unit (GroupInputs::visitUnit), each unit's recovery made once for
    // every case of the set rather than once per batch of cases.
    [[nodiscard]] bool recoversInSequence() const noexcept;
    // The bytes a set of `caseCount` cases keeps while its units are recovered unit by unit: the reactions of the DOFs
    // the recovered units hold and their residual work, per case (the SPC forces and EPSILON of each case add them in
    // the order of the units, after the units are recovered).
    [[nodiscard]] std::uint64_t unitByUnitBytes(std::size_t caseCount) const noexcept;
    // `plan` (an output plan of outputPlanOf) without the interior points and the elements of the recovered units,
    // whose rows each recovered unit gives (unitOutputPlanOf): what the cases' own visit recovers in a set taken unit
    // by unit.
    [[nodiscard]] output::OutputPlan interfaceOutputPlanOf(const output::OutputPlan& plan) const;
    // The displacements and element tables of `plan` that the recovered unit at position `unit` (among the condensed
    // units) gives: its interior points and its elements; nothing else. Throws std::invalid_argument for a position
    // outside the units.
    [[nodiscard]] output::OutputPlan unitOutputPlanOf(const output::OutputPlan& plan, std::size_t unit) const;

    // The answers of the cache stores of the condensation (made off the critical path by the session's writer) in
    // the units' records, a store refused reported (Information "cache-not-stored"). Called at the end of the
    // condensation, or by the caller before record() with Inputs::defersCacheStores; the stores not begun by then made
    // on `pool` (CondensationCache::Session::finishStores). Rethrows what a store threw.
    void finishCacheStores(base::DiagnosticSink& sink, base::ThreadPool* pool);

    // One constraint set: its solved subcases (indices into Inputs::subcases), its SPC set, its FreeBodySupport DOFs,
    // its inertia relief, its selection (MPC set, contact) and the PARAMs of its first subcase.
    struct GroupInputs
    {
        std::int32_t constraintSet = 0;
        std::span<const std::size_t> cases;
        const assembly::SpcSet* spc = nullptr;
        std::vector<base::DofIndex> supports;
        assembly::InertiaReliefMode relief = assembly::InertiaReliefMode::None;
        ConstraintSelection selection;
        linalg::PivotPolicy pivotPolicy;
        assembly::SingularityOptions singularity;
        base::Vec3 referencePoint{};
        // False (the load basis handed over, with Inputs::keepsSystems): the interface problem is made, factorised and
        // kept (takeKeptSets), and no case is solved: neither visited nor recorded beyond the set's interface.
        bool solvesCases = true;
        // Called once per batch of cases before its cases are visited, with the g-set displacements of every case of
        // the batch (one column per case, the first the case `firstCaseInGroup`), those each CaseResult then gives: the
        // element tables of a block of cases recovered at once. Empty: not called, and a batch holds the displacements
        // of one case at a time.
        std::function<void(std::size_t firstCaseInGroup, linalg::ConstDenseMatrixView displacements)> visitBatch;
        // With the units recovered in sequence (recoversInSequence()), called for each recovered unit, in the order of
        // the units, once per batch of cases, with the unit's displacements: the recovery unit by unit, each unit's
        // recovery (its equations built again, its interior factorised again, its basis solutions) made once for every
        // case of the set, its results handed over unit by unit, before the cases are visited. The cases' g-set
        // displacements then hold the interface's points only (the interiors of the recovered units are this call's),
        // their SPC forces, applied loads and checks every unit's: the caller recovers the interiors' rows here
        // (unitOutputPlanOf) and the others in the cases' visit (interfaceOutputPlanOf). Every value has the bits of
        // the recovery batch by batch: each sum of a case keeps its order (the interface, then the units in their
        // order). Empty: the units recovered in every batch.
        struct UnitCases
        {
            std::size_t unit = 0; // its position among the condensed units
            std::int64_t unitId = 0;
            std::size_t firstCaseInGroup = 0;
            // The g-set displacements of the batch's cases (one column per case), valid on the DOFs of the unit's
            // points only (its interior, its boundary points as the interface solved them), with the gauge of the
            // FreeBodySupport DOFs of inertia relief on conditioned supports.
            linalg::ConstDenseMatrixView displacements;
        };
        std::function<void(const UnitCases&)> visitUnit;
        // The phases driven by the caller. exportsInterface: the set's interface problem and the right-hand sides of
        // its boundary basis handed over (takeInterfaceExports; solved too with solvesExport, the solutions kept), no
        // unit recovered, no case visited. boundaryMode GivenEquations: the interface's solve replaced by `given`
        // (equations × the basis's columns, or × the cases with isGivenPerCase); GivenLocal: by the boundary
        // displacements of the condensed unit givenUnitId (its K* rows × the cases), the other local DOFs at zero,
        // taken as they are (the recovery of one unit: only that unit's results mean something). Either refuses inertia
        // relief (its relieved loads come from the solve of its support motions) with a SolverError; the interface is
        // then not factorised (but for solvesExport).
        bool exportsInterface = false;
        bool solvesExport = false;
        reduction::BoundarySolveMode boundaryMode = reduction::BoundarySolveMode::Solve;
        bool isGivenPerCase = false;
        linalg::ConstDenseMatrixView given;
        std::int64_t givenUnitId = 0;
    };

    // The force of a condensed unit not recovered on its boundary DOFs in one case, K*·u_a − f*·α (exact: the forces
    // of its elements at its boundary), on the g set: the contour forces of a zone that is the unit need no recovery.
    // The DOFs of a point generated on a fastener side are left out.
    struct UnitBoundaryForces
    {
        std::int64_t unitId = 0;
        std::vector<base::DofIndex> dofs; // bounded by the unit's boundary (its K* front)
        std::vector<double> forces;
        bool hasInteriorReactions = false; // a constraint inside the unit takes a reaction (hasInteriorReactions)
    };

    // The g-set vectors of one case and its check values (StaticSolution recovers its tables from them).
    struct CaseResult
    {
        std::span<const double> displacements;
        std::span<const double> applied;
        std::span<const double> spcForces;
        std::span<const double> mpcForces; // empty without an m set
        // The elimination `mpcForces` were made with (null without an m set): the split multiPointConstrForce rows read
        // it.
        const assembly::RigidElimination* mpcRelations = nullptr;
        std::span<const std::uint8_t> isConstrained;
        double epsilon = 0.0;
        double externalWork = 0.0;
        std::vector<double> supportAccelerations; // with inertia relief
        // The grid point forces of the units not recovered (null when every unit is recovered).
        const output::StiffnessRecovery::CondensedUnits* gridPointForceUnits = nullptr;
        // The resultant of the reactions of the units not recovered (forces and moments about the basic origin, by
        // their equilibrium): the SPC force resultant of the case adds it.
        std::array<double, 6> unrecoveredReactions{};
        // The boundary forces of each unit not recovered, in the order of the units (empty when every unit is).
        std::span<const UnitBoundaryForces> unitBoundaryForces;
    };
    using CaseVisitor = std::function<void(std::size_t caseInGroup, const CaseResult&)>;

    // Solves a constraint set and calls visit for each of its cases, in order. Returns the GRID POINT SINGULARITY
    // TABLE of the set (empty when nothing was constrained). Throws SolverError (the problems of the constraints,
    // listed in the sink), SingularityError (a mechanism of the interface problem), MemoryBudgetError,
    // OperationCancelledError.
    std::string solveGroup(const GroupInputs& group, const CaseVisitor& visit, base::MemoryBudget& budget,
                           base::DiagnosticSink& sink);

    // ---- Eigenvalue analyses on the condensed path (Craig-Bampton; Inputs::isCraigBampton) ----

    // The vibration modes of the statically condensed model that give the cut-off of a buckling analysis (which asks
    // for no frequency): its BUCKLING_REFERENCE_MODE_COUNT lowest roots (the frequency scale of the assembly).
    static constexpr std::size_t BUCKLING_REFERENCE_MODE_COUNT = 10;

    // The modal part of every unit for a request (reduction::ModeRequest: N modes, or the modes below a cut-off): from
    // the condensation cache (a reduction entry under the unit's key with the request) or computed
    // (reduction::CraigBampton: the fixed-interface modes by Lanczos on the unit's interior, then K_qq, M_aq, M_qq; its
    // interior factorised again first when released, to the bits of its condensation) and stored, with the modes as
    // recovery data. The modes themselves are kept in memory for the units that need them (the recovered ones; every
    // unit with needsEveryUnitsModes, which a buckling analysis sets: its differential stiffness is projected on
    // them): taken from the cache, or computed again to the same bits when the cache does not have them (evicted under
    // cache_recovery_max_size, or never stored), the unit's record naming it ("recomputed", with the cause). Units
    // one after the other, each on every thread. `record` (its request) goes to the run record. Throws std::logic_error
    // without Inputs::isCraigBampton, and as the condensation (SolverError, MemoryBudgetError, CacheError for a cached
    // modal part whose modes, computed again, have other eigenvalues: a stale entry).
    void reduceModes(const reduction::ModeRequest& request, const CraigBamptonRecord& record, bool needsEveryUnitsModes,
                     base::MemoryBudget& budget, base::DiagnosticSink& sink);

    // One eigenvalue solve on the interface problem of a constraint set.
    struct EigenGroupInputs
    {
        GroupInputs group; // its constraints, selection and singularity options (its cases unused)
        linalg::EigenRequest request;
        linalg::LanczosOptions lanczos;
        bool isBuckling = false;
        // False: the statically condensed model (no modal coordinate: the boundary mass M_aa alone), whose roots
        // bound the true ones from above (the cut-off of an ND request).
        bool includesModes = true;
        // A buckling analysis: the g-set displacements of the preload (every unit recovered) and the temperatures of
        // a thermal preload (null: none).
        std::span<const double> preload;
        const elements::ElementTemperatures* temperatures = nullptr;
    };
    // Its interface problem (kept for the recovery of the modes) and the eigenpairs of the reduced problem.
    struct EigenGroup
    {
        std::unique_ptr<reduction::InterfaceProblem> interface;
        linalg::EigenSolution solution; // the vectors on the interface's equations
        std::string singularityTable;   // the table of the singular components of the set (empty: none)
        std::int64_t modalCoordinateCount = 0;
        std::int64_t equationCount = 0;
        // Per g-set DOF: whether it is in the full path's analysis set (f) of the constraint set.
        base::BudgetedBuffer<std::uint8_t> isAnalysisSet;
    };
    // The pencil of the constraint set on the interface: K (the units' K* and K_qq, the interface elements, the
    // relations and constraints of the full path's sets, AUTOSPC on the summed boundary blocks, as solveGroup), and
    // the mass (vibration: the interface elements' and each unit's M_CB) or the differential stiffness (buckling: the
    // interface elements' and each unit's T_CBᵀ·K_d·T_CB from the preload, never cached), solved by the block Lanczos
    // solver (linalg::SparsePencil or BucklingPencil on the interface's symbolic factor). Throws as solveGroup and the
    // Lanczos solver.
    [[nodiscard]] EigenGroup solveEigenGroup(const EigenGroupInputs& inputs, base::MemoryBudget& budget,
                                             base::DiagnosticSink& sink);
    // The mode request of the units for the eigenvalue analysis of a constraint set: the plan's count per unit; with
    // none (0), the cut-off CB_CUTOFF_FACTOR² times the highest eigenvalue requested: the upper bound of the vibration
    // request `reference` when it has no count, otherwise the highest root of `reference` on the statically condensed
    // model (solveEigenGroup without modes: an upper bound of the true one). `record` is filled with the request and
    // where it comes from. Throws as solveEigenGroup. `cutoffFactor`: reduction::CB_CUTOFF_FACTOR but in its
    // calibration.
    [[nodiscard]] reduction::ModeRequest chooseModeRequest(const GroupInputs& group,
                                                           const linalg::EigenRequest& reference,
                                                           const linalg::LanczosOptions& lanczos,
                                                           CraigBamptonRecord& record, base::MemoryBudget& budget,
                                                           base::DiagnosticSink& sink,
                                                           double cutoffFactor = reduction::CB_CUTOFF_FACTOR);
    // The eigenvectors of a solve on the g set (g × modes): the interface's points (their relations expanded), and
    // the interiors of the recovered units, u_o = Ψ·u_a + Φ·q (reduction::CraigBampton::recover; a released interior
    // factor computed again to the same bits); `isKnown` flags the g-set DOFs set (the interiors of the units not
    // recovered are not), `isAnalysisSet` those of them in the full path's analysis set (the components a normalisation
    // compares: scaleEigenvectorsOnSet). Throws MemoryBudgetError, OperationCancelledError.
    struct ModeVectors
    {
        linalg::DenseMatrix vectors;
        base::BudgetedBuffer<std::uint8_t> isKnown;
        base::BudgetedBuffer<std::uint8_t> isAnalysisSet;
    };
    [[nodiscard]] ModeVectors recoverModes(const EigenGroup& group, base::MemoryBudget& budget);

    // ---- Loads a solution makes itself: the static aeroelastic trim on the condensed path (SOL 144: static
    // condensation with the spline nodes kept on the boundaries; inertial loads and inertia relief) ----

    // The constraints of a set kept for its outputs (the full path's Gm, its sets), owned by a LoadGroup.
    struct GroupConstraints;
    // The interface problem of a constraint set factorised with its FreeBodySupport DOFs held (the r set: u_r = 0, no
    // inertia relief: GroupInputs::relief is ignored), for right-hand sides the solution makes: loads at the interface
    // points and rigid-body accelerations of the structure.
    struct LoadGroup
    {
        std::unique_ptr<reduction::InterfaceProblem> interface;
        std::shared_ptr<const GroupConstraints> constraints;
        const assembly::RigidElimination* gm = nullptr; // the full path's Gm of the set (in `constraints`)
        std::string singularityTable;                   // its table of the singular components (empty: none)
        // Per g-set DOF: held by the interface or by a unit (the s and r sets of the full path, the AUTOSPC).
        base::BudgetedBuffer<std::uint8_t> isConstrained;
        // Φ_rᵀ·M·Φ_r about the basic origin of the whole structure: the units' and the interface elements'.
        base::MatN<6> rigidBodyMass{};
    };
    // Throws as solveGroup. Needs Inputs::hasInertiaRelief for a set whose solution applies accelerations (the units'
    // six condensed inertial patterns).
    [[nodiscard]] LoadGroup prepareLoadGroup(const GroupInputs& group, base::MemoryBudget& budget,
                                             base::DiagnosticSink& sink);
    // u (g × k) of loads `loadsG` (g × k) at the interface points and of the inertial loads −M·Φ_r·a of the rigid-body
    // accelerations `accelerations` (6 × k about the basic origin, translations then rotations; no column: none), by
    // the units' condensed inertial patterns and the interface elements' mass: the interface points' displacements
    // (u_m = Gm·u_n, held DOFs zero) and, with `recoversUnits`, the interiors of the recovered units
    // (u_o = K_oo⁻¹·(f_o − K_oa·u_a), a released interior factor computed again to the same bits); the interiors of the
    // others zero. Throws std::invalid_argument for a load on an interior point of a condensed unit (the plan must keep
    // the loaded points on the boundaries) or sizes that disagree, std::logic_error without the inertial patterns.
    void solveLoads(const LoadGroup& group, linalg::ConstDenseMatrixView loadsG,
                    linalg::ConstDenseMatrixView accelerations, bool recoversUnits,
                    linalg::DenseMatrixView displacementsG, base::MemoryBudget& budget);
    // loadsG += −M·Φ_r·a on every point (the units' local inertial loads, the interface elements'): the inertial loads
    // of one rigid-body acceleration `acceleration` (6 values, as solveLoads).
    void addInertialLoads(const LoadGroup& group, std::span<const double> acceleration, std::span<double> loadsG) const;
    // forcesG += K·u at the DOFs the outputs read: the elements of the interface and of the recovered units, and the
    // force of each unit not recovered on its boundary points, K*·u_a − f*_inertial·a (exact: the forces of its
    // elements at its boundary, no recovery needed), whose grid point force rows "UNIT" are added to `unitRows` (sorted
    // by point).
    void addStructureForces(const LoadGroup& group, std::span<const double> displacementsG,
                            std::span<const double> acceleration, std::span<double> forcesG,
                            std::vector<output::StiffnessRecovery::UnitRow>& unitRows,
                            base::MemoryBudget& budget) const;

    // Whether some condensed unit is not recovered.
    [[nodiscard]] bool hasUnrecoveredUnits() const noexcept;
    // The kernels the later solves of solveLoads take, serial and without cancellation: a kept trim structure solves
    // after its run, whose pool and cancellation token may be gone (solutions::TrimLinearisation); the same
    // bits for a given kernel kind. `kernels` must outlive this object's later solves.
    void rebind(const linalg::DenseKernels& kernels) noexcept;
    // The grid point forces of the units not recovered with given rows (a SUBCOM combines those of its subcases).
    [[nodiscard]] output::StiffnessRecovery::CondensedUnits gridPointForceUnits(
        std::span<const output::StiffnessRecovery::UnitRow> rows) const noexcept;

    [[nodiscard]] const CondensedRunRecord& record() const noexcept { return m_record; }
    // At the end of a static run, the interior factors its units still hold given to Inputs::keptFactors for the later
    // analyses of the process (each with a copy of its equations), in the order of the units; `analysisBudget` is the
    // budget this run reserved them in (shared: the kept factors keep it alive). Nothing without keptFactors, with
    // Inputs::keepsSystems (the kept sets hold the factors) or Craig-Bampton. The run must recover nothing afterwards
    // (its factors are gone).
    void offerInteriorFactors(const std::shared_ptr<base::MemoryBudget>& analysisBudget);
    // The condensed units' reduced units and the coefficients of the load cases on their patterns, by plan order (the
    // coefficients copied into `budget`). Throws std::logic_error before the units are condensed (the MemoryPlan
    // stage), MemoryBudgetError.
    [[nodiscard]] std::vector<ReducedUnitResult> reducedUnits(base::MemoryBudget& budget) const;
    // The labels of the solved subcases, in the order of the load cases (the columns of the coefficients).
    [[nodiscard]] std::vector<base::EntityId> caseLabels() const;
    // The largest MAXRATIO ratio of the factorisations (units' interiors and interface problems).
    [[nodiscard]] double largestPivotRatio() const noexcept { return m_largestPivotRatio; }
    // The right-hand sides solved: the patterns of every unit and the interface's vectors.
    [[nodiscard]] std::int64_t solvedVectorCount() const noexcept { return m_solvedVectorCount; }
    // Whether a condensed unit (by ID) is recovered; false for a unit the plan keeps full or does not have.
    [[nodiscard]] bool isRecovered(std::int64_t unitId) const noexcept;
    // Whether a condensed unit holds a DOF in every subcase with a stiffness inside it (its interior reactions then
    // need its recovery: a zone that is the unit takes its contour forces without recovery only when it has none).
    [[nodiscard]] bool hasInteriorReactions(std::int64_t unitId) const noexcept;
    // With Inputs::keepsSystems: the constraint sets solved so far, each with its interface factor and the units, in
    // solve order; the condensed run keeps none of them afterwards.
    [[nodiscard]] std::vector<std::shared_ptr<const CondensedFactorisedSet>> takeKeptSets();
    // With Inputs::keepsSystems: the constraint set solved last (null before the first).
    [[nodiscard]] std::shared_ptr<const CondensedFactorisedSet> lastKeptSet() const
    {
        return m_keptSets.empty() ? nullptr : m_keptSets.back();
    }
    // With GroupInputs::exportsInterface: the interface problems handed over so far, in solve order; the run
    // keeps none of them afterwards.
    [[nodiscard]] std::vector<InterfaceExport> takeInterfaceExports() { return std::move(m_interfaceExports); }
    // The recovery of one unit: `plan` restricted to what the condensed unit `unitId` gives alone: the
    // displacements of its interior and boundary points and the tables of its elements; nothing else. Throws
    // std::invalid_argument for a unit the plan does not condense.
    [[nodiscard]] output::OutputPlan unitOnlyOutputPlanOf(const output::OutputPlan& plan, std::int64_t unitId) const;

private:
    struct Units;  // the condensed units and their patterns
    struct Loads;  // the load cases, the interface's load basis
    struct Memory; // the memory plan, the factors kept

    void condenseUnits(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // Step 4: the memory plan of the run, refused before any unit is condensed when the budget cannot hold it.
    void planMemory(const Inputs& inputs, base::MemoryBudget& budget);
    // Step 5: the units condensed in the scheduler's waves.
    void condenseInWaves(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // The units to recover (before the condensation: the factor retention needs them), and the diagnostic of the
    // outputs of the others (after it: the points whose forces need a recovery come from the units' equations).
    void decideRecoveredUnits(const Inputs& inputs);
    void reportUnrecoveredOutputs(const Inputs& inputs, base::DiagnosticSink& sink);
    // Step 7, some unit not recovered: the points and elements the outputs may take (outputPlanOf), listed once.
    void listKnownOutputs(base::MemoryBudget& budget);
    // The interior points and elements of each recovered unit, and the points and elements of none
    // (interfaceOutputPlanOf), listed once at the first call.
    void listUnitOutputs() const;
    // The constraint set kept with its interface problem, its Gm, its held DOFs and its external units, the
    // units shared by the sets (made at the first).
    void keepSet(const GroupInputs& group, std::unique_ptr<const reduction::InterfaceProblem> interface,
                 std::unique_ptr<const assembly::RigidElimination> gm, base::BudgetedBuffer<std::uint8_t> isConstrained,
                 std::vector<std::shared_ptr<const reduction::ExternalUnitAssembly>> placements,
                 base::MemoryBudget& budget, std::shared_ptr<const CondensedRelief> relief = nullptr);

    Inputs m_inputs;
    std::optional<reduction::CondensationCache::Session> m_cacheSession;
    // The units whose new condensations were given to the session's writer, in order (finishCacheStores).
    std::vector<std::size_t> m_storedUnits;
    std::unique_ptr<Units> m_units;
    std::unique_ptr<Loads> m_loads;
    std::unique_ptr<Memory> m_memory;
    CondensedRunRecord m_record;
    double m_largestPivotRatio = 0.0;
    std::int64_t m_solvedVectorCount = 0;
    std::size_t m_solvedCaseCount = 0; // the cases of the constraint sets solved so far
    // With Inputs::keepsSystems: the units as the kept sets share them (made at the first set), the sets.
    std::shared_ptr<const KeptCondensedUnits> m_keptUnits;
    std::vector<std::shared_ptr<const CondensedFactorisedSet>> m_keptSets;
    // listUnitOutputs: per condensed unit, its interior points and its elements (sorted; empty for a unit not
    // recovered); the points and elements of no recovered unit (sorted).
    struct UnitOutputs
    {
        std::vector<base::EntityId> interiorPoints;
        std::vector<base::EntityId> elements;
    };
    mutable std::vector<UnitOutputs> m_unitOutputs;
    mutable std::vector<base::EntityId> m_interfacePoints;
    mutable std::vector<base::EntityId> m_interfaceElements;
    mutable bool m_isUnitOutputsListed = false;
    std::vector<InterfaceExport> m_interfaceExports; // GroupInputs::exportsInterface
};

} // namespace oos::solutions
