#pragma once

#include "oos/assembly/relief/GridPointWeight.hpp"
#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/output/arrays/EigenvalueTable.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/reduction/cache/CondensationCache.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"
#include "oos/solutions/common/AnalysisPath.hpp"
#include "oos/solutions/common/EigenMethod.hpp"
#include "oos/solutions/common/FullPathEigen.hpp"
#include "oos/solutions/common/ProgressMonitor.hpp"
#include "oos/solutions/common/StructureCache.hpp"
#include "oos/solutions/sol101/CondensedStatics.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace oos::solutions
{

// What a real eigenvalue analysis runs and with what.
struct ModesOptions
{
    // The subcases to solve, in this order; empty: every SUBCASE of the Case Control (the global section of a deck
    // without SUBCASE, as subcase 1).
    std::vector<base::EntityId> subcases;
    // The method of every subcase in place of its METHOD (a request of the caller by count or frequency range).
    std::optional<EigenMethod> method;
    base::ThreadPool* pool = nullptr; // null: one thread; not owned
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    const base::CancellationToken* cancellation = nullptr; // not owned, may be null
    ProgressMonitor* monitor = nullptr;                    // not owned, may be null
    // The memory the result tables may take; nothing: half of the memory budget left when the recovery starts.
    std::optional<std::uint64_t> outputLimitBytes;
    // Fields read by design responses, added to the Case Control outputs.
    std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> requiredOutputs{};
    // The elements of one task of the element recovery (RANGE_ELEMENTS but in the tests of the threaded recovery).
    std::int64_t recoveryRangeElements = output::ElementResultBuilder::RANGE_ELEMENTS;
    // Whether the element tables of every mode are recovered one mode at a time (ModeRecovery::recover), the reference
    // of the recovery of many modes at once (ModeRecovery::elementTables), which the tests compare with.
    bool isRecoveredCaseByCase = false;
    // The condensed path (Craig-Bampton for SOL 103): a plan of the units made on this
    // snapshot (its mode count per unit, or 0 for the cut-off of reduction::CB_CUTOFF_FACTOR); null, or a plan that
    // condenses nothing, for the full path. With it: the condensed units to recover (nothing: those whose points or
    // elements an output asks for), the condensation cache (null: nothing kept; not owned), the name of the model in
    // its entries, and the design stamps of the units a design point drives.
    std::shared_ptr<const reduction::CondensationPlan> condensation;
    std::optional<std::vector<std::int64_t>> recoveredUnits;
    reduction::CondensationCache* cache = nullptr;
    std::string cacheModelTag;
    std::vector<UnitDesignStamp> designStamps;
    // reduction::CB_CUTOFF_FACTOR, unless another value is given to calibrate it.
    double cutoffFactor = reduction::CB_CUTOFF_FACTOR;
    // The design layer on the condensed path: whether the units a design point drives (a non-empty design
    // stamp) are recovered whatever the outputs ask (the eigenvalue derivatives read the mode shapes of their
    // elements), and the units recovered beside them, by ID (CondensedStatics::Inputs, as StaticOptions does for
    // statics).
    bool recoversDrivenUnits = false;
    std::vector<std::int64_t> alsoRecoveredUnits;
    // The structure kept between the analyses of the model (StaticOptions::structures): on the full path
    // the pattern, the ordering and the symbolic factor of each analysis set taken from it when an earlier analysis
    // built them from the same graph, set and positions, else built and kept there. Null: built for this run alone.
    StructureCache* structures = nullptr;
    // The full path's eigenvalue solver: the block Lanczos solver, multilevel substructuring or the factor out of core.
    EigenSolverOptions eigenSolver;
};

// The seconds spent in each stage of a real eigenvalue analysis (summed over the solves).
struct ModesTimings
{
    double setup = 0.0;       // support and element checks, DOF map, connectivity, subcases
    double assembly = 0.0;    // stiffness, mass, m-set elimination, singularities, ordering and symbolic analysis
    double eigenvalues = 0.0; // the block Lanczos solve: factorisations, Lanczos runs
    double recovery = 0.0;    // normalisation, tables, modal effective mass
};

// One eigenvalue solve: the subcases that share it (the same constraints and method), its roots and vectors.
struct ModesSolve
{
    std::vector<base::EntityId> subcases;
    EigenMethod method;
    // The eigenvalues λ = ω², increasing (the rigid-body modes of a free structure first, rounded zeros of either
    // sign).
    std::vector<double> eigenvalues;
    // The normalised eigenvectors, g × modes (one value per DOF of assembly::DofMap; the constrained DOFs zero, the m
    // set from Gm).
    linalg::DenseMatrix eigenvectors;
    // The scale of each normalised vector from the M-orthonormal one: its generalised mass is the square.
    std::vector<double> scales;
    // The Lanczos shifts (factorisations, Sturm counts, runs) of the solve, for the run report.
    std::vector<linalg::ShiftRecord> shifts;
    std::size_t factorisationCount = 0;
    std::size_t verifiedCount = 0; // the roots of the request confirmed by the Sturm counts
    bool isComplete = true;        // false: the shift limit was reached before the request was met (a warning)
    std::int64_t analysisSetSize =
        0; // the equations of the pencil (on the condensed path: the interface's, q included)
    std::int64_t modalCoordinateCount = 0; // the condensed path: the units' modal coordinates (their q sets)
};

// The eigensolution of one subcase.
struct ModesSubcaseResult
{
    base::EntityId subcase = 1;
    std::size_t solve = 0; // its solve in ModesResults::solves
    output::EigenvalueTable eigenvalues;
    std::optional<output::ModalEffectiveMassTable> modalEffectiveMass; // with MEFFMASS
};

// The results of a real eigenvalue analysis and its report.
struct ModesResults
{
    output::ResultArrays arrays;
    std::vector<ModesSolve> solves;
    std::vector<ModesSubcaseResult> subcases; // in the order solved
    std::vector<base::Diagnostic> diagnostics;
    std::vector<std::string> singularityTables;
    std::int64_t gSetSize = 0;
    std::int64_t largestAnalysisSetSize = 0;
    std::string kernel;
    ModesTimings timings;
    std::optional<assembly::GridPointWeight> gridPointWeight; // with PARAM GRDPNT
    // The path taken, and on the condensed path its report (the units, their modes and the cut-off: the report states
    // that the modes are approximate, with the cut-off and the number of modes per unit).
    AnalysisPath path = AnalysisPath::Full;
    std::optional<CondensedRunRecord> condensed;
    // On multilevel substructuring: its report (the cut-off, the nodes and modes, the passes, the times).
    std::optional<linalg::SubstructuringReport> substructuring;
    // With the factor out of core: its plan and the disk's traffic.
    std::optional<OutOfCoreRecord> outOfCore;
    // multilevel substructuring and the factor out of core: their pencil, "elements" or "assembled"; empty on the other
    // paths.
    std::string eigenPencil;
};

// The real eigenvalue solution on the full model (SOL 103): for each subcase,
// K·φ = λ·M·φ on the analysis set (the constraints, MPCs and rigid elements eliminated as in the static solution, the
// grid point singularities constrained by AUTOSPC), the roots of its METHOD (EIGRL, or RealEigenMethod with METHOD =
// LAN) by the block shift-invert Lanczos solver with Sturm checks (linalg::BlockLanczos), the rigid-body modes of a
// free structure included.
//
// The steps:
//  1. the support check (EIGRL and RealEigenMethod read), the element check, the subcases (SUBCASE rows; a SUBCOM has
//  no meaning
//     here and is refused), their METHOD (readEigenMethod: an RealEigenMethod method other than LAN is refused naming
//     it) and their output plans; FreeBodySupport is not separated (the rigid-body modes come from the free stiffness:
//     a warning);
//  2. the subcases are grouped by constraint set and method: one solve per group;
//  3. per group: the constrained system (ConstrainedSystem: Gm, sets, Kgg, Knn, AUTOSPC), Mgg (assembly::MassAssembler:
//     lumped or coupled by PARAM COUPMASS, the non-structural mass, PARAM WTMASS) on the pattern of K and reduced to
//     the n set, the analysis-set ordering and symbolic factor (FsetOrdering), K and M converted with the same
//     selection, the pencil (linalg::SparsePencil) and the solve (the LanczosEigenMethod V1, V2 in cycles converted to
//     λ = (2πf)², ND, MAXSET as the block size: eigenRequestOf); the grid point weight generator (PARAM GRDPNT) from
//     the first Mgg;
//  4. the eigenvectors normalised (scaleEigenvectors: MASS, MAX, POINT), the eigenvalue table (λ, radians, cycles,
//     generalised mass and stiffness), the output volume checked against the limit (OutputVolumeError before the
//     recovery), then the tables of each mode for each subcase of the group (ModeRecovery), and the modal effective
//     mass of the subcases that request MEFFMASS.
//
// With ModesOptions::condensation, the condensed path (Craig-Bampton for SOL 103): the units of the plan condensed
// with their mass once for every group
// (CondensedStatics, stage Condensation), then per group the units' fixed-interface modes (the plan's count, or the
// cut-off of reduction::CB_CUTOFF_FACTOR: CondensedStatics::chooseModeRequest), the reduced eigenproblem on the
// interface (CondensedStatics::solveEigenGroup), the shapes on the interface and in the recovered units, normalised on
// the components of the full path's analysis set they know (scaleEigenvectorsOnSet), and their tables. The modes are
// approximate but with every mode of each unit (equal to the full path's then to round-off); the SPC and MPC forces,
// MEFFMASS and GRDPNT are not produced there (an information diagnostic).
//
// Determinism: the assembly, the Lanczos solver (seeded starts, deterministic factor, solves and products), the
// normalisation and the recovery have the same bits for any number of threads, for a given kernel family.
//
// Thread-safety: run() is const; it uses the pool of the options. Ownership: borrows the snapshot, which must outlive
// the call.
class ModesSolution
{
public:
    ModesSolution(const input::ModelSnapshot& snapshot, ModesOptions options)
        : m_snapshot(&snapshot), m_options(std::move(options))
    {
    }

    // Throws UnsupportedCardError (cards not supported), SolverError (problems of the model, listed; a pencil singular
    // at every shift; contradictory Sturm counts), MemoryBudgetError, OutputVolumeError, OperationCancelledError,
    // std::invalid_argument (a subcase asked for that the Case Control does not have).
    [[nodiscard]] ModesResults run(base::MemoryBudget& budget) const;

private:
    const input::ModelSnapshot* m_snapshot;
    ModesOptions m_options;
};

} // namespace oos::solutions
