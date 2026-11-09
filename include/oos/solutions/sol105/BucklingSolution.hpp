#pragma once

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
#include "oos/reduction/cache/CondensationCache.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"
#include "oos/solutions/common/AnalysisPath.hpp"
#include "oos/solutions/common/EigenMethod.hpp"
#include "oos/solutions/common/FullPathEigen.hpp"
#include "oos/solutions/common/ProgressMonitor.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace oos::solutions
{

// What a linear buckling analysis runs and with what.
struct BucklingOptions
{
    // The buckling subcases to solve, in this order; empty: every subcase with a METHOD (the global section of a deck
    // without SUBCASE, as subcase 1). The static subcases of the deck are solved in any case (the preloads, and the
    // static results of the deck).
    std::vector<base::EntityId> subcases;
    // The method of every buckling subcase in place of its METHOD (a request of the caller by count or range).
    std::optional<EigenMethod> method;
    base::ThreadPool* pool = nullptr; // null: one thread; not owned
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    const base::CancellationToken* cancellation = nullptr; // not owned, may be null
    ProgressMonitor* monitor = nullptr;                    // not owned, may be null
    // The memory the result tables of the buckling shapes may take; nothing: half of the budget left at the recovery.
    std::optional<std::uint64_t> outputLimitBytes;
    std::int64_t recoveryRangeElements = output::ElementResultBuilder::RANGE_ELEMENTS;
    // Whether the element tables of every mode are recovered one mode at a time (ModeRecovery::recover), the reference
    // of the recovery of many modes at once (ModeRecovery::elementTables), which the tests compare with.
    bool isRecoveredCaseByCase = false;
    // The condensed path (in SOL 105, the differential stiffness of a condensed unit is reduced with the same basis):
    // a plan of the units made on this snapshot (its mode count per unit, or 0 for the cut-off of
    // reduction::CB_CUTOFF_FACTOR); null, or a plan that condenses nothing, for the full path. The preload
    // is solved on the condensed path with every unit recovered (each unit's differential stiffness needs its
    // displacements). The units to recover for the buckling shapes (nothing: those an output reaches), the condensation
    // cache (not owned), the name of the model in its entries, the design stamps.
    std::shared_ptr<const reduction::CondensationPlan> condensation;
    std::optional<std::vector<std::int64_t>> recoveredUnits;
    reduction::CondensationCache* cache = nullptr;
    std::string cacheModelTag;
    std::vector<UnitDesignStamp> designStamps;
    // reduction::CB_CUTOFF_FACTOR, unless another value is given to calibrate it.
    double cutoffFactor = reduction::CB_CUTOFF_FACTOR;
    // The design layer (SOL 200 BUCK): the options of the static solution of the preload and static subcases
    // (the outputs the responses read, the factorised constraint sets kept for the sensitivities, the g-set
    // displacements kept, the section loads and zones of its responses); nothing: those of a plain SOL 105. Their
    // subcases, pool, kernel, cancellation, monitor, analysis kind and condensation are set by the buckling solution;
    // the subcases they keep the displacements of are added to those of the preloads. And on the condensed path, the
    // units recovered for the buckling shapes beside those the outputs ask for: the driven units (a non-empty design
    // stamp: their derivatives read the shapes of their elements) and the units of alsoRecoveredUnits.
    std::optional<StaticOptions> preloadOptions;
    bool recoversDrivenUnits = false;
    std::vector<std::int64_t> alsoRecoveredUnits;
    // The structure kept between the analyses of the model (StaticOptions::structures): on the full path
    // the pattern, the ordering and the symbolic factor of each buckling subcase's analysis set, and those of the
    // preloads' constraint sets, taken from it when an earlier analysis built them. Null: built for this run alone.
    StructureCache* structures = nullptr;
    // The full path's eigenvalue solver: the block Lanczos solver, multilevel substructuring or the factor out of core.
    EigenSolverOptions eigenSolver;
};

// The seconds spent in each stage of a buckling analysis (summed over the solves).
struct BucklingTimings
{
    double preload = 0.0;     // the static solution of the preload subcases, with their outputs
    double assembly = 0.0;    // the model, K, K_d, m-set elimination, ordering and symbolic analysis
    double eigenvalues = 0.0; // the block Lanczos solve
    double recovery = 0.0;
};

// The buckling solve of one subcase.
struct BucklingSolve
{
    base::EntityId subcase = 1;
    base::EntityId preloadSubcase = 1; // the static subcase whose internal forces make K_d
    EigenMethod method;
    output::EigenvalueTable eigenvalues; // the load factors, by increasing magnitude (the positive first on a tie)
    // The normalised buckling shapes, g × modes (one value per DOF of assembly::DofMap; constrained DOFs zero).
    linalg::DenseMatrix eigenvectors;
    std::vector<double> scales; // the factor from the K-orthonormal vector: its generalised stiffness is the square
    std::vector<linalg::ShiftRecord> shifts;
    std::size_t factorisationCount = 0;
    std::size_t verifiedCount = 0;
    bool isComplete = true;
    std::int64_t analysisSetSize = 0;
    std::int64_t modalCoordinateCount = 0; // the condensed path: the units' modal coordinates
};

// The results of a buckling analysis: the static solution of the preload subcases (its tables, checks and report:
// the static results of a SOL 105 deck), the buckling solves, and the tables of the buckling shapes appended to
// the static result arrays (analysis "buckling", a step per mode).
struct BucklingResults
{
    StaticResults preload;
    std::vector<BucklingSolve> solves;
    std::vector<base::Diagnostic> diagnostics; // those of the buckling part (the preload's are in preload)
    std::vector<std::string> singularityTables;
    std::int64_t largestAnalysisSetSize = 0;
    BucklingTimings timings;
    // The path of the buckling solves, and on the condensed path their report (CondensedRunRecord: the units, their
    // modes and the cut-off) and the units whose local buckling is not checked (every condensed unit, listed first in
    // the report, a warning in the diagnostics).
    AnalysisPath path = AnalysisPath::Full;
    std::optional<CondensedRunRecord> condensed;
    // On multilevel substructuring: the report of its last solve; out of core: the plan and traffic of its last solve.
    std::optional<linalg::SubstructuringReport> substructuring;
    std::optional<OutOfCoreRecord> outOfCore;
    // multilevel substructuring and the factor out of core: their pencil, "elements" or "assembled"; empty on the other
    // paths.
    std::string eigenPencil;
    std::vector<std::int64_t> uncheckedUnits;
};

// The linear buckling solution on the full model (SOL 105): for each buckling subcase (a subcase with METHOD), the
// static solution of its preload subcase gives the displacements, from which every element gives its differential
// stiffness (ElementFamily::computeDifferentialStiffness, assembled on the pattern of K by
// assembly::DifferentialStiffnessAssembler), and the load factors λ of (K + λ·K_d)·φ = 0 are found by the block Lanczos
// solver in the buckling transformation (linalg::BucklingPencil, BlockLanczos).
//
// The preload of a buckling subcase (STATSUB): the subcase that the column staticSubcaseSelection of subcasesTable
// names when O_o_Reader gives it; otherwise the subcase itself when it selects a LOAD or a
// TEMPERATURE(LOAD) (a deck of one subcase with LOAD and METHOD), else the
// static subcase just before it in the Case Control (the usual deck: SUBCASE 1 with LOAD, SUBCASE 2 with METHOD).
// Every static subcase of the deck (without METHOD) is solved and recovered as SOL 101 does.
//
// The requests (eigenRequestOf for a buckling analysis): LanczosEigenMethod V1, V2 and ND in load-factor units, or
// BucklingEigenMethod L1, L2 and NDP, NDN (its inverse power method replaced by the Lanczos solver over the same
// range). The roots have either sign; ND alone gives the ND roots nearest 0. The shapes are normalised by MAX (the
// default; NORM = MASS is replaced by MAX with a warning) or POINT. The constraints of the eigenproblem are those of
// the buckling subcase.
//
// With BucklingOptions::condensation, the condensed path (Craig-Bampton for SOL 105): the preload solved on the
// condensed path with every unit recovered, the units condensed with their mass for the constraints of the buckling
// subcases, their fixed-interface modes (the plan's count, or the cut-off of reduction::CB_CUTOFF_FACTOR times the
// BUCKLING_REFERENCE_MODE_COUNT-th vibration frequency of the statically condensed model), each unit's differential
// stiffness from its preload projected on its Craig-Bampton basis (never cached), the reduced buckling problem on the
// interface, the shapes on the interface and in the recovered units. The local buckling inside a condensed unit is not
// checked: every condensed unit is listed first (BucklingResults::uncheckedUnits, a warning
// "local-buckling-not-checked").
//
// Determinism: as the static and modal solutions, the same bits for any number of threads, for a given kernel family.
//
// Thread-safety: run() is const; it uses the pool of the options. Ownership: borrows the snapshot.
class BucklingSolution
{
public:
    BucklingSolution(const input::ModelSnapshot& snapshot, BucklingOptions options)
        : m_snapshot(&snapshot), m_options(std::move(options))
    {
    }

    // Throws UnsupportedCardError, SolverError (problems of the model, listed; a stiffness that is not positive
    // definite), SingularityError (a mechanism in the preload), MemoryBudgetError, OutputVolumeError,
    // OperationCancelledError, std::invalid_argument (a subcase asked for that the Case Control does not have).
    [[nodiscard]] BucklingResults run(base::MemoryBudget& budget) const;

private:
    const input::ModelSnapshot* m_snapshot;
    BucklingOptions m_options;
};

} // namespace oos::solutions
