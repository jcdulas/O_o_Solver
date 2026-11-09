#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/output/arrays/EigenvalueTable.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/solutions/common/EigenMethod.hpp"
#include "oos/solutions/common/ProgressMonitor.hpp"
#include "oos/solutions/common/RunCatalog.hpp"
#include "oos/solutions/superelements/SuperelementPart.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace oos::solutions
{

// What a run of a deck with superelements solves, and with what.
struct SuperelementOptions
{
    AnalysisKind analysis = AnalysisKind::Statics; // Statics (SESTATIC, SOL 101) or Modes (SEMODES, SOL 103)
    std::vector<SuperelementPart> parts;           // the superelements upstream of the residual structure
    // Every subcase of the Case Control with the superelements its SUPER command selects, in its order.
    std::vector<SuperelementSubcase> subcases;
    base::ThreadPool* pool = nullptr; // null: one thread; not owned
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    const base::CancellationToken* cancellation = nullptr; // not owned, may be null
    ProgressMonitor* monitor = nullptr;                    // not owned, may be null
};

// The results of one superelement (the residual structure, 0, or a part): its tables in the subcases that apply to it
// (SUPER), and in a modal analysis the eigenvalue table of each of those subcases (the roots of the residual
// structure, given with every superelement's results).
struct SuperelementOutput
{
    std::int32_t superelementId = RESIDUAL_STRUCTURE;
    output::ResultArrays arrays;
    std::vector<output::EigenvalueTable> eigenvalues;
};

// What the run did with one upstream superelement.
struct SuperelementRecord
{
    std::int32_t superelementId = 0;
    bool isExternal = false;
    std::int64_t boundaryDofCount = 0;             // the rows of K* (its a set for an external one)
    std::int64_t interiorDofCount = 0;             // its interior equations (none for an external one)
    std::int64_t modalCoordinateCount = 0;         // its q set kept (Craig-Bampton)
    std::vector<double> fixedInterfaceEigenvalues; // λ of the modes kept, increasing
    std::vector<base::EntityId> subcases;          // the subcases that apply to it
};

// One eigenvalue solve of the residual structure.
struct SuperelementSolve
{
    std::vector<base::EntityId> subcases; // the residual's subcases that share it
    EigenMethod method;
    std::vector<double> eigenvalues;
    std::int64_t analysisSetSize = 0;
    std::size_t factorisationCount = 0;
    bool isComplete = true;
};

// The results of a run of a deck with superelements.
struct SuperelementResults
{
    std::vector<SuperelementOutput> outputs; // the residual structure first, then the parts in the order given
    std::vector<SuperelementRecord> superelements;
    std::vector<SuperelementSolve> solves; // a modal analysis
    std::vector<base::Diagnostic> diagnostics;
    std::vector<std::string> singularityTables;
    std::int64_t residualGSetSize = 0;
    std::int64_t largestAnalysisSetSize = 0;
    std::string kernel;
    double reductionSeconds = 0.0;
    double residualSeconds = 0.0;
    double recoverySeconds = 0.0;
};

// The analysis of a deck with superelements written in it, mapped onto the condensation engine: SESTATIC (SOL 101) and
// SEMODES (SOL 103), one level of superelements upstream of the residual structure.
//
// 1. Each part (BEGIN SUPER) is reduced on its own model onto its boundary (PartReduction: one condensation unit;
//    static condensation, Guyan or Craig-Bampton on its q set); an external superelement (external matrix file) brings
//    its matrices as they are. Each is placed on the residual's DOFs (PlacedSuperelement).
// 2. The residual structure is the interface problem: its own elements, constraints and loads with the reduced
//    superelements' matrices and loads added (ConstrainedSystemOptions: their nodes coupled, K added to Kgg; their
//    mass to Mgg), the boundary DOFs a superelement holds held there too (the union of the permanent constraints),
//    solved by the full path's steps on the residual (the m set eliminated, AUTOSPC on its Knn, the f set factorised:
//    FsetSolver for statics; SparsePencil and BlockLanczos with the METHOD of its subcases for modes, normalised by
//    it).
// 3. Each superelement is recovered from the residual's displacements (its boundary values and generalised
//    coordinates) and its tables made on its own model, in the subcases its SUPER selects (an external superelement
//    has no table).
//
// Subcases (SUPER): a modal analysis solves the residual once per group of its subcases (same SPC, MPC and METHOD);
// a part's tables in a subcase that is not the residual's take the residual's one solve (refused when there are
// several). A static analysis needs every subcase to apply to every superelement (SUPER = ALL, or no SUPER: the
// load sequences of SUPER = n in statics are not supported by this version), the loads of each subcase those of every
// superelement's own Bulk Data.
//
// Not supported, refused before any solve (errors): SUBCOM, thermal loads, inertia relief (SUPORT, PARAM INREL) in a
// deck with superelements; a superelement downstream of another one (several levels), and the refusals of
// PartReduction and PlacedSuperelement. Not produced: GPFORCE and the grid point weight generator (PARAM GRDPNT) of a
// part, OLOAD of a modal analysis (an information diagnostic).
//
// Determinism: the parts reduced one after the other, every step deterministic: the same bits for any number of
// threads, for a kernel family.
//
// Thread-safety: run() is const; it uses the pool of the options. Ownership: borrows the residual's snapshot and shares
// the parts' snapshots, which must outlive the call.
class SuperelementSolution
{
public:
    SuperelementSolution(const input::ModelSnapshot& residual, SuperelementOptions options)
        : m_residual(&residual), m_options(std::move(options))
    {
    }

    // Throws UnsupportedCardError, SolverError (the problems of the model and of the superelements, listed),
    // SingularityError, MemoryBudgetError, OperationCancelledError.
    [[nodiscard]] SuperelementResults run(base::MemoryBudget& budget) const;

private:
    const input::ModelSnapshot* m_residual;
    SuperelementOptions m_options;
};

} // namespace oos::solutions
