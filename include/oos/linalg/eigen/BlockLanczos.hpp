#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/linalg/eigen/ShiftStrategy.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/supernodal/ScratchSpool.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace oos::linalg
{

// The settings of the block Lanczos solver, each with its reason.
struct LanczosOptions
{
    // 7, the default of the EIGRL MAXSET field: it holds the six rigid-body modes of a free structure and a
    // repeated root of multiplicity up to 7 in one block (a block Krylov method finds at most blockSize copies of a
    // root per start), and its 7-column solves use the solve kernels well (FactorSolve's tiles are padded to a
    // multiple of the vector length: 8 columns, and threaded by the subtrees and the rows of the large supernodes).
    static constexpr std::size_t DEFAULT_BLOCK_SIZE = 7;
    // The modes one shift is asked to converge: the basis holds about twice as many (thick restart keeps half), so 40
    // modes give a basis of 105 columns for a block of 7: 0.8 GB at 1 M DOF, and a projected problem of 105 solved in
    // a few milliseconds at every step. More modes are found with more shifts.
    static constexpr std::size_t DEFAULT_MODES_PER_SHIFT = 40;
    // A Ritz pair has converged when ‖Op·y − θ·y‖_M <= tolerance·|θ|: its eigenvalue is then exact to about
    // tolerance²/(relative gap) (Parlett, The Symmetric Eigenvalue Problem, §11.7), well below 10⁻¹², and its vector to
    // about tolerance/(relative gap).
    static constexpr double DEFAULT_CONVERGENCE_TOLERANCE = 1.0e-10;
    // Shifts (factorisations) allowed before the solver stops with what it has found.
    static constexpr std::size_t DEFAULT_MAXIMUM_SHIFTS = 40;
    // Thick restarts allowed in one run; a run that has not converged its target by then hands over to a new shift.
    static constexpr std::size_t DEFAULT_MAXIMUM_RESTARTS = 20;
    static constexpr std::uint64_t DEFAULT_SEED = 0x6E61737472616E5FULL;
    static constexpr std::size_t DEFAULT_INNER_PRODUCT_PIECE_ROWS = 1024;

    std::size_t blockSize = DEFAULT_BLOCK_SIZE;
    // The converged vectors held in memory at most: the later ones go to a scratch
    // spool of `lockedSpool`, read back piece by piece by the projections (the same bits). Unbounded by default.
    std::size_t residentLockedColumns = std::numeric_limits<std::size_t>::max();
    ScratchSpool::Options lockedSpool;
    std::size_t modesPerShift = DEFAULT_MODES_PER_SHIFT;
    double convergenceTolerance = DEFAULT_CONVERGENCE_TOLERANCE;
    std::size_t maximumShifts = DEFAULT_MAXIMUM_SHIFTS;
    std::size_t maximumRestarts = DEFAULT_MAXIMUM_RESTARTS;
    // The seed of the random starting blocks (the same seed, the same bits).
    std::uint64_t seed = DEFAULT_SEED;
    // The rows of a piece of the inner products over the dimension: the pieces, not the threads,
    // fix the order of their sums. 1 024 by default (INNER_PRODUCT_PIECE_ROWS); the tests set fewer to have many
    // pieces on a small pencil.
    std::size_t innerProductPieceRows = DEFAULT_INNER_PRODUCT_PIECE_ROWS;
    // Checked at every block step and passed on by the pencil to its factorisations; the counter is advanced by the
    // modes found (the caller sizes it with the mode count or the Sturm count of the range).
    base::OperationControl control;
};

// One shift of the solution: a factorisation, and a Lanczos run when the Sturm count did not already settle things.
struct ShiftRecord
{
    double shift = 0.0;
    ShiftPurpose purpose = ShiftPurpose::Advance;
    std::size_t sturmCount = 0;
    bool hasLanczosRun = false;
    std::size_t blockSteps = 0;
    std::size_t restarts = 0;
    std::size_t convergedCount = 0;
};

// The eigenpairs of a request.
struct EigenSolution
{
    // Increasing; for a buckling pencil by increasing magnitude (ties: the positive root first), the critical load
    // factor first.
    std::vector<double> eigenvalues;
    // dimension × eigenvalues.size(), M-orthonormal (φᵢᵀ·M·φⱼ = δᵢⱼ); for a buckling pencil K-orthonormal
    // (φᵢᵀ·K·φⱼ = δᵢⱼ, so that φᵢᵀ·B·φᵢ = 1/λᵢ). The sign is the solver's.
    DenseMatrix eigenvectors;
    // ‖Op·y − θ·y‖_M / |θ| of each pair when it converged (below the convergence tolerance).
    std::vector<double> residualBounds;
    std::vector<ShiftRecord> shifts;
    std::size_t factorisationCount = 0;
    // The roots of the request found and confirmed by the Sturm counts (all of them when isComplete).
    std::size_t verifiedCount = 0;
    // False when the shift limit was reached before the request was met: the eigenpairs are those found.
    bool isComplete = false;
};

// The block shift-invert Lanczos solver of K φ = λ M φ (SOL 103): the eigenpairs of a range or the
// lowest of a count, none missed (Sturm checks), the rigid-body modes of free structures included (a negative shift).
//
// Each shift factorises K − σM through the pencil and gives a Sturm point; a Lanczos run at the shift (block size b,
// thick restart in a bounded basis, full reorthogonalisation in the M-inner product: LanczosRun) converges the modes
// nearest it, which are then locked: every later basis is kept M-orthogonal to them, so no mode is found twice and a
// root of high multiplicity is completed by later runs. ShiftStrategy places the shifts and decides when the Sturm
// counts confirm the request (SturmCheck). The method follows Grimes, Lewis & Simon (SIAM J. Matrix Anal. Appl. 15,
// 1994), with a thick restart in place of their selective orthogonalisation.
//
// Memory: the basis and four blocks (LanczosWorkspace) are reserved before the first factorisation, the eigenvectors
// found as they come, in the budget given; the pencil reserves its factor in its own. A basis that does not fit is
// refused with MemoryBudgetError before any work.
//
// Determinism: the random starts are seeded, every product is a DenseKernels product or a sum by pieces of rows fixed
// by the dimension and LanczosOptions::innerProductPieceRows, the pencil's factor and solves have the same bits
// for any number of threads: the solution has the same bits for 1 or 12 threads, for a given kernel family and piece
// size.
//
// Buckling (SpectralTransform::Buckling): the same runs, store and restarts on Op = (K − σB)⁻¹·B in the
// K-inner product (ShiftedPencil). The load factors of either sign are legitimate roots and K − σB counts the roots
// between 0 and σ, so each sign is a side with its own Sturm walk and shift strategy in the coordinate x = |λ|: the
// solve factorises at σ = 0 first (K must be positive definite, else SolverError), runs at σ = 0 where the roots
// nearest 0 of both signs converge first (θ = 1/λ), and extends the verified radius of the side that lags
// (EigenRequest).
//
// Thread-safety: solve() is const and keeps no state; one pencil must not be shared by two solves at once.
class BlockLanczos
{
public:
    // Throws std::invalid_argument when a setting is out of range (block size or modes per shift 0, tolerance outside
    // (0, 10⁻²], no shift or no restart allowed).
    explicit BlockLanczos(const LanczosOptions& options = {});

    [[nodiscard]] const LanczosOptions& options() const noexcept { return m_options; }

    // The eigenpairs of `request`. The kernels run the dense products of the Lanczos basis (their pool, their family).
    // Throws std::invalid_argument for a request that cannot be met (ShiftStrategy); MemoryBudgetError when the basis
    // or the eigenvectors do not fit; SolverError when the pencil is singular at every shift tried (a DOF with neither
    // mass nor stiffness), the Sturm counts contradict each other, or the negative roots are unbounded (for a buckling
    // pencil: when K is not positive definite);
    // OperationCancelledError when cancelled.
    [[nodiscard]] EigenSolution solve(ShiftedPencil& pencil, const EigenRequest& request, const DenseKernels& kernels,
                                      base::MemoryBudget& budget) const;

    // The block size and the basis columns a solve uses for a pencil of `dimension` and a request.
    struct BasisShape
    {
        std::size_t blockSize = 0;
        std::size_t basisColumns = 0;
        std::size_t modesPerShift = 0;
    };
    [[nodiscard]] BasisShape shapeFor(std::size_t dimension, const EigenRequest& request) const noexcept;

    // An upper bound of the bytes a solve reserves before its first factorisation (the basis and its blocks) plus the
    // eigenvectors of `modeCount` modes and the copy of the result; the pencil's factor is not included.
    [[nodiscard]] std::uint64_t estimateMemoryBytes(std::size_t dimension, const EigenRequest& request,
                                                    std::size_t modeCount) const noexcept;

private:
    // The solve of a buckling pencil (see the class comment); BlockLanczosBuckling.cpp.
    [[nodiscard]] EigenSolution solveBuckling(ShiftedPencil& pencil, const EigenRequest& request,
                                              const DenseKernels& kernels, base::MemoryBudget& budget) const;

    LanczosOptions m_options;
};

} // namespace oos::linalg
