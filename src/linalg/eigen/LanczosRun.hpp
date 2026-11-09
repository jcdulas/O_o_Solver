#pragma once

#include "EigenvectorStore.hpp"
#include "MassOrthogonalizer.hpp"

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::linalg
{

// The sizes and limits of one Lanczos run.
struct LanczosRunSettings
{
    std::size_t blockSize = 1;
    // The basis capacity in columns, a multiple of blockSize, at least 3 blocks.
    std::size_t basisColumns = 3;
    // The run stops when the `target` Ritz values nearest the shift (largest |θ|) have converged.
    std::size_t target = 1;
    double convergenceTolerance = 1.0e-10;
    std::size_t maximumRestarts = 1;
    std::uint64_t seed = 0;
    // The rows of a piece of the inner products over the dimension (MassOrthogonalizer).
    std::size_t innerProductPieceRows = INNER_PRODUCT_PIECE_ROWS;
};

// What one run gave (its converged eigenpairs are appended to the store).
struct LanczosRunResult
{
    std::size_t convergedCount = 0;
    // The eigenvalue estimates σ + 1/θ of the Ritz pairs not converged, increasing.
    std::vector<double> ritzEstimates;
    std::size_t blockSteps = 0;
    std::size_t restarts = 0;
    // True when no direction outside the locked eigenvectors was left to start the run from: every root of the pencil
    // has been found.
    bool isSpaceExhausted = false;
};

// The working storage of the runs, reserved once per analysis in the budget: the basis (dimension × basisColumns),
// the next block, the product block and two scratch blocks (dimension × blockSize each).
struct LanczosWorkspace
{
    LanczosWorkspace(base::MemoryBudget& budget, std::size_t dimension, std::size_t basisColumns,
                     std::size_t blockSize);

    DenseMatrix basis;
    DenseMatrix next;
    DenseMatrix product;
    DenseMatrix massScratch;
    DenseMatrix massOfNext;

    // The bytes reserved for these sizes.
    [[nodiscard]] static std::uint64_t estimateBytes(std::size_t dimension, std::size_t basisColumns,
                                                     std::size_t blockSize);
};

// One run of the block shift-invert Lanczos method at the shift of the pencil's current factor, with thick restarts
// and full reorthogonalisation in the M-inner product.
//
// Spectral transformation (Ericsson & Ruhe, Math. Comp. 35 (1980)): the operator Op = (K − σM)⁻¹·M is self-adjoint in
// the M-inner product; its eigenvalues θ = 1/(λ − σ) are largest in magnitude for the λ nearest σ, which the Krylov
// method finds first. With V the M-orthonormal basis (Vᵀ·M·V = I), the relation kept throughout is
//
//     Op·V = V·T + V_next·G      (T symmetric, V_nextᵀ·M·V = 0, V_nextᵀ·M·V_next = I)
//
// with T block tridiagonal (A_j on the diagonal from the projection, B_j below from the QR of the new block) after
// plain steps, and an arrow (diag(θ) and the coupling G·S) after a thick restart (Wu & Simon, SIAM J. Matrix Anal.
// Appl. 22 (2000); Stewart's Krylov-Schur, SIAM J. Matrix Anal. Appl. 23 (2001)). A Ritz pair (θ, y = V·s) of T has
// the residual Op·y − θ·y = V_next·G·s, whose M-norm is ‖G·s‖₂: the pair has converged when ‖G·s‖₂ <= tol·|θ|. Its
// vector is purified, φ = V·s + V_next·G·s/θ = Op·y/θ (Nour-Omid, Parlett, Ericsson & Jensen, BIT 27 (1987)), which
// removes the components outside the range of Op (the null space of a lumped mass), then M-normalised.
//
// Every new block W = Op·V_j is M-orthogonalised twice against the locked eigenvectors and the whole basis (full
// reorthogonalisation), then factorised W = V_next·B column by column (modified Gram-Schmidt, twice); a column whose
// M-norm falls to rounding is a deflation: B gets a zero there and the column is replaced by Op applied to a seeded
// random vector, orthogonalised the same way. Every order is fixed and every product is a DenseKernels product or a
// serial sum: the bits do not depend on the threads.
//
// Thread-safety: not thread-safe. Ownership: references the pencil, the kernels, the workspace and the budget.
class LanczosRun
{
public:
    // The relative M-norm (to ε·√dimension·‖Op‖) below which a new Lanczos column is rank-deficient: the rounding of a
    // projection grows as √n·ε (a random walk of n roundings), and 100 times that leaves a margin while staying 10⁻¹⁰
    // below the norms of genuine directions.
    static constexpr double DEFLATION_FACTOR = 100.0;
    // Random vectors tried to replace one deflated column before it is left empty (the space is exhausted).
    static constexpr std::size_t REPLACEMENT_ATTEMPTS = 3;
    // A Ritz value this small relative to the largest is an empty column of the basis, never an eigenvalue.
    static constexpr double NEGLIGIBLE_RITZ_VALUE = 1.0e-14;
    // The Ritz values of T are exact only to about ε·max|θ| (the rounding of T and the backward error of the projected
    // eigensolver): a root whose |θ| is below this fraction of the largest would be known to worse than ε/ratio =
    // 2·10⁻¹³ relative in θ, and is left to a later shift where it is nearer. Measured: with the rigid-body roots of a
    // free truss in the same basis (θ ratio 3·10⁻⁶) the flexible roots came out 10⁻⁹ wrong without this rule.
    static constexpr double ACCURATE_RITZ_RATIO = 1.0e-3;

    LanczosRun(ShiftedPencil& pencil, const DenseKernels& kernels, LanczosWorkspace& workspace,
               base::MemoryBudget& budget, const LanczosRunSettings& settings, const base::OperationControl& control);

    // Runs at `shift` (the pencil holds the factor of K − shift·M) and appends the converged pairs to the store.
    [[nodiscard]] LanczosRunResult run(double shift, EigenvectorStore& store);

private:
    struct RitzPairs
    {
        std::vector<double> values;    // θ, in decreasing |θ| order
        std::vector<double> vectors;   // p × p, column k for values[k]
        std::vector<double> residuals; // ‖G·s‖₂
        std::vector<unsigned char> isConverged;
        // The Ritz values accurate enough to be accepted (|θ| >= ACCURATE_RITZ_RATIO·max|θ|): the first ones.
        std::size_t accurateCount = 0;
    };

    void startBlock(const EigenvectorStore& store);
    void applyOperator(DenseMatrixView block);
    // Factorises the block (already M-orthogonal to the locked set and the basis) as Q·B; Q replaces the block.
    // Returns the columns filled (fewer than the block's when the space is exhausted: the others are zero).
    std::size_t factoriseBlock(DenseMatrixView block, std::vector<double>* triangle, const EigenvectorStore& store);
    void fillRandom(DenseMatrixView block);
    [[nodiscard]] RitzPairs computeRitzPairs() const;
    void restart(const RitzPairs& ritz);
    void extractConverged(const RitzPairs& ritz, double shift, EigenvectorStore& store, LanczosRunResult& result);
    [[nodiscard]] bool hasConvergedTarget(const RitzPairs& ritz) const;
    [[nodiscard]] ConstDenseMatrixView basisView(std::size_t columns) const;

    ShiftedPencil& m_pencil;
    const DenseKernels& m_kernels;
    LanczosWorkspace& m_workspace;
    base::MemoryBudget& m_budget;
    LanczosRunSettings m_settings;
    base::OperationControl m_control;
    MassOrthogonalizer m_orthogonalizer;
    std::size_t m_dimension = 0;
    std::size_t m_columns = 0;   // columns of the basis in use (p)
    std::size_t m_capacity = 0;  // usable basis columns
    double m_operatorNorm = 0.0; // the largest ‖Op·v‖_M seen: the scale of the deflation test
    std::uint64_t m_randomState = 0;
    std::vector<double> m_projected; // T, capacity × capacity, column-major: bounded by the basis
    std::vector<double> m_coupling;  // G, blockSize × capacity, column-major
    // The norms each column of the block being factorised is compared with for the deflation test.
    std::vector<double> m_referenceNorms;
    std::size_t m_basisInUse = 0;      // the basis columns a replacement column is orthogonalised against
    std::size_t m_nextColumnCount = 0; // the non-empty columns of the next block
};

} // namespace oos::linalg
