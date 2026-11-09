#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <vector>

namespace oos::linalg
{

// A = V·diag(λ)·Vᵀ: the eigenvalues in increasing order, and the orthonormal eigenvectors as the columns of V, each
// signed so that its component of largest magnitude is positive (the first such component on a tie).
struct SymmetricEigenDecomposition
{
    std::vector<double> eigenvalues;
    DenseMatrix eigenvectors;
};

// The complete eigendecomposition of a small dense symmetric matrix: the projected problems of the block Lanczos
// solver (a few hundred at most) and the checks of external units.
//
// Method: Householder reduction to tridiagonal form with the transformations accumulated, then the implicit QL
// algorithm with Wilkinson-type shifts on the tridiagonal matrix (EISPACK tred2 and tql2: Bowdler, Martin, Reinsch and
// Wilkinson, Numer. Math. 11 (1968); as in the public-domain JAMA library). About 9n³ flops: 0.2 GFLOP for n = 300.
// Backward stable: the computed pairs are exact for A + E with ‖E‖ of the order of n·ε·‖A‖ (Golub & Van Loan §8.3).
// Serial, with a fixed order of operations and no library function whose rounding may vary (√ only, correctly rounded):
// the same input gives the same bits.
//
// Thread-safety: stateless; any number of calls may run at once.
class SmallEigenSolver
{
public:
    // The QL sweeps allowed per eigenvalue before giving up (EISPACK uses 30; two or three are typical).
    static constexpr std::size_t MAX_SWEEPS_PER_EIGENVALUE = 30;

    // The decomposition of the symmetric `a` (n × n; its lower triangle is read). The eigenvector matrix is reserved in
    // `budget`. Throws std::invalid_argument when `a` is not square, SolverError when a term is not finite or the QL
    // iteration does not converge.
    [[nodiscard]] static SymmetricEigenDecomposition decompose(ConstDenseMatrixView a, base::MemoryBudget& budget);
};

} // namespace oos::linalg
