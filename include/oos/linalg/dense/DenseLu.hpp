#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <cstdint>

namespace oos::linalg
{

// The LU factorisation with partial pivoting of a dense square matrix that is not symmetric, P·A = L·U (Golub & Van
// Loan, Matrix Computations, 4th ed., algorithm 3.4.1, the right-looking "kji" form): at column k the row of largest
// magnitude at or below the diagonal is swapped up, the multipliers l_ik = a_ik / a_kk stored below it and the trailing
// block updated, a_ij ← a_ij − l_ik·u_kj. Used for the aerodynamic influence matrices of the doublet-lattice method
// (SOL 144), a few thousand boxes: about 2n³/3 flops, 0.6 GFLOP for 1 000 boxes.
//
// Serial, with a fixed order of operations (the trailing update column by column, row by row): the same matrix gives
// the same bits. Backward stable in practice (growth factor bounded by 2^(n−1), small on the matrices met).
//
// Thread-safety: none (a factor is used by one thread at a time). Ownership: owns the factored matrix and the pivots,
// reserved in the budget.
class DenseLu
{
public:
    // Factors `a` (taken over, n × n). Throws std::invalid_argument for a matrix that is not square; a zero (or not
    // finite) pivot leaves the factor singular (isSingular), with singularColumn the first such column.
    DenseLu(DenseMatrix a, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t order() const noexcept { return m_factor.rows(); }
    [[nodiscard]] bool isSingular() const noexcept { return m_singularColumn >= 0; }
    // The first column whose pivot was zero or not finite, −1 when none.
    [[nodiscard]] std::int64_t singularColumn() const noexcept { return m_singularColumn; }
    // The ratio of the smallest to the largest pivot magnitude: an estimate of how near to singular the matrix is.
    [[nodiscard]] double pivotRatio() const noexcept { return m_pivotRatio; }

    // Solves A·X = B in place (`rhs` n × m holds B, then X). Throws std::invalid_argument when the rows differ from the
    // order or the factor is singular.
    void solve(DenseMatrixView rhs) const;

private:
    DenseMatrix m_factor;                        // L below the diagonal (unit diagonal implied), U on and above it
    base::BudgetedBuffer<std::int64_t> m_pivots; // row k was swapped with row m_pivots[k] at step k
    std::int64_t m_singularColumn = -1;
    double m_pivotRatio = 0.0;
};

} // namespace oos::linalg
