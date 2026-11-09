#pragma once

#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <utility>

namespace oos::reduction
{

// The coefficients α of the load cases on a basis of vectors (each case is f_case = Σ_i α_i,case·f_i)
// and the one way a result known on the basis vectors becomes the result of every load case (results are combined,
// not recomputed, per load case in a fixed order): displacements, forces,
// stresses and strains are linear in the loads, so
//     r_case = Σ_i α_i,case·r_i,
// each term of each case summed over i in increasing order by the dense kernels (DenseKernels::subtractProduct: one
// chain of terms per result element, whatever the tiles and the threads), so that a case has the same bits
// alone, in any batch of cases, and with any number of threads.
//
// Thread-safety: immutable once built; combine() may run concurrently. Ownership: owns α (budgeted).
class BasisCoefficients
{
public:
    BasisCoefficients() = default;
    // α: one row per basis vector, one column per load case.
    explicit BasisCoefficients(linalg::DenseMatrix alpha) : m_alpha(std::move(alpha)) {}

    [[nodiscard]] std::size_t size() const noexcept { return m_alpha.rows(); }
    [[nodiscard]] std::size_t caseCount() const noexcept { return m_alpha.columns(); }
    [[nodiscard]] const linalg::DenseMatrix& matrix() const noexcept { return m_alpha; }
    [[nodiscard]] double operator()(std::size_t vector, std::size_t loadCase) const noexcept
    {
        return m_alpha(vector, loadCase);
    }

    // out(:, j) = Σ_i basis(:, i)·α(i, firstCase + j) for the out.columns cases from firstCase: `basis` has size()
    // columns (any number of rows), `out` as many rows. Throws std::invalid_argument on a size mismatch or a batch
    // beyond the cases.
    void combine(linalg::ConstDenseMatrixView basis, std::size_t firstCase, linalg::DenseMatrixView out,
                 const linalg::DenseKernels& kernels) const;

private:
    linalg::DenseMatrix m_alpha;
};

} // namespace oos::reduction
