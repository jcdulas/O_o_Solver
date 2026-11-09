#include "oos/reduction/patterns/BasisCoefficients.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>

namespace oos::reduction
{

void BasisCoefficients::combine(linalg::ConstDenseMatrixView basis, std::size_t firstCase, linalg::DenseMatrixView out,
                                const linalg::DenseKernels& kernels) const
{
    if (basis.columns != size() || basis.rows != out.rows || firstCase > caseCount() ||
        out.columns > caseCount() - firstCase)
        throw std::invalid_argument(base::concatenate(
            {"BasisCoefficients::combine: ", std::to_string(basis.rows), " x ", std::to_string(basis.columns),
             " basis values into ", std::to_string(out.rows), " x ", std::to_string(out.columns), " from case ",
             std::to_string(firstCase), "; expected ", std::to_string(size()),
             " basis columns, as many rows out as in, and a batch within the ", std::to_string(caseCount()),
             " cases"}));
    for (std::size_t j = 0; j < out.columns; ++j)
        for (std::size_t i = 0; i < out.rows; ++i)
            out(i, j) = 0.0;
    if (out.columns == 0 || out.rows == 0 || size() == 0)
        return;
    // 0 − Σ_i b_i·α_i, then the sign changed: IEEE 754 rounding is symmetric, so −(0 − Σ) has the bits of Σ taken in
    // the same order; subtractProduct is the kernels' product with a fixed order of terms for every element.
    const linalg::ConstDenseMatrixView alpha = m_alpha.view().block(0, firstCase, size(), out.columns);
    kernels.subtractProduct(out, basis, alpha);
    for (std::size_t j = 0; j < out.columns; ++j)
        for (std::size_t i = 0; i < out.rows; ++i)
            out(i, j) = -out(i, j);
}

} // namespace oos::reduction
