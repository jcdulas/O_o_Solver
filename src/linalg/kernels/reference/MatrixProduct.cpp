// The scalar reference family: compiled with -ffp-contract=off (cmake/OosKernels.cmake) so that the compiler
// never fuses a multiply and an add, whatever the target flags: the oracle gives the same bits everywhere.

#include "oos/linalg/kernels/MatrixProduct.hpp"

namespace oos::linalg::kernels::reference
{

void multiplyMatrices(const MatrixProductOperands& operands) noexcept
{
    const std::size_t m = operands.rowCount;
    const std::size_t n = operands.columnCount;
    const std::size_t k = operands.innerCount;
    // Loop order j, p, i: for each column of C, the columns of A are added one after the other, scaled by b(p, j).
    // Every c(i, j) is thus summed in increasing p. The compiler may vectorise the i loop (independent elements), which
    // does not change the order of any sum.
    for (std::size_t j = 0; j < n; ++j)
    {
        double* cColumn = operands.c + j * operands.ldc;
        for (std::size_t i = 0; i < m; ++i)
            cColumn[i] = 0.0;
        for (std::size_t p = 0; p < k; ++p)
        {
            const double bValue = operands.b[p + j * operands.ldb];
            const double* aColumn = operands.a + p * operands.lda;
            for (std::size_t i = 0; i < m; ++i)
                cColumn[i] += aColumn[i] * bValue;
        }
    }
}

} // namespace oos::linalg::kernels::reference
