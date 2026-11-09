// The dispatch of the matrix product to its kernel family, compiled at the baseline x86-64 like the rest of the code.

#include "oos/linalg/kernels/MatrixProduct.hpp"

#include "oos/linalg/kernels/KernelKind.hpp"

#include <algorithm>

namespace oos::linalg
{

void multiplyMatrices(KernelKind kind, const MatrixProductOperands& operands) noexcept
{
    switch (kind)
    {
    case KernelKind::Reference:
        kernels::reference::multiplyMatrices(operands);
        return;
#if defined(OOS_WASM)
    case KernelKind::Avx2:
    case KernelKind::Avx512:
        // WebAssembly has no AVX family (never selected there): the reference kernel.
        kernels::reference::multiplyMatrices(operands);
        return;
#else
    case KernelKind::Avx2:
        kernels::avx2::multiplyMatrices(operands);
        return;
    case KernelKind::Avx512:
        kernels::avx512::multiplyMatrices(operands);
        return;
#endif
    }
}

void multiplyRowMajor(KernelKind kind, const double* a, const double* b, double* c, std::size_t m, std::size_t n,
                      std::size_t k) noexcept
{
    // Cᵀ (n × m, column-major = C row-major) = Bᵀ (n × k, column-major = B row-major) · Aᵀ (k × m).
    MatrixProductOperands operands;
    operands.a = b;
    operands.lda = std::max<std::size_t>(n, 1);
    operands.b = a;
    operands.ldb = std::max<std::size_t>(k, 1);
    operands.c = c;
    operands.ldc = std::max<std::size_t>(n, 1);
    operands.rowCount = n;
    operands.columnCount = m;
    operands.innerCount = k;
    multiplyMatrices(kind, operands);
}

} // namespace oos::linalg
