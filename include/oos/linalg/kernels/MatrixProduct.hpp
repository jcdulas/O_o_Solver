#pragma once

#include <cstddef>

// This header is included by the translation units of every kernel family, each compiled with its own target flags
// (-mavx2 -mfma, -mavx512f ...). It must therefore hold no inline function and include no header that defines one: an
// inline function compiled in the AVX2 unit and in a baseline unit is one symbol for the linker, which may keep the
// AVX2 copy and run it on a processor without AVX2. Only declarations and plain data live here.

namespace oos::linalg
{

enum class KernelKind; // kernels/KernelKind.hpp

// The operands of C = A·B, all column-major with leading dimensions (element (i, j) of A is a[i + j·lda]):
// A is rowCount × innerCount, B is innerCount × columnCount, C is rowCount × columnCount and is overwritten.
// C must not overlap A or B. The leading dimensions are at least the row counts of their matrices (at least 1).
struct MatrixProductOperands
{
    const double* a = nullptr;
    std::size_t lda = 1;
    const double* b = nullptr;
    std::size_t ldb = 1;
    double* c = nullptr;
    std::size_t ldc = 1;
    std::size_t rowCount = 0;
    std::size_t columnCount = 0;
    std::size_t innerCount = 0;
};

// C = A·B with the kernel family `kind`. The caller has checked that the processor supports it (KernelSelection);
// calling a SIMD family on a processor without its instructions is undefined behaviour (an illegal instruction).
void multiplyMatrices(KernelKind kind, const MatrixProductOperands& operands) noexcept;

// C = A·B for contiguous row-major A (m × k), B (k × n) and C (m × n), as C arrays are laid out. A row-major
// matrix is the column-major storage of its transpose, so C = A·B is computed as Cᵀ = Bᵀ·Aᵀ by multiplyMatrices,
// without moving any value. Same preconditions as multiplyMatrices.
void multiplyRowMajor(KernelKind kind, const double* a, const double* b, double* c, std::size_t m, std::size_t n,
                      std::size_t k) noexcept;

namespace kernels::reference
{
// c(i, j) = Σ_p a(i, p)·b(p, j), each sum taken in increasing p, every product rounded before its addition (no fused
// multiply-add): the same bits on every machine and every compiler that follows IEEE 754 double arithmetic.
void multiplyMatrices(const MatrixProductOperands& operands) noexcept;
} // namespace kernels::reference

namespace kernels::avx2
{
// c(i, j) = fma(a(i, k-1), b(k-1, j), ... fma(a(i, 0), b(0, j), 0)): one fused multiply-add per term, in increasing p,
// whatever the position of (i, j) in the register blocks; so the bits do not depend on the blocking or the sizes.
void multiplyMatrices(const MatrixProductOperands& operands) noexcept;
} // namespace kernels::avx2

namespace kernels::avx512
{
// The same sequence of fused multiply-adds as the AVX2 family, hence the same bits.
void multiplyMatrices(const MatrixProductOperands& operands) noexcept;
} // namespace kernels::avx512

} // namespace oos::linalg
