#include "KernelFamily.hpp"

namespace oos::linalg
{
namespace
{

constexpr KernelFamily REFERENCE_FAMILY = {&kernels::reference::subtractProductTile,
                                           &kernels::reference::subtractColumnProducts,
                                           &kernels::reference::solvePanel,
                                           &kernels::reference::solveUnitLowerBlock,
                                           &kernels::reference::solveUnitLowerTransposedBlock,
                                           &kernels::reference::subtractRowProducts,
                                           &kernels::reference::solveRowTriangle,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           false};

#if !defined(OOS_WASM)
// The x86 vector families. WebAssembly has no AVX: its build compiles only the reference family (CMakeLists.txt).
constexpr KernelFamily AVX2_FAMILY = {&kernels::avx2::subtractProductTile,
                                      &kernels::avx2::subtractColumnProducts,
                                      &kernels::avx2::solvePanel,
                                      &kernels::avx2::solveUnitLowerBlock,
                                      &kernels::avx2::solveUnitLowerTransposedBlock,
                                      &kernels::avx2::subtractRowProducts,
                                      &kernels::avx2::solveRowTriangle,
                                      &kernels::avx2::packProductColumns,
                                      &kernels::avx2::subtractPackedProduct,
                                      &kernels::avx2::packProductRows,
                                      &kernels::avx2::subtractPrepackedProduct,
                                      true};

constexpr KernelFamily AVX512_FAMILY = {&kernels::avx512::subtractProductTile,
                                        &kernels::avx512::subtractColumnProducts,
                                        &kernels::avx512::solvePanel,
                                        &kernels::avx512::solveUnitLowerBlock,
                                        &kernels::avx512::solveUnitLowerTransposedBlock,
                                        &kernels::avx512::subtractRowProducts,
                                        &kernels::avx512::solveRowTriangle,
                                        &kernels::avx512::packProductColumns,
                                        &kernels::avx512::subtractPackedProduct,
                                        &kernels::avx512::packProductRows,
                                        &kernels::avx512::subtractPrepackedProduct,
                                        true};
#endif

} // namespace

const KernelFamily& kernelFamilyOf(KernelKind kind) noexcept
{
    switch (kind)
    {
    case KernelKind::Reference:
        return REFERENCE_FAMILY;
#if defined(OOS_WASM)
    case KernelKind::Avx2:
    case KernelKind::Avx512:
        return REFERENCE_FAMILY; // never selected in WebAssembly (no CPU feature is detected there)
#else
    case KernelKind::Avx2:
        return AVX2_FAMILY;
    case KernelKind::Avx512:
        return AVX512_FAMILY;
#endif
    }
    return REFERENCE_FAMILY;
}

} // namespace oos::linalg
