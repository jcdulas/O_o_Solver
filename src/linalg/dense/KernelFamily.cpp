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
                                           false,
                                           false};

#if defined(OOS_WASM)
// The vector family of WebAssembly, which its build compiles beside the reference family (CMakeLists.txt).
constexpr KernelFamily WASM128_FAMILY = {&kernels::wasm128::subtractProductTile,
                                         &kernels::wasm128::subtractColumnProducts,
                                         &kernels::wasm128::solvePanel,
                                         &kernels::wasm128::solveUnitLowerBlock,
                                         &kernels::wasm128::solveUnitLowerTransposedBlock,
                                         &kernels::wasm128::subtractRowProducts,
                                         &kernels::wasm128::solveRowTriangle,
                                         &kernels::wasm128::packProductColumns,
                                         &kernels::wasm128::subtractPackedProduct,
                                         &kernels::wasm128::packProductRows,
                                         &kernels::wasm128::subtractPrepackedProduct,
                                         true,
                                         false};
#else
// The x86 vector families. WebAssembly has no AVX: its build does not compile them.
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
                                      true,
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
                                        true,
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
    case KernelKind::Wasm128:
        return WASM128_FAMILY;
#else
    case KernelKind::Avx2:
        return AVX2_FAMILY;
    case KernelKind::Avx512:
        return AVX512_FAMILY;
    case KernelKind::Wasm128:
        return REFERENCE_FAMILY; // compiled in the WebAssembly build alone (never selected elsewhere)
#endif
    }
    return REFERENCE_FAMILY;
}

} // namespace oos::linalg
