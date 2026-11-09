#pragma once

#include <array>
#include <string_view>

namespace oos::linalg
{

// The families of dense kernels, each compiled with its own target flags
// (cmake/OosKernels.cmake):
// - Reference: scalar, fixed loop order, no fused multiply-add: the same bits on every machine, the oracle;
// - Avx2: AVX2 and FMA (Haswell and later, Zen and later);
// - Avx512: AVX-512 F, DQ and VL (Skylake-SP and later server parts, Zen 4). Not validated on the reference PC, which
//   has no AVX-512: compiled, selectable by force, never selected automatically.
enum class KernelKind
{
    Reference,
    Avx2,
    Avx512
};

inline constexpr std::array<KernelKind, 3> ALL_KERNEL_KINDS = {KernelKind::Reference, KernelKind::Avx2,
                                                               KernelKind::Avx512};

// The name of a kernel family as the option `kernel=`, the variable OOS_KERNEL and the reports write it.
[[nodiscard]] constexpr std::string_view kernelName(KernelKind kind) noexcept
{
    switch (kind)
    {
    case KernelKind::Reference:
        return "reference";
    case KernelKind::Avx2:
        return "avx2";
    case KernelKind::Avx512:
        return "avx512";
    }
    return "unknown";
}

} // namespace oos::linalg
