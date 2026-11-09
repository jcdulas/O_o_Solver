#include "oos/base/platform/SubnormalModeScope.hpp"

namespace oos::base
{

namespace
{

#if defined(__x86_64__)

// The bits of the SSE control and status register MXCSR (Intel 64 and IA-32 Architectures Software Developer's
// Manual, vol. 1, figure 10-3): FTZ, flush to zero, bit 15; DAZ, denormals are zero, bit 6. Every x86-64 processor has
// both (DAZ came with SSE2's second steppings, before x86-64).
constexpr std::uint32_t FLUSH_TO_ZERO = 1U << 15;
constexpr std::uint32_t DENORMALS_ARE_ZERO = 1U << 6;
constexpr std::uint32_t FLUSH_BITS = FLUSH_TO_ZERO | DENORMALS_ARE_ZERO;

std::uint32_t readControl() noexcept
{
    return __builtin_ia32_stmxcsr();
}

void writeControl(std::uint32_t control) noexcept
{
    __builtin_ia32_ldmxcsr(control);
}

#endif

} // namespace

bool canFlushSubnormals() noexcept
{
#if defined(__x86_64__)
    return true;
#else
    return false;
#endif
}

SubnormalMode subnormalModeOfThisThread() noexcept
{
#if defined(__x86_64__)
    return (readControl() & FLUSH_BITS) == FLUSH_BITS ? SubnormalMode::Flush : SubnormalMode::Ieee;
#else
    return SubnormalMode::Ieee;
#endif
}

SubnormalModeScope::SubnormalModeScope([[maybe_unused]] SubnormalMode mode) noexcept
{
#if defined(__x86_64__)
    m_saved = readControl();
    const std::uint32_t wanted = mode == SubnormalMode::Flush ? (m_saved | FLUSH_BITS) : (m_saved & ~FLUSH_BITS);
    if (wanted != m_saved)
    {
        writeControl(wanted);
        m_isChanged = true;
    }
#endif
}

SubnormalModeScope::~SubnormalModeScope()
{
#if defined(__x86_64__)
    // Only the two bits of the mode go back: an exception flag the work raised meanwhile stays raised.
    if (m_isChanged)
        writeControl((readControl() & ~FLUSH_BITS) | (m_saved & FLUSH_BITS));
#endif
}

} // namespace oos::base
