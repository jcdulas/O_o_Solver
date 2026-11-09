#pragma once

#include <cstdint>

namespace oos::base
{

// How the floating-point arithmetic of a thread treats the subnormal numbers, those below the smallest normal double
// (DBL_MIN, about 2.2e-308): IEEE 754 gradual underflow (Ieee: the default of every thread, O_o_Solver's reference), or
// flushed to zero (Flush: a subnormal result becomes 0, flush to zero, and a subnormal operand is read as 0, denormals
// are zero), which spares the microcode assist that each vector operation meeting a subnormal costs on x86-64 (the
// solves of a matrix whose inverse decays exponentially fill their solutions with subnormal values). Results
// differ from IEEE ones only where a subnormal value appears.
enum class SubnormalMode : std::uint8_t
{
    Ieee,
    Flush
};

// Whether this machine can flush subnormal numbers: x86-64, its SSE control register MXCSR (bit 15 FTZ, bit 6 DAZ;
// Intel 64 and IA-32 Architectures Software Developer's Manual, vol. 1 §10.2.3); elsewhere false, and Flush is IEEE.
[[nodiscard]] bool canFlushSubnormals() noexcept;

// The subnormal mode of the calling thread now: Flush when both FTZ and DAZ are set, Ieee otherwise.
[[nodiscard]] SubnormalMode subnormalModeOfThisThread() noexcept;

// Sets the subnormal mode of the calling thread for the scope's lifetime and gives the thread back its FTZ and DAZ
// bits as they were when the scope ends; the other bits of the control register (the rounding, the exception masks)
// are never touched, and a scope of the mode the thread is already in writes nothing.
//
// The mode belongs to a thread: O_o_Solver sets it on its own threads only (the thread of a background run, and the
// pool's workers while they run a task pushed from a thread in that mode: base::ThreadPool), never on the caller's
// thread for longer than a call, so that the caller's own code keeps IEEE arithmetic.
//
// Thread-safety: an object of one thread; not copyable nor movable (it restores the thread that made it).
class SubnormalModeScope
{
public:
    explicit SubnormalModeScope(SubnormalMode mode) noexcept;
    ~SubnormalModeScope();
    SubnormalModeScope(const SubnormalModeScope&) = delete;
    SubnormalModeScope& operator=(const SubnormalModeScope&) = delete;
    SubnormalModeScope(SubnormalModeScope&&) = delete;
    SubnormalModeScope& operator=(SubnormalModeScope&&) = delete;

private:
    [[maybe_unused]] std::uint32_t m_saved = 0; // unused where the mode cannot be changed (WebAssembly)
    [[maybe_unused]] bool m_isChanged = false;
};

} // namespace oos::base
