#pragma once

#include <string>

namespace oos::base
{

// What the processor and the operating system together allow the dense kernels to use, read once at run time
// (the kernel family is chosen by CPU detection). An instruction set counts only when the CPU reports it
// (CPUID) and the operating system saves the registers it uses across context switches (XGETBV, XCR0): a CPU with
// AVX-512 under an OS that does not save the ZMM registers must not run AVX-512 code.
//
// A plain value: the tests build their own to play a CPU that lacks a feature.
struct CpuFeatures
{
    std::string vendor; // "GenuineIntel", "AuthenticAMD", ... ("" when CPUID is not available)
    std::string brand;  // the processor brand string, e.g. "12th Gen Intel(R) Core(TM) i5-12400F"
    bool hasAvx2 = false;
    bool hasFma = false;
    bool hasAvx512f = false;
    bool hasAvx512dq = false;
    bool hasAvx512vl = false;
    // The frequencies the CPU reports in CPUID leaf 0x16 (Intel, Skylake and later); 0 when it does not report them.
    // They are the published base and maximum (single-core turbo) frequencies, not the current one.
    int baseFrequencyInMegahertz = 0;
    int maxFrequencyInMegahertz = 0;

    // AVX2 and FMA together: what the AVX2 kernel family is compiled for (-mavx2 -mfma).
    [[nodiscard]] bool hasAvx2Family() const noexcept { return hasAvx2 && hasFma; }
    // AVX-512 F, DQ and VL: what the AVX-512 kernel family is compiled for (-mavx512f -mavx512dq -mavx512vl), which
    // also implies AVX2 and FMA on every CPU that has them.
    [[nodiscard]] bool hasAvx512Family() const noexcept
    {
        return hasAvx512f && hasAvx512dq && hasAvx512vl && hasAvx2Family();
    }
};

// Reads the features of the processor this process runs on. Cheap (a few CPUID instructions), but callers keep the
// result: it does not change while the process lives. On a processor that is not x86-64, every feature is false.
[[nodiscard]] CpuFeatures detectCpuFeatures();

} // namespace oos::base
