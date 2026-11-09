#include "oos/base/platform/CpuFeatures.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define OOS_HAS_CPUID 1
#endif

namespace oos::base
{
namespace
{

#if defined(OOS_HAS_CPUID)

// The bits read below are those of the Intel 64 and IA-32 Architectures Software Developer's Manual, vol. 2A, CPUID
// (tables 3-8 and 3-10) and vol. 1 §13.3 (XCR0); AMD documents the same bits for the same features.
constexpr unsigned LEAF_FEATURES = 1;
constexpr unsigned LEAF_EXTENDED_FEATURES = 7;
constexpr unsigned LEAF_FREQUENCY = 0x16;
constexpr unsigned LEAF_BRAND_FIRST = 0x80000002U;
constexpr unsigned LEAF_BRAND_LAST = 0x80000004U;

constexpr unsigned ECX1_FMA = 1U << 12;
constexpr unsigned ECX1_OSXSAVE = 1U << 27; // the OS enabled XSAVE, so XGETBV may be executed
constexpr unsigned ECX1_AVX = 1U << 28;
constexpr unsigned EBX7_AVX2 = 1U << 5;
constexpr unsigned EBX7_AVX512F = 1U << 16;
constexpr unsigned EBX7_AVX512DQ = 1U << 17;
constexpr unsigned EBX7_AVX512VL = 1U << 31;

// XCR0: the register states the OS saves. AVX needs the SSE (bit 1) and YMM-upper (bit 2) states; AVX-512 also needs
// the opmask (bit 5), ZMM0-15 upper halves (bit 6) and ZMM16-31 (bit 7) states.
constexpr std::uint64_t XCR0_AVX_STATES = 0x6;
constexpr std::uint64_t XCR0_AVX512_STATES = 0xE6;

struct CpuidRegisters
{
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
};

CpuidRegisters readCpuid(unsigned leaf, unsigned subleaf)
{
    CpuidRegisters registers;
    __cpuid_count(leaf, subleaf, registers.eax, registers.ebx, registers.ecx, registers.edx);
    return registers;
}

// XGETBV with ECX = 0 reads XCR0. Written as an instruction rather than the _xgetbv intrinsic, which would need the
// whole translation unit compiled with -mxsave (the baseline code must run on any x86-64).
std::uint64_t readXcr0()
{
    unsigned low = 0;
    unsigned high = 0;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<std::uint64_t>(high) << 32) | low;
}

std::string readVendor(const CpuidRegisters& leafZero)
{
    // The vendor string is EBX, EDX, ECX in that order (12 characters).
    std::array<char, 12> text{};
    std::memcpy(text.data(), &leafZero.ebx, 4);
    std::memcpy(text.data() + 4, &leafZero.edx, 4);
    std::memcpy(text.data() + 8, &leafZero.ecx, 4);
    return {text.data(), text.size()};
}

std::string readBrand()
{
    if (__get_cpuid_max(0x80000000U, nullptr) < LEAF_BRAND_LAST)
        return {};
    std::array<char, 48> text{};
    for (unsigned leaf = LEAF_BRAND_FIRST; leaf <= LEAF_BRAND_LAST; ++leaf)
    {
        const CpuidRegisters registers = readCpuid(leaf, 0);
        char* destination = text.data() + 16 * (leaf - LEAF_BRAND_FIRST);
        std::memcpy(destination, &registers.eax, 4);
        std::memcpy(destination + 4, &registers.ebx, 4);
        std::memcpy(destination + 8, &registers.ecx, 4);
        std::memcpy(destination + 12, &registers.edx, 4);
    }
    std::string brand(text.begin(), std::find(text.begin(), text.end(), '\0'));
    // Some processors pad the brand string with leading spaces.
    const auto first = brand.find_first_not_of(' ');
    return first == std::string::npos ? std::string{} : brand.substr(first);
}

#endif

} // namespace

CpuFeatures detectCpuFeatures()
{
    CpuFeatures features;
#if defined(OOS_HAS_CPUID)
    const unsigned maxLeaf = __get_cpuid_max(0, nullptr);
    if (maxLeaf == 0)
        return features;
    features.vendor = readVendor(readCpuid(0, 0));
    features.brand = readBrand();

    const CpuidRegisters leaf1 = readCpuid(LEAF_FEATURES, 0);
    const bool isOsSavingXsaveState = (leaf1.ecx & ECX1_OSXSAVE) != 0;
    const std::uint64_t xcr0 = isOsSavingXsaveState ? readXcr0() : 0;
    const bool isOsSavingAvxState = (xcr0 & XCR0_AVX_STATES) == XCR0_AVX_STATES;
    const bool isOsSavingAvx512State = (xcr0 & XCR0_AVX512_STATES) == XCR0_AVX512_STATES;
    const bool hasAvx = (leaf1.ecx & ECX1_AVX) != 0 && isOsSavingAvxState;
    features.hasFma = hasAvx && (leaf1.ecx & ECX1_FMA) != 0;

    if (maxLeaf >= LEAF_EXTENDED_FEATURES)
    {
        const CpuidRegisters leaf7 = readCpuid(LEAF_EXTENDED_FEATURES, 0);
        features.hasAvx2 = hasAvx && (leaf7.ebx & EBX7_AVX2) != 0;
        features.hasAvx512f = isOsSavingAvx512State && (leaf7.ebx & EBX7_AVX512F) != 0;
        features.hasAvx512dq = isOsSavingAvx512State && (leaf7.ebx & EBX7_AVX512DQ) != 0;
        features.hasAvx512vl = isOsSavingAvx512State && (leaf7.ebx & EBX7_AVX512VL) != 0;
    }
    if (maxLeaf >= LEAF_FREQUENCY)
    {
        const CpuidRegisters leaf16 = readCpuid(LEAF_FREQUENCY, 0);
        features.baseFrequencyInMegahertz = static_cast<int>(leaf16.eax & 0xFFFFU);
        features.maxFrequencyInMegahertz = static_cast<int>(leaf16.ebx & 0xFFFFU);
    }
#endif
    return features;
}

} // namespace oos::base
