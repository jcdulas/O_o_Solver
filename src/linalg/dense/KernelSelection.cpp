#include "oos/linalg/dense/KernelSelection.hpp"

#include "oos/base/errors/SolverError.hpp"

#include <cctype>
#include <cstdlib>
#include <optional>
#include <utility>

namespace oos::linalg
{
namespace
{

std::string toLowerCase(std::string_view text)
{
    std::string lower(text);
    for (char& character : lower)
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    return lower;
}

bool isAutomatic(std::string_view value)
{
    return value.empty() || toLowerCase(value) == KernelSelection::AUTOMATIC_NAME;
}

std::optional<KernelKind> findKernel(std::string_view value)
{
    const std::string lower = toLowerCase(value);
    for (const KernelKind kind : ALL_KERNEL_KINDS)
        if (lower == kernelName(kind))
            return kind;
    return std::nullopt;
}

std::string describeSource(KernelSelectionSource source)
{
    switch (source)
    {
    case KernelSelectionSource::Option:
        return "the option kernel=";
    case KernelSelectionSource::Environment:
        return std::string("the environment variable ").append(KernelSelection::ENVIRONMENT_VARIABLE);
    case KernelSelectionSource::Automatic:
        return "the automatic choice";
    }
    return {};
}

std::string describeInstructions(KernelKind kind)
{
    switch (kind)
    {
    case KernelKind::Reference:
        return "no SIMD instruction";
    case KernelKind::Avx2:
        return "AVX2 and FMA";
    case KernelKind::Avx512:
        return "AVX-512 F, DQ and VL";
    }
    return {};
}

std::string processorName(const base::CpuFeatures& cpu)
{
    return cpu.brand.empty() ? std::string("this processor") : cpu.brand;
}

std::string listSupportedKernels(const base::CpuFeatures& cpu)
{
    std::string list;
    for (const KernelKind kind : ALL_KERNEL_KINDS)
    {
        if (!isKernelSupported(kind, cpu))
            continue;
        if (!list.empty())
            list.append(", ");
        list.append(kernelName(kind));
    }
    return list;
}

[[noreturn]] void refuseUnknownName(std::string_view value, KernelSelectionSource source)
{
    std::string message("unknown dense kernel '");
    message.append(value).append("' given by ").append(describeSource(source)).append("; expected one of ");
    message.append(KernelSelection::AUTOMATIC_NAME);
    for (const KernelKind kind : ALL_KERNEL_KINDS)
        message.append(", ").append(kernelName(kind));
    throw base::SolverError(message);
}

[[noreturn]] void refuseUnsupported(KernelKind kind, KernelSelectionSource source, const base::CpuFeatures& cpu)
{
    std::string message("the dense kernel '");
    message.append(kernelName(kind))
        .append("' was requested by ")
        .append(describeSource(source))
        .append(", but ")
        .append(processorName(cpu))
        .append(" does not support ")
        .append(describeInstructions(kind))
        .append(" (or the operating system does not enable them); the kernels it supports are: ")
        .append(listSupportedKernels(cpu))
        .append(". Request one of them, or '")
        .append(KernelSelection::AUTOMATIC_NAME)
        .append("' for the automatic choice.");
    throw base::SolverError(message);
}

} // namespace

KernelSelection::KernelSelection(KernelKind kind, KernelSelectionSource source, std::string processorBrand)
    : m_kind(kind), m_source(source), m_processorBrand(std::move(processorBrand))
{
}

KernelSelection KernelSelection::select(const base::CpuFeatures& cpu, std::string_view option,
                                        std::string_view environmentValue)
{
    // The option wins over the environment variable, which wins over the detection.
    const bool isOptionGiven = !isAutomatic(option);
    const bool isEnvironmentGiven = !isAutomatic(environmentValue);
    if (isOptionGiven || isEnvironmentGiven)
    {
        const KernelSelectionSource source =
            isOptionGiven ? KernelSelectionSource::Option : KernelSelectionSource::Environment;
        const std::string_view value = isOptionGiven ? option : environmentValue;
        const std::optional<KernelKind> kind = findKernel(value);
        if (!kind)
            refuseUnknownName(value, source);
        if (!isKernelSupported(*kind, cpu))
            refuseUnsupported(*kind, source, cpu);
        return {*kind, source, processorName(cpu)};
    }
    // AVX-512 is not chosen automatically until it is validated on a machine that has it.
    const KernelKind automatic = cpu.hasAvx2Family() ? KernelKind::Avx2 : KernelKind::Reference;
    return {automatic, KernelSelectionSource::Automatic, processorName(cpu)};
}

KernelSelection KernelSelection::selectForThisProcess(std::string_view option)
{
    // std::getenv is standard C++; the MinGW headers mark it deprecated for clang (clang-tidy) in favour of _dupenv_s,
    // which is not standard. Reading the variable once per selection, on the caller's thread, is safe.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    const char* environmentValue = std::getenv(std::string(ENVIRONMENT_VARIABLE).c_str());
#pragma GCC diagnostic pop
    return select(processorFeatures(), option, environmentValue == nullptr ? std::string_view{} : environmentValue);
}

std::string KernelSelection::describe() const
{
    std::string text(name());
    text.append(" (");
    switch (m_source)
    {
    case KernelSelectionSource::Option:
        text.append("forced by the option kernel=");
        break;
    case KernelSelectionSource::Environment:
        text.append("forced by ").append(ENVIRONMENT_VARIABLE);
        break;
    case KernelSelectionSource::Automatic:
        text.append("automatic");
        break;
    }
    text.append(": ").append(describeInstructions(m_kind)).append(" on ").append(m_processorBrand).append(")");
    return text;
}

bool isKernelSupported(KernelKind kind, const base::CpuFeatures& cpu) noexcept
{
    switch (kind)
    {
    case KernelKind::Reference:
        return true;
    case KernelKind::Avx2:
        return cpu.hasAvx2Family();
    case KernelKind::Avx512:
        return cpu.hasAvx512Family();
    }
    return false;
}

const base::CpuFeatures& processorFeatures()
{
    // Detected on first use (no work at static initialisation) and immutable afterwards.
    static const base::CpuFeatures DETECTED_FEATURES = base::detectCpuFeatures();
    return DETECTED_FEATURES;
}

} // namespace oos::linalg
