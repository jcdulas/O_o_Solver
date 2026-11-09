#pragma once

#include "oos/base/platform/CpuFeatures.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"

#include <string>
#include <string_view>

namespace oos::linalg
{

// Where the choice of a kernel family came from, in order of precedence.
enum class KernelSelectionSource
{
    Option,      // the option `kernel=` of the caller
    Environment, // the environment variable OOS_KERNEL
    Automatic    // the CPU detection
};

// The dense kernel family an analysis uses, chosen once and named in every report.
//
// The choice, in order: the option `kernel=` when it names a family; otherwise OOS_KERNEL when it names one;
// otherwise the automatic choice: AVX2 when the processor and the OS support AVX2 and FMA, the reference kernel
// otherwise. AVX-512 is never chosen automatically while it is not validated; it can be forced.
// "auto" or an empty value names no family and passes the choice on to the next source. A forced family the processor
// lacks is refused (SolverError naming the family, where it was requested, the processor and the families it supports),
// never replaced silently.
//
// An immutable value; thread-safe.
class KernelSelection
{
public:
    static constexpr std::string_view ENVIRONMENT_VARIABLE = "OOS_KERNEL";
    static constexpr std::string_view AUTOMATIC_NAME = "auto";

    // The choice for the processor `cpu`, from the option and the value of the environment variable (each empty when
    // not given). Throws SolverError for an unknown name or a family the processor lacks.
    [[nodiscard]] static KernelSelection select(const base::CpuFeatures& cpu, std::string_view option,
                                                std::string_view environmentValue);

    // The choice for the processor this process runs on, reading OOS_KERNEL now.
    [[nodiscard]] static KernelSelection selectForThisProcess(std::string_view option = {});

    [[nodiscard]] KernelKind kind() const noexcept { return m_kind; }
    [[nodiscard]] std::string_view name() const noexcept { return kernelName(m_kind); }
    [[nodiscard]] KernelSelectionSource source() const noexcept { return m_source; }
    // One line for the reports: "avx2 (automatic: AVX2 and FMA on 12th Gen Intel(R) Core(TM) i5-12400F)".
    [[nodiscard]] std::string describe() const;

private:
    KernelSelection(KernelKind kind, KernelSelectionSource source, std::string processorBrand);

    KernelKind m_kind;
    KernelSelectionSource m_source;
    std::string m_processorBrand;
};

// True when the processor (and its OS) can run the kernel family `kind`.
[[nodiscard]] bool isKernelSupported(KernelKind kind, const base::CpuFeatures& cpu) noexcept;

// The features of the processor this process runs on, detected on first use and kept (they cannot change).
[[nodiscard]] const base::CpuFeatures& processorFeatures();

} // namespace oos::linalg
