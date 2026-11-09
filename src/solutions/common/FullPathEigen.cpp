#include "oos/solutions/common/FullPathEigen.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/ScratchSpaceError.hpp"
#include "oos/base/platform/ScratchFile.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/BucklingPencil.hpp"
#include "oos/linalg/eigen/OutOfCorePencil.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/supernodal/MatrixFrontSource.hpp"
#include "oos/linalg/supernodal/UpdateLayout.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <system_error>

namespace oos::solutions
{
namespace
{

// the multilevel substructuring's options from the solver's (the analysis's default cut-off: SOL 105's on the vibration
// basis higher).
linalg::SubstructuringOptions substructuringOptionsOf(const EigenSolverOptions& options,
                                                      linalg::SpectralTransform transform, bool hasMass,
                                                      const linalg::LanczosOptions& lanczos)
{
    const bool isBuckling = transform == linalg::SpectralTransform::Buckling;
    linalg::SubstructuringOptions substructuring;
    substructuring.cutoffFactor = options.substructuringCutoffFactor > 0.0 ? options.substructuringCutoffFactor
                                  : isBuckling && hasMass ? linalg::SUBSTRUCTURING_BUCKLING_CUTOFF_FACTOR
                                                          : linalg::SUBSTRUCTURING_CUTOFF_FACTOR;
    substructuring.unitDofs = options.substructuringUnitDofs;
    substructuring.lanczos = lanczos;
    substructuring.spool.isOnDisk = !options.scratchFolder.empty();
    substructuring.spool.folder = options.scratchFolder;
    return substructuring;
}

// multilevel substructuring on the given sources (the mass: the vibration basis of a buckling solve, or none).
void solveSubstructuring(const EigenSolverOptions& options, const linalg::SymbolicFactor& symbolic,
                         linalg::FrontSource& stiffness, linalg::FrontSource& rightHand, linalg::FrontSource* mass,
                         linalg::SpectralTransform transform, const linalg::EigenRequest& request,
                         const linalg::LanczosOptions& lanczos, const linalg::DenseKernels& kernels,
                         base::MemoryBudget& budget, FullPathEigen& result)
{
    const bool isBuckling = transform == linalg::SpectralTransform::Buckling;
    const linalg::SubstructuringOptions substructuring =
        substructuringOptionsOf(options, transform, isBuckling && mass != nullptr, lanczos);
    linalg::MultilevelSubstructuring solver(symbolic, stiffness, rightHand, transform, kernels, substructuring,
                                            isBuckling ? mass : nullptr);
    linalg::SubstructuringSolution solved = solver.solve(request, budget);
    result.solution = std::move(solved.solution);
    result.substructuring = std::move(solved.report);
}

// The plan of the factor out of core: the disk for one factor's panels with their delays, beside the margin; in memory
// the factorisation's working storage and the factor's arrays, `matrixBytes` (the shifted matrix of the assembled
// pencil; the element products' work of a pencil from the elements), the Lanczos workspace and as many converged
// vectors as the rest allows (the others spooled). Throws ScratchSpaceError when the disk is short.
struct OutOfCorePlan
{
    OutOfCoreRecord record;
    std::size_t resident = 0;
    linalg::ScratchSpool::Options spool;
};

OutOfCorePlan planOutOfCore(const EigenSolverOptions& options, const linalg::SymbolicFactor& symbolic,
                            std::uint64_t matrixBytes, const linalg::EigenRequest& request,
                            const linalg::LanczosOptions& lanczos, const linalg::DenseKernels& kernels,
                            const base::MemoryBudget& budget)
{
    std::error_code ignored;
    const std::filesystem::path folder = options.scratchFolder.empty()
                                             ? std::filesystem::temp_directory_path(ignored) / "O_o_Solver"
                                             : options.scratchFolder;
    OutOfCorePlan plan;
    OutOfCoreRecord& record = plan.record;
    record.folder = folder.string();
    const auto n = static_cast<std::size_t>(symbolic.dofCount());
    const std::uint64_t panels = static_cast<std::uint64_t>(symbolic.panelValueCount()) * sizeof(double);
    const std::uint64_t perColumn =
        static_cast<std::uint64_t>(linalg::DenseMatrix::alignedLeadingDimension(n)) * sizeof(double);
    const std::size_t threads =
        kernels.options().pool != nullptr ? kernels.options().pool->threadCount() : std::size_t{1};
    const std::uint64_t factorInMemory =
        linalg::SupernodalFactor::estimateMemoryBytes(symbolic, threads, linalg::PivotingMode::Indefinite) - panels +
        matrixBytes;
    const linalg::BlockLanczos solver(lanczos);
    const std::uint64_t workspace = solver.estimateMemoryBytes(n, request, 0);
    std::size_t resident = options.residentLockedColumns;
    if (resident == 0)
    {
        const std::uint64_t available = budget.available();
        const std::uint64_t held = factorInMemory + workspace;
        resident =
            available > held ? static_cast<std::size_t>((available - held) / std::max<std::uint64_t>(perColumn, 1)) : 0;
        resident = std::max(resident, lanczos.blockSize);
    }
    const std::size_t modes = request.modeCount > 0 ? std::min(request.modeCount, n) : n;
    const std::size_t storeColumns = modes + solver.shapeFor(n, request).basisColumns;
    const std::uint64_t spooledVectors =
        storeColumns > resident ? static_cast<std::uint64_t>(storeColumns - resident) * perColumn : 0;
    record.plannedDiskBytes =
        static_cast<std::uint64_t>(DELAY_GROWTH_ALLOWANCE * static_cast<double>(panels)) + spooledVectors;
    record.freeDiskBytes = base::ScratchFile::freeBytes(folder);
    if (options.scratchLimitBytes > 0)
        record.freeDiskBytes = std::min(record.freeDiskBytes, options.scratchLimitBytes);
    record.residentLockedColumns = resident;
    if (record.freeDiskBytes < record.plannedDiskBytes + SCRATCH_MARGIN_BYTES)
        throw base::ScratchSpaceError(
            base::concatenate(
                {"the out-of-core Lanczos solver needs ", base::formatBytes(record.plannedDiskBytes),
                 " of scratch disk (one factor's panels with their delays, and the converged vectors "
                 "beyond the memory) beside a margin of ",
                 base::formatBytes(SCRATCH_MARGIN_BYTES), ", found ", base::formatBytes(record.freeDiskBytes),
                 options.scratchLimitBytes > 0 ? " free or allowed (the options' scratch limit) in '" : " free in '",
                 folder.string(),
                 "'; give a scratch folder on a larger disk (scratch_dir=, or the OOS_SCRATCH "
                 "environment variable), or ask fewer modes"}),
            record.plannedDiskBytes, record.freeDiskBytes);
    plan.resident = resident;
    plan.spool.isOnDisk = true;
    plan.spool.folder = folder;
    return plan;
}

// The block Lanczos solver on an out-of-core pencil, its traffic into the plan's record.
void solveOutOfCore(OutOfCorePlan plan, linalg::OutOfCorePencil& pencil, const linalg::EigenRequest& request,
                    const linalg::LanczosOptions& lanczos, const linalg::DenseKernels& kernels,
                    base::MemoryBudget& budget, FullPathEigen& result)
{
    linalg::LanczosOptions outOfCore = lanczos;
    outOfCore.residentLockedColumns = plan.resident;
    outOfCore.lockedSpool = plan.spool;
    result.solution = linalg::BlockLanczos(outOfCore).solve(pencil, request, kernels, budget);
    const linalg::OutOfCoreTraffic traffic = pencil.traffic();
    OutOfCoreRecord& record = plan.record;
    record.largestFactorBytes = pencil.largestSpoolBytes();
    record.writtenBytes = traffic.writtenBytes;
    record.readBytes = traffic.readBytes;
    record.readRequests = traffic.readRequests;
    record.readSeconds = traffic.readSeconds;
    record.factorisations = pencil.factorisationCount();
    record.sampledPeakBytes = pencil.sampledPeakBytes();
    record.sampledConsumers = pencil.sampledConsumers();
    result.outOfCore = std::move(record);
}

} // namespace

linalg::AmalgamationThresholds orderingThresholdsOf(const EigenSolverOptions& options) noexcept
{
    linalg::AmalgamationThresholds thresholds;
    if (options.solver == EigenSolver::Substructuring)
        thresholds.maxColumns = linalg::UpdateLayout::STREAMED_BLOCK_COLUMNS;
    return thresholds;
}

FullPathEigen solveFullPathEigen(const EigenSolverOptions& options, const linalg::SymbolicFactor& symbolic,
                                 const linalg::SymmetricCsc& stiffness, const linalg::SymmetricCsc& rightHand,
                                 linalg::SpectralTransform transform, const linalg::EigenRequest& request,
                                 const linalg::LanczosOptions& lanczos, const linalg::FactorOptions& factorOptions,
                                 const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                 const linalg::SymmetricCsc* mass)
{
    const bool isBuckling = transform == linalg::SpectralTransform::Buckling;
    FullPathEigen result;
    if (options.solver == EigenSolver::Substructuring)
    {
        linalg::MatrixFrontSource stiffnessSource(stiffness, symbolic);
        // B = M, or B = −K_d.
        linalg::MatrixFrontSource rightHandSource(rightHand, symbolic, isBuckling ? -1.0 : 1.0);
        std::optional<linalg::MatrixFrontSource> massSource;
        if (isBuckling && mass != nullptr)
            massSource.emplace(*mass, symbolic);
        solveSubstructuring(options, symbolic, stiffnessSource, rightHandSource, massSource ? &*massSource : nullptr,
                            transform, request, lanczos, kernels, budget, result);
        return result;
    }
    if (options.solver == EigenSolver::OutOfCore)
    {
        // The shifted matrix K − σB beside the factor (its rows and values, its column offsets).
        const std::uint64_t shifted =
            static_cast<std::uint64_t>(stiffness.nonZeroCount()) * (sizeof(std::int32_t) + sizeof(double)) +
            (static_cast<std::uint64_t>(symbolic.dofCount()) + 1) * sizeof(std::int64_t);
        OutOfCorePlan plan = planOutOfCore(options, symbolic, shifted, request, lanczos, kernels, budget);
        linalg::OutOfCorePencil pencil(symbolic, stiffness, rightHand, transform, kernels, factorOptions, plan.spool,
                                       budget);
        solveOutOfCore(std::move(plan), pencil, request, lanczos, kernels, budget, result);
        return result;
    }
    if (isBuckling)
    {
        linalg::BucklingPencil pencil(symbolic, stiffness, rightHand, kernels, factorOptions, budget);
        result.solution = linalg::BlockLanczos(lanczos).solve(pencil, request, kernels, budget);
    }
    else
    {
        linalg::SparsePencil pencil(symbolic, stiffness, rightHand, kernels, factorOptions, budget);
        result.solution = linalg::BlockLanczos(lanczos).solve(pencil, request, kernels, budget);
    }
    return result;
}

FullPathEigen solveFullPathEigen(const EigenSolverOptions& options, const linalg::SymbolicFactor& symbolic,
                                 linalg::PencilSource& pencil, std::uint64_t productBytes,
                                 linalg::SpectralTransform transform, const linalg::EigenRequest& request,
                                 const linalg::LanczosOptions& lanczos, const linalg::FactorOptions& factorOptions,
                                 const linalg::DenseKernels& kernels, base::MemoryBudget& budget)
{
    FullPathEigen result;
    if (options.solver == EigenSolver::Substructuring)
    {
        solveSubstructuring(options, symbolic, pencil.stiffnessSource(), pencil.rightHandSource(), pencil.massSource(),
                            transform, request, lanczos, kernels, budget, result);
        return result;
    }
    if (options.solver != EigenSolver::OutOfCore)
        throw std::invalid_argument(
            "solveFullPathEigen: a pencil from the elements is solved by multilevel substructuring or with the factor "
            "out of core; the "
            "block "
            "Lanczos solver in memory takes the assembled matrices (give EigenSolver::Substructuring or "
            "EigenSolver::OutOfCore)");
    OutOfCorePlan plan = planOutOfCore(options, symbolic, productBytes, request, lanczos, kernels, budget);
    linalg::OutOfCorePencil outOfCore(symbolic, pencil, transform, kernels, factorOptions, plan.spool, budget);
    solveOutOfCore(std::move(plan), outOfCore, request, lanczos, kernels, budget, result);
    return result;
}

} // namespace oos::solutions
