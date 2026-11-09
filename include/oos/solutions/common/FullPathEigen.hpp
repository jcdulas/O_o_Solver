#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/linalg/eigen/MultilevelSubstructuring.hpp"
#include "oos/linalg/eigen/PencilSource.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace oos::solutions
{

// The eigenvalue solver of the full path of SOL 103 and SOL 105: the block Lanczos solver on the pencil's factor in
// memory (the full path), multilevel substructuring on the elimination tree (approximate to its cut-off), or the block
// Lanczos solver with its factor out of core.
enum class EigenSolver : std::uint8_t
{
    Lanczos,
    Substructuring,
    OutOfCore
};

// Where multilevel substructuring and the factor out of core take their pencil: from the elements, front by front and
// product by product, no matrix of the analysis set assembled (the default); or the assembled matrices (the reference,
// and the outputs that need Kgg or Mgg: SPCFORCES, MPCFORCES, MEFFMASS, PARAM GRDPNT).
enum class EigenPencil : std::uint8_t
{
    Elements,
    Assembled
};

struct EigenSolverOptions
{
    EigenSolver solver = EigenSolver::Lanczos;
    EigenPencil pencil = EigenPencil::Elements;
    // the multilevel substructuring's cut-off factor; 0: the analysis's default (SUBSTRUCTURING_CUTOFF_FACTOR, or
    // SUBSTRUCTURING_BUCKLING_CUTOFF_FACTOR for SOL 105 on the vibration basis).
    double substructuringCutoffFactor = 0.0;
    std::int64_t substructuringUnitDofs = linalg::SUBSTRUCTURING_UNIT_DOFS;
    // The folder of the scratch files (empty: multilevel substructuring keeps its spool in memory; the factor out of
    // core goes to the system's temporary folder).
    std::filesystem::path scratchFolder;
    // The factor out of core: the converged vectors held in memory at most (0: as many as the memory left allows).
    std::size_t residentLockedColumns = 0;
    // The factor out of core: the scratch disk it may use at most, beside the folder's free space (0: that space).
    std::uint64_t scratchLimitBytes = 0;
};

// The plan and the traffic of the factor out of core: the folder, the disk the plan asked and found, the converged
// vectors kept in memory, then what the solve wrote and read.
struct OutOfCoreRecord
{
    std::string folder;
    std::uint64_t plannedDiskBytes = 0;
    std::uint64_t freeDiskBytes = 0;
    std::size_t residentLockedColumns = 0;
    std::uint64_t largestFactorBytes = 0; // one factor's panels on the disk
    std::uint64_t writtenBytes = 0;
    std::uint64_t readBytes = 0;
    std::uint64_t readRequests = 0;
    double readSeconds = 0.0;
    std::size_t factorisations = 0;
    // The budget's largest consumers at the most used of the pencil's samples (OutOfCorePencil::sampledConsumers).
    std::uint64_t sampledPeakBytes = 0;
    std::vector<base::MemoryConsumer> sampledConsumers;
};

// What the full path's eigensolution gave, with the multilevel substructuring's report on that path.
struct FullPathEigen
{
    linalg::EigenSolution solution;
    std::optional<linalg::SubstructuringReport> substructuring;
    std::optional<OutOfCoreRecord> outOfCore;
};

// The factor of K − σB with its delays at most this many times the symbolic panels (measured on a barrel model: 1.08 GB
// of delays beside 4.01 GB at 1/32; the estimate for the full section 1.27): the disk the plan asks for one factor.
inline constexpr double DELAY_GROWTH_ALLOWANCE = 1.3;
// The free space the plan leaves on the scratch disk beside what it asks.
inline constexpr std::uint64_t SCRATCH_MARGIN_BYTES = std::uint64_t{1} << 30;

// The supernodes the ordering of the analysis set makes for a solver: at most 256 columns for multilevel substructuring
// (its separators' dense pencils, linalg::MultilevelSubstructuring), the defaults otherwise.
[[nodiscard]] linalg::AmalgamationThresholds orderingThresholdsOf(const EigenSolverOptions& options) noexcept;

// The eigenpairs of K·φ = λ·B·φ on the analysis set by the options' solver: `rightHand` is M (vibration) or K_d
// (buckling, B = −K_d), both in the order of `symbolic`. `mass`: for buckling on multilevel substructuring, the mass
// whose vibration modes make its basis; null: the nodes' own buckling modes (a model without mass). Throws what the
// solvers throw.
[[nodiscard]] FullPathEigen solveFullPathEigen(
    const EigenSolverOptions& options, const linalg::SymbolicFactor& symbolic, const linalg::SymmetricCsc& stiffness,
    const linalg::SymmetricCsc& rightHand, linalg::SpectralTransform transform, const linalg::EigenRequest& request,
    const linalg::LanczosOptions& lanczos, const linalg::FactorOptions& factorOptions,
    const linalg::DenseKernels& kernels, base::MemoryBudget& budget, const linalg::SymmetricCsc* mass = nullptr);

// The same on a pencil given by its sources and products, no matrix of the analysis set assembled (the pencils from the
// elements): multilevel substructuring on its K, B and (buckling) mass sources, or the block Lanczos solver with the
// factor out of core of its K − σB; `productBytes` the products' work memory beside the factor (the out-of-core
// plan's). Throws std::invalid_argument for the solver in memory, and what the solvers throw.
[[nodiscard]] FullPathEigen solveFullPathEigen(const EigenSolverOptions& options,
                                               const linalg::SymbolicFactor& symbolic, linalg::PencilSource& pencil,
                                               std::uint64_t productBytes, linalg::SpectralTransform transform,
                                               const linalg::EigenRequest& request,
                                               const linalg::LanczosOptions& lanczos,
                                               const linalg::FactorOptions& factorOptions,
                                               const linalg::DenseKernels& kernels, base::MemoryBudget& budget);

} // namespace oos::solutions
