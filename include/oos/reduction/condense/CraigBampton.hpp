#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/reduction/condense/ModalReduction.hpp"
#include "oos/reduction/condense/StaticCondensation.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstddef>
#include <cstdint>

namespace oos::reduction
{

// The fixed-interface modes a unit keeps (a count N or a frequency cut-off, chosen per unit): the lowest
// `count` modes, or with count 0 every mode whose eigenvalue λ = ω² is at most cutoffEigenvalue (CB_CUTOFF_FACTOR
// times the highest frequency requested, squared: solutions::CondensedStatics). A unit never keeps more modes than
// its interior has DOFs with mass (the other roots are infinite).
struct ModeRequest
{
    std::int32_t count = 0;
    double cutoffEigenvalue = 0.0;

    [[nodiscard]] bool isCount() const noexcept { return count > 0; }
};

// The fixed-interface modes of one unit: Φ on its interior equations (the leading interiorDofCount() equations of its
// UnitSystem), M_oo-orthonormal, and their eigenvalues, increasing. The recovery data of the condensation cache
// (CachedRecoveryData).
struct FixedInterfaceModes
{
    linalg::DenseMatrix modes;       // interior equations × N
    linalg::DenseMatrix eigenvalues; // N × 1
    std::size_t factorisationCount = 0;
    // False when the Lanczos solver stopped at its shift limit before the request was met (the modes found are used:
    // the reduction is still a Rayleigh-Ritz projection, only coarser).
    bool isComplete = true;
};

// The Craig-Bampton reduction of a unit (the matrices: ModalReduction): its
// fixed-interface modes by the block Lanczos solver on its interior, and the projections of its matrices onto the
// constraint modes Ψ̃ = [−K_oo⁻¹·K_oa; I] and the modes Φ̃ = [Φ; 0], from its static condensation (whose interior
// factor must be in memory: StaticCondensation::recomputeInteriorFactor first when it was released).
//
// The constraint modes are never formed whole (no recovery matrix, 8·n_o·n_a bytes): a block of
// CONSTRAINT_MODE_BLOCK of them at a time is made by the recovery of the interior (StaticCondensation::recover with
// the boundary values the columns of I), multiplied by the matrix, and condensed by the forward sweep
// (StaticCondensation::condenseLoads: Ψ̃ᵀ·y = y_a − K_ao·K_oo⁻¹·y_o): one interior solve and one sweep per boundary
// DOF, the cost of recovering the unit for every boundary DOF once.
//
// Determinism: the Lanczos solver (seeded starts, deterministic factorisations and products), the recoveries, the
// sparse products (SymmetricCsc::multiply, the columns the pieces of the parallel work) and the dense products
// (DenseKernels) give the same bits for any number of threads, for a kernel family: modes evicted from the cache and
// computed again have the bits of the first computation.
//
// Thread-safety: stateless static functions. Ownership: none.
class CraigBampton
{
public:
    // The boundary DOFs whose constraint modes are formed at once (the columns of the recoveries and products): a
    // block the solve kernels take well (half a tile of FactorSolve, 128 columns, at full vector width), 8·n·64 bytes
    // of work per array.
    static constexpr std::size_t CONSTRAINT_MODE_BLOCK = 64;

    // The fixed-interface modes of the unit: the eigenpairs of (K_oo, M_oo), the leading interior blocks of its
    // matrix() and mass() (UnitSystem::interiorSymbolic, linalg::SparsePencil, linalg::BlockLanczos from the lowest
    // root: K_oo is positive definite, the unit being held at its boundary). The count is limited to the interior DOFs
    // with mass; a unit without interior mass has no mode. Throws std::logic_error when the unit has no mass
    // (UnitModel::keepsMass), and as BlockLanczos::solve (MemoryBudgetError, OperationCancelledError, SolverError).
    [[nodiscard]] static FixedInterfaceModes computeModes(const UnitSystem& system, const ModeRequest& request,
                                                          const linalg::DenseKernels& kernels,
                                                          const linalg::FactorOptions& options,
                                                          base::MemoryBudget& budget);

    // Ψ̃ᵀ·A·Ψ̃ (boundary DOFs square, exactly symmetric: the mean of its two triangles, in a fixed order) for a matrix
    // A on the unit's equations (its mass: M_aa, the reduced boundary mass; its differential stiffness). Throws
    // std::invalid_argument on a size mismatch, std::logic_error when the interior factor was released,
    // MemoryBudgetError, OperationCancelledError.
    [[nodiscard]] static linalg::DenseMatrix projectOnBoundary(const StaticCondensation& condensation,
                                                               const linalg::SymmetricCsc& matrix,
                                                               const linalg::DenseKernels& kernels,
                                                               base::MemoryBudget& budget,
                                                               const base::CancellationToken* cancellation = nullptr);

    // The q part of the reduction (ModalReduction): K_qq = Φᵀ·K_oo·Φ, M_aq = Ψ̃ᵀ·M·Φ̃, M_qq = Φᵀ·M_oo·Φ from the unit's
    // stiffness matrix() and mass() and its modes (interior equations × N, their eigenvalues N × 1). Throws as
    // projectOnBoundary.
    [[nodiscard]] static ModalReduction reduceModes(const StaticCondensation& condensation,
                                                    const linalg::DenseMatrix& modes,
                                                    const linalg::DenseMatrix& eigenvalues,
                                                    const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                                    const base::CancellationToken* cancellation = nullptr);

    // T_CBᵀ·A·T_CB on (a, q) ((a + N) square, exactly symmetric) for a matrix A on the unit's equations, T_CB = [Ψ̃
    // Φ̃], Φ its modes (interior equations × N; no column: Ψ̃ alone): the reduced differential stiffness of SOL 105
    // (computed at each analysis from the preload, never cached). Throws as projectOnBoundary.
    [[nodiscard]] static linalg::DenseMatrix project(const StaticCondensation& condensation,
                                                     const linalg::SymmetricCsc& matrix,
                                                     const linalg::DenseMatrix& modes,
                                                     const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                                     const base::CancellationToken* cancellation = nullptr);

    // The unit's equations of reduced vectors: u_o = Ψ·u_a + Φ·q (the recovery of the interior, then the modes added),
    // u_a copied. boundaryValues: boundary DOFs × c, modalValues: N × c (no row without modes), equationValues: the
    // unit's equations × c, overwritten. Throws as projectOnBoundary.
    static void recover(const StaticCondensation& condensation, const linalg::DenseMatrix& modes,
                        linalg::ConstDenseMatrixView boundaryValues, linalg::ConstDenseMatrixView modalValues,
                        linalg::DenseMatrixView equationValues, const linalg::DenseKernels& kernels,
                        const base::CancellationToken* cancellation = nullptr);
};

// The cut-off of the fixed-interface modes kept by a unit under condensation="auto" and the frequency cut-off of a
// plan (modes and buckling under condensation="auto"): each unit keeps
// its modes up to CB_CUTOFF_FACTOR times the highest frequency requested (the V2 of LanczosEigenMethod; with ND, the
// ND-th frequency of the statically condensed model, an upper bound of the true one), i.e. the eigenvalues up to
// CB_CUTOFF_FACTOR² times the highest eigenvalue requested. Calibrated on seven synthetic assemblies of 9 to 100 plates
// joined by RigidInterpolation-Bushing-RigidInterpolation fasteners (2 900 to 11 800 DOFs, among them two assemblies
// of 100 plates): the largest relative error of the 20 lowest frequencies is 1 to 3.5 % at 2 (the usual engineering
// rule), 0.07 to 0.9 % at 4, 0.01 to 0.10 % at 8, 0.002 to 0.095 % at 16 and 0.0004 to 0.003 % at 20, the modes kept
// growing about as the factor. The error a dropped mode leaves is about ½·p·(f/f_j)² on a frequency f (p the fraction
// of the unit's mass the mode carries, f_j its frequency): the first fixed-interface mode of a plate held at a few
// points carries about 0.6 of its mass (measured on the 100 plates: their mode at 17.5 times the 20th frequency left
// 0.095 %), so 20 is the smallest round factor that keeps such a mode within 0.1 %, every assembly measured with a
// margin of 30. A caller who wants fewer modes gives a count per unit (PlanRequest::modeCount).
inline constexpr double CB_CUTOFF_FACTOR = 20.0;

} // namespace oos::reduction
