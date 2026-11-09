#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/supernodal/ScratchSpool.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::linalg
{

// The interior DOFs of a unit of multilevel substructuring at most: the maximal subtrees of the elimination tree below
// this are the units, each solved by the block Lanczos solver on its own sparse pencil; above them every supernode is a
// separator node with dense modes. Larger units make fewer separator levels (the coupling blocks carried up shrink) but
// longer unit eigensolutions; set by measurement on a series of barrel models of growing size.
inline constexpr std::int64_t SUBSTRUCTURING_UNIT_DOFS = 20000;
// A unit of at most this many interior DOFs solves its modes densely (SmallEigenSolver, about 9n³ flops: 0.15 GFLOP
// at 256) rather than by the block Lanczos solver, whose factorisations and runs cost more on a small pencil.
inline constexpr std::int64_t SUBSTRUCTURING_DENSE_UNIT_DOFS = 256;
// The cut-off factor in frequency (λ_c = f²·λ_ref), set from the errors on a series of barrel models of growing
// size: at 20 the highest root of 5 or 20 modes within 1.4e-4 to 3.4e-3 of the full path's
// on six rungs (the Craig-Bampton path's class at its own CB_CUTOFF_FACTOR, 20), for 5 to 15 % more time than at 10
// (within 5e-4 to 1.4e-2).
inline constexpr double SUBSTRUCTURING_CUTOFF_FACTOR = 20.0;
// The vibration root of the first pass that sets a buckling solve's cut-off on the vibration basis:
// the 10th, the Craig-Bampton path's rule for SOL 105 (CondensedStatics::BUCKLING_REFERENCE_MODE_COUNT),
// buckling asking for no frequency.
inline constexpr std::size_t SUBSTRUCTURING_BUCKLING_REFERENCE_MODES = 10;
// SOL 105 on the vibration basis: the cut-off factor of the units, and the separators' at
// this ratio of it (f_s = 2f). Set from the barrel models: the units' and the separators' truncations both bound the
// load factors' error (f = 20 alone 4.8e-3; the separators at 40, 2.5e-3; the units at 40, 3.5e-3; both at
// 40, 1.1e-3), the separators' modes the cheaper (dense, at most 256 columns); at f = 30, f_s = 60 the 5 lowest
// positive load factors are within 1.0e-3 to 1.6e-3 of the full path's on those models, the
// Craig-Bampton basis's class (0.06 to 0.2 %).
inline constexpr double SUBSTRUCTURING_BUCKLING_CUTOFF_FACTOR = 30.0;
inline constexpr double SUBSTRUCTURING_BUCKLING_SEPARATOR_RATIO = 2.0;

// The options of a multilevel substructuring solve.
struct SubstructuringOptions
{
    double cutoffFactor = SUBSTRUCTURING_CUTOFF_FACTOR;
    // The separators' own cut-off factor (0: cutoffFactor; on the vibration basis of a buckling solve,
    // SUBSTRUCTURING_BUCKLING_SEPARATOR_RATIO times it): their dense modes below (f_s/f)² times the cut-off.
    double separatorCutoffFactor = 0.0;
    std::int64_t unitDofs = SUBSTRUCTURING_UNIT_DOFS;
    // The units of at most this many DOFs solve their modes densely (the tests compare both ways on one unit).
    std::int64_t denseUnitDofs = SUBSTRUCTURING_DENSE_UNIT_DOFS;
    // Every node keeps every one of its modes (the cut-off ignored): multilevel substructuring is then exact (the
    // tests' reference).
    bool keepsEveryMode = false;
    // The block Lanczos solver of the units and of the reduced pencil (its control: cancellation and progress).
    LanczosOptions lanczos;
    // Where the separators' panels and every node's modes go between the two passes.
    ScratchSpool::Options spool;
};

// the multilevel substructuring's memory plan from the symbolic factor before any work: its fixed part (the
// upward pass's fronts and update matrices of the matrices it carries, along the order it runs; the units in flight,
// each its gathered entries, factor and Lanczos workspace bounded by its supernodes' panels; the scratch spool's
// writers and records in memory; the recovered vectors),
// its parts per mode kept (the coupling blocks, the spool's modes in memory) and per mode squared (the reduced pencil:
// its right-hand matrix, which the dense pencil takes, and its factor; the Z blocks while it is assembled), and the
// most modes the memory available holds beside them (the allowance). A first pass by count keeps count × nodes modes:
// it is refused before any work when they exceed the allowance. Bytes.
struct SubstructuringPlan
{
    std::uint64_t availableBytes = 0;
    std::uint64_t stackBytes = 0;
    std::uint64_t unitBytes = 0;
    std::uint64_t vectorBytes = 0;
    std::uint64_t spoolBytes = 0; // its writers' blocks, and in memory the separators' panels
    std::uint64_t fixedBytes = 0;
    std::uint64_t perModeBytes = 0;
    std::uint64_t perModeSquaredBytes = 0;
    std::size_t modeAllowance = 0;

    // The plan's bytes with m modes kept.
    [[nodiscard]] std::uint64_t bytesWith(std::size_t modes) const noexcept
    {
        const auto m = static_cast<std::uint64_t>(modes);
        return fixedBytes + perModeBytes * m + perModeSquaredBytes * m * m;
    }
};

// What a multilevel substructuring solve did (the report's `substructuring` entry).
struct SubstructuringReport
{
    // The memory plan made before the work (its allowance checked after each upward pass).
    SubstructuringPlan plan;
    // The nodes' modes: "vibration" (of (K, M): SOL 103, and SOL 105 with K_d projected) or
    // "buckling" (SOL 105's own buckling modes of (K, −K_d), a model without mass).
    std::string basis;
    // "V2" (the request's upper bound), "first pass" (ND: the ND-th root of the first pass, an upper bound of the
    // true one) or "vibration first pass" (SOL 105 on the vibration basis: its
    // SUBSTRUCTURING_BUCKLING_REFERENCE_MODES-th vibration root), and the eigenvalue λ_ref and cut-off λ_c = f²·λ_ref;
    // the passes made (1 or 2).
    std::string reference;
    double referenceEigenvalue = 0.0;
    double cutoffEigenvalue = 0.0;
    double cutoffFactor = 0.0;
    std::size_t passes = 0;
    std::size_t unitCount = 0;
    std::size_t separatorCount = 0;
    std::size_t unitModes = 0;
    std::size_t separatorModes = 0;
    std::size_t reducedOrder = 0; // the modes kept in all: the order of the reduced pencil
    std::int64_t largestUnitDofs = 0;
    std::int64_t largestBoundary = 0;
    std::int64_t largestCouplingValues = 0; // the largest coupling block carried up (rows × modes)
    // Per root found: λ/λ_c, the cut-off ratio its error grows with; empty on the vibration basis
    // of a buckling solve (the roots load factors, the cut-off an ω²).
    std::vector<double> cutoffRatios;
    // Seconds: the upward passes (fronts, units' modes, separators' modes), the units' eigensolutions among them, the
    // reduced pencil's solve, the recovery.
    double upwardSeconds = 0.0;
    double unitSeconds = 0.0;
    double reducedSeconds = 0.0;
    double recoverySeconds = 0.0;
    std::uint64_t spoolBytes = 0; // the spool's records (on disk or in memory)
    bool isSpoolOnDisk = false;
    std::size_t reducedFactorisations = 0;
    // The budget's largest consumers at the most used of its samples (the end of each upward pass, of the reduced
    // solve and of the recovery), with that use: what the memory of a run is made of (the extrapolations of step
    // 10.42).
    std::uint64_t sampledPeakBytes = 0;
    std::vector<base::MemoryConsumer> sampledConsumers;
};

struct SubstructuringSolution
{
    // The roots by the request's order (EigenSolution: increasing for vibration, by magnitude for buckling) and their
    // vectors on the pencil's equations (B-orthonormal for vibration, K-orthonormal for buckling); the shifts are the
    // reduced pencil's.
    EigenSolution solution;
    SubstructuringReport report;
};

// The eigenpairs of K φ = λ B φ by automated multilevel substructuring on the elimination tree (
// Bennighof & Lehoucq, SIAM J. Sci. Comput. 25 (2004); Gao, Li, Yang & Bai, SIAM J. Matrix Anal.
// Appl. 30 (2008)): B = M for vibration (SOL 103), B = −K_d for buckling (SOL 105), both given front by front.
//
// Method. The supernodes of the symbolic factor (in postorder) form the nodes: the units (the maximal subtrees of at
// most unitDofs interior DOFs) and above them every supernode alone (a separator node). One upward pass makes each
// supernode's fronts of K and B (the sources' terms, then the children's update matrices) and its coupling block C
// (the modes kept below it, against its front's equations), eliminates its columns from F_K (positive definite LDLᵀ:
// U_K, and Ψ = −A_cc⁻¹·A_cr = −L_cc⁻ᵀ·L_rcᵀ), applies the congruence to F_B (U_B = B_rr + Ψᵀ·B_cr + B_rc·Ψ +
// Ψᵀ·B_cc·Ψ) and carries C up (C_r + Ψᵀ·C_c); a unit's root solves the unit's fixed-interface modes on its gathered
// pencil (BlockLanczos below λ_c) and couples them with its boundary (B_bi·Φ − K_bi·Φ·Λ⁻¹); a separator solves its own
// dense pencil (A_cc, B_cc) below λ_c (SmallEigenSolver), adds its block Z = C_cᵀ·Φ_s to the reduced pencil and its
// modes' coupling (B_rc + Ψᵀ·B_cc)·Φ_s to C. The reduced pencil (K̄ = diag(Λ), B̄ = I + Z + Zᵀ, vibration; K̄ = I,
// B̄ = diag(1/λ) + Z + Zᵀ, buckling) is solved by the block Lanczos solver on a DensePencil, then the vectors recovered
// top-down (x_c = Φ_s·η_s − L_cc⁻ᵀ·L_rcᵀ·x_r at a separator, its panel and modes from the spool; at a unit
// x_i = Φ_u·η_u − K_ii⁻¹·K_ib·x_b, the unit factorised again).
//
// The cut-off: λ_c = f²·λ_ref, λ_ref the request's upper bound, or with a count alone the count-th root of a first
// pass whose nodes keep their count lowest modes (an upper bound of the true root: Rayleigh-Ritz on that pass's
// subspace).
//
// Buckling on the vibration basis (given a mass source): the nodes' modes are the fixed-interface
// vibration modes of (K, M) below λ_c (an ω²: f² times the SUBSTRUCTURING_BUCKLING_REFERENCE_MODES-th vibration root of
// a first pass), scaled K-orthonormal (φ/√λ); B = −K_d is carried as a third matrix (its fronts, update matrices by the
// same congruence, and the coupling blocks) and projected: K̄ = I, B̄ = Tᵀ·B·T dense (each node's own block Φᵀ·B̃_cc·Φ
// beside the Z blocks), a unit's coupling C_b = B_bi·Φ − K_bi·K_ii⁻¹·(B_ii·Φ) by one solve with its stiffness factor.
//
// Determinism: one serial postorder, every product a DenseKernels product, every unit's Lanczos seeded: the same bits
// for any number of threads for a given kernel family.
//
// Memory: the stack of the update matrices of K and B (and M on the vibration basis of a buckling solve) and of the
// coupling blocks (Liu's postorder), one unit's pencil
// and Lanczos workspace at a time, the reduced pencil, the vectors; the separators' panels and the nodes' modes in the
// spool (in memory or on disk). Every buffer in the budget (MemoryBudgetError).
//
// Thread-safety: one solve at a time. Ownership: references the symbolic factor, the sources and the kernels, which
// must outlive it.
class MultilevelSubstructuring
{
public:
    // `mass`: for a buckling solve (B = −K_d in `rightHand`), the mass M whose vibration modes make the basis;
    // null: the nodes' own buckling modes. Ignored for vibration (B is M).
    MultilevelSubstructuring(const SymbolicFactor& symbolic, FrontSource& stiffness, FrontSource& rightHand,
                             SpectralTransform transform, const DenseKernels& kernels, SubstructuringOptions options,
                             FrontSource* mass = nullptr);

    // Throws std::invalid_argument for a request without count nor upper bound (vibration) or bound (buckling), or a
    // cut-off factor below 1; SolverError when a front below a root is not positive definite (a mechanism held by
    // nothing); MemoryBudgetError before any work when the plan's fixed part does not fit or its mode allowance is
    // below the count asked, after an upward pass that keeps more modes than the allowance, and from the budget;
    // FileError, OperationCancelledError.
    [[nodiscard]] SubstructuringSolution solve(const EigenRequest& request, base::MemoryBudget& budget);

    // The memory plan of a solve that carries `matrices` matrices up the tree (2, or 3 on the vibration basis of a
    // buckling solve) on `threads` threads, recovering `vectors` vectors, with `availableBytes` of memory
    // (SubstructuringPlan).
    [[nodiscard]] static SubstructuringPlan plan(const SymbolicFactor& symbolic, const SubstructuringOptions& options,
                                                 std::size_t matrices, std::size_t threads, std::size_t vectors,
                                                 std::uint64_t availableBytes);

    // The units the symbolic factor gives for a bound: per supernode the unit it belongs to (−1: a separator).
    [[nodiscard]] static std::vector<std::int32_t> unitsOf(const SymbolicFactor& symbolic, std::int64_t unitDofs);

private:
    const SymbolicFactor& m_symbolic;
    FrontSource& m_stiffness;
    FrontSource& m_rightHand;
    SpectralTransform m_transform;
    const DenseKernels& m_kernels;
    SubstructuringOptions m_options;
    FrontSource* m_mass;
};

} // namespace oos::linalg
