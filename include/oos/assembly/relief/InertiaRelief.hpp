#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::assembly
{

// The most support DOFs inertia relief takes: a body has six rigid-body motions (the r set of inertia relief must be
// statically determinate, at most six DOFs).
inline constexpr std::size_t MAX_SUPPORT_DOFS = 6;

// A body is free apart from its supports when the rigid motion of a unit motion of each support makes no reaction:
// the largest |K·D| of D = Φ·C over every DOF, against the largest stiffness of the model on the
// full path and the full path streamed (the round-off of K·Φ scales with the stiffest element: fasteners 1e8 times the
// skins), against the largest |K·e| of a support DOF on the condensed path, is at most this. For a free
// body K·Φ is the round-off of the element sums and of the condensed units' K* (4e-15 to 1e-11 of a support's stiffness
// on the test models and on large barrel models, both paths), not amplified by the condition number of the
// structure held at its supports, which the solved support motions −K_ll⁻¹·K_lr are (their misfit to the rigid
// motions 1.6e-4 on a free barrel of four frames, 1.3 M DOFs, on the full path, 3.6e-5 on the condensed
// path); a grounded spring of 1e3 N/m on the test plates gives about 1e-6, an SPC or a stiff grounded element 1e-3
// to 1. The full path takes the rigid motions below it, and solves them above; KeptRelief (the sensitivities) and the
// condensed path's relief refuse above it.
inline constexpr double RIGID_SUPPORT_TOLERANCE = 1.0e-8;

// The coefficients of the rigid-body motions whose value at six support DOFs is the identity:
// C = Φ_S⁻¹ (6 × 6, column j at [6·j, 6·j + 5]), Φ_S the rows of the rigid-body motions at the supports (row j the
// support j's, `phiAtSupports` row-major 6 × 6), by Gauss-Jordan elimination with partial pivoting; nothing when there
// are not six supports or Φ_S is singular (its smallest pivot at most 1e-12 of its largest: the supports do not fix
// every rigid motion). D = Φ·C is then the support motion of a free body without the round-off of a solve: D_S = I and
// D a rigid motion.
[[nodiscard]] std::optional<std::array<double, MAX_SUPPORT_DOFS * MAX_SUPPORT_DOFS>> rigidSupportCoefficients(
    std::span<const double> phiAtSupports, std::size_t supportCount);

// PARAM INREL: 0 no inertia relief (a FreeBodySupport then holds its DOFs as a constraint), −1 inertia relief on the
// FreeBodySupport/SUPORT1 DOFs, −2 inertia relief without FreeBodySupport (the supports chosen by the solver).
enum class InertiaReliefMode : std::int8_t
{
    None = 0,
    OnSupports = -1,
    Automatic = -2
};

// The r-set DOFs of a subcase (SUPORT, and the SUPORT1 set the subcase selects), from O_o_Reader's dofSetsTable (the
// selection of a SUPORT1 set given in the table as the subcases that select it). A point the model does not have, or a
// component a scalar point does not have, is an error Diagnostic; an empty result without the table. Sorted, each DOF
// once.
[[nodiscard]] std::vector<base::DofIndex> supportDofsOf(const input::ModelSnapshot& snapshot, const DofMap& dofMap,
                                                        base::EntityId subcase, base::DiagnosticSink& sink);

// The supports of PARAM INREL = −2: six DOFs of the a set (exclusive set b) on which the rigid-body modes are
// independent, chosen by a pivoted Gram-Schmidt on the rows of D (the rigid-body modes of `modes`): at each step the
// DOF whose row is the farthest from the span of the rows taken (the farthest points, which make the best conditioned
// statically determinate support), on a tie the lowest DOF. Fewer than six when the model does not let six motions
// be held (a model on a line); the solution then refuses it.
[[nodiscard]] std::vector<base::DofIndex> chooseAutomaticSupports(const RigidBodyModes& modes, const DofSets& sets);

// The supports a body free apart from its FreeBodySupport DOFs is solved on: six translational DOFs of grid
// points among the candidates (`isCandidate`, one flag per g-set DOF), taken by a pivoted Gram-Schmidt on the rows of
// the rigid-body motions about the candidates' centroid (the rotations divided by the candidates' RMS distance to it,
// so that the six columns weigh alike; Golub & Van Loan, Matrix Computations, §5.4.1): at each step the DOF whose row
// is the farthest from the span of the rows taken, the lowest DOF on a tie. They are the farthest points, whose
// translations hold the body's rotations by the longest lever arms. A body held at one point by its six DOFs (a
// FreeBodySupport of 123456) is held against its rotations by that point's own rotational stiffness alone, a small
// fraction of the body's for thin sheets: its K_ll is ill-conditioned (the stresses of a free barrel of four frames at
// 1/32, 1.3 M DOFs, 1.4e-5 apart with the FreeBodySupport on two points of a frame), where the farthest translations
// give the body's own condition (2e-8 apart). Fewer than six DOFs when the candidates do not fix every rigid motion
// (points on a line).
[[nodiscard]] std::vector<base::DofIndex> chooseConditionedSupports(const DofMap& dofMap,
                                                                    std::span<const std::uint8_t> isCandidate);

// u ← u + Φ·a on every grid point of the g set (`valuesG`, one value per g-set DOF): the rigid-body motion of
// amplitudes `amplitudes` (translations along the basic axes, rotations about them through the basic origin; Φ the
// convention of RigidBodyModes, u_cd = Rᵀ·(t + θ × x), θ_cd = Rᵀ·θ), made point by point without the g × 6 modes
// (the displacements of a free body brought back to the gauge of its FreeBodySupport DOFs). Scalar points do not
// move, nor the DOFs `isHeld` flags (one flag per g-set DOF; empty: none). Throws std::invalid_argument when `valuesG`
// or `isHeld` is not of the g-set size.
void addRigidMotion(const DofMap& dofMap, const std::array<double, RigidBodyModes::MODE_COUNT>& amplitudes,
                    std::span<double> valuesG, std::span<const std::uint8_t> isHeld = {});
// The same on the grid points `gridRows` only (rows of the DOF map's grid table, any order; for a condensed
// unit's points, recovered unit by unit): the values addRigidMotion adds there, to the bit.
void addRigidMotionAt(const DofMap& dofMap, const std::array<double, RigidBodyModes::MODE_COUNT>& amplitudes,
                      std::span<double> valuesG, std::span<const std::int32_t> gridRows,
                      std::span<const std::uint8_t> isHeld = {});

// Φᵀ·v about the basic origin, point by point without the g × 6 modes (RigidBodyModes' convention: the forces along the
// basic axes, then the moments about them; a load vector's resultant): F = Σ R·f_cd, M = Σ x × R·f_cd + R·m_cd over the
// grid points (the relief of a basis vector of the condensed load basis). Scalar points carry no resultant.
// Throws std::invalid_argument when `valuesG` is not of the g-set size.
[[nodiscard]] std::array<double, RigidBodyModes::MODE_COUNT> rigidResultantOf(const DofMap& dofMap,
                                                                              std::span<const double> valuesG);

// Inertia relief of a free body:
// the applied loads P are balanced by the inertia forces of a rigid-body acceleration of the r-set DOFs,
//     ü_r = (Dᵀ·M·D)⁻¹·Dᵀ·P,    P' = P − M·D·ü_r,
// D (g × r) the rigid-body motions of a unit motion of each support DOF (computed from the stiffness by the solution:
// D_l = −K_ll⁻¹·K_lr, D_r = I, expanded to the m set by Gm, zero on the s set), M = Mgg. The structure is then solved
// under P' with the supports held (u_r = 0): the displacements are relative to the supports, and the support reactions
// vanish to round-off. With PARAM INREL = −2 the displacements are then made orthogonal to the rigid-body motions in
// the mass metric, u ← u − D·(Dᵀ·M·D)⁻¹·Dᵀ·M·u, which does not depend on the supports chosen (this project's choice of
// normalisation).
//
// Thread-safety: const methods are safe concurrently. Ownership: owns D, M·D and the reduced mass; borrows M.
class InertiaRelief
{
public:
    // Throws std::invalid_argument when D has more than MAX_SUPPORT_DOFS columns or not the size of M; SolverError
    // when Dᵀ·M·D is singular (a support DOF whose motion moves no mass).
    InertiaRelief(linalg::DenseMatrix supportModes, const linalg::BlockSparseMatrix& mass, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t supportCount() const noexcept { return m_modes.columns(); }
    [[nodiscard]] const linalg::DenseMatrix& supportModes() const noexcept { return m_modes; }
    // M·D.
    [[nodiscard]] const linalg::DenseMatrix& massModes() const noexcept { return m_massModes; }
    // Dᵀ·M·D.
    [[nodiscard]] const base::MatN<6>& reducedMass() const noexcept { return m_reducedMass; }

    // ü_r of the loads P (g set).
    [[nodiscard]] std::array<double, MAX_SUPPORT_DOFS> accelerationsOf(std::span<const double> loadsG) const;
    // P ← P − M·D·ü_r.
    void relieve(std::span<double> loadsG) const;
    // u ← u − D·(Dᵀ·M·D)⁻¹·Dᵀ·M·u (PARAM INREL = −2).
    void removeRigidMotion(std::span<double> displacementsG, base::MemoryBudget& budget) const;
    // The amplitudes of the rigid motion of u in the mass metric, (Dᵀ·M·D)⁻¹·(M·D)ᵀ·u (from M·D, without a product
    // with M): u − D·amplitudes is M-orthogonal to D (the gauge of the refinement's residuals).
    [[nodiscard]] std::array<double, MAX_SUPPORT_DOFS> rigidMotionOf(std::span<const double> displacementsG) const;

private:
    // x = (Dᵀ·M·D)⁻¹·b on the support count.
    [[nodiscard]] std::array<double, MAX_SUPPORT_DOFS> solveReduced(std::array<double, MAX_SUPPORT_DOFS> b) const;

    const linalg::BlockSparseMatrix* m_mass;
    linalg::DenseMatrix m_modes;     // D
    linalg::DenseMatrix m_massModes; // M·D
    base::MatN<6> m_reducedMass;
    base::MatN<6> m_cholesky; // lower triangle of the Cholesky factor of Dᵀ·M·D
};

} // namespace oos::assembly
