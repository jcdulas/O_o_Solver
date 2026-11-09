#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstdint>
#include <span>

namespace oos::reduction
{

// A unit reduced to its boundary: what it gives the interface problem, and all that a cache may keep of it (never an
// interior factor):
// - K* = K_aa − K_ao·K_oo⁻¹·K_oa on its boundary DOFs, named by their labels (grid or reference point, component);
// - the condensed load patterns f*_k = f_a,k − K_ao·K_oo⁻¹·f_o,k, one per vector of the unit's load basis (not per
//   load case), with the stamp of what they depend on (UnitLoadBasis::patternStamp);
// - the six inertial load patterns f*_I = (−M·Φ_r)_a − K_ao·K_oo⁻¹·(−M·Φ_r)_o and the rigid-body mass properties
//   Φ_rᵀ·M·Φ_r about the basic origin (inertia relief on the interface problem, without recovering the unit).
// Never an interior factor nor a recovery matrix (K_oo⁻¹·K_oa): those belong to the condensation in memory
// (StaticCondensation), never to what is kept.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its matrices (budgeted).
class ReducedUnit
{
public:
    ReducedUnit(std::int64_t unitId, base::BudgetedBuffer<UnitDofLabel> labels, linalg::DenseMatrix stiffness,
                linalg::DenseMatrix patterns, std::int64_t patternStamp, linalg::DenseMatrix inertialPatterns,
                const base::MatN<6>& rigidBodyMass);

    [[nodiscard]] std::int64_t unitId() const noexcept { return m_unitId; }
    [[nodiscard]] std::size_t boundaryDofCount() const noexcept { return m_labels.size(); }
    [[nodiscard]] std::span<const UnitDofLabel> labels() const noexcept { return m_labels.span(); }
    // K*, boundaryDofCount() square, both triangles.
    [[nodiscard]] const linalg::DenseMatrix& stiffness() const noexcept { return m_stiffness; }
    // f* (boundaryDofCount() × patterns), the columns in the order of the unit's load basis.
    [[nodiscard]] const linalg::DenseMatrix& patterns() const noexcept { return m_patterns; }
    [[nodiscard]] std::int64_t patternStamp() const noexcept { return m_patternStamp; }
    // f*_I (boundaryDofCount() × 6): translations along the basic axes, rotations about them.
    [[nodiscard]] const linalg::DenseMatrix& inertialPatterns() const noexcept { return m_inertialPatterns; }
    [[nodiscard]] const base::MatN<6>& rigidBodyMass() const noexcept { return m_rigidBodyMass; }
    // The bytes it holds (what a cache entry of it weighs).
    [[nodiscard]] std::uint64_t byteCount() const noexcept;

    // The force of the unit on its boundary DOFs for given boundary displacements and pattern coefficients
    // (checks and grid point forces without recovery): K*·u_a − f*·α, column by column
    // (u_a: boundaryDofCount() × cases, α: patterns × cases, out: boundaryDofCount() × cases). Exact: it is the
    // internal force of the unit's elements at its boundary, K_aa·u_a + K_ao·u_o, less its own loads there. Each
    // term summed in a fixed order by the kernels. Throws std::invalid_argument on a size mismatch.
    void boundaryForces(linalg::ConstDenseMatrixView boundaryDisplacements, linalg::ConstDenseMatrixView coefficients,
                        linalg::DenseMatrixView forces, const linalg::DenseKernels& kernels) const;

private:
    std::int64_t m_unitId = 0;
    base::BudgetedBuffer<UnitDofLabel> m_labels;
    linalg::DenseMatrix m_stiffness;
    linalg::DenseMatrix m_patterns;
    std::int64_t m_patternStamp = 0;
    linalg::DenseMatrix m_inertialPatterns;
    base::MatN<6> m_rigidBodyMass{};
};

} // namespace oos::reduction
