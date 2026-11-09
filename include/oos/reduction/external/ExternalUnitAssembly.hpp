#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/reduction/external/ExternalUnit.hpp"
#include "oos/reduction/external/ExternalZone.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::reduction
{

// An external unit placed on the DOFs of the model (it enters the analysis as a condensed unit does: on the full path,
// a dense block assembled on the contour DOFs): its matrices in the order of
// the zone's contour DOFs (ExternalZone::contourLabels), symmetrised, ½·(A + Aᵀ), once ExternalUnitChecks accepted
// their asymmetry; the scatter of K* and M* into a node-block matrix of the model; the force of the unit on its contour
// for given contour displacements and condensed loads, K*·u_a − f*; and the resultant of K*·u_a, the force its held
// constraints take (zero for a free unit).
//
// Determinism: every sum in a fixed order (the contour order).
//
// Thread-safety: immutable once built. Ownership: shares the unit; owns its placed matrices (budgeted: the budget must
// outlive it) and the zone.
class ExternalUnitAssembly
{
public:
    // Precondition: ExternalUnitChecks accepted the unit for `zone`. Throws std::invalid_argument when a contour label
    // is not one of the unit's (the checks were not made), MemoryBudgetError.
    ExternalUnitAssembly(std::shared_ptr<const ExternalUnit> unit, ExternalZone zone, base::MemoryBudget& budget);

    [[nodiscard]] const ExternalUnit& unit() const noexcept { return *m_unit; }
    [[nodiscard]] const std::shared_ptr<const ExternalUnit>& sharedUnit() const noexcept { return m_unit; }
    [[nodiscard]] const ExternalZone& zone() const noexcept { return m_zone; }
    [[nodiscard]] std::size_t contourSize() const noexcept { return m_zone.contourDofs.size(); }
    [[nodiscard]] std::size_t interiorSize() const noexcept { return m_zone.interiorDofs.size(); }

    // K*, M* (null without one) and the derivatives of the unit (in unit().derivatives() order), contour × contour.
    [[nodiscard]] const linalg::DenseMatrix& stiffness() const noexcept { return m_stiffness; }
    [[nodiscard]] const linalg::DenseMatrix* mass() const noexcept { return m_mass ? &*m_mass : nullptr; }
    [[nodiscard]] const linalg::DenseMatrix& stiffnessDerivative(std::size_t d) const { return m_derivatives[d]; }
    [[nodiscard]] const linalg::DenseMatrix* massDerivative(std::size_t d) const
    {
        return m_massDerivatives[d] ? &*m_massDerivatives[d] : nullptr;
    }
    // The row of the unit's own matrices (and of its load source's f*) of each contour DOF.
    [[nodiscard]] std::span<const std::size_t> unitRowOfContour() const noexcept { return m_unitRow; }

    // The contour nodes of the model's node-block pattern (assembly::DofMap::nodeDofOf), sorted: the clique of K* that
    // the node graph must couple.
    [[nodiscard]] std::vector<std::int32_t> contourNodes(const assembly::DofMap& dofMap) const;

    // Adds K* (M*) onto the contour DOFs of `target`, a g-set matrix on a pattern that couples contourNodes(). Throws
    // std::logic_error when two contour nodes are not coupled.
    void addStiffnessTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& dofMap) const;
    void addMassTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& dofMap) const;
    // The same for any contour × contour matrix of the unit (a derivative of M*: its inertial loads).
    void addMatrixTo(const linalg::DenseMatrix& matrix, linalg::BlockSparseMatrix& target,
                     const assembly::DofMap& dofMap) const;

    // The values of a g-set vector at the contour DOFs (u_a), and at the interior DOFs.
    void gatherContour(std::span<const double> valuesG, std::span<double> contour) const;
    void gatherInterior(std::span<const double> valuesG, std::span<double> interior) const;

    // product = A·contour for a contour × contour matrix A (K*, a derivative), in the contour order.
    static void multiply(const linalg::DenseMatrix& a, std::span<const double> contour, std::span<double> product);

    // The resultant of a contour vector (forces and moments about the basic origin, in the basic system): Σ_a r_a·v_a
    // with r_a the rigid-body motions of the contour DOFs (the virtual work of the six unit motions), as
    // assembly::RigidBodyModes::resultantOf on the g set.
    [[nodiscard]] std::array<double, 6> resultantOf(std::span<const double> contour,
                                                    const assembly::DofMap& dofMap) const;

private:
    std::shared_ptr<const ExternalUnit> m_unit;
    ExternalZone m_zone;
    std::vector<std::size_t> m_unitRow; // bounded by the contour
    linalg::DenseMatrix m_stiffness;
    std::optional<linalg::DenseMatrix> m_mass;
    std::vector<linalg::DenseMatrix> m_derivatives;                    // bounded by the plugin's variables
    std::vector<std::optional<linalg::DenseMatrix>> m_massDerivatives; // bounded by the plugin's variables
};

} // namespace oos::reduction
