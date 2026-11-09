#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"

#include <array>
#include <span>

namespace oos::assembly
{

// The six rigid-body motions of the model about a reference point, as g-set vectors (as in the grid point weight
// generator: the matrix D of the rigid-body transformation from the reference point to the grid
// points). Column k is the motion of a unit translation along the basic axis k (k = 0, 1, 2) or of a unit rotation
// about the basic axis k − 3 through the reference point (k = 3, 4, 5): a grid point at r from the reference point
// moves by u = t + θ × r and rotates by θ, written in its displacement system CD (u_cd = Rᵀ·u_basic); scalar points do
// not move.
//
// Thread-safety: an immutable value. Ownership: owns its matrix (budgeted).
class RigidBodyModes
{
public:
    static constexpr std::size_t MODE_COUNT = 6;

    RigidBodyModes(const DofMap& dofMap, base::Vec3 referencePoint, base::MemoryBudget& budget);

    [[nodiscard]] const linalg::DenseMatrix& modes() const noexcept { return m_modes; }
    [[nodiscard]] base::Vec3 referencePoint() const noexcept { return m_reference; }

    // Dᵀ·v: the resultant of the g-set load vector v about the reference point, in the basic system (forces F1-F3 and
    // moments M1-M3), the CD components turned into the basic ones by the virtual work of the rigid motions.
    [[nodiscard]] std::array<double, MODE_COUNT> resultantOf(std::span<const double> valuesG) const;

    // Dᵀ·M·D: the 6 × 6 rigid-body mass matrix of `mass` about the reference point (MO of the grid point weight
    // generator, before any division by WTMASS).
    [[nodiscard]] base::MatN<6> rigidBodyMass(const linalg::BlockSparseMatrix& mass, base::MemoryBudget& budget) const;

private:
    base::Vec3 m_reference;
    linalg::DenseMatrix m_modes; // g × 6
};

} // namespace oos::assembly
