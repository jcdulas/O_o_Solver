#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"

#include <array>
#include <cstdint>

namespace oos::assembly
{

// The grid point weight generator (PARAM GRDPNT; its table in the text report): the rigid-body mass properties of the
// g-set mass about a reference point, in the units of the input masses (the mass matrix is divided by WTMASS):
//
// - MO = Dᵀ·M·D / WTMASS, the 6 × 6 rigid-body mass matrix about the reference point in the basic system (D the
//   rigid-body modes, RigidBodyModes);
// - S, the principal directions of the translational mass Mt = MO[0:3, 0:3] (rows: the mass axes in the basic system;
//   the identity when Mt is diagonal, as for every isotropic mass);
// - the mass along each axis of S, m_i = (S·Mt·Sᵀ)_ii, and the centre of gravity seen by each of them, from the
//   coupling block Mtr = S·MO[0:3, 3:6]·Sᵀ: along x (y = −Mtr(0,2)/m1, z = Mtr(0,1)/m1), along y (x = Mtr(1,2)/m2,
//   z = −Mtr(1,0)/m2), along z (x = −Mtr(2,1)/m3, y = Mtr(2,0)/m3), the coordinate along its own axis left 0 (it is
//   not defined by that mass);
// - I(S), the inertia about the centre of gravity in the S axes: the tensor Irr = S·MO[3:6, 3:6]·Sᵀ minus the transfer
//   terms of each direction's centre of gravity,
//   Irr − [[m2 z2² + m3 y3², −m3 x3 y3, −m2 x2 z2], [., m1 z1² + m3 x3², −m1 y1 z1], [., ., m1 y1² + m2 x2²]],
//   printed with the products of inertia as +Σ m·x·y (the opposite sign of the tensor terms of MO);
// - I(Q), its principal moments, and Q, the principal axes (row i the axis of I(Q)_i in the S axes), each axis placed
//   at the S axis it is closest to and oriented with that component positive.
//
// Thread-safety: an immutable value.
struct GridPointWeight
{
    std::int32_t referencePoint = 0; // the grid point of PARAM GRDPNT, 0 for the origin of the basic system
    base::Vec3 referencePosition;    // in the basic system
    base::MatN<6> rigidBodyMass;     // MO
    base::Mat3 massAxes = base::Mat3::identity();      // S
    std::array<double, 3> masses{};                    // the mass along each S axis
    std::array<base::Vec3, 3> centresOfGravity{};      // per direction, offsets from the reference point (basic axes)
    base::Mat3 inertia;                                // I(S)
    std::array<double, 3> principalInertia{};          // I(Q)
    base::Mat3 principalAxes = base::Mat3::identity(); // Q

    // From MO (already divided by WTMASS) about `referencePosition`.
    [[nodiscard]] static GridPointWeight fromRigidBodyMass(const base::MatN<6>& rigidBodyMass,
                                                           std::int32_t referencePoint, base::Vec3 referencePosition);

    // The total mass (the mass along the first S axis: the mass of the model when the masses are isotropic).
    [[nodiscard]] double totalMass() const noexcept { return masses[0]; }
    // The centre of gravity of an isotropic mass in the basic system (the reference position plus the offsets): x from
    // the y and z directions, y from x and z,
    // z from x and y (each coordinate averaged over the two directions that define it).
    [[nodiscard]] base::Vec3 centreOfGravity() const noexcept;
};

} // namespace oos::assembly
