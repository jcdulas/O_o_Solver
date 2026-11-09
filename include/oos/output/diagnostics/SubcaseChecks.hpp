#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/math/Vec3.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::output
{

// The resultant of a nodal load vector (the applied loads of OLOAD, the constraint forces of SPCFORCES) about a
// reference point in the basic system: one row per load component (FX, FY, FZ, MX, MY, MZ: the sum over
// the grid points of that component of the force and moment, turned into the basic system), its six columns the
// resultant force T1-T3 and moment R1-R3 it gives about the reference point (the row FX gives T1 = ΣFx,
// R2 = Σ z·Fx, R3 = −Σ y·Fx, with (x, y, z) the position of each grid point from the reference point), and the totals.
// Scalar points are left out (they have no position).
struct LoadResultant
{
    std::array<std::array<double, 6>, 6> rows{}; // rows[component][T1..R3]
    std::array<double, 6> totals{};

    // The resultant of a g-set vector (forces and moments in the CD systems of the grid points).
    [[nodiscard]] static LoadResultant of(const assembly::DofMap& dofMap, std::span<const double> valuesG,
                                          base::Vec3 referencePoint);
};

// The checks of one static subcase: the OLOAD and SPC force resultants, and for a solved subcase EPSILON and the
// external work of the solve (ε = uᵀ·δP / uᵀ·P on the l set, δP = P − K_ll·u the residual of the solve; the external
// work uᵀ·P / 2 on the l set), and the accelerations of the supports under inertia relief (ü_r, one per r-set DOF).
struct SubcaseCheck
{
    std::int32_t subcase = 0;
    bool isCombination = false; // a SUBCOM: no EPSILON nor work (they exist for the solved subcases only)
    LoadResultant appliedLoads;
    LoadResultant spcForces;
    double epsilon = 0.0;
    double externalWork = 0.0;
    std::vector<double> supportAccelerations; // bounded by the r set (at most six)
};

} // namespace oos::output
