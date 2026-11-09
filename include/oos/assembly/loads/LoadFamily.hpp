#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace oos::assembly
{

// The families of static loads (for the load basis of many load cases): the load vectors of the cases are
// grouped by family, and each family takes its own smallest exact basis.
// - Concentrated: FORCE, PointForceLoad1, PointForceLoad2, MOMENT, PointMomentLoad1, PointMomentLoad2, ScalarPointLoad
// (loadsTable, per grid or scalar point);
// - Distributed: ShellPressureLoad, FacePressureLoad (pressureTable) and BeamDistributedLoad (lineLoadsTable), per
// element;
// - Inertial: GravityLoad, ACCEL, AccelNodeLoad, RotationalForceLoad (Mgg times an acceleration field);
// - Thermal: the thermal loads of TEMPERATURE(LOAD);
// - Enforced: the enforced displacements of SPC and EnforcedMotionConstr (a displacement vector of the s set, not a
// load).
enum class LoadFamily : std::uint8_t
{
    Concentrated,
    Distributed,
    Inertial,
    Thermal,
    Enforced
};

inline constexpr std::array<LoadFamily, 5> ALL_LOAD_FAMILIES{
    LoadFamily::Concentrated, LoadFamily::Distributed, LoadFamily::Inertial, LoadFamily::Thermal, LoadFamily::Enforced};

// The name of a family in the reports: "concentrated", "distributed", "inertial", "thermal", "enforced".
[[nodiscard]] constexpr std::string_view nameOf(LoadFamily family) noexcept
{
    switch (family)
    {
    case LoadFamily::Concentrated:
        return "concentrated";
    case LoadFamily::Distributed:
        return "distributed";
    case LoadFamily::Inertial:
        return "inertial";
    case LoadFamily::Thermal:
        return "thermal";
    case LoadFamily::Enforced:
        return "enforced";
    }
    return "";
}

} // namespace oos::assembly
