#pragma once

#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::output
{

// The contributions reported for a monitor point of a trim: the aerodynamic loads of the rigid configuration at the
// trimmed variables, those of the trimmed elastic configuration (restrained), the inertial loads M·a of the trimmed
// accelerations, the applied loads (rigid and restrained).
enum MonitorContribution : std::size_t
{
    RigidAir,
    ElasticRestrained,
    Inertial,
    RigidApplied,
    RestrainedApplied
};
inline constexpr std::size_t MONITOR_CONTRIBUTION_COUNT = 5;

// A MONPNT1 as the caller gives it: O_o_Reader gives the MONPNT1 fields as columns but the AECOMP and AECOMPL only as
// card objects, so the caller resolves the component of each monitor point to its boxes (an AECOMP of AELIST or CAERO
// lists) or its grid points (of SET1 lists), the AECOMPL expanded.
struct MonitorPoint1Definition
{
    std::string name;
    std::string label;
    std::string component; // the AECOMP or AECOMPL
    // The class of the point: the CLASS of a MONPNT1 (GENERAL by default), and for the points the trim adds,
    // "vehicleCoefficients" (the whole vehicle) and "hingeMoment" (the boxes of an AESURF about its hinge line).
    std::string pointClass = "GENERAL";
    std::int32_t axes = 123456;    // AXES: the components (1 to 6) the point reports
    bool isAerodynamic = false;    // boxes (AELIST, CAERO) or grid points (SET1)
    std::vector<std::int32_t> ids; // the box or grid point IDs, bounded by the component
    std::int32_t positionFrameId = 0;
    base::Vec3 pointInPositionFrame; // X, Y, Z
    std::int32_t displacementFrameId = 0;
};

// The integrated loads of a MONPNT1: the resultant of the forces of its component about its point, the forces along the
// axes of CD and the moments about them through the point. A structural component sums the g-set loads of its grid
// points (the loads in their displacement systems turned to the basic one), an aerodynamic component the forces of its
// boxes (along their normals at their force points).
//
// Thread-safety: immutable. Ownership: owns its data.
class MonitorPoint1
{
public:
    // The monitor point of `definition` in the frames of the snapshot. Throws SolverError for a CP or CD the snapshot
    // does not have, or that is not rectangular, and for an ID its component names that the model does not have.
    MonitorPoint1(const MonitorPoint1Definition& definition, const input::ModelSnapshot& snapshot,
                  const assembly::DofMap& dofMap, const aero::AeroReference& reference, const aero::AeroBoxes& boxes);

    [[nodiscard]] const MonitorPoint1Definition& definition() const noexcept { return m_definition; }
    [[nodiscard]] base::Vec3 pointInBasic() const noexcept { return m_point; }
    // Whether component c (0 to 5: fx, fy, fz, mx, my, mz) is among AXES.
    [[nodiscard]] bool reports(std::size_t component) const noexcept { return m_isReported[component]; }

    // The six loads of g-set loads (structural component).
    [[nodiscard]] std::array<double, 6> loadsOfStructure(std::span<const double> loadsG) const;
    // The six loads of box forces, one per box of AeroBoxes, along the normals at the force points (aerodynamic
    // component; the boxes are in the aerodynamic frame, turned to the basic one).
    [[nodiscard]] std::array<double, 6> loadsOfBoxes(std::span<const double> boxForces) const;

private:
    MonitorPoint1Definition m_definition;
    const assembly::DofMap* m_dofMap;
    const aero::AeroBoxes* m_boxes;
    base::Vec3 m_point;    // in the basic system
    base::Mat3 m_cdAxes;   // the axes of CD in the basic system (columns)
    base::Mat3 m_aeroAxes; // the axes of ACSID in the basic system
    base::Vec3 m_aeroOrigin;
    std::array<bool, 6> m_isReported{};
    std::vector<std::int32_t> m_rows;      // the grid rows of a structural component
    std::vector<std::size_t> m_boxIndices; // the boxes of an aerodynamic component
};

} // namespace oos::output
