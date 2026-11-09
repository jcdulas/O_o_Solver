#pragma once

#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::aero
{

// The aerodynamic trim variables of a model and the normalwash a unit value of each puts on the boxes, w_j / V, the
// right-hand side of the doublet lattice (the rigid-body motions and the control surfaces). The boundary condition at
// a control point is that the flow does not cross the box: (V + v_induced)·n = 0, so the normalwash the boxes must
// induce is minus the normal component of the onset flow the variable adds:
//
// - AngleOfAttack α (radians): the flow V(cos α, 0, sin α) of the aerodynamic frame, w = −α n_z;
// - SideSlip β (radians): the flow coming from the side, V(cos β, −sin β, 0), w = β n_y;
// - RollRate, PitchRate, YawRate (p b / 2V, q c / 2V, r b / 2V about the x, y, z axes of the reference frame RCSID, at
// its origin):
//   the body turns at ω, the boxes move at ω × r, the onset flow is −ω × r, w = (ω × r)·n / V;
// - an AeroControlSurface of label L (radians): its boxes turn about the hinge line, the y axis of CID1, by the
// right-hand rule; a
//   normal n turns by δ e × n, w = −EFF δ (x × e)·n on its boxes (a linear downwash).
//
// The accelerations RigidAccelX to RigidAngularAccelZ load the structure, not the aerodynamics; an AeroController and
// the other AeroRigidTrimVariable labels act only through user matrices (not supported).
//
// Thread-safety: immutable. Ownership: copies of the values it needs.
class TrimVariables
{
public:
    // One variable: its label (upper case) and the kind of O_o_Reader's aeroTrimCasesTable ("rigid_body",
    // "control_surface").
    struct Variable
    {
        std::string label;
        bool isControlSurface = false;
    };

    // An AeroControlSurface: its ID and label, its hinge line (the y axis of CID1 through its origin, in the
    // aerodynamic frame), its effectiveness, the reference chord and area of its hinge moment coefficient (CREFC,
    // CREFS), its position and hinge moment limits (PLLIM, PULIM default to ±π/2; no hinge moment limit by default) and
    // its boxes.
    struct ControlSurface
    {
        std::int32_t id = 0;
        std::string label;
        std::int32_t frameId1 = 0;
        std::int32_t aeroListId1 = 0; // 0 when the table does not give it
        base::Vec3 hinge;
        base::Vec3 hingeOrigin;
        double effectiveness = 1.0;
        double referenceChord = 1.0;
        double referenceArea = 1.0;
        double positionLower = -DEFAULT_POSITION_LIMIT;
        double positionUpper = DEFAULT_POSITION_LIMIT;
        std::optional<double> hingeMomentLower;
        std::optional<double> hingeMomentUpper;
        std::vector<std::size_t> boxes; // bounded by the boxes of the surface's AeroBoxList
    };

    // AeroControlSurface PLLIM, PULIM: ±π/2 by default.
    static constexpr double DEFAULT_POSITION_LIMIT = 1.5707963267948966;

    // The rigid-body variables with an aerodynamic effect (AngleOfAttack, SideSlip, RollRate, PitchRate, YawRate) and
    // the AeroControlSurface of the model, in the order of the stability derivatives (the control surfaces by
    // AeroControlSurface ID, then the rigid-body variables in this order). Throws SolverError for an AeroControlSurface
    // whose hinge system is not rectangular or is missing.
    TrimVariables(const input::ModelSnapshot& snapshot, const AeroReference& reference, const AeroBoxes& boxes);

    [[nodiscard]] const std::vector<Variable>& variables() const noexcept { return m_variables; }
    [[nodiscard]] const std::vector<ControlSurface>& controlSurfaces() const noexcept { return m_surfaces; }

    // The normalwash w_j / V of a unit value of variable `label` (exact spelling) on the boxes (`normalwash`: one per
    // box). Throws std::invalid_argument for a label the model does not have.

    // The AeroRigidTrimVariable or AeroControlSurface ID of a variable (upper-case label), 0 when the model does not
    // give it (the AeroRigidTrimVariable IDs come from the AeroRigidTrimVariable card table, when the snapshot has it).
    [[nodiscard]] std::int32_t idOf(std::string_view label) const;
    void normalwashOf(std::string_view label, std::span<double> normalwash) const;

private:
    const AeroBoxes* m_boxes;
    AeroReference m_reference;
    std::vector<Variable> m_variables;      // bounded by the trim variables of the model
    std::vector<ControlSurface> m_surfaces; // bounded by the AeroControlSurface cards
    std::vector<std::pair<std::string, std::int32_t>>
        m_statisticIds; // the AeroRigidTrimVariable (label, ID), bounded by the cards
};

} // namespace oos::aero
