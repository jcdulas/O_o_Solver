#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::aero
{

// What a trim variable is (O_o_Reader's aeroTrimCasesTable "variable_kind"): an AeroRigidTrimVariable label the program
// knows (AERODYNAMIC_RIGID_BODY_LABELS, RIGID_ACCELERATION_LABELS), an AeroControlSurface, an AeroController or another
// AeroRigidTrimVariable label (user), or a label no card defines.
enum class TrimVariableKind : std::uint8_t
{
    RigidBody,
    ControlSurface,
    User,
    Undefined
};

// How a trim case sets a variable (aeroTrimCasesTable "state"): fixed to a value by the TRIM, solved for, given by an
// AeroTrimLink from others, or scheduled by a AeroControlSchedule.
enum class TrimVariableState : std::uint8_t
{
    Fixed,
    Free,
    Linked,
    Scheduled
};

struct TrimSetting
{
    std::string label; // upper case (the labels of the input format are not case-sensitive)
    TrimVariableKind kind = TrimVariableKind::RigidBody;
    TrimVariableState state = TrimVariableState::Free;
    double value = 0.0; // the value of a fixed variable
};

// One independent term of an AeroTrimLink.
struct LinkTerm
{
    std::string label;
    double coefficient = 0.0;
};

// An AeroTrimLink: u_D + Σ C_i·u_i = 0, the dependent variable is minus the combination of its independent ones (the
// rule O_o_Reader's card documents): a link of +1 makes the two surfaces deflect opposite ways (elev_l = −elev_r), a
// link of −1 the same way (ailr_l = ailr_r). `id` is the TRIM set it applies to, 0 for ID = 0 or ALWAYS (every trim
// case). O_o_Reader gives the AeroTrimLink cards as rows without their fields: the caller reads them from the cards and
// the solver takes them as options.
struct AeroLink
{
    std::int32_t id = 0;
    std::string dependent;
    std::vector<LinkTerm> terms; // bounded by the fields of the card
};

// The trim case of a subcase of a static aeroelastic analysis (TRIM, AeroTrim2): the TRIM set its Case Control selects,
// as O_o_Reader's aeroTrimCasesTable gives it: the Mach number, the dynamic pressure, AEQR, and every variable of the
// model with its state; the AeroTrimLink cards of the set (and those of ID 0 or ALWAYS).
//
// Thread-safety: an immutable value. Ownership: plain data.
struct TrimCase
{
    std::int32_t setId = 0;
    base::EntityId subcase = 1;
    double mach = 0.0;
    double dynamicPressure = 0.0;
    // TRIM AEQR (0 to 1, default 1): the share of the aerodynamic loads of the deformation that act on the structure, 0
    // for a rigid trim (the structure deforms under the loads of its rigid shape, without feedback).
    double aeqr = 1.0;
    std::vector<TrimSetting>
        settings; // in the order of aeroTrimCasesTable (AESTAT, AeroControlSurface, AeroController by ID)
    std::vector<AeroLink> links; // those of the linked variables, one per linked variable

    // The trim case that the Case Control of `subcase` selects (its SUBCASE label, or `tableId`, its row of
    // O_o_Reader's subcasesTable: 0 for the global section of a deck without SUBCASE), or nothing when no TRIM selects
    // it. Its problems are error diagnostics of `sink`: two TRIM sets for the subcase, a Mach number of 1 or more, a
    // dynamic pressure not positive, AEQR outside [0, 1], a scheduled variable (CSSCHD, not supported), a linked
    // variable without its AeroTrimLink or with two, an AeroTrimLink that names a variable the model does not have or
    // that is itself linked.
    [[nodiscard]] static std::optional<TrimCase> forSubcase(const input::ModelSnapshot& snapshot,
                                                            base::EntityId subcase, base::EntityId tableId,
                                                            std::span<const AeroLink> links,
                                                            base::DiagnosticSink& sink);

    // The index of the variable `label` (exact spelling), or nothing.
    [[nodiscard]] std::optional<std::size_t> find(std::string_view label) const;
};

// The AeroRigidTrimVariable labels the program knows, as a model file writes them: the rigid-body variables with an
// aerodynamic effect (angle of attack, side slip, roll, pitch and yaw rates), in the order of the derivative tables,
// and the accelerations of the supports in the reference frame (three translations, then three rotations).
inline constexpr std::array<std::string_view, 5> AERODYNAMIC_RIGID_BODY_LABELS{"AngleOfAttack", "SideSlip", "RollRate",
                                                                               "PitchRate", "YawRate"};
inline constexpr std::array<std::string_view, 6> RIGID_ACCELERATION_LABELS{
    "RigidAccelX", "RigidAccelY", "RigidAccelZ", "RigidAngularAccelX", "RigidAngularAccelY", "RigidAngularAccelZ"};

} // namespace oos::aero
