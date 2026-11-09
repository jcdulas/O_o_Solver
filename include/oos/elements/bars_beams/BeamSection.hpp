#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace oos::elements
{

// The section of a bar or beam at one station along its length, in the element system (PBAR, BeamProp): y and z the
// element axes, I1 = ∫y² dA (bending in plane 1, the x-y plane), I2 = ∫z² dA (plane 2, x-z), I12 = ∫yz dA, all about
// the neutral axis.
//
// The element axis (the line between the end points) is the axis of the shear centres of a Beam2 (PBEAM,
// BeamShapeProp): the transverse shears pass through it and the torque is about it. The neutral axis is at
// `neutralAxis` = (N1, N2) from it (PBEAM N1, N2; a BeamShapeProp's from O_o_Reader's shear centre and centroid): the
// axial force acts there and the bending moments are about it. A Bar2's is (0, 0): its axis is the neutral axis and
// the shear centre (PBAR, BarShapeProp). `massCentre` = (M1, M2) is the centre of gravity of the non-structural mass
// from the element axis (PBEAM M1, M2; (0, 0) for the others). The stress recovery points C, D, E, F are (y, z) pairs
// from the neutral axis (a BeamProp gives them from the shear centre: Beam2Element subtracts N1, N2).
struct BeamStation
{
    double position = 0.0;                                 // X/XB: the station as a fraction of the length from end A
    double area = 0.0;                                     // A
    double inertia1 = 0.0;                                 // I1
    double inertia2 = 0.0;                                 // I2
    double productOfInertia = 0.0;                         // I12
    double torsionalConstant = 0.0;                        // J
    double nonStructuralMass = 0.0;                        // NSM, mass per unit length
    std::array<std::array<double, 2>, 4> recoveryPoints{}; // C, D, E, F as (y, z) from the neutral axis
    std::array<double, 2> neutralAxis{};                   // (N1, N2): the neutral axis from the element axis
    std::array<double, 2> massCentre{};                    // (M1, M2): the NSM's centre of gravity from the axis
    bool hasOutput = true; // BeamProp SO: false for an intermediate station with SO = NO (no output there)

    [[nodiscard]] bool hasNeutralAxisOffset() const noexcept { return neutralAxis[0] != 0.0 || neutralAxis[1] != 0.0; }
    [[nodiscard]] bool hasMassCentreOffset() const noexcept { return massCentre[0] != 0.0 || massCentre[1] != 0.0; }
};

// A BeamProp has end A and at most ten more stations.
inline constexpr std::size_t BEAM_MAX_STATIONS = 11;

// The section of a bar or beam along its length: its stations in increasing position, the first at end A (0) and the
// last at end B (1); a BarProp has the same section at both ends. Between two stations each property varies linearly
// with the position, the neutral axis and the NSM's centre of gravity included (a modelling choice for tapered
// CBEAMs; see BeamFlexibility.hpp). The shear factors K1, K2 are those of the whole element.
//
// Fixed capacity (no allocation per element). Invariant (resolveBeamSection and the families ensure it): count ≥ 2,
// positions increasing from 0 to 1.
struct BeamSection
{
    std::array<BeamStation, BEAM_MAX_STATIONS> stations{};
    std::size_t count = 0;
    // K1, K2: the shear area of plane 1 (shear along y) is K1·A, that of plane 2 (along z) K2·A. A factor ≤ 0 means no
    // shear flexibility in that plane (PBAR: K1, K2 blank or 0.0 give an infinite shear stiffness; O_o_Reader
    // gives a blank BarProp factor as −1).
    double shearFactor1 = 0.0;
    double shearFactor2 = 0.0;

    // The section at a position (0 at end A, 1 at end B), each property interpolated linearly between the two
    // stations around it; the recovery points are those of the station at or before the position.
    [[nodiscard]] BeamStation at(double position) const;
    // True when every station has the properties of end A (a BarProp, or a BeamProp with one section).
    [[nodiscard]] bool isUniform() const;
    // True when a station has its neutral axis off the element axis (a Beam2 on a BeamProp with N1, N2, or on a
    // BeamShapeProp whose shear centre is not its centroid).
    [[nodiscard]] bool hasNeutralAxisOffset() const;
    // True when a station has the centre of gravity of its non-structural mass off the element axis (PBEAM M1, M2).
    [[nodiscard]] bool hasMassCentreOffset() const;
};

// A section whose stiffness cannot be formed: the property field as the card names it, and the message.
struct SectionProblem
{
    std::string field;
    std::string message;
};

// The problem of a section, if any: a negative property, an I12 other than 0 with I1·I2 ≤ I12² at a station, or a
// property that is zero at some stations and not at others (the flexibility of a section tapering to a zero bending,
// axial or torsional stiffness is not finite).
[[nodiscard]] std::optional<SectionProblem> findSectionProblem(const BeamSection& section);

} // namespace oos::elements
