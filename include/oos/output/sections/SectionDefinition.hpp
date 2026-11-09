#pragma once

#include "oos/base/math/Vec3.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::output
{

// How the element loads of a section are reduced: the weighted average Σ w·N / Σ w (the default: the mean running
// loads of a panel), or the weighted sum Σ w·N (the axial forces of the stringers of a panel).
enum class SectionReduce : std::uint8_t
{
    Average,
    Sum
};

// The element families a section may hold (a section holds elements of one family only), each with the components of
// its loads:
// - Shell (CQUAD4, CQUADR, CQUAD8, CTRIA3, CTRIAR, CTRIA6): nx, ny, nxy, mx, my, mxy, qx, qy, the element forces per
//   unit length at the centre (the FORCE output), rotated into the section frame;
// - Bar (CBAR): bendingMomentA1, bendingMomentA2, bendingMomentB1, bendingMomentB2, shear1, shear2, axial,
//   torque, in the element frame (the FORCE output of the CBAR);
// - Beam (CBEAM): the seven forces of end A then of end B (bendingMoment1A ... warpingTorqueA, bendingMoment1B
//   ... warpingTorqueB), in the element frame;
// - Rod (CROD, CONROD, CTUBE): axial, torsion.
enum class SectionFamily : std::uint8_t
{
    Shell,
    Bar,
    Beam,
    Rod
};

// The family of an element card, or nothing for a card no section takes.
[[nodiscard]] std::optional<SectionFamily> sectionFamilyOf(std::string_view card) noexcept;
[[nodiscard]] std::string_view nameOf(SectionFamily family) noexcept;
// The components of a family, in the order of the rows of a section.
[[nodiscard]] std::span<const std::string_view> sectionComponents(SectionFamily family) noexcept;
// The largest number of components of a family (a CBEAM: seven at each end).
inline constexpr std::size_t MAX_SECTION_COMPONENTS = 14;

// One element of a section with its weight; no weight: the default of the family (the element area for a shell, 1 for
// a bar, a beam or a rod).
struct SectionMember
{
    std::int32_t elementId = 0;
    std::optional<double> weight;
};

// One section as the caller defines it (a table of `section`, `elementId`, optionally `weight`, and per section its
// frame by an `axis` in the basic system or by a coordinate system `frameId`): its elements, its frame and its
// reduction. The frame matters for the shells only: the x axis of the section frame in each element is the projection
// of the axis (or of the x axis of the rectangular system CID) on the plane of the element, the normal that of the
// element, so that the element forces are rotated about their normal only. A bar, beam or rod section keeps the frame
// of each element (a frame given to it is not used).
struct SectionDefinition
{
    std::int32_t id = 0;    // the label of the section (its value, or its rank among text labels)
    std::string name;       // as the caller names it, for the messages and the tables
    std::int32_t group = 0; // a tag of the caller carried to the labels (a DesignProblem: the response that asks)
    std::vector<SectionMember> members;
    std::optional<base::Vec3> axis; // in the basic system
    std::optional<std::int32_t> frameId;
    SectionReduce reduce = SectionReduce::Average;
};

} // namespace oos::output
