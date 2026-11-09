#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace oos::input
{

// How a model's element offsets behave (the OFFDEF parameter of the MDLPRM card): the shell ZOFFS of CQUAD4, CQUADR,
// CTRIA3, CTRIAR, CQUAD8, CTRIA6 and the WA, WB of CBAR and CBEAM. The stiffness is the same in the three values
// O_o_Solver takes (a shell's ZOFFS along its element normal, turning with it; a bar's or beam's offsets fixed); they
// differ by where the mass and the element loads of an offset element act and by its differential stiffness:
//
// - ElementOffsets (ELMOFF, the default). The offsets act on the stiffness only: the element mass and the element loads
//   (thermal; pressure: PLOAD2, PLOAD4; and the distributed loads PLOAD1 of a bar or beam, applied at the grid points'
//   positions) are computed as on the offset (a bar's between its offset ends, a shell's on its reference plane) and
//   carried to the grid points by the rotation of the frames alone, without the rigid offset. No differential
//   stiffness is formed for an offset element, so one in a buckling analysis is refused, naming ELMZ.
// - LargeRotationOffsets (LROFF): the mass, the loads and the differential stiffness go through the rigid offset, as
//   the stiffness does (the differential stiffness of an offset element is the element's carried through the offset;
//   no extra term for the offset itself). The offset direction of a shell under this value would come from the shell
//   normals at its grid points rather than from the element normal, which O_o_Solver does not form: it takes LROFF for
//   the offsets of bars and beams only (whose WA and WB give the offset direction) and refuses it on a shell with a
//   ZOFFS, naming ELMZ.
// - ElementZ (ELMZ): as LROFF, with the element z direction as a shell's offset direction.
//
// The other OFFDEF values (NODIF, NOTHRM, NODT, NOMASS, NDMTZ) turn off parts of LROFF's differential stiffness,
// thermal loads or mass that O_o_Solver does not form apart: refused.
enum class OffsetDefinition : std::uint8_t
{
    ElementOffsets,
    LargeRotationOffsets,
    ElementZ
};

// Whether the mass of an offset element is on its offset (LROFF, ELMZ) rather than at its grid points (ELMOFF).
[[nodiscard]] constexpr bool carriesMassOnOffsets(OffsetDefinition definition) noexcept
{
    return definition != OffsetDefinition::ElementOffsets;
}

// Whether the element loads of an offset element (thermal, pressure: PLOAD2, PLOAD4; PLOAD1 on a bar or beam) act
// through its offset (LROFF, ELMZ) rather than at its grid points (ELMOFF), and whether its differential stiffness is
// formed (under ELMOFF it is refused).
[[nodiscard]] constexpr bool carriesLoadsOnOffsets(OffsetDefinition definition) noexcept
{
    return definition != OffsetDefinition::ElementOffsets;
}

// The MDLPRM OFFDEF value of a definition ("ElemOffsetsOnly", "LoadRecoveryOffsets", "ElemOffsetZ").
[[nodiscard]] std::string_view offsetDefinitionName(OffsetDefinition definition) noexcept;

// The definition a ModelParams OffsetDefinition value names (case-sensitive; surrounding blanks ignored);
// nothing for a value O_o_Solver does not take (the other OFFDEF values, or a misspelt one).
[[nodiscard]] std::optional<OffsetDefinition> offsetDefinitionOf(std::string_view value) noexcept;

} // namespace oos::input
