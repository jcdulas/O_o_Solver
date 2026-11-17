#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace oos::input
{

// How a model's element offsets behave (the OffsetDefinition parameter of the ModelParams card): the shell ZOFFS of
// ShellQuad4, ShellQuad4Drilling, ShellTria3, ShellTria3Drilling, ShellQuad8, ShellTria6 and the WA, WB of Bar2 and
// Beam2. The stiffness is the same in the three values O_o_Solver takes (a shell's ZOFFS along its element normal,
// turning with it; a bar's or beam's offsets fixed); they differ by where the mass and the element loads of an offset
// element act and by its differential stiffness:
//
// - ElementOffsets (ElemOffsetsOnly, the default). The offsets act on the stiffness only: the element mass and the
//   element loads (thermal; pressure: ShellPressureLoad, FacePressureLoad; and the distributed loads
//   BeamDistributedLoad of a bar or beam, applied at the grid points' positions) are computed as on the offset (a
//   bar's between its offset ends, a shell's on its reference plane) and carried to the grid points by the rotation of
//   the frames alone, without the rigid offset. No differential stiffness is formed for an offset element, so one in
//   a buckling analysis is refused, naming ElemOffsetZ.
// - LargeRotationOffsets (LoadRecoveryOffsets): the mass, the loads and the differential stiffness go through the rigid
//   offset, as the stiffness does (the differential stiffness of an offset element is the element's carried through the
//   offset; no extra term for the offset itself). The offset direction of a shell under this value would come from the
//   shell normals at its grid points rather than from the element normal, which O_o_Solver does not form: it takes
//   LoadRecoveryOffsets for the offsets of bars and beams only (whose WA and WB give the offset direction) and refuses
//   it on a shell with a ZOFFS, naming ElemOffsetZ.
// - ElementZ (ElemOffsetZ): as LoadRecoveryOffsets, with the element z direction as a shell's offset direction.
//
// The other OffsetDefinition values (NoOffsetDifference, NoThermalOffsets, NoTemperatureGradientOffsets, NoMassOffsets,
// NoDampingOffsets) turn off parts of the differential stiffness, thermal loads or mass of LoadRecoveryOffsets that
// O_o_Solver does not form apart: refused.
enum class OffsetDefinition : std::uint8_t
{
    ElementOffsets,
    LargeRotationOffsets,
    ElementZ
};

// Whether the mass of an offset element is on its offset (LoadRecoveryOffsets, ElemOffsetZ) rather than at its grid
// points (ElemOffsetsOnly).
[[nodiscard]] constexpr bool carriesMassOnOffsets(OffsetDefinition definition) noexcept
{
    return definition != OffsetDefinition::ElementOffsets;
}

// Whether the element loads of an offset element (thermal, pressure: PLOAD2, PLOAD4; PLOAD1 on a bar or beam) act
// through its offset (LoadRecoveryOffsets, ElemOffsetZ) rather than at its grid points (ElemOffsetsOnly), and whether
// its differential stiffness is formed (under ElemOffsetsOnly it is refused).
[[nodiscard]] constexpr bool carriesLoadsOnOffsets(OffsetDefinition definition) noexcept
{
    return definition != OffsetDefinition::ElementOffsets;
}

// The ModelParams OffsetDefinition value of a definition ("ElemOffsetsOnly", "LoadRecoveryOffsets", "ElemOffsetZ").
[[nodiscard]] std::string_view offsetDefinitionName(OffsetDefinition definition) noexcept;

// The definition a ModelParams OffsetDefinition value names (case-sensitive; surrounding blanks ignored);
// nothing for a value O_o_Solver does not take (the other OffsetDefinition values, or a misspelt one).
[[nodiscard]] std::optional<OffsetDefinition> offsetDefinitionOf(std::string_view value) noexcept;

} // namespace oos::input
