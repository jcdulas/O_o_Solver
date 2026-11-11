#pragma once

#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/output/sections/SectionDefinition.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace oos::design
{

// Where an element row of a result table lies: the centre (a shell's centre, a rod, a spring), or an end of a bar or
// beam (positions 0 and 1 of a station table).
enum class ItemPlace : std::uint8_t
{
    Centre,
    EndA,
    EndB
};

// The names of the composite failure index columns (ShellRecovery's laminate failure index table) a CFAILURE or CSTRAT
// item reads: the index of the direct stresses and that of the interlaminar shear.
inline constexpr std::string_view FAILURE_INDEX_COLUMN = "failureIndex";
inline constexpr std::string_view BOND_FAILURE_INDEX_COLUMN = "bondFailureIndex";

// What one item code of a DesignResp1 reads in the element result tables of the data contract (the tables O_o_Solver
// makes, output/arrays/ElementResultBuilder and StiffnessRecovery): the table kind, whether it is the composite table
// of the element ("ShellQuad4Composite", the row of the lamina ATTB), the column by its name in the table (so the order
// of the table's columns does not matter), the layer of a shell row (1: Z1 or the membrane strains, 2: Z2 or the
// curvatures; 0: the element has one row per place) and the place.
struct ItemLocation
{
    std::string tableKind; // "stress", "strain", "forces", "strainEnergy", "failureIndices"
    bool isComposite = false;
    std::string column;
    // A second name of the same quantity in another layout of the table, empty when none: a shell's strain table holds
    // the fibre distance, or the curvature flag in its membrane-and-curvature layout.
    std::string otherColumn;
    std::int32_t layer = 0;
    ItemPlace place = ItemPlace::Centre;
    bool isStrengthRatio = false; // CSTRAT: the strength ratio of the failure index column (StrengthRatio)

    // The position of the column among `names` (a table's or a family's column names), or nothing when they do not
    // have it.
    [[nodiscard]] std::optional<std::size_t> columnIn(std::span<const std::string> names) const;
};

// The location of item code `itemCode` of a response of type `type` (STRESS, STRAIN, FORCE, ESE, CSTRESS, CSTRAIN,
// CFAILURE, CSTRAT) on an element of card `card`, from the item codes of the input format, or the reason it is refused
// (a code the format does not give, an element whose items are not supported). Each family has its own table of item
// code -> column name in ItemCodes.cpp, the single source of the mapping (src/design/README.md, "Item codes"):
//
// - rods: the axial and torsional stresses (or strains) and their margins; the axial force and the torque;
// - Bar2: the stresses at the four recovery points of each end, the axial stress, the extremes and the margins;
//   the end moments, the shears, the axial force and the torque;
// - Beam2: the same quantities at end A and at end B (the item codes of end B are those of end A shifted by a fixed
//   step); the intermediate stations are refused (O_o_Solver writes the stations of SO, which do not follow them);
// - shells (centre format; the corner format has the grid point first, its items shifted by one, its corners
//   refused): the fibre distance, the in-plane components, the angle, the principal values and the von Mises value
//   at each fibre (or the membrane strains and the curvatures); the membrane forces, the moments, the shears;
// - ShearPanel4: the maximum and average shears and the margin; the corner and kick forces and the edge shears;
// - springs: the stress and the force; Bushing: the translational and rotational components;
// - composite plies: the ply stresses or strains (the lamina is ATTB); the failure index of the direct stresses and
//   that of the interlaminar shear (or their strength ratios);
// - strain energy: the energy, its percentage of the total and its density.
//
// Items the format defines as "von Mises or maximum shear" give the von Mises value the result tables hold: O_o_Solver
// writes the von Mises column whatever the describer, so STRESS(MAXS) does not turn them into the maximum shear.
[[nodiscard]] std::variant<ItemLocation, std::string> locateItem(ResponseType type, std::string_view card,
                                                                 std::int32_t itemCode);

// A standard Bar2 item in the station layout written under line loads. Shared axial/shear/torque items select
// end A; end-specific bending/stress items retain their end. Margins have no equivalent station column.
[[nodiscard]] std::optional<ItemLocation> barStationLocation(const ItemLocation& location);

// The FORCE item code of force column `component` of a member of a section of family `family` (output::SectionLoads:
// the components of its family, a beam's at end A then at end B) on an element of card `card`: the code whose location
// is that column at the centre, or at end A or B of a beam; 0 when none. A section's rows are then the weighted sums
// of these items (DesignProblem.to_model()'s DesignResp2, the closed second derivatives of the section loads).
[[nodiscard]] std::int32_t sectionForceItemCode(output::SectionFamily family, std::string_view card,
                                                std::size_t component);

} // namespace oos::design
