#include "oos/design/responses/stress_strain_force/ItemCodes.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>

namespace oos::design
{

namespace
{

// One item code of the input format and the column of O_o_Solver's result table that holds its quantity (the names of
// the families' outputColumns and layerOutputColumns, and of StiffnessRecovery's strain energy table).
struct ItemColumn
{
    std::int32_t item;
    std::string_view column;
};

// Rods (Rod2, Rod2InlineSection, Tube2): the stresses and strains share their column names.
constexpr std::array<ItemColumn, 4> ROD_STRESS_ITEMS{{{2, "axial"}, {3, "sma"}, {4, "torsion"}, {5, "smt"}}};
constexpr std::array<ItemColumn, 2> ROD_FORCE_ITEMS{{{2, "axial"}, {3, "torsion"}}};

// Bar2: the four recovery points of end A, the axial stress, the extremes of end A and the margin in tension; the
// points and the extremes of end B and the margin in compression.
constexpr std::array<ItemColumn, 15> BAR_STRESS_ITEMS{{{2, "s1a"},
                                                       {3, "s2a"},
                                                       {4, "s3a"},
                                                       {5, "s4a"},
                                                       {6, "axial"},
                                                       {7, "smaxa"},
                                                       {8, "smina"},
                                                       {9, "msTension"},
                                                       {10, "s1b"},
                                                       {11, "s2b"},
                                                       {12, "s3b"},
                                                       {13, "s4b"},
                                                       {14, "smaxb"},
                                                       {15, "sminb"},
                                                       {16, "msCompression"}}};
constexpr std::array<ItemColumn, 15> BAR_STRAIN_ITEMS{{{2, "e1a"},
                                                       {3, "e2a"},
                                                       {4, "e3a"},
                                                       {5, "e4a"},
                                                       {6, "axial"},
                                                       {7, "emaxa"},
                                                       {8, "emina"},
                                                       {9, "msTension"},
                                                       {10, "e1b"},
                                                       {11, "e2b"},
                                                       {12, "e3b"},
                                                       {13, "e4b"},
                                                       {14, "emaxb"},
                                                       {15, "eminb"},
                                                       {16, "msCompression"}}};
constexpr std::array<ItemColumn, 8> BAR_FORCE_ITEMS{{{2, "bendingMomentA1"},
                                                     {3, "bendingMomentA2"},
                                                     {4, "bendingMomentB1"},
                                                     {5, "bendingMomentB2"},
                                                     {6, "shear1"},
                                                     {7, "shear2"},
                                                     {8, "axial"},
                                                     {9, "torque"}}};

// Beam2 at end A (a station row: the stresses and strains share their column names). The items of station K are
// those of end A plus (K − 1)·10 for the stresses and strains, plus (K − 1)·9 for the forces; end B is K = 11.
constexpr std::array<ItemColumn, 8> BEAM_STRESS_ITEMS{{{4, "sxc"},
                                                       {5, "sxd"},
                                                       {6, "sxe"},
                                                       {7, "sxf"},
                                                       {8, "smax"},
                                                       {9, "smin"},
                                                       {10, "msTension"},
                                                       {11, "msCompression"}}};
constexpr std::array<ItemColumn, 7> BEAM_FORCE_ITEMS{{{4, "bendingMoment1"},
                                                      {5, "bendingMoment2"},
                                                      {6, "shear1"},
                                                      {7, "shear2"},
                                                      {8, "axialForce"},
                                                      {9, "totalTorque"},
                                                      {10, "warpingTorque"}}};
constexpr std::int32_t BEAM_STRESS_STATION_STEP = 10;
constexpr std::int32_t BEAM_FORCE_STATION_STEP = 9;
constexpr std::int32_t BEAM_END_B_STATION = 11;

// Shells in the centre format (ShellQuad4, ShellTria3), at fibre Z1 (or the membrane strains); the items of fibre Z2
// (or the curvatures) follow, SHELL_FIBRE_ITEM_STEP further. The corner format (ShellQuad8, ShellTria6,
// ShellQuad4Drilling, ShellTria3Drilling) puts the grid point first: its items are these plus CORNER_FORMAT_ITEM_SHIFT.
constexpr std::array<ItemColumn, 8> SHELL_STRESS_ITEMS{{{2, "fiberDistance"},
                                                        {3, "oxx"},
                                                        {4, "oyy"},
                                                        {5, "txy"},
                                                        {6, "angle"},
                                                        {7, "omax"},
                                                        {8, "omin"},
                                                        {9, "vonMises"}}};
constexpr std::array<ItemColumn, 8> SHELL_STRAIN_ITEMS{{{2, "fiberDistance"},
                                                        {3, "exx"},
                                                        {4, "eyy"},
                                                        {5, "exy"},
                                                        {6, "angle"},
                                                        {7, "emax"},
                                                        {8, "emin"},
                                                        {9, "vonMises"}}};
constexpr std::array<ItemColumn, 8> SHELL_FORCE_ITEMS{
    {{2, "nx"}, {3, "ny"}, {4, "nxy"}, {5, "mx"}, {6, "my"}, {7, "mxy"}, {8, "qx"}, {9, "qy"}}};
// The strain table in its membrane-and-curvature layout holds the curvature flag where the fibre distance is.
constexpr std::string_view SHELL_FIBRE_COLUMN = "fiberDistance";
constexpr std::string_view SHELL_CURVATURE_FLAG_COLUMN = "fiberCurvature";
constexpr std::int32_t SHELL_FIBRE_ITEM_STEP = 8;
constexpr std::int32_t CORNER_FORMAT_ITEM_SHIFT = 1;

// ShearPanel4: the stresses and strains share their column names; the forces in the order of the format (the
// corner forces, then the kick forces and the edge shears).
constexpr std::array<ItemColumn, 3> SHEAR_PANEL_STRESS_ITEMS{{{2, "maxShear"}, {3, "avgShear"}, {4, "margin"}}};
constexpr std::array<ItemColumn, 16> SHEAR_PANEL_FORCE_ITEMS{{{2, "force41"},
                                                              {3, "force21"},
                                                              {4, "force12"},
                                                              {5, "force32"},
                                                              {6, "force23"},
                                                              {7, "force43"},
                                                              {8, "force34"},
                                                              {9, "force14"},
                                                              {10, "kickForce1"},
                                                              {11, "shear12"},
                                                              {12, "kickForce2"},
                                                              {13, "shear23"},
                                                              {14, "kickForce3"},
                                                              {15, "shear34"},
                                                              {16, "kickForce4"},
                                                              {17, "shear41"}}};

// SpringDofProperty to SpringScalarInline (no strains) and Bushing (the stresses and strains share their column names).
constexpr std::array<ItemColumn, 1> SPRING_STRESS_ITEMS{{{2, "springStress"}}};
constexpr std::array<ItemColumn, 1> SPRING_FORCE_ITEMS{{{2, "springForce"}}};
constexpr std::array<ItemColumn, 6> BUSH_STRESS_ITEMS{
    {{2, "tx"}, {3, "ty"}, {4, "tz"}, {5, "rx"}, {6, "ry"}, {7, "rz"}}};
constexpr std::array<ItemColumn, 6> BUSH_FORCE_ITEMS{
    {{2, "fx"}, {3, "fy"}, {4, "fz"}, {5, "mx"}, {6, "my"}, {7, "mz"}}};
constexpr std::array<ItemColumn, 7> BUSH_1D_RESULT_ITEMS{
    {{2, "elementForce"}, {3, "axialDisplacement"}, {4, "axialVelocity"}, {5, "axialStress"},
     {6, "axialStrain"}, {7, "plasticStrain"}, {8, "isFailed"}}};
constexpr std::array<ItemColumn, 6> FASTENER_STRESS_ITEMS{
    {{2, "forceX"}, {3, "forceY"}, {4, "forceZ"}, {5, "momentX"}, {6, "momentY"}, {7, "momentZ"}}};
constexpr std::array<ItemColumn, 8> GAP_FORCE_ITEMS{
    {{2, "fx"}, {3, "sfy"}, {4, "sfz"}, {5, "u"}, {6, "v"}, {7, "w"}, {8, "sv"}, {9, "sw"}}};

// The plies of a laminated shell (item 2, the lamina number, is the ATTB of the response), and their failure indices.
constexpr std::array<ItemColumn, 9> PLY_STRESS_ITEMS{{{3, "o11"},
                                                      {4, "o22"},
                                                      {5, "t12"},
                                                      {6, "t1z"},
                                                      {7, "t2z"},
                                                      {8, "angle"},
                                                      {9, "major"},
                                                      {10, "minor"},
                                                      {11, "vonMises"}}};
constexpr std::array<ItemColumn, 9> PLY_STRAIN_ITEMS{{{3, "e11"},
                                                      {4, "e22"},
                                                      {5, "e12"},
                                                      {6, "e1z"},
                                                      {7, "e2z"},
                                                      {8, "angle"},
                                                      {9, "major"},
                                                      {10, "minor"},
                                                      {11, "vonMises"}}};
constexpr std::array<ItemColumn, 2> PLY_FAILURE_ITEMS{{{5, FAILURE_INDEX_COLUMN}, {7, BOND_FAILURE_INDEX_COLUMN}}};

// The strain energy of an element (StiffnessRecovery's strain energy table).
constexpr std::array<ItemColumn, 3> ENERGY_ITEMS{{{2, "strainEnergy"}, {3, "percent"}, {4, "strainEnergyDensity"}}};

enum class Family : std::uint8_t
{
    Rod,
    Bar2,
    Beam2,
    ShellCentreFormat, // ShellQuad4, ShellTria3
    ShellCornerFormat, // ShellQuad8, ShellTria6, ShellQuad4Drilling, ShellTria3Drilling
    ShearPanel4,
    Spring,
    Bushing,
    Bushing1D,
    Fastener,
    Gap2,
    Other
};

Family familyOf(std::string_view card)
{
    if (card == "Rod2" || card == "Rod2InlineSection" || card == "Tube2")
        return Family::Rod;
    if (card == "Bar2")
        return Family::Bar2;
    if (card == "Beam2")
        return Family::Beam2;
    if (card == "ShellQuad4" || card == "ShellTria3")
        return Family::ShellCentreFormat;
    if (card == "ShellQuad8" || card == "ShellTria6" || card == "ShellQuad4Drilling" || card == "ShellTria3Drilling")
        return Family::ShellCornerFormat;
    if (card == "ShearPanel4")
        return Family::ShearPanel4;
    if (card == "SpringDofProperty" || card == "SpringDofInline" || card == "SpringScalarProperty" || card == "SpringScalarInline")
        return Family::Spring;
    if (card == "Bushing")
        return Family::Bushing;
    if (card == "Bushing1D")
        return Family::Bushing1D;
    if (card == "Fastener")
        return Family::Fastener;
    if (card == "Gap2")
        return Family::Gap2;
    return Family::Other;
}

std::string tableKindOf(ResponseType type)
{
    switch (type)
    {
    case ResponseType::Stress:
    case ResponseType::CompositeStress:
        return "stress";
    case ResponseType::Strain:
    case ResponseType::CompositeStrain:
        return "strain";
    case ResponseType::Force:
        return "forces";
    case ResponseType::StrainEnergy:
        return "strainEnergy";
    case ResponseType::CompositeFailure:
    case ResponseType::CompositeStrengthRatio:
        return "failureIndices";
    default:
        return "";
    }
}

std::string noItem(ResponseType type, std::string_view card, std::int32_t itemCode, std::string_view expected)
{
    return base::concatenate({"expected an item code of ", responseTypeName(type), " on a ", card, " among ", expected,
                              ", found ", std::to_string(itemCode)});
}

// The entry of `items` for item code `itemCode` shifted back by `shift` (the item of a later fibre, station or format),
// or null.
const ItemColumn* findItem(std::span<const ItemColumn> items, std::int32_t itemCode, std::int32_t shift = 0)
{
    const auto found = std::find_if(items.begin(), items.end(),
                                    [&](const ItemColumn& entry) { return entry.item + shift == itemCode; });
    return found == items.end() ? nullptr : &*found;
}

ItemLocation locationOf(ResponseType type, const ItemColumn& entry)
{
    ItemLocation location;
    location.tableKind = tableKindOf(type);
    location.column = std::string(entry.column);
    return location;
}

// A location in a table of one row per place.
std::variant<ItemLocation, std::string> plainItem(ResponseType type, std::string_view card, std::int32_t itemCode,
                                                  std::span<const ItemColumn> items, std::string_view expected)
{
    const ItemColumn* entry = findItem(items, itemCode);
    if (entry == nullptr)
        return noItem(type, card, itemCode, expected);
    return locationOf(type, *entry);
}

// A shell's centre row: the items of fibre Z1 (layer 1), then those of fibre Z2 (layer 2); the forces have one row.
std::variant<ItemLocation, std::string> shellItem(ResponseType type, std::string_view card, std::int32_t itemCode,
                                                  bool isCornerFormat)
{
    const std::int32_t shift = isCornerFormat ? CORNER_FORMAT_ITEM_SHIFT : 0;
    if (type == ResponseType::Force)
    {
        const ItemColumn* entry = findItem(SHELL_FORCE_ITEMS, itemCode, shift);
        if (entry == nullptr)
            return noItem(type, card, itemCode, isCornerFormat ? "3 to 10 (centre forces)" : "2 to 9 (centre forces)");
        return locationOf(type, *entry);
    }
    const std::span<const ItemColumn> items =
        type == ResponseType::Strain ? std::span<const ItemColumn>(SHELL_STRAIN_ITEMS) : SHELL_STRESS_ITEMS;
    for (const std::int32_t layer : {1, 2})
    {
        const ItemColumn* entry = findItem(items, itemCode, shift + (layer - 1) * SHELL_FIBRE_ITEM_STEP);
        if (entry == nullptr)
            continue;
        ItemLocation location = locationOf(type, *entry);
        location.layer = layer;
        if (type == ResponseType::Strain && entry->column == SHELL_FIBRE_COLUMN)
            location.otherColumn = std::string(SHELL_CURVATURE_FLAG_COLUMN);
        return location;
    }
    return noItem(type, card, itemCode,
                  isCornerFormat ? "3 to 18 (centre, Z1 then Z2; corners refused)" : "2 to 17 (centre, Z1 then Z2)");
}

// A Beam2's station row at end A or end B; the intermediate stations are refused.
std::variant<ItemLocation, std::string> beam2Item(ResponseType type, std::string_view card, std::int32_t itemCode)
{
    const bool isForce = type == ResponseType::Force;
    const std::int32_t step = isForce ? BEAM_FORCE_STATION_STEP : BEAM_STRESS_STATION_STEP;
    const std::span<const ItemColumn> items =
        isForce ? std::span<const ItemColumn>(BEAM_FORCE_ITEMS) : BEAM_STRESS_ITEMS;
    const std::string_view expected =
        isForce ? "4 to 10 at end A, 94 to 100 at end B" : "4 to 11 at end A, 104 to 111 at end B";
    for (const auto& [station, place] : std::array<std::pair<std::int32_t, ItemPlace>, 2>{
             {{1, ItemPlace::EndA}, {BEAM_END_B_STATION, ItemPlace::EndB}}})
    {
        const ItemColumn* entry = findItem(items, itemCode, (station - 1) * step);
        if (entry == nullptr)
            continue;
        ItemLocation location = locationOf(type, *entry);
        location.place = place;
        return location;
    }
    return noItem(type, card, itemCode, expected);
}

// A ply row of a laminated shell: its stresses or strains, or its failure indices (or their strength ratios).
std::variant<ItemLocation, std::string> plyItem(ResponseType type, std::string_view card, std::int32_t itemCode)
{
    std::variant<ItemLocation, std::string> located;
    if (type == ResponseType::CompositeStress)
        located = plainItem(type, card, itemCode, PLY_STRESS_ITEMS, "3 to 11 (ply items)");
    else if (type == ResponseType::CompositeStrain)
        located = plainItem(type, card, itemCode, PLY_STRAIN_ITEMS, "3 to 11 (ply items)");
    else
        located = plainItem(type, card, itemCode, PLY_FAILURE_ITEMS, "5 (direct stresses) and 7 (interlaminar shear)");
    if (auto* location = std::get_if<ItemLocation>(&located))
    {
        location->isComposite = true;
        location->isStrengthRatio = type == ResponseType::CompositeStrengthRatio;
    }
    return located;
}

} // namespace

std::optional<ItemLocation> barStationLocation(const ItemLocation& source)
{
    ItemLocation result = source;
    result.place = ItemPlace::EndA;
    const auto& name = source.column;
    if (name.size() == 3 && (name.front() == 's' || name.front() == 'e') &&
        name[1] >= '1' && name[1] <= '4' && (name.back() == 'a' || name.back() == 'b'))
    {
        constexpr std::array<std::string_view, 4> POINTS{"sxc", "sxd", "sxe", "sxf"};
        result.column = POINTS[static_cast<std::size_t>(name[1] - '1')];
        result.place = name.back() == 'a' ? ItemPlace::EndA : ItemPlace::EndB;
        return result;
    }
    if (name == "bendingMomentA1" || name == "bendingMomentB1") result.column = "bendingMoment1";
    else if (name == "bendingMomentA2" || name == "bendingMomentB2") result.column = "bendingMoment2";
    else if (name == "smaxa" || name == "smaxb" || name == "emaxa" || name == "emaxb") result.column = "smax";
    else if (name == "smina" || name == "sminb" || name == "emina" || name == "eminb") result.column = "smin";
    else if (name != "axial" && name != "shear1" && name != "shear2" && name != "torque")
        return std::nullopt;
    if (name == "bendingMomentB1" || name == "bendingMomentB2" ||
        name == "smaxb" || name == "sminb" || name == "emaxb" || name == "eminb")
        result.place = ItemPlace::EndB;
    return result;
}

std::optional<std::size_t> ItemLocation::columnIn(std::span<const std::string> names) const
{
    for (std::size_t c = 0; c < names.size(); ++c)
        if (names[c] == column || (!otherColumn.empty() && names[c] == otherColumn))
            return c;
    return std::nullopt;
}

std::variant<ItemLocation, std::string> locateItem(ResponseType type, std::string_view card, std::int32_t itemCode)
{
    if (type == ResponseType::StrainEnergy)
        return plainItem(type, card, itemCode, ENERGY_ITEMS, "2 to 4");
    const Family family = familyOf(card);
    const bool isComposite = type == ResponseType::CompositeStress || type == ResponseType::CompositeStrain ||
                             type == ResponseType::CompositeFailure || type == ResponseType::CompositeStrengthRatio;
    if (isComposite)
    {
        if (family != Family::ShellCentreFormat && family != Family::ShellCornerFormat)
            return base::concatenate(
                {"expected a laminated shell (ShellQuad4, ShellTria3, ShellQuad8, ShellTria6, "
                 "ShellQuad4Drilling, ShellTria3Drilling) for ",
                 responseTypeName(type), ", found a ", card});
        return plyItem(type, card, itemCode);
    }
    const bool isForce = type == ResponseType::Force;
    const bool isStrain = type == ResponseType::Strain;
    switch (family)
    {
    case Family::Rod:
        return isForce ? plainItem(type, card, itemCode, ROD_FORCE_ITEMS, "2 and 3")
                       : plainItem(type, card, itemCode, ROD_STRESS_ITEMS, "2 to 5");
    case Family::Bar2:
        if (isForce)
            return plainItem(type, card, itemCode, BAR_FORCE_ITEMS, "2 to 9");
        return plainItem(type, card, itemCode, isStrain ? BAR_STRAIN_ITEMS : BAR_STRESS_ITEMS, "2 to 16");
    case Family::Beam2:
        return beam2Item(type, card, itemCode);
    case Family::ShellCentreFormat:
        return shellItem(type, card, itemCode, false);
    case Family::ShellCornerFormat:
        return shellItem(type, card, itemCode, true);
    case Family::ShearPanel4:
        return isForce ? plainItem(type, card, itemCode, SHEAR_PANEL_FORCE_ITEMS, "2 to 17")
                       : plainItem(type, card, itemCode, SHEAR_PANEL_STRESS_ITEMS, "2 to 4");
    case Family::Spring:
        if (isStrain)
            return base::concatenate({"expected an element with strains for ", responseTypeName(type), ", found a ",
                                      card, " (the scalar springs give stresses and forces only)"});
        return isForce ? plainItem(type, card, itemCode, SPRING_FORCE_ITEMS, "2")
                       : plainItem(type, card, itemCode, SPRING_STRESS_ITEMS, "2");
    case Family::Bushing:
        return plainItem(type, card, itemCode, isForce ? BUSH_FORCE_ITEMS : BUSH_STRESS_ITEMS, "2 to 7");
    case Family::Bushing1D:
        if (isForce)
            return "Bushing1D puts its element force in the stress and strain tables, not in a force table";
        return plainItem(type, card, itemCode, BUSH_1D_RESULT_ITEMS, "2 to 8");
    case Family::Fastener:
        if (isStrain)
            return "Fastener gives forces and stresses, with no strain table";
        return plainItem(type, card, itemCode, isForce ? BUSH_FORCE_ITEMS : FASTENER_STRESS_ITEMS, "2 to 7");
    case Family::Gap2:
        if (!isForce)
            return "Gap2 gives a force table only";
        return plainItem(type, card, itemCode, GAP_FORCE_ITEMS, "2 to 9");
    case Family::Other:
        break;
    }
    return base::concatenate({"expected an element whose ", responseTypeName(type),
                              " items O_o_Solver maps (rods, Bar2, Beam2, shells, ShearPanel4, SpringDofProperty to "
                              "SpringScalarInline, Bushing, Bushing1D, Fastener and Gap2), found a ",
                              card, "; constrain a supported element, or its results through an equation response"});
}

std::int32_t sectionForceItemCode(output::SectionFamily family, std::string_view card, std::size_t component)
{
    // The item codes of the FORCE responses mapped are below this (the largest, 111, a Beam2's end B).
    constexpr std::int32_t LARGEST_FORCE_ITEM_CODE = 111;
    constexpr std::int32_t FIRST_ITEM_CODE = 2;
    // The force columns of a Beam2 station, read at end A then at end B (output::SectionLoads).
    constexpr std::size_t BEAM_STATION_COLUMNS = 7;
    const std::span<const std::string_view> components = output::sectionComponents(family);
    if (component >= components.size())
        return 0;
    // The section's component names its force column; a beam's adds the end, A or B, to the column of the station.
    const bool isBeam = family == output::SectionFamily::Beam;
    const std::string_view column =
        isBeam ? components[component].substr(0, components[component].size() - 1) : components[component];
    const ItemPlace place = !isBeam                            ? ItemPlace::Centre
                            : component < BEAM_STATION_COLUMNS ? ItemPlace::EndA
                                                               : ItemPlace::EndB;
    for (std::int32_t code = FIRST_ITEM_CODE; code <= LARGEST_FORCE_ITEM_CODE; ++code)
    {
        const auto located = locateItem(ResponseType::Force, card, code);
        const ItemLocation* location = std::get_if<ItemLocation>(&located);
        if (location != nullptr && location->tableKind == "forces" && location->column == column &&
            location->place == place)
            return code;
    }
    return 0;
}

} // namespace oos::design
