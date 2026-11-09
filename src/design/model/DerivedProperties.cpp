#include "oos/design/model/DerivedProperties.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/input/tables/SectionTable.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace oos::design
{

namespace
{

const std::string PLIES_TABLE = "laminatePliesTable";
const std::string SUMMARY_TABLE = "laminateSummary";
const std::string SECTIONS_TABLE = "sectionsTable";
constexpr std::array<std::string_view, 7> SECTION_COLUMNS{"area", "i1", "i2", "i12", "j", "k1", "k2"};
constexpr std::array<std::string_view, 8> POINT_COLUMNS{"c1", "c2", "d1", "d2", "e1", "e2", "f1", "f2"};
constexpr std::array<std::string_view, 2> NEUTRAL_AXIS_COLUMNS{"n1", "n2"};

[[nodiscard]] bool isLaminateCard(const std::string& card)
{
    return card == "CompProp" || card == "CompPropGlobal";
}

// The column of a card's designed list: the ply thicknesses of a laminate, the dimensions of a section.
[[nodiscard]] std::string derivedColumnOf(const std::string& card)
{
    if (isLaminateCard(card))
        return "plyThickness";
    return card == "BarShapeProp" ? "dims" : "stationDims";
}

// A value as messages show it (6 significant digits).
[[nodiscard]] std::string numberText(double value)
{
    std::ostringstream text;
    text << std::setprecision(6) << value;
    return text.str();
}

// The stack sum of a laminate's thicknesses, in stack order (O_o_Reader's laminateSummary thickness).
[[nodiscard]] double stackThickness(std::span<const double> thicknesses)
{
    double total = 0.0;
    for (const double t : thicknesses)
        total += t;
    return total;
}

// The rows [first, last) of a table sorted by PID that hold `propId` (std::equal_range); every row when the PID is not
// found there, for a table out of order (the caller filters them).
// PERFORMANCE: a laminate's binding scanned every ply row of the model (30 000 CompProp of 16 plies: quadratic).
[[nodiscard]] std::pair<std::int64_t, std::int64_t> rowsOfPropId(std::span<const std::int32_t> propIds,
                                                                 std::int32_t propId)
{
    const auto range = std::equal_range(propIds.begin(), propIds.end(), propId);
    if (range.first != range.second)
        return {range.first - propIds.begin(), range.second - propIds.begin()};
    return {0, static_cast<std::int64_t>(propIds.size())};
}

} // namespace

bool DerivedProperties::bind(const input::ModelSnapshot& snapshot, std::size_t relation, const std::string& card,
                             std::int32_t propId, const std::string& table, const std::string& column, std::int64_t row,
                             std::int32_t item, std::string& problem)
{
    if (column != derivedColumnOf(card))
    {
        problem = base::concatenate(
            {"the field '", column, "' of ", card, " is not a design target: the elements read ", card,
             isLaminateCard(card) ? " through its ply stack, whose ply thicknesses (CompProp.plyThickness1, "
                                    "CompProp.plyThickness2 ...) are the designed fields"
                                  : " through its section properties, whose dimensions (BarShapeProp.dimension1, "
                                    "BarShapeProp.dimension2 ...) are the designed "
                                    "fields",
             "; design those"});
        return false;
    }
    const bool isLaminate = isLaminateCard(card);
    std::map<std::pair<std::string, std::int32_t>, std::size_t>& groups = isLaminate ? m_laminateOf : m_sectionOf;
    const std::pair<std::string, std::int32_t> key{card, propId};
    auto found = groups.find(key);
    if (found == groups.end())
    {
        const bool isBound = isLaminate ? bindLaminate(snapshot, card, propId, table, row, problem)
                                        : bindSection(snapshot, card, propId, table, column, row, problem);
        if (!isBound)
            return false;
        // A new group is the last of its kind (the section groups follow the laminates in groupOf's numbering).
        found = groups.emplace(key, isLaminate ? m_laminates.size() - 1 : m_sections.size() - 1).first;
    }
    const GroupRef group{isLaminate, found->second};
    if (isLaminate)
        m_laminates[group.index].members.push_back({relation, item});
    else
        m_sections[group.index].members.push_back({relation, item});
    // The relations come in increasing order (the design model's): kept sorted without sorting them again.
    const bool isInOrder = m_groupOfRelation.empty() || m_groupOfRelation.back().first < relation;
    m_groupOfRelation.emplace_back(relation, group);
    if (!isInOrder)
        std::sort(m_groupOfRelation.begin(), m_groupOfRelation.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    return true;
}

bool DerivedProperties::bindLaminate(const input::ModelSnapshot& snapshot, const std::string& card, std::int32_t propId,
                                     const std::string& table, std::int64_t row, std::string& problem)
{
    if (!snapshot.hasTable(input::TableRole::LaminatePlies))
    {
        problem = base::concatenate({"expected the plies of ", card, " ", std::to_string(propId),
                                     " in laminatePliesTable, found no such table; take the snapshot with O_o_Reader's "
                                     "laminate tables"});
        return false;
    }
    Laminate laminate;
    laminate.card = card;
    laminate.propId = propId;
    laminate.table = table;
    laminate.row = row;
    const base::ContractReader& cardTable = *snapshot.findTable(table);
    const std::span<const double> plies = cardTable.lists<double>("plyThickness")[row];
    laminate.plies.assign(plies.begin(), plies.end());
    const base::ContractReader& stack = snapshot.laminatePlies();
    const std::span<const std::int32_t> propIds = stack.values<std::int32_t>("propId");
    const base::DictionaryView types = stack.dictionaryTexts("type");
    const std::span<const std::int32_t> numbers = stack.values<std::int32_t>("ply");
    const base::BooleanView mirrored = stack.booleans("isMirrored");
    const std::span<const double> bottoms = stack.values<double>("zBottom");
    // The PID's rows: laminatePliesTable is sorted by PID, then ply (the data contract; elements::Laminate reads it
    // so), a table out of order scanned.
    const auto [first, last] = rowsOfPropId(propIds, propId);
    for (std::int64_t r = first; r < last; ++r)
    {
        if (propIds[static_cast<std::size_t>(r)] != propId || types[r] != card)
            continue;
        laminate.stackRows.push_back(r);
        // Ply k of the stack (from 1): the card ply k, or, in the mirrored half of LAM = SYM, ply 2n + 1 − k.
        const auto number = static_cast<std::size_t>(numbers[static_cast<std::size_t>(r)]);
        const std::size_t cardPly = mirrored[r] ? 2 * laminate.plies.size() - number : number - 1;
        if (number == 0 || cardPly >= laminate.plies.size())
        {
            problem = base::concatenate({"expected the plies of ", card, " ", std::to_string(propId),
                                         " in laminatePliesTable to number its card's plies, found ply ",
                                         std::to_string(number), "; the snapshot does not come from one model"});
            return false;
        }
        laminate.plyOfStack.push_back(cardPly);
    }
    if (laminate.stackRows.empty())
    {
        problem = base::concatenate(
            {"expected the plies of ", card, " ", std::to_string(propId), " in laminatePliesTable, found none"});
        return false;
    }
    std::vector<double> stackThicknesses;
    for (const std::size_t ply : laminate.plyOfStack)
        stackThicknesses.push_back(laminate.plies[ply]);
    const double total = stackThickness(stackThicknesses);
    const double z0 = bottoms[static_cast<std::size_t>(laminate.stackRows.front())];
    // A blank Z0 of a CompProp is −T/2 and follows the designed thickness (as O_o_Reader's model.design.apply). From
    // data contract 54 the card says whether Z0 was given (z0_given: a given Z0 stays, even equal to −T/2); before it,
    // a Z0 equal to −T/2 to the bit is taken as blank.
    if (cardTable.hasColumn("bottomOffsetGiven") && cardTable.columnType("bottomOffsetGiven") == "uint8")
        laminate.isZ0Blank = cardTable.values<std::uint8_t>(
                                 "bottomOffsetGiven", base::NullPolicy::Allow)[static_cast<std::size_t>(row)] == 0;
    else
        laminate.isZ0Blank = z0 == -0.5 * total;
    laminate.z0 = z0;
    if (snapshot.hasTable(input::TableRole::LaminateSummary))
    {
        const base::ContractReader& summary = snapshot.laminateSummary();
        const std::span<const std::int32_t> summaryPropIds = summary.values<std::int32_t>("propId");
        const base::DictionaryView summaryTypes = summary.dictionaryTexts("type");
        const auto [first, last] = rowsOfPropId(summaryPropIds, propId);
        for (std::int64_t r = first; r < last; ++r)
            if (summaryPropIds[static_cast<std::size_t>(r)] == propId && summaryTypes[r] == card)
                laminate.summaryRow = r;
    }
    m_laminates.push_back(std::move(laminate));
    return true;
}

bool DerivedProperties::bindSection(const input::ModelSnapshot& snapshot, const std::string& card, std::int32_t propId,
                                    const std::string& table, const std::string& column, std::int64_t row,
                                    std::string& problem)
{
    if (!snapshot.hasTable(input::TableRole::Sections))
    {
        problem = base::concatenate({"expected the section properties of ", card, " ", std::to_string(propId),
                                     " in sectionsTable, found no such table; take the snapshot with O_o_Reader's "
                                     "sectionsTable"});
        return false;
    }
    Section section;
    section.card = card;
    section.propId = propId;
    section.table = table;
    section.column = column;
    section.row = row;
    const base::ContractReader& reader = *snapshot.findTable(table);
    section.type = std::string(reader.texts("kind")[row]);
    const std::span<const double> dimensions = reader.lists<double>(column)[row];
    section.dimensions.assign(dimensions.begin(), dimensions.end());
    const input::SectionTable& sections = snapshot.sections();
    const std::optional<input::SectionRows> rows = sections.findProperty(propId);
    if (!rows || rows->count() == 0)
    {
        problem = base::concatenate({"expected the section properties of ", card, " ", std::to_string(propId),
                                     " in sectionsTable, found none"});
        return false;
    }
    for (std::int64_t r = rows->first; r < rows->last; ++r)
        section.stationRows.push_back(r);
    const std::size_t stations = section.stationRows.size();
    if (section.dimensions.empty() || section.dimensions.size() % stations != 0)
    {
        problem = base::concatenate({"expected the dimensions of ", card, " ", std::to_string(propId), " for its ",
                                     std::to_string(stations), " stations of sectionsTable, found ",
                                     std::to_string(section.dimensions.size()), " values"});
        return false;
    }
    section.dimensionCount = section.dimensions.size() / stations;
    m_hasNeutralAxisColumns = sections.reader().hasColumn("n1");
    m_sections.push_back(std::move(section));
    return true;
}

std::optional<std::size_t> DerivedProperties::groupOf(std::size_t relation) const
{
    const auto found = std::lower_bound(m_groupOfRelation.begin(), m_groupOfRelation.end(), relation,
                                        [](const auto& entry, std::size_t key) { return entry.first < key; });
    if (found == m_groupOfRelation.end() || found->first != relation)
        return std::nullopt;
    return found->second.isLaminate ? found->second.index : m_laminates.size() + found->second.index;
}

void DerivedProperties::addLaminate(const Laminate& laminate, input::ColumnOverlay& overlay,
                                    std::span<const double> fields) const
{
    std::vector<double> plies = laminate.plies;
    for (const Member& member : laminate.members)
        plies[static_cast<std::size_t>(member.item)] = fields[member.relation];
    for (std::size_t k = 0; k < plies.size(); ++k)
        if (!(plies[k] > 0.0))
            throw base::SolverError(
                base::concatenate({"the design point gives ply ", std::to_string(k + 1), " of ", laminate.card, " ",
                                   std::to_string(laminate.propId), " the thickness ", numberText(plies[k]),
                                   ", which is not positive; keep the ply thicknesses positive "
                                   "(DesignPropRelation1.minValue of the relation, the bounds of the "
                                   "variable)"}));
    std::vector<double> thicknesses;
    thicknesses.reserve(laminate.plyOfStack.size());
    for (const std::size_t ply : laminate.plyOfStack)
        thicknesses.push_back(plies[ply]);
    const double total = stackThickness(thicknesses);
    double z = laminate.isZ0Blank ? -0.5 * total : laminate.z0;
    const double z0 = z;
    for (std::size_t k = 0; k < thicknesses.size(); ++k)
    {
        const std::int64_t row = laminate.stackRows[k];
        overlay.setValue(PLIES_TABLE, "thickness", {row, input::CellValue::SCALAR_ITEM, thicknesses[k]});
        overlay.setValue(PLIES_TABLE, "zBottom", {row, input::CellValue::SCALAR_ITEM, z});
        z = z + thicknesses[k];
        overlay.setValue(PLIES_TABLE, "zTop", {row, input::CellValue::SCALAR_ITEM, z});
    }
    if (laminate.summaryRow >= 0)
    {
        overlay.setValue(SUMMARY_TABLE, "thickness", {laminate.summaryRow, input::CellValue::SCALAR_ITEM, total});
        overlay.setValue(SUMMARY_TABLE, "z0", {laminate.summaryRow, input::CellValue::SCALAR_ITEM, z0});
    }
    if (laminate.isZ0Blank)
        overlay.setValue(laminate.table, "bottomOffset", {laminate.row, input::CellValue::SCALAR_ITEM, z0});
}

void DerivedProperties::stationShapes(std::span<const double> fields, std::optional<std::size_t> onlyGroup,
                                      std::vector<SectionShape>& shapes,
                                      std::vector<std::pair<std::size_t, std::size_t>>& stations) const
{
    for (std::size_t s = 0; s < m_sections.size(); ++s)
    {
        if (onlyGroup && *onlyGroup != m_laminates.size() + s)
            continue;
        const Section& section = m_sections[s];
        std::vector<double> dimensions = section.dimensions;
        for (const Member& member : section.members)
            dimensions[static_cast<std::size_t>(member.item)] = fields[member.relation];
        for (std::size_t station = 0; station < section.stationRows.size(); ++station)
        {
            const auto first = static_cast<std::ptrdiff_t>(station * section.dimensionCount);
            shapes.push_back(
                {section.type,
                 std::vector<double>(dimensions.begin() + first,
                                     dimensions.begin() + first + static_cast<std::ptrdiff_t>(section.dimensionCount)),
                 {}});
            stations.emplace_back(s, station);
        }
    }
}

namespace
{

std::vector<SectionValues> libraryValues(const SectionLibrary* library, std::span<const SectionShape> shapes,
                                         bool gradient)
{
    if (library == nullptr)
        throw std::invalid_argument(
            "DerivedProperties: a designed BarShapeProp or BeamShapeProp needs the section library (SectionLibrary: "
            "O_o_Reader's "
            "formulas, which the caller gives); a bug of O_o_Solver when the design came from the caller: report it");
    std::vector<SectionValues> values = library->properties(shapes, gradient);
    if (values.size() != shapes.size())
        throw std::invalid_argument(
            base::concatenate({"DerivedProperties: the section library gave ", std::to_string(values.size()),
                               " sections for ", std::to_string(shapes.size()),
                               " asked; a SectionLibrary gives one per shape (a bug of O_o_Solver: report it)"}));
    for (std::size_t k = 0; gradient && k < shapes.size(); ++k)
        if (values[k].gradient.size() != SECTION_VALUE_COUNT * shapes[k].dimensions.size())
            throw std::invalid_argument(
                "DerivedProperties: the section library gave no derivatives where they were asked (a bug of "
                "O_o_Solver: report it)");
    return values;
}

std::array<double, SECTION_VALUE_COUNT> flatten(const SectionValues& value)
{
    std::array<double, SECTION_VALUE_COUNT> flat{value.area, value.i1, value.i2, value.i12,
                                                 value.j,    value.k1, value.k2};
    std::copy(value.points.begin(), value.points.end(), flat.begin() + 7);
    flat[15] = value.neutralAxis[0];
    flat[16] = value.neutralAxis[1];
    return flat;
}

} // namespace

void DerivedProperties::writeSection(input::ColumnOverlay& overlay, std::size_t s, std::size_t station,
                                     const SectionValues& value, std::span<const double> dimensions) const
{
    const Section& section = m_sections[s];
    if (!std::isfinite(value.area) || !(value.area > 0.0))
    {
        std::string text;
        for (const double d : dimensions)
            text.append(text.empty() ? "" : ", ").append(numberText(d));
        throw base::SolverError(base::concatenate(
            {"the design point gives ", section.card, " ", std::to_string(section.propId), " (", section.type,
             section.card == "BeamShapeProp" ? ", station " + std::to_string(station + 1) : std::string(),
             ") the dimensions ", text,
             ", for which the section library gives no section; keep the dimensions within the geometry of the "
             "section (DesignPropRelation1.minValue and DesignPropRelation1.maxValue of the relations, "
             "the bounds of the variables)"}));
    }
    const std::int64_t row = section.stationRows[station];
    const std::array<double, SECTION_COLUMNS.size()> scalars{value.area, value.i1, value.i2, value.i12,
                                                             value.j,    value.k1, value.k2};
    for (std::size_t c = 0; c < SECTION_COLUMNS.size(); ++c)
        overlay.setValue(SECTIONS_TABLE, std::string(SECTION_COLUMNS[c]),
                         {row, input::CellValue::SCALAR_ITEM, scalars[c]});
    for (std::size_t c = 0; c < POINT_COLUMNS.size(); ++c)
        overlay.setValue(SECTIONS_TABLE, std::string(POINT_COLUMNS[c]),
                         {row, input::CellValue::SCALAR_ITEM, value.points[c]});
    // A BeamShapeProp's neutral axis from its shear centre, when the table has the columns (a BarShapeProp's axis is
    // its neutral axis: its rows keep 0).
    if (section.card == "BeamShapeProp" && m_hasNeutralAxisColumns)
        for (std::size_t c = 0; c < NEUTRAL_AXIS_COLUMNS.size(); ++c)
            overlay.setValue(SECTIONS_TABLE, std::string(NEUTRAL_AXIS_COLUMNS[c]),
                             {row, input::CellValue::SCALAR_ITEM, neutralAxisOffset(value.neutralAxis[c], value.area)});
}

void DerivedProperties::addCells(input::ColumnOverlay& overlay, std::span<const double> fields,
                                 std::optional<std::size_t> onlyGroup, const SectionLibrary* library,
                                 const SectionLinearisation* linear, std::span<const double> reference) const
{
    for (std::size_t g = 0; g < m_laminates.size(); ++g)
        if (!onlyGroup || *onlyGroup == g)
            addLaminate(m_laminates[g], overlay, fields);
    // The sections: every station of the groups asked, in one call of the library, or linearised.
    std::vector<SectionShape> shapes;
    std::vector<std::pair<std::size_t, std::size_t>> stations; // (section, station) of each shape
    stationShapes(fields, onlyGroup, shapes, stations);
    if (shapes.empty())
        return;
    if (linear != nullptr)
    {
        for (std::size_t k = 0; k < shapes.size(); ++k)
        {
            const auto [s, station] = stations[k];
            const std::vector<double>& origin = linear->dimensions.at(s).at(station);
            const SectionValues& at = linear->values.at(s).at(station);
            const std::size_t count = origin.size();
            std::array<double, SECTION_VALUE_COUNT> flat = flatten(at);
            for (std::size_t v = 0; v < SECTION_VALUE_COUNT; ++v)
                for (std::size_t d = 0; d < count; ++d)
                    if (const double delta = shapes[k].dimensions[d] - origin[d]; delta != 0.0)
                        flat[v] += at.gradient[v * count + d] * delta;
            SectionValues value;
            value.area = flat[0];
            value.i1 = flat[1];
            value.i2 = flat[2];
            value.i12 = flat[3];
            value.j = flat[4];
            value.k1 = flat[5];
            value.k2 = flat[6];
            std::copy(flat.begin() + 7, flat.begin() + 15, value.points.begin());
            value.neutralAxis = {flat[15], flat[16]};
            writeSection(overlay, s, station, value, shapes[k].dimensions);
        }
        return;
    }
    if (!reference.empty())
    {
        std::vector<SectionShape> origins;
        std::vector<std::pair<std::size_t, std::size_t>> unused;
        stationShapes(reference, onlyGroup, origins, unused);
        for (std::size_t k = 0; k < shapes.size(); ++k)
            shapes[k].reference = origins[k].dimensions;
    }
    const std::vector<SectionValues> values = libraryValues(library, shapes, false);
    for (std::size_t k = 0; k < shapes.size(); ++k)
        writeSection(overlay, stations[k].first, stations[k].second, values[k], shapes[k].dimensions);
}

SectionLinearisation DerivedProperties::linearise(std::span<const double> fields, const SectionLibrary* library,
                                                  std::span<const double> reference,
                                                  std::span<const std::uint8_t> isGroupWanted) const
{
    SectionLinearisation linear;
    linear.dimensions.resize(m_sections.size());
    linear.values.resize(m_sections.size());
    std::vector<SectionShape> shapes;
    std::vector<std::pair<std::size_t, std::size_t>> stations;
    stationShapes(fields, std::nullopt, shapes, stations);
    // The meshes of the point's own dimensions: the values are those of the analysis, to the bit (reference = d0
    // is O_o_Reader's default); or those of `reference`.
    if (reference.empty())
        for (SectionShape& shape : shapes)
            shape.reference = shape.dimensions;
    else
    {
        std::vector<SectionShape> origins;
        std::vector<std::pair<std::size_t, std::size_t>> unused;
        stationShapes(reference, std::nullopt, origins, unused);
        for (std::size_t k = 0; k < shapes.size(); ++k)
            shapes[k].reference = origins[k].dimensions;
    }
    if (!isGroupWanted.empty())
    {
        // Only the stations of the sections wanted.
        std::size_t kept = 0;
        for (std::size_t k = 0; k < shapes.size(); ++k)
        {
            const std::size_t group = m_laminates.size() + stations[k].first;
            if (group >= isGroupWanted.size() || isGroupWanted[group] == 0)
                continue;
            if (kept != k) // never a self-move: it would empty the shape
            {
                shapes[kept] = std::move(shapes[k]);
                stations[kept] = stations[k];
            }
            ++kept;
        }
        shapes.resize(kept);
        stations.resize(kept);
    }
    if (shapes.empty())
        return linear;
    std::vector<SectionValues> values = libraryValues(library, shapes, true);
    for (std::size_t k = 0; k < shapes.size(); ++k)
    {
        const auto [s, station] = stations[k];
        linear.dimensions[s].push_back(shapes[k].dimensions);
        linear.values[s].push_back(std::move(values[k]));
    }
    return linear;
}

} // namespace oos::design
