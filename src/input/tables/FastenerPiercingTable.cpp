#include "oos/input/tables/FastenerPiercingTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <optional>
#include <string>
#include <utility>

namespace oos::input
{

namespace
{

constexpr int SIDE_A = 0;
constexpr int SIDE_B = 1;
constexpr int UNKNOWN = -1;

constexpr std::string_view WHAT_TO_DO = "; take the snapshot again with a O_o_Reader that follows the data contract";

int sideCode(std::string_view side)
{
    if (side == "A")
        return SIDE_A;
    if (side == "B")
        return SIDE_B;
    return UNKNOWN;
}

std::string sideText(std::int32_t elementId, std::string_view side)
{
    return base::concatenate({"element ", std::to_string(elementId), " side ", side});
}

} // namespace

FastenerPiercingTable::FastenerPiercingTable(base::ImportedTable piercings)
    : m_reader(std::move(piercings), contractOf(TableRole::FastenerPiercings))
{
    constexpr base::NullPolicy ALLOW = base::NullPolicy::Allow;
    const std::int64_t sourceRows = m_reader.rowCount();
    const std::span<const std::int32_t> ids = m_reader.values<std::int32_t>("elementId");
    const base::DictionaryView cards = m_reader.dictionaryTexts("card");
    const base::DictionaryView forms = m_reader.dictionaryTexts("form");
    const base::DictionaryView sides = m_reader.dictionaryTexts("side");
    const std::span<const std::int32_t> pointGrids = m_reader.values<std::int32_t>("pointGrid", ALLOW);
    const base::ValidityView hasPointGrid = m_reader.validity("pointGrid");
    const std::array<std::span<const double>, 3> coordinates{
        m_reader.values<double>("x", ALLOW), m_reader.values<double>("y", ALLOW), m_reader.values<double>("z", ALLOW)};
    const base::ValidityView isFound = m_reader.validity("x");
    const std::span<const std::int32_t> elements = m_reader.values<std::int32_t>("element", ALLOW);
    const base::ValidityView hasElement = m_reader.validity("element");
    const base::ListView<std::int32_t> nodeIds = m_reader.lists<std::int32_t>("nodeIds", ALLOW);
    const base::ListView<double> shapeValues = m_reader.lists<double>("shapeValues", ALLOW);
    const base::DictionaryView fields = m_reader.dictionaryTexts("field", ALLOW);
    const base::ValidityView hasField = m_reader.validity("field");
    const base::Utf8View problems = m_reader.texts("problem", ALLOW);
    const base::ValidityView hasProblem = m_reader.validity("problem");

    // The rows by (elementId, side), checked.
    std::vector<std::int64_t> order; // bounded by the rows of the table
    for (std::int64_t row = 0; row < sourceRows; ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        if (sideCode(sides[row]) == UNKNOWN)
            throw base::ContractError(
                base::concatenate({"fastenerPiercings: expected the side A or B, found '", sides[row], "' for element ",
                                   std::to_string(ids[r]), WHAT_TO_DO}));
        if (isFound.isValid(row) && (nodeIds[row].empty() || nodeIds[row].size() != shapeValues[row].size()))
            throw base::ContractError(base::concatenate(
                {"fastenerPiercings: expected one shape function per grid point of the pierced element, found ",
                 std::to_string(nodeIds[row].size()), " grid points and ", std::to_string(shapeValues[row].size()),
                 " shape functions for ", sideText(ids[r], sides[row]), WHAT_TO_DO}));
        order.push_back(row);
    }
    const auto key = [&](std::int64_t row)
    { return std::make_pair(ids[static_cast<std::size_t>(row)], sideCode(sides[row])); };
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) { return key(a) < key(b); });
    for (std::size_t k = 1; k < order.size(); ++k)
    {
        if (key(order[k - 1]) == key(order[k]))
            throw base::ContractError(base::concatenate(
                {"fastenerPiercings: expected one row per fastener side, found ",
                 sideText(ids[static_cast<std::size_t>(order[k])], sides[order[k]]), " twice", WHAT_TO_DO}));
    }
    const auto cardIndex = [&](std::string_view card) -> std::uint8_t
    {
        for (std::size_t index = 0; index < m_cardNames.size(); ++index)
        {
            if (m_cardNames[index] == card)
                return static_cast<std::uint8_t>(index);
        }
        m_cardNames.emplace_back(card);
        return static_cast<std::uint8_t>(m_cardNames.size() - 1);
    };

    m_offsets.push_back(0);
    std::vector<bool> isGiven; // bounded by the sides
    for (const std::int64_t row : order)
    {
        const auto r = static_cast<std::size_t>(row);
        const bool isSideA = sideCode(sides[row]) == SIDE_A;
        if (!isFound.isValid(row))
        {
            // A side the reader did not place: the element reports its field and problem.
            SideFailure failure;
            failure.elementId = ids[r];
            failure.isSideA = isSideA;
            failure.field = hasField.isValid(row) ? std::string(fields[row]) : std::string(isSideA ? "nodeA" : "nodeB");
            failure.message = hasProblem.isValid(row) ? std::string(problems[row])
                                                      : std::string("the reader gives no piercing point and no reason");
            m_failures.push_back(std::move(failure));
            continue;
        }
        const std::int32_t pointGrid = hasPointGrid.isValid(row) ? pointGrids[r] : 0;
        m_fastenerIds.push_back(ids[r]);
        m_cardIndices.push_back(cardIndex(cards[row]));
        m_isSideA.push_back(isSideA ? 1 : 0);
        // The cards that tie a fastener over its cross-section: every Fastener, and the Weld of the two patch forms.
        const bool isOverSection = cards[row] == "Fastener" || forms[row] == "PartPatch" || forms[row] == "ElemPatch";
        m_hasCorners.push_back(isOverSection ? 1 : 0);
        m_pointGrids.push_back(pointGrid);
        isGiven.push_back(pointGrid != 0);
        for (std::size_t c = 0; c < 3; ++c)
            m_piercingPoints.push_back(coordinates[c][r]);
        m_elementIds.push_back(hasElement.isValid(row) ? elements[r] : 0);
        const std::span<const std::int32_t> grids = nodeIds[row];
        const std::span<const double> values = shapeValues[row];
        m_grids.insert(m_grids.end(), grids.begin(), grids.end());
        m_weights.insert(m_weights.end(), values.begin(), values.end());
        m_offsets.push_back(static_cast<std::int64_t>(m_grids.size()));
    }
    m_patchOffsets = m_offsets;
    m_patchGrids = m_grids;
    m_patchWeights = m_weights;
    m_tieOffsets = m_offsets;
    m_tieGrids = m_grids;
    m_cornerFootprint = FootprintSamples(rowCount());
    m_meanFootprint = FootprintSamples(rowCount());
    // The validity bitmap of the point grids, least significant bit first (Arrow's layout, read by ValidityView).
    m_pointGridBits.assign((isGiven.size() + 7) / 8 + 1, 0);
    for (std::size_t row = 0; row < isGiven.size(); ++row)
    {
        if (isGiven[row])
            m_pointGridBits[row / 8] = static_cast<std::uint8_t>(m_pointGridBits[row / 8] | (1U << (row % 8)));
    }
}

namespace
{

// The diameters of the elements of card `card` ("Fastener" or "Weld"), by element ID, sorted: the `diameter` of the
// property of each (FastenerProp, WeldProp), where it is above 0. An element whose property the model does not
// have has no entry.
std::vector<std::pair<std::int32_t, double>> diametersOf(std::string_view card, std::span<const ElementBatch> batches,
                                                         std::span<const CardTable> properties)
{
    std::vector<std::pair<std::int32_t, double>> diameters; // bounded by the elements of the card
    const auto batch = std::find_if(batches.begin(), batches.end(),
                                    [&](const ElementBatch& candidate) { return candidate.card() == card; });
    const std::string_view propertyCard = card == "Fastener" ? "FastenerProp" : "WeldProp";
    const auto table = std::find_if(properties.begin(), properties.end(),
                                    [&](const CardTable& candidate) { return candidate.card() == propertyCard; });
    if (batch == batches.end() || table == properties.end() || !batch->hasProperty() ||
        !table->reader().hasColumn("diameter"))
        return diameters;
    // The rows of the properties by ID.
    const std::span<const std::int32_t> propertyIds = table->ids();
    std::vector<std::pair<std::int32_t, std::size_t>> propertyRows; // bounded by the properties of the card
    for (std::size_t row = 0; row < propertyIds.size(); ++row)
        propertyRows.emplace_back(propertyIds[row], row);
    std::sort(propertyRows.begin(), propertyRows.end());
    const std::span<const double> values = table->reader().values<double>("diameter");
    const std::span<const std::int32_t> elementIds = batch->ids();
    const std::span<const std::int32_t> elementProperties = batch->propertyIds();
    for (std::size_t row = 0; row < elementIds.size(); ++row)
    {
        const auto property = std::lower_bound(propertyRows.begin(), propertyRows.end(),
                                               std::pair<std::int32_t, std::size_t>{elementProperties[row], 0});
        if (property == propertyRows.end() || property->first != elementProperties[row])
            continue;
        const double diameter = values[property->second];
        if (std::isfinite(diameter) && diameter > 0.0)
            diameters.emplace_back(elementIds[row], diameter);
    }
    std::sort(diameters.begin(), diameters.end());
    return diameters;
}

} // namespace

void FastenerPiercingTable::buildPatches(const NodeTable& nodes, std::span<const ElementBatch> batches,
                                         std::span<const CardTable> properties)
{
    // The rows on a shell element (not on a face of grid points), and among them those on a grid point or a side of
    // their element: the others keep the grid points of the pierced element.
    const auto isOnShell = [&](std::int64_t row) { return m_elementIds[static_cast<std::size_t>(row)] != 0; };
    const auto isOnBoundary = [&](std::int64_t row)
    {
        const std::span<const double> values = weights()[row];
        return isOnShell(row) && std::any_of(values.begin(), values.end(), [](double value)
                                             { return std::abs(value) <= PIERCING_ZERO_SHAPE_VALUE; });
    };
    bool hasShellRow = false;
    for (std::int64_t row = 0; row < rowCount() && !hasShellRow; ++row)
        hasShellRow = isOnShell(row);
    if (!hasShellRow)
        return;

    // The diameter of the fastener of a row: that of the property of its card; 0 when the model does not have it.
    const std::vector<std::pair<std::int32_t, double>> fastenerDiameters = diametersOf("Fastener", batches, properties);
    const std::vector<std::pair<std::int32_t, double>> weldDiameters = diametersOf("Weld", batches, properties);
    const auto diameterOf = [&](std::int64_t row)
    {
        const auto& diameters = card(row) == "Fastener" ? fastenerDiameters : weldDiameters;
        const std::int32_t id = m_fastenerIds[static_cast<std::size_t>(row)];
        const auto found =
            std::lower_bound(diameters.begin(), diameters.end(), std::pair<std::int32_t, double>{id, 0.0});
        return found != diameters.end() && found->first == id ? found->second : 0.0;
    };

    const ShellMeshIndex mesh(nodes, batches);
    const double cosine = std::cos(PIERCING_PATCH_ANGLE_DEGREES * std::numbers::pi / 180.0);
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> patchGrids; // bounded by the grid points of the shells around the piercing points
    std::vector<double> patchWeights;
    FootprintSamples corners;
    FootprintSamples mean;
    for (std::int64_t row = 0; row < rowCount(); ++row)
    {
        const std::span<const std::int32_t> ids = grids()[row];
        const std::span<const double> values = weights()[row];
        patchGrids.insert(patchGrids.end(), ids.begin(), ids.end());
        patchWeights.insert(patchWeights.end(), values.begin(), values.end());
        const std::size_t first = static_cast<std::size_t>(offsets.back());
        const std::optional<base::Vec3> normal = isOnShell(row) ? mesh.normalOf(ids) : std::nullopt;
        if (normal && isOnBoundary(row))
        {
            // The grid points the piercing point lies on: those whose shape function is not zero.
            std::vector<std::int32_t> held; // at most the grid points of one element
            for (std::size_t k = 0; k < ids.size(); ++k)
                if (std::abs(values[k]) > PIERCING_ZERO_SHAPE_VALUE)
                    held.push_back(ids[k]);
            // The shell elements at the first of them that hold them all and continue the sheet.
            mesh.forEachShellAt(held.front(), [&](const ShellRef& shell)
            {
                const std::span<const std::int32_t> others = shell.grids;
                const auto isHeldByOther = [&](std::int32_t id)
                { return std::find(others.begin(), others.end(), id) != others.end(); };
                const std::optional<base::Vec3> other = mesh.normalOf(others);
                if (!std::all_of(held.begin(), held.end(), isHeldByOther) || !other ||
                    std::abs(base::dot(*normal, *other)) < cosine)
                    return;
                for (const std::int32_t id : others)
                {
                    if (id == 0 || !mesh.positionOf(id) ||
                        std::find(patchGrids.begin() + static_cast<std::ptrdiff_t>(first), patchGrids.end(), id) !=
                            patchGrids.end())
                        continue;
                    patchGrids.push_back(id);
                    patchWeights.push_back(0.0);
                }
            });
        }
        offsets.push_back(static_cast<std::int64_t>(patchGrids.size()));
        const auto r = static_cast<std::size_t>(row);
        appendFootprint(mesh, m_elementIds[r], values, isOnShell(row) ? diameterOf(row) : 0.0, m_hasCorners[r] != 0,
                        corners, mean);
    }
    m_patchOffsets = std::move(offsets);
    m_patchGrids = std::move(patchGrids);
    m_patchWeights = std::move(patchWeights);
    m_cornerFootprint = std::move(corners);
    m_meanFootprint = std::move(mean);

    // The grid points of the tie of each row: those under its corners, or those of its patch.
    m_tieOffsets.assign(1, 0);
    m_tieGrids.clear();
    for (std::int64_t row = 0; row < rowCount(); ++row)
    {
        const auto first = static_cast<std::ptrdiff_t>(m_tieGrids.size());
        const auto addGrid = [&](std::int32_t id)
        {
            if (std::find(m_tieGrids.begin() + first, m_tieGrids.end(), id) == m_tieGrids.end())
                m_tieGrids.push_back(id);
        };
        for (std::int64_t k = 0; k < m_cornerFootprint.count(row); ++k)
            for (const std::int32_t id : m_cornerFootprint.at(row, k).grids)
                addGrid(id);
        if (m_cornerFootprint.count(row) == 0)
            for (const std::int32_t id : this->patchGrids()[row])
                addGrid(id);
        m_tieOffsets.push_back(static_cast<std::int64_t>(m_tieGrids.size()));
    }
}

std::string_view FastenerPiercingTable::card(std::int64_t row) const
{
    return m_cardNames[m_cardIndices[static_cast<std::size_t>(row)]];
}

std::optional<FastenerPiercingRows> FastenerPiercingTable::findFastener(std::int32_t elementId) const
{
    const auto [first, last] = std::equal_range(m_fastenerIds.begin(), m_fastenerIds.end(), elementId);
    if (first == last)
        return std::nullopt;
    const std::int64_t begin = first - m_fastenerIds.begin();
    const std::int64_t end = last - m_fastenerIds.begin();
    std::int64_t middle = begin;
    while (middle < end && isSideA(middle))
        ++middle;
    return FastenerPiercingRows{{begin, middle}, {middle, end}};
}

const SideFailure* FastenerPiercingTable::findFailure(std::int32_t elementId, bool isSideA) const
{
    const auto found = std::find_if(m_failures.begin(), m_failures.end(), [&](const SideFailure& failure)
                                    { return failure.elementId == elementId && failure.isSideA == isSideA; });
    return found == m_failures.end() ? nullptr : &*found;
}

} // namespace oos::input
