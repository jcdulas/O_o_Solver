#include "oos/input/tables/FastenerPiercingTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <algorithm>
#include <array>
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
    // The validity bitmap of the point grids, least significant bit first (Arrow's layout, read by ValidityView).
    m_pointGridBits.assign((isGiven.size() + 7) / 8 + 1, 0);
    for (std::size_t row = 0; row < isGiven.size(); ++row)
    {
        if (isGiven[row])
            m_pointGridBits[row / 8] = static_cast<std::uint8_t>(m_pointGridBits[row / 8] | (1U << (row % 8)));
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
