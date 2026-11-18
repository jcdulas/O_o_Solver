#include "oos/input/tables/FastenerPiercingTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <optional>
#include <string>
#include <tuple>
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
    m_patchOffsets = m_offsets;
    m_patchGrids = m_grids;
    m_patchWeights = m_weights;
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

// The cards of the shell elements a fastener pierces.
constexpr std::array<std::string_view, 6> SHELL_CARDS{"ShellQuad4",         "ShellTria3",         "ShellQuad8",
                                                      "ShellTria6",         "ShellQuad4Drilling", "ShellTria3Drilling"};

using Point = std::array<double, 3>;

// A shell element at one of its grid points: the batch and the row.
struct ShellAtGrid
{
    std::int32_t gridId = 0;
    std::size_t batch = 0;
    std::int64_t row = 0;
};

// The unit normal of the shell of grid points `ids` (the corners first: three for 3 or 6 grid points, else four):
// along the cross product of two sides of a triangle, of the diagonals of a quadrilateral. Nothing when a corner is
// not a grid point of the model or the corners are on a line.
std::optional<Point> shellNormal(std::span<const std::int32_t> ids,
                                 const std::function<std::optional<Point>(std::int32_t)>& positionOf)
{
    const std::size_t corners = ids.size() == 3 || ids.size() == 6 ? 3 : 4;
    if (ids.size() < corners)
        return std::nullopt;
    std::array<Point, 4> x{};
    for (std::size_t c = 0; c < corners; ++c)
    {
        const std::optional<Point> position = positionOf(ids[c]);
        if (!position)
            return std::nullopt;
        x[c] = *position;
    }
    const auto difference = [&](std::size_t a, std::size_t b)
    { return Point{x[a][0] - x[b][0], x[a][1] - x[b][1], x[a][2] - x[b][2]}; };
    const Point u = corners == 3 ? difference(1, 0) : difference(2, 0);
    const Point v = corners == 3 ? difference(2, 0) : difference(3, 1);
    const Point n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(length > 0.0))
        return std::nullopt;
    return Point{n[0] / length, n[1] / length, n[2] / length};
}

} // namespace

void FastenerPiercingTable::buildPatches(const NodeTable& nodes, std::span<const ElementBatch> batches)
{
    // The rows on a grid point or a side of their element: the others keep the pierced element.
    const auto isOnBoundary = [&](std::int64_t row)
    {
        if (m_elementIds[static_cast<std::size_t>(row)] == 0)
            return false;
        const std::span<const double> values = weights()[row];
        return std::any_of(values.begin(), values.end(),
                           [](double value) { return std::abs(value) <= PIERCING_ZERO_SHAPE_VALUE; });
    };
    bool hasBoundaryRow = false;
    for (std::int64_t row = 0; row < rowCount() && !hasBoundaryRow; ++row)
        hasBoundaryRow = isOnBoundary(row);
    if (!hasBoundaryRow)
        return;

    // The positions of the grid points by ID, and the shell elements at each grid point, sorted by grid point.
    const std::span<const std::int32_t> nodeIds = nodes.ids();
    std::vector<std::pair<std::int32_t, std::int64_t>> nodeRows; // bounded by the grid points
    nodeRows.reserve(nodeIds.size());
    for (std::size_t r = 0; r < nodeIds.size(); ++r)
        nodeRows.emplace_back(nodeIds[r], static_cast<std::int64_t>(r));
    std::sort(nodeRows.begin(), nodeRows.end());
    const base::FixedListView<double> positions = nodes.positionsInBasic();
    const std::function<std::optional<Point>(std::int32_t)> positionOf = [&](std::int32_t id) -> std::optional<Point>
    {
        const auto found =
            std::lower_bound(nodeRows.begin(), nodeRows.end(), std::pair<std::int32_t, std::int64_t>{id, 0});
        if (found == nodeRows.end() || found->first != id)
            return std::nullopt;
        const std::span<const double> x = positions[found->second];
        return Point{x[0], x[1], x[2]};
    };
    std::vector<ShellAtGrid> shells; // bounded by the grid points of the shell elements
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        if (std::find(SHELL_CARDS.begin(), SHELL_CARDS.end(), batches[b].card()) == SHELL_CARDS.end() ||
            !batches[b].hasNodes())
            continue;
        const ConnectivityView connectivity = batches[b].nodes();
        for (std::int64_t row = 0; row < connectivity.size(); ++row)
            for (const std::int32_t id : connectivity[row])
                if (id != 0)
                    shells.push_back({id, b, row});
    }
    std::sort(shells.begin(), shells.end(), [](const ShellAtGrid& a, const ShellAtGrid& b)
              { return std::tie(a.gridId, a.batch, a.row) < std::tie(b.gridId, b.batch, b.row); });

    const double cosine = std::cos(PIERCING_PATCH_ANGLE_DEGREES * std::numbers::pi / 180.0);
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> patchGrids; // bounded by the grid points of the shells around the piercing points
    std::vector<double> patchWeights;
    for (std::int64_t row = 0; row < rowCount(); ++row)
    {
        const std::span<const std::int32_t> ids = grids()[row];
        const std::span<const double> values = weights()[row];
        patchGrids.insert(patchGrids.end(), ids.begin(), ids.end());
        patchWeights.insert(patchWeights.end(), values.begin(), values.end());
        const std::size_t first = static_cast<std::size_t>(offsets.back());
        const std::optional<Point> normal = isOnBoundary(row) ? shellNormal(ids, positionOf) : std::nullopt;
        if (normal)
        {
            // The grid points the piercing point lies on: those whose shape function is not zero.
            std::vector<std::int32_t> held; // at most the grid points of one element
            for (std::size_t k = 0; k < ids.size(); ++k)
                if (std::abs(values[k]) > PIERCING_ZERO_SHAPE_VALUE)
                    held.push_back(ids[k]);
            // The shell elements at the first of them that hold them all and continue the sheet.
            const auto range = std::equal_range(shells.begin(), shells.end(), ShellAtGrid{held.front(), 0, 0},
                                                [](const ShellAtGrid& a, const ShellAtGrid& b)
                                                { return a.gridId < b.gridId; });
            for (auto shell = range.first; shell != range.second; ++shell)
            {
                const std::span<const std::int32_t> others = batches[shell->batch].nodes()[shell->row];
                const auto isHeldByOther = [&](std::int32_t id)
                { return std::find(others.begin(), others.end(), id) != others.end(); };
                const bool holdsAll = std::all_of(held.begin(), held.end(), isHeldByOther);
                if (!holdsAll)
                    continue;
                const std::optional<Point> other = shellNormal(others, positionOf);
                if (!other ||
                    std::abs((*normal)[0] * (*other)[0] + (*normal)[1] * (*other)[1] + (*normal)[2] * (*other)[2]) <
                        cosine)
                    continue;
                for (const std::int32_t id : others)
                {
                    if (id == 0 || !positionOf(id) ||
                        std::find(patchGrids.begin() + static_cast<std::ptrdiff_t>(first), patchGrids.end(), id) !=
                            patchGrids.end())
                        continue;
                    patchGrids.push_back(id);
                    patchWeights.push_back(0.0);
                }
            }
        }
        offsets.push_back(static_cast<std::int64_t>(patchGrids.size()));
    }
    m_patchOffsets = std::move(offsets);
    m_patchGrids = std::move(patchGrids);
    m_patchWeights = std::move(patchWeights);
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
