#include "oos/design/aids/LinearityGeometry.hpp"

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace oos::design
{

namespace
{

// The shell cards and the number of their corners (the first grid points of the card: ShellQuad8 and ShellTria6 list the
// corners before the mid-side points).
constexpr std::array<std::pair<std::string_view, std::size_t>, 6> SHELL_CARDS{{
    {"ShellQuad4", 4},
    {"ShellQuad4Drilling", 4},
    {"ShellQuad8", 4},
    {"ShellTria3", 3},
    {"ShellTria3Drilling", 3},
    {"ShellTria6", 3},
}};

// The lists of the estimate: the elements around a grid point (four in a regular quadrilateral mesh) and the entry of
// a grid point in a lookup by ID.
constexpr std::uint64_t ELEMENTS_AROUND_A_GRID_POINT = 4;
constexpr std::uint64_t HASH_ENTRY_BYTES = 32;

base::Vec3 positionOf(const input::NodeTable& nodes, std::int32_t row)
{
    const std::span<const double> position = nodes.positionsInBasic()[row];
    return {position[0], position[1], position[2]};
}

// The (unnormalised) normal of a shell from its corners: the cross product of the diagonals of a quadrilateral, of two
// edges of a triangle; its length is twice the area (exact for a flat element).
base::Vec3 doubleAreaNormal(const std::vector<base::Vec3>& corners)
{
    if (corners.size() == 4)
        return base::cross(corners[2] - corners[0], corners[3] - corners[1]);
    return base::cross(corners[1] - corners[0], corners[2] - corners[0]);
}

} // namespace

std::size_t LinearityGeometry::cornerCountOf(std::string_view card) noexcept
{
    for (const auto& [name, corners] : SHELL_CARDS)
        if (name == card)
            return corners;
    return 0;
}

std::uint64_t LinearityGeometry::estimateBytes(const input::ModelSnapshot& snapshot)
{
    std::uint64_t elements = 0;
    std::uint64_t corners = 0;
    for (const input::ElementBatch& batch : snapshot.elementBatches())
        if (const std::size_t count = cornerCountOf(batch.card()); count > 0)
        {
            elements += static_cast<std::uint64_t>(batch.rowCount());
            corners += static_cast<std::uint64_t>(batch.rowCount()) * count;
        }
    const std::uint64_t points = std::min(corners, static_cast<std::uint64_t>(snapshot.nodes().rowCount()));
    return elements * (sizeof(Element) + 4 * sizeof(std::int32_t)) +
           points * (sizeof(Point) + ELEMENTS_AROUND_A_GRID_POINT * sizeof(std::size_t) + HASH_ENTRY_BYTES);
}

std::vector<std::int32_t> LinearityGeometry::cornerRowsOf(const input::ElementBatch& batch, std::int64_t row,
                                                          std::size_t corners, std::int64_t nodeCount)
{
    if (!batch.reader().hasColumn("nodesRows") || !batch.reader().validity("nodesRows").isValid(row))
        return {};
    const std::span<const std::int32_t> rows = batch.nodeRows()[row];
    if (rows.size() < corners)
        return {};
    std::vector<std::int32_t> result(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(corners));
    for (const std::int32_t node : result)
        if (node < 0 || node >= nodeCount)
            return {};
    return result;
}

LinearityGeometry::LinearityGeometry(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget)
    : m_reservation(budget.reserve(estimateBytes(snapshot), "linearity indicator geometry"))
{
    const input::NodeTable& nodes = snapshot.nodes();
    const std::span<const std::int32_t> ids = nodes.ids();
    const std::span<const input::ElementBatch> batches = snapshot.elementBatches();
    std::vector<base::Vec3> positions; // bounded by an element's corners
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const input::ElementBatch& batch = batches[b];
        const std::size_t corners = cornerCountOf(batch.card());
        if (corners == 0)
            continue;
        for (std::int64_t row = 0; row < batch.rowCount(); ++row)
        {
            std::vector<std::int32_t> cornerRows = cornerRowsOf(batch, row, corners, nodes.rowCount());
            if (cornerRows.empty())
                continue;
            positions.clear();
            for (const std::int32_t node : cornerRows)
                positions.push_back(positionOf(nodes, node));
            const base::Vec3 normal = doubleAreaNormal(positions);
            const double area = 0.5 * base::norm(normal);
            if (!(area > 0.0))
                continue;
            Element element;
            element.card = batch.card();
            element.elementId = batch.ids()[static_cast<std::size_t>(row)];
            element.batch = b;
            element.row = row;
            element.cornerRows = std::move(cornerRows);
            element.normal = normal / (2.0 * area);
            element.area = area;
            m_elements.push_back(std::move(element));
        }
    }
    // The elements around each node row, in the order of the batches and rows (the first one orients the normal of the
    // grid point): counted, then filled.
    const auto nodeCount = static_cast<std::size_t>(nodes.rowCount());
    std::vector<std::size_t> aroundStart(nodeCount + 1, 0); // bounded by the nodes: within the estimate
    for (const Element& element : m_elements)
        for (const std::int32_t node : element.cornerRows)
            ++aroundStart[static_cast<std::size_t>(node) + 1];
    for (std::size_t node = 0; node < nodeCount; ++node)
        aroundStart[node + 1] += aroundStart[node];
    std::vector<std::size_t> around(aroundStart[nodeCount]); // within the estimate
    std::vector<std::size_t> next(aroundStart.begin(), aroundStart.end() - 1);
    for (std::size_t e = 0; e < m_elements.size(); ++e)
        for (const std::int32_t node : m_elements[e].cornerRows)
            around[next[static_cast<std::size_t>(node)]++] = e;
    std::vector<std::int32_t> rows; // the node rows with an element, by increasing grid ID
    for (std::size_t node = 0; node < nodeCount; ++node)
        if (aroundStart[node + 1] > aroundStart[node])
            rows.push_back(static_cast<std::int32_t>(node));
    std::sort(rows.begin(), rows.end(), [&](std::int32_t a, std::int32_t b)
              { return ids[static_cast<std::size_t>(a)] < ids[static_cast<std::size_t>(b)]; });
    if (rows.empty())
        return;
    const elements::SnapshotIndex index(snapshot, budget);
    const elements::ElementBatchView frames(snapshot.elementBatches().front(), index);
    m_points.reserve(rows.size());
    for (const std::int32_t row : rows)
    {
        Point point;
        point.grid = ids[static_cast<std::size_t>(row)];
        point.row = row;
        point.displacementFrame = frames.displacementFrame(row);
        point.elements.assign(
            around.begin() + static_cast<std::ptrdiff_t>(aroundStart[static_cast<std::size_t>(row)]),
            around.begin() + static_cast<std::ptrdiff_t>(aroundStart[static_cast<std::size_t>(row) + 1]));
        m_points.push_back(std::move(point));
    }
}

} // namespace oos::design
