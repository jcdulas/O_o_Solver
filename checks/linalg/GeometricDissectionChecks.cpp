// The checks of the geometric nested dissection (linalg::GeometricDissection) on the node graph of a panel of shells:
// the order is a permutation, and the factorisation it leads to costs about the same whether the grid points of the
// mesh are in rows or not. A cut of the dissection follows a mesh line only when the grid points are in rows;
// elsewhere it crosses elements, and its separator must still be one line of the mesh (the lightest cover of the edges
// cut), not every node of the elements crossed on one side.

#include "CheckRunner.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using oos::base::Vec3;

namespace
{

constexpr std::uint64_t BUDGET_BYTES = std::uint64_t{1} << 30;
constexpr std::uint32_t SEED = 2024;
// The grid points inside the panel are moved by up to this share of a cell, in each direction of the panel.
constexpr double MOVED_SHARE = 0.3;
// The operations of the factorisation of the mesh with moved grid points, over those of the mesh in rows: at most
// this. Measured: 1.02 to 1.05 (a separator made of every node the elements crossed have on one side gave 1.5 on the
// grid points alone, and 11 to 18 with the edge and interior nodes).
constexpr double MOVED_COST_BOUND = 1.25;

// The DOFs of the nodes of a mesh: its grid points alone (6 DOFs each), or its grid points, its edges and the
// interiors of its elements (3, 17 and 51 DOFs: an element couples all of its nodes, the interior the heaviest).
enum class Nodes
{
    GridPoints,
    GridPointsEdgesInteriors
};
constexpr std::int32_t GRID_POINT_ALONE_DOFS = 6;
constexpr std::int32_t GRID_POINT_DOFS = 3;
constexpr std::int32_t EDGE_DOFS = 17;
constexpr std::int32_t INTERIOR_DOFS = 51;

// The node graph of a square panel of cells × cells quadrilaterals (or of the two triangles of each): the positions
// and the DOFs of the nodes, and the nodes of each element.
class Panel
{
public:
    Panel(int cells, bool isTriangles, Nodes nodes, double movedShare) : m_cells(cells), m_nodes(nodes)
    {
        std::mt19937 random(SEED);
        const auto move = [&]() { return movedShare * (2.0 * static_cast<double>(random()) / 4294967296.0 - 1.0); };
        for (int j = 0; j <= cells; ++j)
        {
            for (int i = 0; i <= cells; ++i)
            {
                const bool isInside = i > 0 && i < cells && j > 0 && j < cells;
                const double x = static_cast<double>(i) + (isInside ? move() : 0.0);
                const double y = static_cast<double>(j) + (isInside ? move() : 0.0);
                addNode({x, y, 0.0}, nodes == Nodes::GridPoints ? GRID_POINT_ALONE_DOFS : GRID_POINT_DOFS);
            }
        }
        for (int j = 0; j < cells; ++j)
        {
            for (int i = 0; i < cells; ++i)
            {
                if (isTriangles)
                {
                    addElement({grid(i, j), grid(i + 1, j), grid(i + 1, j + 1)});
                    addElement({grid(i, j), grid(i + 1, j + 1), grid(i, j + 1)});
                }
                else
                    addElement({grid(i, j), grid(i + 1, j), grid(i + 1, j + 1), grid(i, j + 1)});
            }
        }
    }

    [[nodiscard]] std::int32_t nodeCount() const { return static_cast<std::int32_t>(m_positions.size()); }

    // The order of the nodes by the nested dissection.
    [[nodiscard]] oos::linalg::NodePermutation order(oos::base::MemoryBudget& budget) const
    {
        const oos::linalg::NodeGraph graph = this->graph(budget);
        return oos::linalg::ConstrainedOrdering().order(graph, m_dofs, m_positions, {},
                                                        oos::linalg::OrderingMethod::NestedDissection, budget);
    }

    // The operations of the factorisation in the order of the nested dissection.
    [[nodiscard]] double factorisationCost() const
    {
        oos::base::MemoryBudget budget(BUDGET_BYTES);
        const oos::linalg::NodeGraph graph = this->graph(budget);
        const oos::linalg::NodePermutation order = this->order(budget);
        return oos::linalg::SymbolicFactor::analyse(graph, m_dofs, order, 0, oos::linalg::AmalgamationThresholds{},
                                                    budget)
            .flopCount();
    }

private:
    [[nodiscard]] std::int32_t grid(int i, int j) const { return i + j * (m_cells + 1); }

    std::int32_t addNode(const Vec3& position, std::int32_t dofs)
    {
        m_positions.push_back(position);
        m_dofs.push_back(dofs);
        return nodeCount() - 1;
    }

    // The node of the edge between two grid points, at their middle (created at its first element).
    std::int32_t edgeNode(std::int32_t a, std::int32_t b)
    {
        const std::pair<std::int32_t, std::int32_t> ends{std::min(a, b), std::max(a, b)};
        const auto found = m_edges.find(ends);
        if (found != m_edges.end())
            return found->second;
        const Vec3 middle = 0.5 * (m_positions[static_cast<std::size_t>(a)] + m_positions[static_cast<std::size_t>(b)]);
        return m_edges.emplace(ends, addNode(middle, EDGE_DOFS)).first->second;
    }

    void addElement(const std::vector<std::int32_t>& corners)
    {
        m_elementNodes.insert(m_elementNodes.end(), corners.begin(), corners.end());
        if (m_nodes == Nodes::GridPointsEdgesInteriors)
        {
            Vec3 centre{};
            for (std::size_t k = 0; k < corners.size(); ++k)
            {
                m_elementNodes.push_back(edgeNode(corners[k], corners[(k + 1) % corners.size()]));
                const Vec3& corner = m_positions[static_cast<std::size_t>(corners[k])];
                centre = centre + corner / static_cast<double>(corners.size());
            }
            m_elementNodes.push_back(addNode(centre, INTERIOR_DOFS));
        }
        m_elementFirsts.push_back(static_cast<std::int64_t>(m_elementNodes.size()));
    }

    [[nodiscard]] oos::linalg::NodeGraph graph(oos::base::MemoryBudget& budget) const
    {
        const oos::linalg::NodeGroupList elements{m_elementFirsts, m_elementNodes};
        return oos::linalg::NodeGraph(nodeCount(), std::span<const oos::linalg::NodeGroupList>(&elements, 1), budget);
    }

    int m_cells;
    Nodes m_nodes;
    std::vector<Vec3> m_positions;
    std::vector<std::int32_t> m_dofs;
    std::map<std::pair<std::int32_t, std::int32_t>, std::int32_t> m_edges;
    std::vector<std::int64_t> m_elementFirsts{0};
    std::vector<std::int32_t> m_elementNodes;
};

// The cost of the panel with its grid points moved over its cost with its grid points in rows.
double movedCostRatio(int cells, bool isTriangles, Nodes nodes)
{
    return Panel(cells, isTriangles, nodes, MOVED_SHARE).factorisationCost() /
           Panel(cells, isTriangles, nodes, 0.0).factorisationCost();
}

} // namespace

CHECK_CASE("dissection: the order of a panel with moved grid points is a permutation")
{
    const Panel panel(24, false, Nodes::GridPointsEdgesInteriors, MOVED_SHARE);
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const oos::linalg::NodePermutation order = panel.order(budget);
    runner.expect(order.size() == panel.nodeCount(), "one position per node");
    std::vector<int> timesOrdered(static_cast<std::size_t>(panel.nodeCount()), 0);
    for (const std::int32_t node : order.nodes())
        ++timesOrdered[static_cast<std::size_t>(node)];
    runner.expect(std::all_of(timesOrdered.begin(), timesOrdered.end(), [](int times) { return times == 1; }),
                  "every node ordered once");
}

CHECK_CASE("dissection: moved grid points do not raise the cost of the factorisation, grid points alone")
{
    const double ratio = movedCostRatio(64, false, Nodes::GridPoints);
    runner.expect(ratio <= MOVED_COST_BOUND, "quadrilaterals: cost ratio " + std::to_string(ratio));
}

CHECK_CASE("dissection: moved grid points do not raise the cost of the factorisation, edge and interior nodes")
{
    const double quadrilaterals = movedCostRatio(24, false, Nodes::GridPointsEdgesInteriors);
    runner.expect(quadrilaterals <= MOVED_COST_BOUND, "quadrilaterals: cost ratio " + std::to_string(quadrilaterals));
    const double triangles = movedCostRatio(24, true, Nodes::GridPointsEdgesInteriors);
    runner.expect(triangles <= MOVED_COST_BOUND, "triangles: cost ratio " + std::to_string(triangles));
}
