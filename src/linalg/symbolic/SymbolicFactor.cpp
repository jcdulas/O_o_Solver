#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/linalg/symbolic/ColumnCounts.hpp"
#include "oos/linalg/symbolic/EliminationTree.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

void requireValidInput(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                       const NodePermutation& ordering, std::int32_t trailingNodeCount)
{
    const std::int32_t n = graph.nodeCount();
    if (ordering.size() != n || nodeWeights.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument(base::concatenate(
            {"SymbolicFactor: expected an ordering and one weight per node (", std::to_string(n), "), found ",
             std::to_string(ordering.size()), " positions and ", std::to_string(nodeWeights.size()), " weights"}));
    if (trailingNodeCount < 0 || trailingNodeCount > n)
        throw std::invalid_argument(
            base::concatenate({"SymbolicFactor: expected a trailing count in [0, ", std::to_string(n), "], found ",
                               std::to_string(trailingNodeCount)}));
    std::int64_t dofs = 0;
    for (std::size_t k = 0; k < nodeWeights.size(); ++k)
    {
        if (nodeWeights[k] < 1)
            throw std::invalid_argument(
                base::concatenate({"SymbolicFactor: expected node weights >= 1 (drop the nodes without DOF with "
                                   "NodeGraph::inducedSubgraph), found ",
                                   std::to_string(nodeWeights[k]), " at node ", std::to_string(k)}));
        dofs += nodeWeights[k];
    }
    if (dofs > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument(base::concatenate({"SymbolicFactor: ", std::to_string(dofs),
                                                       " DOFs, more than a DofIndex holds; condense the model in "
                                                       "units"}));
}

// The given order with its interior positions postordered (roots and children in increasing position), the trailing
// positions unchanged: an equivalent order (a topological order of the same tree, hence the same fill) in which every
// subtree is contiguous, as the supernodes need.
NodePermutation postorderInterior(const NodePermutation& ordering, const EliminationTree& tree,
                                  std::int32_t interiorCount, base::MemoryBudget& budget)
{
    const auto n = static_cast<std::size_t>(ordering.size());
    const auto interior = static_cast<std::size_t>(interiorCount);
    base::BudgetedBuffer<std::int32_t> head(budget, interior, "symbolic: first children");
    base::BudgetedBuffer<std::int32_t> next(budget, interior, "symbolic: next siblings");
    base::BudgetedBuffer<std::int32_t> stack(budget, interior, "symbolic: stack");
    base::BudgetedBuffer<std::int32_t> nodes(budget, n, "node permutation");
    std::fill_n(head.data(), interior, EliminationTree::ROOT);
    const auto isInteriorRoot = [&](std::size_t j)
    {
        const std::int32_t parent = tree.parent(static_cast<std::int32_t>(j));
        return parent == EliminationTree::ROOT || parent >= interiorCount;
    };
    for (std::size_t j = interior; j-- > 0;)
    {
        if (isInteriorRoot(j))
            continue;
        const auto parent = static_cast<std::size_t>(tree.parent(static_cast<std::int32_t>(j)));
        next[j] = head[parent];
        head[parent] = static_cast<std::int32_t>(j);
    }
    std::size_t k = 0;
    for (std::size_t root = 0; root < interior; ++root)
    {
        if (!isInteriorRoot(root))
            continue;
        std::size_t top = 0;
        stack[0] = static_cast<std::int32_t>(root);
        while (true)
        {
            const std::int32_t p = stack[top];
            const std::int32_t child = head[static_cast<std::size_t>(p)];
            if (child == EliminationTree::ROOT)
            {
                nodes[k++] = ordering.nodeAt(p);
                if (top == 0)
                    break;
                --top;
            }
            else
            {
                head[static_cast<std::size_t>(p)] = next[static_cast<std::size_t>(child)];
                stack[++top] = child;
            }
        }
    }
    for (std::size_t j = interior; j < n; ++j)
        nodes[j] = ordering.nodeAt(static_cast<std::int32_t>(j));
    return NodePermutation(std::move(nodes), budget);
}

// The supernodes wider than maxColumns cut into chains of pieces of about that many columns (whole positions, a cut
// only at a link of a chain of the elimination tree, so that a piece may be wider), each piece the parent of the one
// before it, the last the cut supernode's parent; every other supernode's parent the piece holding its etree parent's
// position (a child of an amalgamated supernode may hang from any of its positions). The trailing supernode is kept
// whole.
Supernodes cutWideSupernodes(const Supernodes& supernodes, const EliminationTree& tree,
                             std::span<const std::int32_t> positionWeight, std::int64_t maxColumns, bool hasTrailing,
                             base::MemoryBudget& budget)
{
    const std::int32_t count = supernodes.count();
    std::vector<std::int32_t> firsts; // the pieces' first positions, bounded by the positions
    for (std::int32_t s = 0; s < count; ++s)
    {
        const bool isTrailing = hasTrailing && s == count - 1;
        std::int64_t columns = 0;
        firsts.push_back(supernodes.firstPosition(s));
        for (std::int32_t p = supernodes.firstPosition(s); p <= supernodes.lastPosition(s); ++p)
        {
            const std::int64_t weight = positionWeight[static_cast<std::size_t>(p)];
            // A cut only where the elimination tree is a chain (p − 1's parent is p): the piece before it then has the
            // structure of its last column, as the row structures below make it.
            if (!isTrailing && columns > 0 && columns + weight > maxColumns && tree.parent(p - 1) == p)
            {
                firsts.push_back(p);
                columns = 0;
            }
            columns += weight;
        }
    }
    const std::int32_t positions = supernodes.firstPosition(count);
    if (static_cast<std::int32_t>(firsts.size()) == count)
    {
        base::BudgetedBuffer<std::int32_t> firstPosition(budget, static_cast<std::size_t>(count) + 1,
                                                         "supernodes: first positions");
        base::BudgetedBuffer<std::int32_t> parent(budget, static_cast<std::size_t>(count), "supernodes: parents");
        for (std::int32_t s = 0; s <= count; ++s)
            firstPosition[static_cast<std::size_t>(s)] = supernodes.firstPosition(s);
        for (std::int32_t s = 0; s < count; ++s)
            parent[static_cast<std::size_t>(s)] = supernodes.parent(s);
        return Supernodes(std::move(firstPosition), std::move(parent));
    }
    const auto pieces = static_cast<std::int32_t>(firsts.size());
    base::BudgetedBuffer<std::int32_t> firstPosition(budget, firsts.size() + 1, "supernodes: first positions");
    std::copy(firsts.begin(), firsts.end(), firstPosition.data());
    firstPosition[firsts.size()] = positions;
    base::BudgetedBuffer<std::int32_t> pieceOf(budget, static_cast<std::size_t>(positions), "supernodes: pieces");
    for (std::int32_t k = 0; k < pieces; ++k)
        for (std::int32_t p = firstPosition[static_cast<std::size_t>(k)];
             p < firstPosition[static_cast<std::size_t>(k) + 1]; ++p)
            pieceOf[static_cast<std::size_t>(p)] = k;
    base::BudgetedBuffer<std::int32_t> parent(budget, firsts.size(), "supernodes: parents");
    for (std::int32_t k = 0; k < pieces; ++k)
    {
        const std::int32_t last = firstPosition[static_cast<std::size_t>(k) + 1] - 1;
        const std::int32_t above = tree.parent(last);
        parent[static_cast<std::size_t>(k)] =
            above == EliminationTree::ROOT ? Supernodes::ROOT : pieceOf[static_cast<std::size_t>(above)];
    }
    return Supernodes(std::move(firstPosition), std::move(parent));
}

} // namespace

SymbolicFactor SymbolicFactor::analyse(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                       const NodePermutation& ordering, std::int32_t trailingNodeCount,
                                       const AmalgamationThresholds& thresholds, base::MemoryBudget& budget,
                                       const base::OperationControl& control)
{
    requireValidInput(graph, nodeWeights, ordering, trailingNodeCount);
    const std::int32_t n = graph.nodeCount();
    const auto size = static_cast<std::size_t>(n);
    const std::int32_t interiorCount = n - trailingNodeCount;
    SymbolicFactor result;
    {
        const EliminationTree givenTree(graph, ordering, budget);
        result.m_nodeOrder = postorderInterior(ordering, givenTree, interiorCount, budget);
    }
    // The trees and the counts are near-linear passes (a fraction of a second at 10^6 nodes): the control is checked
    // between them, then at every supernode of the row structures, the longest step.
    control.throwIfCancelled();
    const NodePermutation& order = result.m_nodeOrder;
    const EliminationTree tree(graph, order, budget);
    control.throwIfCancelled();
    const ColumnCounts counts(graph, order, tree, tree.postorder(budget).span(), nodeWeights, budget);
    control.throwIfCancelled();

    base::BudgetedBuffer<std::int32_t> positionWeight(budget, size, "symbolic: position weights");
    result.m_firstColumnOfPosition = base::BudgetedBuffer<std::int32_t>(budget, size + 1, "symbolic: first columns");
    for (std::int32_t k = 0; k < n; ++k)
    {
        const auto at = static_cast<std::size_t>(k);
        positionWeight[at] = nodeWeights[static_cast<std::size_t>(order.nodeAt(k))];
        result.m_firstColumnOfPosition[at + 1] = result.m_firstColumnOfPosition[at] + positionWeight[at];
        const std::int64_t w = positionWeight[at];
        result.m_exactEntryCount += w * (w + 1) / 2 + w * counts.belowWeight(k);
        if (k >= interiorCount)
            result.m_trailingDofCount += positionWeight[at];
    }

    const bool hasTrailing = trailingNodeCount > 0;
    result.m_supernodes = Amalgamation(thresholds)
                              .relax(Supernodes::findFundamental(tree, counts, hasTrailing ? interiorCount : n, budget),
                                     positionWeight.span(), counts, hasTrailing, budget);
    if (thresholds.maxColumns > 0)
        result.m_supernodes = cutWideSupernodes(result.m_supernodes, tree, positionWeight.span(), thresholds.maxColumns,
                                                hasTrailing, budget);
    const Supernodes& supernodes = result.m_supernodes;
    const std::int32_t count = supernodes.count();
    const auto supernodeTotal = static_cast<std::size_t>(count);

    // The row structures, node level: R(s) = (adjacency of the nodes of s ∪ R(children)) beyond the last position of
    // s. Its size is known from the column counts (R(s) is the structure of the last column of s), so the arrays are
    // allocated exactly, and the count is checked.
    result.m_rowPositionStart = base::BudgetedBuffer<std::int64_t>(budget, supernodeTotal + 1, "symbolic: row starts");
    result.m_rowStart = base::BudgetedBuffer<std::int64_t>(budget, supernodeTotal + 1, "symbolic: DOF row starts");
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        const std::int32_t last = supernodes.lastPosition(s);
        result.m_rowPositionStart[at + 1] = result.m_rowPositionStart[at] + counts.belowNodeCount(last);
        result.m_rowStart[at + 1] = result.m_rowStart[at] + counts.belowWeight(last);
    }
    result.m_rowPositions = base::BudgetedBuffer<std::int32_t>(
        budget, static_cast<std::size_t>(result.m_rowPositionStart[supernodeTotal]), "symbolic: row positions");
    result.m_rows = base::BudgetedBuffer<std::int32_t>(
        budget, static_cast<std::size_t>(result.m_rowStart[supernodeTotal]), "symbolic: rows");
    {
        base::BudgetedBuffer<std::int32_t> childHead(budget, supernodeTotal, "symbolic: supernode children");
        base::BudgetedBuffer<std::int32_t> childNext(budget, supernodeTotal, "symbolic: supernode siblings");
        base::BudgetedBuffer<std::int32_t> marker(budget, size, "symbolic: markers");
        std::fill_n(childHead.data(), supernodeTotal, Supernodes::ROOT);
        std::fill_n(marker.data(), size, -1);
        for (std::int32_t s = count; s-- > 0;)
        {
            const std::int32_t parent = supernodes.parent(s);
            if (parent == Supernodes::ROOT)
                continue;
            childNext[static_cast<std::size_t>(s)] = childHead[static_cast<std::size_t>(parent)];
            childHead[static_cast<std::size_t>(parent)] = s;
        }
        for (std::int32_t s = 0; s < count; ++s)
        {
            control.throwIfCancelled();
            const auto at = static_cast<std::size_t>(s);
            const std::int32_t last = supernodes.lastPosition(s);
            const std::int64_t end = result.m_rowPositionStart[at + 1];
            std::int64_t cursor = result.m_rowPositionStart[at];
            const auto addRow = [&](std::int32_t position)
            {
                if (position <= last || marker[static_cast<std::size_t>(position)] == s)
                    return;
                marker[static_cast<std::size_t>(position)] = s;
                OOS_ASSERT(cursor < end); // the column counts bound the structure
                result.m_rowPositions[static_cast<std::size_t>(cursor++)] = position;
            };
            for (std::int32_t p = supernodes.firstPosition(s); p <= last; ++p)
            {
                for (const std::int32_t neighbour : graph.neighbours(order.nodeAt(p)))
                    addRow(order.positionOf(neighbour));
            }
            for (std::int32_t child = childHead[at]; child != Supernodes::ROOT;
                 child = childNext[static_cast<std::size_t>(child)])
            {
                for (const std::int32_t position : result.rowPositions(child))
                    addRow(position);
            }
            OOS_ASSERT(cursor == end);
            std::sort(result.m_rowPositions.data() + result.m_rowPositionStart[at], result.m_rowPositions.data() + end);
            // The DOF rows: each row node expanded to its consecutive columns.
            std::int64_t dofCursor = result.m_rowStart[at];
            for (const std::int32_t position : result.rowPositions(s))
            {
                for (std::int32_t c = 0; c < positionWeight[static_cast<std::size_t>(position)]; ++c)
                    result.m_rows[static_cast<std::size_t>(dofCursor++)] = result.firstColumnOfPosition(position) + c;
            }
            OOS_ASSERT(dofCursor == result.m_rowStart[at + 1]);
            control.advance(static_cast<std::uint64_t>(last - supernodes.firstPosition(s) + 1));
        }
    }

    // The estimates, and Liu's stack of update matrices in this postorder.
    base::BudgetedBuffer<std::int64_t> pendingUpdates(budget, supernodeTotal, "symbolic: pending updates");
    std::int64_t stack = 0;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int64_t c = result.columnCount(s);
        const std::int64_t r = result.rowCount(s);
        result.m_panelValueCount += c * (c + r);
        result.m_factorEntryCount += c * (c + 1) / 2 + c * r;
        for (std::int64_t m = r; m < r + c; ++m)
            result.m_flopCount += static_cast<double>(m) * static_cast<double>(m + 1) + 2.0 * static_cast<double>(m);
        const std::int64_t front = (c + r) * (c + r);
        result.m_largestFrontValues = std::max(result.m_largestFrontValues, front);
        result.m_updateStackPeakValues = std::max(result.m_updateStackPeakValues, stack + front);
        stack -= pendingUpdates[static_cast<std::size_t>(s)];
        const std::int32_t parent = supernodes.parent(s);
        if (parent != Supernodes::ROOT && r > 0)
        {
            stack += r * r;
            pendingUpdates[static_cast<std::size_t>(parent)] += r * r;
        }
    }
    return result;
}

std::span<const std::int32_t> SymbolicFactor::rows(std::int32_t supernode) const noexcept
{
    const auto begin = static_cast<std::size_t>(m_rowStart[static_cast<std::size_t>(supernode)]);
    const auto end = static_cast<std::size_t>(m_rowStart[static_cast<std::size_t>(supernode) + 1]);
    return m_rows.span().subspan(begin, end - begin);
}

std::span<const std::int32_t> SymbolicFactor::rowPositions(std::int32_t supernode) const noexcept
{
    const auto begin = static_cast<std::size_t>(m_rowPositionStart[static_cast<std::size_t>(supernode)]);
    const auto end = static_cast<std::size_t>(m_rowPositionStart[static_cast<std::size_t>(supernode) + 1]);
    return m_rowPositions.span().subspan(begin, end - begin);
}

std::uint64_t SymbolicFactor::memoryPeakBytes() const noexcept
{
    const auto values = static_cast<std::uint64_t>(m_panelValueCount + m_updateStackPeakValues);
    const auto indices = static_cast<std::uint64_t>(m_rows.size() + m_rowPositions.size());
    const auto starts = static_cast<std::uint64_t>(m_rowStart.size() + m_rowPositionStart.size());
    return values * sizeof(double) + indices * sizeof(std::int32_t) + starts * sizeof(std::int64_t);
}

} // namespace oos::linalg
