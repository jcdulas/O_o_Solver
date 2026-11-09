#include "oos/reduction/units/DissectionUnits.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/reduction/plan/UnitCostEstimate.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::reduction
{

namespace
{

constexpr std::int32_t NO_UNIT = -1;
constexpr std::uint64_t VALUE_BYTES = sizeof(double);

void requireSize(const char* what, std::size_t found, std::size_t expected)
{
    if (found != expected)
        throw std::invalid_argument(
            base::concatenate({"DissectionUnits: expected ", std::to_string(expected), " ", what,
                               " (one per node or group), found ", std::to_string(found)}));
}

// What the pass in postorder gathers per supernode about its subtree.
struct SubtreeSizes
{
    base::BudgetedBuffer<std::int32_t> first;      // its first supernode
    base::BudgetedBuffer<std::int64_t> panels;     // Σ c(c + r)
    base::BudgetedBuffer<double> flops;            // Σ panelFlopCount
    base::BudgetedBuffer<std::int64_t> dofs;       // Σ c
    base::BudgetedBuffer<std::int64_t> stackPeak;  // Liu's stack of its multifrontal factorisation, the panels apart
    base::BudgetedBuffer<std::int64_t> heldUpdate; // the update matrices of its children seen so far
    base::BudgetedBuffer<std::int64_t> childPeak;  // the largest peak of a child beside the updates before it
};

SubtreeSizes sizeSubtrees(const linalg::SymbolicFactor& symbolic, base::MemoryBudget& budget)
{
    const auto count = static_cast<std::size_t>(symbolic.supernodeCount());
    SubtreeSizes sizes{base::BudgetedBuffer<std::int32_t>(budget, count, "dissection: subtree firsts"),
                       base::BudgetedBuffer<std::int64_t>(budget, count, "dissection: subtree panels"),
                       base::BudgetedBuffer<double>(budget, count, "dissection: subtree flops"),
                       base::BudgetedBuffer<std::int64_t>(budget, count, "dissection: subtree DOFs"),
                       base::BudgetedBuffer<std::int64_t>(budget, count, "dissection: subtree stacks"),
                       base::BudgetedBuffer<std::int64_t>(budget, count, "dissection: held updates"),
                       base::BudgetedBuffer<std::int64_t>(budget, count, "dissection: child peaks")};
    for (std::size_t s = 0; s < count; ++s)
        sizes.first[s] = static_cast<std::int32_t>(s);
    // Postorder: the children of s come before s, in increasing order (Supernodes), so each child is added to its
    // parent's sums before the parent is finished.
    for (std::int32_t s = 0; s < symbolic.supernodeCount(); ++s)
    {
        const auto at = static_cast<std::size_t>(s);
        const std::int64_t c = symbolic.columnCount(s);
        const std::int64_t r = symbolic.rowCount(s);
        sizes.panels[at] += c * (c + r);
        sizes.flops[at] += panelFlopCount(c, r);
        sizes.dofs[at] += c;
        // The front of s in place (its panel and its update matrix) beside the update matrices of its children.
        sizes.stackPeak[at] = std::max(sizes.childPeak[at], sizes.heldUpdate[at] + c * (c + r) + r * r);
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (parent == linalg::Supernodes::ROOT)
            continue;
        const auto up = static_cast<std::size_t>(parent);
        sizes.first[up] = std::min(sizes.first[up], sizes.first[at]);
        sizes.panels[up] += sizes.panels[at];
        sizes.flops[up] += sizes.flops[at];
        sizes.dofs[up] += sizes.dofs[at];
        sizes.childPeak[up] = std::max(sizes.childPeak[up], sizes.heldUpdate[up] + sizes.stackPeak[at]);
        sizes.heldUpdate[up] += r * r;
    }
    return sizes;
}

std::uint64_t recoveryBytesOf(const SubtreeSizes& sizes, std::size_t s) noexcept
{
    return VALUE_BYTES * static_cast<std::uint64_t>(sizes.panels[s] + sizes.stackPeak[s]);
}

} // namespace

DissectionUnits::DissectionUnits(const DissectionInputs& inputs, const DissectionRequest& request,
                                 base::MemoryBudget& budget, base::ThreadPool* pool)
{
    if (inputs.graph == nullptr)
        throw std::invalid_argument("DissectionUnits: no node graph; give the model's graph with its relations");
    const linalg::NodeGraph& graph = *inputs.graph;
    const auto nodeCount = static_cast<std::size_t>(graph.nodeCount());
    requireSize("DOF counts", inputs.dofCounts.size(), nodeCount);
    requireSize("positions", inputs.positions.size(), nodeCount);
    requireSize("element IDs", inputs.elementIds.size(), inputs.elementNodes.groupCount());
    requireSize("one-node element nodes", inputs.pointElementNodes.size(), inputs.pointElementIds.size());
    m_nodePositions = base::BudgetedBuffer<std::int32_t>(budget, nodeCount, "dissection: node positions");
    if (nodeCount == 0)
        return;

    // The order: every node with at least one DOF in the weights (a node held in every subcase keeps its place in the
    // graph; it counts one), the interface nodes last.
    base::BudgetedBuffer<std::int32_t> weights(budget, nodeCount, "dissection: node weights");
    for (std::size_t node = 0; node < nodeCount; ++node)
        weights[node] = std::max(1, inputs.dofCounts[node]);
    std::vector<std::int32_t> trailing(inputs.interfaceNodes.begin(),
                                       inputs.interfaceNodes.end()); // FreeBodySupport points
    std::sort(trailing.begin(), trailing.end());
    trailing.erase(std::unique(trailing.begin(), trailing.end()), trailing.end());
    const linalg::NodePermutation order = linalg::ConstrainedOrdering().order(
        graph, weights.span(), inputs.positions, trailing, linalg::OrderingMethod::NestedDissection, budget, pool);
    const linalg::SymbolicFactor symbolic =
        linalg::SymbolicFactor::analyse(graph, weights.span(), order, static_cast<std::int32_t>(trailing.size()),
                                        linalg::AmalgamationThresholds{}, budget);
    for (std::size_t node = 0; node < nodeCount; ++node)
        m_nodePositions[node] = symbolic.nodeOrder().positionOf(static_cast<std::int32_t>(node));
    m_dofCount = symbolic.dofCount();
    m_factorValueCount = symbolic.panelValueCount();
    m_flopCount = symbolic.flopCount();

    const std::int32_t supernodeCount = symbolic.supernodeCount();
    const auto count = static_cast<std::size_t>(supernodeCount);
    const SubtreeSizes sizes = sizeSubtrees(symbolic, budget);
    const auto isAdmissible = [&](std::int32_t s)
    {
        // The trailing supernode (the FreeBodySupport points) is the interface's whatever its size.
        return !symbolic.isTrailing(s) && recoveryBytesOf(sizes, static_cast<std::size_t>(s)) <= request.maxUnitBytes;
    };

    // The maximal admissible subtrees, in postorder, then consecutive siblings merged while their sum is admissible.
    base::BudgetedBuffer<std::int32_t> unitOf(budget, count, "dissection: unit of each supernode");
    for (std::size_t s = 0; s < count; ++s)
        unitOf[s] = NO_UNIT;
    std::int32_t s = supernodeCount - 1;
    std::vector<std::int32_t> roots; // the unit roots, found from the last supernode down: bounded by the supernodes
    while (s >= 0)
    {
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (isAdmissible(s) && (parent == linalg::Supernodes::ROOT || !isAdmissible(parent)))
        {
            roots.push_back(s);
            s = sizes.first[static_cast<std::size_t>(s)] - 1;
            continue;
        }
        --s;
    }
    std::reverse(roots.begin(), roots.end());
    for (std::size_t k = 0; k < roots.size();)
    {
        const std::int32_t root = roots[k];
        DissectionUnit unit;
        unit.firstSupernode = sizes.first[static_cast<std::size_t>(root)];
        unit.lastSupernode = root;
        const std::int32_t parent = symbolic.supernodes().parent(root);
        std::uint64_t bytes = recoveryBytesOf(sizes, static_cast<std::size_t>(root));
        std::size_t next = k + 1;
        // A sibling whose subtree starts right after this one's (the next child of the same parent in postorder).
        while (next < roots.size() && symbolic.supernodes().parent(roots[next]) == parent &&
               sizes.first[static_cast<std::size_t>(roots[next])] == unit.lastSupernode + 1 &&
               bytes + recoveryBytesOf(sizes, static_cast<std::size_t>(roots[next])) <= request.maxUnitBytes)
        {
            bytes += recoveryBytesOf(sizes, static_cast<std::size_t>(roots[next]));
            unit.lastSupernode = roots[next];
            ++next;
        }
        for (std::size_t member = k; member < next; ++member)
        {
            const auto at = static_cast<std::size_t>(roots[member]);
            unit.interiorDofCount += sizes.dofs[at];
            unit.boundaryDofCount += symbolic.rowCount(roots[member]);
            unit.factorValueCount += sizes.panels[at];
            unit.flopCount += sizes.flops[at];
            unit.stackPeakValues = std::max(unit.stackPeakValues, sizes.stackPeak[at]);
        }
        unit.recoveryBytes = bytes;
        const auto index = static_cast<std::int32_t>(m_units.size());
        for (std::int32_t member = unit.firstSupernode; member <= unit.lastSupernode; ++member)
            unitOf[static_cast<std::size_t>(member)] = index;
        m_units.push_back(std::move(unit));
        k = next;
    }

    // The interface and the depths.
    base::BudgetedBuffer<std::int32_t> depth(budget, count, "dissection: depths");
    for (std::int32_t t = supernodeCount - 1; t >= 0; --t)
    {
        const std::int32_t parent = symbolic.supernodes().parent(t);
        depth[static_cast<std::size_t>(t)] =
            parent == linalg::Supernodes::ROOT ? 0 : depth[static_cast<std::size_t>(parent)] + 1;
    }
    for (std::int32_t t = 0; t < supernodeCount; ++t)
    {
        const auto at = static_cast<std::size_t>(t);
        const std::int64_t c = symbolic.columnCount(t);
        const std::int64_t r = symbolic.rowCount(t);
        const auto level = static_cast<std::size_t>(depth[at]);
        if (m_levels.size() <= level)
            m_levels.resize(level + 1);
        DissectionLevel& entry = m_levels[level];
        entry.depth = depth[at];
        ++entry.supernodeCount;
        entry.factorValueCount += c * (c + r);
        entry.flopCount += panelFlopCount(c, r);
        entry.largestFrontValues = std::max(entry.largestFrontValues, (c + r) * (c + r));
        if (unitOf[at] != NO_UNIT)
            continue;
        ++m_interface.supernodeCount;
        m_interface.dofCount += c;
        m_interface.factorValueCount += c * (c + r);
        m_interface.flopCount += panelFlopCount(c, r);
        m_interface.largestFrontValues = std::max(m_interface.largestFrontValues, (c + r) * (c + r));
    }
    if (m_units.empty())
    {
        m_unplacedElementCount = static_cast<std::int64_t>(inputs.placedElements.size());
        return;
    }

    // The elements: by their first eliminated node, then those of the interface alone.
    base::BudgetedBuffer<std::int32_t> supernodeOfPosition(budget, nodeCount, "dissection: supernode of positions");
    for (std::int32_t t = 0; t < supernodeCount; ++t)
        for (std::int32_t p = symbolic.supernodes().firstPosition(t); p <= symbolic.supernodes().lastPosition(t); ++p)
            supernodeOfPosition[static_cast<std::size_t>(p)] = t;
    base::BudgetedBuffer<std::int32_t> nodeUnit(budget, nodeCount, "dissection: unit touching each node");
    for (std::size_t node = 0; node < nodeCount; ++node)
        nodeUnit[node] = NO_UNIT;
    const auto isPlaced = [&](std::int32_t elementId)
    { return std::binary_search(inputs.placedElements.begin(), inputs.placedElements.end(), elementId); };
    const auto firstSupernodeOf = [&](std::span<const std::int32_t> nodes)
    {
        std::int32_t first = m_nodePositions[static_cast<std::size_t>(nodes.front())];
        for (const std::int32_t node : nodes)
            first = std::min(first, m_nodePositions[static_cast<std::size_t>(node)]);
        return supernodeOfPosition[static_cast<std::size_t>(first)];
    };
    const auto place = [&](std::int32_t elementId, std::span<const std::int32_t> nodes, std::int32_t unit)
    {
        m_units[static_cast<std::size_t>(unit)].elementIds.push_back(elementId);
        for (const std::int32_t node : nodes)
            if (nodeUnit[static_cast<std::size_t>(node)] == NO_UNIT)
                nodeUnit[static_cast<std::size_t>(node)] = unit;
    };
    struct Deferred
    {
        std::int32_t elementId;
        std::int64_t group; // in elementNodes, or −1 − k for the one-node element k
    };
    std::vector<Deferred> deferred; // the elements of the interface alone: bounded by the elements
    const std::size_t groupCount = inputs.elementNodes.groupCount();
    for (std::size_t g = 0; g < groupCount; ++g)
    {
        const std::int32_t elementId = inputs.elementIds[g];
        if (!isPlaced(elementId))
            continue;
        const std::span<const std::int32_t> nodes = inputs.elementNodes.group(g);
        const std::int32_t unit = unitOf[static_cast<std::size_t>(firstSupernodeOf(nodes))];
        if (unit == NO_UNIT)
            deferred.push_back({elementId, static_cast<std::int64_t>(g)});
        else
            place(elementId, nodes, unit);
    }
    for (std::size_t k = 0; k < inputs.pointElementIds.size(); ++k)
    {
        const std::int32_t elementId = inputs.pointElementIds[k];
        if (!isPlaced(elementId))
            continue;
        const std::span<const std::int32_t> nodes(&inputs.pointElementNodes[k], 1);
        const std::int32_t unit = unitOf[static_cast<std::size_t>(firstSupernodeOf(nodes))];
        if (unit == NO_UNIT)
            deferred.push_back({elementId, -1 - static_cast<std::int64_t>(k)});
        else
            place(elementId, nodes, unit);
    }
    for (const Deferred& element : deferred)
    {
        const std::span<const std::int32_t> nodes =
            element.group >= 0 ? inputs.elementNodes.group(static_cast<std::size_t>(element.group))
                               : std::span<const std::int32_t>(
                                     &inputs.pointElementNodes[static_cast<std::size_t>(-1 - element.group)], 1);
        std::int32_t chosen = NO_UNIT;
        std::int32_t best = 0;
        for (const std::int32_t node : nodes)
        {
            const std::int32_t unit = nodeUnit[static_cast<std::size_t>(node)];
            if (unit == NO_UNIT)
                continue;
            const auto score =
                static_cast<std::int32_t>(std::count_if(nodes.begin(), nodes.end(), [&](std::int32_t other)
                                                        { return nodeUnit[static_cast<std::size_t>(other)] == unit; }));
            if (score > best || (score == best && unit < chosen))
            {
                best = score;
                chosen = unit;
            }
        }
        if (chosen == NO_UNIT)
        {
            // The first unit below its first node's supernode, or the first unit.
            const std::int32_t top = firstSupernodeOf(nodes);
            const std::int32_t from = sizes.first[static_cast<std::size_t>(top)];
            chosen = 0;
            for (std::int32_t t = from; t <= top; ++t)
                if (unitOf[static_cast<std::size_t>(t)] != NO_UNIT)
                {
                    chosen = unitOf[static_cast<std::size_t>(t)];
                    break;
                }
        }
        place(element.elementId, nodes, chosen);
    }
    for (DissectionUnit& unit : m_units)
        std::sort(unit.elementIds.begin(), unit.elementIds.end());
    std::erase_if(m_units, [](const DissectionUnit& unit) { return unit.elementIds.empty(); });
}

} // namespace oos::reduction
