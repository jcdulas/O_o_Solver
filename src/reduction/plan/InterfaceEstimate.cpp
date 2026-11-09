#include "oos/reduction/plan/InterfaceEstimate.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"
#include "oos/linalg/ordering/QuotientGraph.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/reduction/plan/UnitCostEstimate.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::reduction
{

namespace
{

constexpr std::int32_t NOT_IN_INTERFACE = -1;

std::int32_t modelNodeOf(const PlanGraph& graph, base::EntityId nodeId)
{
    const std::optional<std::int32_t> node = graph.nodeOf(nodeId);
    if (!node)
        throw std::invalid_argument(base::concatenate(
            {"InterfaceEstimate: point ", std::to_string(nodeId),
             " is not a point of the model; the units and the plan graph must come from one snapshot"}));
    return *node;
}

} // namespace

linalg::NodePermutation InterfaceEstimate::orderOnUnits(const linalg::NodeGraph& graph,
                                                        std::span<const std::int32_t> unitOfNode,
                                                        std::int32_t unitCount,
                                                        std::span<const std::int32_t> nodeWeights,
                                                        std::span<const base::Vec3> coordinates,
                                                        base::MemoryBudget& budget, base::ThreadPool* pool)
{
    const linalg::QuotientGraph quotient(graph, unitOfNode, unitCount, nodeWeights, coordinates, budget);
    const linalg::NodePermutation unitOrder = linalg::ConstrainedOrdering().order(
        quotient.graph(), quotient.unitWeights(),
        coordinates.empty() ? std::span<const base::Vec3>{} : quotient.unitCentroids(), {},
        linalg::OrderingMethod::Automatic, budget, pool);
    return quotient.expandUnitOrder(unitOrder, budget);
}

linalg::RegionSweepCost InterfaceEstimate::regionCostOf(std::uint64_t regionBytes) const noexcept
{
    linalg::RegionSweepCost found;
    for (const linalg::RegionSweepCost& cost : regionCosts)
        if (cost.regionValues * static_cast<std::int64_t>(sizeof(double)) <= static_cast<std::int64_t>(regionBytes) ||
            &cost == &regionCosts.front())
            found = cost;
    return found;
}

InterfaceEstimate InterfaceEstimate::estimate(const PlanGraph& graph, std::span<const InterfacePart> parts,
                                              std::span<const base::EntityId> connectionNodes,
                                              std::span<const RelationKey> interfaceRelations,
                                              base::MemoryBudget& budget, base::ThreadPool* pool,
                                              bool estimatesStreamedSolve)
{
    InterfaceEstimate result;
    const linalg::NodeGraph& model = graph.graph();
    const std::span<const std::int32_t> freeDofs = graph.freeDofCounts();
    base::BudgetedBuffer<std::int32_t> localOf(budget, static_cast<std::size_t>(model.nodeCount()),
                                               "interface point index");
    for (std::size_t node = 0; node < localOf.size(); ++node)
        localOf[node] = NOT_IN_INTERFACE;

    // The interface points, each in the first part that brings it, and their quotient vertices: one per condensed
    // unit (its boundary), one per other point.
    std::size_t candidateCount = connectionNodes.size();
    for (const InterfacePart& part : parts)
        candidateCount += part.isCondensed ? part.boundary.size() : part.nodes.size();
    base::BudgetedBuffer<std::int32_t> modelNodes(budget, candidateCount, "interface points");
    base::BudgetedBuffer<std::int32_t> vertexOf(budget, candidateCount, "interface quotient vertices");
    std::size_t count = 0;
    std::int32_t vertexCount = 0;
    const auto add = [&](std::int32_t node, std::int32_t vertex)
    {
        if (freeDofs[static_cast<std::size_t>(node)] == 0 || localOf[static_cast<std::size_t>(node)] >= 0)
            return false;
        localOf[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(count);
        modelNodes[count] = node;
        vertexOf[count] = vertex;
        ++count;
        return true;
    };
    for (const InterfacePart& part : parts)
    {
        if (part.isCondensed)
        {
            ++result.condensedUnitCount;
            bool isUsed = false;
            for (const base::EntityId nodeId : part.boundary)
                isUsed = add(modelNodeOf(graph, nodeId), vertexCount) || isUsed;
            if (isUsed)
                ++vertexCount;
            continue;
        }
        ++result.fullUnitCount;
        for (const base::EntityId nodeId : part.nodes)
            if (add(modelNodeOf(graph, nodeId), vertexCount))
                ++vertexCount;
    }
    for (const base::EntityId nodeId : connectionNodes)
        if (add(modelNodeOf(graph, nodeId), vertexCount))
        {
            ++vertexCount;
            ++result.connectionNodeCount;
        }
    if (count == 0)
        return result;
    // The interface problem numbers its points in the model's order and its vertices in the order they first appear
    // there (InterfaceProblem::order): so are they here, so that the quotient graph's ordering takes the
    // same choices, its ties included, and the estimate is the factor the analysis makes (a barrel of the section at
    // 1/32: 0.55 GB estimated in the order of the parts, one dense supernode of 0.82 GB made).
    {
        std::vector<std::size_t> byNode(count); // bounded by the interface points
        for (std::size_t k = 0; k < count; ++k)
            byNode[k] = k;
        std::sort(byNode.begin(), byNode.end(),
                  [&](std::size_t a, std::size_t b) { return modelNodes[a] < modelNodes[b]; });
        base::BudgetedBuffer<std::int32_t> sortedNodes(budget, count, "interface points");
        base::BudgetedBuffer<std::int32_t> sortedVertices(budget, count, "interface quotient vertices");
        std::vector<std::int32_t> renumbered(static_cast<std::size_t>(vertexCount), -1); // bounded by the vertices
        std::int32_t nextVertex = 0;
        for (std::size_t k = 0; k < count; ++k)
        {
            const std::size_t old = byNode[k];
            sortedNodes[k] = modelNodes[old];
            std::int32_t& vertex = renumbered[static_cast<std::size_t>(vertexOf[old])];
            if (vertex < 0)
                vertex = nextVertex++;
            sortedVertices[k] = vertex;
            localOf[static_cast<std::size_t>(sortedNodes[k])] = static_cast<std::int32_t>(k);
        }
        for (std::size_t k = 0; k < count; ++k)
        {
            modelNodes[k] = sortedNodes[k];
            vertexOf[k] = sortedVertices[k];
        }
        vertexCount = nextVertex;
    }

    // The couplings each K* adds (its structural pattern): the elimination of a connected set of interior points
    // couples every boundary point adjacent to it (the fill of Gaussian elimination: George and Liu, "Computer
    // Solution of Large Sparse Positive Definite Systems", 1981, §5.1, reachable sets). So each connected component of
    // a unit's interior gives a clique on the boundary points it touches; a boundary point that touches no interior
    // point (a row of spider legs outside the plate's last interior row) is coupled only to its own neighbours. K* is
    // stored dense, but its structurally zero blocks are not assembled. Two passes: count, then fill.
    base::BudgetedBuffer<std::int32_t> visitedBy(budget, static_cast<std::size_t>(model.nodeCount()),
                                                 "interface interior marks");
    for (std::size_t node = 0; node < visitedBy.size(); ++node)
        visitedBy[node] = NOT_IN_INTERFACE;
    std::size_t largestUnit = 0;
    for (const InterfacePart& part : parts)
        largestUnit = std::max(largestUnit, part.isCondensed ? part.nodes.size() : 0);
    base::BudgetedBuffer<std::int32_t> stack(budget, largestUnit, "interface interior search");
    std::int32_t pass = 0;
    std::size_t cliqueCount = 0;
    std::size_t cliqueFill = 0;
    const auto forEachClique = [&](auto&& consume)
    {
        // Each pass marks with new values, so that the marks of a pass never look visited to the next one.
        for (std::size_t p = 0; p < parts.size(); ++p)
        {
            const InterfacePart& part = parts[p];
            if (!part.isCondensed)
                continue;
            const std::int32_t mark = pass * static_cast<std::int32_t>(parts.size()) + static_cast<std::int32_t>(p);
            const auto isBoundary = [&](std::int32_t node)
            {
                return std::binary_search(part.boundary.begin(), part.boundary.end(),
                                          graph.nodeIds()[static_cast<std::size_t>(node)]);
            };
            std::vector<std::int32_t> clique; // bounded by the unit's boundary (the K* front)
            // A boundary point brings its free DOFs, and, for the DOFs a relation of the connection system makes
            // dependent, the points they depend on (the relation is eliminated at the interface: K* couples them).
            const auto addMember = [&](std::int32_t node)
            {
                if (const std::int32_t local = localOf[static_cast<std::size_t>(node)]; local >= 0)
                    clique.push_back(local);
                for (const DependentPoint& entry : graph.dependentsOf(node))
                {
                    if (!std::binary_search(interfaceRelations.begin(), interfaceRelations.end(), entry.relation))
                        continue;
                    for (const std::int32_t independent : entry.independentNodes)
                        if (const std::int32_t local = localOf[static_cast<std::size_t>(independent)]; local >= 0)
                            clique.push_back(local);
                }
            };
            for (const base::EntityId nodeId : part.nodes)
            {
                const std::int32_t start = modelNodeOf(graph, nodeId);
                if (graph.unitDofCount(start, interfaceRelations) == 0 || isBoundary(start) ||
                    visitedBy[static_cast<std::size_t>(start)] == mark)
                    continue;
                // One connected component of the interior, and the boundary points it touches.
                clique.clear();
                std::size_t top = 0;
                stack[top++] = start;
                visitedBy[static_cast<std::size_t>(start)] = mark;
                while (top > 0)
                {
                    const std::int32_t node = stack[--top];
                    for (const std::int32_t neighbour : model.neighbours(node))
                    {
                        if (visitedBy[static_cast<std::size_t>(neighbour)] == mark)
                            continue;
                        if (isBoundary(neighbour))
                        {
                            addMember(neighbour);
                            continue;
                        }
                        if (graph.unitDofCount(neighbour, interfaceRelations) == 0 ||
                            !std::binary_search(part.nodes.begin(), part.nodes.end(),
                                                graph.nodeIds()[static_cast<std::size_t>(neighbour)]))
                            continue;
                        visitedBy[static_cast<std::size_t>(neighbour)] = mark;
                        stack[top++] = neighbour;
                    }
                }
                std::sort(clique.begin(), clique.end());
                clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
                if (clique.size() > 1)
                    consume(clique);
            }
        }
        ++pass;
    };
    std::size_t cliqueSize = 0;
    forEachClique(
        [&](const std::vector<std::int32_t>& clique)
        {
            ++cliqueCount;
            cliqueSize += clique.size();
        });
    base::BudgetedBuffer<std::int64_t> cliqueOffsets(budget, cliqueCount + 1, "interface clique offsets");
    base::BudgetedBuffer<std::int32_t> cliqueNodes(budget, cliqueSize, "interface clique points");
    cliqueOffsets[0] = 0;
    cliqueCount = 0;
    forEachClique(
        [&](const std::vector<std::int32_t>& clique)
        {
            for (const std::int32_t local : clique)
                cliqueNodes[cliqueFill++] = local;
            cliqueOffsets[++cliqueCount] = static_cast<std::int64_t>(cliqueFill);
        });

    // The couplings of the model between interface points: each edge once, as a group of two.
    std::size_t edgeCount = 0;
    for (std::size_t local = 0; local < count; ++local)
        for (const std::int32_t neighbour : model.neighbours(modelNodes[local]))
            edgeCount += localOf[static_cast<std::size_t>(neighbour)] > static_cast<std::int32_t>(local) ? 1 : 0;
    base::BudgetedBuffer<std::int64_t> edgeOffsets(budget, edgeCount + 1, "interface edge offsets");
    base::BudgetedBuffer<std::int32_t> edgeNodes(budget, 2 * edgeCount, "interface edges");
    std::size_t edge = 0;
    edgeOffsets[0] = 0;
    for (std::size_t local = 0; local < count; ++local)
        for (const std::int32_t neighbour : model.neighbours(modelNodes[local]))
        {
            const std::int32_t other = localOf[static_cast<std::size_t>(neighbour)];
            if (other <= static_cast<std::int32_t>(local))
                continue;
            edgeNodes[2 * edge] = static_cast<std::int32_t>(local);
            edgeNodes[2 * edge + 1] = other;
            ++edge;
            edgeOffsets[edge] = static_cast<std::int64_t>(2 * edge);
        }
    const std::vector<linalg::NodeGroupList> groups{
        {edgeOffsets.span(), edgeNodes.span()},
        {cliqueOffsets.span().first(cliqueCount + 1), cliqueNodes.span().first(cliqueFill)}};
    const linalg::NodeGraph interfaceGraph(static_cast<std::int32_t>(count), groups, budget, pool);

    base::BudgetedBuffer<std::int32_t> weights(budget, count, "interface point weights");
    base::BudgetedBuffer<base::Vec3> positions(budget, count, "interface point positions");
    for (std::size_t local = 0; local < count; ++local)
    {
        weights[local] = freeDofs[static_cast<std::size_t>(modelNodes[local])];
        positions[local] = graph.positions()[static_cast<std::size_t>(modelNodes[local])];
    }
    const linalg::NodePermutation order = orderOnUnits(interfaceGraph, vertexOf.span().first(count), vertexCount,
                                                       weights.span(), positions.span(), budget, pool);
    const linalg::SymbolicFactor symbolic = linalg::SymbolicFactor::analyse(interfaceGraph, weights.span(), order, 0,
                                                                            linalg::AmalgamationThresholds{}, budget);
    result.nodeCount = static_cast<std::int64_t>(count);
    result.dofCount = symbolic.dofCount();
    result.factorValueCount = symbolic.panelValueCount();
    result.flopCount = symbolic.flopCount();
    result.memoryPeakBytes = symbolic.memoryPeakBytes();
    result.estimatedSeconds = result.flopCount / ESTIMATE_FLOP_RATE;
    const UnitCostEstimate sizes = UnitCostEstimate::fromGraph(symbolic, interfaceGraph, weights.span());
    result.blockCount = sizes.blockCount;
    result.matrixValueCount = sizes.matrixValueCount;
    result.factorRowCount = sizes.factorRowCount;
    result.supernodeCount = sizes.supernodeCount;
    result.factorisationBytes = sizes.factorisationBytes;
    if (estimatesStreamedSolve)
    {
        // The bounds of CondensedMemoryPlan::interfaceBoundFor: the least region, the powers of two of bytes above it
        // and below the factor, the factor.
        result.streamedSolve = linalg::StreamedSolve::estimate(symbolic, 1);
        constexpr std::int64_t VALUE_BYTES = sizeof(double);
        const std::int64_t least = result.streamedSolve.run.largestPanelValues;
        const std::int64_t whole = result.streamedSolve.panelValues;
        result.regionCosts.push_back(linalg::StreamedSolve::regionCost(symbolic, least));
        for (std::uint64_t bytes = std::bit_ceil(static_cast<std::uint64_t>(least * VALUE_BYTES) + 1);
             static_cast<std::int64_t>(bytes) < whole * VALUE_BYTES; bytes *= 2)
            result.regionCosts.push_back(
                linalg::StreamedSolve::regionCost(symbolic, static_cast<std::int64_t>(bytes) / VALUE_BYTES));
        if (whole > least)
            result.regionCosts.push_back(linalg::StreamedSolve::regionCost(symbolic, whole));
    }
    return result;
}

} // namespace oos::reduction
