#include "oos/linalg/ordering/GeometricDissection.hpp"

#include "TrailingSet.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

// The label of a node that is numbered already: a separator node or a trailing node.
constexpr std::int64_t NUMBERED = -1;

// One run of the dissection. The interior nodes live in `work`; the nodes of a region are work[begin, end) and get the
// output positions [begin, end). Each region has a label unique among the live regions (computed from its range), so
// that "is this neighbour in my region" is one comparison. Regions processed concurrently are disjoint and no edge
// joins them, so they never read or write each other's labels.
class DissectionRun
{
public:
    DissectionRun(const NodeGraph& graph, std::span<const std::int32_t> weights,
                  std::span<const base::Vec3> coordinates, const TrailingSet& trailing,
                  const DissectionOptions& options, const MinimumDegreeOptions& leafOptions, base::MemoryBudget& budget,
                  base::ThreadPool* pool, const base::OperationControl& control)
        : m_control(control),
          m_graph(graph),
          m_weights(weights),
          m_coordinates(coordinates),
          m_options(options),
          m_leafOptions(leafOptions),
          m_budget(budget),
          m_pool(pool),
          m_labelBase(static_cast<std::int64_t>(graph.nodeCount()) + 1),
          m_label(budget, static_cast<std::size_t>(graph.nodeCount()), "dissection: labels"),
          m_work(budget, static_cast<std::size_t>(graph.nodeCount() - trailing.count()), "dissection: regions"),
          m_scratch(budget, m_work.size(), "dissection: scratch"),
          m_order(budget, static_cast<std::size_t>(graph.nodeCount()), "node permutation")
    {
        std::size_t next = 0;
        const std::int64_t whole = labelOf(0, static_cast<std::int32_t>(m_work.size()));
        for (std::int32_t node = 0; node < graph.nodeCount(); ++node)
        {
            if (trailing.contains(node))
            {
                m_label[static_cast<std::size_t>(node)] = NUMBERED;
                continue;
            }
            m_label[static_cast<std::size_t>(node)] = whole;
            m_work[next++] = node;
        }
        for (const std::int32_t node : trailing.nodes())
            m_order[next++] = node;
    }

    // The control is checked at every region (a leaf or a bisection is a small part of the work) and advanced by the
    // nodes numbered: the separators, the leaves, then the trailing nodes.
    [[nodiscard]] base::BudgetedBuffer<std::int32_t> run()
    {
        orderRegion(0, static_cast<std::int32_t>(m_work.size()));
        m_control.advance(m_order.size() - m_work.size());
        return std::move(m_order);
    }

private:
    [[nodiscard]] std::int64_t labelOf(std::int32_t begin, std::int32_t end) const noexcept
    {
        return static_cast<std::int64_t>(end) * m_labelBase + begin;
    }
    [[nodiscard]] std::int64_t& label(std::int32_t node) noexcept { return m_label[static_cast<std::size_t>(node)]; }
    [[nodiscard]] std::int64_t weight(std::int32_t node) const noexcept
    {
        return m_weights.empty() ? 1 : m_weights[static_cast<std::size_t>(node)];
    }
    [[nodiscard]] std::int32_t* work(std::int32_t at) noexcept { return m_work.data() + at; }

    void orderRegion(std::int32_t begin, std::int32_t end)
    {
        m_control.throwIfCancelled();
        if (end - begin <= m_options.leafNodeCount)
        {
            orderLeaf(begin, end);
            return;
        }
        if (splitComponents(begin, end))
            return;
        bisect(begin, end);
    }

    // Splits a disconnected region into its components (breadth-first, from the first node of the region not yet
    // reached) and orders each; false when the region is connected.
    bool splitComponents(std::int32_t begin, std::int32_t end)
    {
        const std::int64_t regionLabel = labelOf(begin, end);
        const std::int64_t reached = -regionLabel - 2; // a value no other live region uses
        std::int32_t* const queue = m_scratch.data();
        std::int32_t tail = begin;
        std::vector<std::int32_t> componentStarts; // bounded by the region's components; few in practice
        for (std::int32_t k = begin; k < end; ++k)
        {
            const std::int32_t seed = *work(k);
            if (label(seed) != regionLabel)
                continue;
            componentStarts.push_back(tail);
            label(seed) = reached;
            queue[tail++] = seed;
            for (std::int32_t head = componentStarts.back(); head < tail; ++head)
            {
                for (const std::int32_t neighbour : m_graph.neighbours(queue[head]))
                {
                    if (label(neighbour) == regionLabel)
                    {
                        label(neighbour) = reached;
                        queue[tail++] = neighbour;
                    }
                }
            }
        }
        OOS_ASSERT(tail == end);
        if (componentStarts.size() == 1)
        {
            for (std::int32_t k = begin; k < end; ++k)
                label(*work(k)) = regionLabel;
            return false;
        }
        std::copy(queue + begin, queue + end, work(begin));
        componentStarts.push_back(end);
        for (std::size_t c = 0; c + 1 < componentStarts.size(); ++c)
        {
            const std::int64_t componentLabel = labelOf(componentStarts[c], componentStarts[c + 1]);
            for (std::int32_t k = componentStarts[c]; k < componentStarts[c + 1]; ++k)
                label(*work(k)) = componentLabel;
        }
        for (std::size_t c = 0; c + 1 < componentStarts.size(); ++c)
            orderRegion(componentStarts[c], componentStarts[c + 1]);
        return true;
    }

    // A cut of a region, its work sorted for it: the first position of the second half, the weight of the separator
    // it makes (the lighter boundary) and of its lighter half.
    struct Cut
    {
        std::int32_t middle = 0;
        std::int64_t separatorWeight = 0;
        std::int64_t lighterHalfWeight = 0;
    };

    // Cuts a connected region (DissectionOptions::componentCuts: at the weighted median of its largest extent, or
    // between its components); the boundary of the lighter side is the separator, numbered last.
    void bisect(std::int32_t begin, std::int32_t end)
    {
        const std::size_t axis = largestExtentAxis(begin, end);
        const bool isFree = m_options.componentCuts == ComponentCuts::Free || m_options.componentOfNode.empty();
        split(begin, end, isFree ? freeCut(begin, end, axis).middle : componentCut(begin, end, axis));
    }

    // The free cut: the weighted median of the coordinate along `axis`.
    Cut freeCut(std::int32_t begin, std::int32_t end, std::size_t axis)
    {
        const auto byCoordinate = [&](std::int32_t a, std::int32_t b)
        {
            const double ca = m_coordinates[static_cast<std::size_t>(a)][axis];
            const double cb = m_coordinates[static_cast<std::size_t>(b)][axis];
            return ca < cb || (ca == cb && a < b);
        };
        std::sort(work(begin), work(end), byCoordinate);
        return evaluate(begin, end,
                        medianCut(begin, end, [&](std::int32_t k)
                                  { return m_coordinates[static_cast<std::size_t>(*work(k))][axis]; }));
    }

    // The cut between whole components (ComponentCuts::Components, Hybrid): the components ranked by their centroids
    // along `axis`, the cut at the weighted median moved to a change of component; the free cut when the region is
    // one component, or for Hybrid when the free cut is much smaller or this one leaves a small half. Returns the cut's
    // middle, the work sorted for it.
    std::int32_t componentCut(std::int32_t begin, std::int32_t end, std::size_t axis)
    {
        const auto count = static_cast<std::size_t>(end - begin);
        // The region's nodes by component (then by node), then one centroid a component.
        struct Member
        {
            std::int32_t component;
            std::int32_t node;
        };
        base::BudgetedBuffer<Member> members(m_budget, count, "dissection: components", base::FOR_OVERWRITE);
        for (std::size_t k = 0; k < count; ++k)
        {
            const std::int32_t node = *work(begin + static_cast<std::int32_t>(k));
            members[k] = {m_options.componentOfNode[static_cast<std::size_t>(node)], node};
        }
        std::sort(members.data(), members.data() + count, [](const Member& a, const Member& b)
                  { return a.component < b.component || (a.component == b.component && a.node < b.node); });
        struct Centroid
        {
            std::int32_t component;
            double position;
            std::int32_t rank;
        };
        std::vector<Centroid> centroids; // bounded by the region's components
        for (std::size_t k = 0; k < count;)
        {
            double moment = 0.0;
            double mass = 0.0;
            std::size_t next = k;
            for (; next < count && members[next].component == members[k].component; ++next)
            {
                const double w = static_cast<double>(weight(members[next].node));
                moment += w * m_coordinates[static_cast<std::size_t>(members[next].node)][axis];
                mass += w;
            }
            centroids.push_back({members[k].component, moment / mass, 0});
            k = next;
        }
        if (centroids.size() < 2)
            return freeCut(begin, end, axis).middle;
        std::vector<std::size_t> byPosition(centroids.size()); // bounded by the region's components
        for (std::size_t c = 0; c < byPosition.size(); ++c)
            byPosition[c] = c;
        std::sort(byPosition.begin(), byPosition.end(),
                  [&](std::size_t a, std::size_t b)
                  {
                      return centroids[a].position < centroids[b].position ||
                             (centroids[a].position == centroids[b].position &&
                              centroids[a].component < centroids[b].component);
                  });
        for (std::size_t r = 0; r < byPosition.size(); ++r)
            centroids[byPosition[r]].rank = static_cast<std::int32_t>(r);
        // The members by rank (within a component by node): the aligned cut's order of the work.
        base::BudgetedBuffer<std::int32_t> rankOf(m_budget, count, "dissection: components", base::FOR_OVERWRITE);
        for (std::size_t k = 0, c = 0; k < count; ++k)
        {
            while (centroids[c].component != members[k].component)
                ++c;
            rankOf[k] = centroids[c].rank;
        }
        base::BudgetedBuffer<std::int32_t> order(m_budget, count, "dissection: components", base::FOR_OVERWRITE);
        for (std::size_t k = 0; k < count; ++k)
            order[k] = static_cast<std::int32_t>(k);
        std::stable_sort(order.data(), order.data() + count, [&](std::int32_t a, std::int32_t b)
                         { return rankOf[static_cast<std::size_t>(a)] < rankOf[static_cast<std::size_t>(b)]; });
        const auto writeAligned = [&]()
        {
            for (std::size_t k = 0; k < count; ++k)
                *work(begin + static_cast<std::int32_t>(k)) = members[static_cast<std::size_t>(order[k])].node;
        };
        writeAligned();
        const Cut aligned =
            evaluate(begin, end,
                     medianCut(begin, end,
                               [&](std::int32_t k)
                               {
                                   return static_cast<double>(
                                       rankOf[static_cast<std::size_t>(order[static_cast<std::size_t>(k - begin)])]);
                               }));
        if (m_options.componentCuts == ComponentCuts::Components)
            return aligned.middle;
        std::int64_t total = 0;
        for (std::int32_t k = begin; k < end; ++k)
            total += weight(*work(k));
        const Cut free = freeCut(begin, end, axis);
        const bool isAlignedTaken =
            static_cast<double>(aligned.separatorWeight) <=
                DissectionOptions::HYBRID_SEPARATOR_RATIO * static_cast<double>(free.separatorWeight) &&
            static_cast<double>(aligned.lighterHalfWeight) >=
                DissectionOptions::HYBRID_LEAST_SHARE * static_cast<double>(total);
        if (!isAlignedTaken)
            return free.middle;
        writeAligned();
        return aligned.middle;
    }

    // The separator and the lighter half of the cut at `middle` of the region's work as it is sorted (the labels set
    // for it; split sets them again).
    Cut evaluate(std::int32_t begin, std::int32_t end, std::int32_t middle)
    {
        const std::int64_t firstLabel = labelOf(begin, middle);
        const std::int64_t secondLabel = labelOf(middle, end);
        for (std::int32_t k = begin; k < end; ++k)
            label(*work(k)) = k < middle ? firstLabel : secondLabel;
        std::int64_t boundary[2] = {0, 0};
        std::int64_t halves[2] = {0, 0};
        for (std::int32_t k = begin; k < end; ++k)
        {
            const std::int32_t node = *work(k);
            const std::size_t side = k < middle ? 0 : 1;
            halves[side] += weight(node);
            if (hasNeighbourLabelled(node, side == 0 ? secondLabel : firstLabel))
                boundary[side] += weight(node);
        }
        const std::size_t separatorSide = boundary[0] <= boundary[1] ? 0 : 1;
        return {middle, boundary[separatorSide],
                std::min(halves[separatorSide] - boundary[separatorSide], halves[1 - separatorSide])};
    }

    // Splits the region at `middle` of its work: the boundary of the lighter side the separator, numbered last; the two
    // halves ordered (in parallel when large).
    void split(std::int32_t begin, std::int32_t end, std::int32_t middle)
    {
        const std::int64_t firstLabel = labelOf(begin, middle);
        const std::int64_t secondLabel = labelOf(middle, end);
        for (std::int32_t k = begin; k < end; ++k)
            label(*work(k)) = k < middle ? firstLabel : secondLabel;
        // The two boundaries: nodes of one side with a neighbour on the other.
        std::int64_t firstBoundaryWeight = 0;
        std::int64_t secondBoundaryWeight = 0;
        std::int32_t firstBoundaryCount = 0;
        std::int32_t secondBoundaryCount = 0;
        for (std::int32_t k = begin; k < end; ++k)
        {
            const std::int32_t node = *work(k);
            const std::int64_t other = k < middle ? secondLabel : firstLabel;
            if (!hasNeighbourLabelled(node, other))
                continue;
            if (k < middle)
            {
                firstBoundaryWeight += weight(node);
                ++firstBoundaryCount;
            }
            else
            {
                secondBoundaryWeight += weight(node);
                ++secondBoundaryCount;
            }
        }
        const bool isFirstSeparator =
            firstBoundaryWeight < secondBoundaryWeight ||
            (firstBoundaryWeight == secondBoundaryWeight && firstBoundaryCount <= secondBoundaryCount);
        // Stable partition into [first side][second side][separator], kept in coordinate order.
        const auto isSeparatorAt = [&](std::int32_t k)
        {
            const bool isFirst = k < middle;
            return isFirst == isFirstSeparator && hasNeighbourLabelled(*work(k), isFirst ? secondLabel : firstLabel);
        };
        std::int32_t* const out = m_scratch.data();
        std::int32_t firstCount = 0;
        std::int32_t secondCount = 0;
        for (std::int32_t k = begin; k < end; ++k)
        {
            if (isSeparatorAt(k))
                continue;
            if (k < middle)
                ++firstCount;
            else
                ++secondCount;
        }
        std::int32_t firstAt = begin;
        std::int32_t secondAt = begin + firstCount;
        std::int32_t separatorAt = begin + firstCount + secondCount;
        for (std::int32_t k = begin; k < end; ++k)
        {
            const std::int32_t node = *work(k);
            if (isSeparatorAt(k))
                out[separatorAt++] = node;
            else if (k < middle)
                out[firstAt++] = node;
            else
                out[secondAt++] = node;
        }
        std::copy(out + begin, out + end, work(begin));
        const std::int32_t secondBegin = begin + firstCount;
        const std::int32_t separatorBegin = secondBegin + secondCount;
        for (std::int32_t k = begin; k < end; ++k)
        {
            const std::int32_t node = *work(k);
            if (k < secondBegin)
                label(node) = labelOf(begin, secondBegin);
            else if (k < separatorBegin)
                label(node) = labelOf(secondBegin, separatorBegin);
            else
            {
                label(node) = NUMBERED;
                m_order[static_cast<std::size_t>(k)] = node;
            }
        }
        m_control.advance(static_cast<std::uint64_t>(end - separatorBegin));
        const bool isParallel =
            m_pool != nullptr && m_pool->threadCount() > 1 && end - begin > m_options.parallelNodeCount;
        if (!isParallel)
        {
            orderRegion(begin, secondBegin);
            orderRegion(secondBegin, separatorBegin);
            return;
        }
        m_pool->runBatch(2,
                         [&](std::size_t half)
                         {
                             if (half == 0)
                                 orderRegion(begin, secondBegin);
                             else
                                 orderRegion(secondBegin, separatorBegin);
                         });
    }

    [[nodiscard]] bool hasNeighbourLabelled(std::int32_t node, std::int64_t wanted) noexcept
    {
        const auto neighbours = m_graph.neighbours(node);
        return std::any_of(neighbours.begin(), neighbours.end(),
                           [&](std::int32_t other) { return label(other) == wanted; });
    }

    [[nodiscard]] std::size_t largestExtentAxis(std::int32_t begin, std::int32_t end) const
    {
        base::Vec3 low = m_coordinates[static_cast<std::size_t>(m_work[static_cast<std::size_t>(begin)])];
        base::Vec3 high = low;
        for (std::int32_t k = begin; k < end; ++k)
        {
            const base::Vec3& point = m_coordinates[static_cast<std::size_t>(m_work[static_cast<std::size_t>(k)])];
            for (std::size_t a = 0; a < 3; ++a)
            {
                low[a] = std::min(low[a], point[a]);
                high[a] = std::max(high[a], point[a]);
            }
        }
        std::size_t axis = 0;
        for (std::size_t a = 1; a < 3; ++a)
        {
            if (high[a] - low[a] > high[axis] - low[axis])
                axis = a;
        }
        return axis;
    }

    // The first position of the second half: the weighted median, moved to the nearest change of the key (the
    // coordinate along the cut's axis, or the component's rank) when that keeps both halves non-empty.
    template <class KeyAt>
    [[nodiscard]] std::int32_t medianCut(std::int32_t begin, std::int32_t end, const KeyAt& coordinateAt) const
    {
        std::int64_t total = 0;
        for (std::int32_t k = begin; k < end; ++k)
            total += weight(m_work[static_cast<std::size_t>(k)]);
        std::int64_t running = 0;
        std::int32_t middle = begin + 1;
        for (std::int32_t k = begin; k < end - 1; ++k)
        {
            running += weight(m_work[static_cast<std::size_t>(k)]);
            middle = k + 1;
            if (2 * running >= total)
                break;
        }
        std::int32_t down = middle;
        while (down > begin + 1 && coordinateAt(down - 1) == coordinateAt(down))
            --down;
        std::int32_t up = middle;
        while (up < end - 1 && coordinateAt(up - 1) == coordinateAt(up))
            ++up;
        const bool canGoDown = coordinateAt(down - 1) != coordinateAt(down);
        const bool canGoUp = coordinateAt(up - 1) != coordinateAt(up);
        if (canGoDown && (!canGoUp || middle - down <= up - middle))
            return down;
        if (canGoUp)
            return up;
        return middle;
    }

    // A leaf: the approximate minimum degree on the leaf and its halo (the numbered neighbours), the halo last.
    void orderLeaf(std::int32_t begin, std::int32_t end)
    {
        if (begin == end)
            return;
        const std::int64_t regionLabel = labelOf(begin, end);
        std::vector<std::int32_t> halo; // bounded by the leaf size times the degree
        for (std::int32_t k = begin; k < end; ++k)
        {
            for (const std::int32_t neighbour : m_graph.neighbours(*work(k)))
            {
                if (label(neighbour) == NUMBERED)
                    halo.push_back(neighbour);
            }
        }
        std::sort(halo.begin(), halo.end());
        halo.erase(std::unique(halo.begin(), halo.end()), halo.end());
        const auto leafCount = static_cast<std::size_t>(end - begin);
        std::vector<std::pair<std::int32_t, std::int32_t>> localOf; // (node, local index), sorted by node
        for (std::size_t k = 0; k < leafCount; ++k)
            localOf.emplace_back(*work(begin + static_cast<std::int32_t>(k)), static_cast<std::int32_t>(k));
        for (std::size_t h = 0; h < halo.size(); ++h)
            localOf.emplace_back(halo[h], static_cast<std::int32_t>(leafCount + h));
        std::sort(localOf.begin(), localOf.end());
        const auto local = [&](std::int32_t node)
        { return std::lower_bound(localOf.begin(), localOf.end(), std::make_pair(node, -1))->second; };

        // Edges leaf-leaf and leaf-halo (the halo-halo edges change no degree of a leaf node).
        const std::size_t localCount = localOf.size();
        std::vector<std::vector<std::int32_t>> adjacency(localCount);
        for (std::size_t k = 0; k < leafCount; ++k)
        {
            const std::int32_t node = *work(begin + static_cast<std::int32_t>(k));
            for (const std::int32_t neighbour : m_graph.neighbours(node))
            {
                const std::int64_t neighbourLabel = label(neighbour);
                if (neighbourLabel == regionLabel)
                    adjacency[k].push_back(local(neighbour));
                else if (neighbourLabel == NUMBERED)
                {
                    const std::int32_t h = local(neighbour);
                    adjacency[k].push_back(h);
                    adjacency[static_cast<std::size_t>(h)].push_back(static_cast<std::int32_t>(k));
                }
            }
        }
        std::vector<std::int64_t> offsets(localCount + 1, 0);
        std::vector<std::int32_t> neighbours;
        std::vector<std::int32_t> localWeights(localCount);
        for (std::size_t u = 0; u < localCount; ++u)
        {
            std::sort(adjacency[u].begin(), adjacency[u].end());
            neighbours.insert(neighbours.end(), adjacency[u].begin(), adjacency[u].end());
            offsets[u + 1] = static_cast<std::int64_t>(neighbours.size());
        }
        for (const auto& [node, index] : localOf)
            localWeights[static_cast<std::size_t>(index)] = static_cast<std::int32_t>(weight(node));
        std::vector<std::int32_t> trailing(halo.size());
        for (std::size_t h = 0; h < halo.size(); ++h)
            trailing[h] = static_cast<std::int32_t>(leafCount + h);
        const NodeGraph leafGraph =
            NodeGraph::fromAdjacency(static_cast<std::int32_t>(localCount), offsets, neighbours, m_budget);
        const NodePermutation leafOrder =
            ApproximateMinimumDegree(m_leafOptions).order(leafGraph, localWeights, trailing, m_budget);
        std::vector<std::int32_t> globalOf(leafCount);
        for (std::size_t k = 0; k < leafCount; ++k)
            globalOf[k] = *work(begin + static_cast<std::int32_t>(k));
        for (std::size_t k = 0; k < leafCount; ++k)
        {
            const std::int32_t node =
                globalOf[static_cast<std::size_t>(leafOrder.nodeAt(static_cast<std::int32_t>(k)))];
            m_order[static_cast<std::size_t>(begin) + k] = node;
            label(node) = NUMBERED;
        }
        m_control.advance(leafCount);
    }

    const base::OperationControl m_control;
    const NodeGraph& m_graph;
    std::span<const std::int32_t> m_weights;
    std::span<const base::Vec3> m_coordinates;
    const DissectionOptions& m_options;
    const MinimumDegreeOptions& m_leafOptions;
    base::MemoryBudget& m_budget;
    base::ThreadPool* m_pool;
    const std::int64_t m_labelBase;
    base::BudgetedBuffer<std::int64_t> m_label;
    base::BudgetedBuffer<std::int32_t> m_work;
    base::BudgetedBuffer<std::int32_t> m_scratch;
    base::BudgetedBuffer<std::int32_t> m_order;
};

} // namespace

GeometricDissection::GeometricDissection(DissectionOptions options, MinimumDegreeOptions leafOptions)
    : m_options(options), m_leafOptions(leafOptions)
{
    if (options.leafNodeCount < 1)
        throw std::invalid_argument(
            base::concatenate({"GeometricDissection: expected a leaf size of at least 1, found ",
                               std::to_string(options.leafNodeCount)}));
}

NodePermutation GeometricDissection::order(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                           std::span<const base::Vec3> coordinates,
                                           std::span<const std::int32_t> trailingNodes, base::MemoryBudget& budget,
                                           base::ThreadPool* pool, const base::OperationControl& control) const
{
    const auto n = static_cast<std::size_t>(graph.nodeCount());
    if (coordinates.size() != n)
        throw std::invalid_argument(
            base::concatenate({"GeometricDissection: expected one coordinate per node (", std::to_string(n),
                               "), found ", std::to_string(coordinates.size())}));
    if (!nodeWeights.empty() && nodeWeights.size() != n)
        throw std::invalid_argument(
            base::concatenate({"GeometricDissection: expected one weight per node (", std::to_string(n), "), found ",
                               std::to_string(nodeWeights.size())}));
    for (std::size_t k = 0; k < nodeWeights.size(); ++k)
    {
        if (nodeWeights[k] < 1)
            throw std::invalid_argument(
                base::concatenate({"GeometricDissection: expected node weights >= 1, found ",
                                   std::to_string(nodeWeights[k]), " at node ", std::to_string(k)}));
    }
    const TrailingSet trailing(graph.nodeCount(), trailingNodes, budget);
    DissectionRun run(graph, nodeWeights, coordinates, trailing, m_options, m_leafOptions, budget, pool, control);
    return NodePermutation(run.run(), budget);
}

} // namespace oos::linalg
