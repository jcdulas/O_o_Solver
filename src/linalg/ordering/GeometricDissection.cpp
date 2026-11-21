#include "oos/linalg/ordering/GeometricDissection.hpp"

#include "TrailingSet.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
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

// The lightest set of nodes that covers every edge of a bipartite graph (the edges a cut of the dissection removes,
// between the nodes of its two sides, here the left and the right ones): every edge has an end in the set, and no set
// that does weighs less.
//
// It is read on the minimum cut of a network: source -> each left node (capacity: its weight), left node -> right node
// for each edge (unbounded), each right node -> sink (capacity: its weight). A cut of finite capacity removes no edge
// of the graph, so it removes, for each of them, the arc from the source to its left end or the arc of its right end
// to the sink: the nodes of the arcs removed are a cover, of the weight of the cut (the weighted form of König's
// theorem, by the max-flow min-cut theorem of L. R. Ford and D. R. Fulkerson, "Flows in Networks", Princeton
// University Press, 1962). The maximum flow is found by blocking flows on the layers of a breadth-first search
// (E. A. Dinic, "Algorithm for solution of a problem of maximum flow in networks with power estimation", Soviet
// Mathematics Doklady 11, 1970).
//
// Among the lightest covers it gives the one with the most left nodes (the nodes the source reaches at the end are the
// fewest a minimum cut can leave on its side): the whole left side when both sides weigh the same.
//
// Determinism: the searches follow the order of the edges given. Thread-safety: one object per cut. Memory: 32 bytes
// per edge and 44 per node, budgeted.
class LightestCover
{
public:
    // leftWeights, rightWeights: the weight (>= 1) of each node; the right nodes joined to left node i are
    // rights[firstRight[i], firstRight[i + 1]). Throws MemoryBudgetError when the network does not fit.
    LightestCover(std::span<const std::int64_t> leftWeights, std::span<const std::int64_t> rightWeights,
                  std::span<const std::int64_t> firstRight, std::span<const std::int32_t> rights,
                  base::MemoryBudget& budget)
        : m_leftCount(leftWeights.size()),
          m_nodeCount(FIRST_LEFT + leftWeights.size() + rightWeights.size()),
          m_firstArc(budget, m_nodeCount + 1, "dissection: cut"),
          m_arcs(budget, 2 * (leftWeights.size() + rightWeights.size() + rights.size()), "dissection: cut",
                 base::FOR_OVERWRITE),
          m_layer(budget, m_nodeCount, "dissection: cut", base::FOR_OVERWRITE),
          m_queue(budget, m_nodeCount, "dissection: cut", base::FOR_OVERWRITE),
          m_nextArc(budget, m_nodeCount, "dissection: cut", base::FOR_OVERWRITE),
          m_path(budget, m_nodeCount, "dissection: cut", base::FOR_OVERWRITE)
    {
        countArcs(rightWeights.size(), firstRight, rights);
        // More than every arc from the source together: an arc of this capacity is never the one that limits a flow.
        std::int64_t unbounded = 1;
        for (const std::int64_t weight : leftWeights)
            unbounded += weight;
        for (std::size_t node = 0; node < m_nodeCount; ++node)
            m_nextArc[node] = m_firstArc[node];
        for (std::size_t i = 0; i < leftWeights.size(); ++i)
            addArc(SOURCE, left(i), leftWeights[i]);
        for (std::size_t j = 0; j < rightWeights.size(); ++j)
            addArc(right(j), SINK, rightWeights[j]);
        for (std::size_t i = 0; i < leftWeights.size(); ++i)
        {
            for (std::int64_t e = firstRight[i]; e < firstRight[i + 1]; ++e)
                addArc(left(i), right(static_cast<std::size_t>(rights[static_cast<std::size_t>(e)])), unbounded);
        }
        while (buildLayers())
            pushBlockingFlow();
    }

    // Whether left node i, or right node j, is in the cover: the left nodes the source does not reach any more (the
    // minimum cut removes their arc from the source), the right nodes it reaches (their arc to the sink).
    [[nodiscard]] bool hasLeft(std::size_t i) const noexcept { return m_layer[left(i)] == UNREACHED; }
    [[nodiscard]] bool hasRight(std::size_t j) const noexcept { return m_layer[right(j)] != UNREACHED; }

private:
    static constexpr std::size_t SOURCE = 0;
    static constexpr std::size_t SINK = 1;
    static constexpr std::size_t FIRST_LEFT = 2;
    static constexpr std::int32_t UNREACHED = -1;

    // An arc of the network. Each one is stored with its reverse, of capacity 0 at first: what an arc carries is
    // taken from its capacity and added to its reverse's, so that a later path may send it back.
    struct Arc
    {
        std::int64_t capacity;     // what the arc can still carry
        std::uint32_t to;          // the node it leads to
        std::uint32_t reverseRank; // the rank of its reverse among the arcs of `to`
    };

    [[nodiscard]] std::size_t left(std::size_t i) const noexcept { return FIRST_LEFT + i; }
    [[nodiscard]] std::size_t right(std::size_t j) const noexcept { return FIRST_LEFT + m_leftCount + j; }
    [[nodiscard]] std::int64_t reverseOf(std::int64_t arc) const noexcept
    {
        const Arc& forward = m_arcs[static_cast<std::size_t>(arc)];
        return m_firstArc[forward.to] + forward.reverseRank;
    }

    // The first arc of each node (the arcs are stored node by node): an arc and its reverse for each node of the
    // graph and each of its edges.
    void countArcs(std::size_t rightCount, std::span<const std::int64_t> firstRight,
                   std::span<const std::int32_t> rights)
    {
        m_firstArc[SOURCE + 1] = static_cast<std::int64_t>(m_leftCount);
        m_firstArc[SINK + 1] = static_cast<std::int64_t>(rightCount);
        for (std::size_t i = 0; i < m_leftCount; ++i)
            m_firstArc[left(i) + 1] = 1 + firstRight[i + 1] - firstRight[i];
        for (std::size_t j = 0; j < rightCount; ++j)
            m_firstArc[right(j) + 1] = 1;
        for (const std::int32_t j : rights)
            ++m_firstArc[right(static_cast<std::size_t>(j)) + 1];
        for (std::size_t node = 0; node < m_nodeCount; ++node)
            m_firstArc[node + 1] += m_firstArc[node];
    }

    void addArc(std::size_t from, std::size_t to, std::int64_t capacity)
    {
        const std::int64_t forward = m_nextArc[from]++;
        const std::int64_t backward = m_nextArc[to]++;
        m_arcs[static_cast<std::size_t>(forward)] = {capacity, static_cast<std::uint32_t>(to),
                                                     static_cast<std::uint32_t>(backward - m_firstArc[to])};
        m_arcs[static_cast<std::size_t>(backward)] = {0, static_cast<std::uint32_t>(from),
                                                      static_cast<std::uint32_t>(forward - m_firstArc[from])};
    }

    // The layer of each node the source reaches through arcs that can still carry flow (its distance in arcs); false
    // when it does not reach the sink: the flow is then maximum, and the nodes reached are the source's side of a
    // minimum cut.
    bool buildLayers()
    {
        for (std::size_t node = 0; node < m_nodeCount; ++node)
            m_layer[node] = UNREACHED;
        m_layer[SOURCE] = 0;
        m_queue[0] = static_cast<std::uint32_t>(SOURCE);
        std::size_t tail = 1;
        for (std::size_t head = 0; head < tail; ++head)
        {
            const std::size_t node = m_queue[head];
            for (std::int64_t a = m_firstArc[node]; a < m_firstArc[node + 1]; ++a)
            {
                const Arc& arc = m_arcs[static_cast<std::size_t>(a)];
                if (arc.capacity > 0 && m_layer[arc.to] == UNREACHED)
                {
                    m_layer[arc.to] = m_layer[node] + 1;
                    m_queue[tail++] = arc.to;
                }
            }
        }
        return m_layer[SINK] != UNREACHED;
    }

    // Sends flow along the paths that go down the layers until none is left (a blocking flow). Each path is followed
    // from the source, depth first, on an explicit stack (a path may be as long as the cut has nodes): a node from
    // which nothing more goes down is taken out of the layers, and the search steps back.
    void pushBlockingFlow()
    {
        for (std::size_t node = 0; node < m_nodeCount; ++node)
            m_nextArc[node] = m_firstArc[node];
        std::size_t pathLength = 0;
        std::size_t node = SOURCE;
        while (true)
        {
            if (node == SINK)
            {
                pushAlongPath(pathLength);
                pathLength = 0;
                node = SOURCE;
                continue;
            }
            const std::int64_t arc = nextArcDown(node);
            if (arc < m_firstArc[node + 1])
            {
                m_path[pathLength++] = arc;
                node = m_arcs[static_cast<std::size_t>(arc)].to;
                continue;
            }
            if (pathLength == 0)
                return;
            m_layer[node] = UNREACHED;
            node = m_arcs[static_cast<std::size_t>(reverseOf(m_path[--pathLength]))].to;
        }
    }

    // The next arc of `node` that can carry flow to the layer below (the arcs before it cannot any more in these
    // layers); the end of its arcs when there is none.
    [[nodiscard]] std::int64_t nextArcDown(std::size_t node) noexcept
    {
        std::int64_t& a = m_nextArc[node];
        for (; a < m_firstArc[node + 1]; ++a)
        {
            const Arc& arc = m_arcs[static_cast<std::size_t>(a)];
            if (arc.capacity > 0 && m_layer[arc.to] == m_layer[node] + 1)
                break;
        }
        return a;
    }

    // Sends along the path (its first `length` arcs, from the source to the sink) what its narrowest arc can carry.
    void pushAlongPath(std::size_t length) noexcept
    {
        std::int64_t pushed = m_arcs[static_cast<std::size_t>(m_path[0])].capacity;
        for (std::size_t k = 1; k < length; ++k)
            pushed = std::min(pushed, m_arcs[static_cast<std::size_t>(m_path[k])].capacity);
        for (std::size_t k = 0; k < length; ++k)
        {
            m_arcs[static_cast<std::size_t>(m_path[k])].capacity -= pushed;
            m_arcs[static_cast<std::size_t>(reverseOf(m_path[k]))].capacity += pushed;
        }
    }

    std::size_t m_leftCount;
    std::size_t m_nodeCount; // the source, the sink, the left nodes then the right nodes
    base::BudgetedBuffer<std::int64_t> m_firstArc;
    base::BudgetedBuffer<Arc> m_arcs;
    base::BudgetedBuffer<std::int32_t> m_layer;
    base::BudgetedBuffer<std::uint32_t> m_queue;  // the nodes reached, in the order of the breadth-first search
    base::BudgetedBuffer<std::int64_t> m_nextArc; // per node: the next arc to fill, then the next arc to try
    base::BudgetedBuffer<std::int64_t> m_path;    // the arcs of the path followed, from the source
};

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
    // it makes (separatorOfCut) and of the lighter of the two halves it leaves.
    struct Cut
    {
        std::int32_t middle = 0;
        std::int64_t separatorWeight = 0;
        std::int64_t lighterHalfWeight = 0;
    };

    // Cuts a connected region (DissectionOptions::componentCuts: at the weighted median of its largest extent, or
    // between its components); the separator of the cut is numbered last.
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
        labelSides(begin, end, middle);
        const base::BudgetedBuffer<std::uint8_t> isSeparator = separatorOfCut(begin, end, middle);
        std::int64_t separator = 0;
        std::int64_t halves[2] = {0, 0};
        for (std::int32_t k = begin; k < end; ++k)
        {
            if (isSeparator[static_cast<std::size_t>(k - begin)] != 0)
                separator += weight(*work(k));
            else
                halves[k < middle ? 0 : 1] += weight(*work(k));
        }
        return {middle, separator, std::min(halves[0], halves[1])};
    }

    // Labels the two sides of the cut at `middle` of the region's work.
    void labelSides(std::int32_t begin, std::int32_t end, std::int32_t middle) noexcept
    {
        const std::int64_t firstLabel = labelOf(begin, middle);
        const std::int64_t secondLabel = labelOf(middle, end);
        for (std::int32_t k = begin; k < end; ++k)
            label(*work(k)) = k < middle ? firstLabel : secondLabel;
    }

    // The edges a cut removes, as LightestCover takes them: the positions in the work of the nodes of each side that
    // have a neighbour across, and for each of the first side its neighbours across (their ranks in `secondPositions`
    // are seconds[firstSecond[i], firstSecond[i + 1])).
    struct CutEdges
    {
        base::BudgetedBuffer<std::int32_t> firstPositions;
        base::BudgetedBuffer<std::int32_t> secondPositions;
        base::BudgetedBuffer<std::int64_t> firstSecond;
        base::BudgetedBuffer<std::int32_t> seconds;
    };

    // A node of the second side of a cut and its rank among those of the cut.
    struct RankedNode
    {
        std::int32_t node;
        std::int32_t rank;
    };

    // The edges of the cut at `middle` of the region's work (the labels set for its two sides). isOnCut: one flag per
    // position of the region, set here for the nodes that have a neighbour across.
    CutEdges edgesOfCut(std::int32_t begin, std::int32_t end, std::int32_t middle, std::span<std::uint8_t> isOnCut)
    {
        const std::int64_t firstLabel = labelOf(begin, middle);
        const std::int64_t secondLabel = labelOf(middle, end);
        std::size_t counts[2] = {0, 0};
        for (std::int32_t k = begin; k < end; ++k)
        {
            const bool isFirst = k < middle;
            if (!hasNeighbourLabelled(*work(k), isFirst ? secondLabel : firstLabel))
                continue;
            isOnCut[static_cast<std::size_t>(k - begin)] = 1;
            ++counts[isFirst ? 0 : 1];
        }
        CutEdges edges{{m_budget, counts[0], "dissection: cut", base::FOR_OVERWRITE},
                       {m_budget, counts[1], "dissection: cut", base::FOR_OVERWRITE},
                       {m_budget, counts[0] + 1, "dissection: cut"},
                       {}};
        // The nodes of the second side sorted, to find the rank of a neighbour across.
        base::BudgetedBuffer<RankedNode> ranked(m_budget, counts[1], "dissection: cut", base::FOR_OVERWRITE);
        std::size_t firstAt = 0;
        std::size_t secondAt = 0;
        for (std::int32_t k = begin; k < end; ++k)
        {
            if (isOnCut[static_cast<std::size_t>(k - begin)] == 0)
                continue;
            if (k < middle)
            {
                edges.firstPositions[firstAt++] = k;
                continue;
            }
            ranked[secondAt] = {*work(k), static_cast<std::int32_t>(secondAt)};
            edges.secondPositions[secondAt++] = k;
        }
        std::sort(ranked.data(), ranked.data() + ranked.size(),
                  [](const RankedNode& a, const RankedNode& b) { return a.node < b.node; });
        const auto isAcross = [&](std::int32_t neighbour) { return label(neighbour) == secondLabel; };
        for (std::size_t i = 0; i < counts[0]; ++i)
        {
            const auto neighbours = m_graph.neighbours(*work(edges.firstPositions[i]));
            edges.firstSecond[i + 1] =
                edges.firstSecond[i] + std::count_if(neighbours.begin(), neighbours.end(), isAcross);
        }
        const auto edgeCount = static_cast<std::size_t>(edges.firstSecond[counts[0]]);
        edges.seconds = base::BudgetedBuffer<std::int32_t>(m_budget, edgeCount, "dissection: cut", base::FOR_OVERWRITE);
        std::size_t at = 0;
        for (std::size_t i = 0; i < counts[0]; ++i)
        {
            for (const std::int32_t neighbour : m_graph.neighbours(*work(edges.firstPositions[i])))
            {
                if (!isAcross(neighbour))
                    continue;
                const RankedNode* const found =
                    std::lower_bound(ranked.data(), ranked.data() + ranked.size(), neighbour,
                                     [](const RankedNode& entry, std::int32_t node) { return entry.node < node; });
                edges.seconds[at++] = found->rank;
            }
        }
        return edges;
    }

    // The separator of the cut at `middle` of the region's work (the labels set for its two sides): one flag per
    // position of the region. It is the lightest set of nodes that covers every edge joining the two sides
    // (LightestCover): a node of every edge the cut removes is taken, so that no edge joins what is left of the sides.
    //
    // PERFORMANCE: the nodes of one side that have a neighbour across (its boundary) are a cover too, and the
    // lightest one where the cut follows a line of the mesh. Elsewhere (a mesh whose grid points are not in rows) the
    // cut crosses elements: all the nodes of an element are neighbours, so the boundary of a side takes, of each
    // element crossed, every node it has on that side, where the lightest cover takes the lighter of its two parts,
    // and follows a line of the mesh that winds along the cut. Measured on the node graph of 2 304 quadrilateral
    // shells of the mixed hybrid formulation (a grid point, an edge and an interior weigh 3, 17 and 51 DOFs), in
    // operations of the factorisation (SymbolicFactor::flopCount): 19 G with the grid points in rows, by both rules;
    // with the grid points moved by 30 % of a cell, 240 G by the boundary of the lighter side and 19.6 G by the
    // lightest cover. On 65 536 shells of the classic formulation: 73 G in rows, 124 G and 73 G moved.
    base::BudgetedBuffer<std::uint8_t> separatorOfCut(std::int32_t begin, std::int32_t end, std::int32_t middle)
    {
        base::BudgetedBuffer<std::uint8_t> flags(m_budget, static_cast<std::size_t>(end - begin), "dissection: cut");
        const CutEdges edges = edgesOfCut(begin, end, middle, flags.span());
        const auto weightsAt = [&](const base::BudgetedBuffer<std::int32_t>& positions)
        {
            base::BudgetedBuffer<std::int64_t> weights(m_budget, positions.size(), "dissection: cut",
                                                       base::FOR_OVERWRITE);
            for (std::size_t i = 0; i < positions.size(); ++i)
                weights[i] = weight(*work(positions[i]));
            return weights;
        };
        const base::BudgetedBuffer<std::int64_t> firstWeights = weightsAt(edges.firstPositions);
        const base::BudgetedBuffer<std::int64_t> secondWeights = weightsAt(edges.secondPositions);
        const LightestCover cover(firstWeights.span(), secondWeights.span(), edges.firstSecond.span(),
                                  edges.seconds.span(), m_budget);
        for (std::size_t i = 0; i < edges.firstPositions.size(); ++i)
            flags[static_cast<std::size_t>(edges.firstPositions[i] - begin)] = cover.hasLeft(i) ? 1 : 0;
        for (std::size_t j = 0; j < edges.secondPositions.size(); ++j)
            flags[static_cast<std::size_t>(edges.secondPositions[j] - begin)] = cover.hasRight(j) ? 1 : 0;
        return flags;
    }

    // Splits the region at `middle` of its work: the separator of the cut (separatorOfCut) numbered last; the two
    // halves ordered (in parallel when large).
    void split(std::int32_t begin, std::int32_t end, std::int32_t middle)
    {
        labelSides(begin, end, middle);
        const base::BudgetedBuffer<std::uint8_t> isSeparator = separatorOfCut(begin, end, middle);
        // Stable partition into [first side][second side][separator], kept in coordinate order.
        const auto isSeparatorAt = [&](std::int32_t k)
        { return isSeparator[static_cast<std::size_t>(k - begin)] != 0; };
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
