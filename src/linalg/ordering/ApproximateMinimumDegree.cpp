#include "oos/linalg/ordering/ApproximateMinimumDegree.hpp"

#include "TrailingSet.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"

#include <algorithm>
#include <cmath>
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

constexpr std::int64_t EMPTY = -1;

// A node index stored where a position or EMPTY could be: −i − 2 is ≤ −2, and flip(flip(i)) = i (the paper's code
// uses the same trick to mark absorbed nodes and the list heads during garbage collection).
[[nodiscard]] constexpr std::int64_t flip(std::int64_t i) noexcept
{
    return -i - 2;
}

enum class NodeStatus : std::uint8_t
{
    Variable, // not eliminated: a principal variable, its list holds elements then variables
    Element,  // eliminated: its list Le holds the variables of its clique
    Dead      // merged into a supervariable, mass-eliminated, absorbed into another element, or dense
};

// A binary min-heap of nodes keyed by (degree, node): the pivot of least degree, ties by the smaller index
// (every tie broken by index). Indexed, so that a node's key changes or the node leaves in O(log n).
class IndexedMinHeap
{
public:
    IndexedMinHeap(std::int32_t nodeCount, base::MemoryBudget& budget)
        : m_heap(budget, static_cast<std::size_t>(nodeCount), "minimum degree: heap"),
          m_position(budget, static_cast<std::size_t>(nodeCount), "minimum degree: heap positions"),
          m_key(budget, static_cast<std::size_t>(nodeCount), "minimum degree: heap keys")
    {
        std::fill_n(m_position.data(), m_position.size(), -1);
    }

    [[nodiscard]] bool isEmpty() const noexcept { return m_size == 0; }

    // Inserts the node, or changes its key.
    void set(std::int32_t node, std::int64_t key)
    {
        const auto n = static_cast<std::size_t>(node);
        std::int32_t at = m_position[n];
        if (at < 0)
        {
            at = m_size++;
            m_heap[static_cast<std::size_t>(at)] = node;
            m_position[n] = at;
            m_key[n] = key;
            siftUp(at);
            return;
        }
        const std::int64_t old = m_key[n];
        m_key[n] = key;
        if (key < old)
            siftUp(at);
        else
            siftDown(at);
    }

    void remove(std::int32_t node)
    {
        const std::int32_t at = m_position[static_cast<std::size_t>(node)];
        if (at < 0)
            return;
        m_position[static_cast<std::size_t>(node)] = -1;
        --m_size;
        if (at == m_size)
            return;
        const std::int32_t last = m_heap[static_cast<std::size_t>(m_size)];
        m_heap[static_cast<std::size_t>(at)] = last;
        m_position[static_cast<std::size_t>(last)] = at;
        siftUp(at);
        siftDown(m_position[static_cast<std::size_t>(last)]);
    }

    [[nodiscard]] std::int32_t popMinimum()
    {
        const std::int32_t node = m_heap[0];
        remove(node);
        return node;
    }

private:
    [[nodiscard]] bool isBefore(std::int32_t a, std::int32_t b) const noexcept
    {
        const std::int64_t keyA = m_key[static_cast<std::size_t>(a)];
        const std::int64_t keyB = m_key[static_cast<std::size_t>(b)];
        return keyA < keyB || (keyA == keyB && a < b);
    }
    void place(std::int32_t at, std::int32_t node) noexcept
    {
        m_heap[static_cast<std::size_t>(at)] = node;
        m_position[static_cast<std::size_t>(node)] = at;
    }
    void siftUp(std::int32_t at) noexcept
    {
        const std::int32_t node = m_heap[static_cast<std::size_t>(at)];
        while (at > 0)
        {
            const std::int32_t parent = (at - 1) / 2;
            if (!isBefore(node, m_heap[static_cast<std::size_t>(parent)]))
                break;
            place(at, m_heap[static_cast<std::size_t>(parent)]);
            at = parent;
        }
        place(at, node);
    }
    void siftDown(std::int32_t at) noexcept
    {
        const std::int32_t node = m_heap[static_cast<std::size_t>(at)];
        while (true)
        {
            std::int32_t child = 2 * at + 1;
            if (child >= m_size)
                break;
            if (child + 1 < m_size &&
                isBefore(m_heap[static_cast<std::size_t>(child) + 1], m_heap[static_cast<std::size_t>(child)]))
                ++child;
            if (!isBefore(m_heap[static_cast<std::size_t>(child)], node))
                break;
            place(at, m_heap[static_cast<std::size_t>(child)]);
            at = child;
        }
        place(at, node);
    }

    base::BudgetedBuffer<std::int32_t> m_heap;
    base::BudgetedBuffer<std::int32_t> m_position; // -1 when the node is not in the heap
    base::BudgetedBuffer<std::int64_t> m_key;
    std::int32_t m_size = 0;
};

// The elimination on the quotient graph (paper §3–§5). The arrays follow the paper's code: pe (start of a node's list
// in iw, or flip(absorbing node), or EMPTY), len (list length), elen (number of elements at the head of a variable's
// list), nv (weight of a principal supervariable; 0 when dead; negated while the variable is in the new element), w
// (the |Le \ Lp| counters of the elements, relative to the flag wflg), degree (approximate external degree of a
// variable, |Le| of an element).
class QuotientGraphElimination
{
public:
    QuotientGraphElimination(const NodeGraph& graph, std::span<const std::int32_t> weights, const TrailingSet& trailing,
                             const MinimumDegreeOptions& options, base::MemoryBudget& budget)
        : m_n(graph.nodeCount()),
          m_trailing(trailing),
          m_options(options),
          m_iwLength(graph.adjacencySize() + graph.adjacencySize() / 5 + 2 * static_cast<std::int64_t>(m_n) + 1),
          m_iw(budget, static_cast<std::size_t>(m_iwLength), "minimum degree: quotient graph"),
          m_pe(budget, count(), "minimum degree: list starts"),
          m_len(budget, count(), "minimum degree: list lengths"),
          m_elen(budget, count(), "minimum degree: element counts"),
          m_nv(budget, count(), "minimum degree: supervariable weights"),
          m_degree(budget, count(), "minimum degree: degrees"),
          m_w(budget, count(), "minimum degree: element counters"),
          m_mark(budget, count(), "minimum degree: marks"),
          m_status(budget, count(), "minimum degree: status"),
          m_pivotStep(budget, count(), "minimum degree: pivot steps"),
          m_heap(m_n, budget)
    {
        std::copy(graph.adjacency().begin(), graph.adjacency().end(), m_iw.data());
        m_pfree = graph.adjacencySize();
        const double denseLimit = std::max(static_cast<double>(ApproximateMinimumDegree::DENSE_MIN_NEIGHBOURS),
                                           m_options.denseFactor * std::sqrt(static_cast<double>(m_n)));
        for (std::int32_t i = 0; i < m_n; ++i)
        {
            const auto k = static_cast<std::size_t>(i);
            m_len[k] = graph.degree(i);
            m_pe[k] = m_len[k] > 0 ? graph.offsets()[k] : EMPTY; // a list of length 0 must not have a start
            m_nv[k] = weights.empty() ? 1 : weights[k];
            m_w[k] = 1;
            m_pivotStep[k] = -1;
            m_status[k] = NodeStatus::Variable;
            m_totalWeight += m_nv[k];
        }
        for (std::int32_t i = 0; i < m_n; ++i)
        {
            std::int64_t degree = 0;
            for (const std::int32_t j : graph.neighbours(i))
                degree += m_nv[static_cast<std::size_t>(j)];
            m_degree[static_cast<std::size_t>(i)] = degree;
        }
        for (std::int32_t i = 0; i < m_n; ++i)
        {
            const auto k = static_cast<std::size_t>(i);
            if (m_trailing.contains(i))
                continue;
            if (static_cast<double>(m_len[k]) > denseLimit)
            {
                // Paper §5: a dense node leaves the graph (weight 0: its neighbours skip it) and is ordered last.
                m_denseNodes.push_back(i);
                m_status[k] = NodeStatus::Dead;
                m_nv[k] = 0;
                m_pe[k] = EMPTY;
                m_len[k] = 0;
                continue;
            }
            m_interiorWeight += m_nv[k];
            m_heap.set(i, m_degree[k]);
        }
    }

    // Runs the elimination and returns the order: the interior nodes, the dense nodes, then the trailing nodes.
    // The control is checked before every pivot (one pivot is at most the work of its element), and advanced by the
    // weight eliminated; the dense and trailing nodes are counted at the end, so the total is the sum of the weights.
    [[nodiscard]] base::BudgetedBuffer<std::int32_t> run(base::MemoryBudget& budget,
                                                         const base::OperationControl& control)
    {
        std::int64_t reportedWeight = 0;
        while (m_eliminatedWeight < m_interiorWeight)
        {
            control.throwIfCancelled();
            OOS_ASSERT(!m_heap.isEmpty());
            eliminate(m_heap.popMinimum());
            control.advance(static_cast<std::uint64_t>(m_eliminatedWeight - reportedWeight));
            reportedWeight = m_eliminatedWeight;
        }
        base::BudgetedBuffer<std::int32_t> order = assembleOrder(budget);
        control.advance(static_cast<std::uint64_t>(m_totalWeight - reportedWeight));
        return order;
    }

private:
    [[nodiscard]] std::size_t count() const noexcept { return static_cast<std::size_t>(m_n); }
    [[nodiscard]] std::int64_t& pe(std::int64_t i) noexcept { return m_pe[static_cast<std::size_t>(i)]; }
    [[nodiscard]] std::int64_t& len(std::int64_t i) noexcept { return m_len[static_cast<std::size_t>(i)]; }
    [[nodiscard]] std::int64_t& elen(std::int64_t i) noexcept { return m_elen[static_cast<std::size_t>(i)]; }
    [[nodiscard]] std::int64_t& nv(std::int64_t i) noexcept { return m_nv[static_cast<std::size_t>(i)]; }
    [[nodiscard]] std::int64_t& degree(std::int64_t i) noexcept { return m_degree[static_cast<std::size_t>(i)]; }
    [[nodiscard]] std::int64_t& w(std::int64_t i) noexcept { return m_w[static_cast<std::size_t>(i)]; }
    [[nodiscard]] std::int32_t& iw(std::int64_t p) noexcept { return m_iw[static_cast<std::size_t>(p)]; }
    [[nodiscard]] NodeStatus& status(std::int64_t i) noexcept { return m_status[static_cast<std::size_t>(i)]; }

    // One pivot step: the variable `me` becomes an element.
    void eliminate(std::int32_t me)
    {
        m_pivotStep[static_cast<std::size_t>(me)] = m_stepCount++;
        const std::int64_t elenme = elen(me);
        std::int64_t nvpiv = nv(me);
        m_eliminatedWeight += nvpiv;
        nv(me) = -nvpiv;
        std::int64_t degme = 0;
        std::int64_t pme1 = 0;
        std::int64_t pme2 = 0;
        constructNewElement(me, elenme, degme, pme1, pme2);
        countExternalVariables(pme1, pme2);
        updateDegrees(me, degme, nvpiv, pme1, pme2);
        degree(me) = degme;
        detectSupervariables();
        finaliseDegrees(me, elenme, degme, nvpiv, pme1, pme2);
        m_largestElement = std::max(m_largestElement, degme);
        m_wflg += m_largestElement + 1; // every counter w(e) set in this step is now below the flag
    }

    // Paper §3, step 2: Lme = (A_me ∪ ⋃_{e ∈ E_me} Le) \ me, each variable once (marked by a negative nv), and the
    // elements of E_me absorbed into me. Built in place when me has no element, at the end of iw otherwise.
    void constructNewElement(std::int32_t me, std::int64_t elenme, std::int64_t& degme, std::int64_t& pme1,
                             std::int64_t& pme2)
    {
        if (elenme == 0)
        {
            pme1 = pe(me);
            pme2 = pme1 - 1;
            for (std::int64_t p = pme1; p < pme1 + len(me); ++p)
            {
                const std::int32_t i = iw(p);
                const std::int64_t nvi = nv(i);
                if (nvi > 0)
                {
                    degme += nvi;
                    nv(i) = -nvi;
                    iw(++pme2) = i;
                    m_heap.remove(i);
                }
            }
        }
        else
        {
            std::int64_t p = pe(me);
            pme1 = m_pfree;
            const std::int64_t slenme = len(me) - elenme;
            for (std::int64_t knt1 = 1; knt1 <= elenme + 1; ++knt1)
            {
                std::int64_t e = 0;
                std::int64_t pj = 0;
                std::int64_t ln = 0;
                if (knt1 > elenme)
                {
                    e = me; // the variables of me itself
                    pj = p;
                    ln = slenme;
                }
                else
                {
                    e = iw(p++);
                    pj = pe(e);
                    ln = len(e);
                }
                for (std::int64_t knt2 = 1; knt2 <= ln; ++knt2)
                {
                    const std::int32_t i = iw(pj++);
                    const std::int64_t nvi = nv(i);
                    if (nvi <= 0)
                        continue;
                    if (m_pfree >= m_iwLength)
                    {
                        // Save the unread parts of me's list and of e's list, compress, then continue.
                        pe(me) = p;
                        len(me) -= knt1;
                        if (len(me) == 0)
                            pe(me) = EMPTY;
                        pe(e) = pj;
                        len(e) = ln - knt2;
                        if (len(e) == 0)
                            pe(e) = EMPTY;
                        collectGarbage(pme1);
                        // The storage of the lists never exceeds the initial adjacency plus one entry per node
                        // (paper §3), less than the workspace.
                        OOS_ASSERT(m_pfree < m_iwLength);
                        pj = pe(e);
                        p = pe(me);
                    }
                    degme += nvi;
                    nv(i) = -nvi;
                    iw(m_pfree++) = i;
                    m_heap.remove(i);
                }
                if (e != me)
                {
                    pe(e) = flip(me); // e is absorbed into me
                    w(e) = 0;
                    status(e) = NodeStatus::Dead;
                }
            }
            pme2 = m_pfree - 1;
        }
        pe(me) = pme1;
        len(me) = pme2 - pme1 + 1;
        status(me) = NodeStatus::Element;
    }

    // Compresses iw: every live list moves to the front, in storage order, then the partial new element
    // [pme1, pfree) after them. The head of each list is replaced by flip(node) during the scan (every list with a
    // start has at least one entry), so the scan finds the lists without another array.
    void collectGarbage(std::int64_t& pme1)
    {
        for (std::int32_t j = 0; j < m_n; ++j)
        {
            const std::int64_t pn = pe(j);
            if (pn >= 0)
            {
                pe(j) = iw(pn);
                iw(pn) = static_cast<std::int32_t>(flip(j));
            }
        }
        std::int64_t psrc = 0;
        std::int64_t pdst = 0;
        const std::int64_t pend = pme1 - 1;
        while (psrc <= pend)
        {
            const std::int64_t j = flip(iw(psrc++));
            if (j < 0)
                continue;
            iw(pdst) = static_cast<std::int32_t>(pe(j));
            pe(j) = pdst++;
            for (std::int64_t knt3 = 0; knt3 + 1 < len(j); ++knt3)
                iw(pdst++) = iw(psrc++);
        }
        const std::int64_t newStart = pdst;
        for (psrc = pme1; psrc < m_pfree; ++psrc)
            iw(pdst++) = iw(psrc);
        pme1 = newStart;
        m_pfree = pdst;
    }

    // Paper §4, step 3: w(e) − wflg = |Le \ Lme| for every element e adjacent to a variable of Lme.
    void countExternalVariables(std::int64_t pme1, std::int64_t pme2)
    {
        for (std::int64_t pme = pme1; pme <= pme2; ++pme)
        {
            const std::int32_t i = iw(pme);
            const std::int64_t eln = elen(i);
            if (eln <= 0)
                continue;
            const std::int64_t nvi = -nv(i);
            const std::int64_t wnvi = m_wflg - nvi;
            for (std::int64_t p = pe(i); p < pe(i) + eln; ++p)
            {
                const std::int32_t e = iw(p);
                std::int64_t we = w(e);
                if (we >= m_wflg)
                    we -= nvi;
                else if (we != 0)
                    we = degree(e) + wnvi; // first visit in this step: |Le| − |i|
                w(e) = we;
            }
        }
    }

    // Paper §4, step 4: the degree bound of each variable of Lme, its list pruned (dead and absorbed entries out, me
    // first), aggressive absorption, mass elimination of the variables left with no other neighbour than me.
    void updateDegrees(std::int32_t me, std::int64_t& degme, std::int64_t& nvpiv, std::int64_t pme1, std::int64_t pme2)
    {
        m_candidates.clear();
        for (std::int64_t pme = pme1; pme <= pme2; ++pme)
        {
            const std::int32_t i = iw(pme);
            const std::int64_t p1 = pe(i);
            const std::int64_t p2 = p1 + elen(i) - 1;
            std::int64_t pn = p1;
            std::uint64_t hash = 0;
            std::int64_t deg = 0;
            for (std::int64_t p = p1; p <= p2; ++p)
            {
                const std::int32_t e = iw(p);
                const std::int64_t we = w(e);
                if (we == 0)
                    continue; // absorbed before
                const std::int64_t dext = we - m_wflg;
                if (dext > 0 || !m_options.hasAggressiveAbsorption)
                {
                    deg += dext;
                    iw(pn++) = e;
                    hash += static_cast<std::uint64_t>(e);
                }
                else
                {
                    pe(e) = flip(me); // Le ⊆ Lme: e is absorbed into me
                    w(e) = 0;
                    status(e) = NodeStatus::Dead;
                }
            }
            elen(i) = pn - p1 + 1; // me will be prepended
            const std::int64_t p3 = pn;
            const std::int64_t p4 = p1 + len(i);
            for (std::int64_t p = p2 + 1; p < p4; ++p)
            {
                const std::int32_t j = iw(p);
                const std::int64_t nvj = nv(j);
                if (nvj > 0) // a live variable outside Lme
                {
                    deg += nvj;
                    iw(pn++) = j;
                    hash += static_cast<std::uint64_t>(j);
                }
            }
            if (elen(i) == 1 && p3 == pn && !m_trailing.contains(i))
            {
                // Mass elimination (paper §5): i is adjacent to me only, so it is eliminated with it.
                pe(i) = flip(me);
                const std::int64_t nvi = -nv(i);
                degme -= nvi;
                nvpiv += nvi;
                m_eliminatedWeight += nvi;
                nv(i) = 0;
                elen(i) = 0;
                len(i) = 0;
                status(i) = NodeStatus::Dead;
                continue;
            }
            degree(i) = std::min(degree(i), deg);
            // me first: the first element moves to the start of the variables, the first variable to the end.
            iw(pn) = iw(p3);
            iw(p3) = iw(p1);
            iw(p1) = me;
            len(i) = pn - p1 + 1;
            m_candidates.emplace_back(hash, i);
        }
    }

    // Paper §5: supervariable detection. Variables of Lme with the same hash of their adjacency are compared; a
    // variable with the same lists as an earlier one is merged into it (weights summed, the later one dead).
    void detectSupervariables()
    {
        std::sort(m_candidates.begin(), m_candidates.end());
        for (std::size_t first = 0; first < m_candidates.size();)
        {
            std::size_t end = first + 1;
            while (end < m_candidates.size() && m_candidates[end].first == m_candidates[first].first)
                ++end;
            for (std::size_t a = first; a + 1 < end; ++a)
            {
                const std::int32_t i = m_candidates[a].second;
                if (status(i) != NodeStatus::Variable)
                    continue;
                ++m_stamp;
                for (std::int64_t p = pe(i); p < pe(i) + len(i); ++p)
                    m_mark[static_cast<std::size_t>(iw(p))] = m_stamp;
                for (std::size_t b = a + 1; b < end; ++b)
                {
                    const std::int32_t j = m_candidates[b].second;
                    if (status(j) != NodeStatus::Variable || len(j) != len(i) || elen(j) != elen(i) ||
                        m_trailing.contains(i) != m_trailing.contains(j))
                        continue;
                    bool isSame = true;
                    for (std::int64_t p = pe(j); p < pe(j) + len(j) && isSame; ++p)
                        isSame = m_mark[static_cast<std::size_t>(iw(p))] == m_stamp;
                    if (!isSame)
                        continue;
                    nv(i) += nv(j); // both negative while in Lme
                    nv(j) = 0;
                    elen(j) = 0;
                    len(j) = 0;
                    pe(j) = flip(i);
                    status(j) = NodeStatus::Dead;
                }
            }
            first = end;
        }
    }

    // Paper §4, step 6: the final degree bound of each principal variable of Lme, back into the heap; Lme compacted.
    void finaliseDegrees(std::int32_t me, std::int64_t elenme, std::int64_t degme, std::int64_t nvpiv,
                         std::int64_t pme1, std::int64_t pme2)
    {
        const std::int64_t remainingWeight = m_totalWeight - m_eliminatedWeight;
        std::int64_t p = pme1;
        for (std::int64_t pme = pme1; pme <= pme2; ++pme)
        {
            const std::int32_t i = iw(pme);
            const std::int64_t nvi = -nv(i);
            if (nvi <= 0)
                continue; // merged or mass-eliminated
            nv(i) = nvi;
            const std::int64_t deg = std::min(degree(i) + degme - nvi, remainingWeight - nvi);
            degree(i) = deg;
            if (!m_trailing.contains(i))
                m_heap.set(i, deg);
            iw(p++) = i;
        }
        nv(me) = nvpiv;
        len(me) = p - pme1;
        if (len(me) == 0)
        {
            pe(me) = EMPTY;
            w(me) = 0;
        }
        if (elenme != 0)
            m_pfree = p; // the tail of the new element is free again
    }

    // The order: interior nodes by the step of the pivot that eliminated them (a merged or mass-eliminated node with
    // its pivot), ties by index; then the dense nodes by index; then the trailing nodes as given.
    [[nodiscard]] base::BudgetedBuffer<std::int32_t> assembleOrder(base::MemoryBudget& budget)
    {
        base::BudgetedBuffer<std::int64_t> stepOf(budget, count(), "minimum degree: steps");
        base::BudgetedBuffer<std::uint8_t> isDense(budget, count(), "minimum degree: dense flags");
        for (const std::int32_t node : m_denseNodes)
            isDense[static_cast<std::size_t>(node)] = 1;
        base::BudgetedBuffer<std::int64_t> stepStart(budget, static_cast<std::size_t>(m_stepCount) + 1,
                                                     "minimum degree: step starts");
        for (std::int32_t j = 0; j < m_n; ++j)
        {
            const auto k = static_cast<std::size_t>(j);
            if (m_trailing.contains(j) || isDense[k] != 0)
                continue;
            stepOf[k] = resolveStep(j);
            ++stepStart[static_cast<std::size_t>(stepOf[k]) + 1];
        }
        for (std::size_t s = 0; s < static_cast<std::size_t>(m_stepCount); ++s)
            stepStart[s + 1] += stepStart[s];
        base::BudgetedBuffer<std::int32_t> order(budget, count(), "node permutation");
        for (std::int32_t j = 0; j < m_n; ++j)
        {
            const auto k = static_cast<std::size_t>(j);
            if (m_trailing.contains(j) || isDense[k] != 0)
                continue;
            order[static_cast<std::size_t>(stepStart[static_cast<std::size_t>(stepOf[k])]++)] = j;
        }
        std::size_t next = static_cast<std::size_t>(stepStart[static_cast<std::size_t>(m_stepCount)]);
        for (const std::int32_t node : m_denseNodes)
            order[next++] = node;
        for (const std::int32_t node : m_trailing.nodes())
            order[next++] = node;
        OOS_ASSERT(next == count());
        return order;
    }

    // The pivot step of a node: its own, or that of the node it was merged or mass-eliminated into, transitively.
    [[nodiscard]] std::int64_t resolveStep(std::int32_t node) noexcept
    {
        std::int64_t current = node;
        while (m_pivotStep[static_cast<std::size_t>(current)] < 0)
            current = flip(pe(current));
        const std::int64_t step = m_pivotStep[static_cast<std::size_t>(current)];
        // Path compression: the nodes of the chain take the step, so that a later chain stops at once.
        for (std::int64_t k = node; m_pivotStep[static_cast<std::size_t>(k)] < 0;)
        {
            const std::int64_t nextNode = flip(pe(k));
            m_pivotStep[static_cast<std::size_t>(k)] = step;
            k = nextNode;
        }
        return step;
    }

    const std::int32_t m_n;
    const TrailingSet& m_trailing;
    const MinimumDegreeOptions m_options;
    const std::int64_t m_iwLength;
    base::BudgetedBuffer<std::int32_t> m_iw;
    base::BudgetedBuffer<std::int64_t> m_pe;
    base::BudgetedBuffer<std::int64_t> m_len;
    base::BudgetedBuffer<std::int64_t> m_elen;
    base::BudgetedBuffer<std::int64_t> m_nv;
    base::BudgetedBuffer<std::int64_t> m_degree;
    base::BudgetedBuffer<std::int64_t> m_w;
    base::BudgetedBuffer<std::int64_t> m_mark;
    base::BudgetedBuffer<NodeStatus> m_status;
    base::BudgetedBuffer<std::int64_t> m_pivotStep;
    IndexedMinHeap m_heap;
    std::vector<std::pair<std::uint64_t, std::int32_t>> m_candidates; // bounded by the largest element (a front)
    std::vector<std::int32_t> m_denseNodes; // at most √n / denseFactor · ... nodes: bounded by the adjacency
    std::int64_t m_pfree = 0;
    std::int64_t m_wflg = 2;
    std::int64_t m_stamp = 0;
    std::int64_t m_largestElement = 0;
    std::int64_t m_stepCount = 0;
    std::int64_t m_totalWeight = 0;
    std::int64_t m_interiorWeight = 0;
    std::int64_t m_eliminatedWeight = 0;
};

} // namespace

NodePermutation ApproximateMinimumDegree::order(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                                std::span<const std::int32_t> trailingNodes, base::MemoryBudget& budget,
                                                const base::OperationControl& control) const
{
    const std::int32_t n = graph.nodeCount();
    if (!nodeWeights.empty() && nodeWeights.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument(
            base::concatenate({"ApproximateMinimumDegree: expected one weight per node (", std::to_string(n),
                               "), found ", std::to_string(nodeWeights.size())}));
    for (std::size_t k = 0; k < nodeWeights.size(); ++k)
    {
        if (nodeWeights[k] < 1)
            throw std::invalid_argument(
                base::concatenate({"ApproximateMinimumDegree: expected node weights >= 1, found ",
                                   std::to_string(nodeWeights[k]), " at node ", std::to_string(k)}));
    }
    const TrailingSet trailing(n, trailingNodes, budget);
    QuotientGraphElimination elimination(graph, nodeWeights, trailing, m_options, budget);
    return NodePermutation(elimination.run(budget, control), budget);
}

} // namespace oos::linalg
