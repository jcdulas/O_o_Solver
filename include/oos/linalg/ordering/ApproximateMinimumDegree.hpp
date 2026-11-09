#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"

#include <cstdint>
#include <span>

namespace oos::linalg
{

// The settings of the minimum degree ordering, fixed constants of the method (Amestoy, Davis and Duff 1996).
struct MinimumDegreeOptions
{
    // Absorb an element whose variables are all in the new element (paper §4: "aggressive absorption"); it tightens
    // the degrees and frees memory.
    bool hasAggressiveAbsorption = true;
    // A node adjacent to more than max(DENSE_MIN_NEIGHBOURS, denseFactor·√n) nodes is "dense": it is left out of the
    // elimination and ordered after the others (paper §5: a dense row would make every degree update cost O(n)).
    double denseFactor = 10.0;
};

// The approximate minimum degree ordering of Amestoy, Davis and Duff, "An approximate minimum degree ordering
// algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996, written from the paper (the ordering of small
// units and of the leaves of the dissection).
//
// The elimination is simulated on the quotient graph: an eliminated node becomes an element, the clique it creates is
// represented by the list of its variables, and the elements adjacent to it are absorbed. The degree of a variable is
// not computed exactly but bounded (paper eq. 4.1):
//   d_i = min( n_left − |i|,  d_i(old) + |Lp \ i|,  |A_i \ i| + |Lp \ i| + Σ_{e ∈ E_i \ p} |Le \ Lp| ),
// with |Le \ Lp| for every element found in one pass over Lp. Indistinguishable variables (same adjacency) are merged
// into supervariables and eliminated together ("mass elimination"), elements contained in the new element are
// absorbed ("aggressive absorption"), and dense nodes are ordered last. Degrees are weighted by the node weights (the
// DOFs of a node: each node is already a supervariable of its DOFs).
//
// Determinism: the pivot is the variable of least approximate degree, ties broken by the smaller index (an indexed
// heap, not the LIFO degree lists of the paper's code); supervariables are detected by sorting on a hash of their
// adjacency, ties by index; the result depends on the graph and the weights only.
//
// Constrained ordering: the trailing nodes stay in the graph as variables (they count in the degrees of their
// neighbours) but are never pivots and never merged with other variables; they are placed last, in the order given.
//
// Thread-safety: order() is const and may run concurrently. Memory: a workspace of about 1.2 times the adjacency plus
// ten arrays of the node count, all budgeted.
class ApproximateMinimumDegree
{
public:
    static constexpr std::int32_t DENSE_MIN_NEIGHBOURS = 16;

    explicit ApproximateMinimumDegree(MinimumDegreeOptions options = {}) : m_options(options) {}

    // nodeWeights: the weight of each node (>= 1), or empty for 1 each. trailingNodes: distinct nodes to place last,
    // in this order. Throws std::invalid_argument on a weight below 1 or a wrong size, a repeated or out-of-range
    // trailing node; MemoryBudgetError when the workspace does not fit.
    // control: checked before every pivot (OperationCancelledError); its counter is advanced by the node weight
    // ordered, the sum of the weights in all (the node count when unweighted).
    [[nodiscard]] NodePermutation order(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                        std::span<const std::int32_t> trailingNodes, base::MemoryBudget& budget,
                                        const base::OperationControl& control = {}) const;

private:
    MinimumDegreeOptions m_options;
};

} // namespace oos::linalg
