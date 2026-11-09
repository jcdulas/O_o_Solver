#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/ApproximateMinimumDegree.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"

#include <cstdint>
#include <span>

namespace oos::linalg
{

// Where the dissection may cut a region holding several components (the full path streamed: the
// components the model's structural elements connect, the connections between them its fasteners and relations):
// - Free: at the weighted median of the region's largest extent, through the components;
// - Components: between whole components only (each component on the side of its centroid's median), a region of one
//   component cut as Free cuts it;
// - Hybrid: between whole components, unless that cut's separator is more than HYBRID_SEPARATOR_RATIO times the free
//   cut's (a cut across a long stringer is a few nodes, its fasteners to the skin many) or its lighter half holds less
//   than HYBRID_LEAST_SHARE of the region (an oversized component): then the free cut.
enum class ComponentCuts : std::uint8_t
{
    Free,
    Components,
    Hybrid
};

// The settings of the geometric dissection, fixed by measurement.
struct DissectionOptions
{
    // The hybrid cut's bounds (ComponentCuts::Hybrid; set by measurement).
    static constexpr double HYBRID_SEPARATOR_RATIO = 2.0;
    static constexpr double HYBRID_LEAST_SHARE = 0.25;

    // A region of at most this many nodes is a leaf, ordered by the constrained minimum degree.
    std::int32_t leafNodeCount = 64;
    // A region of more nodes orders its two halves in parallel (when a pool is given).
    std::int32_t parallelNodeCount = 20'000;
    // The cuts of a region of several components, and the component of each node of the graph (ComponentCuts; empty:
    // every cut free). Borrowed: it must outlive the ordering.
    ComponentCuts componentCuts = ComponentCuts::Free;
    std::span<const std::int32_t> componentOfNode;
};

// Nested dissection on node coordinates (the ordering of large units). A region of nodes is cut in two
// at the weighted median of its coordinates along its largest extent (moved to the nearest change of coordinate, so
// that a row of a structured mesh is not split); the separator is the boundary of the lighter side (the nodes of one
// side adjacent to the other: a vertex cover of the edge cut), numbered after both halves, which are ordered the same
// way recursively. A disconnected region is split into its components, ordered one after the other with no
// separator. A leaf is ordered by the approximate minimum degree on the leaf plus its neighbours in the separators
// already numbered (the halo), the halo constrained last, so that the degrees of the leaf nodes count the separator
// nodes they will fill (A. George, "Nested dissection of a regular finite element mesh", SIAM J. Numer. Anal. 10,
// 1973; the constrained leaves as in CHOLMOD's partition of T. A. Davis).
//
// Determinism: sorts and ties by node index; the two halves are independent (no edge joins them), so ordering them in
// parallel changes nothing: same permutation for any number of threads.
//
// The trailing nodes are numbered last in the order given; they are part of the halo of the leaves next to them.
//
// Thread-safety: order() is const; it uses the pool given. Memory: a few arrays of the node count and the leaf
// workspaces, all budgeted.
class GeometricDissection
{
public:
    explicit GeometricDissection(DissectionOptions options = {}, MinimumDegreeOptions leafOptions = {});

    // coordinates: one position per node. Throws std::invalid_argument on a wrong size, a weight below 1, a repeated
    // or out-of-range trailing node; MemoryBudgetError when the workspace does not fit. control: checked at every
    // region (OperationCancelledError); its counter is advanced by the nodes numbered, the node count in all.
    [[nodiscard]] NodePermutation order(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                        std::span<const base::Vec3> coordinates,
                                        std::span<const std::int32_t> trailingNodes, base::MemoryBudget& budget,
                                        base::ThreadPool* pool = nullptr,
                                        const base::OperationControl& control = {}) const;

private:
    DissectionOptions m_options;
    MinimumDegreeOptions m_leafOptions;
};

} // namespace oos::linalg
