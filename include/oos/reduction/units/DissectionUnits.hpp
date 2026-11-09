#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// What the dissection reads of a model: its node graph with every connection as edges (the element couplings and those
// of the eliminated relations: RigidBodyStar, RigidInterpolation, the fasteners' piercing points, the MPC equations;
// CondensationPlanner's graph), the DOFs of each node not held in every subcase, the positions (the geometric
// dissection), the nodes of every element with its ID (assembly::ElementConnectivity), the elements to place (those of
// the units being replaced, sorted) and the nodes the units must keep on their boundaries (the FreeBodySupport points:
// ordered last, so that they are interface points). Views: the caller owns everything.
struct DissectionInputs
{
    const linalg::NodeGraph* graph = nullptr;
    std::span<const std::int32_t> dofCounts;  // per node
    std::span<const base::Vec3> positions;    // per node
    linalg::NodeGroupList elementNodes;       // per element of two nodes or more: its nodes
    std::span<const std::int32_t> elementIds; // per group of elementNodes
    // The elements of one node: (elementId, node) pairs.
    std::span<const std::int32_t> pointElementIds;
    std::span<const std::int32_t> pointElementNodes;
    std::span<const base::EntityId> placedElements; // sorted: the elements the units are made of
    std::span<const std::int32_t> interfaceNodes;   // node indices kept on the interface
};

// The bound of a unit: the bytes of its recovery, the partial factorisation made again for it (its interior
// factor, the working storage of its fronts and its K*, at 8 bytes a value), estimated on the subtree of the
// dissection it comes from.
struct DissectionRequest
{
    std::uint64_t maxUnitBytes = 0;
};

// One unit of the dissection: a subtree of the elimination tree of the model, or a few consecutive sibling subtrees
// (the small ones merged), and the elements placed in it.
struct DissectionUnit
{
    std::vector<base::EntityId> elementIds; // sorted
    std::int32_t firstSupernode = 0;        // its supernodes [firstSupernode, lastSupernode] in the postorder
    std::int32_t lastSupernode = 0;
    std::int64_t interiorDofCount = 0; // the columns of its supernodes
    std::int64_t boundaryDofCount = 0; // the rows of its roots below them (the order of its K*)
    std::int64_t factorValueCount = 0; // its panels
    double flopCount = 0.0;
    std::int64_t stackPeakValues = 0; // the working storage of its factorisation without the factor (Liu's stack)
    std::uint64_t recoveryBytes = 0;  // the estimate it is bounded by
};

// The part of the elimination tree above the units: the interface problem of a plan of these units, when it is
// ordered as the dissection orders the model.
struct DissectionInterface
{
    std::int64_t supernodeCount = 0;
    std::int64_t dofCount = 0;
    std::int64_t factorValueCount = 0;
    double flopCount = 0.0;
    std::int64_t largestFrontValues = 0;
};

// The sizes of one depth of the elimination tree (the root at depth 0), for the reports and the studies of the bound.
struct DissectionLevel
{
    std::int32_t depth = 0;
    std::int64_t supernodeCount = 0;
    std::int64_t factorValueCount = 0;
    double flopCount = 0.0;
    std::int64_t largestFrontValues = 0;
};

// The automatic units of a model from its nested dissection: the units follow the connectivity of the model rather
// than its components, so that the fasteners are interior to the units wherever the dissection allows and the
// interface is the top separators instead of the cliques of the fastener points.
//
// Method. The node graph is ordered by the geometric nested dissection (linalg::GeometricDissection, the ordering of
// the full path's large models; the interface nodes last) and analysed (linalg::SymbolicFactor). Per supernode, in
// postorder (children before their parent), its subtree's panels, flops and the peak of the working storage of a
// multifrontal factorisation of the subtree (Liu, "The multifrontal method for sparse matrix solution", SIAM Review 34,
// 1992: the front of s, c(c + r) + r² values beside the update matrices of its children, or the peak of a child beside
// the updates of the children before it). A subtree is admissible when its recovery, 8·(panels + stack peak) bytes, is
// at most DissectionRequest::maxUnitBytes; the units are the maximal admissible subtrees (an admissible subtree whose
// parent is not, or a root), then the consecutive admissible siblings of one parent merged while their sum stays
// admissible (a subtree of a few nodes beside a large sibling would be a unit of its own otherwise). Every supernode
// outside the units is the interface's.
//
// The elements. An element couples its nodes, which lie on one path of the elimination tree; it goes to the unit of its
// first eliminated node. One whose nodes are all on the interface goes to the unit that already has the most of them
// (ties: the first unit), or, when none has any, to the first unit below its first node's supernode, or the first unit:
// its nodes become boundary points of that unit (its stiffness enters that unit's K* on its boundary alone). Only the
// elements of `placedElements` are placed (the connections follow by the cluster absorption of the plan: a fastener
// whose two sides are in one unit is interior to it). A unit left with no element is dropped.
//
// Determinism: the ordering and the symbolic analysis depend on the graph, the weights and the positions only (the same
// units whatever the threads); the placement is a pass in element order.
//
// Thread-safety: immutable once built. Ownership: owns its units and arrays (budgeted, the units' element lists bounded
// by the model's elements: std::vector, built once).
class DissectionUnits
{
public:
    // Throws std::invalid_argument for inputs of other sizes than the graph's, MemoryBudgetError.
    DissectionUnits(const DissectionInputs& inputs, const DissectionRequest& request, base::MemoryBudget& budget,
                    base::ThreadPool* pool = nullptr);

    [[nodiscard]] const std::vector<DissectionUnit>& units() const noexcept { return m_units; }
    [[nodiscard]] const DissectionInterface& interface() const noexcept { return m_interface; }
    [[nodiscard]] const std::vector<DissectionLevel>& levels() const noexcept { return m_levels; }
    // The whole tree: its DOFs, panels and flops (the full path's factorisation in this order).
    [[nodiscard]] std::int64_t dofCount() const noexcept { return m_dofCount; }
    [[nodiscard]] std::int64_t factorValueCount() const noexcept { return m_factorValueCount; }
    [[nodiscard]] double flopCount() const noexcept { return m_flopCount; }
    // The position of each node in the elimination order of the dissection (the order of the interface problem).
    [[nodiscard]] std::span<const std::int32_t> nodePositions() const noexcept { return m_nodePositions.span(); }
    // The elements of `placedElements` placed in no unit (none unless no unit could be made).
    [[nodiscard]] std::int64_t unplacedElementCount() const noexcept { return m_unplacedElementCount; }

private:
    std::vector<DissectionUnit> m_units; // bounded by the model's elements
    DissectionInterface m_interface;
    std::vector<DissectionLevel> m_levels; // bounded by the depth of the tree
    std::int64_t m_dofCount = 0;
    std::int64_t m_factorValueCount = 0;
    double m_flopCount = 0.0;
    base::BudgetedBuffer<std::int32_t> m_nodePositions;
    std::int64_t m_unplacedElementCount = 0;
};

} // namespace oos::reduction
