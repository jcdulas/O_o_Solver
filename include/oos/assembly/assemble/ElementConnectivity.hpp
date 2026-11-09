#pragma once

#include "oos/assembly/assemble/ElementMatrixStore.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"

#include <cstdint>
#include <span>

namespace oos::assembly
{

// The nodes each element couples, as the groups of linalg::NodeGraph: one group per element of the set that couples two
// nodes or more, its nodes sorted and distinct (the node numbering of DofMap: grid point rows, then scalar points).
//
// The nodes are those of the DOFs of the element's matrices: the stiffness for an element that has one, the mass
// (lumped) for a mass element (a MassDofInline between two points couples them in the mass matrix, which shares the pattern
// of the stiffness). The snapshot's nodesRows column would give them for most cards without computing a matrix, but not
// for the springs, Bushing and Bar2, and not for the points an element reaches through an offset or a
// spring point; the matrices give every case the same way. The cost is a second computation of the element matrices
// (about 1 µs for a light element, 7 µs for a ShellQuad4), small against the factorisation of the same model; it is
// saved when the caller gives a store (ElementMatrixStore): the stiffness matrices computed here are kept for the
// assembly.
//
// The groups are built on the pool in waves of pieces of consecutive elements and appended in element order: the
// result does not depend on the number of threads. Each group keeps the ID of its element (groupElementIds), and the
// elements that couple one node only (a PointMassInertia, a grounded spring) are listed apart with their node
// (pointElements): the automatic units of the dissection (reduction::DissectionUnits) place every element by its nodes.
//
// Thread-safety: immutable once built. Ownership: owns its groups (budgeted); the budget must outlive it.
class ElementConnectivity
{
public:
    // Elements formed per task of the pool.
    static constexpr std::int64_t ELEMENTS_PER_PIECE = 256;

    // Precondition: the set passed its check (the facets throw on an element the check refuses). stiffnessStore: when
    // given (and kept), receives the stiffness matrices of the elements, in the order of StiffnessAssembler. Throws
    // MemoryBudgetError, OperationCancelledError (between waves).
    ElementConnectivity(const ElementSet& set, const DofMap& dofMap, base::MemoryBudget& budget,
                        base::ThreadPool* pool = nullptr, const base::CancellationToken* cancellation = nullptr,
                        ElementMatrixStore* stiffnessStore = nullptr);

    // An element that couples one node only, and that node.
    struct PointElement
    {
        std::int32_t elementId = 0;
        std::int32_t node = 0;
    };

    [[nodiscard]] linalg::NodeGroupList groups() const noexcept { return {m_offsets.span(), m_nodes.span()}; }
    [[nodiscard]] std::int64_t groupCount() const noexcept { return static_cast<std::int64_t>(m_offsets.size()) - 1; }
    // The element ID of each group, in group order.
    [[nodiscard]] std::span<const std::int32_t> groupElementIds() const noexcept { return m_groupElementIds.span(); }
    // The elements of one node, in element order.
    [[nodiscard]] std::span<const PointElement> pointElements() const noexcept { return m_pointElements.span(); }
    // Per node of the DOF map, the components an element's stiffness has DOFs on (bit c − 1 for component c of a grid
    // point, bit 0 for a scalar point; 0: no stiffness element): a grid point of solids alone has its translations
    // only (its rotations held by AUTOSPC in the analysis; used by the factor estimate made before the analysis).
    [[nodiscard]] std::span<const std::uint8_t> stiffenedComponents() const noexcept
    {
        return m_stiffenedComponents.span();
    }

private:
    base::BudgetedBuffer<std::int64_t> m_offsets; // groupCount + 1
    base::BudgetedBuffer<std::int32_t> m_nodes;
    base::BudgetedBuffer<std::int32_t> m_groupElementIds; // groupCount
    base::BudgetedBuffer<PointElement> m_pointElements;
    base::BudgetedBuffer<std::uint8_t> m_stiffenedComponents; // per node
};

} // namespace oos::assembly
