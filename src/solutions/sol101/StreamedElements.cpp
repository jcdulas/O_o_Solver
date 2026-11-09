#include "oos/solutions/sol101/StreamedElements.hpp"

#include "oos/assembly/assemble/ElementConnectivity.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"

#include <algorithm>
#include <limits>

namespace oos::solutions
{
namespace
{

constexpr std::int32_t NO_EQUATION = std::numeric_limits<std::int32_t>::max();

// Fills offsets (supernodes + 1) and items (by supernode, increasing) from the supernode of each item: a counting sort.
void sortBySupernode(std::span<const std::int32_t> supernodeOfItem, std::span<const std::int32_t> itemIds,
                     std::int32_t supernodeCount, base::BudgetedBuffer<std::int64_t>& offsets,
                     base::BudgetedBuffer<std::int32_t>& items, base::MemoryBudget& budget, const char* label)
{
    const auto total = static_cast<std::size_t>(supernodeCount);
    offsets = base::BudgetedBuffer<std::int64_t>(budget, total + 1, label);
    for (const std::int32_t s : supernodeOfItem)
        ++offsets[static_cast<std::size_t>(s) + 1];
    for (std::size_t s = 0; s < total; ++s)
        offsets[s + 1] += offsets[s];
    items = base::BudgetedBuffer<std::int32_t>(budget, supernodeOfItem.size(), label, base::FOR_OVERWRITE);
    base::BudgetedBuffer<std::int64_t> cursor(budget, total, label, base::FOR_OVERWRITE);
    std::copy_n(offsets.data(), total, cursor.data());
    for (std::size_t k = 0; k < supernodeOfItem.size(); ++k)
        items[static_cast<std::size_t>(cursor[static_cast<std::size_t>(supernodeOfItem[k])]++)] = itemIds[k];
}

} // namespace

StreamedElements::StreamedElements(const Inputs& inputs, base::MemoryBudget& budget) : m_inputs(inputs)
{
    const assembly::ElementSet& set = inputs.model->elementSet();
    const assembly::ElementConnectivity& connectivity = inputs.model->connectivity();
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    const std::int32_t count = inputs.symbolic->supernodeCount();
    const std::int32_t root = std::max(count - 1, 0);
    m_batchFirst.push_back(0);
    for (const assembly::AssemblyBatch& batch : set.batches())
        m_batchFirst.push_back(m_batchFirst.back() + batch.view.rowCount());
    const auto elements = static_cast<std::size_t>(m_batchFirst.back());

    // The first equation each item reaches, over every DOF of its nodes (a superset of its own DOFs: the nodes are
    // whole in the node graph, so the front of that equation's supernode holds them all).
    std::vector<base::DofIndex> dofs; // bounded by a node
    const auto firstEquationOfNodes = [&](std::span<const std::int32_t> nodes)
    {
        std::int32_t first = NO_EQUATION;
        for (const std::int32_t node : nodes)
        {
            nodeDofs(node, dofs);
            for (const base::DofIndex dof : dofs)
            {
                const std::int32_t equation = inputs.selection->equationOfDof(dof);
                if (equation >= 0)
                {
                    first = std::min(first, equation);
                    continue;
                }
                if (const std::optional<std::size_t> row = inputs.gm->findRow(dof))
                    for (const assembly::ConstraintTerm& term : inputs.gm->row(*row))
                    {
                        const std::int32_t reached = inputs.selection->equationOfDof(term.dof);
                        if (reached >= 0)
                            first = std::min(first, reached);
                    }
            }
        }
        return first == NO_EQUATION ? root : supernodeOf(first);
    };

    base::BudgetedBuffer<std::int32_t> supernodeOfElement(budget, elements, "streamed placement: elements",
                                                          base::FOR_OVERWRITE);
    base::BudgetedBuffer<std::int32_t> elementIds(budget, elements, "streamed placement: elements",
                                                  base::FOR_OVERWRITE);
    const linalg::NodeGroupList groups = connectivity.groups();
    const std::span<const std::int32_t> groupIds = connectivity.groupElementIds();
    const std::span<const assembly::ElementConnectivity::PointElement> single = connectivity.pointElements();
    std::size_t group = 0;
    std::size_t point = 0;
    std::size_t element = 0;
    m_nodeSource =
        base::BudgetedBuffer<std::int32_t>(budget, elements, "streamed placement: element nodes", base::FOR_OVERWRITE);
    for (const assembly::AssemblyBatch& batch : set.batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row, ++element)
        {
            // The groups and the single-node elements are in element order (ElementConnectivity).
            const std::int32_t elementId = batch.view.elementId(row);
            std::int32_t supernode = root;
            m_nodeSource[element] = -1;
            if (group < groupIds.size() && groupIds[group] == elementId)
            {
                m_nodeSource[element] = static_cast<std::int32_t>(group);
                supernode = firstEquationOfNodes(groups.group(group++));
            }
            else if (point < single.size() && single[point].elementId == elementId)
            {
                m_nodeSource[element] = -2 - static_cast<std::int32_t>(point);
                supernode = firstEquationOfNodes(std::span<const std::int32_t>(&single[point++].node, 1));
            }
            supernodeOfElement[element] = supernode;
            elementIds[element] = static_cast<std::int32_t>(element);
        }
    sortBySupernode(supernodeOfElement.span(), elementIds.span(), count, m_elementOffsets, m_elements, budget,
                    "streamed placement: elements");

    std::vector<std::int32_t> pointSupernodes; // bounded by the points asked for
    std::vector<std::int32_t> pointIds;
    for (std::int32_t node = 0; node < dofMap.nodeCount(); ++node)
        if (!inputs.isOutputNode.empty() && inputs.isOutputNode[static_cast<std::size_t>(node)] != 0)
        {
            pointSupernodes.push_back(firstEquationOfNodes(std::span<const std::int32_t>(&node, 1)));
            pointIds.push_back(node);
        }
    sortBySupernode(pointSupernodes, pointIds, count, m_pointOffsets, m_points, budget, "streamed placement: points");
}

StreamedElements::Place StreamedElements::placeOf(std::int32_t element) const noexcept
{
    const auto found = std::upper_bound(m_batchFirst.begin(), m_batchFirst.end(), static_cast<std::int64_t>(element));
    const auto batch = static_cast<std::size_t>(found - m_batchFirst.begin()) - 1;
    return {batch, element - m_batchFirst[batch]};
}

std::span<const std::int32_t> StreamedElements::nodesOf(std::int32_t element) const noexcept
{
    const std::int32_t source = m_nodeSource[static_cast<std::size_t>(element)];
    const assembly::ElementConnectivity& connectivity = m_inputs.model->connectivity();
    if (source >= 0)
        return connectivity.groups().group(static_cast<std::size_t>(source));
    if (source == -1)
        return {};
    return std::span<const std::int32_t>(&connectivity.pointElements()[static_cast<std::size_t>(-2 - source)].node, 1);
}

std::span<const std::int32_t> StreamedElements::elementsAt(std::int32_t s) const noexcept
{
    const auto at = static_cast<std::size_t>(s);
    return m_elements.span().subspan(static_cast<std::size_t>(m_elementOffsets[at]),
                                     static_cast<std::size_t>(m_elementOffsets[at + 1] - m_elementOffsets[at]));
}

std::span<const std::int32_t> StreamedElements::pointsAt(std::int32_t s) const noexcept
{
    const auto at = static_cast<std::size_t>(s);
    return m_points.span().subspan(static_cast<std::size_t>(m_pointOffsets[at]),
                                   static_cast<std::size_t>(m_pointOffsets[at + 1] - m_pointOffsets[at]));
}

std::int32_t StreamedElements::supernodeOf(std::int32_t equation) const noexcept
{
    const linalg::SymbolicFactor& symbolic = *m_inputs.symbolic;
    std::int32_t low = 0;
    std::int32_t high = symbolic.supernodeCount() - 1;
    while (low < high)
    {
        const std::int32_t middle = low + (high - low + 1) / 2;
        if (symbolic.firstColumn(middle) <= equation)
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}

void StreamedElements::nodeDofs(std::int32_t node, std::vector<base::DofIndex>& dofs) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    dofs.clear();
    if (node < dofMap.gridCount())
        for (int c = 1; c <= 6; ++c)
            dofs.push_back(dofMap.gridDof(node, c));
    else
        dofs.push_back(dofMap.scalarDof(node - dofMap.gridCount()));
}

void StreamedElements::reduce(std::span<const base::DofIndex> dofs, StreamedReduction& out) const
{
    out.equations.clear();
    out.terms.clear();
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
        const auto at = static_cast<std::int32_t>(d);
        const std::int32_t equation = m_inputs.selection->equationOfDof(dofs[d]);
        if (equation >= 0)
        {
            out.terms.push_back({at, equation, 1.0});
            continue;
        }
        if (const std::optional<std::size_t> row = m_inputs.gm->findRow(dofs[d]))
            for (const assembly::ConstraintTerm& term : m_inputs.gm->row(*row))
            {
                const std::int32_t reached = m_inputs.selection->equationOfDof(term.dof);
                if (reached >= 0)
                    out.terms.push_back({at, reached, term.coefficient});
            }
    }
    for (const StreamedReduction::Term& term : out.terms)
        out.equations.push_back(term.equation);
    std::sort(out.equations.begin(), out.equations.end());
    out.equations.erase(std::unique(out.equations.begin(), out.equations.end()), out.equations.end());
    for (StreamedReduction::Term& term : out.terms)
        term.equation = static_cast<std::int32_t>(
            std::lower_bound(out.equations.begin(), out.equations.end(), term.equation) - out.equations.begin());
}

std::int32_t StreamedElements::placeOf(const StreamedReduction& reduction) const noexcept
{
    if (reduction.equations.empty())
        return std::max(m_inputs.symbolic->supernodeCount() - 1, 0);
    return supernodeOf(reduction.equations.front());
}

std::uint64_t StreamedElements::bytes() const noexcept
{
    return m_nodeSource.size() * sizeof(std::int32_t) +
           (m_elementOffsets.size() + m_pointOffsets.size()) * sizeof(std::int64_t) +
           (m_elements.size() + m_points.size()) * sizeof(std::int32_t) + m_batchFirst.size() * sizeof(std::int64_t);
}

std::uint64_t StreamedElements::estimateBytes(std::int64_t elements, std::int64_t points, std::int64_t supernodes,
                                              std::size_t batches) noexcept
{
    // The placement made (offsets and items) and, while it is made, the supernode of every element and its sort's
    // cursor.
    const auto items = static_cast<std::uint64_t>(elements + points);
    return 2 * static_cast<std::uint64_t>(supernodes + 1) * sizeof(std::int64_t) + items * sizeof(std::int32_t) +
           3 * static_cast<std::uint64_t>(elements) * sizeof(std::int32_t) +
           static_cast<std::uint64_t>(supernodes) * sizeof(std::int64_t) + (batches + 1) * sizeof(std::int64_t);
}

} // namespace oos::solutions
