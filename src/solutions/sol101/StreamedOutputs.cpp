#include "oos/solutions/sol101/StreamedOutputs.hpp"

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/solutions/sol101/BasisFixedRows.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace oos::solutions
{
namespace
{

// The cases recovered together from a chunk (the element tables of a block of cases: ElementResultBuilder's blocks;
// each block makes its elements' recovery slopes again, so a block of 1024 makes them once for up to 1024 cases).
constexpr std::size_t CASE_BLOCK = 1024;
// The combination of the cases (flush): the cases summed together on a row, the slots of a task of the pool.
constexpr std::size_t CASE_LANES = 8;
constexpr std::size_t COMBINATION_SLOTS = 256;

// The ranges of sorted rows.
std::vector<elements::ElementRange> rangesOfRows(std::vector<elements::ElementRange>& rows)
{
    std::sort(rows.begin(), rows.end(),
              [](const elements::ElementRange& a, const elements::ElementRange& b) { return a.first < b.first; });
    std::vector<elements::ElementRange> ranges; // bounded by the rows
    for (const elements::ElementRange& row : rows)
    {
        if (!ranges.empty() && ranges.back().last == row.first)
            ranges.back().last = row.last;
        else if (ranges.empty() || ranges.back().last < row.first)
            ranges.push_back(row);
    }
    return ranges;
}

// One table made of the pieces of the chunks, its rows in the full path's order: by element ID for an element table
// (the batches of a type are in ID order), the rows of one element in their order.
output::ElementResultTable mergeElementTables(std::vector<output::ElementResultTable>& pieces,
                                              base::MemoryBudget& budget)
{
    std::size_t rowCount = 0;
    for (const output::ElementResultTable& piece : pieces)
        rowCount += piece.rowCount();
    struct Row
    {
        std::int32_t id;
        std::size_t piece;
        std::size_t row;
    };
    std::vector<Row> order; // bounded by the rows
    order.reserve(rowCount);
    for (std::size_t p = 0; p < pieces.size(); ++p)
        for (std::size_t r = 0; r < pieces[p].rowCount(); ++r)
            order.push_back({pieces[p].ids[r], p, r});
    std::stable_sort(order.begin(), order.end(), [](const Row& a, const Row& b) { return a.id < b.id; });
    output::ElementResultTable table;
    const output::ElementResultTable& first = pieces.front();
    table.kind = first.kind;
    table.elementType = first.elementType;
    table.subcase = first.subcase;
    table.analysis = first.analysis;
    table.components = first.components;
    table.hasPositions = first.hasPositions;
    const std::size_t width = first.componentCount();
    table.ids = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result ids", base::FOR_OVERWRITE);
    table.nodeIds =
        base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result grid points", base::FOR_OVERWRITE);
    table.layers = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result layers", base::FOR_OVERWRITE);
    table.values = base::BudgetedBuffer<double>(budget, rowCount * width, "element result values", base::FOR_OVERWRITE);
    if (table.hasPositions)
        table.positions =
            base::BudgetedBuffer<double>(budget, rowCount, "element result positions", base::FOR_OVERWRITE);
    for (std::size_t k = 0; k < order.size(); ++k)
    {
        const output::ElementResultTable& piece = pieces[order[k].piece];
        const std::size_t r = order[k].row;
        table.ids[k] = piece.ids[r];
        table.nodeIds[k] = piece.nodeIds[r];
        table.layers[k] = piece.layers[r];
        if (table.hasPositions)
            table.positions[k] = piece.positions[r];
        std::copy_n(piece.values.data() + r * width, width, table.values.data() + k * width);
    }
    pieces.clear();
    return table;
}

} // namespace

StreamedOutputs::StreamedOutputs(const Inputs& inputs, base::MemoryBudget& budget)
    : m_inputs(inputs),
      m_budget(budget),
      m_columns(inputs.coefficients->rows()),
      m_caseCount(inputs.coefficients->columns())
{
    const assembly::DofMap& dofMap = inputs.elements->model().dofMap();
    const output::ElementOutputSelection selection = output::elementSelectionOf(*inputs.plan);
    m_hasStress = selection.stress.isRequested || selection.strain.isRequested || selection.forces.isRequested;
    m_asksDisplacements = inputs.plan->of(output::OutputKind::Displacement).isRequested;
    m_slotOfNode = base::BudgetedBuffer<std::int32_t>(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                                      "streamed outputs: chunk slots", base::FOR_OVERWRITE);
    std::fill_n(m_slotOfNode.data(), m_slotOfNode.size(), -1);
    m_gridSlots = base::BudgetedBuffer<std::int32_t>(budget, static_cast<std::size_t>(dofMap.gridCount()),
                                                     "streamed outputs: window slots", base::FOR_OVERWRITE);
    m_scalarSlots = base::BudgetedBuffer<std::int32_t>(budget, static_cast<std::size_t>(dofMap.scalarCount()),
                                                       "streamed outputs: window slots", base::FOR_OVERWRITE);
    std::fill_n(m_gridSlots.data(), m_gridSlots.size(), -1);
    std::fill_n(m_scalarSlots.data(), m_scalarSlots.size(), -1);
    m_rowsOfBatch.resize(inputs.elements->model().elementSet().batches().size());
    m_supportValues.assign(inputs.supports.size(), std::vector<double>(m_columns, 0.0));
    m_isSupportSolved.assign(inputs.supports.size(), 0);
    m_nodal.resize(m_caseCount);
    m_element.resize(m_caseCount);
    m_isBasis = inputs.basisQuantities != nullptr || !inputs.isForceNode.empty();
    if (m_isBasis)
    {
        if (m_caseCount != m_columns)
            throw std::invalid_argument(base::concatenate(
                {"StreamedOutputs: the load basis takes its vectors as the cases (", std::to_string(m_columns),
                 "), found ", std::to_string(m_caseCount), " cases (a bug of O_o_Solver: report it)"}));
        m_basis = std::make_shared<KeptStreamedBasis>();
        if (inputs.basisQuantities != nullptr)
        {
            m_basis->quantityCount = inputs.basisQuantities->quantityCount();
            m_basis->termCount = inputs.basisQuantities->termCount();
            m_basis->quantities = base::BudgetedBuffer<double>(budget, m_columns * m_basis->quantityCount,
                                                               "streamed load basis quantities");
            m_hasStress = !inputs.basisQuantities->elements().empty();
            // The force tables of the map's elements alone (sorted: QuantityMap::elements), with the describers given.
            m_inputs.basisSelection.stress.isRequested = false;
            m_inputs.basisSelection.strain.isRequested = false;
            m_inputs.basisSelection.forces.isRequested = true;
            m_inputs.basisSelection.forces.isAll = false;
            m_inputs.basisSelection.forces.members = inputs.basisQuantities->elements();
        }
        else
            m_hasStress = false;
        m_asksDisplacements = false;
        m_forceRowsOfBatch.resize(m_rowsOfBatch.size());
        m_forceNames.assign(output::GridPointForceTable::FIXED_ROW_NAMES.begin(),
                            output::GridPointForceTable::FIXED_ROW_NAMES.end());
        m_contributions.resize(m_columns);
    }
}

std::shared_ptr<KeptStreamedBasis> StreamedOutputs::takeBasis()
{
    return std::move(m_basis);
}

void StreamedOutputs::presetSupportValues(linalg::ConstDenseMatrixView values)
{
    if (values.rows != m_inputs.supports.size() || values.columns != m_columns)
        throw std::invalid_argument(base::concatenate(
            {"StreamedOutputs::presetSupportValues: expected ", std::to_string(m_inputs.supports.size()), " x ",
             std::to_string(m_columns), " values (the FreeBodySupport DOFs x the vectors), found ",
             std::to_string(values.rows), " x ", std::to_string(values.columns)}));
    for (std::size_t j = 0; j < values.rows; ++j)
    {
        for (std::size_t b = 0; b < m_columns; ++b)
            m_supportValues[j][b] = values(j, b);
        m_isSupportSolved[j] = 1;
    }
}

bool StreamedOutputs::isGaugeReady() const noexcept
{
    return std::all_of(m_isSupportSolved.begin(), m_isSupportSolved.end(), [](std::uint8_t solved) { return solved; });
}

std::uint64_t StreamedOutputs::chunkBytes(std::int64_t values, std::size_t columns, std::size_t cases) noexcept
{
    // The chunk's displacements and gauge rows, a block of cases' displacements (the chunk's DOFs × CASE_BLOCK) and its
    // element tables (at most about as many values again as the displacements of the block, with their labels).
    const auto chunk = static_cast<std::uint64_t>(std::max<std::int64_t>(values, 0));
    const std::uint64_t dofs = columns == 0 ? 0 : chunk / columns;
    const std::uint64_t block = dofs * std::min(cases, CASE_BLOCK);
    return (chunk + 7 * dofs + 4 * block) * sizeof(double);
}

std::int64_t StreamedOutputs::chunkValuesFor(std::uint64_t bytes, std::size_t columns, std::size_t cases) noexcept
{
    // chunkBytes: per DOF of the chunk, its columns, its gauge row and the block's cases with their tables.
    const std::uint64_t perDof = (columns + 7 + 4 * std::min(cases, CASE_BLOCK)) * sizeof(double);
    const std::uint64_t dofs = std::max<std::uint64_t>(bytes / perDof, 6);
    return static_cast<std::int64_t>(dofs * std::max<std::size_t>(columns, 1));
}

void StreamedOutputs::addNode(std::int32_t node, std::int32_t supernode, linalg::ConstDenseMatrixView columnValues,
                              linalg::ConstDenseMatrixView rowValues)
{
    if (m_slotOfNode[static_cast<std::size_t>(node)] >= 0)
        return;
    const StreamedElements& elements = *m_inputs.elements;
    const linalg::SymbolicFactor& symbolic = elements.symbolic();
    const std::int32_t firstColumn = symbolic.firstColumn(supernode);
    const std::int32_t columnCount = symbolic.columnCount(supernode);
    const std::span<const std::int32_t> rows = symbolic.rows(supernode);
    // The solution of an equation of the front (columns, then rows).
    const auto valueOf = [&](std::int32_t equation, std::size_t column) -> double
    {
        if (equation >= firstColumn && equation < firstColumn + columnCount)
            return columnValues(static_cast<std::size_t>(equation - firstColumn), column);
        const auto found = std::lower_bound(rows.begin(), rows.end(), equation);
        if (found == rows.end() || *found != equation)
            throw base::InternalError(base::concatenate({"the full path streamed: equation ", std::to_string(equation),
                                                         " of a node placed at supernode ", std::to_string(supernode),
                                                         " is not in its front (a bug of O_o_Solver: report it)"}));
        return rowValues(static_cast<std::size_t>(found - rows.begin()), column);
    };
    std::vector<base::DofIndex> dofs; // bounded by a node
    elements.nodeDofs(node, dofs);
    StreamedReduction reduction;
    elements.reduce(dofs, reduction);
    const auto slot = static_cast<std::int32_t>(m_slotNodes.size());
    m_slotOfNode[static_cast<std::size_t>(node)] = slot;
    m_slotNodes.push_back(node);
    const std::size_t first = m_values.size();
    m_valueOffset.push_back(static_cast<std::int64_t>(first));
    m_values.resize(first + dofs.size() * m_columns, 0.0);
    const std::size_t gaugeFirst = m_gauge.size();
    m_gauge.resize(gaugeFirst + dofs.size() * 6, 0.0);
    for (const StreamedReduction::Term& term : reduction.terms)
    {
        const std::int32_t equation = reduction.equations[static_cast<std::size_t>(term.equation)];
        const auto d = static_cast<std::size_t>(term.dof);
        for (std::size_t b = 0; b < m_columns; ++b)
            m_values[first + d * m_columns + b] += term.coefficient * valueOf(equation, b);
    }
    if (!m_inputs.supports.empty())
        for (std::size_t d = 0; d < dofs.size(); ++d)
        {
            const std::array<double, 6> row = m_inputs.gaugeRowOf(dofs[d]);
            std::copy(row.begin(), row.end(), m_gauge.begin() + static_cast<std::ptrdiff_t>(gaugeFirst + d * 6));
        }
}

void StreamedOutputs::take(std::int32_t supernode, linalg::ConstDenseMatrixView columnValues,
                           linalg::ConstDenseMatrixView rowValues)
{
    const StreamedElements& elements = *m_inputs.elements;
    const linalg::SymbolicFactor& symbolic = elements.symbolic();
    // The FreeBodySupport DOFs' values, as the conditioned solve gives them.
    const std::int32_t firstColumn = symbolic.firstColumn(supernode);
    for (std::size_t j = 0; j < m_inputs.supports.size(); ++j)
    {
        const std::int32_t equation = elements.selection().equationOfDof(m_inputs.supports[j]);
        if (equation >= firstColumn && equation < firstColumn + symbolic.columnCount(supernode))
        {
            for (std::size_t b = 0; b < m_columns; ++b)
                m_supportValues[j][b] = columnValues(static_cast<std::size_t>(equation - firstColumn), b);
            m_isSupportSolved[j] = 1;
        }
    }
    const std::span<const assembly::AssemblyBatch> batches = elements.model().elementSet().batches();
    if (m_isBasis)
    {
        // The load basis: the map's elements (their force tables), and the elements at the force nodes (their grid
        // point forces).
        const std::span<const std::int32_t> mapped = m_inputs.basisQuantities != nullptr
                                                         ? m_inputs.basisQuantities->elements()
                                                         : std::span<const std::int32_t>{};
        for (const std::int32_t element : elements.elementsAt(supernode))
        {
            const StreamedElements::Place place = elements.placeOf(element);
            const assembly::AssemblyBatch& batch = batches[place.batch];
            const std::int32_t elementId = batch.view.elementId(place.row);
            const std::span<const std::int32_t> nodes = elements.nodesOf(element);
            const bool isMapped = batch.family->facets().has(elements::Facet::Recovery) &&
                                  std::binary_search(mapped.begin(), mapped.end(), elementId);
            const bool atForceNode = !m_inputs.isForceNode.empty() &&
                                     batch.family->facets().has(elements::Facet::Stiffness) &&
                                     std::any_of(nodes.begin(), nodes.end(), [&](std::int32_t node)
                                                 { return m_inputs.isForceNode[static_cast<std::size_t>(node)] != 0; });
            if (!isMapped && !atForceNode)
                continue;
            for (const std::int32_t node : nodes)
                addNode(node, supernode, columnValues, rowValues);
            if (isMapped)
                m_rowsOfBatch[place.batch].push_back({place.row, place.row + 1});
            if (atForceNode)
                m_forceRowsOfBatch[place.batch].push_back({place.row, place.row + 1});
        }
    }
    else if (m_hasStress)
    {
        const output::ElementOutputSelection selection = output::elementSelectionOf(*m_inputs.plan);
        for (const std::int32_t element : elements.elementsAt(supernode))
        {
            const StreamedElements::Place place = elements.placeOf(element);
            const assembly::AssemblyBatch& batch = batches[place.batch];
            if (!batch.family->facets().has(elements::Facet::Recovery))
                continue;
            const std::int32_t elementId = batch.view.elementId(place.row);
            if (!selection.stress.takes(elementId) && !selection.strain.takes(elementId) &&
                !selection.forces.takes(elementId))
                continue;
            for (const std::int32_t node : elements.nodesOf(element))
                addNode(node, supernode, columnValues, rowValues);
            m_rowsOfBatch[place.batch].push_back({place.row, place.row + 1});
        }
    }
    if (m_asksDisplacements)
        for (const std::int32_t node : elements.pointsAt(supernode))
        {
            addNode(node, supernode, columnValues, rowValues);
            m_points.push_back(node);
        }
    if (static_cast<std::int64_t>(m_values.size()) >= m_inputs.chunkValues && isGaugeReady())
        flush();
}

void StreamedOutputs::flush()
{
    if (m_slotNodes.empty())
        return;
    if (!isGaugeReady())
        throw std::logic_error(
            "StreamedOutputs: a chunk handed over before the FreeBodySupport DOFs were solved (a bug of O_o_Solver: "
            "report it)");
    ++m_chunkCount;
    m_largestChunkValues = std::max(m_largestChunkValues, static_cast<std::int64_t>(m_values.size()));
    const StreamedElements& elements = *m_inputs.elements;
    const assembly::DofMap& dofMap = elements.model().dofMap();
    const linalg::DenseMatrix& alpha = *m_inputs.coefficients;

    // The gauge of the FreeBodySupport DOFs: the amplitudes C·u_r of each vector's rigid motion, removed from the
    // displacement rows of each case, u_c ← u_c − (Σ coefficient·Φ)·(Σ_b α(b, c)·C·u_r,b), the FreeBodySupport DOFs at
    // rest exactly; the element tables are made from the displacements as solved on the conditioned supports, whose
    // element results do not depend on the FreeBodySupport DOFs (the gauge's rigid motion would add its round-off to
    // the stiff elements: 1e-6 of the forces of fasteners 1e8 times the skins between two FreeBodySupport points).
    std::vector<std::array<double, 6>> amplitudes; // per vector (bounded by the basis); empty without FreeBodySupport
    if (!m_inputs.supports.empty())
    {
        const std::array<double, 36>& c = *m_inputs.supportCoefficients;
        const std::size_t r = m_inputs.supports.size();
        amplitudes.assign(m_columns, {});
        for (std::size_t b = 0; b < m_columns; ++b)
            for (std::size_t mode = 0; mode < 6; ++mode)
            {
                double sum = 0.0;
                for (std::size_t j = 0; j < r; ++j)
                    sum += c[6 * j + mode] * m_supportValues[j][b];
                amplitudes[b][mode] = sum;
            }
    }
    // The gauge's motion of a slot's DOF d for the amplitudes `a`.
    const auto gaugeMotion = [&](std::int32_t node, std::size_t d, const std::array<double, 6>& a)
    {
        const auto slot = static_cast<std::size_t>(m_slotOfNode[static_cast<std::size_t>(node)]);
        const std::size_t gaugeFirst =
            static_cast<std::size_t>(m_valueOffset[slot]) / std::max<std::size_t>(m_columns, 1) * 6;
        double motion = 0.0;
        for (std::size_t mode = 0; mode < 6; ++mode)
            motion += m_gauge[gaugeFirst + d * 6 + mode] * a[mode];
        return motion;
    };
    const auto isSupport = [&](std::int32_t node, std::size_t d)
    {
        return std::any_of(m_inputs.supports.begin(), m_inputs.supports.end(),
                           [&](base::DofIndex support)
                           {
                               const linalg::NodeDof nodeDof = dofMap.nodeDofOf(support);
                               return nodeDof.node == node && static_cast<std::size_t>(nodeDof.component) == d;
                           });
    };

    // The slots of the window: the grid points, then the scalar points.
    std::span<std::int32_t> gridSlots = m_gridSlots.span();
    std::span<std::int32_t> scalarSlots = m_scalarSlots.span();
    std::size_t gridCount = 0;
    std::size_t scalarCount = 0;
    for (const std::int32_t node : m_slotNodes)
    {
        if (node < dofMap.gridCount())
            gridSlots[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(gridCount++);
        else
            scalarSlots[static_cast<std::size_t>(node - dofMap.gridCount())] = static_cast<std::int32_t>(scalarCount++);
    }
    const std::size_t rows = 6 * gridCount + scalarCount;
    const auto rowOfSlotDof = [&](std::int32_t node, std::size_t d) -> std::size_t
    {
        if (node < dofMap.gridCount())
            return 6 * static_cast<std::size_t>(gridSlots[static_cast<std::size_t>(node)]) + d;
        return 6 * gridCount +
               static_cast<std::size_t>(scalarSlots[static_cast<std::size_t>(node - dofMap.gridCount())]);
    };
    std::vector<std::vector<elements::ElementRange>> ranges(m_rowsOfBatch.size()); // per batch
    for (std::size_t b = 0; b < m_rowsOfBatch.size(); ++b)
        ranges[b] = rangesOfRows(m_rowsOfBatch[b]);
    std::vector<std::vector<elements::ElementRange>> forceRanges(m_forceRowsOfBatch.size()); // per batch
    for (std::size_t b = 0; b < m_forceRowsOfBatch.size(); ++b)
        forceRanges[b] = rangesOfRows(m_forceRowsOfBatch[b]);
    output::RecoveryWindow window;
    window.rowsOfBatch = ranges;
    window.slots.gridSlots = gridSlots;
    window.slots.scalarSlots = scalarSlots;
    window.gridSlotCount = gridCount;
    window.scalarSlotCount = scalarCount;
    std::sort(m_points.begin(), m_points.end());
    m_points.erase(std::unique(m_points.begin(), m_points.end()), m_points.end());
    const output::OutputRequestPlan& displacement = m_inputs.plan->of(output::OutputKind::Displacement);
    const output::ElementOutputSelection selection = output::elementSelectionOf(*m_inputs.plan);

    if (m_isBasis)
    {
        // The load basis: the vectors' displacements on the window's rows as they are (α the identity),
        // the map's elements' force tables reduced by the map, the elements at the force nodes their grid point
        // forces; the element results from the displacements as solved, as the cases' tables are made.
        linalg::DenseMatrix vectors(m_budget, rows, m_columns, "streamed outputs: displacements of the vectors");
        for (std::size_t slot = 0; slot < m_slotNodes.size(); ++slot)
        {
            const std::int32_t node = m_slotNodes[slot];
            const auto first = static_cast<std::size_t>(m_valueOffset[slot]);
            const std::size_t dofCount = node < dofMap.gridCount() ? 6 : 1;
            for (std::size_t d = 0; d < dofCount; ++d)
                for (std::size_t b = 0; b < m_columns; ++b)
                    vectors(rowOfSlotDof(node, d), b) = m_values[first + d * m_columns + b];
        }
        if (m_hasStress)
        {
            std::vector<std::vector<output::ElementResultTable>> tables(m_columns);
            m_inputs.builder->buildWindow(m_inputs.caseLabels, vectors.view(), m_caseCount, window, m_budget, tables,
                                          m_inputs.pool, m_inputs.cancellation, &m_inputs.basisSelection);
            const std::size_t count = m_basis->quantityCount;
            for (std::size_t b = 0; b < m_columns; ++b)
                m_inputs.basisQuantities->reduce(tables[b],
                                                 std::span<double>(m_basis->quantities.data() + b * count, count));
        }
        if (!m_inputs.isForceNode.empty())
        {
            const output::StiffnessRecovery stiffness(elements.model().elementSet(), dofMap, m_inputs.pool);
            output::RecoveryWindow forceWindow = window;
            forceWindow.rowsOfBatch = forceRanges;
            const output::StiffnessRecovery::PointForces forces = stiffness.pointForces(
                m_inputs.isForceNode, forceRanges, vectors.view(), m_budget, m_inputs.cancellation, &forceWindow);
            // The names of this chunk's contributions into the run's list.
            std::vector<std::int32_t> nameOf(forces.names.size(), 0); // bounded by the element types
            for (std::size_t k = output::StiffnessRecovery::FIRST_ELEMENT_NAME; k < forces.names.size(); ++k)
            {
                const auto found = std::find(m_forceNames.begin(), m_forceNames.end(), forces.names[k]);
                nameOf[k] = static_cast<std::int32_t>(found - m_forceNames.begin());
                if (found == m_forceNames.end())
                    m_forceNames.push_back(forces.names[k]);
            }
            for (std::size_t b = 0; b < m_columns; ++b)
                for (output::StiffnessRecovery::PointContribution contribution : forces.columns[b])
                {
                    contribution.name = nameOf[static_cast<std::size_t>(contribution.name)];
                    m_contributions[b].push_back(contribution);
                }
        }
    }
    for (std::size_t c0 = 0; c0 < m_caseCount && !m_isBasis; c0 += CASE_BLOCK)
    {
        const std::size_t count = std::min(CASE_BLOCK, m_caseCount - c0);
        // u_c = Σ_b α(b, c)·u_b on the window's rows, each sum in the order of b from zero (the same bits whatever
        // the blocks and the threads).
        // PERFORMANCE: α of the block transposed (the cases contiguous), CASE_LANES cases summed together per row, the
        // slots in pieces on the pool: the combination was a scalar loop on one thread, 8e10 products for 300 cases of
        // the 4-frame barrel at 1/32.
        linalg::DenseMatrix cases(m_budget, rows, count, "streamed outputs: displacements of the cases");
        base::BudgetedBuffer<double> byVector(m_budget, m_columns * count, "streamed outputs: coefficients",
                                              base::FOR_OVERWRITE);
        for (std::size_t b = 0; b < m_columns; ++b)
            for (std::size_t c = 0; c < count; ++c)
                byVector[b * count + c] = alpha(b, c0 + c);
        const auto combine = [&](std::size_t piece)
        {
            const std::size_t last = std::min(m_slotNodes.size(), (piece + 1) * COMBINATION_SLOTS);
            for (std::size_t slot = piece * COMBINATION_SLOTS; slot < last; ++slot)
            {
                const std::int32_t node = m_slotNodes[slot];
                const auto first = static_cast<std::size_t>(m_valueOffset[slot]);
                const std::size_t dofCount = node < dofMap.gridCount() ? 6 : 1;
                for (std::size_t d = 0; d < dofCount; ++d)
                {
                    const std::size_t row = rowOfSlotDof(node, d);
                    const double* const u = m_values.data() + first + d * m_columns;
                    for (std::size_t c = 0; c < count; c += CASE_LANES)
                    {
                        const std::size_t lanes = std::min(CASE_LANES, count - c);
                        std::array<double, CASE_LANES> sums{};
                        for (std::size_t b = 0; b < m_columns; ++b)
                        {
                            const double ub = u[b];
                            const double* const ab = byVector.data() + b * count + c;
                            for (std::size_t k = 0; k < lanes; ++k)
                                sums[k] += ab[k] * ub;
                        }
                        for (std::size_t k = 0; k < lanes; ++k)
                            cases(row, c + k) = sums[k];
                    }
                }
            }
        };
        const std::size_t pieces = (m_slotNodes.size() + COMBINATION_SLOTS - 1) / COMBINATION_SLOTS;
        if (m_inputs.pool != nullptr && m_inputs.pool->threadCount() > 1 && pieces > 1)
            m_inputs.pool->runBatch(pieces, combine);
        else
            for (std::size_t piece = 0; piece < pieces; ++piece)
                combine(piece);
        // The displacement rows of the chunk's points.
        if (m_asksDisplacements)
            for (std::size_t c = 0; c < count; ++c)
            {
                std::vector<std::int32_t> taken; // bounded by the chunk's points
                for (const std::int32_t node : m_points)
                {
                    const base::EntityId id =
                        node < dofMap.gridCount() ? dofMap.gridId(node) : dofMap.scalarId(node - dofMap.gridCount());
                    if (displacement.isAll ||
                        std::binary_search(displacement.members.begin(), displacement.members.end(), id))
                        taken.push_back(node);
                }
                if (taken.empty())
                    continue;
                output::NodalResultTable table;
                table.kind = "displacements";
                table.subcase = m_inputs.caseLabels[c0 + c];
                table.ids = base::BudgetedBuffer<std::int32_t>(m_budget, taken.size(), "displacement ids");
                table.pointTypes = base::BudgetedBuffer<std::int8_t>(m_budget, taken.size(), "displacement types");
                table.values = base::BudgetedBuffer<double>(m_budget, taken.size() * 6, "displacement values");
                std::array<double, 6> caseAmplitudes{}; // Σ_b α(b, c)·C·u_r,b
                for (std::size_t b = 0; b < amplitudes.size(); ++b)
                    for (std::size_t mode = 0; mode < 6; ++mode)
                        caseAmplitudes[mode] += alpha(b, c0 + c) * amplitudes[b][mode];
                for (std::size_t k = 0; k < taken.size(); ++k)
                {
                    const std::int32_t node = taken[k];
                    const bool isGrid = node < dofMap.gridCount();
                    table.ids[k] = isGrid ? dofMap.gridId(node) : dofMap.scalarId(node - dofMap.gridCount());
                    table.pointTypes[k] =
                        static_cast<std::int8_t>(isGrid ? output::PointType::Grid : output::PointType::Scalar);
                    for (std::size_t d = 0; d < (isGrid ? 6U : 1U); ++d)
                    {
                        double value = cases(rowOfSlotDof(node, d), c);
                        if (!amplitudes.empty())
                            value = isSupport(node, d) ? 0.0 : value - gaugeMotion(node, d, caseAmplitudes);
                        table.values[6 * k + d] = value;
                    }
                }
                if (m_inputs.envelopes != nullptr)
                    m_inputs.envelopes->offer(table);
                if (m_inputs.keepsCaseTables || m_inputs.envelopes == nullptr || !m_inputs.envelopes->takes(table.kind))
                    m_nodal[c0 + c].push_back(std::move(table));
            }
        // The element tables of the chunk's elements.
        if (m_hasStress)
        {
            std::vector<std::vector<output::ElementResultTable>> tables(count);
            m_inputs.builder->buildWindow(m_inputs.caseLabels.subspan(c0, count), cases.view(), m_caseCount, window,
                                          m_budget, tables, m_inputs.pool, m_inputs.cancellation, &selection);
            for (std::size_t c = 0; c < count; ++c)
                for (output::ElementResultTable& table : tables[c])
                {
                    if (m_inputs.envelopes != nullptr)
                        m_inputs.envelopes->offer(table);
                    if (m_inputs.keepsCaseTables || m_inputs.envelopes == nullptr ||
                        !m_inputs.envelopes->takes(table.kind))
                        m_element[c0 + c].push_back(std::move(table));
                }
        }
    }

    // The chunk released.
    for (const std::int32_t node : m_slotNodes)
    {
        m_slotOfNode[static_cast<std::size_t>(node)] = -1;
        if (node < dofMap.gridCount())
            gridSlots[static_cast<std::size_t>(node)] = -1;
        else
            scalarSlots[static_cast<std::size_t>(node - dofMap.gridCount())] = -1;
    }
    m_slotNodes.clear();
    m_values.clear();
    m_values.shrink_to_fit();
    m_gauge.clear();
    m_valueOffset.clear();
    for (std::vector<elements::ElementRange>& rowsOfBatch : m_rowsOfBatch)
        rowsOfBatch.clear();
    for (std::vector<elements::ElementRange>& rowsOfBatch : m_forceRowsOfBatch)
        rowsOfBatch.clear();
    m_points.clear();
}

void StreamedOutputs::finish()
{
    flush();
    if (m_isBasis)
    {
        // The grid point forces of every vector at the fixed rows' nodes: the contributions by node (the order of the
        // elements within a node that of the chunks), the fixed rows made as statics makes them (BasisFixedRows).
        if (m_inputs.fixedRows != nullptr && !m_inputs.isForceNode.empty())
        {
            const assembly::DofMap& dofMap = m_inputs.elements->model().dofMap();
            output::StiffnessRecovery::PointForces forces;
            forces.names = m_forceNames;
            forces.columns = std::move(m_contributions);
            for (std::vector<output::StiffnessRecovery::PointContribution>& made : forces.columns)
                std::stable_sort(made.begin(), made.end(),
                                 [](const output::StiffnessRecovery::PointContribution& a,
                                    const output::StiffnessRecovery::PointContribution& b) { return a.node < b.node; });
            const std::span<const std::int32_t> nodes = m_inputs.fixedRows->nodes.span();
            m_basis->rowNodes.assign(nodes.begin(), nodes.end());
            const output::StiffnessRecovery stiffness(m_inputs.elements->model().elementSet(), dofMap, m_inputs.pool);
            BasisFixedRows::Work work;
            for (std::size_t b = 0; b < m_columns; ++b)
            {
                const std::vector<output::StiffnessRecovery::PointFixedRows> fixed = BasisFixedRows::rowsOf(
                    *m_inputs.fixedRows, b, m_basis->rowNodes, forces, b, {}, dofMap, m_budget, work);
                m_basis->gridPointForces.push_back(
                    stiffness.pointForceTable(m_inputs.caseLabels[b], forces, b, m_basis->rowNodes, fixed, m_budget));
            }
        }
        return;
    }
    // Each case's tables in the full path's order: a nodal table by point (grid points then scalar points, by ID),
    // an element table by element ID within its kind and type.
    for (std::size_t c = 0; c < m_caseCount; ++c)
    {
        std::vector<output::NodalResultTable>& nodal = m_nodal[c];
        if (nodal.size() > 1 || (nodal.size() == 1 && nodal.front().rowCount() > 1))
        {
            std::size_t rowCount = 0;
            for (const output::NodalResultTable& piece : nodal)
                rowCount += piece.rowCount();
            std::vector<std::tuple<std::int8_t, std::int32_t, std::size_t, std::size_t>> order; // bounded by the rows
            for (std::size_t p = 0; p < nodal.size(); ++p)
                for (std::size_t r = 0; r < nodal[p].rowCount(); ++r)
                    order.emplace_back(nodal[p].pointTypes[r], nodal[p].ids[r], p, r);
            std::sort(order.begin(), order.end());
            output::NodalResultTable table;
            table.kind = nodal.front().kind;
            table.subcase = nodal.front().subcase;
            table.ids = base::BudgetedBuffer<std::int32_t>(m_budget, rowCount, "displacement ids");
            table.pointTypes = base::BudgetedBuffer<std::int8_t>(m_budget, rowCount, "displacement types");
            table.values = base::BudgetedBuffer<double>(m_budget, rowCount * 6, "displacement values");
            for (std::size_t k = 0; k < order.size(); ++k)
            {
                const auto [type, id, p, r] = order[k];
                table.ids[k] = id;
                table.pointTypes[k] = type;
                std::copy_n(nodal[p].values.data() + 6 * r, 6, table.values.data() + 6 * k);
            }
            nodal.clear();
            nodal.push_back(std::move(table));
        }
        std::vector<output::ElementResultTable>& element = m_element[c];
        std::vector<output::ElementResultTable> merged; // bounded by the kinds × types
        while (!element.empty())
        {
            const std::string kind = element.front().kind;
            const std::string type = element.front().elementType;
            std::vector<output::ElementResultTable> pieces; // the pieces of one kind and type
            for (std::size_t k = 0; k < element.size();)
                if (element[k].kind == kind && element[k].elementType == type)
                {
                    pieces.push_back(std::move(element[k]));
                    element.erase(element.begin() + static_cast<std::ptrdiff_t>(k));
                }
                else
                    ++k;
            merged.push_back(mergeElementTables(pieces, m_budget));
        }
        element = std::move(merged);
    }
}

} // namespace oos::solutions
