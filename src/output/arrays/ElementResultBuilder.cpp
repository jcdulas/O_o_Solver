#include "oos/output/arrays/ElementResultBuilder.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace oos::output
{

namespace
{

// The kinds of tables made, in this order for each batch.
struct KindOfTable
{
    const char* kind;
    elements::ElementOutput output;
};
constexpr std::array<KindOfTable, 3> KINDS{{{"stress", elements::ElementOutput::Stresses},
                                            {"strain", elements::ElementOutput::Strains},
                                            {"forces", elements::ElementOutput::Forces}}};

// The kinds of the layer tables (laminated shells), after the others.
struct KindOfLayerTable
{
    const char* kind;
    elements::LayerOutput output;
};
constexpr std::array<KindOfLayerTable, 3> LAYER_KINDS{{{"stress", elements::LayerOutput::Stresses},
                                                       {"strain", elements::LayerOutput::Strains},
                                                       {"failureIndices", elements::LayerOutput::FailureIndices}}};

// The output a family gives for a kind of table, or nothing (no columns): the strains as membrane strains and
// curvatures when the family gives that form and the fibres are not asked for.
std::optional<elements::ElementOutput> outputOf(const elements::ElementFamily& family, elements::ElementOutput output,
                                                bool isStrainAtFibres)
{
    if (output == elements::ElementOutput::Strains && !isStrainAtFibres &&
        !family.outputColumns(elements::ElementOutput::StrainCurvatures).empty())
        return elements::ElementOutput::StrainCurvatures;
    if (family.outputColumns(output).empty())
        return std::nullopt;
    return output;
}

// The bytes a block holds, for its charge to the budget.
std::uint64_t bytesOf(const elements::ElementResultBlock& block)
{
    const std::size_t rows = block.size();
    return rows * (sizeof(std::int64_t) + sizeof(std::int32_t) + sizeof(elements::ResultPlace) +
                   block.columns().size() * sizeof(double));
}

// A block of rows and the reservation that charges it to the budget.
struct ChargedBlock
{
    elements::ElementResultBlock block;
    base::MemoryReservation charge;
};

// What a table keeps of the rows of the blocks: the centre rows only of the elements that also have corner rows
// (CENTER), and the maximum shear in place of the von Mises column (MAXS).
struct TableShaping
{
    bool isCentreOnly = false;
    bool isMaximumShear = false;
};

// The plane-stress components a von Mises column can be turned into a maximum shear from: the shells' element-frame
// stresses (oxx, oyy, txy) and the plies' (o11, o22, t12).
struct PlaneStressColumns
{
    std::size_t normal1 = 0;
    std::size_t normal2 = 0;
    std::size_t shear = 0;
    std::size_t vonMises = 0;
};

std::optional<PlaneStressColumns> planeStressColumnsOf(const std::vector<std::string>& columns)
{
    const auto find = [&](std::string_view name) -> std::optional<std::size_t>
    {
        for (std::size_t k = 0; k < columns.size(); ++k)
            if (columns[k] == name)
                return k;
        return std::nullopt;
    };
    const std::optional<std::size_t> vonMises = find("vonMises");
    if (!vonMises)
        return std::nullopt;
    for (const auto& names :
         {std::array<std::string_view, 3>{"oxx", "oyy", "txy"}, std::array<std::string_view, 3>{"o11", "o22", "t12"}})
    {
        const std::optional<std::size_t> a = find(names[0]);
        const std::optional<std::size_t> b = find(names[1]);
        const std::optional<std::size_t> s = find(names[2]);
        if (a && b && s)
            return PlaneStressColumns{*a, *b, *s, *vonMises};
    }
    return std::nullopt;
}

// The ranges of a batch's rows that recover() processes: the rows not in `excluded` (sorted), cut every
// `rangeElements` rows.
std::vector<elements::ElementRange> rangesOf(std::int64_t rowCount, const std::vector<std::int64_t>& excluded,
                                             std::int64_t rangeElements)
{
    std::vector<elements::ElementRange> ranges; // bounded by rowCount / rangeElements + excluded.size() + 1
    std::size_t next = 0;
    std::int64_t row = 0;
    while (row < rowCount)
    {
        while (next < excluded.size() && excluded[next] < row)
            ++next;
        if (next < excluded.size() && excluded[next] == row)
        {
            ++row;
            continue;
        }
        std::int64_t end = rowCount;
        if (next < excluded.size())
            end = std::min(end, excluded[next]);
        end = std::min(end, row + rangeElements);
        ranges.push_back({row, end});
        row = end;
    }
    return ranges;
}

// One table from the rows of `blocks`, in the order of the blocks, or sorted by element row (stable) when
// `sortByElementRow`; nothing when the blocks hold no row (the result tables hold no empty table).
void appendTable(const char* kind, std::string elementType, std::int32_t subcase,
                 const std::vector<const elements::ElementResultBlock*>& blocks, bool sortByElementRow,
                 base::MemoryBudget& budget, std::vector<ElementResultTable>& tables, TableShaping shaping = {})
{
    struct RowRef
    {
        std::int64_t elementRow;
        const elements::ElementResultBlock* block;
        std::size_t index;
    };
    std::size_t rowCount = 0;
    for (const elements::ElementResultBlock* block : blocks)
        rowCount += block->size();
    if (rowCount == 0)
        return;
    base::BudgetedBuffer<RowRef> order(budget, rowCount, "element result row order");
    std::size_t next = 0;
    for (const elements::ElementResultBlock* block : blocks)
        for (std::size_t i = 0; i < block->size(); ++i)
            order[next++] = {block->elementRow(i), block, i};
    if (sortByElementRow)
        std::stable_sort(order.data(), order.data() + rowCount,
                         [](const RowRef& left, const RowRef& right) { return left.elementRow < right.elementRow; });
    if (shaping.isCentreOnly)
    {
        // An element with a centre row (grid point 0) and other rows keeps its centre row (the CENTER describer); an
        // element without a centre row, or with rows at stations, keeps all its rows. The rows of an element are
        // contiguous in the order.
        std::size_t kept = 0;
        std::size_t first = 0;
        while (first < rowCount)
        {
            std::size_t last = first + 1;
            while (last < rowCount && order[last].elementRow == order[first].elementRow &&
                   order[last].block->elementId(order[last].index) == order[first].block->elementId(order[first].index))
                ++last;
            // A bar or beam (rows at stations away from end A) keeps all its rows: its grid points label stations.
            bool hasCentre = false;
            bool hasStations = false;
            for (std::size_t r = first; r < last; ++r)
            {
                const elements::ResultPlace place = order[r].block->place(order[r].index);
                hasCentre = hasCentre || place.nodeId == 0;
                hasStations = hasStations || place.position != 0.0;
            }
            hasCentre = hasCentre && !hasStations;
            for (std::size_t r = first; r < last; ++r)
                if (!hasCentre || order[r].block->place(order[r].index).nodeId == 0)
                    order[kept++] = order[r];
            first = last;
        }
        rowCount = kept;
        if (rowCount == 0)
            return;
    }

    ElementResultTable table;
    table.kind = kind;
    table.elementType = std::move(elementType);
    table.subcase = subcase;
    table.components = blocks.front()->columns();
    const std::size_t width = table.components.size();
    const std::optional<PlaneStressColumns> planeColumns =
        shaping.isMaximumShear ? planeStressColumnsOf(table.components) : std::nullopt;
    // Unwrapped once: GCC 12 -O3 reports the optional's members as maybe uninitialised inside the row loop.
    const bool isMaximumShear = planeColumns.has_value();
    const PlaneStressColumns plane = planeColumns.value_or(PlaneStressColumns{});
    if (isMaximumShear)
        table.components[plane.vonMises] = "maxShear";
    table.ids = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result ids");
    table.nodeIds = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result grid points");
    table.layers = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result layers");
    table.values = base::BudgetedBuffer<double>(budget, rowCount * width, "element result values");
    // A table has stations when a place lies away from end A: the bars and beams (positions 0 and 1 at least); the
    // shells and springs have none, and their tables carry no positions.
    for (std::size_t r = 0; r < rowCount && !table.hasPositions; ++r)
        table.hasPositions = order[r].block->place(order[r].index).position != 0.0;
    if (table.hasPositions)
        table.positions = base::BudgetedBuffer<double>(budget, rowCount, "element result positions");
    for (std::size_t r = 0; r < rowCount; ++r)
    {
        const elements::ElementResultBlock& block = *order[r].block;
        const std::size_t i = order[r].index;
        const elements::ResultPlace place = block.place(i);
        table.ids[r] = block.elementId(i);
        table.nodeIds[r] = place.nodeId;
        table.layers[r] = place.layer;
        if (table.hasPositions)
            table.positions[r] = place.position;
        const std::span<const double> values = block.values(i);
        double* row = table.values.data() + r * width;
        std::copy(values.begin(), values.end(), row);
        if (isMaximumShear)
        {
            // The maximum shear of a plane stress state, (σ1 − σ2)/2 = √(((σa − σb)/2)² + τ²) (Mohr's circle).
            const double halfDifference = 0.5 * (row[plane.normal1] - row[plane.normal2]);
            row[plane.vonMises] = std::sqrt(halfDifference * halfDifference + row[plane.shear] * row[plane.shear]);
        }
    }
    tables.push_back(std::move(table));
}

} // namespace

bool ElementKindSelection::takes(std::int32_t elementId) const
{
    if (!isRequested)
        return false;
    return isAll || std::binary_search(members.begin(), members.end(), elementId);
}

bool ElementKindSelection::selectsSameAs(const ElementKindSelection& other) const
{
    if (isRequested != other.isRequested)
        return false;
    if (!isRequested)
        return true;
    return isAll == other.isAll && isCentreOnly == other.isCentreOnly &&
           (isAll || std::equal(members.begin(), members.end(), other.members.begin(), other.members.end()));
}

bool ElementOutputSelection::selectsSameAs(const ElementOutputSelection& other) const
{
    return stress.selectsSameAs(other.stress) && strain.selectsSameAs(other.strain) &&
           forces.selectsSameAs(other.forces) && isStrainAtFibres == other.isStrainAtFibres &&
           isMaximumShear == other.isMaximumShear;
}

namespace
{

// The element types whose outputs the CENTER and CORNER describers choose between: the four-node quadrilaterals and
// the three-node triangles (CQUAD4, CQUADR, CTRIAR, CTRIA3), which have a centre row and corner rows. The other
// element types write all their rows whatever the describer: the solids their centre and vertex grid points, the
// higher-order shells their corners, the bars their stations.
bool isCentreOrCornerElement(std::string_view card)
{
    return card == "ShellQuad4" || card == "ShellQuad4Drilling" || card == "ShellTria3Drilling" || card == "ShellTria3";
}

// The ranges of the rows of a batch an element selection takes, without the rows under line loads (`excluded`,
// sorted), cut every `rangeElements` rows; within the rows of `window` when one is given.
// The rows of a batch a build recovers (RecoveryWindow): every row, or the ranges given (sorted, disjoint).
struct BatchRows
{
    bool isWhole = true;
    std::span<const elements::ElementRange> ranges;
};

// The ranges of rows a build recovers, cut every `rangeElements` rows.
std::vector<elements::ElementRange> rangesOf(std::int64_t rowCount, const BatchRows& rows, std::int64_t rangeElements)
{
    if (rows.isWhole)
        return rangesOf(rowCount, std::vector<std::int64_t>{}, rangeElements);
    std::vector<elements::ElementRange> ranges; // bounded by the rows given
    for (const elements::ElementRange& range : rows.ranges)
        for (std::int64_t row = range.first; row < range.last; row += rangeElements)
            ranges.push_back({row, std::min(range.last, row + rangeElements)});
    return ranges;
}

std::vector<elements::ElementRange> selectedRanges(const elements::ElementBatchView& view,
                                                   const ElementKindSelection& selection,
                                                   const std::vector<std::int64_t>& excluded,
                                                   std::int64_t rangeElements,
                                                   std::optional<elements::ElementRange> window = std::nullopt,
                                                   const BatchRows& rows = {});

// The rows of `selection` in each range of `rows` (a windowed build).
std::vector<elements::ElementRange> selectedRangesWithin(const elements::ElementBatchView& view,
                                                         const ElementKindSelection& selection,
                                                         const std::vector<std::int64_t>& excluded,
                                                         std::int64_t rangeElements, const BatchRows& rows)
{
    std::vector<elements::ElementRange> ranges; // bounded by the rows given
    for (const elements::ElementRange& range : rows.ranges)
    {
        const std::vector<elements::ElementRange> taken =
            selectedRanges(view, selection, excluded, rangeElements, range);
        ranges.insert(ranges.end(), taken.begin(), taken.end());
    }
    return ranges;
}

std::vector<elements::ElementRange> selectedRanges(const elements::ElementBatchView& view,
                                                   const ElementKindSelection& selection,
                                                   const std::vector<std::int64_t>& excluded,
                                                   std::int64_t rangeElements,
                                                   std::optional<elements::ElementRange> window, const BatchRows& rows)
{
    if (!rows.isWhole && !window)
        return selectedRangesWithin(view, selection, excluded, rangeElements, rows);
    if (selection.isAll && !window)
        return rangesOf(view.rowCount(), excluded, rangeElements);
    std::vector<elements::ElementRange> ranges; // bounded by the selected rows
    std::size_t next = 0;
    std::int64_t row = window ? window->first : 0;
    const std::int64_t rowCount = window ? window->last : view.rowCount();
    while (row < rowCount)
    {
        while (next < excluded.size() && excluded[next] < row)
            ++next;
        const bool isExcluded = next < excluded.size() && excluded[next] == row;
        if (isExcluded || !selection.takes(view.elementId(row)))
        {
            ++row;
            continue;
        }
        std::int64_t end = row + 1;
        while (end < rowCount && end - row < rangeElements && selection.takes(view.elementId(end)) &&
               !(next < excluded.size() && excluded[next] == end))
            ++end;
        ranges.push_back({row, end});
        row = end;
    }
    return ranges;
}

// The output kind of one kind of table of a batch under a selection, the selection of that kind and the shaping of its
// table; nothing when the kind is not requested or the family gives no columns for it.
struct KindPlan
{
    elements::ElementOutput output;
    const ElementKindSelection* selection;
    TableShaping shaping;
};

std::optional<KindPlan> planOf(const assembly::AssemblyBatch& batch, const KindOfTable& kind,
                               const ElementOutputSelection& chosen)
{
    const ElementKindSelection& kindSelection = kind.output == elements::ElementOutput::Stresses  ? chosen.stress
                                                : kind.output == elements::ElementOutput::Strains ? chosen.strain
                                                                                                  : chosen.forces;
    if (!kindSelection.isRequested)
        return std::nullopt;
    const std::optional<elements::ElementOutput> output = outputOf(*batch.family, kind.output, chosen.isStrainAtFibres);
    if (!output)
        return std::nullopt;
    return KindPlan{*output, &kindSelection,
                    TableShaping{kindSelection.isCentreOnly && isCentreOrCornerElement(batch.view.card()),
                                 chosen.isMaximumShear && kind.output == elements::ElementOutput::Stresses}};
}

// Runs task(0) ... task(count − 1) on the pool, or on the caller.
void runTasks(base::ThreadPool* pool, std::size_t count, const std::function<void(std::size_t)>& task)
{
    if (pool != nullptr && count > 1)
        pool->runBatch(count, task);
    else
        for (std::size_t i = 0; i < count; ++i)
            task(i);
}

// Whether two selections take the same elements (their rows recovered alike; the shaping of their tables apart).
bool takesSameElements(const ElementKindSelection& left, const ElementKindSelection& right)
{
    if (left.isRequested != right.isRequested || left.isAll != right.isAll)
        return false;
    return left.isAll ||
           std::equal(left.members.begin(), left.members.end(), right.members.begin(), right.members.end());
}

// The tables of one kind of one batch for one case from its blocks (recoverKind, applyLineLoads): the plain rows, and
// the rows of the elements under line loads (`loaded`, when `hasLoadedRows`) in one table sorted by element row when
// the family writes both in one layout, else in a table of their own after the plain one.
void appendKindTables(const elements::ElementFamily& family, const char* kindName, const KindPlan& plan,
                      std::int32_t subcase, std::vector<const elements::ElementResultBlock*> plainBlocks,
                      const elements::ElementResultBlock* loaded, base::MemoryBudget& budget,
                      std::vector<ElementResultTable>& tables)
{
    const std::string plainType = family.resultElementType(false);
    const std::string loadedType = family.resultElementType(true);
    if (plainBlocks.empty() && loaded == nullptr)
        return;
    if (loaded != nullptr && loadedType == plainType)
    {
        plainBlocks.push_back(loaded);
        appendTable(kindName, plainType, subcase, plainBlocks, true, budget, tables, plan.shaping);
        return;
    }
    if (!plainBlocks.empty())
        appendTable(kindName, plainType, subcase, plainBlocks, false, budget, tables, plan.shaping);
    if (loaded != nullptr)
        appendTable(kindName, loadedType, subcase, {loaded}, false, budget, tables, plan.shaping);
}

// The rows of the elements under line loads of one kind of one batch for one case (recoverKind's rule): `loadedRows`
// the batch rows of every element the loads name (sorted, once each), `loaded` the rows of those the selection takes,
// recovered under their loads when the family gives such rows; false when it gives none.
bool recoverLoadedRows(const assembly::AssemblyBatch& batch, const KindPlan& plan,
                       const elements::NodalDisplacements& displacements,
                       std::span<const elements::ElementLineLoad> allLoads, std::vector<std::int64_t>& loadedRows,
                       ChargedBlock& loaded, base::MemoryBudget& budget)
{
    const elements::ElementFamily& family = *batch.family;
    loadedRows.clear();
    for (const elements::ElementLineLoad& load : allLoads)
        if (loadedRows.empty() || loadedRows.back() != load.elementRow)
            loadedRows.push_back(load.elementRow);
    std::vector<elements::ElementLineLoad> selectedLoads; // bounded by the loads of the subcase on the batch
    for (const elements::ElementLineLoad& load : allLoads)
        if (plan.selection->takes(batch.view.elementId(load.elementRow)))
            selectedLoads.push_back(load);
    if (selectedLoads.empty() || family.outputColumnsUnderLineLoads(plan.output).empty())
        return false;
    family.recoverUnderLineLoads(batch.view, selectedLoads, plan.output, displacements, loaded.block);
    loaded.charge = budget.reserve(bytesOf(loaded.block), "element result blocks");
    return true;
}

// The tables of one kind of one batch for one case, from the family's recovery of each element (the reference path):
// the elements under line loads (`allLoads`) recovered in their own layout.
void recoverKind(const assembly::AssemblyBatch& batch, const char* kindName, const KindPlan& plan, std::int32_t subcase,
                 const elements::NodalDisplacements& displacements, std::span<const elements::ElementLineLoad> allLoads,
                 std::int64_t rangeLength, base::MemoryBudget& budget, std::vector<ElementResultTable>& tables,
                 base::ThreadPool* pool, const BatchRows& rows = {})
{
    const elements::ElementFamily& family = *batch.family;
    std::vector<std::int64_t> loadedRows; // bounded by the loads of the subcase on the batch
    ChargedBlock loaded;
    const bool hasLoadedRows = recoverLoadedRows(batch, plan, displacements, allLoads, loadedRows, loaded, budget);
    const std::vector<elements::ElementRange> ranges =
        selectedRanges(batch.view, *plan.selection, loadedRows, rangeLength, std::nullopt, rows);
    std::vector<ChargedBlock> plain(ranges.size()); // bounded by the ranges of the batch
    runTasks(pool, ranges.size(),
             [&](std::size_t r)
             {
                 family.recover(batch.view, ranges[r], plan.output, displacements, plain[r].block);
                 plain[r].charge = budget.reserve(bytesOf(plain[r].block), "element result blocks");
             });
    std::vector<const elements::ElementResultBlock*> plainBlocks;
    for (const ChargedBlock& charged : plain)
        plainBlocks.push_back(&charged.block);
    appendKindTables(family, kindName, plan, subcase, std::move(plainBlocks), hasLoadedRows ? &loaded.block : nullptr,
                     budget, tables);
}

// The layer tables of one layer kind of one batch for one case (the laminated elements; a layered element has no line
// loads): the stresses and failure indices with STRESS, the strains with STRAIN. Returns whether a table was appended.
bool recoverLayerKind(const assembly::AssemblyBatch& batch, const KindOfLayerTable& kind,
                      const ElementOutputSelection& chosen, std::int32_t subcase,
                      const elements::NodalDisplacements& displacements, std::int64_t rangeLength,
                      base::MemoryBudget& budget, std::vector<ElementResultTable>& tables, base::ThreadPool* pool,
                      const BatchRows& rows = {})
{
    const elements::ElementFamily& family = *batch.family;
    if (family.layerOutputColumns(kind.output).empty())
        return false;
    const ElementKindSelection& kindSelection =
        kind.output == elements::LayerOutput::Strains ? chosen.strain : chosen.stress;
    if (!kindSelection.isRequested)
        return false;
    const std::vector<elements::ElementRange> layerRanges =
        selectedRanges(batch.view, kindSelection, {}, rangeLength, std::nullopt, rows);
    std::vector<ChargedBlock> layers(layerRanges.size()); // bounded by the ranges of the batch
    runTasks(pool, layerRanges.size(),
             [&](std::size_t r)
             {
                 family.recoverLayers(batch.view, layerRanges[r], kind.output, displacements, layers[r].block);
                 layers[r].charge = budget.reserve(bytesOf(layers[r].block), "element result blocks");
             });
    std::vector<const elements::ElementResultBlock*> blocks;
    for (const ChargedBlock& charged : layers)
        blocks.push_back(&charged.block);
    if (blocks.empty())
        return false;
    const std::size_t before = tables.size();
    const TableShaping shaping{false, chosen.isMaximumShear && kind.output == elements::LayerOutput::Stresses};
    appendTable(kind.kind, family.layerResultElementType(), subcase, blocks, false, budget, tables, shaping);
    return tables.size() > before;
}

// The tables of the kinds of one batch for one case without line loads, kinds[k] those of KINDS[k] (k < 3) and of
// LAYER_KINDS[k − 3], for the kinds `isIncluded` names: the kinds whose selections take the same elements recovered
// together, each element resolved and formulated once for all of them (ElementFamily::recoverOutputs: each output's
// rows those of its recovery alone), each kind's tables those recoverKind and recoverLayerKind make.
// PERFORMANCE: a case's tables resolved and formulated every element again for each kind (a CQUAD4 three times, a
// laminate six: its plate, place states and laminate); here once.
// Measured: on the 1-case recovery configurations.
constexpr std::size_t ALL_KINDS = 6;
using KindTablesOfCase = std::array<std::vector<ElementResultTable>, ALL_KINDS>;

KindTablesOfCase recoverKindsTogether(const assembly::AssemblyBatch& batch, const ElementOutputSelection& chosen,
                                      const std::array<bool, ALL_KINDS>& isIncluded, std::int32_t subcase,
                                      const elements::NodalDisplacements& displacements, std::int64_t rangeLength,
                                      base::MemoryBudget& budget, base::ThreadPool* pool, const BatchRows& rows = {})
{
    const elements::ElementFamily& family = *batch.family;
    // The kinds asked of the family: the element kinds, then the layer kinds.
    struct AskedKind
    {
        std::size_t index; // in KINDS, then LAYER_KINDS
        const char* kindName;
        elements::RecoveryOutput output;
        const ElementKindSelection* selection;
        TableShaping shaping;
        std::optional<KindPlan> plan; // of an element kind
    };
    std::vector<AskedKind> asked; // bounded by the kinds
    for (std::size_t k = 0; k < KINDS.size(); ++k)
        if (const std::optional<KindPlan> plan = planOf(batch, KINDS[k], chosen); plan && isIncluded[k])
            asked.push_back(
                {k, KINDS[k].kind, elements::RecoveryOutput::of(plan->output), plan->selection, plan->shaping, plan});
    for (std::size_t k = 0; k < LAYER_KINDS.size(); ++k)
    {
        const KindOfLayerTable& kind = LAYER_KINDS[k];
        const ElementKindSelection& kindSelection =
            kind.output == elements::LayerOutput::Strains ? chosen.strain : chosen.stress;
        if (!isIncluded[KINDS.size() + k] || family.layerOutputColumns(kind.output).empty() ||
            !kindSelection.isRequested)
            continue;
        asked.push_back({KINDS.size() + k, kind.kind, elements::RecoveryOutput::ofLayers(kind.output), &kindSelection,
                         TableShaping{false, chosen.isMaximumShear && kind.output == elements::LayerOutput::Stresses},
                         std::nullopt});
    }
    std::vector<std::vector<std::size_t>> together; // groups of kinds, bounded by the kinds
    for (std::size_t k = 0; k < asked.size(); ++k)
    {
        const auto same =
            std::find_if(together.begin(), together.end(), [&](const std::vector<std::size_t>& group)
                         { return takesSameElements(*asked[group.front()].selection, *asked[k].selection); });
        if (same == together.end())
            together.push_back({k});
        else
            same->push_back(k);
    }
    std::vector<std::vector<ChargedBlock>> blocks(asked.size()); // kind × range
    for (const std::vector<std::size_t>& group : together)
    {
        const std::vector<elements::ElementRange> ranges =
            selectedRanges(batch.view, *asked[group.front()].selection, {}, rangeLength, std::nullopt, rows);
        std::vector<elements::RecoveryOutput> outputs; // of the group
        for (const std::size_t k : group)
        {
            outputs.push_back(asked[k].output);
            blocks[k].resize(ranges.size());
        }
        runTasks(pool, ranges.size(),
                 [&](std::size_t r)
                 {
                     std::vector<elements::ElementResultBlock> parts(group.size());
                     family.recoverOutputs(batch.view, ranges[r], outputs, displacements, parts);
                     for (std::size_t g = 0; g < group.size(); ++g)
                     {
                         ChargedBlock& charged = blocks[group[g]][r];
                         charged.block = std::move(parts[g]);
                         charged.charge = budget.reserve(bytesOf(charged.block), "element result blocks");
                     }
                 });
    }
    KindTablesOfCase made;
    for (std::size_t k = 0; k < asked.size(); ++k)
    {
        std::vector<const elements::ElementResultBlock*> kindBlocks;
        for (const ChargedBlock& charged : blocks[k])
            kindBlocks.push_back(&charged.block);
        std::vector<ElementResultTable>& kindTables = made[asked[k].index];
        if (asked[k].plan)
            appendKindTables(family, asked[k].kindName, *asked[k].plan, subcase, std::move(kindBlocks), nullptr, budget,
                             kindTables);
        else if (!kindBlocks.empty())
            appendTable(asked[k].kindName, family.layerResultElementType(), subcase, kindBlocks, false, budget,
                        kindTables, asked[k].shaping);
        blocks[k].clear(); // the rows of a kind released once its table is made
    }
    return made;
}

// Appends the tables of `kindTables` to `tables`.
void moveTables(std::vector<ElementResultTable>& kindTables, std::vector<ElementResultTable>& tables)
{
    for (ElementResultTable& table : kindTables)
        tables.push_back(std::move(table));
    kindTables.clear();
}

// The rows of a block a table keeps (TableShaping::isCentreOnly: an element with a centre row and other rows, not at
// stations, keeps its centre rows; appendTable's rule), one flag per row; the rows of an element are contiguous.
std::vector<std::uint8_t> keptRowsOf(const elements::ElementResultBlock& block, bool isCentreOnly)
{
    std::vector<std::uint8_t> isKept(block.size(), 1); // bounded by the block
    if (!isCentreOnly)
        return isKept;
    std::size_t first = 0;
    while (first < block.size())
    {
        std::size_t last = first + 1;
        while (last < block.size() && block.elementRow(last) == block.elementRow(first) &&
               block.elementId(last) == block.elementId(first))
            ++last;
        bool hasCentre = false;
        bool hasStations = false;
        for (std::size_t r = first; r < last; ++r)
        {
            hasCentre = hasCentre || block.place(r).nodeId == 0;
            hasStations = hasStations || block.place(r).position != 0.0;
        }
        hasCentre = hasCentre && !hasStations;
        for (std::size_t r = first; r < last; ++r)
            isKept[r] = !hasCentre || block.place(r).nodeId == 0 ? 1 : 0;
        first = last;
    }
    return isKept;
}

// The family's calls for an element or a layer output (elements::RecoveryOutput).
std::vector<std::string> columnsOf(const elements::ElementFamily& family, const elements::RecoveryOutput& output)
{
    return output.isLayer ? family.layerOutputColumns(output.layerOutput) : family.outputColumns(output.output);
}

std::vector<elements::ColumnForm> formsOf(const elements::ElementFamily& family, const elements::RecoveryOutput& output)
{
    return output.isLayer ? family.layerOutputColumnForms(output.layerOutput) : family.outputColumnForms(output.output);
}

void recoverOutput(const elements::ElementFamily& family, const elements::ElementBatchView& view,
                   elements::ElementRange range, const elements::RecoveryOutput& output,
                   const elements::NodalDisplacements& displacements, elements::ElementResultBlock& block)
{
    if (output.isLayer)
        family.recoverLayers(view, range, output.layerOutput, displacements, block);
    else
        family.recover(view, range, output.output, displacements, block);
}

// The slopes of the rows of the element of batch row `row` (RecoverySlopes) of a family that gives none of its own: the
// differences of its recoveries at unit displacements of its stiffness DOFs (rows(e_d) − rows(0), exact for a Linear
// column, as the sensitivities take them: design/sensitivity/ElementItemFunction), `scratch` a zero g-set vector it
// leaves at zero.
void unitRecoverySlopes(const elements::ElementFamily& family, const elements::ElementBatchView& view, std::int64_t row,
                        const elements::RecoveryOutput& output, std::span<const elements::ColumnForm> forms,
                        std::span<double> scratch, std::size_t gridValues, elements::RecoverySlopes& slopes,
                        const elements::DisplacementSlots* slots)
{
    elements::ElementMatrixBlock stiffness;
    family.computeStiffness(view, {row, row + 1}, stiffness);
    const std::span<const elements::ElementDof> dofs =
        stiffness.empty() ? std::span<const elements::ElementDof>{} : stiffness.dofs(0);
    const std::span<double> grid = scratch.first(gridValues);
    const std::span<double> scalar = scratch.subspan(gridValues);
    const elements::NodalDisplacements unit{grid, scalar, nullptr, slots};
    const auto at = [&](const elements::ElementDof& dof) -> double&
    {
        return dof.kind == input::PointKind::Scalar ? scalar[unit.slotOf(dof)]
                                                    : grid[unit.slotOf(dof) * 6 + dof.component - 1];
    };
    elements::ElementResultBlock atRest;
    recoverOutput(family, view, {row, row + 1}, output, unit, atRest);
    slopes.reset(dofs, atRest.columns().size());
    for (std::size_t r = 0; r < atRest.size(); ++r)
        slopes.appendRow(atRest.place(r));
    elements::ElementResultBlock moved;
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
        at(dofs[d]) = 1.0;
        recoverOutput(family, view, {row, row + 1}, output, unit, moved);
        at(dofs[d]) = 0.0;
        if (moved.size() != atRest.size())
            throw std::logic_error(base::concatenate(
                {"vectorised recovery: the rows of element ", std::to_string(view.elementId(row)), " of ",
                 std::string(view.card()),
                 " depend on its displacements, so it has no slopes; its family must not complete its rows (a bug "
                 "of O_o_Solver: report it)"}));
        for (std::size_t r = 0; r < atRest.size(); ++r)
            for (std::size_t c = 0; c < forms.size() && c < atRest.columns().size(); ++c)
                if (forms[c].kind == elements::ColumnForm::Kind::Linear)
                    slopes.slopes(r, c)[d] = moved.values(r)[c] - atRest.values(r)[c];
    }
}

// The index of a DOF of an element in the g set (DofMap: six values per grid point row, then the scalar points).
std::size_t gSetIndexOf(const elements::ElementDof& dof, std::size_t gridValues,
                        const elements::NodalDisplacements& layout)
{
    return dof.kind == input::PointKind::Scalar ? gridValues + layout.slotOf(dof)
                                                : layout.slotOf(dof) * 6 + dof.component - 1;
}

// One vectorised kind of table of a batch: its table kind, the output of its table's rows and the output whose slopes
// make its Linear values (the same, the ply states of the failure indices: ElementFamily::vectorisedLayerSource, or the
// forces of a CWELD's stresses: ElementFamily::vectorisedSource), its selection and the shaping of its table.
struct VectorisedKind
{
    const char* kindName = nullptr;
    elements::RecoveryOutput table;
    elements::RecoveryOutput source;
    const ElementKindSelection* selection = nullptr;
    TableShaping shaping;
};

// What vectoriseKinds makes of one kind: the table of each case (none when the kind has no row) and the batch row of
// the element of each table row (the same in every case: applyLineLoads reads them).
struct VectorisedTables
{
    std::vector<ElementResultTable> tables;
    std::vector<std::int64_t> elementRows; // bounded by the table's rows
};

// The cases of a block by their temperatures (buildCases): the temperatures of each group (null: none), in the order
// they first appear, and the group of each case.
struct TemperatureGroups
{
    std::vector<const elements::ElementTemperatures*> temperatures; // bounded by the cases of the block
    std::vector<std::size_t> groupOfCase;
};

// The vectorised kinds of one batch for every case of a block, from the elements' recovery slopes (ElementResultBuilder
// class comment): the batch cut into ranges; the rows at rest of every range, kind and temperature group first (their
// places and constant values: the source recovered at zero displacements under the group's temperatures, the same rows
// in every group), so that each case's table is allocated once and every range writes its rows at its offset; then,
// range by range on the pool, each element's slopes of every kind once and its rows for every case, completed by the
// family. Returns per kind the tables of the cases.
std::vector<VectorisedTables> vectoriseKinds(
    const assembly::AssemblyBatch& batch, std::span<const VectorisedKind> kinds, std::span<const std::int32_t> subcases,
    linalg::ConstDenseMatrixView displacementsG, std::size_t gridValues, std::span<const double> zero,
    const TemperatureGroups& groups, std::int64_t rangeLength, base::MemoryBudget& budget, base::ThreadPool* pool,
    const BatchRows& batchRows = {}, const elements::DisplacementSlots* slots = nullptr)
{
    const elements::ElementFamily& family = *batch.family;
    const elements::ElementBatchView& view = batch.view;
    const std::size_t kindCount = kinds.size();
    const std::size_t caseCount = subcases.size();
    const std::size_t groupCount = groups.temperatures.size();
    const std::vector<elements::ElementRange> ranges = rangesOf(view.rowCount(), batchRows, rangeLength);
    const elements::NodalDisplacements layout{{}, {}, nullptr, slots};

    // What each kind holds of a range: its source rows at rest per temperature group (the elements its selection
    // takes), the rows its table keeps, and the offset of its first kept row in the tables.
    struct RangeRows
    {
        std::vector<ChargedBlock> rests;  // per temperature group
        std::vector<std::uint8_t> isKept; // bounded by the range's rows
        std::size_t kept = 0;
        std::size_t offset = 0;
    };
    std::vector<std::vector<RangeRows>> rows(kindCount); // kind × range
    for (std::vector<RangeRows>& kindRows : rows)
        kindRows.resize(ranges.size());
    // The kinds whose selections take the same elements make their rows at rest together: each element resolved and
    // formulated once for all of them (ElementFamily::recoverOutputs, the rows of each output those of its recovery).
    std::vector<std::vector<std::size_t>> together; // groups of kinds, bounded by the kinds
    for (std::size_t k = 0; k < kindCount; ++k)
    {
        const auto same =
            std::find_if(together.begin(), together.end(), [&](const std::vector<std::size_t>& group)
                         { return takesSameElements(*kinds[group.front()].selection, *kinds[k].selection); });
        if (same == together.end())
            together.push_back({k});
        else
            same->push_back(k);
    }
    runTasks(pool, ranges.size(),
             [&](std::size_t r)
             {
                 std::vector<elements::RecoveryOutput> outputs; // of the kinds of a group with rows
                 std::vector<std::size_t> outputKinds;
                 std::vector<elements::ElementResultBlock> parts;
                 for (const std::vector<std::size_t>& group : together)
                 {
                     const std::vector<elements::ElementRange> taken =
                         selectedRanges(view, *kinds[group.front()].selection, {}, rangeLength, ranges[r]);
                     for (const std::size_t k : group)
                     {
                         rows[k][r].rests.resize(groupCount);
                         for (ChargedBlock& rest : rows[k][r].rests)
                             rest.block.reset(columnsOf(family, kinds[k].source));
                     }
                     for (std::size_t g = 0; g < groupCount; ++g)
                     {
                         // The rows do not depend on the temperatures: a kind without rows in the first group has
                         // none in the others (a PSHELL batch's ply tables).
                         outputs.clear();
                         outputKinds.clear();
                         for (const std::size_t k : group)
                             if (g == 0 || rows[k][r].rests[0].block.size() > 0)
                             {
                                 outputs.push_back(kinds[k].source);
                                 outputKinds.push_back(k);
                             }
                         if (outputs.empty())
                             continue;
                         parts.resize(outputs.size());
                         const elements::NodalDisplacements atRest{zero.first(gridValues), zero.subspan(gridValues),
                                                                   groups.temperatures[g], slots};
                         for (const elements::ElementRange& piece : taken)
                         {
                             family.recoverOutputs(view, piece, outputs, atRest, parts);
                             for (std::size_t o = 0; o < outputs.size(); ++o)
                             {
                                 elements::ElementResultBlock& rest = rows[outputKinds[o]][r].rests[g].block;
                                 const elements::ElementResultBlock& part = parts[o];
                                 for (std::size_t i = 0; i < part.size(); ++i)
                                     rest.append(part.elementRow(i), part.elementId(i), part.place(i), part.values(i));
                             }
                         }
                         for (const std::size_t k : outputKinds)
                         {
                             ChargedBlock& rest = rows[k][r].rests[g];
                             if (g > 0 && rest.block.size() != rows[k][r].rests[0].block.size())
                                 throw std::logic_error(base::concatenate(
                                     {"vectorised recovery: the rows of ", std::string(view.card()),
                                      " at rest depend on the temperatures (a bug of O_o_Solver: report it)"}));
                             rest.charge = budget.reserve(bytesOf(rest.block), "vectorised recovery rows at rest");
                         }
                     }
                     for (const std::size_t k : group)
                     {
                         RangeRows& range = rows[k][r];
                         range.isKept = keptRowsOf(range.rests[0].block, kinds[k].shaping.isCentreOnly);
                         for (const std::uint8_t kept : range.isKept)
                             range.kept += kept;
                     }
                 }
             });

    // Per kind: the columns of its source and of its table, the Linear columns of its source, the tables of the cases
    // with their row labels.
    struct KindTables
    {
        std::vector<std::string> sourceColumns;
        std::vector<std::string> components; // the table's
        std::size_t width = 0;               // the table's
        std::vector<std::size_t> linear;     // the Linear columns of the source
        bool isMadeFromSource = false;       // the table's rows made from the source's (not completed in place)
        bool isMaximumShear = false;
        PlaneStressColumns plane;
        bool hasPositions = false;
        VectorisedTables made;
    };
    std::vector<KindTables> made(kindCount);
    for (std::size_t k = 0; k < kindCount; ++k)
    {
        KindTables& kind = made[k];
        const VectorisedKind& plan = kinds[k];
        std::size_t rowCount = 0;
        for (RangeRows& range : rows[k])
        {
            range.offset = rowCount;
            rowCount += range.kept;
            for (std::size_t i = 0; i < range.rests[0].block.size() && !kind.hasPositions; ++i)
                kind.hasPositions = range.isKept[i] != 0 && range.rests[0].block.place(i).position != 0.0;
        }
        if (rowCount == 0)
            continue; // the result tables hold no empty table
        kind.sourceColumns = columnsOf(family, plan.source);
        kind.components = columnsOf(family, plan.table);
        kind.width = kind.components.size();
        kind.isMadeFromSource = !(plan.source == plan.table);
        const std::vector<elements::ColumnForm> forms = formsOf(family, plan.source);
        if (forms.size() != kind.sourceColumns.size())
            throw std::logic_error(base::concatenate({"vectorised recovery: ", std::string(view.card()), " declares ",
                                                      std::to_string(forms.size()), " column forms for ",
                                                      std::to_string(kind.sourceColumns.size()),
                                                      " columns (a bug of O_o_Solver: report it)"}));
        for (std::size_t c = 0; c < forms.size(); ++c)
            if (forms[c].kind == elements::ColumnForm::Kind::Linear)
                kind.linear.push_back(c);
        const std::optional<PlaneStressColumns> planeColumns =
            plan.shaping.isMaximumShear ? planeStressColumnsOf(kind.components) : std::nullopt;
        kind.isMaximumShear = planeColumns.has_value();
        kind.plane = planeColumns.value_or(PlaneStressColumns{});
        kind.made.elementRows.reserve(rowCount);
        for (const RangeRows& range : rows[k])
            for (std::size_t i = 0; i < range.rests[0].block.size(); ++i)
                if (range.isKept[i] != 0)
                    kind.made.elementRows.push_back(range.rests[0].block.elementRow(i));
        const std::string elementType =
            plan.table.isLayer ? family.layerResultElementType() : family.resultElementType(false);
        for (std::size_t c = 0; c < caseCount; ++c)
        {
            ElementResultTable table;
            table.kind = plan.kindName;
            table.elementType = elementType;
            table.subcase = subcases[c];
            table.components = kind.components;
            if (kind.isMaximumShear)
                table.components[kind.plane.vonMises] = "maxShear";
            // Every row is written by the ranges below: no zeros first.
            table.ids = base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result ids", base::FOR_OVERWRITE);
            table.nodeIds =
                base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result grid points", base::FOR_OVERWRITE);
            table.layers =
                base::BudgetedBuffer<std::int32_t>(budget, rowCount, "element result layers", base::FOR_OVERWRITE);
            table.values = base::BudgetedBuffer<double>(budget, rowCount * kind.width, "element result values",
                                                        base::FOR_OVERWRITE);
            table.hasPositions = kind.hasPositions;
            if (kind.hasPositions)
                table.positions =
                    base::BudgetedBuffer<double>(budget, rowCount, "element result positions", base::FOR_OVERWRITE);
            kind.made.tables.push_back(std::move(table));
        }
    }

    std::vector<elements::RecoveryOutput> sources; // per kind
    for (const VectorisedKind& kind : kinds)
        sources.push_back(kind.source);
    runTasks(pool, ranges.size(),
             [&](std::size_t r)
             {
                 // The kept rows of the range per kind and case (in the columns of the source), charged to the budget
                 // while they are held.
                 std::vector<std::vector<elements::ElementResultBlock>> blocks(kindCount);
                 std::uint64_t bytes = 0;
                 for (std::size_t k = 0; k < kindCount; ++k)
                 {
                     blocks[k].resize(caseCount);
                     for (elements::ElementResultBlock& block : blocks[k])
                     {
                         block.reset(made[k].sourceColumns);
                         block.reserve(rows[k][r].kept);
                     }
                     bytes += rows[k][r].kept * caseCount *
                              (std::max(made[k].sourceColumns.size(), made[k].width) * sizeof(double) +
                               sizeof(std::int64_t) + sizeof(std::int32_t) + sizeof(elements::ResultPlace));
                 }
                 const base::MemoryReservation charge = budget.reserve(bytes, "vectorised recovery blocks");
                 // The unit displacements of a family without slopes of its own (unitRecoverySlopes), made when needed.
                 std::optional<base::BudgetedBuffer<double>> scratch;
                 std::vector<std::size_t> next(kindCount, 0); // per kind, the next row of its rows at rest
                 std::vector<std::size_t> taking;             // the kinds whose rows hold the element
                 std::vector<elements::RecoveryOutput> takingSources;
                 std::vector<elements::RecoverySlopes> takingSlopes;
                 std::vector<double> local;  // the element's displacements of every case, DOF-major
                 std::vector<double> sums;   // the Linear values of a row for every case
                 std::vector<double> values; // the values of a row, column-major over the cases
                 std::vector<double> rowValues;
                 for (std::int64_t elementRow = ranges[r].first; elementRow < ranges[r].last; ++elementRow)
                 {
                     taking.clear();
                     takingSources.clear();
                     for (std::size_t k = 0; k < kindCount; ++k)
                     {
                         const elements::ElementResultBlock& rest = rows[k][r].rests[0].block;
                         if (next[k] < rest.size() && rest.elementRow(next[k]) == elementRow)
                         {
                             taking.push_back(k);
                             takingSources.push_back(sources[k]);
                         }
                     }
                     if (taking.empty())
                         continue;
                     takingSlopes.resize(taking.size());
                     if (!family.recoverySlopesOfOutputs(view, elementRow, takingSources, takingSlopes))
                         for (std::size_t t = 0; t < taking.size(); ++t)
                         {
                             const elements::RecoveryOutput& source = takingSources[t];
                             const bool isGiven =
                                 source.isLayer
                                     ? family.layerRecoverySlopes(view, elementRow, source.layerOutput, takingSlopes[t])
                                     : family.recoverySlopes(view, elementRow, source.output, takingSlopes[t]);
                             if (isGiven)
                                 continue;
                             if (!scratch)
                             {
                                 scratch.emplace(budget, zero.size(), "vectorised recovery unit displacements",
                                                 base::FOR_OVERWRITE);
                                 std::fill_n(scratch->data(), zero.size(), 0.0);
                             }
                             unitRecoverySlopes(family, view, elementRow, source, formsOf(family, source),
                                                scratch->span(), gridValues, takingSlopes[t], slots);
                         }
                     for (std::size_t t = 0; t < taking.size(); ++t)
                     {
                         const std::size_t k = taking[t];
                         const elements::RecoverySlopes& elementSlopes = takingSlopes[t];
                         const RangeRows& range = rows[k][r];
                         const elements::ElementResultBlock& structure = range.rests[0].block;
                         const std::size_t first = next[k];
                         std::size_t last = first;
                         while (last < structure.size() && structure.elementRow(last) == elementRow)
                             ++last;
                         next[k] = last;
                         if (elementSlopes.size() != last - first ||
                             elementSlopes.columnCount() != made[k].sourceColumns.size())
                             throw std::logic_error(base::concatenate(
                                 {"vectorised recovery: the slopes of element ",
                                  std::to_string(view.elementId(elementRow)), " of ", std::string(view.card()),
                                  " have ", std::to_string(elementSlopes.size()), " rows where its recovery has ",
                                  std::to_string(last - first), " (a bug of O_o_Solver: report it)"}));
                         const std::vector<elements::ElementDof>& dofs = elementSlopes.dofs();
                         const std::size_t dofCount = dofs.size();
                         local.resize(dofCount * caseCount);
                         for (std::size_t d = 0; d < dofCount; ++d)
                         {
                             const std::size_t index = gSetIndexOf(dofs[d], gridValues, layout);
                             for (std::size_t c = 0; c < caseCount; ++c)
                                 local[d * caseCount + c] = displacementsG(index, c);
                         }
                         const std::size_t columnCount = made[k].sourceColumns.size();
                         values.resize(columnCount * caseCount);
                         sums.resize(caseCount);
                         for (std::size_t i = first; i < last; ++i)
                         {
                             if (range.isKept[i] == 0)
                                 continue;
                             // The rows at rest of each case's temperatures: its constant values.
                             for (std::size_t c = 0; c < caseCount; ++c)
                             {
                                 const std::span<const double> constant =
                                     range.rests[groups.groupOfCase[c]].block.values(i);
                                 for (std::size_t column = 0; column < columnCount; ++column)
                                     values[column * caseCount + c] = constant[column];
                             }
                             // PERFORMANCE: the Linear column of every case at once, value(0) + Σ_d S(row, column,
                             // d)·u_d with d in the element's order for each case (its chain of terms, so its bits),
                             // the cases contiguous: the inner loop over the cases is vectorised; the sums were made
                             // case by case, each reading the row's slopes again.
                             // Measured: on the recovery configurations of several cases.
                             for (const std::size_t column : made[k].linear)
                             {
                                 const std::span<const double> slope = elementSlopes.slopes(i - first, column);
                                 double* const sum = sums.data();
                                 const double* const constant = values.data() + column * caseCount;
                                 for (std::size_t c = 0; c < caseCount; ++c)
                                     sum[c] = constant[c];
                                 for (std::size_t d = 0; d < dofCount; ++d)
                                 {
                                     const double factor = slope[d];
                                     const double* const u = local.data() + d * caseCount;
                                     for (std::size_t c = 0; c < caseCount; ++c)
                                         sum[c] += factor * u[c];
                                 }
                                 std::copy(sums.begin(), sums.end(),
                                           values.begin() + static_cast<std::ptrdiff_t>(column * caseCount));
                             }
                             rowValues.resize(columnCount);
                             for (std::size_t c = 0; c < caseCount; ++c)
                             {
                                 for (std::size_t column = 0; column < columnCount; ++column)
                                     rowValues[column] = values[column * caseCount + c];
                                 blocks[k][c].append(elementRow, structure.elementId(i), structure.place(i), rowValues);
                             }
                         }
                     }
                 }
                 std::vector<elements::ElementResultBlock> madeBlocks; // the rows of a table made from its source's
                 for (std::size_t k = 0; k < kindCount; ++k)
                 {
                     if (made[k].made.tables.empty())
                         continue;
                     const VectorisedKind& plan = kinds[k];
                     std::span<elements::ElementResultBlock> tableBlocks = blocks[k];
                     if (plan.table.isLayer && made[k].isMadeFromSource)
                     {
                         madeBlocks.resize(caseCount);
                         for (elements::ElementResultBlock& block : madeBlocks)
                             block.reset(made[k].components);
                         family.completeRecoveredLayerRows(view, plan.table.layerOutput, blocks[k], madeBlocks);
                         tableBlocks = madeBlocks;
                     }
                     else if (plan.table.isLayer)
                     {
                         family.completeRecoveredLayerRows(view, plan.table.layerOutput, {}, blocks[k]);
                     }
                     else if (made[k].isMadeFromSource)
                     {
                         madeBlocks.resize(caseCount);
                         for (elements::ElementResultBlock& block : madeBlocks)
                             block.reset(made[k].components);
                         family.completeRecoveredRowsFromSource(view, plan.table.output, blocks[k], madeBlocks);
                         tableBlocks = madeBlocks;
                     }
                     else
                     {
                         family.completeRecoveredRows(view, plan.table.output, blocks[k]);
                     }
                     const std::size_t width = made[k].width;
                     for (std::size_t c = 0; c < caseCount; ++c)
                     {
                         ElementResultTable& table = made[k].made.tables[c];
                         const elements::ElementResultBlock& block = tableBlocks[c];
                         if (block.size() != rows[k][r].kept || block.columns().size() != width)
                             throw std::logic_error(base::concatenate(
                                 {"vectorised recovery: ", std::string(view.card()), " made ",
                                  std::to_string(block.size()), " rows of ", plan.kindName, " where its slopes gave ",
                                  std::to_string(rows[k][r].kept), " (a bug of O_o_Solver: report it)"}));
                         for (std::size_t i = 0; i < block.size(); ++i)
                         {
                             const std::size_t at = rows[k][r].offset + i;
                             const elements::ResultPlace place = block.place(i);
                             table.ids[at] = block.elementId(i);
                             table.nodeIds[at] = place.nodeId;
                             table.layers[at] = place.layer;
                             if (made[k].hasPositions)
                                 table.positions[at] = place.position;
                             const std::span<const double> recovered = block.values(i);
                             double* row = table.values.data() + at * width;
                             std::copy(recovered.begin(), recovered.end(), row);
                             if (made[k].isMaximumShear)
                             {
                                 // appendTable's maximum shear, (σ1 − σ2)/2 = √(((σa − σb)/2)² + τ²).
                                 const PlaneStressColumns& plane = made[k].plane;
                                 const double halfDifference = 0.5 * (row[plane.normal1] - row[plane.normal2]);
                                 row[plane.vonMises] =
                                     std::sqrt(halfDifference * halfDifference + row[plane.shear] * row[plane.shear]);
                             }
                         }
                     }
                 }
             });
    std::vector<VectorisedTables> result(kindCount);
    for (std::size_t k = 0; k < kindCount; ++k)
        result[k] = std::move(made[k].made);
    return result;
}

// The tables of one kind of one case whose elements of the batch carry line loads (`allLoads`), from its vectorised
// table (`vectorised`, its rows' batch rows `elementRows`; no table when the kind had no row): the rows of the loaded
// elements replaced by their recovery under the loads, recoverKind's tables (its plain rows the vectorised ones).
void applyLineLoads(const assembly::AssemblyBatch& batch, const char* kindName, const KindPlan& plan,
                    std::int32_t subcase, const elements::NodalDisplacements& displacements,
                    std::span<const elements::ElementLineLoad> allLoads, const ElementResultTable* vectorised,
                    std::span<const std::int64_t> elementRows, base::MemoryBudget& budget,
                    std::vector<ElementResultTable>& tables)
{
    const elements::ElementFamily& family = *batch.family;
    std::vector<std::int64_t> loadedRows; // bounded by the loads of the subcase on the batch
    ChargedBlock loaded;
    const bool hasLoadedRows = recoverLoadedRows(batch, plan, displacements, allLoads, loadedRows, loaded, budget);
    ChargedBlock plain;
    plain.block.reset(family.outputColumns(plan.output));
    if (vectorised != nullptr)
    {
        const std::size_t width = vectorised->componentCount();
        for (std::size_t r = 0; r < vectorised->rowCount(); ++r)
        {
            if (std::binary_search(loadedRows.begin(), loadedRows.end(), elementRows[r]))
                continue;
            const elements::ResultPlace place{vectorised->nodeIds[r], vectorised->layers[r],
                                              vectorised->hasPositions ? vectorised->positions[r] : 0.0};
            // A maximum shear in the von Mises column is made again by appendTable from the same components: the
            // same value.
            plain.block.append(elementRows[r], vectorised->ids[r], place,
                               std::span<const double>(vectorised->values.data() + r * width, width));
        }
        plain.charge = budget.reserve(bytesOf(plain.block), "element result blocks");
    }
    std::vector<const elements::ElementResultBlock*> plainBlocks;
    if (plain.block.size() > 0)
        plainBlocks.push_back(&plain.block);
    appendKindTables(family, kindName, plan, subcase, std::move(plainBlocks), hasLoadedRows ? &loaded.block : nullptr,
                     budget, tables);
}

} // namespace

ElementResultBuilder::ElementResultBuilder(const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                                           std::int64_t rangeElements)
    : m_set(&set), m_dofMap(&dofMap), m_rangeElements(rangeElements)
{
    if (rangeElements < 1)
        throw std::invalid_argument(
            base::concatenate({"ElementResultBuilder: expected a positive range of elements, "
                               "found ",
                               std::to_string(rangeElements), "; pass RANGE_ELEMENTS or a smaller positive count"}));
}

std::int64_t ElementResultBuilder::rangeLengthOf(std::int64_t rows, const base::ThreadPool* pool) const noexcept
{
    if (pool == nullptr || pool->threadCount() <= 1)
        return m_rangeElements;
    // PERFORMANCE: a batch cut into RANGES_PER_THREAD ranges per thread (each element's rows depend on its own data
    // only, so the ranges change no bit): a mid-size batch was one or two ranges of RANGE_ELEMENTS, on as many threads.
    // Measured: on a wing box deck and a deck of fasteners.
    const auto tasks = static_cast<std::int64_t>(pool->threadCount()) * RANGES_PER_THREAD;
    const std::int64_t even = (rows + tasks - 1) / tasks;
    return std::min(m_rangeElements, std::max(MIN_RANGE_ELEMENTS, even));
}

void ElementResultBuilder::build(std::int32_t subcase, std::span<const double> displacementsG,
                                 std::span<const std::vector<elements::ElementLineLoad>> lineLoads,
                                 base::MemoryBudget& budget, std::vector<ElementResultTable>& tables,
                                 base::ThreadPool* pool, const base::CancellationToken* cancellation,
                                 const elements::ElementTemperatures* temperatures,
                                 const ElementOutputSelection* selection) const
{
    if (displacementsG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"ElementResultBuilder: expected ", std::to_string(m_dofMap->gSetSize()),
                               " displacements (the g set), found ", std::to_string(displacementsG.size())}));
    const std::span<const assembly::AssemblyBatch> batches = m_set->batches();
    if (!lineLoads.empty() && lineLoads.size() != batches.size())
        throw std::invalid_argument(base::concatenate(
            {"ElementResultBuilder: expected the line loads of each of the ", std::to_string(batches.size()),
             " batches of the element set (or none), found ", std::to_string(lineLoads.size()), " lists"}));
    const ElementOutputSelection everything;
    const ElementOutputSelection& chosen = selection != nullptr ? *selection : everything;
    // The g-set order of DofMap is that of NodalDisplacements: six values per grid point row, then the scalar points.
    const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap->gridCount());
    const elements::NodalDisplacements displacements{displacementsG.first(gridValues),
                                                     displacementsG.subspan(gridValues), temperatures};

    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const assembly::AssemblyBatch& batch = batches[b];
        if (!batch.family->facets().has(elements::Facet::Recovery) || batch.view.rowCount() == 0)
            continue;
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        const std::span<const elements::ElementLineLoad> allLoads =
            lineLoads.empty() ? std::span<const elements::ElementLineLoad>{} : std::span(lineLoads[b]);
        const std::int64_t rangeLength = rangeLengthOf(batch.view.rowCount(), pool);
        if (allLoads.empty())
        {
            std::array<bool, ALL_KINDS> isIncluded{};
            isIncluded.fill(true);
            KindTablesOfCase made =
                recoverKindsTogether(batch, chosen, isIncluded, subcase, displacements, rangeLength, budget, pool);
            for (std::vector<ElementResultTable>& kindTables : made)
                moveTables(kindTables, tables);
            continue;
        }
        for (const KindOfTable& kind : KINDS)
            if (const std::optional<KindPlan> plan = planOf(batch, kind, chosen))
                recoverKind(batch, kind.kind, *plan, subcase, displacements, allLoads, rangeLength, budget, tables,
                            pool);
        for (const KindOfLayerTable& kind : LAYER_KINDS)
            recoverLayerKind(batch, kind, chosen, subcase, displacements, rangeLength, budget, tables, pool);
    }
}

void ElementResultBuilder::buildCases(std::span<const std::int32_t> subcases,
                                      linalg::ConstDenseMatrixView displacementsG, std::size_t vectorisedCases,
                                      base::MemoryBudget& budget, std::span<std::vector<ElementResultTable>> tables,
                                      base::ThreadPool* pool, const base::CancellationToken* cancellation,
                                      const ElementOutputSelection* selection, std::span<const CaseLoads> loads) const
{
    buildCasesIn(subcases, displacementsG, vectorisedCases, budget, tables, pool, cancellation, selection, loads,
                 nullptr);
}

void ElementResultBuilder::buildWindow(std::span<const std::int32_t> subcases,
                                       linalg::ConstDenseMatrixView displacements, std::size_t vectorisedCases,
                                       const RecoveryWindow& window, base::MemoryBudget& budget,
                                       std::span<std::vector<ElementResultTable>> tables, base::ThreadPool* pool,
                                       const base::CancellationToken* cancellation,
                                       const ElementOutputSelection* selection) const
{
    buildCasesIn(subcases, displacements, vectorisedCases, budget, tables, pool, cancellation, selection, {}, &window);
}

void ElementResultBuilder::buildCasesIn(std::span<const std::int32_t> subcases,
                                        linalg::ConstDenseMatrixView displacementsG, std::size_t vectorisedCases,
                                        base::MemoryBudget& budget, std::span<std::vector<ElementResultTable>> tables,
                                        base::ThreadPool* pool, const base::CancellationToken* cancellation,
                                        const ElementOutputSelection* selection, std::span<const CaseLoads> loads,
                                        const RecoveryWindow* window) const
{
    const auto gSize = window != nullptr ? 6 * window->gridSlotCount + window->scalarSlotCount
                                         : static_cast<std::size_t>(m_dofMap->gSetSize());
    const elements::DisplacementSlots* slots = window != nullptr ? &window->slots : nullptr;
    if (displacementsG.rows != gSize || displacementsG.columns != subcases.size() || tables.size() != subcases.size())
        throw std::invalid_argument(base::concatenate(
            {"ElementResultBuilder::buildCases: expected the displacements of the g set (", std::to_string(gSize),
             " rows) and a table list for each of the ", std::to_string(subcases.size()), " cases, found ",
             std::to_string(displacementsG.rows), " × ", std::to_string(displacementsG.columns), " and ",
             std::to_string(tables.size()), " lists"}));
    const std::span<const assembly::AssemblyBatch> batches = m_set->batches();
    if (!loads.empty() && loads.size() != subcases.size())
        throw std::invalid_argument(base::concatenate(
            {"ElementResultBuilder::buildCases: expected the loads of each of the ", std::to_string(subcases.size()),
             " cases (or none), found ", std::to_string(loads.size())}));
    for (const CaseLoads& caseLoads : loads)
        if (!caseLoads.lineLoads.empty() && caseLoads.lineLoads.size() != batches.size())
            throw std::invalid_argument(
                base::concatenate({"ElementResultBuilder::buildCases: expected the line loads of each of the ",
                                   std::to_string(batches.size()), " batches of the element set (or none), found ",
                                   std::to_string(caseLoads.lineLoads.size()), " lists"}));
    const ElementOutputSelection everything;
    const ElementOutputSelection& chosen = selection != nullptr ? *selection : everything;
    const auto gridValues = window != nullptr
                                ? 6 * window->gridSlotCount
                                : static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap->gridCount());
    // The cases by their temperatures: the cases of one group share their rows at rest.
    TemperatureGroups groups;
    std::vector<elements::NodalDisplacements> cases; // bounded by the block
    for (std::size_t c = 0; c < subcases.size(); ++c)
    {
        const elements::ElementTemperatures* temperatures = loads.empty() ? nullptr : loads[c].temperatures;
        const auto found = std::find(groups.temperatures.begin(), groups.temperatures.end(), temperatures);
        groups.groupOfCase.push_back(static_cast<std::size_t>(found - groups.temperatures.begin()));
        if (found == groups.temperatures.end())
            groups.temperatures.push_back(temperatures);
        const std::span<const double> column(displacementsG.pointer(0, c), gSize);
        cases.push_back({column.first(gridValues), column.subspan(gridValues), temperatures, slots});
    }
    const auto lineLoadsOf = [&](std::size_t c, std::size_t b) -> std::span<const elements::ElementLineLoad>
    {
        if (loads.empty() || loads[c].lineLoads.empty())
            return {};
        return loads[c].lineLoads[b];
    };
    std::optional<base::BudgetedBuffer<double>> zero; // the displacements of the rows at rest, made when first needed

    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const assembly::AssemblyBatch& batch = batches[b];
        const elements::ElementFamily& family = *batch.family;
        if (!family.facets().has(elements::Facet::Recovery) || batch.view.rowCount() == 0)
            continue;
        BatchRows rows;
        if (window != nullptr)
        {
            if (b >= window->rowsOfBatch.size() || window->rowsOfBatch[b].empty())
                continue;
            rows = {false, window->rowsOfBatch[b]};
        }
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        const std::int64_t rangeLength = rangeLengthOf(batch.view.rowCount(), pool);
        // The kinds the family completes and the cases reach the minimum of: vectorised together (their slopes made
        // once per element for all of them), the element kinds first, then the layer kinds.
        std::array<std::optional<KindPlan>, KINDS.size()> plans;
        std::array<std::optional<std::size_t>, KINDS.size()> vectorisedIndex;
        std::array<std::optional<std::size_t>, LAYER_KINDS.size()> vectorisedLayerIndex;
        std::vector<VectorisedKind> vectorised; // bounded by the kinds
        for (std::size_t k = 0; k < KINDS.size(); ++k)
        {
            plans[k] = planOf(batch, KINDS[k], chosen);
            if (!plans[k])
                continue;
            const std::size_t minimum = family.vectorisedRecoveryMinCases(plans[k]->output);
            if (minimum > 0 && vectorisedCases >= minimum)
            {
                vectorisedIndex[k] = vectorised.size();
                const elements::RecoveryOutput output = elements::RecoveryOutput::of(plans[k]->output);
                const elements::RecoveryOutput source =
                    elements::RecoveryOutput::of(family.vectorisedSource(plans[k]->output));
                vectorised.push_back({KINDS[k].kind, output, source, plans[k]->selection, plans[k]->shaping});
            }
        }
        for (std::size_t k = 0; k < LAYER_KINDS.size(); ++k)
        {
            const elements::LayerOutput output = LAYER_KINDS[k].output;
            const ElementKindSelection& kindSelection =
                output == elements::LayerOutput::Strains ? chosen.strain : chosen.stress;
            const std::size_t minimum = family.vectorisedLayerRecoveryMinCases(output);
            if (family.layerOutputColumns(output).empty() || !kindSelection.isRequested || minimum == 0 ||
                vectorisedCases < minimum)
                continue;
            vectorisedLayerIndex[k] = vectorised.size();
            vectorised.push_back(
                {LAYER_KINDS[k].kind, elements::RecoveryOutput::ofLayers(output),
                 elements::RecoveryOutput::ofLayers(family.vectorisedLayerSource(output)), &kindSelection,
                 TableShaping{false, chosen.isMaximumShear && output == elements::LayerOutput::Stresses}});
        }
        std::vector<VectorisedTables> made; // per vectorised kind
        if (!vectorised.empty())
        {
            if (!zero)
                zero.emplace(budget, gSize, "vectorised recovery zero displacements");
            made = vectoriseKinds(batch, vectorised, subcases, displacementsG, gridValues, zero->span(), groups,
                                  std::min(rangeLength, VECTORISED_RANGE_ELEMENTS), budget, pool, rows, slots);
        }
        // The kinds recovered case by case of a case without line loads, together (recoverKindsTogether): their
        // tables per case, the layer kinds' taken by the loop of the layers.
        std::array<bool, ALL_KINDS> isTogether{};
        for (std::size_t k = 0; k < KINDS.size(); ++k)
            isTogether[k] = plans[k] && !vectorisedIndex[k];
        for (std::size_t k = 0; k < LAYER_KINDS.size(); ++k)
            isTogether[KINDS.size() + k] = !vectorisedLayerIndex[k];
        const bool hasTogether = std::find(isTogether.begin(), isTogether.end(), true) != isTogether.end();
        std::vector<std::optional<KindTablesOfCase>> togetherTables(subcases.size()); // per case
        for (std::size_t c = 0; c < subcases.size(); ++c)
        {
            if (hasTogether && lineLoadsOf(c, b).empty())
                togetherTables[c] = recoverKindsTogether(batch, chosen, isTogether, subcases[c], cases[c], rangeLength,
                                                         budget, pool, rows);
            for (std::size_t k = 0; k < KINDS.size(); ++k)
            {
                if (!plans[k])
                    continue;
                const std::span<const elements::ElementLineLoad> caseLoads = lineLoadsOf(c, b);
                if (!vectorisedIndex[k] && togetherTables[c])
                {
                    moveTables((*togetherTables[c])[k], tables[c]);
                    continue;
                }
                if (!vectorisedIndex[k])
                {
                    recoverKind(batch, KINDS[k].kind, *plans[k], subcases[c], cases[c], caseLoads, rangeLength, budget,
                                tables[c], pool, rows);
                    continue;
                }
                VectorisedTables& kindTables = made[*vectorisedIndex[k]];
                ElementResultTable* table = kindTables.tables.empty() ? nullptr : &kindTables.tables[c];
                if (!caseLoads.empty())
                {
                    applyLineLoads(batch, KINDS[k].kind, *plans[k], subcases[c], cases[c], caseLoads, table,
                                   kindTables.elementRows, budget, tables[c]);
                    continue;
                }
                if (table != nullptr)
                    tables[c].push_back(std::move(*table));
            }
        }
        // The layers (the plies of the laminates); a kind recovered case by case whose rows do not depend on the
        // displacements: a batch whose first case has none has none for the others either (a PSHELL batch: every
        // element resolved for nothing).
        for (std::size_t k = 0; k < LAYER_KINDS.size(); ++k)
            for (std::size_t c = 0; c < subcases.size(); ++c)
            {
                if (vectorisedLayerIndex[k])
                {
                    VectorisedTables& kindTables = made[*vectorisedLayerIndex[k]];
                    if (kindTables.tables.empty())
                        break;
                    tables[c].push_back(std::move(kindTables.tables[c]));
                    continue;
                }
                if (togetherTables[c])
                {
                    moveTables((*togetherTables[c])[KINDS.size() + k], tables[c]);
                    continue;
                }
                if (!recoverLayerKind(batch, LAYER_KINDS[k], chosen, subcases[c], cases[c], rangeLength, budget,
                                      tables[c], pool, rows) &&
                    c == 0)
                    break;
            }
    }
}

} // namespace oos::output
