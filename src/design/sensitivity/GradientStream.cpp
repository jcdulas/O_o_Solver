#include "oos/design/sensitivity/GradientStream.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OutputVolumeError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/design/sensitivity/SolutionReads.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::design
{

namespace
{

constexpr std::uint64_t DOUBLE_BYTES = sizeof(double);
// The pieces the columns of a block solve are prepared and reduced in (each piece with its own g-set vector and column
// of equations): at most this many, whatever the threads, so that the memory of the stream does not depend on them.
constexpr std::size_t STREAM_COLUMN_PIECES = 8;

// The columns of one piece when `count` columns are cut into at most STREAM_COLUMN_PIECES pieces.
[[nodiscard]] std::size_t columnsPerPiece(std::size_t count) noexcept
{
    return std::max<std::size_t>(1, base::pieceCount(count, STREAM_COLUMN_PIECES));
}

// a·b in 64 bits, saturated (a size estimate never wraps round).
[[nodiscard]] std::uint64_t multiplySaturated(std::uint64_t a, std::uint64_t b) noexcept
{
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        return std::numeric_limits<std::uint64_t>::max();
    return a * b;
}

[[nodiscard]] std::uint64_t addSaturated(std::uint64_t a, std::uint64_t b) noexcept
{
    return b > std::numeric_limits<std::uint64_t>::max() - a ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

void requireBlock(const GradientStreamOptions& options)
{
    if (options.block == 0)
        throw std::invalid_argument(
            "the gradient stream: expected a block of at least one column, found 0; give block= a positive "
            "number (256 by default)");
}

// The rows of the stream by subcase: per subcase (the index in inputs.subcases), the rows that read the
// displacements, in row order.
class RowsBySubcase
{
public:
    explicit RowsBySubcase(const GradientStreamInputs& inputs) : m_rows(inputs.subcases.size())
    {
        std::map<std::int32_t, std::size_t> indexOf;
        for (std::size_t s = 0; s < inputs.subcases.size(); ++s)
            indexOf[inputs.subcases[s].label] = s;
        m_subcaseOfRow.assign(inputs.rows.size(), NO_SUBCASE);
        for (std::size_t r = 0; r < inputs.rows.size(); ++r)
        {
            const ItemLinearisation& item = inputs.linearised->at(inputs.rows[r]);
            if (!item.readsDisplacements())
                continue;
            const auto found = indexOf.find(item.subcase);
            if (found == indexOf.end())
                // no remedy: the stream factorises every subcase its rows read; a missing one is a bug.
                throw std::invalid_argument(base::concatenate(
                    {"the gradient stream: row ", std::to_string(r), " reads the displacements of subcase ",
                     std::to_string(item.subcase), ", which has no factorised system in the stream"}));
            m_rows[found->second].push_back(r);
            m_subcaseOfRow[r] = found->second;
        }
    }

    [[nodiscard]] const std::vector<std::size_t>& of(std::size_t subcase) const { return m_rows[subcase]; }
    // The subcase whose displacements a row reads, or NO_SUBCASE for a row that reads none.
    [[nodiscard]] std::size_t subcaseOf(std::size_t row) const { return m_subcaseOfRow[row]; }

    static constexpr std::size_t NO_SUBCASE = std::numeric_limits<std::size_t>::max();

private:
    std::vector<std::vector<std::size_t>> m_rows; // bounded by the rows
    std::vector<std::size_t> m_subcaseOfRow;
};

// The sizes of the systems the stream solves on (the largest of its subcases).
struct SystemSizes
{
    std::uint64_t equations = 0;
    std::uint64_t gSize = 0;
};

[[nodiscard]] SystemSizes measureSystems(const GradientStreamInputs& inputs)
{
    SystemSizes sizes;
    for (const StreamSubcase& subcase : inputs.subcases)
    {
        sizes.equations =
            std::max<std::uint64_t>(sizes.equations, static_cast<std::uint64_t>(subcase.system->equationCount()));
        sizes.gSize =
            std::max<std::uint64_t>(sizes.gSize, static_cast<std::uint64_t>(subcase.system->dofMap().gSetSize()));
    }
    return sizes;
}

// The largest memory the stream's buffers hold at once, in an orientation (GradientStream.hpp, "Memory").
[[nodiscard]] std::uint64_t estimateBlockPeak(StreamOrientation orientation, std::uint64_t rows, std::uint64_t block,
                                              const SystemSizes& sizes)
{
    // A g-set vector and a column of equations per piece of a block's columns, beside the blocks.
    const std::uint64_t vectors = multiplySaturated(addSaturated(sizes.gSize, sizes.equations),
                                                    std::min<std::uint64_t>(block, STREAM_COLUMN_PIECES));
    std::uint64_t values = 0;
    if (orientation == StreamOrientation::ByColumns) // rows × B of G, equations × B of right-hand sides
        values = addSaturated(multiplySaturated(rows, block), multiplySaturated(sizes.equations, block));
    else // (gSize + equations) × B of adjoint vectors and right-hand sides, B × B of a tile
        values = addSaturated(multiplySaturated(addSaturated(sizes.gSize, sizes.equations), block),
                              multiplySaturated(block, block));
    return multiplySaturated(addSaturated(values, vectors), DOUBLE_BYTES);
}

// Tracks the memory of the stream's buffers alive at once.
class PeakTracker
{
public:
    void hold(std::uint64_t bytes) noexcept
    {
        m_current += bytes;
        m_peak = std::max(m_peak, m_current);
    }
    void drop(std::uint64_t bytes) noexcept { m_current -= std::min(m_current, bytes); }
    [[nodiscard]] std::uint64_t peak() const noexcept { return m_peak; }

private:
    std::uint64_t m_current = 0;
    std::uint64_t m_peak = 0;
};

// The direct method by blocks of columns.
void streamByColumns(const GradientStreamInputs& inputs, const GradientStreamOptions& options,
                     const RowsBySubcase& rowsBySubcase, GradientSink& sink, base::MemoryBudget& budget,
                     GradientStreamReport& report)
{
    const std::size_t rowCount = inputs.rows.size();
    const std::size_t columnCount = inputs.columnCount;
    PeakTracker tracker;
    for (std::size_t first = 0; first < columnCount; first += options.block)
    {
        const std::size_t count = std::min(options.block, columnCount - first);
        base::BudgetedBuffer<double> block(budget, rowCount * count, "gradient stream block of G");
        tracker.hold(block.reservedBytes());
        // Every row starts at its explicit part in these columns (the global items keep it).
        for (std::size_t r = 0; r < rowCount; ++r)
            rowSlice(inputs.linearised->at(inputs.rows[r]).explicitPart, first, block.span().subspan(r * count, count));
        for (std::size_t s = 0; s < inputs.subcases.size(); ++s)
        {
            const std::vector<std::size_t>& rows = rowsBySubcase.of(s);
            if (rows.empty())
                continue;
            const StreamSubcase& subcase = inputs.subcases[s];
            std::vector<std::size_t> variables; // bounded by the block
            for (std::size_t j = first; j < first + count; ++j)
                if (inputs.loads->hasLoad(j, subcase.loadIndex))
                    variables.push_back(j);
            if (variables.empty())
                continue;
            const SensitivitySystem& system = *subcase.system;
            const auto gSize = static_cast<std::size_t>(system.dofMap().gSetSize());
            const auto equations = static_cast<std::size_t>(system.equationCount());
            linalg::DenseMatrix rightHandSides(budget, equations, variables.size(), "gradient stream right-hand sides");
            const std::size_t pieceColumns = columnsPerPiece(variables.size());
            const std::uint64_t solveBytes = multiplySaturated((gSize + equations) * DOUBLE_BYTES,
                                                               base::pieceCount(variables.size(), pieceColumns)) +
                                             multiplySaturated(equations, variables.size()) * DOUBLE_BYTES;
            tracker.hold(solveBytes);
            // The condensed path recovers the units the rows read only.
            std::vector<ItemKey> keys; // bounded by the rows of the subcase
            for (const std::size_t r : rows)
                keys.push_back(inputs.rows[r]);
            const base::BudgetedBuffer<std::uint8_t> reads =
                SolutionReads::ofWeights(system, *inputs.linearised, keys, budget);
            // K_ll·du_l/dx_j = (Tᵀ·b_j)_l, du_g/dx_j = T·du_l/dx_j (DirectSensitivity).
            // PERFORMANCE: the pseudo-loads of a block are assembled and carried to the equations, and the solutions
            // carried back to the g set and reduced to the rows of G, by pieces of columns in parallel: each column is
            // computed alone, as before (the same bits), and each piece writes its own columns. These passes over the
            // g set cost about half of a column's solve at 1 M DOFs on one thread.
            // Measured: 512 variables on 1 M DOFs, 6 threads, the stream 17.7 s → 15.0 s.
            base::parallelFor(inputs.pool, variables.size(), pieceColumns,
                              [&](std::size_t begin, std::size_t end)
                              {
                                  base::BudgetedBuffer<double> vectorG(budget, gSize, "gradient stream pseudo-load");
                                  base::BudgetedBuffer<double> column(budget, equations, "gradient stream equations");
                                  for (std::size_t b = begin; b < end; ++b)
                                  {
                                      inputs.loads->assemble(variables[b], subcase.loadIndex, vectorG.span());
                                      system.toEquationsInPlace(vectorG.span(), column.span());
                                      for (std::size_t e = 0; e < equations; ++e)
                                          rightHandSides(e, b) = column[e];
                                  }
                              });
            if (equations > 0)
                system.solve(rightHandSides.view(), reads.span());
            report.solves += static_cast<std::int64_t>(variables.size());
            base::parallelFor(inputs.pool, variables.size(), pieceColumns,
                              [&](std::size_t begin, std::size_t end)
                              {
                                  base::BudgetedBuffer<double> vectorG(budget, gSize, "gradient stream displacements");
                                  base::BudgetedBuffer<double> column(budget, equations, "gradient stream equations");
                                  for (std::size_t b = begin; b < end; ++b)
                                  {
                                      for (std::size_t e = 0; e < equations; ++e)
                                          column[e] = rightHandSides(e, b);
                                      system.fromEquations(column.span(), vectorG.span());
                                      const std::size_t c = variables[b] - first;
                                      for (const std::size_t r : rows)
                                          block[r * count + c] +=
                                              dotSparse(inputs.linearised->at(inputs.rows[r]).weights, vectorG.span());
                                  }
                              });
            tracker.drop(solveBytes);
        }
        sink.consume({0, rowCount, first, count, block.span()});
        ++report.blocks;
        tracker.drop(block.reservedBytes());
    }
    report.measuredPeakBytes = tracker.peak();
}

// The adjoint method by blocks of rows, each row block streamed by column blocks.
void streamByRows(const GradientStreamInputs& inputs, const GradientStreamOptions& options,
                  const RowsBySubcase& rowsBySubcase, GradientSink& sink, base::MemoryBudget& budget,
                  GradientStreamReport& report)
{
    const std::size_t rowCount = inputs.rows.size();
    const std::size_t columnCount = inputs.columnCount;
    const SystemSizes sizes = measureSystems(inputs);
    const auto gSize = static_cast<std::size_t>(sizes.gSize);
    PeakTracker tracker;
    for (std::size_t firstRow = 0; firstRow < rowCount; firstRow += options.block)
    {
        const std::size_t count = std::min(options.block, rowCount - firstRow);
        // λ_g of the block's rows (zero for a row that reads no displacement), one g-set vector each.
        base::BudgetedBuffer<double> adjoints(budget, gSize * count, "gradient stream adjoint vectors");
        tracker.hold(adjoints.reservedBytes());
        std::vector<std::uint8_t> hasAdjoint(count, 0); // bounded by the block
        for (std::size_t s = 0; s < inputs.subcases.size(); ++s)
        {
            std::vector<std::size_t> rows; // the block's rows of this subcase, bounded by the block
            for (const std::size_t r : rowsBySubcase.of(s))
                if (r >= firstRow && r < firstRow + count)
                    rows.push_back(r);
            if (rows.empty())
                continue;
            const SensitivitySystem& system = *inputs.subcases[s].system;
            const auto subcaseG = static_cast<std::size_t>(system.dofMap().gSetSize());
            const auto equations = static_cast<std::size_t>(system.equationCount());
            base::BudgetedBuffer<double> vectorG(budget, subcaseG, "gradient stream adjoint weights");
            base::BudgetedBuffer<double> column(budget, equations, "gradient stream equations");
            linalg::DenseMatrix rightHandSides(budget, equations, rows.size(), "gradient stream right-hand sides");
            const std::uint64_t solveBytes = vectorG.reservedBytes() + column.reservedBytes() +
                                             multiplySaturated(equations, rows.size()) * DOUBLE_BYTES;
            tracker.hold(solveBytes);
            // The condensed path recovers the adjoint vectors in the driven units only.
            const base::BudgetedBuffer<std::uint8_t> reads =
                SolutionReads::ofPseudoLoads(system, *inputs.loads, inputs.subcases[s].loadIndex, budget);
            // K_ll·λ_l = (Tᵀ·cᵀ)_l, λ_g = T·λ_l (AdjointSensitivity; an equation item with its chained weights).
            for (std::size_t b = 0; b < rows.size(); ++b)
            {
                std::fill(vectorG.data(), vectorG.data() + subcaseG, 0.0);
                for (const auto& [dof, weight] : inputs.linearised->at(inputs.rows[rows[b]]).weights)
                    vectorG[static_cast<std::size_t>(dof)] = weight;
                system.toEquations(vectorG.span(), column.span());
                for (std::size_t e = 0; e < equations; ++e)
                    rightHandSides(e, b) = column[e];
            }
            if (equations > 0)
                system.solve(rightHandSides.view(), reads.span());
            report.solves += static_cast<std::int64_t>(rows.size());
            for (std::size_t b = 0; b < rows.size(); ++b)
            {
                for (std::size_t e = 0; e < equations; ++e)
                    column[e] = rightHandSides(e, b);
                const std::size_t k = rows[b] - firstRow;
                std::fill(adjoints.data() + k * gSize, adjoints.data() + (k + 1) * gSize, 0.0);
                system.fromEquations(column.span(), adjoints.span().subspan(k * gSize, subcaseG));
                hasAdjoint[k] = 1;
            }
            tracker.drop(solveBytes);
        }
        // dr_i/dx_j = ∂r_i/∂x_j + λ_iᵀ·b_j, column block by column block.
        const std::size_t tileWidth = std::min(options.block, std::max<std::size_t>(columnCount, 1));
        base::BudgetedBuffer<double> tile(budget, count * tileWidth, "gradient stream tile of G");
        base::BudgetedBuffer<double> products(budget, tileWidth, "gradient stream products");
        tracker.hold(tile.reservedBytes() + products.reservedBytes());
        for (std::size_t firstColumn = 0; firstColumn < columnCount; firstColumn += options.block)
        {
            const std::size_t columns = std::min(options.block, columnCount - firstColumn);
            for (std::size_t k = 0; k < count; ++k)
            {
                const std::size_t r = firstRow + k;
                const std::span<double> tileRow = tile.span().subspan(k * columns, columns);
                rowSlice(inputs.linearised->at(inputs.rows[r]).explicitPart, firstColumn, tileRow);
                if (hasAdjoint[k] == 0)
                    continue;
                const StreamSubcase& subcase = inputs.subcases[rowsBySubcase.subcaseOf(r)];
                const std::span<double> rowProducts = products.span().first(columns);
                inputs.loads->dotRange(
                    subcase.loadIndex,
                    adjoints.span().subspan(k * gSize, static_cast<std::size_t>(subcase.system->dofMap().gSetSize())),
                    firstColumn, rowProducts);
                for (std::size_t c = 0; c < columns; ++c)
                    tileRow[c] += rowProducts[c];
            }
            sink.consume({firstRow, count, firstColumn, columns, tile.span().first(count * columns)});
            ++report.blocks;
        }
        tracker.drop(tile.reservedBytes() + products.reservedBytes());
        tracker.drop(adjoints.reservedBytes());
    }
    report.measuredPeakBytes = tracker.peak();
}

} // namespace

std::string_view streamOrientationName(StreamOrientation orientation) noexcept
{
    return orientation == StreamOrientation::ByColumns ? "columns" : "rows";
}

GradientSize GradientStream::size(std::size_t rowCount, std::size_t columnCount) noexcept
{
    return {rowCount, columnCount, multiplySaturated(multiplySaturated(rowCount, columnCount), DOUBLE_BYTES)};
}

void GradientStream::requireDestination(const GradientSize& size, std::string_view destination,
                                        std::uint64_t capacityBytes)
{
    if (size.bytes <= capacityBytes)
        return;
    std::vector<std::string> remedies{
        "give sink= a callable, which receives G block by block and holds none of it",
        "ask fewer rows: gradients(x, mode=\"screened\") differentiates the retained responses only",
        "ask fewer columns: link variables with add_link, or design fewer fields"};
    throw base::OutputVolumeError(
        base::concatenate({"the full gradient matrix G (", std::to_string(size.rowCount), " items × ",
                           std::to_string(size.columnCount), " variables, ", base::formatBytes(size.bytes),
                           ") does not fit its destination, ", destination, ", which can take ",
                           base::formatBytes(capacityBytes), "; nothing was computed. Remedies: ", remedies[0], "; ",
                           remedies[1], "; ", remedies[2]}),
        size.bytes, remedies);
}

GradientStreamPlan GradientStream::plan(const GradientStreamInputs& inputs, const GradientStreamOptions& options)
{
    requireBlock(options);
    GradientStreamPlan plan;
    plan.size = size(inputs.rows.size(), inputs.columnCount);
    const RowsBySubcase rowsBySubcase(inputs);
    for (std::size_t s = 0; s < inputs.subcases.size(); ++s)
    {
        if (rowsBySubcase.of(s).empty())
            continue;
        plan.solvesByRows += static_cast<std::int64_t>(rowsBySubcase.of(s).size());
        for (std::size_t j = 0; j < inputs.columnCount; ++j)
            plan.solvesByColumns += inputs.loads->hasLoad(j, inputs.subcases[s].loadIndex) ? 1 : 0;
    }
    switch (options.method)
    {
    case SensitivityMethod::Direct:
        plan.orientation = StreamOrientation::ByColumns;
        break;
    case SensitivityMethod::Adjoint:
    case SensitivityMethod::Chained:
        plan.orientation = StreamOrientation::ByRows;
        break;
    case SensitivityMethod::Automatic:
        // The fewer solves; the direct order on a tie (MethodChoice).
        plan.orientation =
            plan.solvesByRows < plan.solvesByColumns ? StreamOrientation::ByRows : StreamOrientation::ByColumns;
        break;
    }
    plan.blockPeakBytes =
        estimateBlockPeak(plan.orientation, inputs.rows.size(), options.block, measureSystems(inputs));
    return plan;
}

GradientStreamReport GradientStream::run(const GradientStreamInputs& inputs, const GradientStreamOptions& options,
                                         GradientSink& sink, base::MemoryBudget& budget)
{
    GradientStreamReport report;
    report.plan = plan(inputs, options);
    const RowsBySubcase rowsBySubcase(inputs);
    if (report.plan.orientation == StreamOrientation::ByColumns)
        streamByColumns(inputs, options, rowsBySubcase, sink, budget, report);
    else
        streamByRows(inputs, options, rowsBySubcase, sink, budget, report);
    return report;
}

} // namespace oos::design
