#include "FrontAssembly.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/threading/ParallelFor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

// The elements of a range of a front's columns one task assembles: 512 kB of doubles, a few dozen microseconds.
constexpr std::size_t ASSEMBLY_PIECE_VALUES = std::size_t{1} << 16;
// The fronts of at least this many elements in their lower triangle (2 MB: an order of about 720) are assembled by
// ranges of columns on the pool.
constexpr std::size_t PARALLEL_ASSEMBLY_VALUES = std::size_t{1} << 18;

[[noreturn]] void throwOutsideStructure(std::int32_t row, std::int32_t column)
{
    throw std::invalid_argument(base::concatenate(
        {"supernodal factorisation: the matrix entry (", std::to_string(row), ", ", std::to_string(column),
         ") lies outside the structure of the symbolic factor; build the SymmetricCsc with a DofSelection in the order "
         "of SymbolicFactor::nodeOrder(), on the graph the symbolic analysis was made on"}));
}

// The positions in the front of the rows of a child's update matrix: delayed columns into the delayed range (from
// delayedStart), the rows R(child), ascending, merged against the columns and R(s) of the front.
void mapChildRows(const FrontLayout& layout, const UpdateMatrix& child, std::int32_t delayedStart,
                  std::vector<std::int32_t>& positions)
{
    positions.resize(static_cast<std::size_t>(child.order));
    const auto delayed = static_cast<std::int32_t>(child.delayedEquations.size());
    for (std::int32_t i = 0; i < delayed; ++i)
        positions[static_cast<std::size_t>(i)] = delayedStart + i;
    const std::int32_t endColumn = layout.firstColumn + layout.columnCount;
    std::size_t cursor = 0;
    for (std::size_t k = 0; k < child.rows.size(); ++k)
    {
        const std::int32_t row = child.rows[k];
        std::int32_t position = 0;
        if (row >= layout.firstColumn && row < endColumn)
            position = row - layout.firstColumn;
        else
        {
            while (cursor < layout.rows.size() && layout.rows[cursor] < row)
                ++cursor;
            // The rows of a child lie in the columns and rows of its parent (the symbolic factor's invariant).
            OOS_ASSERT(cursor < layout.rows.size() && layout.rows[cursor] == row);
            position = layout.candidateCount() + static_cast<std::int32_t>(cursor);
        }
        positions[static_cast<std::size_t>(delayed) + k] = position;
    }
}

// A range of consecutive rows of a child's update matrix whose positions in the front are consecutive too.
struct PositionRun
{
    std::size_t firstRow = 0;
    std::size_t firstPosition = 0;
    std::size_t length = 0;
};

// The runs of increasing positions (a child without delayed rows): a grid point's DOFs, and the rows a child shares in
// a block with its parent, map to consecutive positions.
void collectRuns(const std::vector<std::int32_t>& positions, std::vector<PositionRun>& runs)
{
    runs.clear();
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        const auto position = static_cast<std::size_t>(positions[i]);
        if (!runs.empty() && runs.back().firstPosition + runs.back().length == position)
            ++runs.back().length;
        else
            runs.push_back({i, position, 1});
    }
}

// target[k] += source[k] for k < count: the adds of one run, the loop the compiler vectorises (each element is the
// same single addition as one at a time).
void addRun(double* target, const double* source, std::size_t count) noexcept
{
    for (std::size_t k = 0; k < count; ++k)
        target[k] += source[k];
}

// A child's update matrix mapped into its parent's front.
struct ChildMap
{
    std::vector<std::int32_t> positions; // per row of the child, its position in the front
    std::vector<PositionRun> runs;       // without delayed rows: the runs of consecutive positions
};

// The columns of a front, cut into ranges of about ASSEMBLY_PIECE_VALUES elements (from its order alone: the pieces
// never depend on the threads, and each element is written by one piece, with the same sum, whatever the pieces).
std::vector<std::size_t> pieceStarts(std::size_t order, bool isCut)
{
    std::vector<std::size_t> starts{0};
    if (!isCut)
        return starts;
    std::size_t values = 0;
    for (std::size_t p = 0; p < order; ++p)
    {
        values += order - p;
        if (values >= ASSEMBLY_PIECE_VALUES && p + 1 < order)
        {
            starts.push_back(p + 1);
            values = 0;
        }
    }
    return starts;
}

} // namespace

struct FrontAssembler::Maps
{
    std::vector<ChildMap> maps;
    bool hasDelayedRows = false;
};

FrontAssembler::FrontAssembler(const FrontLayout& layout, const SymmetricCsc* matrix,
                               std::span<const UpdateMatrix> children, FrontStorage front)
    : m_layout(layout), m_matrix(matrix), m_children(children), m_front(front), m_maps(std::make_unique<Maps>())
{
    const auto order = static_cast<std::size_t>(layout.order());
    const auto candidates = static_cast<std::size_t>(layout.candidateCount());
    const std::size_t rowCount = order - candidates;
    OOS_ASSERT(front.panel.rows == order && front.panel.columns == candidates &&
                      front.contribution.rows == rowCount && front.contribution.columns == rowCount);
    m_maps->maps.resize(children.size());
    std::int32_t delayedStart = layout.columnCount;
    for (std::size_t c = 0; c < children.size(); ++c)
    {
        mapChildRows(layout, children[c], delayedStart, m_maps->maps[c].positions);
        delayedStart += static_cast<std::int32_t>(children[c].delayedEquations.size());
        if (children[c].delayedEquations.empty())
            collectRuns(m_maps->maps[c].positions, m_maps->maps[c].runs);
        else
            m_maps->hasDelayedRows = true;
    }
    OOS_ASSERT(delayedStart == layout.candidateCount());
}

FrontAssembler::~FrontAssembler() = default;

bool FrontAssembler::isCut() const noexcept
{
    return !m_maps->hasDelayedRows;
}

std::pair<std::size_t, std::size_t> FrontAssembler::childColumnsIn(std::size_t child, std::size_t first,
                                                                   std::size_t end) const
{
    const std::vector<std::int32_t>& positions = m_maps->maps[child].positions;
    const auto begin = static_cast<std::size_t>(
        std::lower_bound(positions.begin(), positions.end(), static_cast<std::int32_t>(first)) - positions.begin());
    const auto stop = static_cast<std::size_t>(
        std::lower_bound(positions.begin(), positions.end(), static_cast<std::int32_t>(end)) - positions.begin());
    return {begin, stop};
}

void FrontAssembler::assembleColumns(std::size_t first, std::size_t end, AssemblyPhase phase) const
{
    const FrontLayout& layout = m_layout;
    const SymmetricCsc* matrix = m_matrix;
    const std::span<const UpdateMatrix> children = m_children;
    const FrontStorage& front = m_front;
    const std::vector<ChildMap>& maps = m_maps->maps;
    const auto order = static_cast<std::size_t>(layout.order());
    const auto candidates = static_cast<std::size_t>(layout.candidateCount());
    const std::size_t rowCount = order - candidates;
    // The update matrix's columns: square of its view's leading dimension, or by blocks of columns (UpdateLayout).
    const UpdateLayout ownLayout{static_cast<std::int64_t>(rowCount), front.contributionBlockColumns};
    const auto contributionColumn = [&](std::size_t t) -> double*
    {
        if (!ownLayout.isBlocked())
            return front.contribution.data + t * front.contribution.leadingDimension;
        return front.contribution.data + ownLayout.columnOffset(static_cast<std::int64_t>(t));
    };
    // Element (i, j) of the front, i >= j: in the panel when j is a candidate, else in the update matrix.
    const auto element = [&](std::size_t i, std::size_t j) -> double&
    { return j < candidates ? front.panel(i, j) : contributionColumn(j - candidates)[i - candidates]; };
    // Column j of a child's update matrix (element i at [i], i >= j).
    const auto childColumn = [](const UpdateMatrix& child, std::size_t j) -> const double*
    {
        const UpdateLayout layout{child.order, child.blockColumns};
        return child.values + layout.columnOffset(static_cast<std::int64_t>(j));
    };

    const std::int32_t endColumn = layout.firstColumn + layout.columnCount;
    const bool clears = phase != AssemblyPhase::ChildrenOnly;
    const bool addsChildren = phase != AssemblyPhase::ClearOnly;
    {
        for (std::size_t j = first; j < std::min(end, candidates) && clears; ++j)
        {
            double* column = &front.panel(0, j);
            std::fill(column + (front.isPanelClearedWhole ? 0 : j), column + order, 0.0);
        }
        for (std::size_t p = std::max(first, candidates); p < end && clears; ++p)
        {
            const std::size_t t = p - candidates;
            double* column = contributionColumn(t);
            std::fill(column + t, column + rowCount, 0.0);
        }

        // The entries of the matrix in the columns of the supernode (lower triangle: rows >= the column).
        const auto lastEntryColumn = static_cast<std::int32_t>(
            matrix == nullptr ? 0 : std::min(end, static_cast<std::size_t>(layout.columnCount)));
        for (auto j = static_cast<std::int32_t>(first); j < lastEntryColumn; ++j)
        {
            const auto offsets = matrix->columnOffsets();
            const auto rowIndices = matrix->rowIndices();
            const auto values = matrix->values();
            const std::int32_t column = layout.firstColumn + j;
            double* target = &front.panel(0, static_cast<std::size_t>(j));
            auto cursor = layout.rows.begin();
            for (auto k = offsets[static_cast<std::size_t>(column)]; k < offsets[static_cast<std::size_t>(column) + 1];
                 ++k)
            {
                const std::int32_t row = rowIndices[static_cast<std::size_t>(k)];
                std::size_t position = 0;
                if (row < endColumn)
                {
                    if (row < column)
                        throwOutsideStructure(row, column);
                    position = static_cast<std::size_t>(row - layout.firstColumn);
                }
                else
                {
                    cursor = std::lower_bound(cursor, layout.rows.end(), row);
                    if (cursor == layout.rows.end() || *cursor != row)
                        throwOutsideStructure(row, column);
                    position = candidates + static_cast<std::size_t>(cursor - layout.rows.begin());
                }
                target[position] += values[static_cast<std::size_t>(k)];
            }
        }

        // The update matrices of the children, one after the other.
        for (std::size_t c = 0; c < children.size() && addsChildren; ++c)
        {
            const UpdateMatrix& child = children[c];
            const std::vector<std::int32_t>& positions = maps[c].positions;
            const std::vector<PositionRun>& runs = maps[c].runs;
            const auto childOrder = static_cast<std::size_t>(child.order);
            if (!child.delayedEquations.empty())
            {
                // Without a cut: every column of the front is in this range.
                for (std::size_t j = 0; j < childOrder; ++j)
                {
                    const auto pj = static_cast<std::size_t>(positions[j]);
                    const double* source = childColumn(child, j);
                    for (std::size_t i = j; i < childOrder; ++i)
                    {
                        const auto pi = static_cast<std::size_t>(positions[i]);
                        if (pi >= pj)
                            element(pi, pj) += source[i];
                        else
                            element(pj, pi) += source[i];
                    }
                }
                continue;
            }
            // PERFORMANCE: the positions increase with the rows, so the lower triangle maps to the lower triangle, the
            // columns of the child landing in [first, end) are consecutive, and each column is added by runs of
            // consecutive positions (vectorised), each element receiving the same addition as one at a time.
            // Measured with the front in place (one thread): the shell of 100 000 DOFs 0.52 -> 0.50 s, the 3D
            // grid of 50 000 DOFs 1.64 -> 1.45 s.
            const auto firstChildColumn = static_cast<std::size_t>(
                std::lower_bound(positions.begin(), positions.end(), static_cast<std::int32_t>(first)) -
                positions.begin());
            const auto endChildColumn = static_cast<std::size_t>(
                std::lower_bound(positions.begin(), positions.end(), static_cast<std::int32_t>(end)) -
                positions.begin());
            std::size_t run = 0;
            for (std::size_t j = firstChildColumn; j < endChildColumn; ++j)
            {
                while (runs[run].firstRow + runs[run].length <= j)
                    ++run;
                const auto pj = static_cast<std::size_t>(positions[j]);
                // Rows of the column at position p go to target[p - shift].
                const bool isCandidate = pj < candidates;
                double* target = isCandidate ? &front.panel(0, pj) : contributionColumn(pj - candidates);
                const std::size_t shift = isCandidate ? 0 : candidates;
                const double* source = childColumn(child, j);
                for (std::size_t k = run; k < runs.size(); ++k)
                {
                    const std::size_t firstRow = std::max(runs[k].firstRow, j);
                    const std::size_t endRow = runs[k].firstRow + runs[k].length;
                    addRun(target + (runs[k].firstPosition + (firstRow - runs[k].firstRow) - shift), source + firstRow,
                           endRow - firstRow);
                }
            }
        }
    }
}

namespace
{

void assembleFrontPieces(const FrontLayout& layout, const SymmetricCsc* matrix, std::span<const UpdateMatrix> children,
                         FrontStorage front, base::ThreadPool* pool, const base::CancellationToken* cancellation,
                         AssemblyPhase phase)
{
    const FrontAssembler assembler(layout, matrix, children, front);
    const auto order = static_cast<std::size_t>(layout.order());
    // PERFORMANCE: a large front of the top of the tree is assembled by ranges of its columns on the pool (its
    // clearing, its first touches and its extend-add are memory-bound: one thread left the others waiting). Each
    // element belongs to one range and receives the same sum, in the same order, as in one pass. A child with delayed
    // rows (indefinite mode) maps a column to several: such a front is assembled in one pass.
    // Measured (six threads): the 3D grid of 100 000 DOFs 1.42 -> 1.33 s, of
    // 200 000 DOFs 5.04 -> 4.78 s, the shell of 1 M DOFs 3.57 -> 3.46 s.
    const bool isCut = pool != nullptr && assembler.isCut() && order * (order + 1) / 2 >= PARALLEL_ASSEMBLY_VALUES;
    const std::vector<std::size_t> starts = pieceStarts(order, isCut);
    if (starts.size() == 1)
    {
        assembler.assembleColumns(0, order, phase);
        return;
    }
    base::parallelFor(
        pool, starts.size(), 1,
        [&](std::size_t begin, std::size_t endPiece)
        {
            for (std::size_t piece = begin; piece < endPiece; ++piece)
                assembler.assembleColumns(starts[piece], piece + 1 < starts.size() ? starts[piece + 1] : order, phase);
        },
        cancellation);
}

} // namespace

void assembleFront(const FrontLayout& layout, const SymmetricCsc& matrix, std::span<const UpdateMatrix> children,
                   FrontStorage front, base::ThreadPool* pool, const base::CancellationToken* cancellation)
{
    assembleFrontPieces(layout, &matrix, children, front, pool, cancellation, AssemblyPhase::Whole);
}

void assembleFrontInPhases(const FrontLayout& layout, std::span<const UpdateMatrix> children, FrontStorage front,
                           const std::function<void()>& addOwnTerms, base::ThreadPool* pool,
                           const base::CancellationToken* cancellation)
{
    assembleFrontPieces(layout, nullptr, children, front, pool, cancellation, AssemblyPhase::ClearOnly);
    if (addOwnTerms)
        addOwnTerms();
    assembleFrontPieces(layout, nullptr, children, front, pool, cancellation, AssemblyPhase::ChildrenOnly);
}

void clearRightHandSides(DenseMatrixView columns, DenseMatrixView rows)
{
    for (std::size_t j = 0; j < columns.columns; ++j)
        std::fill_n(&columns(0, j), columns.rows, 0.0);
    for (std::size_t j = 0; j < rows.columns; ++j)
        std::fill_n(&rows(0, j), rows.rows, 0.0);
}

void addChildRightHandSides(const FrontLayout& layout, std::span<const UpdateRightHandSides> children,
                            DenseMatrixView columns, DenseMatrixView rows)
{
    OOS_ASSERT(layout.delayedEquations.empty());
    std::vector<std::int32_t> positions; // per row of a child, bounded by the front
    const auto candidates = static_cast<std::size_t>(layout.columnCount);
    for (const UpdateRightHandSides& child : children)
    {
        const UpdateMatrix asMatrix{child.values, static_cast<std::int32_t>(child.rows.size()), {}, child.rows};
        mapChildRows(layout, asMatrix, layout.columnCount, positions);
        const std::size_t childRows = child.rows.size();
        for (std::size_t j = 0; j < columns.columns; ++j)
        {
            const double* source = child.values + j * childRows;
            for (std::size_t i = 0; i < childRows; ++i)
            {
                const auto position = static_cast<std::size_t>(positions[i]);
                if (position < candidates)
                    columns(position, j) += source[i];
                else
                    rows(position - candidates, j) += source[i];
            }
        }
    }
}

} // namespace oos::linalg
