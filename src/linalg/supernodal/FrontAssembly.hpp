#pragma once

#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/UpdateLayout.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace oos::base
{
class CancellationToken;
class ThreadPool;
} // namespace oos::base

namespace oos::linalg
{

// The update matrix a child hands to its parent front (order × order, lower triangle, stored as `layout` says:
// square of leading dimension order unless blockColumns is set): its first delayedCount rows are the columns the
// child could not eliminate (their equations in delayedEquations), the others the rows R(child) of the symbolic factor,
// ascending.
struct UpdateMatrix
{
    const double* values = nullptr;
    std::int32_t order = 0;
    std::span<const std::int32_t> delayedEquations;
    std::span<const std::int32_t> rows;
    std::int32_t blockColumns = 0; // UpdateLayout: 0 square
};

// The front of one supernode (internal to linalg/supernodal): its positions are
//   [0, columnCount)                    the columns of the supernode, equations firstColumn + k;
//   [columnCount, candidateCount())     the columns delayed into it by its children, in child order;
//   [candidateCount(), order())         its rows R(s), ascending.
// The candidates (the first two ranges) are fully summed: they may be eliminated in this front.
struct FrontLayout
{
    std::int32_t supernode = 0;
    std::int32_t firstColumn = 0;
    std::int32_t columnCount = 0;
    std::vector<std::int32_t> delayedEquations; // the columns delayed into the front (indefinite mode only)
    std::span<const std::int32_t> rows;

    [[nodiscard]] std::int32_t delayedCount() const noexcept
    {
        return static_cast<std::int32_t>(delayedEquations.size());
    }
    [[nodiscard]] std::int32_t candidateCount() const noexcept { return columnCount + delayedCount(); }
    [[nodiscard]] std::int32_t order() const noexcept
    {
        return candidateCount() + static_cast<std::int32_t>(rows.size());
    }
};

// Where a front is assembled: its candidate columns over all its rows (the panel, order × candidateCount) and the
// trailing block of its rows R(s) (the update matrix it leaves to its parent, rowCount × rowCount, lower triangle).
// Either one matrix of the front's order (whole: the panel is its first columns, the update matrix its trailing block,
// and its upper triangle is neither cleared nor written), or two buffers (split: the panel in place in the factor,
// cleared whole so that its entries above the diagonal block are the zeros of the factor's layout, and the update
// matrix apart, handed to the parent as it is).
struct FrontStorage
{
    DenseMatrixView panel;
    DenseMatrixView contribution; // its data, rows and columns; laid out as contributionBlockColumns says
    bool isPanelClearedWhole = false;
    std::int32_t contributionBlockColumns = 0; // UpdateLayout of the update matrix: 0 square (its leading dimension)

    [[nodiscard]] static FrontStorage whole(DenseMatrixView front, std::size_t candidateCount) noexcept
    {
        const std::size_t rowCount = front.rows - candidateCount;
        const DenseMatrixView trailing =
            rowCount == 0 ? DenseMatrixView{} : front.block(candidateCount, candidateCount, rowCount, rowCount);
        return {front.block(0, 0, front.rows, candidateCount), trailing, false};
    }
    [[nodiscard]] static FrontStorage split(DenseMatrixView panel, DenseMatrixView contribution,
                                            std::int32_t contributionBlockColumns = 0) noexcept
    {
        return {panel, contribution, true, contributionBlockColumns};
    }
};

// Assembles a front: clears it (its lower triangle; a split panel whole), adds the entries of the
// matrix in the columns of the supernode, then the update matrices of the children, in the order given (extend-add).
// Every entry of the front is thus the same sum in the same order whoever assembles it and wherever it is stored. With
// a pool, a large front without delayed rows is assembled by ranges of its columns on it (the same sums).
// Throws std::invalid_argument when a matrix entry lies outside the structure of the symbolic factor (the matrix was
// not built in the order and on the graph of the symbolic analysis); OperationCancelledError when the token is
// cancelled between two ranges.
void assembleFront(const FrontLayout& layout, const SymmetricCsc& matrix, std::span<const UpdateMatrix> children,
                   FrontStorage front, base::ThreadPool* pool = nullptr,
                   const base::CancellationToken* cancellation = nullptr);

// The same assembly with the front's own terms given by the caller instead of a matrix (a front source):
// the front cleared, then addOwnTerms() (the caller adds its terms, on this thread), then the update matrices of the
// children in the order given. Every element is thus 0 + the caller's terms + the children's, the order assembleFront
// gives a matrix's entries (a caller that adds a matrix's entries of the supernode's columns gets its bits). The
// clearing and the extend-add of a large front by ranges of its columns on the pool, as assembleFront.
void assembleFrontInPhases(const FrontLayout& layout, std::span<const UpdateMatrix> children, FrontStorage front,
                           const std::function<void()>& addOwnTerms, base::ThreadPool* pool = nullptr,
                           const base::CancellationToken* cancellation = nullptr);

// What one pass over a range of a front's columns does (FrontAssembler): everything (clear, the matrix's entries, the
// children), the clearing alone, or the children alone (assembleFrontInPhases, whose caller adds its own terms between
// the two).
enum class AssemblyPhase : std::uint8_t
{
    Whole,
    ClearOnly,
    ChildrenOnly
};

// A front's assembly by ranges of its columns (the tasks of the streamed top part assemble a front by
// blocks of columns, each block as soon as the children's update columns it reads are final). The children's maps are
// made once here. Each element of the front belongs to one range and receives, in a range's pass, the sum it receives
// in assembleFront: its clearing, the matrix's entries (Whole), then the children in order.
//
// Thread-safety: the passes of disjoint ranges may run at once. Ownership: references the layout, the matrix, the
// children and the front, which outlive it.
class FrontAssembler
{
public:
    // A child with delayed rows maps a column to several (indefinite mode): such a front is assembled in one range.
    FrontAssembler(const FrontLayout& layout, const SymmetricCsc* matrix, std::span<const UpdateMatrix> children,
                   FrontStorage front);
    FrontAssembler(const FrontAssembler&) = delete;
    FrontAssembler& operator=(const FrontAssembler&) = delete;
    ~FrontAssembler();

    // Whether a range may be less than the whole front (no child with delayed rows).
    [[nodiscard]] bool isCut() const noexcept;
    // The columns [first, end) of the front (positions), the phase's contributions. Throws std::invalid_argument for a
    // matrix entry outside the symbolic structure.
    void assembleColumns(std::size_t first, std::size_t end, AssemblyPhase phase) const;
    // The columns of child `child`'s update matrix whose positions lie in [first, end): [begin, stop).
    [[nodiscard]] std::pair<std::size_t, std::size_t> childColumnsIn(std::size_t child, std::size_t first,
                                                                     std::size_t end) const;

private:
    struct Maps;
    const FrontLayout& m_layout;
    const SymmetricCsc* m_matrix;
    std::span<const UpdateMatrix> m_children;
    FrontStorage m_front;
    std::unique_ptr<Maps> m_maps;
};

// The right-hand sides a child hands to its parent front with its update matrix: rows × columns,
// column-major (leading dimension rows), its rows R(child) ascending.
struct UpdateRightHandSides
{
    const double* values = nullptr;
    std::span<const std::int32_t> rows;
};

// The right-hand sides of a front (no delayed column: positive definite mode), cleared, then the caller's
// (addOwnTerms, called between the two by the caller of assembleFrontInPhases: here they are only cleared), then the
// children's added row by row in the order given: `columns` the rows of the front's columns (columnCount × k), `rows`
// those of its rows R(s) (rowCount × k).
void clearRightHandSides(DenseMatrixView columns, DenseMatrixView rows);
void addChildRightHandSides(const FrontLayout& layout, std::span<const UpdateRightHandSides> children,
                            DenseMatrixView columns, DenseMatrixView rows);

} // namespace oos::linalg
