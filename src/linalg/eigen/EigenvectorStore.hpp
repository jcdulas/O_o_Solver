#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/ScratchSpool.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace oos::linalg
{

class EigenvectorStore;

// The locked set as the projections read it (MassOrthogonalizer): its first columns resident in memory, the others on
// a scratch spool by pieces of rows, read back one piece at a time.
class LockedVectors
{
public:
    LockedVectors(ConstDenseMatrixView resident, const EigenvectorStore* store) noexcept
        : m_resident(resident), m_store(store)
    {
    }
    // A locked set held in memory whole.
    explicit LockedVectors(ConstDenseMatrixView resident) noexcept : m_resident(resident) {}

    [[nodiscard]] const ConstDenseMatrixView& resident() const noexcept { return m_resident; }
    [[nodiscard]] std::size_t spooledCount() const noexcept;
    [[nodiscard]] std::size_t columns() const noexcept { return m_resident.columns + spooledCount(); }
    // Rows [first, end) of the spooled columns (a piece of the store's pieceRows: first a multiple of it) into `into`,
    // column-major with the leading dimension end − first.
    void readPiece(std::size_t first, std::size_t end, double* into) const;

private:
    ConstDenseMatrixView m_resident;
    const EigenvectorStore* m_store = nullptr;
};

// The eigenpairs converged so far by the block Lanczos solver, over all its shifts, in the order they converged: the
// vectors (M-orthonormal columns) are the locked set every later Lanczos basis is kept M-orthogonal to, so that a mode
// is never found twice (Grimes, Lewis & Simon 1994 §3.3: orthogonality against the eigenvectors of earlier shifts).
//
// Partly out of core: at most residentColumns vectors are held in memory; the later ones
// go to a scratch spool by slabs (the columns of one commit), each slab stored by pieces of pieceRows rows (one record
// a piece: its rows of the slab's columns), the pieces the projections sum by: a piece of every spooled column
// is read back in one record a slab, with the values it was written with.
//
// Invariant: columns [0, count()) hold the vectors, the resident ones first; eigenvalue k and residual bound k belong
// to column k. The resident matrix grows by GROWTH_FACTOR when full (up to residentColumns), the new one reserved in
// the budget before the copy.
//
// Thread-safety: as a std::vector; readPiece() is safe concurrently. Ownership: owns its vectors and spool (budgeted);
// the budget must outlive it.
class EigenvectorStore
{
public:
    static constexpr double GROWTH_FACTOR = 1.5;

    // Throws MemoryBudgetError when the initial capacity does not fit, FileError when the spool cannot be made.
    EigenvectorStore(base::MemoryBudget& budget, std::size_t dimension, std::size_t initialCapacity,
                     std::size_t residentColumns = std::numeric_limits<std::size_t>::max(),
                     const ScratchSpool::Options& spool = {}, std::size_t pieceRows = 1024);

    [[nodiscard]] std::size_t count() const noexcept { return m_eigenvalues.size(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return m_dimension; }
    [[nodiscard]] std::size_t residentCount() const noexcept { return m_residentCount; }
    [[nodiscard]] std::size_t pieceRows() const noexcept { return m_pieceRows; }
    // The locked set for the projections.
    [[nodiscard]] LockedVectors locked() const noexcept;
    // Column k into `out` (dimension() values). Throws FileError.
    void copyColumn(std::size_t k, double* out) const;
    [[nodiscard]] const std::vector<double>& eigenvalues() const noexcept { return m_eigenvalues; }
    [[nodiscard]] const std::vector<double>& residualBounds() const noexcept { return m_residualBounds; }
    // The view of `columnCount` new columns after the last one, growing the matrix when needed (MemoryBudgetError when
    // the growth does not fit); the caller writes them, then commits them with their eigenvalues.
    [[nodiscard]] DenseMatrixView prepareColumns(std::size_t columnCount);
    // Appends the eigenpairs of the first values.size() columns prepared. Throws FileError.
    void commit(const std::vector<double>& values, const std::vector<double>& residualBounds);
    // The vectors on the spool and their bytes there.
    [[nodiscard]] std::size_t spooledCount() const noexcept { return count() - m_residentCount; }
    [[nodiscard]] std::uint64_t spooledBytes() const noexcept { return m_spool ? m_spool->heldBytes() : 0; }

private:
    friend class LockedVectors;
    // The columns of one commit beyond the resident ones: per piece of rows, its record.
    struct Slab
    {
        std::size_t firstColumn = 0; // among the spooled columns
        std::size_t columnCount = 0;
        std::vector<SpoolRecord> pieces; // bounded by the pieces of the dimension
    };

    base::MemoryBudget& m_budget;
    std::size_t m_dimension = 0;
    std::size_t m_residentLimit = 0;
    std::size_t m_pieceRows = 1024;
    DenseMatrix m_vectors;
    std::size_t m_residentCount = 0;
    DenseMatrix m_staging; // the prepared columns of a commit that goes (partly) to the spool
    bool m_isStaged = false;
    ScratchSpool::Options m_spoolOptions;
    std::unique_ptr<ScratchSpool> m_spool;
    std::vector<Slab> m_slabs;            // bounded by the commits
    std::vector<double> m_eigenvalues;    // bounded by the modes found
    std::vector<double> m_residualBounds; // bounded by the modes found
};

} // namespace oos::linalg
