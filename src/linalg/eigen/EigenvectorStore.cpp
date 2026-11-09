#include "EigenvectorStore.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::linalg
{
namespace
{

constexpr const char* STORE_LABEL = "Lanczos: eigenvectors found";

} // namespace

std::size_t LockedVectors::spooledCount() const noexcept
{
    return m_store == nullptr ? 0 : m_store->spooledCount();
}

void LockedVectors::readPiece(std::size_t first, std::size_t end, double* into) const
{
    if (m_store == nullptr || m_store->spooledCount() == 0)
        return;
    const std::size_t pieceRows = m_store->m_pieceRows;
    if (first % pieceRows != 0 || end - first > pieceRows)
        throw std::logic_error(base::concatenate(
            {"LockedVectors::readPiece: rows [", std::to_string(first), ", ", std::to_string(end),
             ") are not one piece of ", std::to_string(pieceRows), " rows (a bug of O_o_Solver: report it)"}));
    const std::size_t piece = first / pieceRows;
    const std::size_t rows = end - first;
    for (const EigenvectorStore::Slab& slab : m_store->m_slabs)
        m_store->m_spool->read(slab.pieces[piece], into + slab.firstColumn * rows);
}

EigenvectorStore::EigenvectorStore(base::MemoryBudget& budget, std::size_t dimension, std::size_t initialCapacity,
                                   std::size_t residentColumns, const ScratchSpool::Options& spool,
                                   std::size_t pieceRows)
    : m_budget(budget),
      m_dimension(dimension),
      m_residentLimit(residentColumns),
      m_pieceRows(std::max<std::size_t>(pieceRows, 1)),
      m_vectors(budget, dimension, std::max<std::size_t>(std::min(initialCapacity, residentColumns), 1), STORE_LABEL),
      m_spoolOptions(spool)
{
}

LockedVectors EigenvectorStore::locked() const noexcept
{
    return {ConstDenseMatrixView{m_vectors.data(), m_dimension, m_residentCount, m_vectors.leadingDimension()}, this};
}

void EigenvectorStore::copyColumn(std::size_t k, double* out) const
{
    if (k < m_residentCount)
    {
        std::copy_n(m_vectors.data() + k * m_vectors.leadingDimension(), m_dimension, out);
        return;
    }
    const std::size_t spooled = k - m_residentCount;
    for (const Slab& slab : m_slabs)
    {
        if (spooled < slab.firstColumn || spooled >= slab.firstColumn + slab.columnCount)
            continue;
        std::vector<double> piece; // bounded by a piece of a slab
        const std::size_t column = spooled - slab.firstColumn;
        for (std::size_t p = 0; p < slab.pieces.size(); ++p)
        {
            const std::size_t first = p * m_pieceRows;
            const std::size_t rows = std::min(m_pieceRows, m_dimension - first);
            piece.resize(rows * slab.columnCount);
            m_spool->read(slab.pieces[p], piece.data());
            std::copy_n(piece.data() + column * rows, rows, out + first);
        }
        return;
    }
    throw std::out_of_range(base::concatenate(
        {"EigenvectorStore::copyColumn: no column ", std::to_string(k), " (a bug of O_o_Solver: report it)"}));
}

DenseMatrixView EigenvectorStore::prepareColumns(std::size_t columnCount)
{
    const std::size_t needed = count() + columnCount;
    if (needed > m_residentLimit)
    {
        // Beyond the resident columns: the batch staged, then committed to memory up to the limit and the rest to
        // the spool.
        m_staging = DenseMatrix(m_budget, m_dimension, columnCount, "Lanczos: eigenvectors staged for the spool");
        m_isStaged = true;
        return m_staging.view();
    }
    if (needed > m_vectors.columns())
    {
        const auto grown =
            static_cast<std::size_t>(std::ceil(GROWTH_FACTOR * static_cast<double>(m_vectors.columns())));
        DenseMatrix larger(m_budget, m_dimension, std::min(std::max(needed, grown), m_residentLimit), STORE_LABEL);
        for (std::size_t j = 0; j < count(); ++j)
            std::copy_n(&m_vectors(0, j), m_dimension, &larger(0, j));
        m_vectors = std::move(larger);
    }
    m_isStaged = false;
    return m_vectors.view().block(0, count(), m_dimension, columnCount);
}

void EigenvectorStore::commit(const std::vector<double>& values, const std::vector<double>& residualBounds)
{
    if (values.size() != residualBounds.size())
        throw std::logic_error("EigenvectorStore::commit: as many residual bounds as eigenvalues expected");
    if (!m_isStaged)
    {
        if (count() + values.size() > m_vectors.columns())
            throw std::logic_error("EigenvectorStore::commit: more eigenpairs than columns prepared");
        m_residentCount += values.size();
    }
    else
    {
        if (values.size() > m_staging.columns())
            throw std::logic_error("EigenvectorStore::commit: more eigenpairs than columns prepared");
        // The columns spooled before this commit (count() not yet grown, m_residentCount not yet either).
        const std::size_t spooledBefore = count() - m_residentCount;
        // The resident columns first, up to the limit.
        const std::size_t room = m_residentLimit > m_residentCount ? m_residentLimit - m_residentCount : 0;
        const std::size_t resident = std::min(room, values.size());
        if (resident > 0)
        {
            if (m_residentCount + resident > m_vectors.columns())
            {
                DenseMatrix larger(m_budget, m_dimension, m_residentCount + resident, STORE_LABEL);
                for (std::size_t j = 0; j < m_residentCount; ++j)
                    std::copy_n(&m_vectors(0, j), m_dimension, &larger(0, j));
                m_vectors = std::move(larger);
            }
            for (std::size_t j = 0; j < resident; ++j)
                std::copy_n(&m_staging(0, j), m_dimension, &m_vectors(0, m_residentCount + j));
            m_residentCount += resident;
        }
        const std::size_t spooled = values.size() - resident;
        if (spooled > 0)
        {
            if (!m_spool)
                m_spool = std::make_unique<ScratchSpool>(m_spoolOptions, m_budget);
            Slab slab;
            slab.firstColumn = spooledBefore;
            slab.columnCount = spooled;
            const std::size_t pieces = (m_dimension + m_pieceRows - 1) / m_pieceRows;
            slab.pieces.resize(pieces);
            std::vector<double> piece; // bounded by a piece of the slab
            ScratchSpool::Writer writer(*m_spool);
            for (std::size_t p = 0; p < pieces; ++p)
            {
                const std::size_t first = p * m_pieceRows;
                const std::size_t rows = std::min(m_pieceRows, m_dimension - first);
                piece.resize(rows * spooled);
                for (std::size_t j = 0; j < spooled; ++j)
                    std::copy_n(&m_staging(first, resident + j), rows, piece.data() + j * rows);
                writer.append(piece.data(), piece.size() * sizeof(double), slab.pieces[p]);
            }
            writer.flush();
            m_slabs.push_back(std::move(slab));
        }
        m_staging = DenseMatrix();
        m_isStaged = false;
    }
    m_eigenvalues.insert(m_eigenvalues.end(), values.begin(), values.end());
    m_residualBounds.insert(m_residualBounds.end(), residualBounds.begin(), residualBounds.end());
}

} // namespace oos::linalg
