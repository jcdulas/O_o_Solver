#include "oos/assembly/assemble/ElementMatrixStore.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::assembly
{

ElementMatrixStore::ElementMatrixStore(base::MemoryBudget& budget, std::uint64_t byteLimit)
    : m_budget(&budget), m_byteLimit(byteLimit), m_isKept(true)
{
}

bool ElementMatrixStore::appendPiece(std::vector<std::int64_t> elements, linalg::ContributionBlock block)
{
    if (elements.size() != block.size())
        throw std::invalid_argument(base::concatenate(
            {"ElementMatrixStore::appendPiece: expected one element per "
             "matrix, found ",
             std::to_string(elements.size()), " elements for ", std::to_string(block.size()), " matrices"}));
    if (!std::is_sorted(elements.begin(), elements.end()) ||
        (!elements.empty() && !m_pieces.empty() && !m_pieces.back().elements.empty() &&
         elements.front() < m_pieces.back().elements.back()))
        throw std::invalid_argument(
            "ElementMatrixStore::appendPiece: expected the elements in order, from piece to "
            "piece; append the pieces in the order of the element list");
    if (!m_isKept)
        return false;
    if (elements.empty())
        return true;
    // The bytes the piece holds: its element indices, and the DOFs, values and offsets of its matrices.
    std::uint64_t bytes = elements.size() * (sizeof(std::int64_t) + 2 * sizeof(std::size_t));
    for (std::size_t i = 0; i < block.size(); ++i)
        bytes += block.dofs(i).size() * sizeof(linalg::NodeDof) + block.matrix(i).size() * sizeof(double);
    if (m_keptBytes + bytes > m_byteLimit)
    {
        release();
        return false;
    }
    base::MemoryReservation reservation;
    try
    {
        reservation = m_budget->reserve(bytes, "element stiffness matrices kept for the assembly");
    }
    catch (const base::MemoryBudgetError&)
    {
        // The store is an optimisation: when the budget cannot hold it, the assembly computes the matrices again.
        release();
        return false;
    }
    m_keptBytes += bytes;
    m_pieces.push_back(Piece{std::move(elements), std::move(block), std::move(reservation)});
    return true;
}

void ElementMatrixStore::release() noexcept
{
    m_isKept = false;
    std::vector<Piece>().swap(m_pieces);
    m_keptBytes = 0;
}

void ElementMatrixStore::writeContributions(std::size_t begin, std::size_t end,
                                            linalg::ContributionBlock& contributions) const
{
    const auto first = static_cast<std::int64_t>(begin);
    const auto last = static_cast<std::int64_t>(end);
    // The first piece whose last element is at least `begin`.
    auto piece = std::partition_point(m_pieces.begin(), m_pieces.end(),
                                      [first](const Piece& candidate) { return candidate.elements.back() < first; });
    for (; piece != m_pieces.end() && piece->elements.front() < last; ++piece)
    {
        const auto from = static_cast<std::size_t>(
            std::lower_bound(piece->elements.begin(), piece->elements.end(), first) - piece->elements.begin());
        for (std::size_t i = from; i < piece->elements.size() && piece->elements[i] < last; ++i)
            contributions.append(piece->block.dofs(i), piece->block.matrix(i));
    }
}

} // namespace oos::assembly
