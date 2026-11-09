#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/sparse/ContributionBlock.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::assembly
{

// The element stiffness matrices computed by the connectivity pass (ElementConnectivity), kept for the assembly of Kgg
// (StiffnessAssembler) so that each element matrix is computed once. The matrices are kept on
// the node DOFs of linalg as the pieces of the connectivity pass made them (one ContributionBlock per piece of
// consecutive elements, moved in without a copy), in the order of the elements of the batches that have the stiffness
// facet (the order of the sums of the assembly), each with the index of its element in that list.
//
// The store holds at most the bytes given at its construction (a fraction of the memory budget chosen by the caller):
// a piece that would pass that limit, or that the budget refuses, drops everything kept, and the assembly computes the
// matrices again (isKept() is then false).
//
// Thread-safety: appendPiece() and release() are not thread-safe; the const methods may run concurrently once it is
// filled. Ownership: owns its pieces and their reservations; the budget must outlive it.
class ElementMatrixStore
{
public:
    ElementMatrixStore() = default;
    ElementMatrixStore(base::MemoryBudget& budget, std::uint64_t byteLimit);

    // Appends the matrices of one piece: elements[i] is the element (in the stiffness list) of matrix i of `block`,
    // non-decreasing within the piece and from piece to piece. Returns whether the store is still kept. Throws
    // std::invalid_argument when the counts differ or the elements go back.
    bool appendPiece(std::vector<std::int64_t> elements, linalg::ContributionBlock block);
    // Gives the memory back; the store is no longer kept.
    void release() noexcept;

    [[nodiscard]] bool isKept() const noexcept { return m_isKept; }
    [[nodiscard]] std::uint64_t keptBytes() const noexcept { return m_keptBytes; }

    // Appends to `contributions` the matrices of the elements [begin, end) of the stiffness list: the
    // ContributionSource of linalg::SparseAssembler. Precondition: isKept().
    void writeContributions(std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions) const;

private:
    struct Piece
    {
        std::vector<std::int64_t> elements; // bounded by a piece of the connectivity pass
        linalg::ContributionBlock block;
        base::MemoryReservation reservation;
    };

    base::MemoryBudget* m_budget = nullptr;
    std::uint64_t m_byteLimit = 0;
    std::uint64_t m_keptBytes = 0;
    bool m_isKept = false;
    std::vector<Piece> m_pieces; // bounded by the elements / ElementConnectivity::ELEMENTS_PER_PIECE
};

} // namespace oos::assembly
