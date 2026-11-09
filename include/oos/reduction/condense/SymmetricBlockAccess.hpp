#pragma once

#include "oos/linalg/sparse/SparsePattern.hpp"

#include <cstdint>
#include <span>

namespace oos::reduction
{

// The terms of a symmetric matrix stored on a block pattern (linalg::BlockSparseMatrix: the lower triangle of the
// node blocks, a diagonal block complete) read and written as those of the full matrix, by node-major DOF, as
// assembly::MsetElimination reads Kgg: term (i, j) of two different nodes is the one value stored in the block of the
// larger node; the diagonal block of a node holds both of its triangles. Used by the eliminations and the changes of
// basis made in place on a unit's matrices (UnitSystem) and on the interface problem's (InterfaceProblem).
//
// Thread-safety: as the values it views. Ownership: views the pattern, the DOF offsets and the values, which must
// outlive it.
class SymmetricBlockAccess
{
public:
    // firstDof: the node-major DOF offsets of the pattern's nodes (nodeCount + 1 entries).
    SymmetricBlockAccess(const linalg::SparsePattern& pattern, std::span<const std::int64_t> firstDof,
                         std::span<double> values) noexcept
        : m_pattern(&pattern), m_firstDof(firstDof), m_values(values)
    {
    }

    // Term (i, j), 0 when the nodes of i and j are not coupled by the pattern.
    [[nodiscard]] double read(std::int64_t i, std::int64_t j) const;
    // Adds to term (i, j) of the full matrix; the caller adds (j, i) too, so a term of the upper triangle of two nodes
    // is skipped (its partner writes the stored value). Throws std::logic_error when the nodes are not coupled.
    void add(std::int64_t i, std::int64_t j, double value) const;
    // Sets terms (i, j) and (j, i). Throws std::logic_error when the nodes are not coupled.
    void setSymmetric(std::int64_t i, std::int64_t j, double value) const;

    // The node of a node-major DOF.
    [[nodiscard]] std::int32_t nodeOf(std::int64_t dof) const noexcept;

private:
    // The position of term (i, j) in the values, or -1 when the nodes are not coupled; isStored: whether (i, j) is the
    // stored triangle (the larger node's block, or a diagonal block).
    [[nodiscard]] std::int64_t positionOf(std::int64_t i, std::int64_t j, bool& isStored) const;
    static void requireCoupled(std::int64_t position, std::int64_t i, std::int64_t j);

    const linalg::SparsePattern* m_pattern;
    std::span<const std::int64_t> m_firstDof;
    std::span<double> m_values;
};

} // namespace oos::reduction
