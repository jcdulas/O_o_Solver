#pragma once

#include "oos/elements/core/ElementDof.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::elements
{

// The element matrices (stiffness or mass) of a range of elements of one batch, as a facet gives them: per element its
// DOFs and its dense symmetric matrix over those DOFs, row-major and complete (both triangles), in the displacement
// systems of its grid points. The assembler scatters them; the tests read them.
//
// Storage is contiguous for the whole range (structure of arrays: offsets into one DOF array and one value array), so
// that a facet fills it without an allocation per element once the block has grown. A block holds one range at a
// time: its size is bounded by the range length the caller chooses times the largest element (std::vector is the
// right container).
//
// Thread-safety: none for writes; one block per thread. Ownership: owns its values.
class ElementMatrixBlock
{
public:
    // Forgets the elements held, keeping the memory.
    void clear() noexcept;

    // Appends the matrix of the element of row `elementRow` of the batch. Precondition: values.size() is
    // dofs.size()². Throws std::invalid_argument otherwise.
    void append(std::int64_t elementRow, std::span<const ElementDof> dofs, std::span<const double> values);

    [[nodiscard]] std::size_t size() const noexcept { return m_elementRows.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_elementRows.empty(); }
    [[nodiscard]] std::int64_t elementRow(std::size_t i) const { return m_elementRows[i]; }
    [[nodiscard]] std::span<const ElementDof> dofs(std::size_t i) const;
    // The n × n matrix of element i, row-major: term (r, c) at r·n + c, n = dofs(i).size().
    [[nodiscard]] std::span<const double> matrix(std::size_t i) const;

private:
    std::vector<std::int64_t> m_elementRows;
    std::vector<std::size_t> m_dofOffsets{0};
    std::vector<ElementDof> m_dofs;
    std::vector<std::size_t> m_valueOffsets{0};
    std::vector<double> m_values;
};

} // namespace oos::elements
