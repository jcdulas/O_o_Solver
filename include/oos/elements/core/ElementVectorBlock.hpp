#pragma once

#include "oos/elements/core/ElementDof.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::elements
{

// The equivalent nodal loads of a list of distributed loads on the elements of one batch, as the loads facet gives
// them: per load its element's batch row, its DOFs and the load on each, in the displacement systems of the grid
// points. The load vector builder scatters them; the tests read them. One entry per load (an element under two
// pressures of one set appears twice: the builder adds them).
//
// Bounded by the number of loads the caller passes times the DOFs of the largest element (std::vector).
//
// Thread-safety: none for writes; one block per thread. Ownership: owns its values.
class ElementVectorBlock
{
public:
    // Forgets the entries held, keeping the memory.
    void clear() noexcept;

    // Appends the load of the element of batch row `elementRow`. Throws std::invalid_argument when the number of
    // values is not the number of DOFs.
    void append(std::int64_t elementRow, std::span<const ElementDof> dofs, std::span<const double> values);

    [[nodiscard]] std::size_t size() const noexcept { return m_elementRows.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_elementRows.empty(); }
    [[nodiscard]] std::int64_t elementRow(std::size_t i) const { return m_elementRows[i]; }
    [[nodiscard]] std::span<const ElementDof> dofs(std::size_t i) const;
    [[nodiscard]] std::span<const double> values(std::size_t i) const;

private:
    std::vector<std::int64_t> m_elementRows;
    std::vector<std::size_t> m_offsets{0};
    std::vector<ElementDof> m_dofs;
    std::vector<double> m_values;
};

} // namespace oos::elements
