#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace oos::input
{

// The row of each ID of a table column (grid points, scalar points, materials, coordinate systems), found by binary
// search in a copy of the (ID, row) pairs sorted by ID. O_o_Reader gives the row-index columns nodesRows and propIdRow
// for most element cards, but not for the grid points of the springs, CBUSH, CONM1, CONM2 and the scalar masses
// (these come without a row-index column), nor for the material of a property: the elements find those rows here.
//
// The pairs grow with the model, so they live in a BudgetedBuffer of the budget given at construction (8 bytes per
// ID). Building is O(n log n), a lookup O(log n).
//
// Thread-safety: immutable once built; concurrent lookups are safe. Ownership: owns its pairs; the budget must outlive
// the index. Movable, not copyable.
class IdIndex
{
public:
    IdIndex() = default;
    // Throws ContractError when an ID appears twice (O_o_Reader guarantees unique IDs per table), MemoryBudgetError
    // when the pairs do not fit the budget. `label` names the index in both messages ("grid point IDs").
    IdIndex(std::span<const std::int32_t> ids, base::MemoryBudget& budget, const std::string& label);

    [[nodiscard]] std::size_t size() const noexcept { return m_pairs.size(); }
    // The row of `id` in the column the index was built from, or nothing when the column does not hold it.
    [[nodiscard]] std::optional<std::int32_t> find(std::int32_t id) const noexcept;

private:
    // The pair type is trivially copyable, as BudgetedBuffer requires (std::pair is not).
    struct IdRow
    {
        std::int32_t id;
        std::int32_t row;
    };

    base::BudgetedBuffer<IdRow> m_pairs;
};

} // namespace oos::input
