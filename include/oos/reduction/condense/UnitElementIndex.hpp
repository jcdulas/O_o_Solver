#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstdint>
#include <optional>
#include <span>

namespace oos::reduction
{

// Where an element of the model is: its batch (an index in ElementSet::batches()) and its row in it.
struct ElementPlace
{
    base::EntityId elementId = 0;
    std::uint32_t batchIndex = 0;
    std::int64_t row = 0;
};

// The elements of a model that have a matrix (the stiffness or the mass facet), by ID, so that the elements of a unit
// are found without a pass over the model per unit (a model of 10 000 units). Built once per model.
//
// Thread-safety: immutable once built. Ownership: owns its index (budgeted: one entry per element); borrows nothing.
class UnitElementIndex
{
public:
    // Throws std::invalid_argument when two batches hold the same element ID (element IDs are unique in a deck), and
    // MemoryBudgetError.
    UnitElementIndex(const assembly::ElementSet& elements, base::MemoryBudget& budget);

    // The place of an element, or nothing for an ID no batch with a matrix holds (a rigid element is a relation).
    [[nodiscard]] std::optional<ElementPlace> find(base::EntityId elementId) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return m_places.size(); }

private:
    base::BudgetedBuffer<ElementPlace> m_places; // sorted by ID
};

} // namespace oos::reduction
