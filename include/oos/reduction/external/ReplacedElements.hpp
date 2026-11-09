#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/core/ElementRegistry.hpp"

#include <span>
#include <vector>

namespace oos::reduction
{

// The elements external units replace and the element families that leave them out (ReplacedElementFamily): a
// registry with one ReplacedElementFamily per family of a base registry, on which an assembly::ElementSet gives every
// assembler, recovery and sensitivity of an analysis the model without the zones of its external units. The element
// batches stay those of the snapshot; only the families change.
//
// Thread-safety: immutable once built. Ownership: owns the sorted IDs and the wrapping families; borrows the base
// registry, which must outlive it. Not movable (the families borrow the IDs).
class ReplacedElements
{
public:
    // `replaced`: the IDs of the replaced elements, in any order (duplicates removed).
    ReplacedElements(const elements::ElementRegistry& base, std::vector<base::EntityId> replaced);
    ReplacedElements(const ReplacedElements&) = delete;
    ReplacedElements& operator=(const ReplacedElements&) = delete;
    ReplacedElements(ReplacedElements&&) = delete;
    ReplacedElements& operator=(ReplacedElements&&) = delete;
    ~ReplacedElements() = default;

    [[nodiscard]] const elements::ElementRegistry& registry() const noexcept { return m_registry; }
    [[nodiscard]] std::span<const base::EntityId> ids() const noexcept { return m_ids; }
    [[nodiscard]] bool isReplaced(base::EntityId elementId) const noexcept;

private:
    std::vector<base::EntityId> m_ids; // sorted; bounded by the zones of the external units
    elements::ElementRegistry m_registry;
};

} // namespace oos::reduction
