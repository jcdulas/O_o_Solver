#pragma once

#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace oos::design
{

// Where an element of the snapshot is: its card, its batch (index in ModelSnapshot::elementBatches()), its row and its
// property (ID and card; 0 and "" for an element without a property).
struct ElementEntry
{
    std::int32_t elementId = 0;
    std::string card;
    std::size_t batch = 0;
    std::int64_t row = 0;
    std::int32_t propId = 0;
    std::string propertyCard;
};

// The elements of a snapshot by ID, and the elements of the properties a DesignResp1 names (DRESP1: ownerKind =
// "Elements", the ATTi are elements; PTYPE a property card, the ATTi are property IDs of that card, since property IDs
// are unique within a property card only).
//
// The property card of an element is the propType O_o_Reader gives with propIdRow; without it, the property table of
// the snapshot that holds the PID among those the element's card can take (its ElementFamily::propertyCards()).
//
// Thread-safety: immutable once built. Ownership: owns its entries (bounded by the elements of the model: one entry
// each, the index the responses need; not a buffer of the analysis).
class ElementCatalog
{
public:
    explicit ElementCatalog(const input::ModelSnapshot& snapshot);

    [[nodiscard]] const input::ModelSnapshot& snapshot() const noexcept { return *m_snapshot; }

    [[nodiscard]] const ElementEntry* find(std::int32_t elementId) const noexcept;
    // The elements of property `propId` of card `propertyCard`, in increasing element ID (empty when none).
    [[nodiscard]] std::vector<const ElementEntry*> elementsOfProperty(std::string_view propertyCard,
                                                                      std::int32_t propId) const;
    // Whether the snapshot has property `propId` in the table of `propertyCard`.
    [[nodiscard]] bool hasProperty(std::string_view propertyCard, std::int32_t propId) const;
    [[nodiscard]] const std::vector<ElementEntry>& entries() const noexcept { return m_entries; }

private:
    const input::ModelSnapshot* m_snapshot;
    std::vector<ElementEntry> m_entries; // in increasing element ID
    std::unordered_map<std::int32_t, std::size_t> m_byId;
    std::map<std::pair<std::string, std::int32_t>, std::vector<std::size_t>> m_byProperty;
};

} // namespace oos::design
