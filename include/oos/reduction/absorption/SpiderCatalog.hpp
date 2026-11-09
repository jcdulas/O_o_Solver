#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/input/tables/FastenerPiercingTable.hpp"
#include "oos/reduction/absorption/Spider.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace oos::reduction
{

// Every spider of a model, from O_o_Reader's tables:
// - each row of `spiders` (O_o_Reader spiders(contact="merge")): an RigidBodyStar or RigidInterpolation, its
//   reference, legs, topology, eligibility and revision as given;
// - each side of each Fastener and Weld of the piercing table (with its is_eligible and blocking_connections
// columns): a
//   spider whose legs are the grid points of the element the side pierces (the shell element of a PROP, ELEM, ELPAT,
//   PARTPAT or ELEMID side, the face of grid points of a GRIDID side), eligible when every row of the side is, blocked
//   by the union of their blocking connections; its reference the GA or GB of the card (0 when the point is
//   generated). A side the piercing table leaves out (an ALIGN Weld, a point projected on nothing) is no spider.
//   Its legs are on one component by construction when the table's component_id is set; else isSpider is decided by the
//   units (SpiderAbsorption).
// Sorted by key (elementId, then side A before B).
//
// Thread-safety: immutable once built. Ownership: owns its spiders (their number is the number of RigidBodyStar,
// RigidInterpolation and fastener sides of the model, each with a list of legs: a std::vector read once per plan, the
// size of the producer's tables, which the snapshot holds anyway).
class SpiderCatalog
{
public:
    // Either table may be null (none in the snapshot): no spider of that kind. Throws base::ContractError when a table
    // breaks its contract (a list of leg components or weights of another length than the legs, an elementId twice).
    SpiderCatalog(const base::ContractReader* spiders, const input::FastenerPiercingTable* piercings);

    [[nodiscard]] std::span<const Spider> spiders() const noexcept { return m_spiders; }
    // The index of a spider in spiders(), or nothing.
    [[nodiscard]] std::optional<std::size_t> indexOf(const SpiderKey& key) const noexcept;
    // Whether the catalog had a spiders table (false: no RigidBodyStar or RigidInterpolation is ever absorbed, and the
    // plan says why).
    [[nodiscard]] bool hasRigidSpiders() const noexcept { return m_hasRigidSpiders; }

private:
    std::vector<Spider> m_spiders;
    bool m_hasRigidSpiders = false;
};

} // namespace oos::reduction
