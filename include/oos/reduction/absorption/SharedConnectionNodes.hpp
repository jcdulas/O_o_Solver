#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/ids/EntityId.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::reduction
{

// One connection that touches a grid point: kind ("rigid", "rigidInterpolation", "mpc", "fastener", "connector",
// "contact"), card, ID (the element ID, the set ID of an MPC equation) and, for an MPC, the number of the equation in
// its set.
struct SharedConnection
{
    std::string kind;
    std::string card;
    base::EntityId id = 0;
    std::int32_t equation = 0;
};

// Per grid point, the connections that touch it (from O_o_Reader's connectionSharing query), known before running so
// that the user sees which grid points are touched by several relations or by a Fastener, and why the boundary is
// where it is. Read in place from O_o_Reader's connectionSharing, whose rows are by increasing nodeId; the absorption
// report names with it the connections that make a spider ineligible.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: borrows the table, which must outlive it.
class SharedConnectionNodes
{
public:
    // Throws base::ContractError when the rows are not by strictly increasing nodeId or the lists of a row disagree in
    // length.
    explicit SharedConnectionNodes(const base::ContractReader& table);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_table->rowCount(); }
    [[nodiscard]] base::EntityId nodeIdOf(std::int64_t row) const noexcept
    {
        return m_nodeIds[static_cast<std::size_t>(row)];
    }
    [[nodiscard]] std::vector<SharedConnection> connectionsOf(std::int64_t row) const;
    // The connections at a grid point (none when no connection touches it).
    [[nodiscard]] std::vector<SharedConnection> connectionsAt(base::EntityId nodeId) const;

private:
    const base::ContractReader* m_table;
    std::span<const std::int32_t> m_nodeIds;
    base::TextListView m_kinds;
    base::TextListView m_cards;
    base::ListView<std::int32_t> m_ids;
    base::ListView<std::int32_t> m_equations;
};

} // namespace oos::reduction
