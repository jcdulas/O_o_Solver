#include "oos/reduction/absorption/SharedConnectionNodes.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>

namespace oos::reduction
{

SharedConnectionNodes::SharedConnectionNodes(const base::ContractReader& table)
    : m_table(&table),
      m_nodeIds(table.values<std::int32_t>("nodeId")),
      m_kinds(table.textLists("kinds")),
      m_cards(table.textLists("cards")),
      m_ids(table.lists<std::int32_t>("ids")),
      m_equations(table.lists<std::int32_t>("equations"))
{
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        if (row > 0 && m_nodeIds[r] <= m_nodeIds[r - 1])
            throw base::ContractError(base::concatenate(
                {"connectionSharing: expected the grid points by increasing nodeId (O_o_Reader's default sort), found ",
                 std::to_string(m_nodeIds[r]), " after ", std::to_string(m_nodeIds[r - 1]),
                 "; take the table without sort="}));
        const auto count = static_cast<std::int64_t>(m_ids[row].size());
        if (m_kinds.count(row) != count || m_cards.count(row) != count ||
            static_cast<std::int64_t>(m_equations[row].size()) != count)
            throw base::ContractError(base::concatenate(
                {"connectionSharing: expected kinds, cards, ids and equations of one length for grid point ",
                 std::to_string(m_nodeIds[r]), ", found ", std::to_string(m_kinds.count(row)), ", ",
                 std::to_string(m_cards.count(row)), ", ", std::to_string(count), " and ",
                 std::to_string(m_equations[row].size()),
                 "; the table does not follow the data contract: take the snapshot again with a O_o_Reader that "
                 "follows it"}));
    }
}

std::vector<SharedConnection> SharedConnectionNodes::connectionsOf(std::int64_t row) const
{
    std::vector<SharedConnection> connections; // bounded by the connections at one point
    const std::span<const std::int32_t> ids = m_ids[row];
    const std::span<const std::int32_t> equations = m_equations[row];
    for (std::size_t k = 0; k < ids.size(); ++k)
    {
        const auto item = static_cast<std::int64_t>(k);
        connections.push_back(SharedConnection{std::string(m_kinds.at(row, item)), std::string(m_cards.at(row, item)),
                                               ids[k], equations[k]});
    }
    return connections;
}

std::vector<SharedConnection> SharedConnectionNodes::connectionsAt(base::EntityId nodeId) const
{
    const auto found = std::lower_bound(m_nodeIds.begin(), m_nodeIds.end(), nodeId);
    if (found == m_nodeIds.end() || *found != nodeId)
        return {};
    return connectionsOf(found - m_nodeIds.begin());
}

} // namespace oos::reduction
