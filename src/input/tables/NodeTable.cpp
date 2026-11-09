#include "oos/input/tables/NodeTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <string>
#include <utility>

namespace oos::input
{

namespace
{

constexpr std::size_t ROTATION_TERMS = 9; // one 3 x 3 matrix per grid point

} // namespace

NodeTable::NodeTable(base::ImportedTable nodes, std::optional<base::ImportedTable> scalarPoints,
                     BorrowedArray<double> cdRotations)
    : m_nodes(std::move(nodes), contractOf(TableRole::Nodes)), m_cdRotations(std::move(cdRotations))
{
    if (scalarPoints)
        m_scalarPoints.emplace(std::move(*scalarPoints), contractOf(TableRole::ScalarPoints));
    (void)ids();
    const auto expected = static_cast<std::size_t>(rowCount()) * ROTATION_TERMS;
    if (!m_cdRotations.empty() && m_cdRotations.size() != expected)
        throw base::ContractError(base::concatenate(
            {"the Node.displacementFrameId rotations of the grid points: expected ", std::to_string(expected),
             " values (a 3 x 3 matrix for each of the ", std::to_string(rowCount()),
             " grid points of the node table), found ", std::to_string(m_cdRotations.size()),
             "; take the rotations and the node table from the same model (displacementRotations(deck) of "
             "O_o_Reader)"}));
}

std::span<const std::int32_t> NodeTable::scalarPointIds() const
{
    if (!m_scalarPoints)
        return {};
    return m_scalarPoints->values<std::int32_t>("id");
}

} // namespace oos::input
