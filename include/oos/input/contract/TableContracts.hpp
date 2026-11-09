#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <string>
#include <string_view>

namespace oos::input
{

// The table contracts of the snapshot: for each table O_o_Solver reads, the columns it reads with their data contract
// types, as O_o_Reader's data contract (version 68) records them. Every name is a neutral name: the tables are
// O_o_Reader's queries (singlePointConstrTable), the columns the neutral names of the fields (setId, nodeId) or
// camelCase (isDependent), the card types the neutral card names (ShellQuad4).
//
// A contract lists the columns the solver reads, not every column the table has: a column the solver does not use may
// change without breaking it.

[[nodiscard]] base::TableContract contractOf(TableRole role);

// The error for a table the snapshot does not have: it says that the snapshot was taken without it and, for an
// optional table, what the table holds (prerequisiteOf).
[[nodiscard]] base::ContractError missingTableError(TableRole role);

// A reader of `table` checked against the contract of `role` (ContractError when it breaks it).
[[nodiscard]] base::ContractReader readTable(TableRole role, base::ImportedTable table);

// O_o_Reader names the own ID of every card "id", and the property ID of an element card "propId".
inline constexpr std::string_view ELEMENT_ID_COLUMN = "id";
inline constexpr std::string_view ELEMENT_PROPERTY_COLUMN = "propId";
inline constexpr std::string_view PROPERTY_ID_COLUMN = "id";

// The contract of the table of one element card type (cardTable(deck, "ShellQuad4")): its ID, and when the card has
// them, the property ID, the grid points (a fixed list for one card type, a list otherwise) and the row-index columns
// nodesRows, propIdRow and propType that O_o_Reader computes. The other fields are read by the element's own code
// through the batch's reader.
[[nodiscard]] base::TableContract elementBatchContract(std::string_view card);

// The contract of the table of a card type keyed by one integer ID column (property cards by PROPERTY_ID_COLUMN, method
// cards such as LanczosEigenMethod by "setId").
[[nodiscard]] base::TableContract cardTableContract(std::string_view card, std::string_view idColumn);

} // namespace oos::input
