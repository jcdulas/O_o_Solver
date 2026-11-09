#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/design/optimiser/OptimisationParameters.hpp"

namespace oos::design
{

// The DesignOptimParams of a deck as O_o_Solver reads it (the table O_o_Solver proposes to O_o_Reader, which does not
// deliver it yet):
//
// designParametersTable (proposed), one row per PARAMi/VALi pair of the DesignOptimParams entry: name text (PARAMi, as
//   written) · integer_value int64 (VALi when the field is an integer, else null) · real_value float64 (VALi when it
//   is a real, else null) · text_value text (VALi when it is a character value: OPTCOD; else null). Exactly one value
//   per row.
[[nodiscard]] base::TableContract designParametersContract();

// The parameters of the table (the defaults for the others). ContractError when a row breaks the contract;
// OptimisationError for a parameter refused by OptimisationParameters::set.
[[nodiscard]] OptimisationParameters readDesignParameters(const base::ContractReader& table);

} // namespace oos::design
