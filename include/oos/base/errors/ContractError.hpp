#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <string>

namespace oos::base
{

// A table or column of O_o_Reader does not have the name or type the data contract promises, or a
// prerequisite table is missing; the message names the table, the column, the type expected and the type found.
class ContractError : public SolverError
{
public:
    explicit ContractError(const std::string& message) : SolverError(message) {}
};

} // namespace oos::base
