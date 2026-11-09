#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <string>

namespace oos::base
{

// A card, field value or option that the solver does not support; raised only after every such item has been collected
// in a SupportReport, so that the user sees all of them at once.
class UnsupportedCardError : public SolverError
{
public:
    explicit UnsupportedCardError(const std::string& message) : SolverError(message) {}
};

} // namespace oos::base
