#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <string>

namespace oos::base
{

// The optimisation cannot proceed: an inconsistent design model, an approximate problem that fails, a response that
// cannot be evaluated; the message names the design entity (DESVAR, DRESP1 ...) and its ID.
class OptimisationError : public SolverError
{
public:
    explicit OptimisationError(const std::string& message) : SolverError(message) {}
};

} // namespace oos::base
