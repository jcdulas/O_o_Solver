#pragma once

#include <stdexcept>
#include <string>

namespace oos::base
{

// The root of every error the solver raises at its boundary. A problem found in the model during an
// analysis is a Diagnostic; an exception is for a misuse of the API, an exhausted budget, an I/O failure, a broken
// contract with O_o_Reader, or a model that cannot be solved at all. Every message says what was expected, what was
// found, where, and what to do; ErrorMessage.hpp builds such messages.
//
// The subclasses live in their own headers of base/errors/. An invalid argument of a public function (a negative
// size, an unknown option name) is std::invalid_argument, which the caller reports as an invalid argument.
class SolverError : public std::runtime_error
{
public:
    explicit SolverError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace oos::base
