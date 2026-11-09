#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <string>

namespace oos::base
{

// A file cannot be created, written, renamed or read (an output file, a temporary file of the cache); the message names
// the path, the operation and the reason the system gave.
class FileError : public SolverError
{
public:
    explicit FileError(const std::string& message) : SolverError(message) {}
};

} // namespace oos::base
