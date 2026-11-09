#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <string>

namespace oos::base
{

// The condensation cache cannot be read, written or trusted (a bad checksum, an unknown format version, a lock held by
// another process); the message names the file and what to do (clear the cache, free disk space).
class CacheError : public SolverError
{
public:
    explicit CacheError(const std::string& message) : SolverError(message) {}
};

} // namespace oos::base
