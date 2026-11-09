#pragma once

#include "oos/base/errors/SolverError.hpp"

namespace oos::base
{

// The operation was cancelled (Ctrl-C, a progress callback that asked to stop). It is raised at a point
// where nothing is left half done: a file being written is still a temporary file, and is removed.
class OperationCancelledError : public SolverError
{
public:
    OperationCancelledError() : SolverError("the operation was cancelled (by an interruption or a progress callback)")
    {
    }
};

} // namespace oos::base
