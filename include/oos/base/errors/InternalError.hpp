#pragma once

#include <stdexcept>
#include <string>

namespace oos::base
{

// An internal invariant does not hold: a bug of O_o_Solver, never a problem of the model or a misuse of the API. Not a
// SolverError, so that a caller catching SolverError does not take a bug for a model problem. Raised by
// OOS_ASSERT.
class InternalError : public std::logic_error
{
public:
    explicit InternalError(const std::string& message) : std::logic_error(message) {}
};

// Throws InternalError with the condition, the file and the line.
[[noreturn]] void failAssertion(const char* condition, const char* file, int line);

} // namespace oos::base

// Checks an invariant of the code. It stays in the optimised build of the ordinary suite (OOS_ASSERTIONS, see
// CMakeLists.txt) and compiles to nothing otherwise; the condition must have no side effect.
#if defined(OOS_ASSERTIONS)
#define OOS_ASSERT(condition)                                            \
    do                                                                          \
    {                                                                           \
        if (!(condition))                                                       \
            ::oos::base::failAssertion(#condition, __FILE__, __LINE__); \
    } while (false)
#else
#define OOS_ASSERT(condition) \
    do                               \
    {                                \
    } while (false)
#endif
