#include "oos/base/errors/InternalError.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <string>

namespace oos::base
{

void failAssertion(const char* condition, const char* file, int line)
{
    throw InternalError(concatenate({"internal error: '", condition, "' does not hold at ", file, ":",
                                     std::to_string(line), "; this is a bug of O_o_Solver, please report it"}));
}

} // namespace oos::base
