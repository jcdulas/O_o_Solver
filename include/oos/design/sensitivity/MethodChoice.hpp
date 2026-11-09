#pragma once

#include "oos/design/sensitivity/DesignGradients.hpp"

#include <cstdint>

namespace oos::design
{

// The choice of the order of the sensitivities of one subcase by its counted solves (the responses are
// counted in items, never in cards; the choice counts the solves of the direct method, of the adjoint on the argument
// items, an item shared by several equations counted once, and of the chained adjoint, and takes the fewest):
//   direct  = the design variables with a non-zero pseudo-load in the subcase,
//   adjoint = the items read with the displacements: the plain items asked for and every argument item of the
//             equations asked for (through every level of equations), each once,
//   chained = the plain items asked for themselves plus the equation items asked for (one solve each, whatever the
//             number of their arguments).
// Items without displacement weights (a weight, a volume, an equation of variables only) need no solve and are not
// counted. On a tie the direct order is taken, then the adjoint (the simplest orders first). A forced method is kept
// as asked; the counts are reported either way.
//
// Thread-safety: stateless.
class MethodChoice
{
public:
    // The counts and the order for a subcase. `forced`: Automatic, or the order to take.
    [[nodiscard]] static SubcaseSensitivity choose(std::int32_t subcase, std::int64_t variables,
                                                   std::int64_t argumentItems, std::int64_t requestedItems,
                                                   SensitivityMethod forced) noexcept;
};

} // namespace oos::design
