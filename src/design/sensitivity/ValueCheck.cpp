// ValueCheck: an item recomputed for its derivatives against the analysis' value.

#include "oos/design/sensitivity/ValueCheck.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace oos::design
{

namespace
{

// A value with its significant digits (std::to_string prints 1e-11 as 0.000000).
std::string shown(double value)
{
    std::ostringstream text;
    text.precision(10);
    text << value;
    return text.str();
}

} // namespace

bool ValueCheck::agrees(const ResponseValues& values, std::size_t item, double recomputed, double terms) noexcept
{
    const double analysed = values.values[item];
    const double difference = std::abs(recomputed - analysed);
    if (recomputed == analysed || difference <= VALUE_CHECK_TOLERANCE * std::abs(analysed) ||
        difference <= ROUNDING_CHECK_TOLERANCE * terms)
        return true;
    double largest = 0.0;
    for (const double value : values.values)
        largest = std::max(largest, std::abs(value));
    return difference <= VALUE_CHECK_TOLERANCE * VALUE_CHECK_FLOOR * largest;
}

void ValueCheck::require(const ResponseValues& values, std::size_t item, double recomputed, double terms)
{
    if (agrees(values, item, recomputed, terms))
        return;
    const ResponseItem& label = values.items[item];
    throw base::OptimisationError(base::concatenate(
        {"the sensitivity of response ", std::to_string(values.id), " (", values.label, "), item ",
         std::to_string(label.component), " of ", std::to_string(label.entity), " in subcase ",
         std::to_string(label.subcase), ": the item recomputed for its derivatives is ", shown(recomputed),
         ", the analysis gave ", shown(values.values[item]), " (the rounding of its recovery ",
         shown(ROUNDING_CHECK_TOLERANCE * terms), ")",
         "; the derivative would be that of another quantity (a STRAIN request with FIBER, a Bar2 under "
         "BeamDistributedLoad read "
         "in its station layout), so it is not returned. Report the model if the request is a plain one"}));
}

} // namespace oos::design
