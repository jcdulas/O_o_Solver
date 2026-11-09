#include "oos/design/approximation/ExplicitConstraintGroup.hpp"

#include <algorithm>

namespace oos::design
{

namespace
{

bool isIn(const std::vector<std::int32_t>& list, std::int32_t value) noexcept
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

} // namespace

bool ExplicitConstraintGroup::contains(const ConstraintValue& constraint, std::optional<std::int32_t> region,
                                       bool isElementResponse, bool isGridResponse) const noexcept
{
    return isIn(constraintSets, constraint.setId) || isIn(responses, constraint.responseId) ||
           (region && isIn(regions, *region)) || (isElementResponse && isIn(elements, constraint.label.entity)) ||
           (isGridResponse && isIn(grids, constraint.label.entity));
}

} // namespace oos::design
