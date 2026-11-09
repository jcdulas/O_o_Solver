#include "oos/solutions/superelements/SuperelementPart.hpp"

#include <algorithm>

namespace oos::solutions
{

bool SuperelementSubcase::selects(std::int32_t superelementId) const noexcept
{
    return superelements.empty() || std::binary_search(superelements.begin(), superelements.end(), superelementId);
}

} // namespace oos::solutions
