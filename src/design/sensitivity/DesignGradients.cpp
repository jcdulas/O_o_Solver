#include "oos/design/sensitivity/DesignGradients.hpp"

namespace oos::design
{

bool isItemDifferentiated(const GradientOptions& options, std::int32_t responseId, std::size_t item) noexcept
{
    for (const ItemMask& mask : options.itemMasks)
        if (mask.responseId == responseId)
            return item < mask.isRetained.size() && mask.isRetained[item] != 0;
    return true;
}

} // namespace oos::design
