#pragma once

#include "oos/design/external/ItemStatus.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::design
{

// The label of one response item (responses are counted in items: one grid point component, one element value, one
// output of an equation). `entity` is the grid point (DISP, SPCFORCE), the element (STRESS ... CSTRAT), the
// label of a by-label equation, or 0 (WEIGHT, VOLUME, FRMASS, an equation by the card rules); `component` the
// displacement component (1 to 6) or the item code (ATTA), or the output index of an equation (from 0);
// `layer` the lamina of a composite item (ATTB), 0 otherwise. Subcase 0 is a global item (evaluated once).
//
// The sensitivities form the adjoint right-hand side ∂r/∂u of an item from this label: the grid DOF of a DISP item,
// the element and item code of an element item.
struct ResponseItem
{
    std::int32_t subcase = 0;
    std::int32_t entity = 0;
    std::int32_t component = 0;
    std::int32_t layer = 0;

    friend bool operator==(const ResponseItem&, const ResponseItem&) = default;
};

// The values of one design response at a design point: one per item, in the item order (subcases in the order they
// were analysed, global items first; inside a subcase the order of the ATTi, then of the components or items).
//
// `statuses`: the status of each item when some item is not ok (an external response, an equation that
// reads one), empty when every item is ok.
struct ResponseValues
{
    std::int32_t id = 0;
    std::string label;
    std::vector<ResponseItem> items;
    std::vector<double> values;
    std::vector<ItemStatus> statuses;

    [[nodiscard]] std::size_t itemCount() const noexcept { return items.size(); }
    [[nodiscard]] ItemStatus statusOf(std::size_t item) const noexcept
    {
        return statuses.empty() ? ItemStatus::Ok : statuses[item];
    }
    [[nodiscard]] bool isOk(std::size_t item) const noexcept { return statusOf(item) == ItemStatus::Ok; }
};

} // namespace oos::design
