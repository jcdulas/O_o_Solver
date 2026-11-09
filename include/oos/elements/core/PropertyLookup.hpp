#pragma once

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/input/tables/CardTable.hpp"

#include <cstdint>
#include <string_view>

namespace oos::elements
{

// The property of an element: the table of its card type and its row there.
struct PropertyRef
{
    const input::CardTable* table = nullptr;
    std::int32_t row = 0;
};

// The property columns of a batch whose elements take one property type (`expectedCard`, "SpringProp"), read once per
// facet call: the PIDs, O_o_Reader's propIdRow (the row of the property in the table of its type, null when the PID
// is not defined) and propType (that type), and the table of that type (nullptr when the model has none).
struct PropertyRows
{
    std::string_view expectedCard;
    std::span<const std::int32_t> propIds;
    std::span<const std::int32_t> rows;
    base::ValidityView valid;
    base::DictionaryView types;
    const input::CardTable* table = nullptr;
    std::span<const std::int32_t> tableIds; // the PIDs of `table` by row (empty without a table)

    PropertyRows(const ElementBatchView& view, std::string_view expectedCard);
};

// The property of the element of batch row `row`; a PID that the model does not define, or that names another
// property type, is a problem on the field PID. The row propIdRow gives is used when the property table holds the PID
// there; otherwise the PID decides (O_o_Reader c985f57 gives propIdRow in the reading order of the property cards while
// its property tables are sorted by ID: wrong rows in a deck whose properties are not in ID order).
[[nodiscard]] Resolution<PropertyRef> resolveProperty(const ElementBatchView& view, const PropertyRows& properties,
                                                      std::int64_t row);

} // namespace oos::elements
