#include "oos/elements/core/PropertyLookup.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <string>

namespace oos::elements
{

PropertyRows::PropertyRows(const ElementBatchView& view, std::string_view card)
    : expectedCard(card),
      propIds(view.batch().propertyIds()),
      rows(view.batch().propertyRows()),
      valid(view.columns().validity("propIdRow")),
      types(view.batch().propertyTypes()),
      table(view.propertyTable(card)),
      tableIds(table != nullptr ? table->ids() : std::span<const std::int32_t>{})
{
}

Resolution<PropertyRef> resolveProperty(const ElementBatchView& view, const PropertyRows& properties, std::int64_t row)
{
    using Result = Resolution<PropertyRef>;
    const std::string_view expectedCard = properties.expectedCard;
    const std::int32_t propId = properties.propIds[static_cast<std::size_t>(row)];
    const input::CardTable* table = properties.table;
    if (!properties.valid.isValid(row) || table == nullptr)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_PROPERTY, "propId",
            base::concatenate({"expected the ID of a ", expectedCard, ", found ", std::to_string(propId),
                               ", which the model does not define; define the property or correct the card"})));
    if (properties.types[row] != expectedCard)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_PROPERTY, "propId",
            base::concatenate({"expected a ", expectedCard, " for property ", std::to_string(propId), ", found a ",
                               properties.types[row], "; ", view.card(), " takes a ", expectedCard})));
    std::int32_t propertyRow = properties.rows[static_cast<std::size_t>(row)];
    const std::span<const std::int32_t> ids = properties.tableIds;
    const bool isRowOfPropId =
        propertyRow >= 0 && propertyRow < table->rowCount() && ids[static_cast<std::size_t>(propertyRow)] == propId;
    if (!isRowOfPropId)
    {
        // The tables of O_o_Reader are sorted by ID; a linear search covers one that would not be.
        const auto sorted = std::lower_bound(ids.begin(), ids.end(), propId);
        auto found = sorted != ids.end() && *sorted == propId ? sorted : std::find(ids.begin(), ids.end(), propId);
        if (found != ids.end())
            propertyRow = static_cast<std::int32_t>(found - ids.begin());
    }
    if (propertyRow < 0 || propertyRow >= table->rowCount() || ids[static_cast<std::size_t>(propertyRow)] != propId)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_PROPERTY, "propId",
            base::concatenate({"expected the row of ", expectedCard, " ", std::to_string(propId), " below ",
                               std::to_string(table->rowCount()), ", found ", std::to_string(propertyRow),
                               "; the element and property tables come from different snapshots"})));
    return Result::success({table, propertyRow});
}

} // namespace oos::elements
