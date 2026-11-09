#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace oos::design
{

// The status of one item of a response: an external response (a callable or
// supplied values) may give one per item. A failed item is NaN with its status, the other items keep their values, and
// the point has the status "partial"; an item a selection leaves out (select=) is not_evaluated, which is not a
// failure: it is excluded from the constraints, the screening and the gradients, and counted in the report. An equation
// item that reads an item that is not ok takes its status (failed before not_evaluated).
enum class ItemStatus : std::uint8_t
{
    Ok,
    Failed,
    NotEvaluated
};

// "ok", "failed", "not_evaluated".
[[nodiscard]] constexpr std::string_view itemStatusName(ItemStatus status) noexcept
{
    switch (status)
    {
    case ItemStatus::Ok:
        return "ok";
    case ItemStatus::Failed:
        return "failed";
    case ItemStatus::NotEvaluated:
        return "not_evaluated";
    }
    return "";
}

[[nodiscard]] constexpr std::optional<ItemStatus> findItemStatus(std::string_view name) noexcept
{
    if (name == "ok")
        return ItemStatus::Ok;
    if (name == "failed")
        return ItemStatus::Failed;
    if (name == "not_evaluated")
        return ItemStatus::NotEvaluated;
    return std::nullopt;
}

// The status of an item that reads two items of these statuses: the worse (failed, then not_evaluated, then ok).
[[nodiscard]] constexpr ItemStatus combinedStatus(ItemStatus a, ItemStatus b) noexcept
{
    if (a == ItemStatus::Failed || b == ItemStatus::Failed)
        return ItemStatus::Failed;
    if (a == ItemStatus::NotEvaluated || b == ItemStatus::NotEvaluated)
        return ItemStatus::NotEvaluated;
    return ItemStatus::Ok;
}

} // namespace oos::design
