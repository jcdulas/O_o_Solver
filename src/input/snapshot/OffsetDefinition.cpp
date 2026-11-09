#include "oos/input/snapshot/OffsetDefinition.hpp"

#include <array>
#include <utility>

namespace oos::input
{

namespace
{

constexpr std::array<std::pair<std::string_view, OffsetDefinition>, 3> NAMES{{
    {"ElemOffsetsOnly", OffsetDefinition::ElementOffsets},
    {"LoadRecoveryOffsets", OffsetDefinition::LargeRotationOffsets},
    {"ElemOffsetZ", OffsetDefinition::ElementZ},
}};

std::string_view trimmed(std::string_view text) noexcept
{
    const std::size_t first = text.find_first_not_of(' ');
    if (first == std::string_view::npos)
        return {};
    const std::size_t last = text.find_last_not_of(' ');
    return text.substr(first, last - first + 1);
}

} // namespace

std::string_view offsetDefinitionName(OffsetDefinition definition) noexcept
{
    for (const auto& [name, value] : NAMES)
        if (value == definition)
            return name;
    return {};
}

std::optional<OffsetDefinition> offsetDefinitionOf(std::string_view value) noexcept
{
    const std::string_view text = trimmed(value);
    for (const auto& [name, definition] : NAMES)
        if (name == text)
            return definition;
    return std::nullopt;
}

} // namespace oos::input
