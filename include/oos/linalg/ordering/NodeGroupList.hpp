#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// A list of groups of nodes in compressed form: group g holds nodes[offsets[g], offsets[g + 1]). The nodes of one
// group are all coupled with each other: the grid points of an element, or the extra couplings that the connectivity
// does not show (the legs of an absorbed spider, the nodes of an internal constraint). A view: it owns nothing.
//
// Invariants (checked by the users, which throw std::invalid_argument): offsets has groupCount + 1 nondecreasing
// entries, offsets[0] = 0 and offsets.back() = nodes.size(). A node may appear twice in a group (it counts once).
struct NodeGroupList
{
    std::span<const std::int64_t> offsets;
    std::span<const std::int32_t> nodes;

    [[nodiscard]] std::size_t groupCount() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
    [[nodiscard]] std::span<const std::int32_t> group(std::size_t g) const
    {
        const auto begin = static_cast<std::size_t>(offsets[g]);
        return nodes.subspan(begin, static_cast<std::size_t>(offsets[g + 1]) - begin);
    }
};

} // namespace oos::linalg
