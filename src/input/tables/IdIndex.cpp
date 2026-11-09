#include "oos/input/tables/IdIndex.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace oos::input
{

namespace
{

// The row count of an index, checked before its pairs are reserved (sizes are checked before any allocation).
std::size_t checkedRowCount(std::size_t rows, const std::string& label)
{
    if (rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw base::ContractError(
            base::concatenate({"index of ", label, ": expected at most 2^31 - 1 rows, found ", std::to_string(rows),
                               "; the model is larger than O_o_Reader's 32-bit IDs allow: split it into parts "
                               "(superelements) or external units"}));
    return rows;
}

} // namespace

IdIndex::IdIndex(std::span<const std::int32_t> ids, base::MemoryBudget& budget, const std::string& label)
    : m_pairs(budget, checkedRowCount(ids.size(), label), label)
{
    for (std::size_t row = 0; row < ids.size(); ++row)
        m_pairs[row] = {ids[row], static_cast<std::int32_t>(row)};
    const std::span<IdRow> pairs = m_pairs.span();
    std::sort(pairs.begin(), pairs.end(), [](const IdRow& a, const IdRow& b) { return a.id < b.id; });
    const auto duplicate =
        std::adjacent_find(pairs.begin(), pairs.end(), [](const IdRow& a, const IdRow& b) { return a.id == b.id; });
    if (duplicate != pairs.end())
        throw base::ContractError(base::concatenate(
            {"index of ", label, ": expected unique IDs, found ID ", std::to_string(duplicate->id), " at rows ",
             std::to_string(duplicate->row), " and ", std::to_string((duplicate + 1)->row),
             "; O_o_Reader gives unique IDs per table, so the table was built by hand: remove the duplicate"}));
}

std::optional<std::int32_t> IdIndex::find(std::int32_t id) const noexcept
{
    const std::span<const IdRow> pairs = m_pairs.span();
    const auto found = std::lower_bound(pairs.begin(), pairs.end(), id,
                                        [](const IdRow& pair, std::int32_t key) { return pair.id < key; });
    if (found == pairs.end() || found->id != id)
        return std::nullopt;
    return found->row;
}

} // namespace oos::input
