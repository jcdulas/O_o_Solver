#pragma once

#include <cstdint>

namespace oos::base
{

// The offsets of a variable-size list column, read in place in either Arrow layout: int64 ("+L", what O_o_Reader
// gives) or int32 ("+l"). The views of list columns (ListView, NestedListView) hold one; ContractReader checks them
// once (non-decreasing, within the child) before handing a view out.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
class ListOffsets
{
public:
    ListOffsets() = default;
    // Exactly one of the pointers is not null; it points at the offset of row 0.
    ListOffsets(const std::int32_t* narrow, const std::int64_t* wide) : m_narrow(narrow), m_wide(wide) {}

    // The offset of a row into the child values; [size] is the end of the last row. One predictable branch per read
    // instead of a widened copy of the offsets.
    [[nodiscard]] std::int64_t operator[](std::int64_t row) const noexcept
    {
        return m_wide != nullptr ? m_wide[row] : static_cast<std::int64_t>(m_narrow[row]);
    }

private:
    const std::int32_t* m_narrow = nullptr;
    const std::int64_t* m_wide = nullptr;
};

} // namespace oos::base
