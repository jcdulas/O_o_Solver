#include "oos/linalg/supernodal/FactorStorage.hpp"

namespace oos::linalg
{

bool FactorStorage::fits(const SymbolicFactor& symbolic) const noexcept
{
    if (m_values.size() != static_cast<std::size_t>(symbolic.panelValueCount()))
        return false;
    std::size_t rows = 0;
    for (std::int32_t s = 0; s < symbolic.supernodeCount(); ++s)
        rows += static_cast<std::size_t>(symbolic.columnCount(s) + symbolic.rowCount(s));
    return m_rows.size() == rows;
}

} // namespace oos::linalg
