#pragma once

#include <cstdint>
#include <limits>

namespace oos::base
{

// The integer widths: the IDs of the input deck (grid points, elements, properties, materials, sets) are 32-bit, as
// O_o_Reader stores them; DOF indices are 32-bit, so a model above 2 147 483 647 DOFs is refused; offsets into
// sparse structures and non-zero counts are 64-bit (Offset); dense matrices are indexed with std::size_t.
using EntityId = std::int32_t;
using DofIndex = std::int32_t;
using Offset = std::int64_t;

// The largest number of DOFs a model may have: every DOF index must fit a DofIndex.
inline constexpr std::int64_t MAX_DOF_COUNT = std::numeric_limits<DofIndex>::max();

} // namespace oos::base
