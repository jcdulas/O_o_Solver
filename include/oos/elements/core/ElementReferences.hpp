#pragma once

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/input/tables/CoordinateSystemTable.hpp"

#include <cstdint>

namespace oos::elements
{

// The row of the grid point `gridId` that the field `field` of the element of batch row `row` names, for the cards
// whose table gives grid point IDs without a row column. A scalar point or an unknown ID is a
// problem.
[[nodiscard]] Resolution<std::int32_t> resolveGridRow(const ElementBatchView& view, std::int64_t row,
                                                      std::int32_t gridId, const char* field);

// The row of grid point `gridId` from the row `givenRow` that a row-index column (nodesRows) gives for it: that row
// when the node table holds the grid point there, else the row found by the ID. O_o_Reader c985f57 gives nodesRows in
// the reading order of the GRID cards while its node table is sorted by ID, so the rows are wrong in a deck whose GRIDs
// are not in ID order (an O_o_Reader issue): the ID decides.
// Precondition: the grid point exists (the caller checked it); throws ContractError otherwise.
[[nodiscard]] std::int32_t checkedGridRow(const ElementBatchView& view, std::int32_t givenRow, std::int32_t gridId);

// The rectangular coordinate system `frameId` (0 the basic system) that the field `field` names. An unknown system is a
// problem; so is a cylindrical or spherical one, which the elements using this do not support yet (their local axes
// depend on the point where they are evaluated, which is not chosen yet).
[[nodiscard]] Resolution<input::CoordinateFrame> resolveRectangularFrame(const ElementBatchView& view, std::int64_t row,
                                                                         std::int32_t frameId, const char* field);

} // namespace oos::elements
