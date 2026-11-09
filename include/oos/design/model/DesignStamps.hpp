#pragma once

#include "oos/design/model/DesignModel.hpp"
#include "oos/design/model/FieldTargets.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::design
{

// The design stamp of a unit no design variable drives (empty).
inline constexpr std::uint64_t EMPTY_DESIGN_STAMP = 0;

// The design stamp of every unit of `targets` (in the order of FieldTargets::unitIds()) for the field values of a
// design point, part of the cache key: a hash of the values a design point gives to the fields of the unit and to the
// positions of its grid points.
//
// Method: FNV-1a (Fowler, Noll, Vo; 64 bits) over, for each relation that changes the unit in relation order, the
// identity of its field (target kind, card, ID, column, item) and the bits of the value it gives; then, for each shape
// variable that moves grid points of the unit (DVGRID), in basis order, the ID of every grid point it moves
// and the bits of that point's position at the design point (FieldTargets::movedPositions: the sum of every shape's
// move). The stamp depends on the field values and the positions only, not on x: a design point that changes a
// variable driving no field and no grid point of a unit leaves its stamp unchanged, and two points giving the same
// values give the same stamp (a basis that also moves points of other units hashes them too: a stamp may change when
// only those move, a condensation made again, never one kept wrongly). A driven unit's stamp is never
// EMPTY_DESIGN_STAMP (a hash that falls on it is moved to 1). Precondition: `fields` has one value per relation of the
// model, `shapes` one per shape basis (empty when the model has none).
[[nodiscard]] std::vector<std::uint64_t> computeDesignStamps(const DesignModel& model, const FieldTargets& targets,
                                                             const std::vector<FieldValue>& fields,
                                                             std::span<const double> shapes = {});

} // namespace oos::design
