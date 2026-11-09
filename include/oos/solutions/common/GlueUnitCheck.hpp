#pragma once

#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/assembly/constraints/RigidElementEquations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/reduction/plan/PlanGraph.hpp"
#include "oos/reduction/units/UnitSet.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::solutions
{

// Whether a row of Gm is a glue equation (assembly::GlueConstraints).
[[nodiscard]] bool isGlueSource(assembly::ConstraintSource source) noexcept;

// What the glue equations of a model do across the units of a condensation. Each glued grid point is one relation of
// the plan graph (reduction::glueRelationKey of its ID), its rows depending on the corners of its face. The units come
// from O_o_Reader's components, whose contact rule is not that of the glue equations, so a point may be glued to a face
// of another unit even where the components say nothing:
// - a point whose equations have all their DOFs in one unit is eliminated inside it, before the condensation
//   (insidePointCount);
// - a point whose equations join units (its face's corners in another unit, or a point no unit holds) is eliminated
//   at the interface: its relation is one of interfaceRelations, which CondensationPlan takes with the connection
//   system's, so that reduction::BoundaryPromotion keeps its dependent DOFs as boundary DOFs of its unit and puts it
//   and every corner of its face on the boundaries of the units that hold them; a report line per contact table says
//   so. No equation is ever dropped.
struct GlueUnitCheck
{
    std::vector<reduction::RelationKey> interfaceRelations; // sorted, distinct: one per glued point at the interface
    std::int64_t insidePointCount = 0;
    std::int64_t crossingPointCount = 0;
    std::vector<std::string> reportLines; // one per contact table whose glue joins units
};

// Marks no unit in unitOfGridRows.
inline constexpr std::int64_t NO_UNIT = -1;

// The unit of each grid point (by grid row of dofMap; NO_UNIT for a point no unit holds): the points a unit lists (a
// group's), else the component of each point (componentNodes) through UnitSet::unitOfComponent.
[[nodiscard]] base::BudgetedBuffer<std::int64_t> unitOfGridRows(const reduction::UnitSet& units,
                                                                const input::ModelSnapshot& snapshot,
                                                                const assembly::DofMap& dofMap,
                                                                base::MemoryBudget& budget);

// The check of the glue rows against the units (the rows of GlueConstraints, each with the corners of its face; a
// corner made dependent by a rigid element is its own relation's business, BoundaryPromotion's rule). Deterministic:
// the rows in their order, the outputs sorted.
[[nodiscard]] GlueUnitCheck checkGlueAgainstUnits(const assembly::GlueConstraints& glue, const assembly::DofMap& dofMap,
                                                  std::span<const std::int64_t> unitOfGridRow);

} // namespace oos::solutions
