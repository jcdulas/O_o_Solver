#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::output
{

// The grid point forces of one subcase in the layout of the O_o_Reader data contract (the result tables of kind
// "nodePointForces"): for each grid point of the request, one row per contribution, its six components in the
// displacement system CD of the grid point (stored frame "nodeOutputSystem"):
//   appliedLoad             the applied loads (OLOAD),
//   singlePointConstrForce  the forces of the single-point constraints (when the point has constrained DOFs),
//   multiPointConstrForce   the forces of the multipoint constraints and rigid elements (when nonzero; for a reader of
//                           O_o_Reader's data contract 56 or later, one row per relation acting at the grid point
//                           instead, a zero one included: elementIds the rigid element's ID or minus the MPC card's
//                           set ID, the rows of a point summing to the single row;
//                           StiffnessRecovery::NodalState::mpcRelations),
//   one row per element connected, the force the element exerts on the grid point, −(K_e·u_e − f_e,thermal),
//   total                   their sum, zero to round-off (the equilibrium of the grid point).
// The row labels: the grid point (ids), the element (elementIds, 0 for the other rows) and its name (texts
// "elementName": the fixed rows above, or the result element type of the element's family, "Bar2", "ShellQuad4");
// names index a small dictionary whose first entries are FIXED_ROW_NAMES.
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted).
struct GridPointForceTable
{
    static constexpr int COMPONENT_COUNT = 6;
    // The names of the rows that are not an element's, in the order of the first entries of `names`.
    static constexpr std::array<const char*, 4> FIXED_ROW_NAMES{"appliedLoad", "singlePointConstrForce",
                                                                "multiPointConstrForce", "total"};

    std::int32_t subcase = 1;
    std::string analysis = "static";
    std::vector<std::string> names;             // the dictionary of element names, bounded by the element types
    base::BudgetedBuffer<std::int32_t> nodeIds; // one per row
    base::BudgetedBuffer<std::int32_t> elementIds;
    base::BudgetedBuffer<std::int32_t> nameIndices; // into names
    base::BudgetedBuffer<double> values;            // rows × 6, row-major

    [[nodiscard]] std::size_t rowCount() const noexcept { return nodeIds.size(); }
};

} // namespace oos::output
