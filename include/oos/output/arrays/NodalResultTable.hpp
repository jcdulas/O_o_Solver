#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"

#include <cstdint>
#include <string>

namespace oos::output
{

// The point type of a row of a nodal result table, as the result tables of O_o_Reader number them
// (point_types: 1 a grid point, 2 a scalar point).
enum class PointType : std::int8_t
{
    Grid = 1,
    Scalar = 2
};

// One nodal result table in the layout of the O_o_Reader data contract (the result tables: the kinds "displacements",
// "singlePointConstrForces", "multiPointConstrForces", ...): one static step, one row per point (the grid points, then
// the scalar points, in the order of the node table), six components t1, t2, t3, r1, r2, r3 in the displacement system
// CD of the grid point (stored frame "nodeOutputSystem"); a scalar point has its value in t1 and zeros elsewhere.
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted).
struct NodalResultTable
{
    static constexpr int COMPONENT_COUNT = 6;

    std::string kind; // "displacements", "singlePointConstrForces", "multiPointConstrForces"
    std::int32_t subcase = 1;
    std::string analysis = "static"; // "static", "modes", "buckling"
    // The step of a modal or buckling table: the mode number (from 1) and its eigenvalue (λ = ω² for a mode, the load
    // factor for a buckling mode); 0 for a static table. The caller presents the modes of a subcase as one table with
    // one step per mode.
    std::int32_t step = 0;
    double eigenvalue = 0.0;
    base::BudgetedBuffer<std::int32_t> ids;       // one per row
    base::BudgetedBuffer<std::int8_t> pointTypes; // PointType of each row
    base::BudgetedBuffer<double> values;          // rows × 6, row-major

    [[nodiscard]] std::size_t rowCount() const noexcept { return ids.size(); }
};

} // namespace oos::output
