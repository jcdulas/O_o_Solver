#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oos::output
{

// One element result table in the layout of the O_o_Reader data contract (the result tables: the kinds "stress",
// "strain", "forces" with an element type such as "ShellQuad4" or "Bar2LoadStations"): one static step, one row per
// place of an element (a CQUAD4 has its centre and its corners, one row per fibre or per layer at each), the values in
// the columns the element family names (the component names of O_o_Reader's result tables, in the element frame).
//
// The row labels are those of O_o_Reader's result tables: the element ID, the grid point of the place (0 for the
// centre), the layer, and the position along a bar or beam when the table has stations (`hasPositions`).
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted).
struct ElementResultTable
{
    std::string kind;        // "stress", "strain", "forces"
    std::string elementType; // "ShellQuad4", "Bar2LoadStations", ...
    std::int32_t subcase = 1;
    std::string analysis = "static"; // "static", "modes", "buckling"
    // The step of a modal or buckling table, as NodalResultTable::step and eigenvalue.
    std::int32_t step = 0;
    double eigenvalue = 0.0;
    std::vector<std::string> components;        // bounded by the layout of one element type
    base::BudgetedBuffer<std::int32_t> ids;     // the element ID of each row
    base::BudgetedBuffer<std::int32_t> nodeIds; // the grid point of the place, 0 for the centre
    base::BudgetedBuffer<std::int32_t> layers;
    base::BudgetedBuffer<double> positions; // empty unless hasPositions
    bool hasPositions = false;
    base::BudgetedBuffer<double> values; // rows × components, row-major

    [[nodiscard]] std::size_t rowCount() const noexcept { return ids.size(); }
    [[nodiscard]] std::size_t componentCount() const noexcept { return components.size(); }
};

} // namespace oos::output
