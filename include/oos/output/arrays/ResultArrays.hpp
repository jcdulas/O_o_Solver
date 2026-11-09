#pragma once

#include "oos/output/arrays/ElementResultTable.hpp"
#include "oos/output/arrays/GridPointForceTable.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"

#include <vector>

namespace oos::output
{

// The result tables of an analysis in the layouts of the O_o_Reader data contract, in the order they were made (per
// subcase: displacements, applied loads, SPC forces, MPC forces; the element tables of the subcase by batch, then
// stress, strain, forces, strain energy; the grid point forces). The caller (the WebAssembly bindings) hands them to
// the JavaScript layer.
//
// Thread-safety: immutable once built. Ownership: owns its tables.
struct ResultArrays
{
    std::vector<NodalResultTable> nodalTables;             // bounded by subcases × kinds
    std::vector<ElementResultTable> elementTables;         // bounded by subcases × element types × kinds
    std::vector<GridPointForceTable> gridPointForceTables; // bounded by the subcases
};

} // namespace oos::output
