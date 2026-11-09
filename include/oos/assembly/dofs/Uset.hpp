#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"

#include <ostream>
#include <span>

namespace oos::assembly
{

// The set table of the text report (PARAM USETPRT): the members of each set in the g-set order of DofMap (no
// resequencing): per set, a title "Set <NAME>", a column header 1 to 10, then the members ten per line, each written
// "<point ID>-<component>" (component 0 for a scalar point), the line starting with the position of its first member
// in the set ("     11=").
//
// The layout is this project's own; the content (which DOF is in which set) is what the tests check.
//
// Thread-safety: stateless.
class Uset
{
public:
    // Writes the table of each of `sets`, in the order given.
    static void write(std::ostream& out, const DofSets& dofSets, const DofMap& dofMap, std::span<const DofSet> sets);
};

} // namespace oos::assembly
