#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <vector>

namespace oos::design
{

// The mesh distortion checks of a design point that moves grid points, reported as diagnostics: the element quality
// measures the element families already compute in their check() (a re-entrant or collapsed shell, a Jacobian that
// changes sign in a solid, coincident end points of a bar or a rod, an orientation vector along the axis, a warp above
// the reporting ratio of ShellQuad4...), run on the snapshot of the design point and compared with those of the base
// snapshot: every problem of an element at the design point that the base model does not have is a distortion the shape
// variables cause. Each becomes a diagnostic of code design_code::SHAPE_DISTORTION, of the element and field the family
// names, with the family's severity: an error (the element cannot be computed: the design point is refused before its
// analysis, DesignProblem::evaluateStatics) or a warning (the analysis runs; the warning goes with its results).
//
// A problem is matched with the base's by its code, card, element ID and field (not its message, which carries the
// measured value): an element warped at the base design and still warped at the point is not reported again.
//
// Thread-safety: stateless. Ownership: none kept.
class ShapeDistortion
{
public:
    // The problems of the element checks of a snapshot (every batch of the standard registry), as its families report
    // them. Throws as SnapshotIndex.
    [[nodiscard]] static std::vector<base::Diagnostic> elementProblems(const input::ModelSnapshot& snapshot,
                                                                       base::MemoryBudget& budget);
    // The distortions of a design point: the problems of `point` (elementProblems of its snapshot) that `baseProblems`
    // (those of the base snapshot) does not have, as SHAPE_DISTORTION diagnostics.
    [[nodiscard]] static std::vector<base::Diagnostic> distortions(const std::vector<base::Diagnostic>& baseProblems,
                                                                   const std::vector<base::Diagnostic>& point);
};

} // namespace oos::design
