#pragma once

#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"

#include <cstddef>
#include <span>

namespace oos::elements::mixedHybrid
{

// The number of quadrature points of a shell element chosen from its shape (spec section 6; a deviation from the
// reference, section 12, whose rule has 2k + 4 points per direction on every element).
//
// The integrands of an element are polynomials of the reference coordinates times factors of its geometry: the area
// element j, the maps of the membrane strains and of the moments to the local frame, the Christoffel symbols, the
// normal, and the section laws where the thickness varies or the section is offset. On an affine element (a flat
// parallelogram, a flat triangle) those factors are constants and a rule of 5 points is exact; on an element that is
// not, they are rational or algebraic and a Gauss rule converges on them exponentially, the faster the nearer the
// element is to an affine one. A curved element of a regular mesh is integrated to the round-off by 7 or 8 points,
// a distorted one needs more.
//
// The choice measures that convergence on the element itself. For each rule of n points per direction from the
// smallest, the integrals of those geometric factors against polynomials of the highest degree the element multiplies
// them by (degree 8 in each coordinate: the square of a displacement function of degree 4) are formed on the rules of
// n and of n + 1 points; the first n for which every integral changes by less than the tolerance, relative to the
// integral of the absolute value of its factor, is the rule of the element. No rule goes beyond the largest, which is
// the rule of the reference: an element that does not converge before it keeps it.
//
// Thread-safety: a pure function (the reference elements of the rules are built once and shared).
struct ShellRuleChoice
{
    // The smallest and the largest number of points per direction. The largest is the rule of the reference.
    std::size_t minimumPoints = 6;
    std::size_t maximumPoints = 12;
    // The relative change of the integrals of the geometric factors between two successive rules below which the
    // smaller one is taken.
    double tolerance = 1e-10;
};

// The points per direction of the rule of the element of `cornerCount` corners and displacement degree `degree` whose
// map, weight, axes, section and offset `shape` gives (its tables of jets are not read: the choice forms its own on
// each rule it tries). The largest rule when a point of the element has no area (the build then refuses the element).
// `edgeDegrees`: the degree of each edge of the element when some are below `degree` (empty: `degree` on every edge);
// the map of `shape` is given on the displacement functions of that element.
[[nodiscard]] std::size_t chooseShellPoints(std::size_t cornerCount, std::size_t degree,
                                            const ShellElementDefinition& shape, const ShellRuleChoice& choice,
                                            std::span<const std::size_t> edgeDegrees = {});

} // namespace oos::elements::mixedHybrid
