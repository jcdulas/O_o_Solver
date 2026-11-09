#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace oos::elements
{

// The geometry of the higher-order shells ShellQuad8 and ShellTria6 from their grid points alone (HigherOrderShell), and
// the directors their nodes share at the grid points of a model.

// A higher-order shell whose height off the mean plane of its corners is at most this fraction of its size is flat.
inline constexpr double HIGHER_ORDER_SHELL_FLATNESS = 1.0e-8;
// The smallest cosine allowed between the surface normal of a curved element and its element normal: cos 45°.
inline constexpr double HIGHER_ORDER_SHELL_MIN_NORMAL_COSINE = 0.70710678118654752;

// The geometry of one element: the element frame of its corners (the ShellQuad4 frame for the ShellQuad8, the ShellTria3
// frame for the ShellTria6, origin at the mean of the corners), its nodes in that frame (projected on its plane when the
// element is flat) and its own directors there: the unit normal of its isoparametric surface at each node
// (MindlinShell:: nodalDirectors), the element normal n at every node of a flat element.
template <std::size_t N>
struct HigherOrderShellGeometry
{
    ShellCorners<N> nodes;                    // the grid points (rows, IDs, basic positions, CD frames)
    base::Mat3 axes = base::Mat3::identity(); // columns x, y, n of the element frame in basic
    double edgeAngle = 0.0;                   // the edge G1→G2 from the x axis (the reference of THETA)
    bool isCurved = false;
    std::array<base::Vec3, N> position{};      // in the element frame; z = 0 on a flat element
    std::array<base::Vec3, N> ownDirector{};   // in the element frame
};

// The geometry of the element of batch row `row` (Shape: SerendipityQuadShape or QuadraticTriangleShape); refused: a
// blank edge grid point, an unknown grid point, corners on a line, a map whose Jacobian on the element plane is not
// positive, a curved element whose normal turns by more than 45° from the plane of its corners.
template <typename Shape>
[[nodiscard]] Resolution<HigherOrderShellGeometry<Shape::NODES>> resolveHigherOrderShellGeometry(
    const ElementBatchView& view, const ShellConnectivity& connectivity, std::int64_t row);

// The directors shared at the grid points of a model (Hughes & Liu, "Nonlinear finite element analysis of shells:
// part I. Three-dimensional shells", CMAME 26 (1981) 331-362: one fibre direction per node of the mesh; Bathe, Finite
// Element Procedures (1996), §5.4: the director vectors defined at the nodes). Each element computes its own normal at
// each of its nodes; two curved elements meeting at a grid point give it normals that differ by the interpolation
// error of their surfaces (about α³/2 for elements of half-angle of arc α), which leaves the rotation about them a near
// mechanism. So, at each grid point, over every ShellQuad8 and ShellTria6 of the snapshot:
// - the elements are grouped into smooth groups: two elements whose own normals there make an angle below
//   HIGHER_ORDER_SHELL_SHARP_EDGE_DEGREES (in direction, whatever their orientation) are in one group, and the groups
//   are the connected sets of that relation; a fold, a T-junction or a stiffener meeting the skin keeps one director
//   per smooth side;
// - a group of one element, a group of flat elements only (a flat mesh, or flat facets meeting at small angles: a
//   faceted curved surface keeps the behaviour it had, each facet holding the others' normal rotation by its bending),
//   or a group whose normals agree to round-off (HIGHER_ORDER_SHELL_SAME_NORMAL_SINE) keeps each element's own normal,
//   so that a model of flat elements gives the bits it gave before;
// - any other group (with at least one curved element) shares the normalised sum of its elements' unit normals (each
//   turned to the orientation of the group's first element), with equal weights: a design choice of this project (an
//   element's own normal there is already within its interpolation error of the surface's, so a weighting by angle
//   or by area would move the director by less than that error);
// the director of the group then has no stiffness about it at that grid point, exactly, and the automatic constraint
// of singular DOFs removes it there. The elements of other cards (CQUAD4 ...) keep their own normals: at a grid point
// they share with a higher-order shell the two normals differ by an angle of the order of their facets', a stiff pivot.
//
// Thread-safety: immutable once built. Built once per snapshot index (higherOrderShellDirectors).
inline constexpr double HIGHER_ORDER_SHELL_SHARP_EDGE_DEGREES = 30.0;
inline constexpr double HIGHER_ORDER_SHELL_SAME_NORMAL_SINE = 1.0e-12;

class HigherOrderShellDirectors
{
public:
    // The directors of every ShellQuad8 and ShellTria6 of the snapshot of `index`; the elements whose geometry is refused
    // take no part (check() reports them).
    [[nodiscard]] static HigherOrderShellDirectors build(const SnapshotIndex& index);

    // The director (unit, in basic, up to its sign: the caller turns it to its own normal) shared by node `node` of the
    // element of batch row `row` of the card `card` ("ShellQuad8" or "ShellTria6"), or nothing: the element keeps its own
    // normal there.
    [[nodiscard]] std::optional<base::Vec3> sharedDirector(std::string_view card, std::int64_t row,
                                                           std::size_t node) const;

private:
    // Per card, N values per row (row by row, node by node); a zero vector where the element keeps its own normal.
    std::vector<base::Vec3> m_quadrilaterals;
    std::vector<base::Vec3> m_triangles;
};

// The directors of the snapshot of `index`, built on first use and kept by the index (SnapshotIndex::computedOnce).
[[nodiscard]] const HigherOrderShellDirectors& higherOrderShellDirectors(const SnapshotIndex& index);

} // namespace oos::elements
