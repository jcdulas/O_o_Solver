#pragma once

#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"
#include "oos/elements/mixedHybrid/geometry/Patch.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The geometry of the shell elements from the reading of the mesh (MeshTopology.hpp): the surface of each element, the
// denominator W of its rational map, the moments of the shared edge curves that the isoparametric map of the elements
// takes (document 34 of the reference, 4.1), and the report of the geometry read (document 34, section 9).
//
// Invariants: the surface of an element is a rational patch X = N / W with W = 1 at the corners and, on an edge, the
// denominator of the edge conic, the same from both elements. Thread-safety: plain values and pure functions.

// The surface of one shell element, its axes (x_e, y_e) and the direction of its side G1-G2.
struct ElementSurface
{
    Patch patch;
    std::array<Vector3, 2> axes{};
    Vector3 side{};
};

// The surface of shell `element` from the reading of the mesh: the rational patch of its four grids and edge curves
// with its centre control point fitted (RationalPatch::fitCentre), or the rational triangle with its interior fitted
// inside the rational form (TrianglePatch::fitInteriorRational: the map is then exactly in the rational space of degree
// 4, document 34 of the reference, 4.1). A facet (all its edges raw) keeps its plain patch. An explicit surface
// (a cylinder or a sphere patch, validation) is taken instead, as the rational patch of its corners and edge arcs when
// that is the same surface (asRationalPatch).
// Reference: src/tdnns/analysis.rs::element_surface (its geometric part; the thickness is cornerThicknesses).
[[nodiscard]] ElementSurface elementSurface(const MeshReadingInput& input, const Topology& topology,
                                            std::size_t element, const Patch* explicitSurface = nullptr);

// The four corner values of the thickness field of a shell from the thicknesses at its grids: those of a
// quadrilateral as they are; on a triangle (three values) [t0, t1, t1 + t2 − t0, t2].
// Reference: src/tdnns/analysis.rs::element_surface (its thickness part).
[[nodiscard]] std::array<double, 4> cornerThicknesses(std::size_t cornerCount,
                                                      const std::array<double, 4>& gridThicknesses);

// The denominator W of the rational map of a patch as a polynomial of (ξ, η): Σ_ij w_ij B_i(s) B_j(t) on a
// quadrilateral (biquadratic), Σ_k L_k² + 2 w_k L_k L_k+1 on a triangle (quadratic); 1 on a surface that is not
// rational.
// Reference: src/tdnns/analysis.rs::weight_poly.
[[nodiscard]] Polynomial2 weightPolynomial(const Patch& patch);

// An explicit cylinder or sphere patch as the rational patch of its corners and edge arcs, when that patch is the same
// surface to the round-off (1e-12 relative, on 7 × 7 points): the rational map is then exact. Empty otherwise.
// Reference: src/tdnns/analysis.rs::as_rational.
[[nodiscard]] std::optional<Patch> asRationalPatch(const Patch& patch);

// The moments of the curve of one edge against the Legendre polynomials P_l(2τ − 1), l < k − 1, τ from the lower to
// the higher grid: ∫ P_l W X dτ and ∫ P_l W dτ, read on the first element of the edge.
struct EdgeMoments
{
    std::vector<Vector3> position;
    std::vector<double> weight;
};

// The points of the rule of the edge moments (document 34 of the reference: 14 Gauss points).
inline constexpr std::size_t EDGE_MOMENT_RULE_POINTS = 14;

// The edge moments of every edge of the topology, of degree k (k − 1 moments per edge), with `rulePoints` Gauss points
// on [0, 1]. An edge that no shell has gets none.
// Reference: src/tdnns/analysis.rs::Analysis::new (the block of edge_geo and edge_w).
[[nodiscard]] std::vector<EdgeMoments> edgeMoments(const MeshReadingInput& input, const Topology& topology,
                                                   std::span<const ElementSurface> surfaces,
                                                   std::span<const Polynomial2> weights, std::size_t degree,
                                                   std::size_t rulePoints = EDGE_MOMENT_RULE_POINTS);

// The reading of the geometry of a mesh, edge by edge (document 34 of the reference, section 9): the largest angle
// between the normals of the elements of the edge along it (a fold of the reading where the shell is meant to be
// smooth is a real fold for this element), whether the edge is declared raw or sharp, and its number of elements.
// The reference's report sees nothing of what the reading does in the surface itself: a flat plate whose outline is
// rounded at a corner has no kink. The turn of the outline says it.
// Reference: src/tdnns/analysis.rs::EdgeGeometry (with `sharp`, `outlineTurn` and `given` added).
struct EdgeGeometry
{
    EdgeKey grids{};
    std::size_t elements = 0;
    // The state the reading gave the edge (Topology::states): raw, sharp, or neither (smooth, or an edge of one
    // element), where it comes from, and the angle (radians) measured between the normals of its elements at its
    // grids, which the fold angle is compared with.
    bool raw = false;
    bool sharp = false;
    EdgeStateOrigin origin = EdgeStateOrigin::Outline;
    double angle = 0.0;
    // The largest angle (radians) between the normals of two elements of the edge, along it.
    double kink = 0.0;
    // On an edge of one shell (the outline of the mesh): how far the reading bends the outline in its surface
    // (outlineTurn). Zero on the other edges, whose curve in the surface does not move the outline.
    double outlineTurn = 0.0;
    // The curve of the edge is given by the model (a mid-side grid point), not read from the mesh. The reports of the
    // geometry module leave it false: only the owner of the mid-side grids knows (ShellPart::geometryReport).
    bool given = false;
};

// The tangent (not unit) at grid `from` of the curve of the edge between grids `from` and `to`: from the grid towards
// the control point of the conic of the edge, chord / 2 + 2 b with b the bulge (Surface.hpp, "edge conics").
[[nodiscard]] Vector3 edgeEndTangent(const MeshReadingInput& input, const Topology& topology, std::size_t from,
                                     std::size_t to);

// The largest angle (radians), at the two grids of edge `edge` of one shell, between the curve of the edge and its
// chord, seen along the normal of the corners of that shell: atan2(|t · b|, t · u) with t the end tangent, u the chord
// direction from that grid and b = n × u across the chord in the plane of the shell. Zero on a straight edge and on
// an edge that curves along the normal alone (a circle of a cylinder). The reading has no angle criterion: where the
// outline of the mesh turns at a grid through which its line goes on (a corner that no raw edge declares), the two
// outline edges there are read as one curve, half the turn of the corner away from their chords, and the area of the
// shells changes (document 26 of the reference, section 7, limit 4). Zero on an edge that is not of one shell.
[[nodiscard]] double outlineTurn(const MeshReadingInput& input, const Topology& topology, std::size_t edge);

// Below this sine of the angle between the two edge curves of a shell at one of its corners, the map of the shell is
// not invertible there: the curves are tangent to each other to the round-off (a cusp), or cross (negative: the shell
// folds over itself). The limit is as loose as that of the Jacobian inside the elements: a thin corner is accepted.
inline constexpr double DEGENERATE_CORNER_SINE = 1e-6;

// The sine of the angle at corner `corner` of shell `element` from its edge to the next grid to its edge to the
// previous grid, about the normal of the corners of the shell: (t_next × t_previous) · n / (|t_next| |t_previous|)
// with the end tangents of the two edge curves (edgeEndTangent). Positive on a sound corner whatever the grid the
// element starts from; zero when an edge curve has no tangent there (its control point on the grid).
[[nodiscard]] double cornerSine(const MeshReadingInput& input, const Topology& topology, std::size_t element,
                                std::size_t corner);

// A corner of a shell where its edge curves leave no angle (cornerSine ≤ DEGENERATE_CORNER_SINE).
struct DegenerateCorner
{
    std::size_t element = 0;
    // The position of the corner among the grids of the element.
    std::size_t corner = 0;
    double sine = 0.0;
};

// The degenerate corners of the shells of a mesh, in the order of the shells and of their corners. A mesh line that
// the reading takes round a corner of the outline can come tangent to the other edge of an element there (two
// boundary edges at a right angle beside a triangle whose diagonal ends at the corner): nothing can be computed at
// such a corner (the normal of the surface is 0 / 0), so the element must be refused.
[[nodiscard]] std::vector<DegenerateCorner> degenerateCorners(const MeshReadingInput& input, const Topology& topology);

// The position and the unit normal of the surface of a shell at a point of one of its edges.
using EdgeSurfacePoint = std::pair<Vector3, Vector3>;
// The surface of shell `element` at the parameter τ in [0, 1] of its edge `localEdge`, in the element's own direction
// (from its corner localEdge to the next).
using EdgeSurfaceEvaluator = std::function<EdgeSurfacePoint(std::size_t element, std::size_t localEdge, double tau)>;

// The report of every edge from any evaluator of the elements on their edges, at the parameters `tau` of each element
// (the Gauss points of the edge rule of the elements in the reference, which reads their isoparametric map): for each
// point of one element, the closest point of the other on the edge, and the angle between the two surfaces there,
// whatever their orientations.
// Reference: src/tdnns/analysis.rs::Analysis::geometry_report.
[[nodiscard]] std::vector<EdgeGeometry> geometryReport(const MeshReadingInput& input, const Topology& topology,
                                                       std::span<const double> tau, const EdgeSurfaceEvaluator& point);

// The same report from the patches of the elements, at `pointsPerEdge` Gauss points of each edge (2k + 4 in the
// reference).
[[nodiscard]] std::vector<EdgeGeometry> geometryReport(const MeshReadingInput& input, const Topology& topology,
                                                       std::span<const ElementSurface> surfaces,
                                                       std::size_t pointsPerEdge);

} // namespace oos::elements::mixedHybrid
