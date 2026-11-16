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
// Reference: src/tdnns/analysis.rs::EdgeGeometry (with `sharp` added).
struct EdgeGeometry
{
    EdgeKey grids{};
    std::size_t elements = 0;
    bool raw = false;
    bool sharp = false;
    // The largest angle (radians) between the normals of two elements of the edge, along it.
    double kink = 0.0;
};

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
