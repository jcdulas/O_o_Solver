#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSpaces.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// Everything a shell element of the mixed hybrid formulation needs from its reference domain, tabulated at its
// quadrature points: the bases of its spaces (ShellSpaces.hpp) with the place of each function, their values and
// derivatives at the points of the surface rule and of the edge rules, and the functionals of the membrane (Regge)
// interpolation. Built once per (corner count, degree k ≥ 2, points per direction) — the formulation uses k = 4 and
// 2k + 4 = 12 points per direction (the integrands of an element that is not affine are rational, and this rule
// converges on them exponentially; spec section 3).
//
// Orderings (those of the reference implementation, which the element and the global numbering follow):
// - the surface rule: referenceSurfaceRule(cornerCount, n), point q;
// - the edge rule: unitIntervalRule(n) on the parameter τ ∈ [0, 1] of edge k (corner k to corner k + 1), point g;
// - the tables [q · count + i]: point after point, the functions of the basis in their order;
// - the edge tables [k][g · count + i]: edge k, then point after point.
// The moment and membrane bases end with their three reference constants (splitConstants): an element replaces them by
// the constant states of its local frame.
//
// Mapping of the members to the reference (tdnns/space.rs::RefElement): cornerCount nv, degree k, shearDegree p,
// h1Count n1, hcurlCount nc, momentCount ns, reggeCount nr, h1Kinds h1_kind, hcurlKinds hc_kind, points pts, weights
// wts, h1, hcurl hc, moments hhj, regge rg, edgeParameters etau, edgeWeights ewts, h1Edge h1e, hcurlEdge hce,
// momentsEdge hhje, reggeInteriorMoments rg_int, reggeTest rg_test, reggeEdge rge, h1Basis h1_b, hcurlBasis hc_b,
// momentBasis hhj_b, reggeBasis rg_b.
//
// Thread-safety: immutable once built. shellReferenceElement() builds each table once, under a lock, and hands out a
// reference that stays valid for the life of the program and may be read from any thread.
// Reference: tdnns/space.rs::RefElement

// The bases of the reference element at one reference point where the results of every element are read (its centre
// or a corner): the values the basis polynomials give there, kept so that they are evaluated once for all the
// elements of the shape and all their cases.
struct ReferencePointValues
{
    ReferencePoint point{};
    // [i]: value, ∂ξ, ∂η, ∂ξξ, ∂ξη, ∂ηη of the displacement functions (before the division by an element's weight).
    std::vector<ReferenceJet> h1;
    // [i]: g_ξ, g_η.
    std::vector<std::array<double, 2>> hcurl;
    // [j]: S^11, S^22, S^12.
    std::vector<std::array<double, 3>> moments;
    // [m]: R_11, R_22, R_12.
    std::vector<std::array<double, 3>> regge;
};

// The displacement functions of an element tabulated on a rule, without its other spaces: what the choice of the rule
// of an element (chooseShellPoints) reads. The reference element of a rule holds the same tables.
struct ShellDisplacementTables
{
    std::size_t cornerCount = 0;
    std::size_t h1Count = 0;
    // The surface rule and the values of the functions at its points (h1 [q · h1Count + i]: value, ∂ξ, ∂η, ∂ξξ, ∂ξη,
    // ∂ηη).
    std::vector<std::array<double, 2>> points;
    std::vector<double> weights;
    std::vector<ReferenceJet> h1;
    // The edge rule and the values of the functions at its points ([k][g · h1Count + i]: value, ∂ξ, ∂η).
    std::vector<double> edgeParameters;
    std::vector<double> edgeWeights;
    std::vector<std::vector<EdgeJet>> h1Edge;
};

struct ShellReferenceElement
{
    std::size_t cornerCount = 0;
    // The degree k of the displacement.
    std::size_t degree = 0;
    // The degree p = k − 1 of the shear, moments, membrane strains and edge rotations.
    std::size_t shearDegree = 0;
    // The degree of the trace of the displacement on each edge: `degree`, or less on an edge that the element shares
    // with one of a lower degree (a transition element: h1TransitionSpace, hcurlTransitionSpace, momentTransitionSpace
    // and membraneTransition). The trace of the shear, the rotation about the edge, the normal-normal trace of the
    // moments and the tangential-tangential trace of the membrane strains then have the degree edgeDegrees[e] − 1.
    std::vector<std::size_t> edgeDegrees;
    std::size_t h1Count = 0;
    std::size_t hcurlCount = 0;
    std::size_t momentCount = 0;
    std::size_t reggeCount = 0;
    std::vector<FunctionKind> h1Kinds;
    std::vector<FunctionKind> hcurlKinds;
    // The surface rule.
    std::vector<std::array<double, 2>> points;
    std::vector<double> weights;
    // [q · h1Count + i]: value, ∂ξ, ∂η, ∂ξξ, ∂ξη, ∂ηη.
    std::vector<ReferenceJet> h1;
    // [q · hcurlCount + i]: g_ξ, g_η, ∂ξ g_ξ, ∂η g_ξ, ∂ξ g_η, ∂η g_η.
    std::vector<std::array<double, 6>> hcurl;
    // [q · momentCount + j]: S^11, S^22, S^12.
    std::vector<std::array<double, 3>> moments;
    // [q · reggeCount + m]: R_11, R_22, R_12.
    std::vector<std::array<double, 3>> regge;
    // The edge rule on [0, 1].
    std::vector<double> edgeParameters;
    std::vector<double> edgeWeights;
    // [k][g · h1Count + i]: value, ∂ξ, ∂η at the edge points.
    std::vector<std::vector<EdgeJet>> h1Edge;
    // [k][g · hcurlCount + i]: g_ξ, g_η.
    std::vector<std::vector<std::array<double, 2>>> hcurlEdge;
    // [k][g · momentCount + j]: S^11, S^22, S^12.
    std::vector<std::vector<std::array<double, 3>>> momentsEdge;
    // The membrane interpolation: its interior functionals and their test polynomials of zero mean at the surface
    // points [q · reggeInteriorMoments.size() + m]. The raw functionals are the moments of the tangential-tangential
    // trace of each edge against P_l(2τ − 1), l = 0 … shearDegree, then those, then the three constants; those of the
    // interpolation are the raw ones, or their combinations `membraneCombination` (membraneTransition) when it is not
    // empty.
    std::vector<ReggeMoment> reggeInteriorMoments;
    std::vector<double> reggeTest;
    Matrix membraneCombination;
    // [k][g · reggeCount + m]: the membrane basis at the edge points (covariant 11, 22, 12).
    std::vector<std::vector<std::array<double, 3>>> reggeEdge;
    // The basis polynomials, for values away from the quadrature points.
    std::vector<Field> h1Basis;
    std::vector<Field> hcurlBasis;
    std::vector<Field> momentBasis;
    std::vector<Field> reggeBasis;
    // The bases at the centre, then at the corners in their order.
    std::vector<ReferencePointValues> outputPoints;

    // The bases at (ξ, η) when it is one of outputPoints (the same coordinates, to the bit), null otherwise: the
    // values the basis polynomials give there.
    [[nodiscard]] const ReferencePointValues* valuesAt(double xi, double eta) const noexcept;

    // The rotation unknowns of edge e (its coefficients on the Legendre polynomials P_l, l < rotationCount(e)), and
    // the rank of its first one among the rotation unknowns of the element, which follow the shear unknowns edge
    // after edge.
    [[nodiscard]] std::size_t rotationCount(std::size_t edge) const { return edgeDegrees[edge]; }
    // The functionals of the membrane interpolation from the raw ones (the combination, then the constants), and the
    // weights of the raw ones from weights on them.
    [[nodiscard]] Vector membraneFunctionals(Vector raw) const;
    [[nodiscard]] Vector rawMembraneWeights(const Vector& weights) const;
    [[nodiscard]] std::size_t firstRotation(std::size_t edge) const;
    [[nodiscard]] std::size_t rotationUnknownCount() const { return firstRotation(cornerCount); }

    // The reference element of `cornerCount` corners (3 or 4) for the displacement degree k ≥ 2, with
    // `pointsPerDirection` points per direction on the surface and on the edges, and the degree of each edge
    // (`edgeDegrees`, each from 2 to k; empty: k on every edge). Checks the unisolvence of the membrane functionals
    // on the reference domain (InternalError otherwise); an element checks its own.
    [[nodiscard]] static ShellReferenceElement build(std::size_t cornerCount, std::size_t degree,
                                                     std::size_t pointsPerDirection,
                                                     std::span<const std::size_t> edgeDegrees = {});
};

// The displacement tables of (cornerCount, degree, pointsPerDirection, edgeDegrees), built on the first call and
// shared. Safe to call from several threads.
[[nodiscard]] const ShellDisplacementTables& shellDisplacementTables(std::size_t cornerCount, std::size_t degree,
                                                                     std::size_t pointsPerDirection,
                                                                     std::span<const std::size_t> edgeDegrees = {});

// The points per direction of the rules of an element of degree k: 2k + 4 (spec section 3).
[[nodiscard]] constexpr std::size_t shellPointsPerDirection(std::size_t degree)
{
    return 2 * degree + 4;
}

// The reference element of (cornerCount, degree, pointsPerDirection, edgeDegrees), built on the first call and
// shared: every later call with the same arguments returns the same object. Safe to call from several threads.
[[nodiscard]] const ShellReferenceElement& shellReferenceElement(
    std::size_t cornerCount, std::size_t degree, std::size_t pointsPerDirection,
    std::span<const std::size_t> edgeDegrees = {});

} // namespace oos::elements::mixedHybrid
