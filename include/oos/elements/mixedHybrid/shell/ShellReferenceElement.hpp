#pragma once

#include "oos/elements/mixedHybrid/shell/ShellSpaces.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <cstddef>
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
struct ShellReferenceElement
{
    std::size_t cornerCount = 0;
    // The degree k of the displacement.
    std::size_t degree = 0;
    // The degree p = k − 1 of the shear, moments, membrane strains and edge rotations.
    std::size_t shearDegree = 0;
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
    // points [q · reggeInteriorMoments.size() + m].
    std::vector<ReggeMoment> reggeInteriorMoments;
    std::vector<double> reggeTest;
    // [k][g · reggeCount + m]: the membrane basis at the edge points (covariant 11, 22, 12).
    std::vector<std::vector<std::array<double, 3>>> reggeEdge;
    // The basis polynomials, for values away from the quadrature points.
    std::vector<Field> h1Basis;
    std::vector<Field> hcurlBasis;
    std::vector<Field> momentBasis;
    std::vector<Field> reggeBasis;

    // The reference element of `cornerCount` corners (3 or 4) for the displacement degree k ≥ 2, with
    // `pointsPerDirection` points per direction on the surface and on the edges. Checks the unisolvence of the
    // membrane functionals on the reference domain (InternalError otherwise); an element checks its own.
    [[nodiscard]] static ShellReferenceElement build(std::size_t cornerCount, std::size_t degree,
                                                     std::size_t pointsPerDirection);
};

// The points per direction of the rules of an element of degree k: 2k + 4 (spec section 3).
[[nodiscard]] constexpr std::size_t shellPointsPerDirection(std::size_t degree)
{
    return 2 * degree + 4;
}

// The reference element of (cornerCount, degree, pointsPerDirection), built on the first call and shared: every later
// call with the same arguments returns the same object. Safe to call from several threads.
[[nodiscard]] const ShellReferenceElement& shellReferenceElement(std::size_t cornerCount, std::size_t degree,
                                                                 std::size_t pointsPerDirection);

} // namespace oos::elements::mixedHybrid
