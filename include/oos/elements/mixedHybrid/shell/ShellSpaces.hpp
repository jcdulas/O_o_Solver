#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The reference spaces of the shell of the mixed hybrid formulation (M. Neunteufel, J. Schöberl, The
// Hellan-Herrmann-Johnson and TDNNS methods for linear and nonlinear shells, Computers & Structures 305 (2024)
// 107543; spec section 3.1):
//
// | space          | triangle         | quadrilateral                             | components                    |
// |----------------|------------------|-------------------------------------------|-------------------------------|
// | displacement   | P_k              | Q_k                                       | 1 (H1, per Cartesian axis)    |
// | shear          | P_p²             | Q_(p,p+1) × Q_(p+1,p)                     | covariant (ξ, η), H(curl)     |
// | moments        | P_p symmetric    | the quadrilateral HHJ space, 3p²+10p+5    | contravariant (11, 22, 12)    |
// | membrane       | P_p symmetric    | 11: Q_(p,p+1), 22: Q_(p+1,p), 12: Q_(p,p) | covariant (11, 22, 12), Regge |
//
// with p = k − 1. Everything is a polynomial of the reference coordinates (ξ, η) on the square [−1, 1]² or on the
// triangle (−1, −1), (1, −1), (−1, 1) (geometry/ReferenceShapes.hpp; edge k from corner k to corner k + 1, its
// parameter τ ∈ [0, 1]).
//
// A space with degrees of freedom (displacement, shear) is the span of its generators; its basis is the dual of the
// degrees of freedom: the values at the corners (displacement only), then the moments of the traces on the edges
// against the Legendre polynomials P_l(2τ − 1), edge after edge, then a basis of their common null space (the
// interior functions). A function of an edge has a zero trace on the other edges and an interior function zero traces
// on every edge, so that the functions of an edge are shared by the elements of that edge. The interior functions
// are an L2-orthonormal basis of the null space: any such basis gives the same element up to an orthogonal change of
// its interior unknowns. The broken spaces (moments, membrane strains) are given by an L2-orthonormal basis.
//
// The quadrilateral moment space is the Hellan-Herrmann-Johnson space of the square built from Airy-type stress
// functions (fields with two derivatives of products of a linear function of one edge and bubbles), of dimension
// 3p² + 10p + 5; the generators below follow the construction of Neunteufel and Schöberl (2024) and are checked by
// their dimension.
//
// Symmetric eigenvalue problems go through jacobiEigen (core/JacobiEigen.hpp), pseudo-inverses through pseudoInverse.
//
// Reference: tdnns/space.rs (the spaces; the reference element is ShellReferenceElement.hpp).
//
// Thread-safety: pure functions.

// A function of a space: its components (1 for the displacement, (ξ, η) for the shear, (11, 22, 12) for the
// symmetric tensors).
// Reference: tdnns/space.rs::Field
using Field = std::vector<Polynomial2>;

// Where a basis function is attached.
// Reference: tdnns/space.rs::Kind
struct FunctionKind
{
    enum class Place
    {
        // The value at a corner (displacement only).
        Vertex,
        // The moment `moment` of the trace on edge `entity`.
        Edge,
        // Zero traces on every edge.
        Interior
    };

    Place place = Place::Interior;
    // The corner (Vertex) or the edge (Edge); zero for an interior function.
    std::size_t entity = 0;
    // The Legendre degree l of an edge function; zero otherwise.
    std::size_t moment = 0;

    [[nodiscard]] static FunctionKind vertex(std::size_t corner) { return {Place::Vertex, corner, 0}; }
    [[nodiscard]] static FunctionKind edge(std::size_t edgeIndex, std::size_t l) { return {Place::Edge, edgeIndex, l}; }
    [[nodiscard]] static FunctionKind interior() { return {Place::Interior, 0, 0}; }

    [[nodiscard]] bool isInterior() const { return place == Place::Interior; }

    friend bool operator==(const FunctionKind&, const FunctionKind&) = default;
};

// A basis with the place of each function.
struct SpaceBasis
{
    std::vector<Field> functions;
    std::vector<FunctionKind> kinds;
};

// An interior functional of the membrane (Regge) interpolation: the moment of covariant component `component`
// (0: 11, 1: 22, 2: 12) against ξ^i η^j.
// Reference: the (usize, usize, usize) of tdnns/space.rs::regge_interior
struct ReggeMoment
{
    std::size_t component = 0;
    std::size_t i = 0;
    std::size_t j = 0;

    friend bool operator==(const ReggeMoment&, const ReggeMoment&) = default;
};

// The displacement space of degree k ≥ 2 (Q_k on the square, P_k on the triangle), one component: the corner
// functions, the k − 1 functions of each edge (moments l = 0 … k − 2), then the interior functions.
// Reference: tdnns/space.rs::h1_space
[[nodiscard]] SpaceBasis h1Space(std::size_t cornerCount, std::size_t degree);

// The shear space of degree p (covariant components ξ, η): the full P_p² on the triangle, Q_(p,p+1) × Q_(p+1,p) on the
// square. Degrees of freedom of the edges: the moments of t̂ · γ (t̂ the reference edge vector, corner k + 1 − corner
// k) against P_l(2τ − 1), l = 0 … p; then the interior functions.
// Reference: tdnns/space.rs::hcurl_space
[[nodiscard]] SpaceBasis hcurlSpace(std::size_t cornerCount, std::size_t degree);

// The spaces of an element whose neighbours across some edges have a lower degree (a transition element): the
// functions of `full` = h1Space(cornerCount, degree) whose trace on edge e is a polynomial of degree edgeDegrees[e]
// at most, 2 ≤ edgeDegrees[e] ≤ degree, with the corner values and the moments l = 0 … edgeDegrees[e] − 2 of each
// edge as degrees of freedom; and the functions of `full` = hcurlSpace(cornerCount, degree) whose tangential trace on
// edge e has the degree edgeDegrees[e] at most, with its moments l = 0 … edgeDegrees[e]. An edge then carries the
// unknowns of the element of the lower degree, and the two elements share them one for one. The interior functions
// are those of the full spaces. With every edge at the degree of the element they are the full spaces. Not in the
// reference, whose elements have one degree.
//
// The displacement: a trace of degree n on an edge is q = Σ_j c_j P_j(2τ − 1), j ≤ n, whose moments are
// m_j = c_j / (2j + 1). The corner values a (τ = 0) and b (τ = 1) and the moments j ≤ n − 2 give its two last
// coefficients: with s = (−1)^(n−1), r_a = a − Σ_{j ≤ n−2} c_j (−1)^j and r_b = b − Σ_{j ≤ n−2} c_j,
//     c_(n−1) = (r_b + s r_a) / 2,   c_n = (r_b − s r_a) / 2,
// so that the functions of the corners and of the kept moments each take a share of the functions of the moments
// n − 1 and n of the full space; the functions of its higher moments are left out. The shear: the functions of the
// moments above the degree of the edge are left out (a trace of that degree has none).
[[nodiscard]] SpaceBasis h1TransitionSpace(const SpaceBasis& full, std::size_t cornerCount, std::size_t degree,
                                           std::span<const std::size_t> edgeDegrees);
[[nodiscard]] SpaceBasis hcurlTransitionSpace(const SpaceBasis& full, std::size_t degree,
                                              std::span<const std::size_t> edgeDegrees);

// The moments of a transition element: the functions S of `full` (the moment space of degree p = `degree` with its
// three constants last, splitConstants of hhjSpace) that do no work on the functions the element leaves out on its
// edges of a lower degree:
// - the rotation about edge e has the degree edgeDegrees[e]: the normal-normal trace n̂ · S n̂ on the edge (n̂ the unit
//   normal of the edge in the reference domain), which works with it (ShellElement.hpp: −∫ m_μμ (β · μ − a_T) ds), has
//   that degree at most;
// - the trace of the displacement on edge e has the degree edgeDegrees[e] + 1: for each function φ of the
//   displacement space `displacement` = h1Space(cornerCount, p + 1) of a moment l ≥ edgeDegrees[e] of edge e (the
//   ones h1TransitionSpace leaves out), the work of the moments on the deflection φ,
//       ∫ S : ∇̂²φ dÂ − Σ_f ∫_f (n̂ · S n̂) ∂φ/∂n̂ dŝ  (f the edges of the reference domain),
//   is zero.
// The element condensed on the functions left out would be the same: their work would vanish because nothing holds
// them, and the moments that do no work on them are those that remain. Without it the moments, too many for the
// displacement functions that remain, would make the element too stiff in bending. The work above is the one of the
// reference coordinates: on an affine element it is the work of the moments on the surface up to a constant factor,
// the part of the conormal derivative of φ along the edge having no work of its own on a moment of that degree
// (φ has zero corner values and zero moments of a lower degree, so that its derivative along the edge is orthogonal to
// the polynomials of degree edgeDegrees[e]). The result is an L2-orthonormal basis of the functions orthogonal to
// the constants, then the three constants (which do no work on φ). With every edge at `degree` it is `full`. Not in
// the reference.
[[nodiscard]] std::vector<Field> momentTransitionSpace(const std::vector<Field>& full,
                                                       const SpaceBasis& displacement, std::size_t cornerCount,
                                                       std::size_t degree, std::span<const std::size_t> edgeDegrees);

// The membrane strains of a transition element and the functionals of their interpolation. `full` is the membrane space
// of degree p = `degree` with its three constants last (splitConstants of reggeSpace), `displacement` the space
// h1Space(cornerCount, p + 1), and edgeDegrees[e] the degree of the tangential-tangential trace on edge e
// (the degree of the displacement trace there, less one). The functionals of the full space, F_i, are the moments of
// the tangential-tangential trace of each edge against P_l(2τ − 1), l = 0 … p, then the moments of the components of
// reggeInterior against polynomials of zero mean, then the three moments of the constants.
//
// Each discarded displacement function φ_i (edge moment l ≥ edgeDegrees[e]) takes its two strain directions with it:
// with ε(φ_i c) = sym(∇̂ φ_i ⊗ c) for the two constant covectors c, Π ε the strain of `full` that has the functionals
// of ε and V = span Π ε(φ_i c), the strains are
//       span{Π ε(u c), u a displacement function kept} ⊕ {ε in `full` orthogonal to every Π ε(u c), u in `full`},
// two functions less for each φ_i, and the functionals the combinations of F_i (the edges and the inside) that are zero
// on V. This pairs the membrane interpolation with the displacement space on every geometry.
struct MembraneTransition
{
    // The strains, L2-orthonormal and orthogonal to the three constants, then the constants.
    std::vector<Field> strains;
    // The functionals: one row per functional kept but the constants, each column the one of F_i in their order (the
    // edges, then inside).
    Matrix functionals;
};
// With every edge at `degree`, `strains` is `full` and `functionals` is empty.
[[nodiscard]] MembraneTransition membraneTransition(const std::vector<Field>& full, const SpaceBasis& displacement,
                                                    std::size_t cornerCount, std::size_t degree,
                                                    std::span<const std::size_t> edgeDegrees);

// The broken moment space of degree p (contravariant components 11, 22, 12): P_p symmetric on the triangle, the
// quadrilateral HHJ space (dimension 3p² + 10p + 5) on the square. L2-orthonormal basis on the reference domain.
// Reference: tdnns/space.rs::hhj_space
[[nodiscard]] std::vector<Field> hhjSpace(std::size_t cornerCount, std::size_t degree);

// The membrane strain (Regge) space of degree p (covariant components 11, 22, 12): P_p symmetric on the triangle;
// 11 in Q_(p,p+1), 22 in Q_(p+1,p), 12 in Q_(p,p) on the square. L2-orthonormal basis.
// Reference: tdnns/space.rs::regge_space
[[nodiscard]] std::vector<Field> reggeSpace(std::size_t cornerCount, std::size_t degree);

// An orthonormal basis of a tensor space (components 11, 22, 12) that contains the three reference constants,
// reordered: an orthonormal basis of the complement of the constants first, then the three constants (E11, E22, E12).
// In an element the three constants are replaced by the constant states of its local frame (document 34 of the
// reference, section 4.3; ShellElement.hpp). Throws InternalError when a constant is not in the space.
// Reference: tdnns/space.rs::split_constants
[[nodiscard]] std::vector<Field> splitConstants(const std::vector<Field>& basis, std::size_t cornerCount);

// The interior functionals of the membrane interpolation: moments of component c against ξ^i η^j, without the three
// moments of the constants (c, 0, 0), which an element replaces by the work of the three constant membrane forces of
// its local frame. Triangle: every component, i + j < p, i + j > 0. Square: (12, i ≤ p, j ≤ p), (22, i < p, j ≤ p),
// (11, i ≤ p, j < p), in that order, without i = j = 0.
// Reference: tdnns/space.rs::regge_interior
[[nodiscard]] std::vector<ReggeMoment> reggeInterior(std::size_t cornerCount, std::size_t degree);

} // namespace oos::elements::mixedHybrid
