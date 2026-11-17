#pragma once

#include "oos/elements/mixedHybrid/core/HessianScalar.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"

#include <array>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The nonlinear shell element of the mixed hybrid formulation (document 34 of the reference, section 2.5; M.
// Neunteufel, J. Schöberl, The Hellan-Herrmann-Johnson method for nonlinear shells, Computers & Structures 225 (2019)
// 106109), total Lagrangian on the unknowns of the linear element (ShellElement.hpp), φ = X + u:
// - membrane: the Green strain E_αβ = ½ (∂_α φ · ∂_β φ − a_αβ), interpolated in the Regge space by the same functionals
//   as the linear strain;
// - bending: −(∂_αβ φ · ν̂ − b_αβ), ν̂ the unit normal of the deformed piece, plus the shear term (linear in γ);
// - kink: on each side of an edge, the angle of ν̂ about the deformed tangent t̂, measured from the normal carried by
//   the minimal rotation that takes the tangent to t̂; the angle is followed from the last accepted state (the minimal
//   rotation from the reference tangent is singular when the edge has turned back on itself); the hybrid unknown is
//   the rotation of the edge frame;
// - shear: linear in γ.
// The first and second derivatives of the point measures are computed by forward differentiation (HessianScalar over
// the 15 variables (φ_,1, φ_,2, φ_,11, φ_,22, φ_,12) of a surface point, the 6 variables (φ_,1, φ_,2) of an edge
// point). The linear element is the first variation at u = 0: the tangent at the reference equals the linear
// stiffness. The geometric stiffness of a buckling analysis is the second variation of the measures at u = 0 with the
// stresses of the linear solution (membrane, moments in the element and at the kinks).
//
// Reference: tdnns/nonlinear.rs (bending_measure, edge_angle, NlPre, EdgePt, prepare, evaluate, accept).
//
// Thread-safety: pure functions, except acceptNonlinearState which updates the preparation it is given (one owner).

// The bending measure (covariant 11, 22, 12) of the Kirchhoff part at a point, −(φ_,αβ · ν̂ − b_αβ), with its gradient
// and Hessian over the variables [φ_,1, φ_,2, φ_,11, φ_,22, φ_,12] (3 components each); `b` the reference curvature
// (b_11, b_22, b_12).
// Reference: tdnns/nonlinear.rs::bending_measure
[[nodiscard]] std::array<HessianScalar<15>, 3> bendingMeasure(const std::array<double, 15>& variables,
                                                              const std::array<double, 3>& b);

// The angle of the deformed normal of a side about its deformed edge tangent, from the reference normal `nu` carried
// by the minimal rotation taking the reference tangent `t` to the deformed one:
//   t̂ = unit(φ_,α t̂^α),  ν̂ = unit(φ_,1 × φ_,2),  s = t + t̂,  r = ν − s (s · ν) / (1 + t · t̂),
//   angle = atan2((r × ν̂) · t̂, r · ν̂),
// over the variables [φ_,1, φ_,2] at the edge point; `tv` = dξ / dτ (the reference edge vector).
// Reference: tdnns/nonlinear.rs::edge_angle
[[nodiscard]] HessianScalar<6> edgeAngle(const std::array<double, 6>& variables, const std::array<double, 2>& tv,
                                         const SpaceVector& t, const SpaceVector& nu);

// An edge point of the nonlinear element.
// Reference: tdnns/nonlinear.rs::EdgePt (t_prev, nu_prev, theta_prev, g, xt, tv, len, t, mc, mmu, te, we)
struct NonlinearEdgePoint
{
    // The tangent, the normal of the side and the angle of the side about the edge at the last accepted state (the
    // reference at first): the angle is followed from there, so that the edge may turn by any amount.
    SpaceVector previousTangent{};
    SpaceVector previousNormal{};
    double previousAngle = 0.0;
    PointGeometry g;
    // X_τ and the reference edge vector.
    SpaceVector xt{};
    std::array<double, 2> tv{};
    // |X_τ| and X_τ / |X_τ|.
    double length = 0.0;
    SpaceVector tangent{};
    // (μ · a^1, μ · a^2), μ the unit conormal.
    std::array<double, 2> mc{};
    // m_μμ of every moment function.
    std::vector<double> normalMoments;
    // The canonical edge parameter τ_e and W there.
    double canonicalParameter = 0.0;
    double weight = 0.0;
};

// The data of an element that do not change with its unknowns.
// Reference: tdnns/nonlinear.rs::NlPre (geos, x2, bref, edge, kr, cr, ks)
struct NonlinearPreparation
{
    std::vector<PointGeometry> geometries;
    // X_,11, X_,22, X_,12 at the surface points.
    std::vector<std::array<SpaceVector, 3>> secondDerivatives;
    // b_11, b_22, b_12 at the surface points.
    std::vector<std::array<double, 3>> referenceCurvature;
    // [k][g]
    std::vector<std::vector<NonlinearEdgePoint>> edges;
    // ∫ Rᵀ Ā R (reggeCount²), ∫ Sᵀ D⁻¹ Bᵀ R (momentCount × reggeCount), ∫ γᵀ H γ (unknownCount²).
    Matrix kr;
    Matrix cr;
    Matrix ks;
};

// The precomputed data of an element.
// Reference: tdnns/nonlinear.rs::prepare
[[nodiscard]] NonlinearPreparation prepareNonlinear(const ShellReferenceElement& re, const ShellElementDefinition& def);

// The energy, the internal forces and the tangent of the element at its unknowns x.
// Reference: the tuple of tdnns/nonlinear.rs::evaluate
struct NonlinearEvaluation
{
    double energy = 0.0;
    Vector forces;
    Matrix tangent;
};

// The energy, internal forces and tangent of the element at its unknowns x; `momentTerm` (h) and `membraneTerm` (f_R)
// the thermal terms of the moments and of the membrane (null: no temperature). With `geometric` the result is the
// geometric stiffness instead (in `tangent`; energy and forces zero): the second variation of the measures at the
// reference with the stresses of the state x read as a linear solution.
// Reference: tdnns/nonlinear.rs::evaluate
[[nodiscard]] NonlinearEvaluation evaluateNonlinear(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                    const ShellElementMatrices& matrices,
                                                    const NonlinearPreparation& preparation, std::span<const double> x,
                                                    const Vector* momentTerm, const Vector* membraneTerm,
                                                    bool geometric);

// The gradient of φᵀ·K(x)·φ with respect to the preload unknowns x, K(x) the geometric stiffness evaluateNonlinear
// gives with `geometric`: entry k is φᵀ·K(e_k)·φ, the thermal terms (constant in x) left out. K is linear in the
// stresses of x and they are linear in x, so the gradient is the transpose of the stress maps applied to the products
// of φ with the second variations of the measures: one pass over the points instead of one K per unknown. Used by the
// derivative of a buckling factor through its preload.
[[nodiscard]] Vector geometricStiffnessGradient(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                const ShellElementMatrices& matrices,
                                                const NonlinearPreparation& preparation, std::span<const double> phi);

// Takes the state x as the new start of the angles of the edges (after a converged step).
// Reference: tdnns/nonlinear.rs::accept
void acceptNonlinearState(const ShellReferenceElement& re, const ShellElementDefinition& def,
                          NonlinearPreparation& preparation, std::span<const double> x);

} // namespace oos::elements::mixedHybrid
