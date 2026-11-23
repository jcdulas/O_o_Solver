#pragma once

#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The geometry of a shell element of the mixed hybrid formulation at a point of its reference domain (document 34 of
// the reference, section 2.1): the rational isoparametric map X = P_X / W (ShellElementDefinition.hpp) and its first
// and second derivatives, the covariant and contravariant bases, the normal, the Christoffel symbols, the derivatives
// of the normal, the local frame of the element and the angle of the frame of its constant states.
//
// Notation: a_α = ∂_α X, a_αβ = a_α · a_β, a^α the contravariant basis (a^α · a_β = δ), j = |a_1 × a_2|,
// ν = a_1 × a_2 / j, b_αβ = ∂_αβ X · ν, Γ^λ_αβ = a^λ · ∂_αβ X, ∂_α ν = −b_αβ a^β (Weingarten).
//
// Reference: tdnns/element.rs (Geo, geo_from, rational, element_jets, h1_at, geo_at, geo_q, geo_e).
//
// Thread-safety: plain values and pure functions.

// The geometry at one point.
// Reference: tdnns/element.rs::Geo
struct PointGeometry
{
    SpaceVector x{};
    // The covariant basis a_α = ∂X/∂ξ_α.
    SurfaceFrame a{};
    // The contravariant basis a^α.
    SurfaceFrame ac{};
    SpaceVector nu{};
    // The area element |a_1 × a_2|.
    double j = 0.0;
    // christoffel[l][a][b] = Γ^l_ab = a^l · ∂_ab X (zero at edge points).
    std::array<Matrix22, 2> christoffel{};
    // ∂ν/∂ξ_α (zero at edge points).
    SurfaceFrame dnu{};
    // The local orthonormal frame (e1, e2) of the element (tangentFrameXy of its axes).
    SurfaceFrame e{};
    // The angle from e1 to the frame of the constant states, about ν: that frame is the element axes carried by the
    // minimal rotation of the element normal z = unit(x × y) to ν, so that a rotation of the axes turns it by the same
    // angle everywhere (the constant states do not depend on the grid point an element starts from; document 34 of
    // the reference, section 4.3).
    double phi = 0.0;

    // c[α][I] = e_I · a^α: covariant components to the local frame.
    [[nodiscard]] Matrix22 cov() const;
    // d[α][I] = e_I · a_α: contravariant components to the local frame.
    [[nodiscard]] Matrix22 con() const;
    // K_IJ = e_I · ∂ν/∂s_J (the curvature tensor with the sign of the current element).
    [[nodiscard]] Matrix22 curvature() const;
};

// The jet of the displacement function i at the point.
using JetOfFunction = std::function<ReferenceJet(std::size_t)>;

// The geometry from the isoparametric map X = Σ_i x_i N_i, given the jets of the displacement functions (value, ∂ξ,
// ∂η, ∂ξξ, ∂ξη, ∂ηη). With `second` false (edge points) the Christoffel symbols and the derivatives of the normal are
// left at zero. The angle φ of the constant states: with z = unit(x × y) and the minimal rotation from z to ν,
// f1 = x − (z + ν)(ν · x) / (1 + z · ν), φ = atan2((e1 × f1) · ν, e1 · f1).
// Reference: tdnns/element.rs::geo_from
[[nodiscard]] PointGeometry geometryFrom(std::span<const SpaceVector> mapCoefficients, const JetOfFunction& jet,
                                         const SurfaceFrame& axes, bool second, const SpaceVector& origin = {});

// The jet of f = N / W from those of N and W:
//   f = N / W,  f_α = (N_α − f W_α) / W,  f_αβ = (N_αβ − f_α W_β − f_β W_α − f W_αβ) / W.
// Reference: tdnns/element.rs::rational
[[nodiscard]] ReferenceJet rationalJet(const ReferenceJet& n, const ReferenceJet& w);

// The displacement functions N_i / W of an element at its surface points [q · h1Count + i] (jets) and edge points
// [k][g · h1Count + i] (value, ∂ξ, ∂η), and W at the edge points [k][g]: the rational isoparametric space of a patch of
// denominator W.
// Reference: the tuple of tdnns/element.rs::element_jets
struct ElementJets
{
    std::vector<ReferenceJet> volume;
    std::vector<std::vector<EdgeJet>> edge;
    std::vector<std::vector<double>> edgeWeight;
};

// Reference: tdnns/element.rs::element_jets
[[nodiscard]] ElementJets elementJets(const ShellReferenceElement& re, const Polynomial2& weight);

// The displacement functions N_i / W at any reference point (jets).
// Reference: tdnns/element.rs::h1_at
[[nodiscard]] std::vector<ReferenceJet> h1At(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                             double xi, double eta);

// The geometry at any reference point (ξ, η).
// Reference: tdnns/element.rs::geo_at
[[nodiscard]] PointGeometry geometryAt(const ShellReferenceElement& re, const ShellElementDefinition& def, double xi,
                                       double eta);

// The geometry at surface point q.
// Reference: tdnns/element.rs::geo_q
[[nodiscard]] PointGeometry volumeGeometry(const ShellElementDefinition& def, std::size_t h1Count, std::size_t q);

// The geometry at point g of edge k (no second derivatives).
// Reference: tdnns/element.rs::geo_e
[[nodiscard]] PointGeometry edgeGeometry(const ShellElementDefinition& def, std::size_t h1Count, std::size_t k,
                                         std::size_t g);

} // namespace oos::elements::mixedHybrid
