#pragma once

#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// What a shell element of the mixed hybrid formulation needs besides its reference element (ShellReferenceElement):
// its rational isoparametric map, its section, its offset, its axes and the direction of its edges. A view: it holds
// no data, the owner (ShellElementData, or the caller) keeps every array alive while the view is used.
//
// The map (document 34 of the reference, section 4.1; spec section 3.1): the surface of the element is the rational
// patch X = P_X / W, P_X = Σ_i x_i N_i on the displacement basis N_i of the reference element (Q_k or P_k), W the
// weight polynomial of the patch (1 at the corners; on an edge the denominator of the conic of that edge, the same
// from both elements; W = 1 for a polynomial map). The displacement functions of the element are N_i / W, so that a
// rigid motion ω × X is in the space. The coefficients x_i follow the dual basis: the corner positions, then the
// Legendre moments of X W along each edge in the element's own direction, then the L2 projection of the interior
// (ShellElementData.hpp, mapCoefficients).
//
// Edge signs: +1 where the element runs edge k (corner k to corner k + 1) from its lower to its higher grid point (the
// canonical direction of the edge), −1 otherwise. The rotation about edge k is a_T = s_k â(τ_e) / (|X_τ| W²), â the
// edge unknown in the canonical parameter τ_e (document 34 of the reference, section 4.2).
//
// Reference: tdnns/element.rs::ElemDef. Members: mapCoefficients xg, weight w, volumeJets h1v, edgeJets h1ev,
// edgeWeights wedge, section, offset, axes, edgeSigns edge_sign, kirchhoff, mapReferenceConstants (the reference's flag
// of the published spaces).
//
// Thread-safety: a view of immutable data; may be read from several threads.
struct ShellElementDefinition
{
    // The coefficients x_i of the map P_X on the displacement basis (one per function, its own orientation of the
    // edges).
    std::span<const SpaceVector> mapCoefficients;
    // The denominator W of the rational map (the constant 1 for a polynomial map).
    const Polynomial2* weight = nullptr;
    // The displacement functions N_i / W at the surface points [q · h1Count + i] (elementJets).
    std::span<const ReferenceJet> volumeJets;
    // The same at the edge points [k][g · h1Count + i]: value, ∂ξ, ∂η.
    std::span<const std::vector<EdgeJet>> edgeJets;
    // W at the edge points [k][g].
    std::span<const std::vector<double>> edgeWeights;
    const ShellSection* section = nullptr;
    // The offset of the section from the surface of the grid points, along the normal.
    double offset = 0.0;
    // The element axes (x, y) of the local frame (tangentFrameXy).
    SurfaceFrame axes{};
    // s_k = +1 where the element runs edge k from its lower to its higher grid point, −1 otherwise.
    std::span<const double> edgeSigns;
    // No transverse shear flexibility: the shear unknowns are held by the caller (γ = 0).
    bool kirchhoff = false;
    // True: the three reference constants of the moment and membrane spaces are mapped like the other functions (the
    // spaces of the publications, for a check against them); false (the formulation): they are replaced by the
    // constant states of the local frame (document 34 of the reference, section 4.3).
    bool mapReferenceConstants = false;
};

} // namespace oos::elements::mixedHybrid
