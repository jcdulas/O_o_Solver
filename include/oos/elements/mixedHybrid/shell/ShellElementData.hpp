#pragma once

#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementLoads.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// One shell element of the mixed hybrid formulation in an analysis, and what is computed from it alone (the parts of
// the reference's analysis that need no global model): its rotation as a linear map of its unknowns, the coefficients
// of its isoparametric map, the local vectors of the rigid motions, the condensation of its interior unknowns and its
// geometric stiffness.
//
// The local unknowns are those of ShellElement.hpp (3 per displacement function, the shear functions, the edge
// rotations, shellRotationUnknown). The global unknown of local unknown i is dofs[i], with the sign signs[i]: local =
// sign × global. The integrator fills dofs and signs (the reference: grid translations, edge moments 3 l + c with the
// sign (−1)^l where the element runs the edge against its canonical direction, edge shear l with the sign +1 for the
// first element of the edge and −1 for the others (the joint rule γ_L · t_L + γ_R · t_R = 0) times that same (−1)^l,
// edge rotations without sign, then the interior unknowns of the element).
//
// Reference: tdnns/analysis.rs (ShellElem, rotation_rows_of, InteriorProjection, Condensed, the isoparametric map and
// the rigid modes of Analysis::new and Analysis::rigid_modes, the element part of Analysis::geometric_stiffness).
//
// Thread-safety: ShellElementData is a plain value, immutable once built; the functions are pure.

// A shell element in an analysis.
// Reference: tdnns/analysis.rs::ShellElem. Members: cornerCount nv, nodes, mapCoefficients xg, section, offset, axes,
// edgeSigns edge_sign, edges, kirchhoff, matrices mat, dofs, signs sg, thermalMoments thermal, thermalMembrane
// thermal_fr, mapReferenceConstants (the reference's flag of the published spaces), weight w, volumeJets h1v,
// edgeJets h1ev, edgeWeights wedge. (The copies xref, yref of the axes are left out.)
struct ShellElementData
{
    std::size_t cornerCount = 0;
    // The degree k of its displacement and the degree of the trace on each of its edges (the lowest degree of the
    // elements of the edge; `degree` on every edge where the elements all have one degree): what its reference
    // element is built with (ShellReferenceElement::edgeDegrees). Not in the reference, whose elements have one degree.
    std::size_t degree = 0;
    std::vector<std::size_t> edgeDegrees;
    // The grid points of the corners (numbers of the caller).
    std::vector<std::size_t> nodes;
    // The coefficients of the isoparametric map (the element's own orientation of the edges).
    std::vector<SpaceVector> mapCoefficients;
    ShellSection section;
    double offset = 0.0;
    SurfaceFrame axes{};
    std::vector<double> edgeSigns;
    // The edges of the element (numbers of the caller), edge k from corner k to corner k + 1.
    std::vector<std::size_t> edges;
    bool kirchhoff = false;
    ShellElementMatrices matrices;
    std::vector<std::size_t> dofs;
    std::vector<double> signs;
    // The thermal term h of the moments and f_R of the membrane, per load case (empty without temperature).
    std::vector<std::optional<Vector>> thermalMoments;
    std::vector<std::optional<Vector>> thermalMembrane;
    bool mapReferenceConstants = false;
    // The denominator of the rational map and the displacement functions at the points (elementJets).
    Polynomial2 weight;
    std::vector<ReferenceJet> volumeJets;
    std::vector<std::vector<EdgeJet>> edgeJets;
    std::vector<std::vector<double>> edgeWeights;
    // The points per direction of the rule those tables are on (0: no table). An element is read with the reference
    // element of that rule, and its matrices are integrated on it.
    std::size_t pointsPerDirection = 0;

    // The view of the element for the kernels (valid while this object lives and is not changed).
    // Reference: tdnns/analysis.rs::ShellElem::def
    [[nodiscard]] ShellElementDefinition definition() const;
    // The element's unknowns from the model's: signs[i] u[dofs[i]].
    // Reference: tdnns/analysis.rs::ShellElem::local
    [[nodiscard]] Vector local(std::span<const double> u) const;
};

// The rotation at a reference point (ξ, η) as a linear map of the element's unknowns (3 × unknownCount):
// θ = ν × β plus the in-plane rotation of the surface about ν, ω = ½ (e2 · ∂_1 u − e1 · ∂_2 u) (∂_I the derivative
// along e_I), β = −(∇u)ᵀ ν + γ. At a corner it is the corner rotation the mean rotation of a grid point is made of.
// Reference: tdnns/analysis.rs::rotation_rows_of
[[nodiscard]] Matrix rotationRowsOf(const ShellReferenceElement& re, const ShellElementDefinition& def, double xi,
                                    double eta);

// θ = rotationRowsOf · x at (ξ, η).
// Reference: tdnns/analysis.rs::Analysis::rotation (its element part)
[[nodiscard]] SpaceVector rotationAt(const ShellReferenceElement& re, const ShellElementDefinition& def, double xi,
                                     double eta, std::span<const double> x);

// The interior part of a field on the displacement space: the L2 projection (reference measure) of the field on the
// interior functions, the corner and edge coefficients given: G c = ∫ N_interior (f − Σ_boundary N_i x_i), G the
// Gram matrix of the interior functions.
// Reference: tdnns/analysis.rs::InteriorProjection
struct InteriorProjection
{
    // The displacement functions that are interior, in their order.
    std::vector<std::size_t> interior;
    Matrix gramInverse;

    // Throws InternalError when the Gram matrix is singular (a bug of the reference element).
    [[nodiscard]] static InteriorProjection build(const ShellReferenceElement& re);
};

// The points per direction of the rule of the edge moments of the map.
// Reference: tdnns/analysis.rs::EDGE_RULE
inline constexpr std::size_t MAP_EDGE_POINTS = 14;

// The position X(ξ, η) of the surface of an element (its patch).
using SurfaceAt = std::function<SpaceVector(double, double)>;

// The Legendre moments l = 0 … k − 2 of X W along edge `edge` of an element, k = `degree` the degree of the edge,
// in the canonical direction of the edge (from its lower grid point; `forward` when the element runs the edge that
// way): ∫_0^1 X W P_l(2τ_e − 1) dτ_e by the rule of MAP_EDGE_POINTS points. The reference computes them once per
// edge from its first element, and every element of the edge takes them (mapCoefficients), so that the edge is the
// same curve from both sides.
// Reference: the edge loop of tdnns/analysis.rs::Analysis::new
[[nodiscard]] std::vector<SpaceVector> edgeMapMoments(std::size_t cornerCount, std::size_t degree, std::size_t edge,
                                                      bool forward, const SurfaceAt& surface,
                                                      const Polynomial2& weight);

// The coefficients x_i of the map P_X = X W of an element on its displacement basis: the corner positions, the edge
// moments in the element's own direction ((−1)^l times the canonical moment where edgeSigns[k] < 0), and the L2
// projection of X W on the interior functions. `canonicalMoments[k]` are those of edge k of the element
// (edgeMapMoments).
// Reference: the isoparametric map of tdnns/analysis.rs::Analysis::new
[[nodiscard]] std::vector<SpaceVector> mapCoefficients(const ShellReferenceElement& re,
                                                       const InteriorProjection& projection,
                                                       std::span<const SpaceVector> corners,
                                                       const std::vector<std::vector<SpaceVector>>& canonicalMoments,
                                                       std::span<const double> edgeSigns, const SurfaceAt& surface,
                                                       const Polynomial2& weight);

// The local unknowns of the six rigid motions of the element about the origin: the translations along x, y, z (the
// constant c = c W / W: the coefficients of W on the displacement basis), then the rotations about x, y, z (ω × x_i on
// the displacement coefficients, the edge rotations (2l + 1) ∫ W² ω · (s_k X_τ) P_l(2τ_e − 1) dτ), the shear zero.
// Reference: the element part of tdnns/analysis.rs::Analysis::rigid_modes
[[nodiscard]] std::array<Vector, 6> rigidMotionUnknowns(const ShellReferenceElement& re,
                                                        const ShellElementDefinition& def);
// The same with the interior projection of `re` given (InteriorProjection::build), for a caller that has it: it
// depends on the reference element alone and takes more to build than the rigid motions of one element.
[[nodiscard]] std::array<Vector, 6> rigidMotionUnknowns(const ShellReferenceElement& re,
                                                        const ShellElementDefinition& def,
                                                        const InteriorProjection& projection);

// The elimination of the interior unknowns of one element (static condensation):
//   K_c = K_oo − K_oi K_ii⁻¹ K_io,  f_c = f_o − K_oi K_ii⁻¹ f_i,  u_i = K_ii⁻¹ (f_i − K_io u_o),
// K_ii factorised after its scaling to a unit diagonal. (The formulation keeps the shell interiors as unknowns in
// statics, spec section 15, decision 6; this is the reference's element-level elimination.)
// Reference: tdnns/analysis.rs::Condensed
struct CondensedInterior
{
    // The local indices of the interior unknowns and of the others.
    std::vector<std::size_t> inner;
    std::vector<std::size_t> outer;
    // The Cholesky factor of the scaled K_ii (empty when there is no interior unknown) and the scales.
    std::optional<Cholesky> factor;
    Vector scale;
    // K_oi.
    Matrix koi;

    // K_ii⁻¹ v.
    // Reference: tdnns/analysis.rs::Condensed::solve
    [[nodiscard]] Vector solve(const Vector& v) const;
    // K_oo − K_oi K_ii⁻¹ K_io (symmetric), on the outer unknowns.
    [[nodiscard]] Matrix condensedStiffness(const Matrix& stiffness) const;
    // u_i from the outer unknowns and the interior loads.
    [[nodiscard]] Vector recoverInterior(const Vector& interiorLoad, const Vector& outerUnknowns) const;
};

// The condensation of the element stiffness on the interior unknowns marked in `isInterior` (one flag per local
// unknown); empty when K_ii is not positive definite (the reference refuses the element).
// Reference: the construction of tdnns/analysis.rs::Condensed in Analysis::new
[[nodiscard]] std::optional<CondensedInterior> condenseInterior(const Matrix& stiffness,
                                                                const std::vector<bool>& isInterior);

// The geometric stiffness of the element for its local preload unknowns x (a linear solution): the second variation
// of the membrane, bending and kink measures at the reference with the stresses of x (evaluateNonlinear with
// `geometric`), less the load stiffness of a following pressure when `followerPressure` is given.
// Reference: the element part of tdnns/analysis.rs::Analysis::geometric_stiffness
[[nodiscard]] Matrix elementGeometricStiffness(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                               const ShellElementMatrices& matrices, std::span<const double> x,
                                               const Vector* momentTerm, const Vector* membraneTerm,
                                               const PressureAt* followerPressure);

} // namespace oos::elements::mixedHybrid
