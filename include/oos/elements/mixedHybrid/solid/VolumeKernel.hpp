#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/solid/SolidData.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeGeometry.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The kernel of a solid element of the mixed hybrid formulation, every shape (documents 28 and 35 of the reference;
// the specification, section 3.3): a hybrid stress element in the sense of T. H. H. Pian (Derivation of element
// stiffness matrices by assumed stress distributions, AIAA Journal 2 (1964) 1333-1336), with stresses in equilibrium
// weakly.
//
// - The six stress components are functions of the space of degree d of the shape (volumeSpace), made orthonormal on
//   the element by the Cholesky factor of their Gram matrix: with that basis φ the complementary energy is the
//   compliance on each function, and eliminating the stresses leaves the displacement element of the fields below
//   with their strains projected on the stress space,
//     K = Σ_c E_cᵀ Σ_c' D_cc' E_c',   E_c = ∫ φ ε_c(field) dV.
// - The fields are the functions of the element (VolumeGeometry.hpp), three components each, and the interior fields
//   (bubbles) volumeBubble × volumeSpace(d − 1), three components each. The bubbles put the stresses in equilibrium
//   weakly and are eliminated by the Cholesky factor of their own stiffness after a diagonal scaling (the equilibrated
//   Cholesky): K = K_qq − K_bqᵀ K_bb⁻¹ K_bq, and their amplitudes follow the parameters, b = T q, T = −K_bb⁻¹ K_bq.
//   On an element with straight edges the weak equilibrium is the exact one, div σ = 0 (document 28, section 3.4).
// - The rule is that of the shape with d + 4 points per direction (volumeRule), with the Jacobian of the element at
//   each point. The measure is |det J|: either orientation of the corners is accepted; an element whose Jacobian
//   changes sign (it folds) is refused.
//
// Local unknowns (the parameters q of the element, parameterCount() = 3 × geometry().functionCount()): unknown
// 3 j + c is the displacement component c (x, y, z of the basic axes) of function j of the element, numbered as in
// VolumeGeometry.hpp: the corners (three translations each, in the order of the input format), then edge e function m
// at j = nv + e·ec + m (edges in the order of shapeEdges, functions written from the lower number of the edge), then
// face f function i at j = firstFace(f) + i (faces in the order of shapeFaces, functions written from the numbers of
// the face). The unknowns of an edge or face are shared by every element that has it: the integrator maps them to
// one entity per edge and face. The interior fields of a modal analysis (interiorFields) come after them, as
// unknowns of the element alone.
//
// Stress coefficients β (6 × stressFunctionCount()): component c (σxx, σyy, σzz, τxy, τyz, τzx) of function i at
// c · N + i. β = stressOperator() q + Σ the particular stresses of the loads applied (stressCoefficients).
//
// Thread-safety: immutable once built; its functions are const and may be called concurrently. Ownership: a value that
// holds its own copy of the geometry.

// A point of a face for an integral on it.
// Reference: the tuples of volume::VolumeKernel::face_points_of
struct VolumeFacePoint
{
    SolidPoint reference; // the reference point of the element
    SolidPoint position;  // the point in the basic axes
    SolidPoint tangentS;  // a1 = dX / dc1, c the coordinates of the face (faceRule)
    SolidPoint tangentT;  // a2 = dX / dc2
    double weight = 0.0;  // of the face rule (area of its parameter domain)
    double outward = 1.0; // ±1: the sign that makes a1 × a2 point out of the element
};

// A point of a face with its area vector.
// Reference: the tuples of volume::VolumeKernel::face_points
struct VolumeFaceAreaPoint
{
    SolidPoint reference;
    SolidPoint position;
    SolidPoint area; // a1 × a2 times the weight, outward: the normal times the area the point stands for
};

// Reference: volume::VolumeKernel
class VolumeKernel
{
public:
    using TractionField = std::function<SolidPoint(const SolidPoint&)>;
    using StressField = std::function<VoigtVector(const SolidPoint&)>;
    using PressureField = std::function<double(const SolidPoint&)>;

    // The matrices of the element of geometry `geometry`, its corners numbered `order` (the same numbers as the
    // geometry was built with), for the material, the options, the mass when `withMass`, and the loads (their results
    // in the same order, loads()). Refused for degrees out of range (trace 1 to 4, stress at least the trace, a
    // geometry built for other degrees, interior fields other than 0, 3 or 12), for a flat or folded element, and for
    // matrices that are not positive definite.
    // Reference: volume::volume_kernel
    [[nodiscard]] static SolidResult<VolumeKernel> build(const VolumeGeometry& geometry,
                                                         std::span<const base::EntityId> order,
                                                         const SolidMaterial& material, const SolidOptions& options,
                                                         bool withMass, std::span<const SolidLoad> loads);

    // This kernel with the consistent mass of density `density`, the mass build() gives with `withMass` (the same
    // values): its fields are tabulated again, its other matrices copied. For a kernel built without the mass, when an
    // analysis asks for it later (the units of a condensed analysis): about a third of a build. Not for the interior
    // fields of a modal analysis, which need the bubble matrices build() does not keep. Refused as build() for the
    // geometry (not for a kernel that build() gave).
    [[nodiscard]] SolidResult<VolumeKernel> withMass(double density) const;

    // This kernel without its stiffness, its mass and its interior fields: the matrices on the parameters that an
    // analysis assembles once. What is left serves everything else: the stresses of the element (stressCoefficients,
    // affineBodyStresses, stressAt), its geometry (functionsAt, gradients, the points and the loads of its faces), its
    // volume, and, by its bubble operator, its geometric stiffness (geometric, geometricGradient) and its mass
    // (withMass). About five sixths of the bytes of a kernel without mass, for a caller that keeps the kernels of
    // many elements after their stiffness is assembled. hasStiffness() is false on it and stiffness() is empty.
    [[nodiscard]] VolumeKernel withoutStiffness() const;
    [[nodiscard]] bool hasStiffness() const { return !m_stiffness.empty(); }

    // The bytes of the matrices of the kernel (the rest is small).
    [[nodiscard]] std::size_t bytes() const;

    // Why build() refuses the geometry (the degrees and the options, an element that is flat or folds at a point of
    // its rule), without forming a matrix: about a sixth of a build. Empty when the geometry is accepted: build() may
    // still refuse the element for matrices that are not positive definite.
    [[nodiscard]] static std::string geometryRefusal(const VolumeGeometry& geometry, const SolidOptions& options);

    // `p`, `pf`, `d`: the trace, face and stress degrees.
    [[nodiscard]] std::size_t traceDegree() const { return m_traceDegree; }
    [[nodiscard]] std::size_t faceDegree() const { return m_faceDegree; }
    [[nodiscard]] std::size_t stressDegree() const { return m_stressDegree; }
    // `nq`: the parameters of the element (three per function).
    [[nodiscard]] std::size_t parameterCount() const { return m_parameterCount; }
    // `n_bubble`: the bubble amplitudes (three per bubble), eliminated.
    [[nodiscard]] std::size_t bubbleCount() const { return m_bubbleCount; }
    // N: the stress functions per component, volumeSpaceDimension(shape, d).
    [[nodiscard]] std::size_t stressFunctionCount() const { return m_stressFunctionCount; }
    // `k`: the stiffness on the parameters (symmetric).
    [[nodiscard]] const Matrix& stiffness() const { return m_stiffness; }
    // `mass`: the consistent mass on the parameters, the bubbles following statically (symmetric); empty without mass.
    [[nodiscard]] const std::optional<Matrix>& mass() const { return m_mass; }
    // `stress_op`: the stress coefficients per unit parameter (6 N × nq).
    [[nodiscard]] const Matrix& stressOperator() const { return m_stressOperator; }
    // `bubble_op`: T, the bubble amplitudes per unit parameter (n_bubble × nq).
    [[nodiscard]] const Matrix& bubbleOperator() const { return m_bubbleOperator; }
    // `loads`: the result of each load given to build(), in its order.
    [[nodiscard]] const std::vector<SolidLoadResult>& loads() const { return m_loads; }
    // `internal`: the interior fields of a modal analysis (with the mass, interiorFieldCount > 0 and a density).
    [[nodiscard]] const std::optional<SolidInteriorFields>& interiorFields() const { return m_interiorFields; }
    // `volume`: the volume of the element (the sum of the weights of its rule).
    [[nodiscard]] double volume() const { return m_volume; }
    // `shape`: the geometry and the functions of the element.
    [[nodiscard]] const VolumeGeometry& geometry() const { return m_geometry; }
    // `order`: the numbers of its corners, in the order of its shape.
    [[nodiscard]] std::span<const base::EntityId> order() const { return m_order; }

    // The stress coefficients for the parameters q and the loads `loadIndices` (indices in loads()) applied:
    // β = stressOperator q + Σ particularStresses.
    // Reference: tdnns::solid::Solids::beta
    [[nodiscard]] Vector stressCoefficients(std::span<const double> q, std::span<const std::size_t> loadIndices) const;

    // The particular stress coefficients of a body force per unit volume that is affine in the position,
    // f_c(x) = constant_c + Σ_m gradient(c, m) x_m (basic axes; `coefficients`: the three constants, then
    // gradient(c, m) at 3 + 3 m + c), to add to stressCoefficients(): the weight of the element, the inertia of a
    // rotation. Every kernel keeps those of the twelve unit fields (a few milliseconds of its build), so that the
    // stresses of a case with inertial loads need no kernel of their own. Not in the reference.
    static constexpr std::size_t AFFINE_BODY_FIELDS = 12;
    [[nodiscard]] Vector affineBodyStresses(std::span<const double, AFFINE_BODY_FIELDS> coefficients) const;

    // The stresses (σxx, σyy, σzz, τxy, τyz, τzx) at the reference point r for the stress coefficients β.
    // Reference: volume::VolumeKernel::stress_at
    [[nodiscard]] VoigtVector stressAt(std::span<const double> beta, const SolidPoint& r) const;

    // The same at the barycentric coordinates l of a tetrahedron.
    // Reference: volume::VolumeKernel::stress
    [[nodiscard]] VoigtVector stressAtBarycentric(std::span<const double> beta, const std::array<double, 4>& l) const;

    // The values of the functions of the element at the reference point r.
    // Reference: volume::VolumeKernel::functions_at
    [[nodiscard]] std::vector<double> functionsAt(const SolidPoint& r) const;

    // The gradients (basic axes) of the functions of the element at the reference point r.
    // Reference: volume::VolumeKernel::gradients
    [[nodiscard]] std::vector<SolidPoint> gradients(const SolidPoint& r) const;

    // The geometric stiffness on the parameters for the stresses of coefficients β:
    //   K_g = ∫ σ_ab u_k,a v_k,b dV
    // on the functions of the element and the bubbles, the bubbles eliminated as in the stiffness
    // (K_g = [I; T]ᵀ G [I; T]); symmetric (document 28, section 3.6).
    // Reference: volume::VolumeKernel::geometric
    [[nodiscard]] SolidResult<Matrix> geometric(std::span<const double> beta) const;

    // Isotropic Poisson derivative at fixed nodal displacements, temperatures and body forces.
    [[nodiscard]] SolidResult<Matrix> geometricPoissonRate(std::span<const double> beta, const SolidMaterial& material,
                                                           double young) const;
    // The gradient of φᵀ·K_g(β(q))·φ with respect to the parameters q, φ on the parameters (its first
    // parameterCount() values), β(q) = stressOperator q: entry k is φᵀ·K_g(β(e_k))·φ (the particular stresses of the
    // loads, constant in q, are left out). K_g is linear in β, so this is stressOperatorᵀ applied to the products of
    // ∇ψ with itself (ψ = [I; T] φ) against each stress function: one pass over the points instead of one K_g per
    // parameter. Used by the derivative of a buckling factor through its preload.
    [[nodiscard]] SolidResult<Vector> geometricGradient(std::span<const double> phi) const;

    // The local face that has the local corners `corners` (three of them, or all of its corners); empty if none.
    // Reference: volume::VolumeKernel::face_of
    [[nodiscard]] std::optional<std::size_t> faceOf(std::span<const std::size_t> corners) const;

    // The points of the face that has the local corners `corners` (three of them), rule of n points per direction
    // (faceRule), each with its area vector, outward. Empty if no face has these corners.
    // Reference: volume::VolumeKernel::face_points
    [[nodiscard]] std::vector<VolumeFaceAreaPoint> facePoints(const std::array<std::size_t, 3>& corners,
                                                              std::size_t n) const;

    // The points of face `face` (order of shapeFaces), rule of n points per direction.
    // Reference: volume::VolumeKernel::face_points_of
    [[nodiscard]] std::vector<VolumeFacePoint> facePointsOf(std::size_t face, std::size_t n) const;

    // The values of the functions of the element and their derivatives along the two coordinates of face `face`, at
    // the face coordinates c.
    // Reference: volume::VolumeKernel::face_functions_d
    [[nodiscard]] std::vector<Dual<double>> faceFunctionsWithDerivatives(std::size_t face,
                                                                         const std::array<double, 2>& c) const;

    // The loads on the parameters of a force per unit area `traction` (basic axes), of a stress tensor `stress`
    // acting on the face (the traction σ n), or of a pressure `pressure` along the inward normal (the traction −p n),
    // on face `face`, rule of n points per direction (the reference takes n = p + 4). Each is Σ_points N_j(r) t dA.
    // Reference: tdnns::solid::Solids::loads (the part of one element)
    [[nodiscard]] Vector faceTractionLoad(std::size_t face, const TractionField& traction, std::size_t n) const;
    [[nodiscard]] Vector faceStressLoad(std::size_t face, const StressField& stress, std::size_t n) const;
    [[nodiscard]] Vector facePressureLoad(std::size_t face, const PressureField& pressure, std::size_t n) const;

    // The load stiffness of a pressure that follows face `face`: the force on dc1 dc2 is −p a1 × a2 (a1 × a2 outward);
    // when the face moves by U it changes by −p (U,1 × a2 + a1 × U,2). Its symmetric part is kept (document 28,
    // section 3.6), on the parameters, rule of n points per direction (the reference takes n = p + 4). With the sign of
    // the reference: the matrix added to the geometric stiffness.
    // Reference: tdnns::solid::Solids::follower (the part of one element)
    [[nodiscard]] Matrix followerPressureStiffness(std::size_t face, const PressureField& pressure,
                                                   std::size_t n) const;

private:
    VolumeKernel() = default;

    // Σ_points N_j(r) t(point) on face `face`, t the traction the point carries.
    [[nodiscard]] Vector faceLoad(std::size_t face, std::size_t n,
                                  const std::function<SolidPoint(const VolumeFacePoint&)>& tractionAt) const;

    std::size_t m_traceDegree = 0;
    std::size_t m_faceDegree = 0;
    std::size_t m_stressDegree = 0;
    std::size_t m_parameterCount = 0;
    std::size_t m_bubbleCount = 0;
    std::size_t m_stressFunctionCount = 0;
    Matrix m_stiffness;
    std::optional<Matrix> m_mass;
    Matrix m_stressOperator;
    Matrix m_bubbleOperator;
    std::vector<SolidLoadResult> m_loads;
    Matrix m_affineBodyStresses; // 6 N × AFFINE_BODY_FIELDS: the particular stresses of the unit affine body forces
    std::optional<SolidInteriorFields> m_interiorFields;
    double m_volume = 0.0;
    VolumeGeometry m_geometry;
    std::vector<base::EntityId> m_order;
    // `quad`: the points of the rule per direction.
    std::size_t m_pointsPerDirection = 0;
    // `lphi`: the Cholesky factor of the Gram matrix of the stress basis on the element.
    Matrix m_stressGramFactor;
};

} // namespace oos::elements::mixedHybrid
