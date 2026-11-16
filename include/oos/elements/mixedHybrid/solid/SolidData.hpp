#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace oos::elements::mixedHybrid
{

// The plain data of a solid element of the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md section
// 3.3): its material, the options of its kernel, its loads, what the kernel gives for each load, its interior fields of
// a modal analysis, and the result type of the constructions that may refuse an element.
//
// Conventions: basic axes; stresses (σxx, σyy, σzz, τxy, τyz, τzx) and strains (εxx, εyy, εzz, γxy, γyz, γzx), the
// shear strains engineering ones (γ = 2ε).
//
// Thread-safety: plain value types. A SolidLoad holds callables, which must be safe to call from several threads at
// once when the kernels of several elements are built in parallel (they are called on the building thread only).

// A point or a vector in the basic axes (or a point of a reference element).
using SolidPoint = std::array<double, 3>;
// Stresses or strains in the order above.
using VoigtVector = std::array<double, 6>;
using VoigtMatrix = std::array<std::array<double, 6>, 6>;

// The material of a solid.
// Reference: solid::SolidMaterial
struct SolidMaterial
{
    // σ = D ε on (εxx, εyy, εzz, γxy, γyz, γzx), basic axes: any symmetric positive definite law (21 constants).
    VoigtMatrix stiffness{};
    double density = 0.0;
    // α: a temperature rise ΔT is the free strain α ΔT on each direct strain.
    double thermalExpansion = 0.0;

    // The isotropic law of Young's modulus E and Poisson's ratio ν, with the Lamé constants
    //   λ = E ν / ((1 + ν)(1 − 2ν)),   μ = E / (2 (1 + ν)),
    // D_ij = λ + 2μ δ_ij for the direct strains, μ for each engineering shear strain.
    // Reference: solid::SolidMaterial::isotropic
    [[nodiscard]] static SolidMaterial isotropic(double youngModulus, double poissonRatio, double density,
                                                 double thermalExpansion);
};

// The degrees and options of the kernel of a solid. The defaults are those of the reference (document 28, section 4:
// quartic traces with complete faces and stresses of the same degree; three interior fields in a modal analysis).
// Reference: solid::SolidOptions (and its Default)
struct SolidOptions
{
    // `trace`: the degree of the displacement on the faces, 1 (grid points only) to 4.
    std::size_t traceDegree = 4;
    // `stress`: the degree of the stresses in the space of the shape, at least traceDegree.
    std::size_t stressDegree = 4;
    // `internal_modes`: interior fields kept as unknowns in a modal analysis: 0, 3 (the answer of the interior to a
    // uniform acceleration of the element) or 12 (to a linear one).
    std::size_t interiorFieldCount = 3;
    // `face`: the degree of the functions inside the faces, at most traceDegree (the kernel takes the smaller of the
    // two); with less, the faces carry fewer unknowns than the complete polynomials of degree traceDegree ask for.
    std::size_t faceDegree = 4;
    // `quad`: points of the rule per direction; stressDegree + 4 when empty.
    std::optional<std::size_t> pointsPerDirection;
};

// A load on the volume of a solid: a force per unit volume, or the free strain of a temperature rise.
// Reference: solid::SolidLoad
struct SolidLoad
{
    enum class Kind
    {
        Body,    // force per unit volume at a point (basic axes)
        Thermal, // free expansion α ΔT of the material at a point, on the three direct strains
    };

    Kind kind = Kind::Body;
    // Kind::Body: the force per unit volume at a point.
    std::function<SolidPoint(const SolidPoint&)> force;
    // Kind::Thermal: α ΔT at a point.
    std::function<double(const SolidPoint&)> freeStrain;

    [[nodiscard]] static SolidLoad body(std::function<SolidPoint(const SolidPoint&)> forcePerVolume);
    [[nodiscard]] static SolidLoad thermal(std::function<double(const SolidPoint&)> freeStrainAt);

    // The force of an acceleration field `acceleration` (gravity, or a uniform acceleration of the model) on a material
    // of density ρ: the body force ρ a. Its stresses are the particular stresses of the kernel; its nodal forces may
    // come from the mass matrix instead (document 28, section 2).
    // Reference: tdnns::solid::build (the loads of `LoadCase::accel`)
    [[nodiscard]] static SolidLoad uniformAcceleration(double density, const SolidPoint& acceleration);

    // The six unit inertia loads of a rigid acceleration, for the stresses of an inertia relief: mode 0, 1, 2 the
    // force ρ e_m of a unit translation along axis m; mode 3, 4, 5 the force ρ (e_{m−3} × x) of a unit angular
    // acceleration about axis m − 3 through the origin.
    // Reference: tdnns::solid::build (the `relief` loads)
    [[nodiscard]] static SolidLoad rigidInertia(double density, std::size_t mode);
};

// What the kernel gives for a load.
// Reference: solid::SolidLoadResult
struct SolidLoadResult
{
    // `f`: the load on the parameters of the element (the boundary unknowns, VolumeKernel::parameterCount()), the
    // bubbles eliminated.
    Vector force;
    // `beta`: the particular stress coefficients of the load: the stresses of the element are
    // stressOperator · q + particularStresses (VolumeKernel::stressCoefficients).
    Vector particularStresses;
};

// The interior fields of an element in a modal analysis, its boundary held: orthonormal for the mass, of diagonal
// stiffness. Each is one more unknown of the element, coupled to its parameters by the mass only (document 28,
// section 3.5): stiffness λ_i on its diagonal, mass 1 on its diagonal and massCoupling(q, i) with parameter q.
// Reference: solid::SolidInternal
struct SolidInteriorFields
{
    // `lam`: the stiffness of each field (unsorted: in the order of the eigenvalue problem that made them).
    Vector stiffnesses;
    // `mqe`: the mass coupling the parameters (rows) with the fields (columns).
    Matrix massCoupling;
};

// What a construction of the solid kernel gives: its value, or the reason the element is refused (a folded element,
// degrees out of range, functions that do not reproduce their traces). A refusal is a property of the model, which the
// caller turns into a diagnostic of the element: never an exception. The Result<T, String> of the reference.
template <class T>
class SolidResult
{
public:
    // A success (implicit, so that a function can return its value as it is).
    SolidResult(T value) : m_value(std::move(value)) {}

    [[nodiscard]] static SolidResult refused(std::string reason)
    {
        SolidResult result;
        result.m_refusal = std::move(reason);
        return result;
    }

    [[nodiscard]] bool ok() const { return m_value.has_value(); }
    // The value; only when ok().
    [[nodiscard]] const T& value() const { return *m_value; }
    [[nodiscard]] T& value() { return *m_value; }
    // Why the element is refused; empty when ok().
    [[nodiscard]] const std::string& refusal() const { return m_refusal; }

private:
    SolidResult() = default;

    std::optional<T> m_value;
    std::string m_refusal;
};

} // namespace oos::elements::mixedHybrid
