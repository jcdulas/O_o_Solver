#include "oos/elements/shells/PlateConstitutive.hpp"

#include <cmath>

namespace oos::elements
{

namespace
{

template <typename T>
[[nodiscard]] Matrix3<T> matrix3Of(const base::Mat3& m)
{
    Matrix3<T> result;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            result(i, j) = T(m(i, j));
    return result;
}

[[nodiscard]] Matrix3<Complex> stepped(const Matrix3<double>& value, const Matrix3<double>& rate)
{
    Matrix3<Complex> result;
    for (std::size_t k = 0; k < 9; ++k)
        result.values[k] = complexStepped(value.values[k], rate.values[k]);
    return result;
}

// The rotation of a shear-strain vector (γxz, γyz) into a frame turned by `angle` (the in-plane case of spec §1.3's
// U2, written for the physical order x, y).
[[nodiscard]] Matrix2<double> shearStrainRotation(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    Matrix2<double> u;
    u(0, 0) = c;
    u(0, 1) = s;
    u(1, 0) = -s;
    u(1, 1) = c;
    return u;
}

// Spec §2.4: Ex (or Ey) of the bending card. A IsoMat or OrthoMat2D gives the modulus of its plane-stress matrix along
// the axis, G11 − G12²/G22 (E of a IsoMat, E1 of a OrthoMat2D, exactly); a AnisoMat2D or a laminate its raw diagonal
// term.
template <typename T>
[[nodiscard]] T bendingYoungModulus(const PlateSectionValues<T>& section, std::size_t axis)
{
    const std::size_t other = 1 - axis;
    if (section.bendingCard == BendingCard::General)
        return section.bending(axis, axis);
    return section.bending(axis, axis) - section.bending(0, 1) * section.bending(0, 1) / section.bending(other, other);
}

// The transverse shear moduli G3 (on γxz, γyz, material frame) of the bending card, which NASTRAN-95's QUAD4 uses
// without MID3 (spec §1.2): diag(G, G) of a IsoMat, [[G11, G12], [G12, G22]] of a AnisoMat2D (a laminate alike),
// diag(G1Z, G2Z) of a OrthoMat2D (G12 for a blank one).
template <typename T>
[[nodiscard]] Matrix2<T> bendingCardShear(const PlateSectionValues<T>& section)
{
    Matrix2<T> g;
    switch (section.bendingCard)
    {
    case BendingCard::Isotropic:
        g(0, 0) = section.bending(2, 2);
        g(1, 1) = section.bending(2, 2);
        break;
    case BendingCard::General:
        g(0, 0) = section.bending(0, 0);
        g(0, 1) = section.bending(0, 1);
        g(1, 0) = section.bending(0, 1);
        g(1, 1) = section.bending(1, 1);
        break;
    case BendingCard::Orthotropic:
        g(0, 0) = T(section.bendingCardShear[0]);
        g(1, 1) = T(section.bendingCardShear[1]);
        break;
    }
    return g;
}

// The solution x of a·x = b for a small dense system (Gauss elimination with partial pivoting on the real parts).
template <std::size_t N>
[[nodiscard]] std::array<double, N> solved(ScalarMatrix<double, N> a, std::array<double, N> b)
{
    for (std::size_t k = 0; k < N; ++k)
    {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < N; ++i)
            if (std::abs(a(i, k)) > std::abs(a(pivot, k)))
                pivot = i;
        if (a(pivot, k) == 0.0)
            return {}; // a singular stiffness: no thermal strain can be formed
        for (std::size_t j = 0; j < N; ++j)
            std::swap(a(k, j), a(pivot, j));
        std::swap(b[k], b[pivot]);
        for (std::size_t i = k + 1; i < N; ++i)
        {
            const double factor = a(i, k) / a(k, k);
            for (std::size_t j = k; j < N; ++j)
                a(i, j) -= factor * a(k, j);
            b[i] -= factor * b[k];
        }
    }
    std::array<double, N> x{};
    for (std::size_t k = N; k-- > 0;)
    {
        double sum = b[k];
        for (std::size_t j = k + 1; j < N; ++j)
            sum -= a(k, j) * x[j];
        x[k] = sum / a(k, k);
    }
    return x;
}

} // namespace

Matrix3<double> engineeringStrainRotation(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    Matrix3<double> u;
    u(0, 0) = c * c;
    u(0, 1) = s * s;
    u(0, 2) = c * s;
    u(1, 0) = s * s;
    u(1, 1) = c * c;
    u(1, 2) = -c * s;
    u(2, 0) = -2.0 * c * s;
    u(2, 1) = 2.0 * c * s;
    u(2, 2) = c * c - s * s;
    return u;
}

Matrix3<double> stressRotation(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    Matrix3<double> r;
    r(0, 0) = c * c;
    r(0, 1) = s * s;
    r(0, 2) = 2.0 * c * s;
    r(1, 0) = s * s;
    r(1, 1) = c * c;
    r(1, 2) = -2.0 * c * s;
    r(2, 0) = -c * s;
    r(2, 1) = c * s;
    r(2, 2) = c * c - s * s;
    return r;
}

PlateSectionValues<double> plateSectionValuesOf(const ShellSection& section, const ShellMaterial* bendingMaterial)
{
    PlateSectionValues<double> values;
    values.hasMembrane = section.hasMembrane;
    values.hasBending = section.hasBending;
    values.hasCoupling = section.hasCoupling;
    values.hasShearFlexibility = section.hasShearFlexibility;
    values.isLaminate = section.isLaminate();
    values.membrane = matrix3Of<double>(section.membraneMatrix);
    values.bending = matrix3Of<double>(section.bendingMatrix);
    // ShellSection keeps the coupling in the lamination theory's sign, −G4 of a ShellProp MID4 (ShellSection.hpp).
    values.coupling = scaledBy(matrix3Of<double>(section.couplingMatrix), -1.0);
    values.transverseShear = section.transverseShearStiffness;
    values.bendingRatio = section.bendingRatio;
    values.shearRatio = section.shearRatio;
    values.density = section.density;
    values.nonStructuralMass = section.nonStructuralMass;
    values.bendingCard = BendingCard::General;
    if (bendingMaterial != nullptr && !section.isLaminate())
    {
        switch (bendingMaterial->type)
        {
        case ShellMaterialType::IsoMat:
            values.bendingCard = BendingCard::Isotropic;
            break;
        case ShellMaterialType::AnisoMat2D:
            values.bendingCard = BendingCard::General;
            break;
        case ShellMaterialType::OrthoMat2D:
            values.bendingCard = BendingCard::Orthotropic;
            // Spec §2.4: a zero G1Z or G2Z of the OrthoMat2D is replaced by its G12.
            for (std::size_t k = 0; k < 2; ++k)
                values.bendingCardShear[k] = bendingMaterial->transverseShear[k] > 0.0
                                                 ? bendingMaterial->transverseShear[k]
                                                 : bendingMaterial->g12;
            break;
        }
    }
    return values;
}

PlateSectionValues<double> plateSectionRatesOf(const PlateSectionValues<double>& values, const ShellSectionRate& rate,
                                               double densityRate)
{
    PlateSectionValues<double> rates = values;
    rates.membrane = matrix3Of<double>(rate.membraneMatrix);
    rates.bending = matrix3Of<double>(rate.bendingMatrix);
    rates.coupling = scaledBy(matrix3Of<double>(rate.couplingMatrix), -1.0);
    rates.transverseShear = rate.transverseShearStiffness;
    rates.bendingRatio = 0.0;
    rates.shearRatio = 0.0;
    rates.density = densityRate;
    rates.nonStructuralMass = 0.0;
    rates.bendingCardShear = {};
    // A coupling zero at the point moves off zero along a MID4 term (shellSectionRate): the coupling is then formed.
    rates.hasCoupling = values.hasCoupling || rate.movesCoupling;
    return rates;
}

PlateSectionValues<Complex> complexStepped(const PlateSectionValues<double>& values,
                                           const PlateSectionValues<double>& rates)
{
    PlateSectionValues<Complex> result;
    result.hasMembrane = values.hasMembrane;
    result.hasBending = values.hasBending;
    result.hasCoupling = values.hasCoupling || rates.hasCoupling;
    result.hasShearFlexibility = values.hasShearFlexibility;
    result.isLaminate = values.isLaminate;
    result.bendingCard = values.bendingCard;
    result.membrane = stepped(values.membrane, rates.membrane);
    result.bending = stepped(values.bending, rates.bending);
    result.coupling = stepped(values.coupling, rates.coupling);
    for (std::size_t k = 0; k < 2; ++k)
        result.transverseShear[k] = complexStepped(values.transverseShear[k], rates.transverseShear[k]);
    result.bendingCardShear = values.bendingCardShear;
    result.bendingRatio = complexStepped(values.bendingRatio, rates.bendingRatio);
    result.shearRatio = complexStepped(values.shearRatio, rates.shearRatio);
    result.density = complexStepped(values.density, rates.density);
    result.nonStructuralMass = complexStepped(values.nonStructuralMass, rates.nonStructuralMass);
    return result;
}

template <typename T>
PlateConstitutive<T> plateConstitutiveOf(const PlateSectionValues<T>& section, double materialAngle)
{
    PlateConstitutive<T> c;
    c.hasMembrane = section.hasMembrane;
    c.hasBending = section.hasBending;
    c.hasCoupling = section.hasCoupling;
    c.hasShearFlexibility = section.hasShearFlexibility;
    c.bendingRatio = section.bendingRatio;
    c.shearRatio = section.shearRatio;
    // Spec §1.3: ε_material = U3 ε_element, so G_element = U3ᵀ G U3.
    const Matrix3<double> u3 = engineeringStrainRotation(materialAngle);
    c.membrane = congruenceOf(section.membrane, u3);
    c.bending = congruenceOf(section.bending, u3);
    c.coupling = congruenceOf(section.coupling, u3);
    if (!section.hasBending)
        return c;

    // The transverse shear along the material axes, then rotated: G3_e = U2ᵀ G3 U2.
    const Matrix2<double> u2 = shearStrainRotation(materialAngle);
    Matrix2<T> shear;
    Matrix2<T> compliance;
    if (section.hasShearFlexibility)
    {
        for (std::size_t k = 0; k < 2; ++k)
        {
            const bool isRigid = !(realPart(section.transverseShear[k]) > 0.0);
            // A direction MID3 leaves rigid (a blank G1Z or G2Z of a OrthoMat2D) has no shear compliance (the TRIA3
            // compliance counts 0 there); the QUAD4, whose shear-correction factors need a modulus, takes the
            // in-plane shear modulus of the bending card there, the replacement NASTRAN-95 applies to a zero G1Z/G2Z
            // in those factors (spec §2.4). A choice of this project: NASTRAN-95 has no rigid direction.
            shear(k, k) = isRigid ? section.shearRatio * section.bending(2, 2) : section.transverseShear[k];
            compliance(k, k) = isRigid ? T(0.0) : T(1.0) / section.transverseShear[k];
        }
        c.shearModulusX = shear(0, 0) / section.shearRatio;
        c.shearModulusY = shear(1, 1) / section.shearRatio;
    }
    else
    {
        // Spec §1.2: without MID3 the QUAD4 takes the bending card's G3 (with TS/T); the TRIA3 none (zero
        // compliance: its shear is then the bending relief alone).
        const Matrix2<T> g3 = bendingCardShear(section);
        shear = scaledBy(g3, section.shearRatio);
        c.shearModulusX = g3(0, 0);
        c.shearModulusY = g3(1, 1);
    }
    c.transverseShear = congruenceOf(shear, u2);
    c.transverseShearCompliance = congruenceOf(compliance, u2);
    c.youngX = bendingYoungModulus(section, 0);
    c.youngY = bendingYoungModulus(section, 1);
    return c;
}

template PlateConstitutive<double> plateConstitutiveOf(const PlateSectionValues<double>&, double);
template PlateConstitutive<Complex> plateConstitutiveOf(const PlateSectionValues<Complex>&, double);

PlateThermalStrain plateThermalStrainOf(const ShellSection& section, double materialAngle,
                                        const PlateConstitutive<double>& constitutive,
                                        const ShellTemperatureChange& temperature, double thickness)
{
    PlateThermalStrain strain;
    if (temperature.isZero())
        return strain;
    // The element sees one temperature change, the mean of its grid point values (spec §6.1: NASTRAN-95 passes only
    // their arithmetic mean to the element); ShellTemperatureChange::deltaT is that mean.
    const double deltaT = temperature.deltaT;
    const double gradient = temperature.gradient;
    if (!section.isLaminate())
    {
        // Spec §6.2, §6.3: ε_m = α_m ΔT and the bending strain z·α_b T′, the expansion coefficients turned to the
        // element frame by U3⁻¹ (a strain-like vector: α_e = U3(−φ) α_m).
        const Matrix3<double> back = engineeringStrainRotation(-materialAngle);
        if (section.hasMembrane)
            strain.membrane = productOf(back, std::array<double, 3>{section.membraneExpansion[0] * deltaT,
                                                                    section.membraneExpansion[1] * deltaT,
                                                                    section.membraneExpansion[2] * deltaT});
        if (section.hasBending)
            strain.gradient = productOf(back, std::array<double, 3>{section.bendingExpansion[0] * gradient,
                                                                    section.bendingExpansion[1] * gradient,
                                                                    section.bendingExpansion[2] * gradient});
        strain.isZero = false;
        return strain;
    }
    // Spec §6.4: a laminate's thermal resultants (lamination theory: N = ∫σ dz, M = ∫σ z dz, material frame) turned
    // into the free strains of its section [A B; B D]·[ε; κ] = [N_T; M_T], A = t G1, B = −t² G4, D = (12I/T³) t³/12 G2.
    const ThermalResultants resultants = section.thermalResultants(deltaT, gradient, thickness);
    const Matrix3<double> toElement = transposeOf(engineeringStrainRotation(materialAngle)); // σ_e = U3ᵀ σ_m
    const std::array<double, 3> force = productOf(toElement, resultants.force);
    const std::array<double, 3> moment = productOf(toElement, resultants.moment);
    const double inertia = constitutive.bendingRatio * thickness * thickness * thickness / 12.0;
    ScalarMatrix<double, 6> abd;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            abd(i, j) = constitutive.hasMembrane ? thickness * constitutive.membrane(i, j) : (i == j ? 1.0 : 0.0);
            abd(3 + i, 3 + j) = constitutive.hasBending ? inertia * constitutive.bending(i, j) : (i == j ? 1.0 : 0.0);
            const double b = constitutive.hasCoupling ? -thickness * thickness * constitutive.coupling(i, j) : 0.0;
            abd(i, 3 + j) = b;
            abd(3 + i, j) = b;
        }
    const std::array<double, 6> free =
        solved<6>(abd, {force[0], force[1], force[2], moment[0], moment[1], moment[2]});
    for (std::size_t k = 0; k < 3; ++k)
    {
        strain.membrane[k] = constitutive.hasMembrane ? free[k] : 0.0;
        strain.gradient[k] = constitutive.hasBending ? free[3 + k] : 0.0;
    }
    strain.isZero = false;
    return strain;
}

} // namespace oos::elements
