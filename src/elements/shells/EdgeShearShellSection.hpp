#pragma once

// What the ShellQuad4 and ShellTria3 kernels share (docs/shell-quad4-formulation.md, "the document", sections 4, 7
// and 13; docs/shell-tria3-formulation.md): the section at a point, the factors of the residual bending flexibility of
// the transverse shear, and the constants of the two elements.

#include "oos/elements/shells/PlateConstitutive.hpp"
#include "oos/elements/shells/ShellScalar.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace oos::elements::edge_shear_detail
{

// c_K: a section without transverse shear flexibility takes H = c_K · 12 max(D11, D22) / S (S the area of the
// element: sectionAt), a shear flexibility below 1e-4 of the bending one.
inline constexpr double RIGID_SHEAR_FACTOR = 1.0e4;
// The residual bending flexibility of the transverse shear (MacNeal, 1978): a beam segment of length a under end
// loads has a linear moment and a constant shear force, and its complementary energy gives the stiffness of the mean
// shear strain, 1 / K = 1 / H + a² / (12 D).
inline constexpr double RESIDUAL_BENDING_FACTOR = 12.0;
// Below this angle (radians) between the element axes and the local frame of a point, the section is not turned.
inline constexpr double NO_ROTATION = 1.0e-14;

// The section of a point in its local frame (document section 4): N = A e0 + B κ, M = B e0 + D κ, Q = H γ.
template <typename T>
struct PointSection
{
    Matrix3<T> a;
    Matrix3<T> b;
    Matrix3<T> d;
    Matrix2<T> h;
};

// The section at a point of thickness t: A = t G1, D = (12I/T³) t³/12 G2, B = −t² G4 (G4 has the sign of a ShellProp
// MID4, whose moments are −∫σ z dz), H = t κG, each in the local frame of the point (`angle`: frameAngleOf). A section
// without transverse shear flexibility takes H = c_K · 12 max(D11, D22) / S, S = `elementArea`: 4 j at a point of a
// quadrilateral, the area of the parallelogram of a triangle (twice its own).
template <typename T>
[[nodiscard]] PointSection<T> sectionAt(const PlateConstitutive<T>& c, const T& t, double angle, double elementArea)
{
    PointSection<T> section;
    if (c.hasMembrane)
        section.a = scaledBy(c.membrane, t);
    if (c.hasBending)
        section.d = scaledBy(c.bending, c.bendingRatio * t * t * t / 12.0);
    if (c.hasCoupling && c.hasMembrane && c.hasBending)
        section.b = scaledBy(c.coupling, -(t * t));
    Matrix2<T> shear;
    if (c.hasBending && c.hasShearFlexibility)
        shear = scaledBy(c.transverseShear, t);
    if (std::abs(angle) > NO_ROTATION)
    {
        const Matrix3<double> u3 = engineeringStrainRotation(angle);
        section.a = congruenceOf(section.a, u3);
        section.b = congruenceOf(section.b, u3);
        section.d = congruenceOf(section.d, u3);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        Matrix2<double> u2;
        u2(0, 0) = cosine;
        u2(0, 1) = sine;
        u2(1, 0) = -sine;
        u2(1, 1) = cosine;
        shear = congruenceOf(shear, u2);
    }
    if (c.hasBending && !c.hasShearFlexibility)
    {
        const T rigid = (RIGID_SHEAR_FACTOR * RESIDUAL_BENDING_FACTOR / elementArea) *
                        largerOf(section.d(0, 0), section.d(1, 1));
        shear(0, 0) = rigid;
        shear(1, 1) = rigid;
    }
    section.h = shear;
    return section;
}

// The factors of the transverse shear of a point of the ShellQuad4 (document section 7), s(z) = z / (1 + z):
// s_x0 = s(12 D*11 / (H11 a²)), s_x1 = s(12 (D*11 / a² + c_t D*33 / b²) / H11), and the same along y with a and b
// exchanged; c_t = `twistFactor`. D*kk = Dkk − Bkk² / Akk is the bending stiffness of a strip in cylindrical bending
// whose membrane force along the strip is zero (Dkk when B is zero or when that value is not positive).
template <typename T>
struct ShearFactors
{
    T evenX{}; // s_x0
    T oddX{};  // s_x1
    T evenY{};
    T oddY{};
};

// s(z) = z / (1 + z).
template <typename T>
[[nodiscard]] T saturatedOf(const T& z)
{
    return z / (T(1.0) + z);
}

// D*11, D*22, D*33 of a section: D*kk = Dkk − Bkk² / Akk, or Dkk when B is zero or when that value is not positive.
template <typename T>
[[nodiscard]] std::array<T, 3> stripBendingOf(const PointSection<T>& section)
{
    std::array<T, 3> strip{};
    for (std::size_t k = 0; k < 3; ++k)
    {
        strip[k] = section.d(k, k);
        if (realPart(section.b(k, k)) != 0.0 && realPart(section.a(k, k)) != 0.0)
        {
            const T reduced = section.d(k, k) - section.b(k, k) * section.b(k, k) / section.a(k, k);
            if (realPart(reduced) > 0.0)
                strip[k] = reduced;
        }
    }
    return strip;
}

template <typename T>
[[nodiscard]] ShearFactors<T> shearFactorsOf(const PointSection<T>& section, double a, double b, double twistFactor)
{
    const std::array<T, 3> strip = stripBendingOf(section);
    const double a2 = a * a;
    const double b2 = b * b;
    ShearFactors<T> factors;
    factors.evenX = saturatedOf(RESIDUAL_BENDING_FACTOR * strip[0] / (section.h(0, 0) * a2));
    factors.oddX =
        saturatedOf(RESIDUAL_BENDING_FACTOR * (strip[0] / a2 + twistFactor * strip[2] / b2) / section.h(0, 0));
    factors.evenY = saturatedOf(RESIDUAL_BENDING_FACTOR * strip[1] / (section.h(1, 1) * b2));
    factors.oddY =
        saturatedOf(RESIDUAL_BENDING_FACTOR * (strip[1] / b2 + twistFactor * strip[2] / a2) / section.h(1, 1));
    return factors;
}

// The 6 × 6 matrix C = [[A, B], [B, D]] of a section.
template <typename T>
[[nodiscard]] ScalarMatrix<T, 6> planeMatrixOf(const PointSection<T>& section)
{
    ScalarMatrix<T, 6> c;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 3; ++k)
        {
            c(i, k) = section.a(i, k);
            c(i, 3 + k) = section.b(i, k);
            c(3 + i, k) = section.b(i, k);
            c(3 + i, 3 + k) = section.d(i, k);
        }
    return c;
}

} // namespace oos::elements::edge_shear_detail
