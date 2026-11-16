#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The Gauss points per layer of the complementary energy of the transverse shear: τ(z) is quadratic in z, its square
// over a constant modulus of degree 4, integrated exactly by 3 points.
constexpr std::size_t SHEAR_ENERGY_POINTS = 3;

Matrix33 scaled(const Matrix33& m, double s)
{
    Matrix33 out = m;
    for (auto& row : out)
        for (double& x : row)
            x *= s;
    return out;
}

Voigt3 scaled(const Voigt3& v, double s)
{
    return {v[0] * s, v[1] * s, v[2] * s};
}

// The inverse of a 2 × 2 matrix, empty when its determinant is zero.
std::optional<Matrix22> inverse2(const Matrix22& m)
{
    const double det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    if (det == 0.0)
        return std::nullopt;
    return Matrix22{{{m[1][1] / det, -m[0][1] / det}, {-m[1][0] / det, m[0][0] / det}}};
}

// Gᵀ S G for 2 × 2 matrices.
Matrix22 congruence2(const Matrix22& g, const Matrix22& s)
{
    Matrix22 out{};
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            for (std::size_t k = 0; k < 2; ++k)
                for (std::size_t l = 0; l < 2; ++l)
                    out[i][j] += g[k][i] * s[k][l] * g[l][j];
    return out;
}

// [A B; Bᵀ D] as a 6 × 6 matrix.
Matrix stiffness6(const Matrix33& a, const Matrix33& b, const Matrix33& d)
{
    Matrix m(6, 6);
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            m(i, j) = a[i][j];
            m(i, 3 + j) = b[i][j];
            m(3 + i, j) = b[j][i];
            m(3 + i, 3 + j) = d[i][j];
        }
    }
    return m;
}

// The two gradients of the in-plane stress that a unit shear force brings, for the stress F of a unit moment:
// (σxx,x + σxy,y ; σxy,x + σyy,y) = R(F) (Qx, Qy), R(F) = [[F_00, F_21], [F_20, F_11]].
Matrix22 restrictToGradient(const Matrix33& f)
{
    return {{{f[0][0], f[2][1]}, {f[2][0], f[1][1]}}};
}

Matrix22 negated(const Matrix22& m)
{
    return {{{-m[0][0], -m[0][1]}, {-m[1][0], -m[1][1]}}};
}

Matrix33 product33(const Matrix33& a, const Matrix33& b)
{
    Matrix33 out{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k)
                out[i][j] += a[i][k] * b[k][j];
    return out;
}

} // namespace

// τ(z) per unit shear force: g0 + f0 (z − z0) + f1 (z² − z0²) / 2.
Matrix22 PlyShear::at(double z0, double z) const
{
    const double d1 = z - z0;
    const double d2 = 0.5 * (z * z - z0 * z0);
    Matrix22 g = g0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            g[i][j] += f0[i][j] * d1 + f1[i][j] * d2;
    return g;
}

Matrix33 isotropicReducedStiffness(double youngModulus, double poissonRatio)
{
    const double c = youngModulus / (1.0 - poissonRatio * poissonRatio);
    return {{{c, c * poissonRatio, 0.0}, {c * poissonRatio, c, 0.0}, {0.0, 0.0, 0.5 * c * (1.0 - poissonRatio)}}};
}

Matrix33 orthotropicReducedStiffness(double e1, double e2, double nu12, double g12, double angle)
{
    const double nu21 = nu12 * e2 / e1;
    const double den = 1.0 - nu12 * nu21;
    const Matrix33 q{{{e1 / den, nu12 * e2 / den, 0.0}, {nu12 * e2 / den, e2 / den, 0.0}, {0.0, 0.0, g12}}};
    return rotateStiffness(q, angle);
}

Matrix33 strainTransformation(double angle)
{
    const double s = std::sin(angle);
    const double c = std::cos(angle);
    return {{{c * c, s * s, c * s}, {s * s, c * c, -c * s}, {-2.0 * c * s, 2.0 * c * s, c * c - s * s}}};
}

Matrix33 rotateStiffness(const Matrix33& m, double angle)
{
    const Matrix33 t = strainTransformation(angle);
    Matrix33 out{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k)
                for (std::size_t l = 0; l < 3; ++l)
                    out[i][j] += t[k][i] * m[k][l] * t[l][j];
    return out;
}

Voigt3 rotateResultant(const Voigt3& v, double angle)
{
    const Matrix33 t = strainTransformation(angle);
    Voigt3 out{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            out[i] += t[k][i] * v[k];
    return out;
}

Section Section::isotropic(double youngModulus, double poissonRatio, double thickness, double shearFactor,
                           double density, double expansion)
{
    const Matrix33 q = isotropicReducedStiffness(youngModulus, poissonRatio);
    const double g = youngModulus / (2.0 * (1.0 + poissonRatio));
    const double t3 = thickness * thickness * thickness;
    const double ta = youngModulus * expansion * thickness / (1.0 - poissonRatio);
    const double td = youngModulus * expansion * t3 / (12.0 * (1.0 - poissonRatio));
    Section s;
    s.a = scaled(q, thickness);
    s.d = scaled(q, t3 / 12.0);
    s.h = Matrix22{{{shearFactor * g * thickness, 0.0}, {0.0, shearFactor * g * thickness}}};
    s.referenceThickness = thickness;
    s.i0 = density * thickness;
    s.i2 = density * t3 / 12.0;
    s.thermalA = {ta, ta, 0.0};
    s.thermalD = {td, td, 0.0};
    Ply ply;
    ply.q = q;
    ply.alpha = {expansion, expansion, 0.0};
    ply.z0 = -0.5 * thickness;
    ply.z1 = 0.5 * thickness;
    ply.shearModuli = {{{g, 0.0}, {0.0, g}}};
    s.plies.push_back(ply);
    if (std::optional<ShearDistribution> distribution = s.shearDistribution())
        s.shear = std::move(distribution->plies);
    return s;
}

Section Section::laminate(std::span<const LaminateLayer> layers)
{
    double total = 0.0;
    for (const LaminateLayer& layer : layers)
        total += layer.thickness;
    Section s;
    s.referenceThickness = total;
    double z = -0.5 * total;
    for (const LaminateLayer& layer : layers)
    {
        const Matrix33 q = orthotropicReducedStiffness(layer.e1, layer.e2, layer.nu12, layer.g12, layer.angle);
        const double z0 = z;
        const double z1 = z + layer.thickness;
        // ∫ (1, z, z²) dz over the layer
        const double m0 = z1 - z0;
        const double m1 = 0.5 * (z1 * z1 - z0 * z0);
        const double m2 = (z1 * z1 * z1 - z0 * z0 * z0) / 3.0;
        // the thermal expansion in the section axes is strain-like: ε_section = T(−angle) ε_ply
        const Matrix33 tm = strainTransformation(-layer.angle);
        const Voigt3 plyExpansion{layer.alpha1, layer.alpha2, 0.0};
        Voigt3 expansion{};
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t k = 0; k < 3; ++k)
                expansion[i] += tm[i][k] * plyExpansion[k];
        Voigt3 qa{};
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t k = 0; k < 3; ++k)
                qa[i] += q[i][k] * expansion[k];
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 3; ++j)
            {
                s.a[i][j] += q[i][j] * m0;
                s.b[i][j] += q[i][j] * m1;
                s.d[i][j] += q[i][j] * m2;
            }
            s.thermalA[i] += qa[i] * m0;
            s.thermalB[i] += qa[i] * m1;
            s.thermalD[i] += qa[i] * m2;
        }
        // the transverse shear moduli [[g13, 0], [0, g23]] turned by the ply angle
        const double sn = std::sin(layer.angle);
        const double cs = std::cos(layer.angle);
        const Matrix22 shearModuli{{{layer.g13 * cs * cs + layer.g23 * sn * sn, (layer.g13 - layer.g23) * sn * cs},
                                    {(layer.g13 - layer.g23) * sn * cs, layer.g13 * sn * sn + layer.g23 * cs * cs}}};
        s.i0 += layer.density * m0;
        s.i1 += layer.density * m1;
        s.i2 += layer.density * m2;
        Ply ply;
        ply.q = q;
        ply.alpha = expansion;
        ply.z0 = z0;
        ply.z1 = z1;
        ply.angle = layer.angle;
        ply.shearModuli = (layer.g13 > 0.0 && layer.g23 > 0.0) ? shearModuli : Matrix22{};
        s.plies.push_back(ply);
        z = z1;
    }
    if (std::optional<ShearDistribution> distribution = s.shearDistribution())
    {
        s.shear = std::move(distribution->plies);
        s.h = distribution->stiffness;
    }
    return s;
}

std::optional<ShearDistribution> Section::shearDistribution() const
{
    if (plies.empty())
        return std::nullopt;
    const std::optional<Matrix> inverseAbd = inverse(stiffness6(a, b, d));
    if (!inverseAbd)
        return std::nullopt;
    // the strains of a unit moment without membrane force: e0 = β M, κ = δ M
    Matrix33 beta{};
    Matrix33 delta{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            beta[i][j] = (*inverseAbd)(i, 3 + j);
            delta[i][j] = (*inverseAbd)(3 + i, 3 + j);
        }
    }
    const LineRule gauss = gaussLegendre(SHEAR_ENERGY_POINTS);
    ShearDistribution out;
    out.plies.reserve(plies.size());
    Matrix22 g{};
    Matrix22 compliance{};
    for (const Ply& ply : plies)
    {
        PlyShear plyShear;
        plyShear.g0 = g;
        plyShear.f0 = negated(restrictToGradient(product33(ply.q, beta)));
        plyShear.f1 = negated(restrictToGradient(product33(ply.q, delta)));
        // a layer rigid in transverse shear stores no complementary energy
        const std::optional<Matrix22> flexibility = inverse2(ply.shearModuli);
        if (flexibility && ply.shearModuli[0][0] > 0.0 && ply.shearModuli[1][1] > 0.0)
        {
            const double half = 0.5 * (ply.z1 - ply.z0);
            for (std::size_t k = 0; k < SHEAR_ENERGY_POINTS; ++k)
            {
                const double zk = ply.z0 + half * (gauss.points[k] + 1.0);
                const Matrix22 energy = congruence2(plyShear.at(ply.z0, zk), *flexibility);
                for (std::size_t i = 0; i < 2; ++i)
                    for (std::size_t j = 0; j < 2; ++j)
                        compliance[i][j] += energy[i][j] * (gauss.weights[k] * half);
            }
        }
        g = plyShear.at(ply.z0, ply.z1);
        out.plies.push_back(plyShear);
    }
    const double det = compliance[0][0] * compliance[1][1] - compliance[0][1] * compliance[1][0];
    const std::optional<Matrix22> stiffness = inverse2(compliance);
    if (stiffness && det > 0.0)
    {
        const Matrix22& h = *stiffness;
        const double offDiagonal = 0.5 * (h[0][1] + h[1][0]);
        out.stiffness = Matrix22{{{h[0][0], offDiagonal}, {offDiagonal, h[1][1]}}};
    }
    return out;
}

SectionPoint Section::at(double thickness, double angle) const
{
    const double r = thickness / referenceThickness;
    const Matrix33 aScaled = scaled(a, r);
    Matrix22 hMaterial{};
    if (h)
    {
        hMaterial = {{{(*h)[0][0] * r, (*h)[0][1] * r}, {(*h)[1][0] * r, (*h)[1][1] * r}}};
    }
    else
    {
        const double m = kirchhoffRatio * (aScaled[0][0] + aScaled[1][1] + aScaled[2][2]) / 3.0;
        hMaterial = {{{m, 0.0}, {0.0, m}}};
    }
    // γ_material = R γ_local with R = [[c, s], [−s, c]]: H_local = Rᵀ H R
    const double s = std::sin(angle);
    const double c = std::cos(angle);
    const Matrix22 rotation{{{c, s}, {-s, c}}};
    SectionPoint point;
    point.a = rotateStiffness(aScaled, angle);
    point.b = rotateStiffness(scaled(b, r * r), angle);
    point.d = rotateStiffness(scaled(d, r * r * r), angle);
    point.h = congruence2(rotation, hMaterial);
    point.mass = {r * (i0 - nonStructuralMass) + nonStructuralMass, r * r * i1, r * r * r * i2};
    point.thermalA = rotateResultant(scaled(thermalA, r), angle);
    point.thermalB = rotateResultant(scaled(thermalB, r * r), angle);
    point.thermalD = rotateResultant(scaled(thermalD, r * r * r), angle);
    point.thickness = thickness;
    return point;
}

ShellSection::ShellSection(Section section, const std::array<double, 4>& thickness, const SpaceVector& direction)
    : m_section(std::move(section)), m_thickness(thickness), m_direction(direction)
{
}

ShellSection ShellSection::uniform(Section section, const SpaceVector& direction)
{
    const double t = section.referenceThickness;
    return ShellSection(std::move(section), {t, t, t, t}, direction);
}

double ShellSection::thicknessAt(double u, double v) const
{
    return 0.25 * ((1.0 - u) * (1.0 - v) * m_thickness[0] + (1.0 + u) * (1.0 - v) * m_thickness[1] +
                   (1.0 + u) * (1.0 + v) * m_thickness[2] + (1.0 - u) * (1.0 + v) * m_thickness[3]);
}

double ShellSection::angle(const SurfaceFrame& frame) const
{
    const SpaceVector& dir = m_direction;
    const double ax = frame[0][0] * dir[0] + frame[0][1] * dir[1] + frame[0][2] * dir[2];
    const double ay = frame[1][0] * dir[0] + frame[1][1] * dir[1] + frame[1][2] * dir[2];
    return std::atan2(ay, ax);
}

SectionPoint ShellSection::at(double u, double v, const SurfaceFrame& frame) const
{
    return m_section.at(thicknessAt(u, v), angle(frame));
}

} // namespace oos::elements::mixedHybrid
