#include "oos/elements/mixedHybrid/shell/ShellElementResults.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementStresses.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace oos::elements::mixedHybrid
{

namespace
{

// The step of the central differences of the resultants in the reference coordinates.
constexpr double GRADIENT_STEP = 1e-5;
// The relative tolerance with which a distance through the thickness is found in a ply.
constexpr double PLY_TOLERANCE = 1e-12;

} // namespace

std::array<std::array<double, 8>, 8> sectionCompliance(const SectionPoint& point)
{
    const Matrix mi = membraneBendingCompliance(point);
    const Matrix22& h = point.h;
    const double det = h[0][0] * h[1][1] - h[1][0] * h[0][1];
    if (det == 0.0)
        throw base::InternalError("singular transverse shear section");
    std::array<std::array<double, 8>, 8> c{};
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            c[i][j] = mi(i, j);
    c[6][6] = h[1][1] / det;
    c[6][7] = -h[0][1] / det;
    c[7][6] = -h[1][0] / det;
    c[7][7] = h[0][0] / det;
    return c;
}

std::array<double, 8> gridResultants(const ShellReferenceElement& re, const ShellElementData& element, double xi,
                                     double eta, std::span<const double> x, const Vector* momentTerm,
                                     const TemperatureAtPosition& temperatureAt)
{
    return gridResultants(re, element, xi, eta, x, solutionCoefficients(element.matrices, x, momentTerm),
                          temperatureAt);
}

std::array<double, 8> gridResultants(const ShellReferenceElement& re, const ShellElementData& element, double xi,
                                     double eta, std::span<const double> x,
                                     const ShellSolutionCoefficients& coefficients,
                                     const TemperatureAtPosition& temperatureAt)
{
    const ShellElementDefinition def = element.definition();
    std::array<double, 2> temperatures{0.0, 0.0};
    if (temperatureAt)
        temperatures = temperatureAt(geometryAt(re, def, xi, eta).x);
    return resultants(re, def, coefficients, x, temperatures, xi, eta);
}

std::array<double, 8> sectionResultants(const ShellReferenceElement& re, const ShellElementData& element, double xi,
                                        double eta, std::span<const double> x, const Vector* momentTerm,
                                        const TemperatureAtPosition& temperatureAt)
{
    return sectionResultants(re, element, xi, eta, x, solutionCoefficients(element.matrices, x, momentTerm),
                             temperatureAt);
}

std::array<double, 8> sectionResultants(const ShellReferenceElement& re, const ShellElementData& element, double xi,
                                        double eta, std::span<const double> x,
                                        const ShellSolutionCoefficients& coefficients,
                                        const TemperatureAtPosition& temperatureAt)
{
    std::array<double, 8> r = gridResultants(re, element, xi, eta, x, coefficients, temperatureAt);
    const double z0 = element.offset;
    if (z0 == 0.0)
        return r;
    // the forces of the reference surface of the section: N_g = μ0 Pᵀ N, M_g = μ0 Pᵀ (z0 N + M)
    const PointGeometry g = geometryAt(re, element.definition(), xi, eta);
    const double mu0 = offsetArea(g, z0);
    const Matrix33 p = offsetMap(g, z0);
    Matrix33 pt{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            pt[i][j] = p[j][i];
    const std::optional<Matrix33> pit = inverse33(pt);
    if (!pit)
        throw base::InternalError("singular offset map");
    const Voigt3 ng = times33(*pit, {r[0] / mu0, r[1] / mu0, r[2] / mu0});
    const Voigt3 mp = times33(*pit, {r[3] / mu0, r[4] / mu0, r[5] / mu0});
    for (std::size_t i = 0; i < 3; ++i)
    {
        r[i] = ng[i];
        r[3 + i] = mp[i] - ng[i] * z0;
    }
    return r;
}

ShellPointState pointStateOf(const ShellReferenceElement& re, const ShellElementData& element, double xi, double eta,
                             const std::array<double, 8>& resultants, const TemperatureAtPosition& temperatureAt)
{
    const PointGeometry g = geometryAt(re, element.definition(), xi, eta);
    // the section about its own reference surface, where the resultants are given
    const SectionPoint sp = sectionAt(element.section, 0.0, xi, eta, g.e);
    const std::array<std::array<double, 8>, 8> compliance = sectionCompliance(sp);
    ShellPointState state;
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            state.strains[i] += compliance[i][j] * resultants[j];
    if (temperatureAt)
    {
        state.thermal = temperatureAt(g.x);
        const auto [e0, k0] = freeStrains(sp, g.curvature(), state.thermal);
        for (std::size_t i = 0; i < 3; ++i)
        {
            state.strains[i] += e0[i];
            state.strains[3 + i] += k0[i];
        }
    }
    state.x = add3(g.x, scl3(g.nu, element.offset));
    state.frame = g.e;
    state.normal = g.nu;
    state.resultants = resultants;
    state.angle = element.section.angle(g.e);
    state.thickness = sp.thickness;
    return state;
}

ShellPointState pointState(const ShellReferenceElement& re, const ShellElementData& element, double xi, double eta,
                           std::span<const double> x, const Vector* momentTerm,
                           const TemperatureAtPosition& temperatureAt)
{
    return pointState(re, element, xi, eta, x, solutionCoefficients(element.matrices, x, momentTerm), temperatureAt);
}

ShellPointState pointState(const ShellReferenceElement& re, const ShellElementData& element, double xi, double eta,
                           std::span<const double> x, const ShellSolutionCoefficients& coefficients,
                           const TemperatureAtPosition& temperatureAt)
{
    const std::array<double, 8> r = sectionResultants(re, element, xi, eta, x, coefficients, temperatureAt);
    return pointStateOf(re, element, xi, eta, r, temperatureAt);
}

ShellPointState cornerState(const ShellReferenceElement& re, const ShellElementData& element, std::size_t corner,
                            std::span<const double> x, const Vector* momentTerm,
                            const TemperatureAtPosition& temperatureAt)
{
    return cornerState(re, element, corner, x, solutionCoefficients(element.matrices, x, momentTerm), temperatureAt);
}

ShellPointState cornerState(const ShellReferenceElement& re, const ShellElementData& element, std::size_t corner,
                            std::span<const double> x, const ShellSolutionCoefficients& coefficients,
                            const TemperatureAtPosition& temperatureAt)
{
    const ReferencePoint point = referenceCorner(element.cornerCount, corner);
    return pointState(re, element, point[0], point[1], x, coefficients, temperatureAt);
}

std::pair<Voigt3, Voigt3> plyStresses(const ShellElementData& element, const ShellPointState& state, double z)
{
    const Section& section = element.section.section();
    const double r = state.thickness / section.referenceThickness;
    const Matrix33 t = strainTransformation(state.angle);
    const Voigt3 em = times33(t, {state.strains[0], state.strains[1], state.strains[2]});
    const Voigt3 km = times33(t, {state.strains[3], state.strains[4], state.strains[5]});
    const double zr = z / r;
    const double tolerance = PLY_TOLERANCE * section.referenceThickness;
    const Ply* ply = nullptr;
    for (const Ply& candidate : section.plies)
    {
        if (zr >= candidate.z0 - tolerance && zr <= candidate.z1 + tolerance)
        {
            ply = &candidate;
            break;
        }
    }
    if (!ply)
        ply = zr < section.plies.front().z0 ? &section.plies.front() : &section.plies.back();
    const double dt = state.thermal[0] + z * state.thermal[1];
    Voigt3 sm{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            sm[i] += ply->q[i][j] * (em[j] + z * km[j] - ply->alpha[j] * dt);
    const Matrix33 tp = strainTransformation(-ply->angle);
    const Voigt3 sp = transposeTimes33(tp, sm);
    return {sm, sp};
}

std::pair<std::array<double, 2>, std::array<double, 2>> transverseShear(
    const ShellReferenceElement& re, const ShellElementData& element, double xi, double eta, std::span<const double> x,
    const Vector* momentTerm, const TemperatureAtPosition& temperatureAt, double z)
{
    return transverseShear(re, element, xi, eta, x, solutionCoefficients(element.matrices, x, momentTerm),
                           temperatureAt, z);
}

std::pair<std::array<double, 2>, std::array<double, 2>> transverseShear(
    const ShellReferenceElement& re, const ShellElementData& element, double xi, double eta, std::span<const double> x,
    const ShellSolutionCoefficients& coefficients, const TemperatureAtPosition& temperatureAt, double z)
{
    const Section& section = element.section.section();
    const PointGeometry g = geometryAt(re, element.definition(), xi, eta);
    const SectionPoint sp = sectionAt(element.section, 0.0, xi, eta, g.e);
    const double angle = element.section.angle(g.e);
    const double r = sp.thickness / section.referenceThickness;
    const Matrix22 ci = g.cov();
    const double h = GRADIENT_STEP;
    const auto at = [&](double du, double dv)
    { return sectionResultants(re, element, xi + du, eta + dv, x, coefficients, temperatureAt); };
    const std::array<double, 8> up = at(h, 0.0);
    const std::array<double, 8> um = at(-h, 0.0);
    const std::array<double, 8> vp = at(0.0, h);
    const std::array<double, 8> vm = at(0.0, -h);
    // the gradients of (N, M) along e1 and e2, then those of the strains (ε0, κ)
    const Matrix inv = membraneBendingCompliance(sp);
    std::array<Vector, 2> de;
    for (std::size_t i = 0; i < 2; ++i)
    {
        Vector grad(6);
        for (std::size_t c = 0; c < 6; ++c)
            grad[c] = ci[0][i] * (up[c] - um[c]) / (2.0 * h) + ci[1][i] * (vp[c] - vm[c]) / (2.0 * h);
        de[i] = inv * grad;
    }
    // the divergence of the in-plane stress of a ply per unit z⁰ and z¹: (σ_xx,x + σ_xy,y ; σ_xy,x + σ_yy,y)
    const auto divergence = [&](const Matrix33& q)
    {
        std::array<std::array<Voigt3, 2>, 2> s{};
        for (std::size_t i = 0; i < 2; ++i)
        {
            s[i][0] = times33(q, {de[i][0], de[i][1], de[i][2]});
            s[i][1] = times33(q, {de[i][3], de[i][4], de[i][5]});
        }
        const std::array<double, 2> d0{s[0][0][0] + s[1][0][2], s[0][0][2] + s[1][0][1]};
        const std::array<double, 2> d1{s[0][1][0] + s[1][1][2], s[0][1][2] + s[1][1][1]};
        return std::pair{d0, d1};
    };
    std::array<double, 2> tau{};
    std::array<double, 2> area{};
    std::optional<std::array<double, 2>> tauZ;
    std::size_t layer = section.plies.size() - 1;
    for (std::size_t k = 0; k < section.plies.size(); ++k)
    {
        const Ply& ply = section.plies[k];
        const auto [d0, d1] = divergence(rotateStiffness(ply.q, angle));
        const double z0 = ply.z0 * r;
        const double z1 = ply.z1 * r;
        const auto span = [&](double zz)
        {
            return std::array<double, 2>{-(d0[0] * (zz - z0) + 0.5 * d1[0] * (zz * zz - z0 * z0)),
                                         -(d0[1] * (zz - z0) + 0.5 * d1[1] * (zz * zz - z0 * z0))};
        };
        if (!tauZ && z <= z1 + PLY_TOLERANCE * sp.thickness)
        {
            const std::array<double, 2> s = span(std::max(z, z0));
            tauZ = std::array<double, 2>{tau[0] + s[0], tau[1] + s[1]};
            layer = k;
        }
        const std::array<double, 2> sm = span(0.5 * (z0 + z1));
        const std::array<double, 2> s1 = span(z1);
        for (std::size_t i = 0; i < 2; ++i)
            area[i] += (z1 - z0) * (tau[i] + (4.0 * sm[i] + s1[i]) / 6.0);
        tau = {tau[0] + s1[0], tau[1] + s1[1]};
    }
    const double zb = section.plies.front().z0 * r;
    const double zt = section.plies.back().z1 * r;
    const std::array<double, 2> tz = tauZ.value_or(tau);
    const double f = std::clamp((z - zb) / (zt - zb), 0.0, 1.0);
    const std::array<double, 2> tl{tz[0] - f * tau[0], tz[1] - f * tau[1]};
    const std::array<double, 8> q0 = at(0.0, 0.0);
    const std::array<double, 2> missing{q0[6] - (area[0] - 0.5 * (zt - zb) * tau[0]),
                                        q0[7] - (area[1] - 0.5 * (zt - zb) * tau[1])};
    const double s = std::sin(angle);
    const double c = std::cos(angle);
    const auto rotate = [&](const std::array<double, 2>& v)
    { return std::array<double, 2>{c * v[0] + s * v[1], -s * v[0] + c * v[1]}; };
    std::array<double, 2> tm = rotate(tl);
    const std::array<double, 2> mm = rotate(missing);
    if (layer < section.shear.size())
    {
        const Ply& ply = section.plies[layer];
        const Matrix22 gz = section.shear[layer].at(ply.z0, std::clamp(z / r, ply.z0, ply.z1));
        for (std::size_t i = 0; i < 2; ++i)
            tm[i] += (gz[i][0] * mm[0] + gz[i][1] * mm[1]) / r;
    }
    const double sinPly = std::sin(section.plies[layer].angle);
    const double cosPly = std::cos(section.plies[layer].angle);
    return {tm, {cosPly * tm[0] + sinPly * tm[1], -sinPly * tm[0] + cosPly * tm[1]}};
}

} // namespace oos::elements::mixedHybrid
