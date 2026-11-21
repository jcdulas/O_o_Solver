#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"

#include <cmath>

namespace oos::elements::mixedHybrid
{

Matrix22 PointGeometry::cov() const
{
    return {{{dot3(e[0], ac[0]), dot3(e[1], ac[0])}, {dot3(e[0], ac[1]), dot3(e[1], ac[1])}}};
}

Matrix22 PointGeometry::con() const
{
    return {{{dot3(e[0], a[0]), dot3(e[1], a[0])}, {dot3(e[0], a[1]), dot3(e[1], a[1])}}};
}

Matrix22 PointGeometry::curvature() const
{
    const Matrix22 c = cov();
    Matrix22 k{};
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t jj = 0; jj < 2; ++jj)
            for (std::size_t alpha = 0; alpha < 2; ++alpha)
                k[i][jj] += dot3(e[i], dnu[alpha]) * c[alpha][jj];
    return k;
}

namespace
{

// geometryFrom for any way of giving the jet of function i: the tables of an element are read in place, without the
// call through a std::function that geometryFrom makes for each function at each point.
template <typename JetAt>
PointGeometry geometryWith(std::span<const SpaceVector> mapCoefficients, const JetAt& jet, const SurfaceFrame& axes,
                           bool second)
{
    PointGeometry g;
    // d2[α][β] = ∂_αβ X
    std::array<SurfaceFrame, 2> d2{};
    for (std::size_t i = 0; i < mapCoefficients.size(); ++i)
    {
        const SpaceVector& xi = mapCoefficients[i];
        const ReferenceJet& t = jet(i);
        for (std::size_t c = 0; c < 3; ++c)
        {
            g.x[c] += xi[c] * t[0];
            g.a[0][c] += xi[c] * t[1];
            g.a[1][c] += xi[c] * t[2];
            if (second)
            {
                d2[0][0][c] += xi[c] * t[3];
                d2[0][1][c] += xi[c] * t[4];
                d2[1][1][c] += xi[c] * t[5];
            }
        }
    }
    d2[1][0] = d2[0][1];
    const SpaceVector normal = cross3(g.a[0], g.a[1]);
    g.j = norm3(normal);
    g.nu = scl3(normal, 1.0 / g.j);
    // the metric a_αβ and its inverse a^αβ; a^α = a^αβ a_β
    const Matrix22 metric{{{dot3(g.a[0], g.a[0]), dot3(g.a[0], g.a[1])}, {dot3(g.a[1], g.a[0]), dot3(g.a[1], g.a[1])}}};
    const double det = metric[0][0] * metric[1][1] - metric[0][1] * metric[1][0];
    const Matrix22 inverse{{{metric[1][1] / det, -metric[0][1] / det}, {-metric[1][0] / det, metric[0][0] / det}}};
    g.ac = {add3(scl3(g.a[0], inverse[0][0]), scl3(g.a[1], inverse[0][1])),
            add3(scl3(g.a[0], inverse[1][0]), scl3(g.a[1], inverse[1][1]))};
    Matrix22 b{};
    for (std::size_t alpha = 0; alpha < 2; ++alpha)
    {
        for (std::size_t beta = 0; beta < 2; ++beta)
        {
            b[alpha][beta] = dot3(d2[alpha][beta], g.nu);
            for (std::size_t l = 0; l < 2; ++l)
                g.christoffel[l][alpha][beta] = dot3(g.ac[l], d2[alpha][beta]);
        }
    }
    // ∂ν/∂ξ_α = −b_αβ a^β
    g.dnu = {scl3(add3(scl3(g.ac[0], b[0][0]), scl3(g.ac[1], b[0][1])), -1.0),
             scl3(add3(scl3(g.ac[0], b[1][0]), scl3(g.ac[1], b[1][1])), -1.0)};
    g.e = tangentFrameXy(g.nu, axes[0], axes[1]);
    // the element x axis carried by the minimal rotation from the element normal z to ν
    const SpaceVector z = unit3(cross3(axes[0], axes[1]));
    const SpaceVector zn = add3(z, g.nu);
    const SpaceVector f1 = sub3(axes[0], scl3(zn, dot3(g.nu, axes[0]) / (1.0 + dot3(z, g.nu))));
    g.phi = std::atan2(dot3(cross3(g.e[0], f1), g.nu), dot3(g.e[0], f1));
    return g;
}

} // namespace

PointGeometry geometryFrom(std::span<const SpaceVector> mapCoefficients, const JetOfFunction& jet,
                           const SurfaceFrame& axes, bool second)
{
    return geometryWith(mapCoefficients, jet, axes, second);
}

ReferenceJet rationalJet(const ReferenceJet& n, const ReferenceJet& w)
{
    const double f = n[0] / w[0];
    const double fa = (n[1] - f * w[1]) / w[0];
    const double fb = (n[2] - f * w[2]) / w[0];
    const double faa = (n[3] - 2.0 * fa * w[1] - f * w[3]) / w[0];
    const double fab = (n[4] - fa * w[2] - fb * w[1] - f * w[4]) / w[0];
    const double fbb = (n[5] - 2.0 * fb * w[2] - f * w[5]) / w[0];
    return {f, fa, fb, faa, fab, fbb};
}

ElementJets elementJets(const ShellReferenceElement& re, const Polynomial2& weight)
{
    const std::size_t n1 = re.h1Count;
    ElementJets jets;
    jets.volume.reserve(re.points.size() * n1);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const ReferenceJet wj = weight.jet(re.points[q][0], re.points[q][1]);
        for (std::size_t i = 0; i < n1; ++i)
            jets.volume.push_back(rationalJet(re.h1[q * n1 + i], wj));
    }
    jets.edge.resize(re.cornerCount);
    jets.edgeWeight.resize(re.cornerCount);
    for (std::size_t k = 0; k < re.cornerCount; ++k)
    {
        for (std::size_t g = 0; g < re.edgeParameters.size(); ++g)
        {
            const ReferencePoint x = referenceEdgePointAndDirection(re.cornerCount, k, re.edgeParameters[g]).first;
            const ReferenceJet wj = weight.jet(x[0], x[1]);
            jets.edgeWeight[k].push_back(wj[0]);
            for (std::size_t i = 0; i < n1; ++i)
            {
                const EdgeJet& a = re.h1Edge[k][g * n1 + i];
                const ReferenceJet r =
                    rationalJet({a[0], a[1], a[2], 0.0, 0.0, 0.0}, {wj[0], wj[1], wj[2], 0.0, 0.0, 0.0});
                jets.edge[k].push_back({r[0], r[1], r[2]});
            }
        }
    }
    return jets;
}

std::vector<ReferenceJet> h1At(const ShellReferenceElement& re, const ShellElementDefinition& def, double xi,
                               double eta)
{
    const ReferenceJet wj = def.weight->jet(xi, eta);
    std::vector<ReferenceJet> out;
    out.reserve(re.h1Basis.size());
    for (const Field& f : re.h1Basis)
        out.push_back(rationalJet(f[0].jet(xi, eta), wj));
    return out;
}

PointGeometry geometryAt(const ShellReferenceElement& re, const ShellElementDefinition& def, double xi, double eta)
{
    const std::vector<ReferenceJet> jets = h1At(re, def, xi, eta);
    return geometryWith(
        def.mapCoefficients, [&](std::size_t i) -> const ReferenceJet& { return jets[i]; }, def.axes, true);
}

PointGeometry volumeGeometry(const ShellElementDefinition& def, std::size_t h1Count, std::size_t q)
{
    return geometryWith(
        def.mapCoefficients, [&](std::size_t i) -> const ReferenceJet& { return def.volumeJets[q * h1Count + i]; },
        def.axes, true);
}

PointGeometry edgeGeometry(const ShellElementDefinition& def, std::size_t h1Count, std::size_t k, std::size_t g)
{
    return geometryWith(
        def.mapCoefficients,
        [&](std::size_t i)
        {
            const EdgeJet& t = def.edgeJets[k][g * h1Count + i];
            return ReferenceJet{t[0], t[1], t[2], 0.0, 0.0, 0.0};
        },
        def.axes, false);
}

} // namespace oos::elements::mixedHybrid
