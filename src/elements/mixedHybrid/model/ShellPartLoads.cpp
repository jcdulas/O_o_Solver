#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellNonlinear.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include <cmath>

namespace oos::elements::mixedHybrid
{

namespace
{

// A couple about the normal of the shell is refused when it exceeds this share of the couple.
constexpr double NORMAL_COUPLE_TOLERANCE = 1e-8;

// f_3i+c = ∫ N_i f_c dA, f(q, g) the force per unit area of the surface of the grid points at surface point q.
// Reference: tdnns/element.rs::area_load
Vector areaLoadAtPoints(const ShellReferenceElement& re, const ShellElementDefinition& def,
                        const std::function<SpaceVector(std::size_t, const PointGeometry&)>& force)
{
    const std::size_t n1 = re.h1Count;
    Vector out(shellUnknownCount(re), 0.0);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(def, n1, q);
        const SpaceVector v = force(q, g);
        const double s = re.weights[q] * g.j;
        for (std::size_t i = 0; i < n1; ++i)
        {
            const double n = def.volumeJets[q * n1 + i][0] * s;
            for (std::size_t c = 0; c < 3; ++c)
                out[3 * i + c] += n * v[c];
        }
    }
    return out;
}

} // namespace

Vector ShellPart::areaForceLoad(const ShellElementData& element, const ShellAreaForce& force) const
{
    const ShellReferenceElement& re = referenceElement(element);
    const double z0 = element.offset;
    // per unit area of the reference surface of the section, at the offset: μ0 = j(z0) / j
    return areaLoadAtPoints(re, element.definition(),
                            [&](std::size_t q, const PointGeometry& g)
                            {
                                const ShellLoadPoint point{re.points[q][0], re.points[q][1], add3(g.x, scl3(g.nu, z0)),
                                                           g.nu};
                                return scl3(force(point), offsetArea(g, z0));
                            });
}

Vector ShellPart::pressureLoad(const ShellElementData& element, const std::array<double, 4>& corners) const
{
    const ShellReferenceElement& re = referenceElement(element);
    const double z0 = element.offset;
    return areaLoadAtPoints(re, element.definition(),
                            [&](std::size_t q, const PointGeometry& g)
                            {
                                const double p =
                                    cornerInterpolation(element.cornerCount, corners, re.points[q][0], re.points[q][1]);
                                return scl3(g.nu, p * offsetArea(g, z0));
                            });
}

Vector ShellPart::inertiaLoad(const ShellElementData& element, const SpaceVector& acceleration) const
{
    return mixedHybrid::inertiaLoad(referenceElement(element), element.definition(), acceleration);
}

ShellThermalLoad ShellPart::thermalLoad(const ShellElementData& element,
                                        const TemperatureAtPosition& temperatures) const
{
    // the temperatures at the points of the surface of the grid points
    return mixedHybrid::thermalLoad(referenceElement(element), element.definition(), element.matrices,
                                    [&](const PointGeometry& g) { return temperatures(g.x); });
}

std::variant<ShellEdgeLoad, std::string> ShellPart::edgeLoad(std::size_t gridA, std::size_t gridB,
                                                             const EdgeTractionField& load) const
{
    const auto found = m_topology.edgeIndex.find(EdgeKey::of(gridA, gridB));
    if (gridA >= gridCount() || gridB >= gridCount() || found == m_topology.edgeIndex.end())
        return std::string("edge load: no edge of the shells between the two grids");
    return edgeLoadAt(found->second, gridA, gridB,
                      [&](double fraction, const SpaceVector&) { return load(fraction); }, false);
}

std::variant<ShellEdgeLoad, std::string> ShellPart::edgeLoadAt(std::size_t e, std::size_t gridA, std::size_t gridB,
                                                               const EdgePointLoad& load,
                                                               bool takesNormalCouple) const
{
    if (m_topology.edgeElements[e].empty())
        return std::string("edge load: no shell on the edge between the two grids");
    const std::size_t ie = m_topology.edgeElements[e].front();
    const ShellElementData el = element(ie);
    const ShellReferenceElement& re = referenceElement(el);
    const ShellElementDefinition def = el.definition();
    const std::size_t n1 = re.h1Count;
    const std::size_t nc = re.hcurlCount;
    const std::size_t kk = localEdgeOf(ie, e);
    const double sk = el.edgeSigns[kk];
    const SpaceVector& xa = m_input.grids[gridA].position;
    const SpaceVector chord = sub3(m_input.grids[gridB].position, xa);
    const double chordSquared = dot3(chord, chord);
    Vector fl(shellUnknownCount(re), 0.0);
    for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
    {
        const double t = re.edgeParameters[gi];
        const PointGeometry g = edgeGeometry(def, n1, kk, gi);
        const auto tv = referenceEdgePointAndDirection(el.cornerCount, kk, t).second;
        const SpaceVector xt = add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1]));
        const double len = norm3(xt);
        const SpaceVector tt = scl3(xt, 1.0 / len);
        const SpaceVector mu = cross3(tt, g.nu);
        // the load at the chord fraction of the point from grid A
        const std::array<double, 6> v = load(dot3(sub3(g.x, xa), chord) / chordSquared, g.x);
        const SpaceVector force{v[0], v[1], v[2]};
        const SpaceVector couple{v[3], v[4], v[5]};
        if (!takesNormalCouple &&
            std::fabs(dot3(couple, g.nu)) > NORMAL_COUPLE_TOLERANCE * std::fmax(norm3(couple), 1e-300) &&
            norm3(couple) > 0.0)
            return "edge load on " + gridName(gridA) + "-" + gridName(gridB) +
                   ": a couple about the normal of the shell is not a load of this shell (" +
                   std::to_string(std::fabs(dot3(couple, g.nu)) / norm3(couple)) + " of the couple)";
        const double w = re.edgeWeights[gi];
        // the couple about the edge (in the canonical direction) and about the conormal
        const double mt = dot3(couple, scl3(tt, sk));
        const double mm = dot3(couple, mu);
        // The couple about the normal works on the rotation of the edge line in the surface: du/ds = ω × t gives
        // μ · du/ds = ω · (t × μ) = −ω · ν.
        const double mn = takesNormalCouple ? dot3(couple, g.nu) : 0.0;
        for (std::size_t i = 0; i < n1; ++i)
        {
            const EdgeJet& h = def.edgeJets[kk][gi * n1 + i];
            const double dtau = h[1] * tv[0] + h[2] * tv[1];
            for (std::size_t c = 0; c < 3; ++c)
                fl[3 * i + c] += w * len * force[c] * h[0] + w * (mm * g.nu[c] - mn * mu[c]) * dtau;
        }
        // the couple about μ works on −β · t: β = γ − ∇uᵀ ν, its part −ν · du/dτ above, γ · X_τ here
        for (std::size_t i = 0; i < nc; ++i)
        {
            const std::array<double, 2>& h = re.hcurlEdge[kk][gi * nc + i];
            fl[3 * n1 + i] -= w * mm * (h[0] * tv[0] + h[1] * tv[1]);
        }
        // the couple about the edge works on a_T = s_k â / (|X_τ| W²): ∫ m_t â / W² dτ
        const double te = sk > 0.0 ? t : 1.0 - t;
        const double we = def.edgeWeights[kk][gi];
        for (std::size_t l = 0; l < re.rotationCount(kk); ++l)
            fl[shellRotationUnknown(re, kk, l)] += w * mt * legendreValue(l, 2.0 * te - 1.0) / (we * we);
    }
    return ShellEdgeLoad{ie, std::move(fl)};
}

ShellRecovery ShellPart::recover(const ShellElementData& element, std::span<const double> x, const Vector* momentTerm,
                                 const TemperatureAtPosition& temperatures, std::span<const double> fibres) const
{
    const ShellReferenceElement& re = referenceElement(element);
    // PERFORMANCE: the coefficients of the moments and of the membrane strains of the solution, formed once for the
    // centre and the corners (and the four points about each that the transverse shear of a fibre reads). Each point
    // formed them again: two products by the matrices of the element, most of the cost of a point.
    // Measured on the recovery of 10 modes of 1 024 quadrilaterals, with the element taken as the part keeps it
    // (no table of its points built for a case without temperatures): 3.9 s to 2.5 s.
    const ShellSolutionCoefficients coefficients = solutionCoefficients(element.matrices, x, momentTerm);
    const auto pointResults = [&](double xi, double eta, ShellPointState state)
    {
        ShellRecoveryPoint point;
        point.state = std::move(state);
        for (const double z : fibres)
        {
            point.stresses.push_back(plyStresses(element, point.state, z));
            point.transverseShear.push_back(transverseShear(re, element, xi, eta, x, coefficients, temperatures, z));
        }
        return point;
    };
    ShellRecovery out;
    const ReferencePoint centre = referenceCentre(element.cornerCount);
    out.centre = pointResults(centre[0], centre[1],
                              pointState(re, element, centre[0], centre[1], x, coefficients, temperatures));
    for (std::size_t c = 0; c < element.cornerCount; ++c)
    {
        const ReferencePoint corner = referenceCorner(element.cornerCount, c);
        out.corners.push_back(
            pointResults(corner[0], corner[1], cornerState(re, element, c, x, coefficients, temperatures)));
    }
    return out;
}

Matrix ShellPart::geometricStiffness(const ShellElementData& element, std::span<const double> x,
                                     const Vector* momentTerm, const Vector* membraneTerm,
                                     const PressureAt* followerPressure) const
{
    return elementGeometricStiffness(referenceElement(element), element.definition(), element.matrices, x,
                                     momentTerm, membraneTerm, followerPressure);
}

Vector ShellPart::geometricStiffnessGradient(const ShellElementData& element, std::span<const double> phi) const
{
    const ShellReferenceElement& re = referenceElement(element);
    const ShellElementDefinition def = element.definition();
    return mixedHybrid::geometricStiffnessGradient(re, def, element.matrices, prepareNonlinear(re, def), phi);
}

} // namespace oos::elements::mixedHybrid
