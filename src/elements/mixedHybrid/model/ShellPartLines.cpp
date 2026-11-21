#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"

#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The halvings of the search of the edge parameter at a fraction of the chord (2⁻⁶⁰: the round-off of the parameter).
constexpr int CHORD_FRACTION_HALVINGS = 60;

} // namespace

LineKinematics shellLineKinematics(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                   std::size_t localEdge, double tau, bool forward)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nc = re.hcurlCount;
    const std::size_t nv = re.cornerCount;
    const std::size_t p = re.shearDegree;
    const std::size_t nx = shellUnknownCount(re);
    const auto [xp, tv] = referenceEdgePointAndDirection(nv, localEdge, tau);
    const std::vector<ReferenceJet> jets = h1At(re, def, xp[0], xp[1]);
    const PointGeometry g = geometryFrom(def.mapCoefficients, [&](std::size_t i) { return jets[i]; }, def.axes, true);
    // X_τ = a_α t̂^α, its length and the unit tangent in the element's direction
    const SpaceVector xt = add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1]));
    const double len = norm3(xt);
    const SpaceVector te = scl3(xt, 1.0 / len);
    // the second derivative of the edge, X_αβ = Γ^λ_αβ a_λ + b_αβ ν, b_αβ = −a_α · ∂_β ν (symmetrised)
    SpaceVector xtt{};
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t b = 0; b < 2; ++b)
        {
            const double bab = -0.5 * (dot3(g.a[a], g.dnu[b]) + dot3(g.a[b], g.dnu[a]));
            const SpaceVector tangential =
                add3(scl3(g.a[0], g.christoffel[0][a][b]), scl3(g.a[1], g.christoffel[1][a][b]));
            const SpaceVector xab = add3(tangential, scl3(g.nu, bab));
            xtt = add3(xtt, scl3(xab, tv[a] * tv[b]));
        }
    const SpaceVector dt = scl3(sub3(xtt, scl3(te, dot3(te, xtt))), 1.0 / (len * len));
    const SpaceVector dn = scl3(add3(scl3(g.dnu[0], tv[0]), scl3(g.dnu[1], tv[1])), 1.0 / len);
    const SpaceVector mu = cross3(te, g.nu);
    const double we = def.weight->evaluate(xp[0], xp[1]);
    const double sk = def.edgeSigns[localEdge];
    Matrix du(3, nx);
    Matrix uu(3, nx);
    for (std::size_t i = 0; i < n1; ++i)
    {
        const ReferenceJet& h = jets[i];
        const double ds = (h[1] * tv[0] + h[2] * tv[1]) / len;
        for (std::size_t c = 0; c < 3; ++c)
        {
            du(c, 3 * i + c) = ds;
            uu(c, 3 * i + c) = h[0];
        }
    }
    Matrix th(3, nx);
    // θ · μ = ν · du/ds ; θ · ν = −μ · du/ds
    for (std::size_t col = 0; col < 3 * n1; ++col)
    {
        const SpaceVector d{du(0, col), du(1, col), du(2, col)};
        const double tm = dot3(g.nu, d);
        const double tn = -dot3(mu, d);
        for (std::size_t r = 0; r < 3; ++r)
            th(r, col) += tm * mu[r] + tn * g.nu[r];
    }
    // θ · t = a_T = s_k â(τ_e) / (|X_τ| W²)
    const double tg = sk > 0.0 ? tau : 1.0 - tau;
    for (std::size_t l = 0; l <= p; ++l)
    {
        const double v = sk * legendreValue(l, 2.0 * tg - 1.0) / (len * we * we);
        for (std::size_t r = 0; r < 3; ++r)
            th(r, 3 * n1 + nc + localEdge * (p + 1) + l) += v * te[r];
    }
    const double s = forward ? 1.0 : -1.0;
    LineKinematics kin;
    kin.point = g.x;
    kin.fraction = forward ? tau : 1.0 - tau;
    kin.tangent = scl3(te, s);
    kin.fractionRate = 1.0 / len;
    kin.normal = g.nu;
    kin.tangentRate = dt;
    kin.normalRate = scl3(dn, s);
    kin.displacement = std::move(uu);
    kin.displacementRate = du * s;
    kin.rotation = std::move(th);
    return kin;
}

void ShellPart::placeLines()
{
    const std::size_t ng = m_input.grids.size();
    m_lineEdges.assign(m_input.lines.size(), std::nullopt);
    for (std::size_t il = 0; il < m_input.lines.size(); ++il)
    {
        const std::array<std::size_t, 2>& grids = m_input.lines[il].element.grids;
        const std::size_t a = grids[0];
        const std::size_t b = grids[1];
        if (a >= ng || b >= ng || a == b)
        {
            addProblem(ShellPartProblem::Subject::Line, il, "its grids are not two grids of the shells");
            continue;
        }
        const auto found = m_topology.edgeIndex.find(EdgeKey::of(a, b));
        if (found == m_topology.edgeIndex.end())
        {
            addProblem(
                ShellPartProblem::Subject::Line, il,
                "it is not on an edge of the shells (between grids " + gridName(a) + " and " + gridName(b) + ")");
            continue;
        }
        // on an edge of shells: the kinematics of the first shell of the edge
        const std::size_t e = found->second;
        const std::size_t ie = m_topology.edgeElements[e].front();
        const std::size_t kk = localEdgeOf(ie, e);
        m_lineEdges[il] = LineEdge{e, ie, kk, m_input.shells[ie].corners.at(kk) == a};
    }
}

std::optional<LineSource> ShellPart::lineSource(std::size_t line) const
{
    if (line >= m_lineEdges.size() || !m_lineEdges[line])
        return std::nullopt;
    const LineEdge& at = *m_lineEdges[line];
    return LineSource::onShellEdge(at.shell, at.localEdge, at.forward);
}

std::variant<LineData, std::string> ShellPart::lineData(std::size_t line) const
{
    if (line >= m_input.lines.size())
        return std::string("it is not on an edge of the shells");
    return lineData(line, m_input.lines[line].section, true);
}

std::variant<LineData, std::string> ShellPart::lineData(std::size_t line, const LineSection& section,
                                                        bool withMass) const
{
    if (line >= m_lineEdges.size() || !m_lineEdges[line])
        return std::string("it is not on an edge of the shells");
    const LineEdge& at = *m_lineEdges[line];
    const ShellPartLine& input = m_input.lines[line];
    const ShellElementData shell = element(at.shell);
    const ShellReferenceElement& re = referenceElement(shell);
    const ShellElementDefinition def = shell.definition();
    // the rule of the edges of the shell: ds = w dτ / (dλ/ds) with dλ/ds = 1 / |X_τ|
    std::vector<WeightedLineKinematics> points;
    points.reserve(re.edgeParameters.size());
    for (std::size_t g = 0; g < re.edgeParameters.size(); ++g)
    {
        LineKinematics kin = shellLineKinematics(re, def, at.localEdge, re.edgeParameters[g], at.forward);
        const double ds = re.edgeWeights[g] / kin.fractionRate;
        points.push_back({std::move(kin), ds});
    }
    const std::array<std::size_t, 2> rotations{6 * input.element.grids[0] + 3, 6 * input.element.grids[1] + 3};
    return lineCore(LineSource::onShellEdge(at.shell, at.localEdge, at.forward), points, shellUnknownCount(re),
                    m_input.degree, true, input.element, section, shell.dofs, shell.signs, rotations,
                    m_input.withMass && withMass);
}

LineKinematics ShellPart::lineKinematics(std::size_t line, double fraction) const
{
    if (line >= m_lineEdges.size() || !m_lineEdges[line])
        throw base::InternalError("shell part: the line is not on an edge of the shells");
    const LineEdge& at = *m_lineEdges[line];
    const ShellElementData shell = element(at.shell);
    const ShellReferenceElement& re = referenceElement(shell);
    const ShellElementDefinition def = shell.definition();
    const std::array<std::size_t, 2>& grids = m_input.lines[line].element.grids;
    const SpaceVector& xa = m_input.grids[grids[0]].position;
    const SpaceVector chord = sub3(m_input.grids[grids[1]].position, xa);
    // the chord coordinate of the edge point at the element's parameter τ (monotonic along the edge)
    const auto chordFraction = [&](double tau)
    {
        const ReferencePoint x = referenceEdgePointAndDirection(re.cornerCount, at.localEdge, tau).first;
        const PointGeometry g = geometryAt(re, def, x[0], x[1]);
        return dot3(sub3(g.x, xa), chord) / dot3(chord, chord);
    };
    double lo = 0.0;
    double hi = 1.0;
    for (int halving = 0; halving < CHORD_FRACTION_HALVINGS; ++halving)
    {
        const double mid = 0.5 * (lo + hi);
        if (chordFraction(at.forward ? mid : 1.0 - mid) < fraction)
            lo = mid;
        else
            hi = mid;
    }
    const double t = 0.5 * (lo + hi);
    return shellLineKinematics(re, def, at.localEdge, at.forward ? t : 1.0 - t, at.forward);
}

std::variant<LineState, std::string> ShellPart::lineState(std::size_t line, const LineData& data, double fraction,
                                                          std::span<const double> x,
                                                          const LineTemperatureField& temperature) const
{
    const LineKinematics kin = lineKinematics(line, fraction);
    const ShellPartLine& input = m_input.lines[line];
    return lineStateOfCase(data, kin, input.element, input.section, x, temperature);
}

} // namespace oos::elements::mixedHybrid
