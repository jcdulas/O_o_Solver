#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/geometry/EdgeSupport.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace oos::elements::mixedHybrid
{

namespace
{

// A direction is in the span of the held axes when its components along the free axes are below this (squared).
constexpr double SPAN_TOLERANCE_SQUARED = 1e-18;
// A direction across the edge whose slope a held rotation fixes is kept when its part independent of the earlier
// ones is longer than this.
constexpr double INDEPENDENT_DIRECTION = 1e-8;

// v is in the span of the held axes: its components along the axes not held vanish.
bool inHeldSpan(const SpaceVector& v, const std::array<bool, 3>& held)
{
    double free = 0.0;
    for (std::size_t c = 0; c < 3; ++c)
        if (!held[c])
            free += v[c] * v[c];
    return free < SPAN_TOLERANCE_SQUARED;
}

SpaceVector basisVector(std::size_t c)
{
    SpaceVector e{};
    e[c] = 1.0;
    return e;
}

// The terms of a row kept: those whose magnitude exceeds `threshold` times the largest (every one that is not zero
// with a zero threshold), by increasing unknown.
std::vector<ShellPartTerm> keptTerms(const std::map<std::size_t, double>& row, double threshold)
{
    double top = 0.0;
    for (const auto& [unknown, value] : row)
        top = std::max(top, std::fabs(value));
    std::vector<ShellPartTerm> out;
    for (const auto& [unknown, value] : row)
        if (std::fabs(value) > threshold * top)
            out.push_back({unknown, value});
    return out;
}

// The rod that carries the values held at the two grids of an edge along it (document 34 of the reference, section
// 5): linear along the chord, cubic Hermite across it with the rotations held at both grids, in the directions
// t × e_c of the held rotation axes (Q the projection on them):
//   u(s) = (1 − s) u_a + s u_b + b(s) Q (u_a − u_b) + h2(s) L θ_a × t + h4(s) L θ_b × t,
//   b = s (1 − s)(1 − 2s), h2 = s − 2s² + s³, h4 = −s² + s³,
// s the chord coordinate, L the chord, t its direction. Its derivatives with respect to the grid values are the rows
// of the map that carries a reaction back.
struct CarriedRod
{
    SpaceVector ua{};
    SpaceVector ub{};
    SpaceVector ta{};
    SpaceVector tb{};
    SpaceVector tangent{};
    double length = 0.0;
    // The unit directions q across the edge whose slope a held rotation fixes, and Q = Σ q qᵀ.
    std::vector<SpaceVector> controlled;
    Matrix33 across{};

    [[nodiscard]] SpaceVector at(double s) const
    {
        const double h2 = s - 2.0 * s * s + s * s * s;
        const double h4 = -s * s + s * s * s;
        const double bubble = s * (1.0 - s) * (1.0 - 2.0 * s);
        const SpaceVector d = sub3(ua, ub);
        SpaceVector dAcross{};
        for (const SpaceVector& q : controlled)
            dAcross = add3(dAcross, scl3(q, dot3(d, q)));
        const SpaceVector linear = add3(scl3(ua, 1.0 - s), scl3(ub, s));
        return add3(add3(linear, scl3(dAcross, bubble)),
                    add3(scl3(cross3(ta, tangent), h2 * length), scl3(cross3(tb, tangent), h4 * length)));
    }

    // ∂u_c(s)/∂(value) for the translation c' of grid a (end 0) or b (end 1), and for its rotation c'.
    [[nodiscard]] double translationRate(double s, std::size_t end, std::size_t c, std::size_t cp) const
    {
        const double bubble = s * (1.0 - s) * (1.0 - 2.0 * s);
        const double delta = c == cp ? 1.0 : 0.0;
        return end == 0 ? (1.0 - s) * delta + bubble * across[c][cp] : s * delta - bubble * across[c][cp];
    }
    [[nodiscard]] double rotationRate(double s, std::size_t end, std::size_t c, std::size_t cp) const
    {
        const double h = end == 0 ? s - 2.0 * s * s + s * s * s : -s * s + s * s * s;
        return h * length * cross3(basisVector(cp), tangent)[c];
    }
};

} // namespace

ShellSupports ShellPart::supportsOf(std::span<const ShellGridSupport> grids) const
{
    const std::size_t ng = m_input.grids.size();
    const std::size_t k = m_input.degree;
    const std::size_t p = k - 1;
    ShellSupports out;
    // the supports of each grid (several entries of one grid add up)
    std::vector<ShellGridSupport> supportOf(ng);
    for (std::size_t g = 0; g < ng; ++g)
        supportOf[g].grid = g;
    for (const ShellGridSupport& support : grids)
    {
        if (support.grid >= ng)
            continue;
        for (std::size_t c = 0; c < 6; ++c)
            if (support.held[c])
            {
                supportOf[support.grid].held[c] = true;
                supportOf[support.grid].values[c] = support.values[c];
            }
    }
    const auto value = [&](std::size_t g, std::size_t c)
    { return supportOf[g].held[c] ? supportOf[g].values[c] : 0.0; };

    // the entity DOFs held, by unknown: a later rule replaces an earlier one
    std::map<std::size_t, ShellDerivedSupport> fixed;
    const auto hold = [&](std::size_t unknown, double v, std::vector<ShellGridTerm> carried)
    { fixed[unknown] = ShellDerivedSupport{unknown, v, std::move(carried)}; };

    // the shear held on the junctions (no orientation-free continuity exists between more than two pieces) and in the
    // Kirchhoff elements
    for (std::size_t e = 0; e < m_topology.edges.size(); ++e)
        if (m_topology.edgeElements[e].size() >= 3)
            for (std::size_t l = 0; l <= p; ++l)
                hold(m_entities[edgeEntity(e)].firstUnknown + edgeShearDof(l), 0.0, {});
    for (std::size_t ie = 0; ie < m_shells.size(); ++ie)
    {
        if (!m_shells[ie].kirchhoff)
            continue;
        const ShellReferenceElement& re = referenceElement(m_shells[ie].cornerCount);
        for (std::size_t i = 0; i < re.hcurlCount; ++i)
            hold(m_shells[ie].dofs[3 * re.h1Count + i], 0.0, {});
    }

    // the supports along the external edges
    for (std::size_t e = 0; e < m_topology.edges.size(); ++e)
    {
        const std::size_t ga = m_topology.edges[e].lower;
        const std::size_t gb = m_topology.edges[e].higher;
        const EdgeSupport held = edgeHeld({}, m_topology.edges[e], m_topology.edgeElements[e].size(),
                                          supportOf[ga].held, supportOf[gb].held, true);
        if (held == EdgeSupport{})
            continue;
        const std::size_t ie = m_topology.edgeElements[e].front();
        const ShellElementData el = element(ie);
        const ShellReferenceElement& re = referenceElement(el.cornerCount);
        const ShellElementDefinition def = el.definition();
        const std::size_t kk = localEdgeOf(ie, e);
        const std::size_t first = m_entities[edgeEntity(e)].firstUnknown;
        const double sk = el.edgeSigns[kk];
        const bool forward = sk > 0.0;

        // the prescribed motion of the edge: the grid values carried along it as by a rod, its moments taken on the
        // isoparametric edge
        CarriedRod rod;
        const SpaceVector& xa = m_input.grids[ga].position;
        const SpaceVector chord = sub3(m_input.grids[gb].position, xa);
        rod.length = norm3(chord);
        rod.tangent = scl3(chord, 1.0 / rod.length);
        for (std::size_t c = 0; c < 3; ++c)
        {
            rod.ua[c] = value(ga, c);
            rod.ub[c] = value(gb, c);
            rod.ta[c] = held.rotations[c] ? value(ga, 3 + c) : 0.0;
            rod.tb[c] = held.rotations[c] ? value(gb, 3 + c) : 0.0;
        }
        // the directions across the edge whose slope a held rotation fixes: t × e_c
        for (std::size_t c = 0; c < 3; ++c)
        {
            if (!held.rotations[c])
                continue;
            SpaceVector v = cross3(rod.tangent, basisVector(c));
            for (const SpaceVector& q : rod.controlled)
                v = sub3(v, scl3(q, dot3(v, q)));
            if (norm3(v) > INDEPENDENT_DIRECTION)
                rod.controlled.push_back(unit3(v));
        }
        for (const SpaceVector& q : rod.controlled)
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    rod.across[r][c] += q[r] * q[c];

        // the moments of the trace (against P_l(2τ_e − 1) W) and of the rotation about the edge
        // (∫ W² θ · X_τ P_l, θ linear along the chord), with their rates with respect to the grid values
        std::vector<SpaceVector> traceMoments(k - 1, SpaceVector{});
        std::vector<double> rotationMoments(p + 1, 0.0);
        // [l][c][end][c'] for the translations and the rotations of the grids, [l][end][c'] for the edge rotation
        std::vector<std::array<std::array<SpaceVector, 2>, 3>> traceByTranslation(k - 1);
        std::vector<std::array<std::array<SpaceVector, 2>, 3>> traceByRotation(k - 1);
        std::vector<std::array<SpaceVector, 2>> rotationByRotation(p + 1);
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const PointGeometry g = edgeGeometry(def, re.h1Count, kk, gi);
            const auto tv = referenceEdgePointAndDirection(el.cornerCount, kk, re.edgeParameters[gi]).second;
            const double te = forward ? re.edgeParameters[gi] : 1.0 - re.edgeParameters[gi];
            const double sc = dot3(sub3(g.x, xa), rod.tangent) / rod.length;
            const double we = def.edgeWeights[kk][gi];
            const SpaceVector uv = rod.at(sc);
            for (std::size_t l = 0; l + 1 < k; ++l)
            {
                const double f = re.edgeWeights[gi] * legendreValue(l, 2.0 * te - 1.0) * we;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    traceMoments[l][c] += f * uv[c];
                    for (std::size_t end = 0; end < 2; ++end)
                        for (std::size_t cp = 0; cp < 3; ++cp)
                        {
                            traceByTranslation[l][c][end][cp] += f * rod.translationRate(sc, end, c, cp);
                            traceByRotation[l][c][end][cp] += f * rod.rotationRate(sc, end, c, cp);
                        }
                }
            }
            // the rotation about the edge: W² θ · X_τ_e, X_τ_e = s_k X_τ in the canonical direction
            const SpaceVector xt = scl3(add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1])), sk);
            const SpaceVector theta = add3(scl3(rod.ta, 1.0 - sc), scl3(rod.tb, sc));
            for (std::size_t l = 0; l <= p; ++l)
            {
                const double f = re.edgeWeights[gi] * legendreValue(l, 2.0 * te - 1.0) * we * we;
                rotationMoments[l] += f * dot3(theta, xt);
                for (std::size_t cp = 0; cp < 3; ++cp)
                {
                    rotationByRotation[l][0][cp] += f * (1.0 - sc) * xt[cp];
                    rotationByRotation[l][1][cp] += f * sc * xt[cp];
                }
            }
        }
        const std::array<std::size_t, 2> ends{ga, gb};
        for (std::size_t c = 0; c < 3; ++c)
        {
            if (!held.translations[c])
                continue;
            for (std::size_t l = 0; l + 1 < k; ++l)
            {
                std::vector<ShellGridTerm> carried;
                for (std::size_t end = 0; end < 2; ++end)
                {
                    for (std::size_t cp = 0; cp < 3; ++cp)
                        if (supportOf[ends[end]].held[cp] && traceByTranslation[l][c][end][cp] != 0.0)
                            carried.push_back({ends[end], cp, traceByTranslation[l][c][end][cp]});
                    for (std::size_t cp = 0; cp < 3; ++cp)
                        if (held.rotations[cp] && traceByRotation[l][c][end][cp] != 0.0)
                            carried.push_back({ends[end], 3 + cp, traceByRotation[l][c][end][cp]});
                }
                hold(first + 3 * l + c, traceMoments[l][c], std::move(carried));
            }
        }
        // what the held rotations reach: the edge rotation where the tangent is in their span, the shear where the
        // conormal is (with the normal displacement held)
        bool rotationHeld = true;
        bool shearHeld = true;
        bool normalHeld = true;
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const PointGeometry g = edgeGeometry(def, re.h1Count, kk, gi);
            const auto tv = referenceEdgePointAndDirection(el.cornerCount, kk, re.edgeParameters[gi]).second;
            const SpaceVector t = unit3(add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1])));
            const SpaceVector mu = cross3(t, g.nu);
            rotationHeld = rotationHeld && inHeldSpan(t, held.rotations);
            shearHeld = shearHeld && inHeldSpan(mu, held.rotations);
            normalHeld = normalHeld && inHeldSpan(g.nu, held.translations);
        }
        const bool anyRotation = held.rotations[0] || held.rotations[1] || held.rotations[2];
        if (rotationHeld && anyRotation)
            for (std::size_t l = 0; l <= p; ++l)
            {
                // the coefficient on P_l: (2l + 1) times the moment
                const double scale = static_cast<double>(2 * l + 1);
                std::vector<ShellGridTerm> carried;
                for (std::size_t end = 0; end < 2; ++end)
                    for (std::size_t cp = 0; cp < 3; ++cp)
                        if (held.rotations[cp] && rotationByRotation[l][end][cp] != 0.0)
                            carried.push_back({ends[end], 3 + cp, scale * rotationByRotation[l][end][cp]});
                hold(first + edgeRotationDof(l), scale * rotationMoments[l], std::move(carried));
            }
        if (shearHeld && anyRotation)
        {
            if (normalHeld)
                for (std::size_t l = 0; l <= p; ++l)
                    hold(first + edgeShearDof(l), 0.0, {});
            else
                out.notes.push_back("edge " + gridName(ga) + "-" + gridName(gb) +
                                    ": rotation across the edge held with the normal displacement free: not "
                                    "represented");
        }
    }

    // the rotations held with a value at grids of shells without line elements: they act only through the edges that
    // hold them
    std::set<std::size_t> lineGrids = m_input.lineGrids;
    for (const ShellPartLine& line : m_input.lines)
        lineGrids.insert(line.element.grids.begin(), line.element.grids.end());
    std::vector<bool> isCorner(ng, false);
    for (const ShellElementData& shell : m_shells)
        for (const std::size_t g : shell.nodes)
            isCorner[g] = true;
    std::size_t rotationsAtGrids = 0;
    std::string named;
    for (std::size_t g = 0; g < ng; ++g)
    {
        if (!isCorner[g] || lineGrids.contains(g))
            continue;
        bool counted = false;
        for (std::size_t c = 3; c < 6; ++c)
            if (supportOf[g].held[c] && supportOf[g].values[c] != 0.0)
            {
                ++rotationsAtGrids;
                counted = true;
            }
        if (counted)
            named += (named.empty() ? "" : ", ") + gridName(g);
    }
    if (rotationsAtGrids > 0)
        out.notes.push_back(std::to_string(rotationsAtGrids) +
                            " rotation components held at grids act only through the edges that hold them (grids " +
                            named + ")");

    out.held.reserve(fixed.size());
    for (auto& [unknown, support] : fixed)
        out.held.push_back(std::move(support));
    return out;
}

std::vector<ShellPartTerm> ShellPart::meanRotationRow(std::size_t grid, std::size_t component,
                                                      double relativeThreshold) const
{
    return meanRotationRows(grid, relativeThreshold)[component];
}

std::array<std::vector<ShellPartTerm>, 3> ShellPart::meanRotationRows(std::size_t grid, double relativeThreshold) const
{
    const std::vector<std::pair<std::size_t, std::size_t>> at = shellsAt(grid);
    if (at.empty())
        return {};
    const double count = static_cast<double>(at.size());
    // the rotation of each shell at its corner there (exact for a rigid motion), with equal weights
    std::array<std::map<std::size_t, double>, 3> rows;
    for (const auto& [ie, corner] : at)
    {
        const ShellElementData el = element(ie);
        const ReferencePoint x = referenceCorner(el.cornerCount, corner);
        const Matrix r = rotationRowsOf(referenceElement(el.cornerCount), el.definition(), x[0], x[1]);
        for (std::size_t component = 0; component < 3; ++component)
            for (std::size_t i = 0; i < r.columns(); ++i)
                if (r(component, i) != 0.0)
                    rows[component][el.dofs[i]] += el.signs[i] * r(component, i) / count;
    }
    return {keptTerms(rows[0], relativeThreshold), keptTerms(rows[1], relativeThreshold),
            keptTerms(rows[2], relativeThreshold)};
}

void ShellPart::affineField(const std::array<SpaceVector, 3>& gradient, const SpaceVector& translation,
                            std::span<double> entityValues) const
{
    const std::size_t first = firstEntityUnknown();
    // the rotation of the field: the axial vector of the skew part of its gradient, ω = ½ (A32 − A23, A13 − A31,
    // A21 − A12)
    const SpaceVector omega{0.5 * (gradient[2][1] - gradient[1][2]), 0.5 * (gradient[0][2] - gradient[2][0]),
                            0.5 * (gradient[1][0] - gradient[0][1])};
    for (std::size_t ie = 0; ie < m_shells.size(); ++ie)
    {
        const ShellElementData el = element(ie);
        const ShellReferenceElement& re = referenceElement(el.cornerCount);
        const std::size_t n1 = re.h1Count;
        const std::size_t p = re.shearDegree;
        // the six rigid motions about the origin on the local unknowns: the coefficients c_i of the constant 1, and the
        // edge rotations of a unit rotation about each axis
        const std::array<Vector, 6> rigid = rigidMotionUnknowns(re, el.definition());
        Vector local(shellUnknownCount(re), 0.0);
        for (std::size_t i = 0; i < n1; ++i)
        {
            const double constant = rigid[0][3 * i];
            const SpaceVector& x = el.mapCoefficients[i];
            for (std::size_t c = 0; c < 3; ++c)
                local[3 * i + c] = translation[c] * constant + dot3(gradient[c], x);
        }
        for (std::size_t kk = 0; kk < el.cornerCount; ++kk)
            for (std::size_t l = 0; l <= p; ++l)
            {
                const std::size_t slot = 3 * n1 + re.hcurlCount + kk * (p + 1) + l;
                local[slot] = omega[0] * rigid[3][slot] + omega[1] * rigid[4][slot] + omega[2] * rigid[5][slot];
            }
        for (std::size_t i = 0; i < local.size(); ++i)
            if (el.dofs[i] >= first)
                entityValues[el.dofs[i] - first] = el.signs[i] * local[i];
    }
}

ShellDisplacementRows ShellPart::displacementRows(std::size_t shell, double xi, double eta,
                                                  const std::function<bool(const FunctionKind&)>& keep) const
{
    const ShellElementData el = element(shell);
    const ShellReferenceElement& re = referenceElement(el.cornerCount);
    const std::vector<ReferenceJet> h = h1At(re, el.definition(), xi, eta);
    std::array<std::map<std::size_t, double>, 3> rows;
    ShellDisplacementRows out;
    for (std::size_t i = 0; i < re.h1Count; ++i)
    {
        const double v = h[i][0];
        for (std::size_t c = 0; c < 3; ++c)
            out.point[c] += v * el.mapCoefficients[i][c];
        if ((keep && !keep(re.h1Kinds[i])) || v == 0.0)
            continue;
        for (std::size_t c = 0; c < 3; ++c)
            rows[c][el.dofs[3 * i + c]] += el.signs[3 * i + c] * v;
    }
    for (std::size_t c = 0; c < 3; ++c)
        for (const auto& [unknown, coefficient] : rows[c])
            out.rows[c].push_back({unknown, coefficient});
    return out;
}

std::vector<ShellDisplacementRows> ShellPart::displacementRowsAt(
    std::size_t shell, std::span<const ReferencePoint> points,
    const std::function<bool(const FunctionKind&)>& keep) const
{
    // the functions N_i / W at a point need the weight W and the basis only (h1At), not the tables of the element
    const ShellElementData& el = m_shells[shell];
    const ShellReferenceElement& re = referenceElement(el.cornerCount);
    const ShellElementDefinition def = el.definition();
    std::vector<ShellDisplacementRows> out;
    out.reserve(points.size());
    for (const ReferencePoint& point : points)
    {
        const std::vector<ReferenceJet> h = h1At(re, def, point[0], point[1]);
        std::array<std::map<std::size_t, double>, 3> rows;
        ShellDisplacementRows at;
        for (std::size_t i = 0; i < re.h1Count; ++i)
        {
            const double v = h[i][0];
            for (std::size_t c = 0; c < 3; ++c)
                at.point[c] += v * el.mapCoefficients[i][c];
            if ((keep && !keep(re.h1Kinds[i])) || v == 0.0)
                continue;
            for (std::size_t c = 0; c < 3; ++c)
                rows[c][el.dofs[3 * i + c]] += el.signs[3 * i + c] * v;
        }
        for (std::size_t c = 0; c < 3; ++c)
            for (const auto& [unknown, coefficient] : rows[c])
                at.rows[c].push_back({unknown, coefficient});
        out.push_back(std::move(at));
    }
    return out;
}

ReferencePoint ShellPart::edgeReferencePoint(std::size_t shell, std::size_t localEdge, double s) const
{
    const ShellElementData& el = m_shells[shell];
    const double tau = el.edgeSigns[localEdge] > 0.0 ? s : 1.0 - s;
    return referenceEdgePointAndDirection(el.cornerCount, localEdge, tau).first;
}

ShellEdgeRotationFrame ShellPart::edgeRotationFrame(std::size_t shell, std::size_t localEdge, double s) const
{
    const ShellElementData& el = m_shells[shell];
    const ShellReferenceElement& re = referenceElement(el.cornerCount);
    const ReferencePoint x = edgeReferencePoint(shell, localEdge, s);
    const PointGeometry g = geometryAt(re, el.definition(), x[0], x[1]);
    // X_s = s_k X_τ: the tangent of the map along the direction of the edge in the reference domain, turned to the
    // canonical direction of the edge
    const ReferencePoint direction = referenceEdgePointAndDirection(el.cornerCount, localEdge, 0.0).second;
    const SpaceVector xs = scl3(add3(scl3(g.a[0], direction[0]), scl3(g.a[1], direction[1])), el.edgeSigns[localEdge]);
    const double w = el.weight.evaluate(x[0], x[1]);
    return {unit3(xs), w * w * norm3(xs)};
}

} // namespace oos::elements::mixedHybrid
