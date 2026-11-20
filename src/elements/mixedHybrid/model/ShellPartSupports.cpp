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

// A point of the reference element is on an edge within this (of the parameter of the edge, and of its length across
// it): the round-off of a point placed on an edge by the shape functions of a face.
constexpr double ON_EDGE_TOLERANCE = 1e-9;
// Two edges of a shell at a point make a corner when 1 − cos² of the angle of their tangents exceeds this.
constexpr double CORNER_EDGES_APART = 1e-6;
// An edge is curved when a point of it leaves its chord by more than this fraction of the chord.
constexpr double CURVED_EDGE_RATIO = 1e-10;
// Two directions held at the two grids of an edge are one when they are parallel within this (1 − |cosine|).
constexpr double PARALLEL_TOLERANCE = 1e-10;

// The axes a grid holds in its translations (`first` 0) or its rotations (`first` 3): the basic axes of its flags and
// its directions that are no basic axis.
std::vector<SpaceVector> heldAxesOf(const ShellGridSupport& support, std::size_t first)
{
    std::vector<SpaceVector> axes;
    for (std::size_t c = 0; c < 3; ++c)
        if (support.held[first + c])
            axes.push_back(basisVector(c));
    for (const ShellSkewAxis& axis : support.skewAxes)
        if (axis.component >= first && axis.component < first + 3)
            axes.push_back(axis.direction);
    return axes;
}

// v is in the span of `axes` (unit vectors): what is left of it once its parts along them are taken off vanishes.
bool inSpanOf(const SpaceVector& v, const std::vector<SpaceVector>& axes)
{
    std::vector<SpaceVector> basis; // orthonormal, at most three
    for (const SpaceVector& axis : axes)
    {
        SpaceVector rest = axis;
        for (const SpaceVector& q : basis)
            rest = sub3(rest, scl3(q, dot3(rest, q)));
        if (norm3(rest) > INDEPENDENT_DIRECTION)
            basis.push_back(unit3(rest));
    }
    SpaceVector rest = v;
    for (const SpaceVector& q : basis)
        rest = sub3(rest, scl3(q, dot3(rest, q)));
    return dot3(rest, rest) < SPAN_TOLERANCE_SQUARED;
}

bool hasSkewRotation(const ShellGridSupport& support)
{
    return std::any_of(support.skewAxes.begin(), support.skewAxes.end(),
                       [](const ShellSkewAxis& axis) { return axis.component >= 3; });
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
        ShellGridSupport& merged = supportOf[support.grid];
        for (std::size_t c = 0; c < 6; ++c)
            if (support.held[c])
            {
                merged.held[c] = true;
                merged.values[c] = support.values[c];
            }
        if (support.skewAxes.empty())
            continue;
        // a grid that holds directions off the basic axes gives its held translation and rotation as whole vectors
        merged.values = support.values;
        merged.skewAxes.insert(merged.skewAxes.end(), support.skewAxes.begin(), support.skewAxes.end());
    }
    for (const ShellGridSupport& support : grids)
        if (support.grid < ng && support.followsRelation)
            supportOf[support.grid].followsRelation = true;
    // the grids of the outline of the shells: those of an edge that one shell has
    std::vector<bool> isOnOutline(ng, false);
    for (std::size_t e = 0; e < m_topology.edges.size(); ++e)
        if (m_topology.edgeElements[e].size() == 1)
        {
            isOnOutline[m_topology.edges[e].lower] = true;
            isOnOutline[m_topology.edges[e].higher] = true;
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

    // the supports along the edges whose two grids hold a component. The reference reads the external edges only and
    // states the others on the edges; the model files state supports at grid points only, so an edge inside the shells
    // is read too (as an external one) when one of its grids is inside them: a line of supports across a plate holds
    // the plate along it. An inner edge between two grids of the outline is not: the diagonal of a triangle at a
    // corner or the edge across a strip one element wide joins two supported sides, and holding it would hold the
    // shell between them. Two grids that follow a relation of the model (a rigid element) move the edge between them
    // wherever it is.
    constexpr std::size_t READ_AS_EXTERNAL = 1;
    for (std::size_t e = 0; e < m_topology.edges.size(); ++e)
    {
        const std::size_t ga = m_topology.edges[e].lower;
        const std::size_t gb = m_topology.edges[e].higher;
        const bool isExternal = m_topology.edgeElements[e].size() == 1;
        const bool followsRelation = supportOf[ga].followsRelation && supportOf[gb].followsRelation;
        if (!isExternal && !followsRelation && isOnOutline[ga] && isOnOutline[gb])
            continue;
        const EdgeSupport held =
            edgeHeld({}, m_topology.edges[e], READ_AS_EXTERNAL, supportOf[ga].held, supportOf[gb].held, true);
        // the directions off the basic axes: the edge reads them when both its grids hold some
        const bool hasSkew = !supportOf[ga].skewAxes.empty() && !supportOf[gb].skewAxes.empty();
        if (held == EdgeSupport{} && !hasSkew)
            continue;
        // An edge with a grid that follows a relation of the model (a rigid element) must move with it in every rigid
        // motion, or the relation would act as a link to the ground: it follows only when its two grids have their
        // three translations, and, where it is curved, their three rotations too (the rotation about the chord, which
        // moves the points of a curved edge, is not known from two translations). Otherwise it keeps to its grids.
        const bool readsRelation = supportOf[ga].followsRelation || supportOf[gb].followsRelation;
        const bool allTranslations = held.translations[0] && held.translations[1] && held.translations[2];
        const bool allRotations = held.rotations[0] && held.rotations[1] && held.rotations[2];
        if (readsRelation && (!allTranslations || hasSkew))
            continue;
        // with rotations held off the basic axes, the spans of the held rotation axes replace the flags of the basic
        // ones (the same tests when the axes are basic)
        const bool skewRotations = hasSkew && (hasSkewRotation(supportOf[ga]) || hasSkewRotation(supportOf[gb]));
        const std::vector<SpaceVector> rotationAxesA = heldAxesOf(supportOf[ga], 3);
        const std::vector<SpaceVector> rotationAxesB = heldAxesOf(supportOf[gb], 3);
        const std::vector<SpaceVector> translationAxesA = heldAxesOf(supportOf[ga], 0);
        const std::vector<SpaceVector> translationAxesB = heldAxesOf(supportOf[gb], 0);
        const bool bothTurn = !rotationAxesA.empty() && !rotationAxesB.empty();
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
            if (skewRotations && bothTurn)
            {
                rod.ta[c] = supportOf[ga].values[3 + c];
                rod.tb[c] = supportOf[gb].values[3 + c];
            }
        }
        // the directions across the edge whose slope a held rotation fixes: t × e_c (t × r for each rotation axis r
        // both grids hold, when they are off the basic axes)
        std::vector<SpaceVector> turning;
        if (!skewRotations)
        {
            for (std::size_t c = 0; c < 3; ++c)
                if (held.rotations[c])
                    turning.push_back(basisVector(c));
        }
        else
        {
            for (const SpaceVector& axis : rotationAxesA)
                if (inSpanOf(axis, rotationAxesB))
                    turning.push_back(axis);
        }
        for (const SpaceVector& axis : turning)
        {
            SpaceVector v = cross3(rod.tangent, axis);
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
        // [l][end]: the moment l of the function of the chord that is 1 at the grid `end` and 0 at the other
        std::vector<std::array<double, 2>> linearMoments(k - 1, std::array<double, 2>{});
        double largestOffset = 0.0; // of the points of the edge from its chord
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const PointGeometry g = edgeGeometry(def, re.h1Count, kk, gi);
            const auto tv = referenceEdgePointAndDirection(el.cornerCount, kk, re.edgeParameters[gi]).second;
            const double te = forward ? re.edgeParameters[gi] : 1.0 - re.edgeParameters[gi];
            const double sc = dot3(sub3(g.x, xa), rod.tangent) / rod.length;
            const double we = def.edgeWeights[kk][gi];
            // the rod gives the motion of the point of the chord under the point of the edge; the point itself, at
            // δ from it, moves by θ × δ more, θ the rotation of the rod there (linear between the grids): the motion
            // carried is then that of a rigid body in every rigid motion of the two grids, along a curved edge too
            const SpaceVector offset = sub3(sub3(g.x, xa), scl3(rod.tangent, sc * rod.length));
            largestOffset = std::max(largestOffset, norm3(offset));
            const SpaceVector theta = add3(scl3(rod.ta, 1.0 - sc), scl3(rod.tb, sc));
            const SpaceVector uv = add3(rod.at(sc), cross3(theta, offset));
            for (std::size_t l = 0; l + 1 < k; ++l)
            {
                const double f = re.edgeWeights[gi] * legendreValue(l, 2.0 * te - 1.0) * we;
                linearMoments[l][0] += f * (1.0 - sc);
                linearMoments[l][1] += f * sc;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    traceMoments[l][c] += f * uv[c];
                    for (std::size_t end = 0; end < 2; ++end)
                        for (std::size_t cp = 0; cp < 3; ++cp)
                        {
                            const double share = end == 0 ? 1.0 - sc : sc;
                            traceByTranslation[l][c][end][cp] += f * rod.translationRate(sc, end, c, cp);
                            traceByRotation[l][c][end][cp] +=
                                f * (rod.rotationRate(sc, end, c, cp) + share * cross3(basisVector(cp), offset)[c]);
                        }
                }
            }
            // the rotation about the edge: W² θ · X_τ_e, X_τ_e = s_k X_τ in the canonical direction
            const SpaceVector xt = scl3(add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1])), sk);
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
        // a curved edge that reads a relation without the rotations of its grids keeps to them; one that reads it with
        // some of them only follows in its displacement, not in its rotation
        if (readsRelation && !allRotations && largestOffset > CURVED_EDGE_RATIO * rod.length)
            continue;
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
                    // (off the basic axes, every component: the caller keeps those the grid holds)
                    for (std::size_t cp = 0; cp < 3; ++cp)
                        if ((held.rotations[cp] || (skewRotations && bothTurn)) &&
                            traceByRotation[l][c][end][cp] != 0.0)
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
            if (!hasSkew)
            {
                rotationHeld = rotationHeld && inHeldSpan(t, held.rotations);
                shearHeld = shearHeld && inHeldSpan(mu, held.rotations);
                normalHeld = normalHeld && inHeldSpan(g.nu, held.translations);
                continue;
            }
            rotationHeld = rotationHeld && inSpanOf(t, rotationAxesA) && inSpanOf(t, rotationAxesB);
            shearHeld = shearHeld && inSpanOf(mu, rotationAxesA) && inSpanOf(mu, rotationAxesB);
            normalHeld = normalHeld && inSpanOf(g.nu, translationAxesA) && inSpanOf(g.nu, translationAxesB);
        }
        const bool anyRotation =
            (hasSkew ? bothTurn : (held.rotations[0] || held.rotations[1] || held.rotations[2])) &&
            (!readsRelation || allRotations);
        if (rotationHeld && anyRotation)
            for (std::size_t l = 0; l <= p; ++l)
            {
                // the coefficient on P_l: (2l + 1) times the moment
                const double scale = static_cast<double>(2 * l + 1);
                std::vector<ShellGridTerm> carried;
                for (std::size_t end = 0; end < 2; ++end)
                    for (std::size_t cp = 0; cp < 3; ++cp)
                        if ((held.rotations[cp] || skewRotations) && rotationByRotation[l][end][cp] != 0.0)
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
        if (!hasSkew)
            continue;
        // the translations held along a direction d off the basic axes, the same at both grids: the trace of the
        // displacement along d is that of the chord, Σ_c d_c q_(l, c) = m_l(a) (u_a · d) + m_l(b) (u_b · d) for each
        // moment l (m_l the moments of the two linear functions of the chord, u · d the held component of the grid)
        bool differs = false;
        for (const ShellSkewAxis& axisA : supportOf[ga].skewAxes)
        {
            if (axisA.component >= 3)
                continue;
            // the same direction at the other grid; the same component along another direction there is a
            // displacement system that turns along the edge, which the edge does not carry
            const ShellSkewAxis* partner = nullptr;
            bool turns = false;
            for (const ShellSkewAxis& axisB : supportOf[gb].skewAxes)
            {
                if (axisB.component >= 3)
                    continue;
                if (std::fabs(dot3(axisA.direction, axisB.direction)) >= 1.0 - PARALLEL_TOLERANCE)
                    partner = &axisB;
                else
                    turns = turns || axisB.component == axisA.component;
            }
            if (partner == nullptr)
            {
                differs = differs || turns;
                continue;
            }
            const double sign = dot3(axisA.direction, partner->direction) < 0.0 ? -1.0 : 1.0;
            for (std::size_t l = 0; l + 1 < k; ++l)
            {
                ShellSupportRow row;
                for (std::size_t c = 0; c < 3; ++c)
                    if (axisA.direction[c] != 0.0)
                        row.unknowns.push_back({first + 3 * l + c, axisA.direction[c]});
                row.grids.push_back({ga, axisA.component, -linearMoments[l][0]});
                row.grids.push_back({gb, partner->component, -sign * linearMoments[l][1]});
                out.rows.push_back(std::move(row));
            }
        }
        if (differs)
            ++out.uncarriedEdges;
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

std::vector<std::pair<std::size_t, std::size_t>> ShellPart::rotationShellsAt(std::size_t grid) const
{
    std::vector<std::pair<std::size_t, std::size_t>> at = shellsAt(grid);
    const bool anyMoments =
        std::any_of(at.begin(), at.end(), [&](const auto& shell) { return carriesMoments(shell.first); });
    if (anyMoments)
        std::erase_if(at, [&](const auto& shell) { return !carriesMoments(shell.first); });
    return at;
}

std::optional<SpaceVector> ShellPart::commonNormalAt(std::size_t grid) const
{
    // Two normals are one when they differ by less than a degree (the kink the geometry report names).
    const double sameNormal = std::cos(1.0 * 3.14159265358979323846 / 180.0);
    const std::vector<std::pair<std::size_t, std::size_t>> at = rotationShellsAt(grid);
    SpaceVector first{};
    SpaceVector sum{};
    for (std::size_t i = 0; i < at.size(); ++i)
    {
        const ShellElementData el = element(at[i].first);
        const ReferencePoint x = referenceCorner(el.cornerCount, at[i].second);
        SpaceVector nu = geometryAt(referenceElement(el.cornerCount), el.definition(), x[0], x[1]).nu;
        if (i == 0)
            first = nu;
        // the side a shell's normal points to is its numbering's: the normals are compared up to their sign
        const double cosine = dot3(nu, first);
        if (std::fabs(cosine) < sameNormal)
            return std::nullopt;
        sum = add3(sum, scl3(nu, cosine < 0.0 ? -1.0 : 1.0));
    }
    if (at.empty())
        return std::nullopt;
    return unit3(sum);
}

std::array<std::vector<ShellPartTerm>, 3> ShellPart::meanRotationRows(std::size_t grid, double relativeThreshold) const
{
    const std::vector<std::pair<std::size_t, std::size_t>> at = rotationShellsAt(grid);
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

ShellPart::PointRows ShellPart::pointRows(std::size_t shell, double xi, double eta) const
{
    const ShellElementData& el = m_shells[shell];
    const ShellReferenceElement& re = referenceElement(el.cornerCount);
    const ShellElementDefinition def = el.definition();
    const std::vector<ReferenceJet> h = h1At(re, def, xi, eta);
    const Matrix rotation = rotationRowsOf(re, def, xi, eta);
    std::array<std::map<std::size_t, double>, 6> rows;
    PointRows out;
    for (std::size_t i = 0; i < re.h1Count; ++i)
    {
        const double v = h[i][0];
        for (std::size_t c = 0; c < 3; ++c)
            out.point[c] += v * el.mapCoefficients[i][c];
        if (v == 0.0)
            continue;
        for (std::size_t c = 0; c < 3; ++c)
            rows[c][el.dofs[3 * i + c]] += el.signs[3 * i + c] * v;
    }
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t i = 0; i < rotation.columns(); ++i)
            if (rotation(r, i) != 0.0)
                rows[3 + r][el.dofs[i]] += el.signs[i] * rotation(r, i);
    for (std::size_t r = 0; r < 6; ++r)
        for (const auto& [unknown, coefficient] : rows[r])
            if (coefficient != 0.0)
                out.rows[r].push_back({unknown, coefficient});
    return out;
}

ShellPart::PointRows ShellPart::pointRowsOnEdges(std::size_t shell, double xi, double eta) const
{
    PointRows out = pointRows(shell, xi, eta);
    const ShellElementData& el = m_shells[shell];
    // The edges of the reference element the point lies on (two at a corner), each with the parameter of the edge
    // from its lower grid.
    struct EdgeAt
    {
        std::size_t localEdge = 0;
        double s = 0.0;
    };
    std::array<EdgeAt, 2> edgesAt{};
    std::size_t edgeCount = 0;
    for (std::size_t k = 0; k < el.cornerCount && edgeCount < edgesAt.size(); ++k)
    {
        const auto [start, direction] = referenceEdgePointAndDirection(el.cornerCount, k, 0.0);
        const double lengthSquared = direction[0] * direction[0] + direction[1] * direction[1];
        const double tau = ((xi - start[0]) * direction[0] + (eta - start[1]) * direction[1]) / lengthSquared;
        const double across = (xi - start[0]) * direction[1] - (eta - start[1]) * direction[0];
        if (std::fabs(across) > ON_EDGE_TOLERANCE * std::sqrt(lengthSquared) || tau < -ON_EDGE_TOLERANCE ||
            tau > 1.0 + ON_EDGE_TOLERANCE)
            continue;
        const double clamped = std::clamp(tau, 0.0, 1.0);
        edgesAt[edgeCount++] = {k, el.edgeSigns[k] > 0.0 ? clamped : 1.0 - clamped};
    }
    if (edgeCount == 0)
        return out;

    // The directions d_i the rotation about each edge is carried by: θ = θ_element + Σ_i d_i (r_i − t_i · θ_element),
    // r_i the rotation about edge i and t_i its unit tangent. One edge: d = t. Two edges (a corner): the vectors of
    // their plane with d_i · t_j = δ_ij, so that θ · t_i = r_i for both; edges that run on (no corner between them)
    // carry nothing more than one does.
    std::array<ShellEdgeRotationFrame, 2> frames{};
    for (std::size_t i = 0; i < edgeCount; ++i)
        frames[i] = edgeRotationFrame(shell, edgesAt[i].localEdge, edgesAt[i].s);
    std::array<SpaceVector, 2> carriers{frames[0].tangent, SpaceVector{}};
    if (edgeCount == 2)
    {
        const double cosine = dot3(frames[0].tangent, frames[1].tangent);
        const double determinant = 1.0 - cosine * cosine;
        if (determinant < CORNER_EDGES_APART)
            edgeCount = 1;
        else
        {
            carriers[0] = scl3(sub3(frames[0].tangent, scl3(frames[1].tangent, cosine)), 1.0 / determinant);
            carriers[1] = scl3(sub3(frames[1].tangent, scl3(frames[0].tangent, cosine)), 1.0 / determinant);
        }
    }
    std::array<std::map<std::size_t, double>, 3> rows;
    for (std::size_t r = 0; r < 3; ++r)
        for (const ShellPartTerm& term : out.rows[3 + r])
            rows[r][term.unknown] += term.coefficient;
    const std::array<std::map<std::size_t, double>, 3> ofElement = rows;
    for (std::size_t i = 0; i < edgeCount; ++i)
    {
        const std::size_t edge = m_topology.edgeIndex.at(
            EdgeKey::of(m_input.shells[shell].corners.at(edgesAt[i].localEdge),
                        m_input.shells[shell].corners.at(edgesAt[i].localEdge + 1)));
        const std::size_t first = m_entities[edgeEntity(edge)].firstUnknown;
        for (std::size_t r = 0; r < 3; ++r)
        {
            if (carriers[i][r] == 0.0)
                continue;
            // − d_i (t_i · θ_element): the rotation of the element about the edge taken off
            for (std::size_t c = 0; c < 3; ++c)
                for (const auto& [unknown, coefficient] : ofElement[c])
                    rows[r][unknown] -= carriers[i][r] * frames[i].tangent[c] * coefficient;
            // + d_i r_i, r_i = Σ_l a_l P_l(2s − 1) / (W² |X_s|)
            for (std::size_t l = 0; l < m_input.degree; ++l)
                rows[r][first + edgeRotationDof(l)] +=
                    carriers[i][r] * legendreValue(l, 2.0 * edgesAt[i].s - 1.0) / frames[i].measure;
        }
    }
    for (std::size_t r = 0; r < 3; ++r)
    {
        out.rows[3 + r].clear();
        for (const auto& [unknown, coefficient] : rows[r])
            if (coefficient != 0.0)
                out.rows[3 + r].push_back({unknown, coefficient});
    }
    return out;
}

ShellPart::EdgeMoments ShellPart::edgeMoments(std::size_t edge) const
{
    EdgeMoments out;
    if (m_topology.edgeElements[edge].empty())
        return out;
    const std::size_t k = m_input.degree;
    // the edge as its first shell runs it (the same curve from every shell of the edge)
    const std::size_t ie = m_topology.edgeElements[edge].front();
    const ShellElementData el = element(ie);
    const ShellReferenceElement& re = referenceElement(el.cornerCount);
    const ShellElementDefinition def = el.definition();
    const std::size_t kk = localEdgeOf(ie, edge);
    const std::size_t first = m_entities[edgeEntity(edge)].firstUnknown;
    const double sk = el.edgeSigns[kk];
    const std::size_t pointCount = re.edgeParameters.size();
    out.displacementWeights.assign(k - 1, std::vector<double>(pointCount, 0.0));
    out.rotationWeights.assign(k, std::vector<SpaceVector>(pointCount, SpaceVector{}));
    for (std::size_t l = 0; l + 1 < k; ++l)
        for (std::size_t c = 0; c < 3; ++c)
            out.displacementUnknowns.push_back(first + 3 * l + c);
    for (std::size_t l = 0; l < k; ++l)
        out.rotationUnknowns.push_back(first + edgeRotationDof(l));
    for (std::size_t gi = 0; gi < pointCount; ++gi)
    {
        const PointGeometry g = edgeGeometry(def, re.h1Count, kk, gi);
        const auto tv = referenceEdgePointAndDirection(el.cornerCount, kk, re.edgeParameters[gi]).second;
        // the parameter of the edge from its lower grid, and the weight W of the map there
        const double s = sk > 0.0 ? re.edgeParameters[gi] : 1.0 - re.edgeParameters[gi];
        const double we = def.edgeWeights[kk][gi];
        out.points.push_back(g.x);
        out.parameters.push_back(s);
        for (std::size_t l = 0; l + 1 < k; ++l)
            out.displacementWeights[l][gi] = re.edgeWeights[gi] * legendreValue(l, 2.0 * s - 1.0) * we;
        // X_s = s_k X_τ: the tangent of the map in the canonical direction of the edge
        const SpaceVector xs = scl3(add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1])), sk);
        for (std::size_t l = 0; l < k; ++l)
            out.rotationWeights[l][gi] = scl3(
                xs, static_cast<double>(2 * l + 1) * re.edgeWeights[gi] * legendreValue(l, 2.0 * s - 1.0) * we * we);
    }
    return out;
}

ShellPart::InteriorMoments ShellPart::interiorMoments(std::size_t shell) const
{
    InteriorMoments out;
    const ShellElementData el = element(shell);
    const ShellReferenceElement& re = referenceElement(el.cornerCount);
    const InteriorProjection& projection = projectionOf(el.cornerCount);
    const std::size_t n1 = re.h1Count;
    const std::size_t interiorCount = projection.interior.size();
    const std::size_t pointCount = re.points.size();
    if (interiorCount == 0)
        return out;
    // a(i, q) = Σ_j G⁻¹(i, j) w_q N_j(q): the coefficient of the interior function i from the value at point q
    Matrix a(interiorCount, pointCount);
    std::vector<double> mapWeights(pointCount, 1.0);
    for (std::size_t q = 0; q < pointCount; ++q)
    {
        // the point of the surface X = Σ_i N_i x_i / W and the denominator W of the map there
        mapWeights[q] = el.weight.evaluate(re.points[q][0], re.points[q][1]);
        SpaceVector point{};
        for (std::size_t i = 0; i < n1; ++i)
            point = add3(point, scl3(el.mapCoefficients[i], re.h1[q * n1 + i][0]));
        out.points.push_back(scl3(point, 1.0 / mapWeights[q]));
        for (std::size_t j = 0; j < interiorCount; ++j)
        {
            const double f = re.weights[q] * re.h1[q * n1 + projection.interior[j]][0];
            for (std::size_t i = 0; i < interiorCount; ++i)
                a(i, q) += projection.gramInverse(i, j) * f;
        }
    }
    for (std::size_t i = 0; i < interiorCount; ++i)
        for (std::size_t c = 0; c < 3; ++c)
        {
            // the unknown of the model is its sign times the coefficient of the element (ShellElementData::local)
            const std::size_t local = 3 * projection.interior[i] + c;
            const double sign = el.signs[local];
            out.unknowns.push_back(el.dofs[local]);
            std::vector<double> weights(pointCount);
            for (std::size_t q = 0; q < pointCount; ++q)
                weights[q] = sign * a(i, q) * mapWeights[q];
            out.weights.push_back(std::move(weights));
            // − Σ_q a(i, q) N_b(q) on each corner and edge function b
            std::map<std::size_t, double> row;
            for (std::size_t b = 0; b < n1; ++b)
            {
                if (re.h1Kinds[b].isInterior())
                    continue;
                double carried = 0.0;
                for (std::size_t q = 0; q < pointCount; ++q)
                    carried += a(i, q) * re.h1[q * n1 + b][0];
                row[el.dofs[3 * b + c]] -= sign * carried * el.signs[3 * b + c];
            }
            out.terms.push_back(keptTerms(row, 0.0));
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

ShellEdgeFibre ShellPart::edgeFibre(std::size_t shell, std::size_t localEdge, double s) const
{
    const ShellElementData& el = m_shells[shell];
    const ShellPartShell& input = m_input.shells[shell];
    const ReferencePoint x = edgeReferencePoint(shell, localEdge, s);
    const PointGeometry g = geometryAt(referenceElement(el.cornerCount), el.definition(), x[0], x[1]);
    ShellEdgeFibre fibre;
    fibre.point = g.x;
    fibre.normal = g.nu;
    fibre.offset = el.offset;
    fibre.thickness = input.section.referenceThickness;
    if (input.gridThicknesses)
    {
        // linear between the two corners of the edge, tau the parameter of the shell along it
        const double tau = el.edgeSigns[localEdge] > 0.0 ? s : 1.0 - s;
        const std::size_t next = (localEdge + 1) % el.cornerCount;
        fibre.thickness = (1.0 - tau) * (*input.gridThicknesses)[localEdge] + tau * (*input.gridThicknesses)[next];
    }
    return fibre;
}

} // namespace oos::elements::mixedHybrid
