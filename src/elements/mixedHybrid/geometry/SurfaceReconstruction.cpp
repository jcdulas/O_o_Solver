#include "SurfaceReconstruction.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace oos::elements::mixedHybrid::detail
{

namespace
{

// Below this squared sine of the angle of three grids (relative to the product of the squared lengths) they are on a
// straight line: no circle.
constexpr double STRAIGHT_LINE_TOLERANCE = 1e-24;
// Below this sine of the angle of two boundary edges (relative to the product of their lengths) the boundary is
// straight at a grid.
constexpr double STRAIGHT_BOUNDARY_TOLERANCE = 1e-9;
// Below this norm the cross product of the two line tangents of an element at a grid gives no normal.
constexpr double PARALLEL_TANGENTS = 1e-12;
// Gauss-Newton iterations of a fan normal on the unit sphere.
constexpr int FAN_NORMAL_ITERATIONS = 12;
// The highest level of the conditions of a fan normal (see fanNormalAt).
constexpr std::size_t LAST_FAN_LEVEL = 4;
// The coefficients of the quadric fitted at a grid, and the least number of other grids it needs.
constexpr std::size_t QUADRIC_COEFFICIENTS = 8;
constexpr std::size_t QUADRIC_LEAST_GRIDS = 9;
// The relative norm below which a column of the fit is left out (a plane), and the ridge of its normal equations.
constexpr double QUADRIC_COLUMN_TOLERANCE = 1e-14;
constexpr double QUADRIC_RIDGE = 1e-12;
// The variance added to the residual of the fit: a perfect fit is still known to the round-off.
constexpr double QUADRIC_VARIANCE_FLOOR = 1e-20;
// Guards of divisions whose terms may all vanish.
constexpr double TINY = 1e-300;

// A seed direction not along n: x, unless n is near x.
Vector3 seedAcross(const Vector3& n)
{
    return std::abs(n[0]) < 0.6 ? Vector3{1.0, 0.0, 0.0} : Vector3{0.0, 1.0, 0.0};
}

// The position of grid g among the grids of a shell.
std::size_t cornerOf(const ShellCorners& shell, std::size_t g)
{
    for (std::size_t k = 0; k < shell.count; ++k)
        if (shell.grids[k] == g)
            return k;
    throw base::InternalError("mesh reading: grid not in the element");
}

// The grids next to grid g in a shell: the next one, then the previous one.
std::array<std::size_t, 2> neighboursIn(const ShellCorners& shell, std::size_t g)
{
    const std::size_t nv = shell.count;
    const std::size_t p = cornerOf(shell, g);
    return {shell.grids[(p + 1) % nv], shell.grids[(p + nv - 1) % nv]};
}

void pushUnique(std::vector<std::size_t>& list, std::size_t value)
{
    if (std::find(list.begin(), list.end(), value) == list.end())
        list.push_back(value);
}

Vector3 negated(const Vector3& v)
{
    return scl3(v, -1.0);
}

} // namespace

SurfaceReconstruction::SurfaceReconstruction(const MeshReadingInput& input, const Topology& topology)
    : m_input(input), m_topology(topology)
{
    for (std::size_t e = 0; e < topology.edges.size(); ++e)
    {
        if (topology.edgeElements[e].empty())
            continue;
        m_nodeEdges[topology.edges[e].lower].push_back(e);
        m_nodeEdges[topology.edges[e].higher].push_back(e);
    }
}

const std::vector<std::size_t>& SurfaceReconstruction::elementsOf(std::size_t a, std::size_t b) const
{
    return m_topology.edgeElements[m_topology.indexOf(a, b)];
}

const Vector3* SurfaceReconstruction::sectionNormal(std::size_t a, std::size_t b) const
{
    const auto it = m_section.find({a, b});
    return it == m_section.end() ? nullptr : &it->second;
}

// The grid d of element e such that a − d is its other edge at grid a.
// Reference: Reconstruction::other.
std::size_t SurfaceReconstruction::other(std::size_t e, std::size_t a, std::size_t b) const
{
    const std::array<std::size_t, 2> next = neighboursIn(shell(e), a);
    return next[0] == b ? next[1] : next[0];
}

// The unit direction into element e, normal to its edge a − d.
// Reference: Reconstruction::across.
Vector3 SurfaceReconstruction::across(std::size_t e, std::size_t a, std::size_t d) const
{
    const ShellCorners& s = shell(e);
    Vector3 centroid{};
    for (std::size_t k = 0; k < s.count; ++k)
        centroid = add3(centroid, scl3(position(s.grids[k]), 1.0 / static_cast<double>(s.count)));
    const Vector3 ed = unit3(sub3(position(d), position(a)));
    const Vector3 v = sub3(centroid, position(a));
    return unit3(sub3(v, scl3(ed, dot3(v, ed))));
}

// The element that faces e across its edge a − d: the other element of an edge of two; at a junction of more, the
// element whose direction into it is the most opposite, when e is the most opposite for it too.
// Reference: Reconstruction::facing.
std::optional<std::size_t> SurfaceReconstruction::facing(std::size_t e, std::size_t a, std::size_t d)
{
    const EdgeKey key = EdgeKey::of(a, d);
    auto found = m_facing.find(key);
    if (found == m_facing.end())
    {
        const std::vector<std::size_t>& es = m_topology.edgeElements[m_topology.indexOf(a, d)];
        std::map<std::size_t, std::size_t> pair;
        if (es.size() == 2)
        {
            pair[es[0]] = es[1];
            pair[es[1]] = es[0];
        }
        else if (es.size() > 2)
        {
            std::vector<Vector3> dirs;
            for (const std::size_t el : es)
                dirs.push_back(across(el, key.lower, key.higher));
            std::vector<std::size_t> best(es.size());
            for (std::size_t i = 0; i < es.size(); ++i)
            {
                // the first of the most opposite directions
                std::optional<std::size_t> bestOther;
                for (std::size_t o = 0; o < es.size(); ++o)
                    if (o != i && (!bestOther || dot3(dirs[i], dirs[o]) < dot3(dirs[i], dirs[*bestOther])))
                        bestOther = o;
                best[i] = *bestOther;
            }
            for (std::size_t i = 0; i < es.size(); ++i)
            {
                const std::size_t o = best[i];
                if (best[o] == i && dot3(dirs[i], dirs[o]) < 0.0)
                    pair[es[i]] = es[o];
            }
        }
        found = m_facing.emplace(key, std::move(pair)).first;
    }
    const auto it = found->second.find(e);
    if (it == found->second.end())
        return std::nullopt;
    return it->second;
}

// Reference: Reconstruction::has_triangle.
bool SurfaceReconstruction::hasTriangle(std::size_t a) const
{
    const auto it = m_nodeEdges.find(a);
    if (it == m_nodeEdges.end())
        return false;
    for (const std::size_t e : it->second)
        for (const std::size_t el : m_topology.edgeElements[e])
            if (shell(el).count == 3)
                return true;
    return false;
}

// Reference: Reconstruction::triangle_edge.
bool SurfaceReconstruction::triangleEdge(std::size_t a, std::size_t b) const
{
    for (const std::size_t el : elementsOf(a, b))
        if (shell(el).count == 3)
            return true;
    return false;
}

// The elements around grid a from element e away from its edge a − b, each with the grid of its far edge at a, and
// whether the walk came back to edge a − b (a closed fan) or met the boundary of the mesh.
// Reference: Reconstruction::ring.
std::pair<std::vector<SurfaceReconstruction::RingStep>, bool> SurfaceReconstruction::ring(std::size_t e, std::size_t a,
                                                                                          std::size_t b)
{
    const auto edgesAtA = m_nodeEdges.find(a);
    const std::size_t limit = (edgesAtA == m_nodeEdges.end() ? 0 : edgesAtA->second.size()) + 1;
    std::vector<RingStep> out{{e, other(e, a, b)}};
    while (true)
    {
        const auto [el, far] = out.back();
        if (far == b)
            return {out, true};
        const std::optional<std::size_t> f = facing(el, a, far);
        if (!f || out.size() >= limit)
            return {out, false};
        out.emplace_back(*f, other(*f, a, far));
    }
}

// The grids c such that edge a − c continues edge a → b at a, each with "the continuation is smooth".
//
// Between quadrilaterals the edge that continues a → b is the one across one quadrilateral on each side: a
// quadrilateral is a quarter of a turn.
//
// At a grid that a triangle has, nothing says what part of a turn a triangle is. Inside the mesh the edge that
// continues a → b is then the one that faces it: the edge at a whose chord is the most opposite to that of a → b,
// when a → b is the most opposite to it too (no threshold). An edge that no edge faces back ends there. On the
// boundary of the mesh only the boundary goes on, from one boundary edge to the other, unless the grid has a single
// element, or two triangles and nothing else: nothing tells a corner cut by a diagonal from a straight boundary with
// two triangles, and a corner is the safe reading. The other lines end on the boundary.
// Reference: Reconstruction::candidates (cached: a function of the mesh alone).
const std::map<std::size_t, bool>& SurfaceReconstruction::candidates(std::size_t a, std::size_t b)
{
    if (const auto it = m_candidates.find({a, b}); it != m_candidates.end())
        return it->second;
    std::map<std::size_t, bool> found = candidatesAt(a, b);
    return m_candidates.emplace(GridPair{a, b}, std::move(found)).first->second;
}

std::map<std::size_t, bool> SurfaceReconstruction::candidatesAt(std::size_t a, std::size_t b)
{
    std::map<std::size_t, bool> found;
    const std::vector<std::size_t>& es = elementsOf(a, b);
    if (!hasTriangle(a))
    {
        for (const std::size_t e : es)
        {
            const std::size_t dd = other(e, a, b);
            if (const std::optional<std::size_t> f = facing(e, a, dd))
            {
                const std::size_t c = other(*f, a, dd);
                // the line goes on through a only if the edge it crosses there is smooth
                const bool smooth = !isRaw(a, dd);
                bool& entry = found.try_emplace(c, false).first->second;
                entry = entry || smooth;
            }
        }
        return found;
    }
    const bool boundary = es.size() == 1;
    for (const std::size_t e : es)
    {
        const std::pair<std::vector<RingStep>, bool> walk = ring(e, a, b);
        const std::vector<RingStep>& steps = walk.first;
        const bool closed = walk.second;
        std::size_t quadrilaterals = 0;
        for (const RingStep& r : steps)
            if (shell(r.first).count == 4)
                ++quadrilaterals;
        const std::size_t triangles = steps.size() - quadrilaterals;
        // the edges crossed before reaching the far edge of element i are all smooth
        const auto smoothUpTo = [&](std::size_t i)
        {
            for (std::size_t m = 0; m < i; ++m)
                if (isRaw(a, steps[m].second))
                    return false;
            return true;
        };
        std::optional<std::size_t> step;
        if (closed && triangles == 0)
        {
            // quadrilaterals only on this side of a junction: across two of them
            if (steps.size() >= 2)
                step = 1;
        }
        else if (closed)
        {
            const auto direction = [&](std::size_t h) { return unit3(sub3(position(h), position(a))); };
            // the most opposite edge at a (the lowest grid number of two equal ones)
            const auto opposite = [&](std::size_t h) -> std::optional<std::size_t>
            {
                const Vector3 dh = direction(h);
                std::optional<std::size_t> best;
                double bestDot = 0.0;
                for (const RingStep& r : steps)
                {
                    const std::size_t c = r.second;
                    if (c == h)
                        continue;
                    const double dc = dot3(dh, direction(c));
                    if (!best || dc < bestDot || (dc == bestDot && c < *best))
                    {
                        best = c;
                        bestDot = dc;
                    }
                }
                return best;
            };
            const std::optional<std::size_t> c = opposite(b);
            if (c && opposite(*c) == b)
                for (std::size_t i = 0; i < steps.size(); ++i)
                    if (steps[i].second == *c)
                    {
                        step = i;
                        break;
                    }
        }
        else if (boundary)
        {
            // the boundary goes on through a when the declared through grids say so, else when the elements at a
            // make more than a corner (3 per quadrilateral, 2 per triangle: at least 5)
            const bool goesOn =
                m_input.throughGrids ? m_input.throughGrids->contains(a) : 3 * quadrilaterals + 2 * triangles >= 5;
            if (goesOn)
                step = steps.size() - 1;
        }
        if (step && steps[*step].second != b)
        {
            const bool smooth = smoothUpTo(*step);
            bool& entry = found.try_emplace(steps[*step].second, false).first->second;
            entry = entry || smooth;
        }
    }
    return found;
}

// The mesh line of edge a − b says something of its tangent at a: the line goes on at a, or ends there and goes on
// at b.
// Reference: Reconstruction::informed.
bool SurfaceReconstruction::informed(std::size_t a, std::size_t b)
{
    const Continuation c = continuation(a, b);
    return c.smooth || (c.openEnd && continuation(b, a).smooth);
}

// The grid before a on the mesh line of edge a → b, when the line continues there smoothly and without branching.
// Reference: Reconstruction::previous.
std::optional<std::size_t> SurfaceReconstruction::previous(std::size_t a, std::size_t b)
{
    const std::map<std::size_t, bool>& found = candidates(a, b);
    std::vector<std::size_t> smooth;
    for (const auto& [c, s] : found)
        if (s && c != b)
            smooth.push_back(c);
    if (smooth.size() == 1 && found.size() == 1)
        return smooth[0];
    return std::nullopt;
}

// The circle through three consecutive grids of a mesh line (in that order): its curvature, its centre and its
// binormal T × N, from the circumcentre
//   o = [(q·q) (n × p) + (p·p) (q × n)] / (2 |n|²),   p = x0 − x1, q = x2 − x1, n = p × q,
// relative to x1. None on a straight line.
// Reference: Reconstruction::circle.
std::optional<SurfaceReconstruction::Circle> SurfaceReconstruction::circle(const Vector3& x0, const Vector3& x1,
                                                                           const Vector3& x2)
{
    const Vector3 p = sub3(x0, x1);
    const Vector3 q = sub3(x2, x1);
    const Vector3 n = cross3(p, q);
    const double n2 = dot3(n, n);
    if (n2 <= STRAIGHT_LINE_TOLERANCE * dot3(p, p) * dot3(q, q))
        return std::nullopt;
    const Vector3 o = scl3(add3(scl3(cross3(n, p), dot3(q, q)), scl3(cross3(q, n), dot3(p, p))), 0.5 / n2);
    const Vector3 t = sub3(x2, x0);
    return Circle{1.0 / norm3(o), add3(x1, o), unit3(cross3(t, o))};
}

// The correction of the tangent at a grid for the variation of the curvature along the mesh line. The tangent of the
// circle through three grids of parameters p_j is off by a sixth of the third derivative of the position times the
// product of (p_i − p_j), and the part of that derivative normal to the tangent is dκ/ds N + κ τ B (the Frenet
// formulas). dκ/ds and τ come from the circles through the neighbouring triples: both vanish on a circle whatever the
// spacing, so a circle keeps its exact tangent. `line`: the grids along the line in the direction of the tangent;
// `it`: the index of the grid of the tangent in it; `base`: the three grids of its circle; `product`: that product.
// The derivatives are those, at the grid of the tangent, of the polynomial through the circles of the line: three
// inside a line (centred), two next to a boundary and at an open end.
// Reference: Reconstruction::curvature_variation.
Vector3 SurfaceReconstruction::curvatureVariation(const std::vector<Vector3>& line, std::size_t it,
                                                  const std::array<Vector3, 3>& base, double product)
{
    const std::optional<Circle> own = circle(base[0], base[1], base[2]);
    if (!own)
        return {};
    // the abscissae of the grids, then the circles through the consecutive triples
    std::vector<double> s(line.size(), 0.0);
    for (std::size_t k = 1; k < line.size(); ++k)
        s[k] = s[k - 1] + norm3(sub3(line[k], line[k - 1]));
    struct Sample
    {
        double abscissa;
        double curvature;
        Vector3 binormal;
    };
    std::vector<Sample> samples;
    for (std::size_t m = 1; m + 1 < line.size(); ++m)
    {
        const std::optional<Circle> c = circle(line[m - 1], line[m], line[m + 1]);
        if (c)
            samples.push_back({s[m] - s[it], c->curvature, c->binormal});
        else
            samples.push_back({s[m] - s[it], 0.0, own->binormal});
    }
    // the three samples nearest to the grid of the tangent
    while (samples.size() > 3)
    {
        if (std::abs(samples.front().abscissa) > std::abs(samples.back().abscissa))
            samples.erase(samples.begin());
        else
            samples.pop_back();
    }
    if (samples.size() < 2)
        return {};
    // the weights of the derivative at abscissa 0 of the interpolating polynomial (Lagrange)
    std::vector<double> w;
    if (samples.size() == 2)
    {
        const double h = samples[1].abscissa - samples[0].abscissa;
        w = {-1.0 / h, 1.0 / h};
    }
    else
    {
        const double x0 = samples[0].abscissa;
        const double x1 = samples[1].abscissa;
        const double x2 = samples[2].abscissa;
        w = {(-x1 - x2) / ((x0 - x1) * (x0 - x2)), (-x0 - x2) / ((x1 - x0) * (x1 - x2)),
             (-x0 - x1) / ((x2 - x0) * (x2 - x1))};
    }
    double dk = 0.0;
    Vector3 db{};
    for (std::size_t q = 0; q < samples.size(); ++q)
    {
        dk += w[q] * samples[q].curvature;
        db = add3(db, scl3(samples[q].binormal, w[q]));
    }
    const Vector3 n = unit3(sub3(own->centre, line[it]));
    // dB/ds = −τ N
    const double torsion = -dot3(db, n);
    return scl3(add3(scl3(n, dk), scl3(own->binormal, own->curvature * torsion)), product / 6.0);
}

// How far the circle through the three grids line[mid − 1 .. mid + 1] is from the circles through the neighbouring
// triples of the line, as an angle of the tangent: half the spacing times the difference of the curvatures (and of
// the planes). Zero on a circle; of the second order on a smooth line; of the order of the kink where the line
// zigzags or kinks, including a symmetric kink, where the variation of the curvature read on both sides cancels.
// Reference: Reconstruction::circle_spread.
double SurfaceReconstruction::circleSpread(const std::vector<Vector3>& line, std::size_t mid)
{
    const auto at = [&](std::size_t m) { return circle(line[m - 1], line[m], line[m + 1]); };
    if (mid < 1 || mid + 1 >= line.size())
        return 0.0;
    const std::optional<Circle> base = at(mid);
    const double k0 = base ? base->curvature : 0.0;
    double out = 0.0;
    for (const std::size_t m : {mid - 1, mid + 1})
    {
        if (m < 1 || m + 1 >= line.size())
            continue;
        const std::optional<Circle> neighbour = at(m);
        const double k1 = neighbour ? neighbour->curvature : 0.0;
        const double turn = base && neighbour ? k0 * norm3(sub3(neighbour->binormal, base->binormal)) : 0.0;
        const double h = norm3(sub3(line[m], line[mid]));
        out = std::fmax(out, 0.5 * h * std::sqrt((k1 - k0) * (k1 - k0) + turn * turn));
    }
    return out;
}

// The unit normal of the plane of the boundary of the mesh at grid a, for the mesh line that ends at a along edge
// a → b: the plane through a and its two neighbours on the boundary (through a and the next two grids of the
// boundary at a corner). On a straight boundary every plane through it qualifies: the one whose normal is nearest to
// the tangent t is returned. None when a is not a plain boundary grid.
// Reference: Reconstruction::boundary_plane.
std::optional<Vector3> SurfaceReconstruction::boundaryPlane(std::size_t a, std::size_t b, const Vector3& t)
{
    const auto edgesAtA = m_nodeEdges.find(a);
    if (edgesAtA == m_nodeEdges.end())
        return std::nullopt;
    std::vector<std::size_t> near;
    for (const std::size_t e : edgesAtA->second)
    {
        if (m_topology.edgeElements[e].size() != 1)
            continue;
        const EdgeKey edge = m_topology.edges[e];
        const std::size_t g = edge.lower == a ? edge.higher : edge.lower;
        if (g != b)
            near.push_back(g);
    }
    std::size_t ip = 0;
    std::size_t iq = 0;
    if (near.size() == 2)
    {
        // where the boundary does not go on through a (a corner of the boundary, met by a line that ends there: the
        // diagonal of a triangle), the plane through a and its two neighbours is that of the surface, not one the
        // surface could be symmetric about
        if (!candidates(a, near[0]).contains(near[1]))
            return std::nullopt;
        ip = near[0];
        iq = near[1];
    }
    else if (near.size() == 1)
    {
        const std::optional<std::size_t> before = previous(near[0], a);
        if (!before)
            return std::nullopt;
        ip = near[0];
        iq = *before;
    }
    else
    {
        return std::nullopt;
    }
    const Vector3 d1 = sub3(position(ip), position(a));
    const Vector3 d2 = sub3(position(iq), position(a));
    const Vector3 n = cross3(d1, d2);
    if (norm3(n) > STRAIGHT_BOUNDARY_TOLERANCE * norm3(d1) * norm3(d2))
        return unit3(n);
    const Vector3 l = unit3(sub3(position(iq), position(ip)));
    const Vector3 v = sub3(t, scl3(l, dot3(t, l)));
    if (norm3(v) > 0.0)
        return unit3(v);
    return std::nullopt;
}

// The positions of the grids after b on the mesh line of edge a → b (at most `count`).
// Reference: Reconstruction::chain.
std::vector<Vector3> SurfaceReconstruction::chain(std::size_t a, std::size_t b, std::size_t count)
{
    std::vector<std::size_t> visited{a, b};
    std::vector<Vector3> out;
    while (out.size() < count)
    {
        const std::size_t p = visited[visited.size() - 2];
        const std::size_t q = visited[visited.size() - 1];
        const std::optional<std::size_t> next = previous(q, p);
        if (!next)
            break;
        out.push_back(position(*next));
        visited.push_back(*next);
    }
    return out;
}

// (smooth, tangent, open end) at grid a of edge a → b. The tangent is the mean of the tangents at a of the parabolas
// through c, a, b over the smooth continuations c; where the line branches (a grid shared by three or five elements)
// their spread is its uncertainty.
// Reference: Reconstruction::continuation.
SurfaceReconstruction::Continuation SurfaceReconstruction::continuation(std::size_t a, std::size_t b)
{
    if (const auto it = m_continuation.find({a, b}); it != m_continuation.end())
        return it->second;
    const std::map<std::size_t, bool>& found = candidates(a, b);
    const Vector3 d = sub3(position(b), position(a));
    const double length = norm3(d);
    const Vector3 u = scl3(d, 1.0 / length);
    Vector3 sum{};
    std::vector<Vector3> list;
    for (const auto& [c, smooth] : found)
    {
        if (!smooth || c == b)
            continue;
        const Vector3 dc = sub3(position(c), position(a));
        const double lc = norm3(dc);
        // the tangent of the parabola through c, a, b
        const Vector3 t = unit3(sub3(scl3(u, lc), scl3(dc, length / lc)));
        sum = add3(sum, t);
        list.push_back(t);
    }
    const bool any = !list.empty();
    const Continuation out{any, any ? unit3(sum) : u, found.empty()};
    double spread = 0.0;
    for (const Vector3& t : list)
        spread = std::fmax(spread, norm3(sub3(t, out.tangent)));
    m_branch[{a, b}] = spread;
    m_continuation[{a, b}] = out;
    return out;
}

// The unit tangent at grid a of the mesh line of edge a → b.
// Reference: Reconstruction::line_tangent.
Vector3 SurfaceReconstruction::lineTangent(std::size_t a, std::size_t b)
{
    if (const auto it = m_lineTangent.find({a, b}); it != m_lineTangent.end())
        return it->second;
    const Continuation own = continuation(a, b);
    Vector3 t = own.tangent;
    bool mirrored = false;
    if (!own.smooth && own.openEnd)
    {
        const Continuation back = continuation(b, a);
        if (back.smooth)
        {
            const Vector3 u = unit3(sub3(position(b), position(a)));
            // the mirror image of the tangent at the other end
            t = sub3(back.tangent, scl3(u, 2.0 * dot3(back.tangent, u)));
            mirrored = true;
        }
    }
    t = unit3(t);
    const Vector3 tCircle = t;
    double spread = 0.0;
    const Vector3 xa = position(a);
    const Vector3 xb = position(b);
    if (own.smooth)
    {
        // inside a mesh line: grids ..., c, a, b, ... in the direction of the tangent
        const std::vector<Vector3> before = chain(b, a, 2);
        const std::vector<Vector3> after = chain(a, b, 1);
        if (!before.empty() && previous(a, b))
        {
            const Vector3 xc = before[0];
            std::vector<Vector3> line(before.rbegin(), before.rend());
            const std::size_t it = line.size();
            line.push_back(xa);
            line.push_back(xb);
            line.insert(line.end(), after.begin(), after.end());
            const double hb = norm3(sub3(xb, xa));
            const double hc = norm3(sub3(xc, xa));
            const Vector3 correction = curvatureVariation(line, it, {xc, xa, xb}, -hb * hc);
            t = unit3(add3(t, correction));
            spread = circleSpread(line, it);
        }
    }
    else if (mirrored)
    {
        // open end of a mesh line: grids a, b, b2, b3
        const std::vector<Vector3> after = chain(a, b, 2);
        if (!after.empty())
        {
            std::vector<Vector3> line{xa, xb};
            line.insert(line.end(), after.begin(), after.end());
            const double h1 = norm3(sub3(xb, xa));
            const double h2 = norm3(sub3(after[0], xb));
            const Vector3 correction = curvatureVariation(line, 0, {xa, xb, after[0]}, h1 * (h1 + h2));
            const Vector3 t0 = t;
            t = unit3(add3(t, correction));
            spread = circleSpread(line, 1);
            // Symmetry or cut? A surface symmetric about the plane of its boundary meets that plane at a right
            // angle: the line would end along the normal np of the plane. delta: how far the prolonged tangent is
            // from np; sigma: how uncertain that tangent is, measured by the correction it has just received. The
            // tangent moves towards np by sigma² / (sigma² + delta²): all the way when np is within the uncertainty
            // (symmetry), not at all when it is far outside (cut).
            if (std::optional<Vector3> plane = boundaryPlane(a, b, t))
            {
                const Vector3 np = dot3(*plane, t) < 0.0 ? negated(*plane) : *plane;
                const double sigma2 = dot3(sub3(t, t0), sub3(t, t0));
                const double delta2 = dot3(sub3(np, t), sub3(np, t));
                const double w = sigma2 / (sigma2 + delta2 + TINY);
                t = unit3(add3(t, scl3(sub3(np, t), w)));
            }
        }
    }
    const auto branch = m_branch.find({a, b});
    const double branchSpread = branch == m_branch.end() ? 0.0 : branch->second;
    m_tangentUncertainty[{a, b}] = norm3(sub3(t, tCircle)) + spread + branchSpread;
    m_lineTangent[{a, b}] = t;
    return t;
}

// The square of the uncertainty of the line tangent of edge a → b at a (zero when it was not computed).
double SurfaceReconstruction::tangentUncertaintySquared(std::size_t a, std::size_t b) const
{
    const auto it = m_tangentUncertainty.find({a, b});
    return it == m_tangentUncertainty.end() ? 0.0 : it->second * it->second;
}

// The elements around grid g that continue element e without crossing a raw edge (e included).
// Reference: Reconstruction::fan.
std::vector<std::size_t> SurfaceReconstruction::fan(std::size_t e, std::size_t g)
{
    std::vector<std::size_t> out{e};
    for (std::size_t k = 0; k < out.size(); ++k)
    {
        const std::size_t el = out[k];
        for (const std::size_t d : neighboursIn(shell(el), g))
        {
            if (isRaw(g, d))
                continue;
            if (const std::optional<std::size_t> f = facing(el, g, d))
                pushUnique(out, *f);
        }
    }
    return out;
}

// Reference: Reconstruction::fan_normal (cached fanNormalAt).
std::optional<SurfaceReconstruction::FanNormal> SurfaceReconstruction::fanNormal(std::size_t e, std::size_t a,
                                                                                 std::size_t level)
{
    const auto key = std::make_tuple(e, a, level);
    if (const auto it = m_fanNormals.find(key); it != m_fanNormals.end())
        return it->second;
    const std::optional<FanNormal> out = fanNormalAt(e, a, level);
    m_fanNormals[key] = out;
    return out;
}

// The normal at grid a, on the side of element e, from the edges at a in the fan of e: the unit vector that best
// meets, by least squares, one condition per edge. None unless two of the edges at least are different lines. The
// conditions, by level:
// 1. for a line that goes on through a: the normal is at a right angle to its tangent. The circle through three
//    grids of a line of the surface is tangent to the surface at the middle one on a sphere, and on a cylinder when
//    the grids are evenly spaced, whatever the line;
// 2. also, for a line that ends at a: on the boundary of the mesh, the same condition with its tangent prolonged
//    from its other grid; inside, the chord condition with the normal of level 1 at its other grid b: the chord
//    makes opposite angles with the two normals, n_a · (b − a) = −n_b · (b − a). That is exact on a sphere and on a
//    cylinder for every chord, and of the second order on a smooth surface;
// 3. also the chord condition with a normal of level 2, and the tangents prolonged from the other grid wherever they
//    are (corners of the boundary);
// 4. also, for an edge whose line ends at both grids and whose other grid has no normal of the levels above, the
//    chord condition with the normal of the facets there (facetNormal): a strip one element wide is read as straight
//    across the strip, as one row of quadrilaterals is.
// With the normal comes its uncertainty, a squared angle: the mean square of what the conditions leave (lines that do
// not agree on a tangent plane). Two lines always agree: the uncertainty is then that of their tangents, twice the
// mean of their squares. With three lines or more that quantity is left out: it measures how far a line is from a
// circle, in its own plane too, and a helix on a cylinder is far from a circle while its tangent is exactly in the
// tangent plane.
// Reference: Reconstruction::fan_normal_at.
std::optional<SurfaceReconstruction::FanNormal> SurfaceReconstruction::fanNormalAt(std::size_t e, std::size_t a,
                                                                                   std::size_t level)
{
    std::vector<std::size_t> ends;
    for (const std::size_t f : fan(e, a))
        for (const std::size_t h : neighboursIn(shell(f), a))
            pushUnique(ends, h);
    // conditions n · direction = value, one per edge; squared uncertainties of the tangents among them
    std::vector<NormalCondition> conditions;
    std::vector<double> uncertainties;
    const auto addLineTangent = [&](std::size_t h)
    {
        conditions.push_back({h, lineTangent(a, h), 0.0});
        uncertainties.push_back(tangentUncertaintySquared(a, h));
    };
    for (const std::size_t h : ends)
    {
        if (continuation(a, h).smooth)
        {
            addLineTangent(h);
            continue;
        }
        if (level < 2)
            continue;
        const bool external = elementsOf(a, h).size() == 1;
        if (external && informed(a, h))
        {
            addLineTangent(h);
            continue;
        }
        std::optional<Vector3> far = farNormal(e, a, h, 1);
        if (!far && level >= 3)
            far = farNormal(e, a, h, 2);
        if (far)
        {
            const Vector3 u = unit3(sub3(position(h), position(a)));
            conditions.push_back({h, u, -dot3(*far, u)});
        }
        else if (level >= 3 && informed(a, h))
        {
            addLineTangent(h);
        }
        else if (level >= 4)
        {
            const Vector3 nb = facetNormal(e, a, h);
            const Vector3 u = unit3(sub3(position(h), position(a)));
            conditions.push_back({h, u, -dot3(nb, u)});
        }
    }
    // the number of different lines among the edges of the conditions
    std::vector<std::size_t> group(conditions.size());
    for (std::size_t i = 0; i < conditions.size(); ++i)
        group[i] = i;
    for (std::size_t i = 0; i < conditions.size(); ++i)
    {
        const std::map<std::size_t, bool>& ci = candidates(a, conditions[i].grid);
        for (std::size_t j = i + 1; j < conditions.size(); ++j)
        {
            if (ci.contains(conditions[j].grid) || candidates(a, conditions[j].grid).contains(conditions[i].grid))
            {
                const std::size_t gi = group[i];
                const std::size_t gj = group[j];
                for (std::size_t& g : group)
                    if (g == gj)
                        g = gi;
            }
        }
    }
    std::size_t lines = 0;
    for (std::size_t i = 0; i < conditions.size(); ++i)
        if (group[i] == i)
            ++lines;
    if (lines < 2)
        return std::nullopt;
    // Gauss-Newton on the unit sphere, from the normal of the element, in a tangent basis (p, q) at the iterate
    Vector3 n = m_topology.normals[e];
    for (int iteration = 0; iteration < FAN_NORMAL_ITERATIONS; ++iteration)
    {
        const Vector3 seed = seedAcross(n);
        const Vector3 p = unit3(sub3(seed, scl3(n, dot3(seed, n))));
        const Vector3 q = cross3(n, p);
        double a11 = 0.0;
        double a12 = 0.0;
        double a22 = 0.0;
        double b1 = 0.0;
        double b2 = 0.0;
        for (const NormalCondition& c : conditions)
        {
            const double r = dot3(c.direction, n) - c.value;
            const double dp = dot3(c.direction, p);
            const double dq = dot3(c.direction, q);
            a11 += dp * dp;
            a12 += dp * dq;
            a22 += dq * dq;
            b1 -= dp * r;
            b2 -= dq * r;
        }
        const double det = a11 * a22 - a12 * a12;
        if (!(det > 0.0))
            return std::nullopt;
        const double x = (a22 * b1 - a12 * b2) / det;
        const double y = (a11 * b2 - a12 * b1) / det;
        n = unit3(add3(n, add3(scl3(p, x), scl3(q, y))));
    }
    if (!(std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])))
        return std::nullopt;
    double left = 0.0;
    for (const NormalCondition& c : conditions)
    {
        const double r = dot3(c.direction, n) - c.value;
        left += r * r;
    }
    left /= static_cast<double>(conditions.size());
    double tangentsTerm = 0.0;
    if (!uncertainties.empty() && lines < 3)
    {
        double sum = 0.0;
        for (const double v : uncertainties)
            sum += v;
        tangentsTerm = 2.0 * sum / static_cast<double>(uncertainties.size());
    }
    return FanNormal{n, left + tangentsTerm};
}

// The normal of the facets at grid b of edge a − b, on the side of element e: the mean of the normals of the elements
// around b, less its part along the line that goes on through b, or that ends at b on the boundary and goes on at its
// other grid, when there is one.
// Reference: Reconstruction::facet_normal.
Vector3 SurfaceReconstruction::facetNormal(std::size_t e, std::size_t a, std::size_t b)
{
    const Vector3 n0 = m_topology.normals[e];
    Vector3 sum{};
    std::vector<std::size_t> ends;
    for (const std::size_t f : elementsOf(a, b))
        for (const std::size_t g : fan(f, b))
        {
            const Vector3& v = m_topology.normals[g];
            sum = add3(sum, dot3(v, n0) < 0.0 ? negated(v) : v);
            for (const std::size_t h : neighboursIn(shell(g), b))
                pushUnique(ends, h);
        }
    Vector3 n = unit3(sum);
    for (const std::size_t h : ends)
    {
        const bool external = elementsOf(b, h).size() == 1;
        if (continuation(b, h).smooth || (external && informed(b, h)))
        {
            const Vector3 t = lineTangent(b, h);
            n = unit3(sub3(n, scl3(t, dot3(n, t))));
            break;
        }
    }
    return n;
}

// The normal of fanNormal at grid b of edge a − b, over the elements of the edge, on the side of element e.
// Reference: Reconstruction::far_normal.
std::optional<Vector3> SurfaceReconstruction::farNormal(std::size_t e, std::size_t a, std::size_t b, std::size_t level)
{
    const Vector3 n0 = m_topology.normals[e];
    Vector3 sum{};
    bool any = false;
    for (const std::size_t f : elementsOf(a, b))
        if (const std::optional<FanNormal> v = fanNormal(f, b, level))
        {
            sum = add3(sum, dot3(v->normal, n0) < 0.0 ? negated(v->normal) : v->normal);
            any = true;
        }
    if (!any)
        return std::nullopt;
    return unit3(sum);
}

// The normal of a triangle e at its grid a from fanNormal, at the first level that gives one, with its uncertainty.
// Reference: Reconstruction::triangle_normal.
std::optional<SurfaceReconstruction::FanNormal> SurfaceReconstruction::triangleNormal(std::size_t e, std::size_t a)
{
    if (shell(e).count != 3)
        return std::nullopt;
    for (std::size_t level = 1; level <= LAST_FAN_LEVEL; ++level)
        if (const std::optional<FanNormal> n = fanNormal(e, a, level))
            return n;
    return std::nullopt;
}

// The unit normal of element e at its grid a according to the mesh lines: normal to the tangents of the lines of its
// two edges there. A triangle takes the normal of fanNormal instead, at the first level that gives one: its own two
// edges are often a line that ends at the grid (near every boundary) or says nothing (an edge whose line goes on at
// neither end), and the triangles around a grid then have one tangent plane.
// Reference: Reconstruction::line_normal.
Vector3 SurfaceReconstruction::lineNormal(std::size_t e, std::size_t a)
{
    const std::array<std::size_t, 2> own = neighboursIn(shell(e), a);
    const Vector3 n0 = m_topology.normals[e];
    if (const std::optional<FanNormal> fanned = triangleNormal(e, a))
        return fanned->normal;
    const Vector3 v = cross3(lineTangent(a, own[0]), lineTangent(a, own[1]));
    const double length = norm3(v);
    if (length < PARALLEL_TANGENTS)
        return n0;
    const Vector3 unit = scl3(v, 1.0 / length);
    return dot3(unit, n0) < 0.0 ? negated(unit) : unit;
}

// Reference: Reconstruction::fit (cached fitAt).
std::optional<SurfaceReconstruction::QuadricFit> SurfaceReconstruction::fit(std::size_t e, std::size_t g)
{
    if (const auto it = m_fits.find({e, g}); it != m_fits.end())
        return it->second;
    const std::optional<QuadricFit> out = fitAt(e, g);
    m_fits[{e, g}] = out;
    return out;
}

// The normal of the surface at grid g, on the side of element e, from a quadric fitted on the neighbourhood of the
// grid, with the weight in [0, 1] it is given against the mesh lines.
//
// The tangent of a mesh line at a grid is that of the circle through three grids of the line. When the line is a
// smooth curve of the surface it lies in the tangent plane; when the line zigzags, or kinks from one block of the
// mesh to the next, it does not. Where the lines cannot be trusted the tangent plane is read from the surface.
//
// The grids are those of the elements around g and around its neighbours (two rings, without crossing a raw edge).
// In axes (x, y, z) at g, z along the normal of the element, lengths over the size of the neighbourhood, the quadric
// through g is written with a unit coefficient on the height,
//   z = a x + b y + c x² + d x y + e y² + f z² + g x z + h y z,
// and its eight coefficients are a linear least-squares problem (columns scaled, ridge of 1e-12). A plane, a sphere,
// a cylinder, an ellipsoid, a paraboloid are found exactly; the normal at g is along (−a, −b, 1). The covariance of
// (a, b) is that of the regression, (residual variance + 1e-20)(XᵀX + ridge)⁻¹: the squared angle by which the normal
// is uncertain.
//
// The weight compares two uncertainties, both angles: d_l, that of the mesh lines of the element at the grid, and
// d_f, that of the fit (the uncertainty of its normal, and what the distance between the fitted normal and that of
// the lines has beyond d_l). The weight of the fit is d_l⁴ / (d_l⁴ + d_f⁴): the more certain of the two is taken, and
// they share only when their uncertainties are alike. Mesh lines that are circles of the surface keep their own
// tangents: a cylinder or a sphere meshed along its circles is unchanged.
// Reference: Reconstruction::fit.
std::optional<SurfaceReconstruction::QuadricFit> SurfaceReconstruction::fitAt(std::size_t e, std::size_t g)
{
    std::set<std::size_t> nodes;
    for (const std::size_t f : fan(e, g))
        for (const std::size_t h : shell(f).view())
            for (const std::size_t ff : fan(f, h))
                for (const std::size_t q : shell(ff).view())
                    nodes.insert(q);
    nodes.erase(g);
    // eight coefficients: the fit must be over-determined to say anything
    if (nodes.size() < QUADRIC_LEAST_GRIDS)
        return std::nullopt;
    const Vector3 x0 = position(g);
    double size = 0.0;
    for (const std::size_t h : nodes)
        size = std::fmax(size, norm3(sub3(position(h), x0)));
    // the axes at g: z along the normal of the element
    const Vector3 ez = m_topology.normals[e];
    const Vector3 seed = seedAcross(ez);
    const Vector3 ex = unit3(sub3(seed, scl3(ez, dot3(seed, ez))));
    const Vector3 ey = cross3(ez, ex);
    const std::size_t np = nodes.size();
    Matrix xm(np, QUADRIC_COEFFICIENTS);
    Vector zv(np, 0.0);
    std::size_t r = 0;
    for (const std::size_t h : nodes)
    {
        const Vector3 d = scl3(sub3(position(h), x0), 1.0 / size);
        const double x = dot3(d, ex);
        const double y = dot3(d, ey);
        const double z = dot3(d, ez);
        const std::array<double, QUADRIC_COEFFICIENTS> row{x, y, x * x, x * y, y * y, z * z, x * z, y * z};
        for (std::size_t c = 0; c < QUADRIC_COEFFICIENTS; ++c)
            xm(r, c) = row[c];
        zv[r] = z;
        ++r;
    }
    // the columns scaled to unit norm (a column that vanishes, on a plane, stays out)
    std::array<double, QUADRIC_COEFFICIENTS> columnNorm{};
    double largestNorm = 0.0;
    for (std::size_t c = 0; c < QUADRIC_COEFFICIENTS; ++c)
    {
        columnNorm[c] = euclideanNorm(xm.column(c));
        largestNorm = std::fmax(largestNorm, columnNorm[c]);
    }
    std::array<double, QUADRIC_COEFFICIENTS> inverse{};
    for (std::size_t c = 0; c < QUADRIC_COEFFICIENTS; ++c)
    {
        inverse[c] = columnNorm[c] > QUADRIC_COLUMN_TOLERANCE * largestNorm ? 1.0 / columnNorm[c] : 0.0;
        xm.scaleColumn(c, inverse[c]);
    }
    Matrix ata = transposeTimes(xm, xm);
    for (std::size_t c = 0; c < QUADRIC_COEFFICIENTS; ++c)
        ata(c, c) += QUADRIC_RIDGE + (inverse[c] == 0.0 ? 1.0 : 0.0);
    const std::optional<Cholesky> factor = Cholesky::factor(ata);
    if (!factor)
        return std::nullopt;
    const Vector beta = factor->solve(transposeTimes(xm, zv));
    Vector residual = xm * beta;
    for (std::size_t i = 0; i < np; ++i)
        residual[i] -= zv[i];
    const double sigma2 = dot(residual, residual) / static_cast<double>(np) + QUADRIC_VARIANCE_FLOOR;
    const Matrix covariance = factor->inverse();
    const double slopeX = beta[0] * inverse[0];
    const double slopeY = beta[1] * inverse[1];
    const Vector3 n = unit3(add3(ez, add3(scl3(ex, -slopeX), scl3(ey, -slopeY))));
    const double dq2 =
        sigma2 * (covariance(0, 0) * inverse[0] * inverse[0] + covariance(1, 1) * inverse[1] * inverse[1]);
    // the uncertainty of the mesh lines at g: the two edges of e there
    double dl2 = 0.0;
    for (const std::size_t h : neighboursIn(shell(e), g))
    {
        static_cast<void>(lineTangent(g, h));
        dl2 += tangentUncertaintySquared(g, h);
    }
    // a triangle: the uncertainty of the normal it takes from its fan
    if (const std::optional<FanNormal> fanned = triangleNormal(e, g))
        dl2 = fanned->uncertainty;
    // A fit that is farther from the normal of the mesh lines than the lines are uncertain is wrong by the difference
    // (a surface that is not a quadric, read on a narrow strip). Lines that zigzag or kink have a large uncertainty
    // and do not constrain it.
    const Vector3 dn = sub3(n, lineNormal(e, g));
    const double df2 = dq2 + std::fmax(dot3(dn, dn) - dl2, 0.0);
    // the more certain of the two is taken; they share only when their uncertainties are alike
    return QuadricFit{n, dl2 * dl2 / (dl2 * dl2 + df2 * df2 + TINY), dq2, sigma2 * size * size};
}

// The unit normal of the surface at grid a on the side of element e: that of the mesh lines and that of the local
// fit, each by its weight.
// Reference: Reconstruction::normal.
Vector3 SurfaceReconstruction::normal(std::size_t e, std::size_t a)
{
    const Vector3 nl = lineNormal(e, a);
    if (const std::optional<QuadricFit> f = fit(e, a))
        return unit3(add3(scl3(nl, 1.0 - f->weight), scl3(f->normal, f->weight)));
    return nl;
}

// The normal, the weight and the variances of the fit at grid g of an edge, over the elements of the edge (their
// normals turned to the side of the normal of the frame of the edge).
// Reference: the closure `at` of Reconstruction::tangent.
SurfaceReconstruction::EdgeGridNormal SurfaceReconstruction::edgeGridNormal(const std::vector<std::size_t>& owners,
                                                                            const Vector3& frameNormal, std::size_t g)
{
    Vector3 n{};
    EdgeGridNormal out;
    const double count = static_cast<double>(owners.size());
    for (const std::size_t e : owners)
    {
        const Vector3 v = normal(e, g);
        n = add3(n, dot3(v, frameNormal) < 0.0 ? negated(v) : v);
        if (const std::optional<QuadricFit> f = fit(e, g))
        {
            out.weight += f->weight / count;
            out.normalVariance += f->normalVariance / count;
            out.positionVariance += f->positionVariance / count;
        }
    }
    out.normal = unit3(n);
    return out;
}

// The unit tangent at grid a of the curve of edge a → b: that of its mesh line where the line agrees with the
// surface, that of a plane section where it does not (see fitAt).
// Reference: Reconstruction::tangent.
Vector3 SurfaceReconstruction::tangent(std::size_t a, std::size_t b)
{
    if (const auto it = m_tangent.find({a, b}); it != m_tangent.end())
        return it->second;
    const Vector3 line = lineTangent(a, b);
    const std::size_t idx = m_topology.indexOf(a, b);
    const std::vector<std::size_t>& owners = m_topology.edgeElements[idx];
    const Vector3 frameNormal = m_topology.frames[idx][2];
    // the variance of the tangent of the line
    const double lineVariance = tangentUncertaintySquared(a, b);
    double variance = lineVariance;
    Vector3 t{};
    if (owners.size() > 2)
    {
        // a junction of more than two elements: the line is geometry for all of them
        t = line;
    }
    else
    {
        const EdgeGridNormal atA = edgeGridNormal(owners, frameNormal, a);
        // that of the estimate taken, the line or the fit, and that of the direction of the chord between two grids
        // known to within the residual of the fit
        const Vector3 chord = sub3(position(b), position(a));
        const double chord2 = dot3(chord, chord);
        variance =
            (1.0 - atA.weight) * lineVariance + atA.weight * atA.normalVariance + 2.0 * atA.positionVariance / chord2;
        // the mesh line, in the tangent plane at a
        const Vector3 inPlane = unit3(sub3(line, scl3(atA.normal, dot3(line, atA.normal))));
        // An edge of a triangle is a plane section of the surface between its two grids (below), unless it is on the
        // boundary and its line says something: the lines of a mesh of triangles are short near every boundary and
        // zigzag on any mesh that is not regular, and a short line cannot tell how certain it is.
        const bool sectionOnly = triangleEdge(a, b) && (owners.size() == 2 || !(informed(a, b) || informed(b, a)));
        const double weight = sectionOnly ? 1.0 : atA.weight;
        if (sectionOnly)
            m_section[{a, b}] = atA.normal;
        if (owners.size() == 1 && !sectionOnly)
        {
            // boundary: the line is geometry
            t = inPlane;
        }
        else
        {
            // Between two elements the line is geometry only as far as it agrees with the surface: where it does not
            // (weight of the fit), the edge is a plane section. The plane holds the chord and the mean of the normals
            // at the two grids; the tangent at a is its line in the tangent plane at a, so that the two end tangents
            // and the chord are in one plane whatever the twist of the normals along the edge.
            const EdgeGridNormal atB = edgeGridNormal(owners, frameNormal, b);
            const Vector3 u = unit3(chord);
            const Vector3 m = cross3(u, add3(atA.normal, atB.normal));
            const Vector3 s = cross3(m, atA.normal);
            const Vector3 section = unit3(dot3(s, u) < 0.0 ? negated(s) : s);
            t = unit3(add3(scl3(inPlane, 1.0 - weight), scl3(section, weight)));
        }
    }
    m_tangentVariance[{a, b}] = variance;
    m_tangent[{a, b}] = t;
    return t;
}

Vector3 SurfaceReconstruction::bulge(std::size_t a, std::size_t b)
{
    if (isRaw(a, b))
        return {};
    const Vector3 d = sub3(position(b), position(a));
    const double length = norm3(d);
    const Vector3 u = scl3(d, 1.0 / length);
    const Vector3 ta = tangent(a, b);
    const Vector3 tb = tangent(b, a);
    // the tangents scaled to a unit component along the chord, less the chord
    const Vector3 sigmaA = sub3(scl3(ta, 1.0 / dot3(ta, u)), u);
    const Vector3 sigmaB = add3(scl3(tb, -1.0 / dot3(tb, u)), u);
    const Vector3 sum = add3(sigmaA, sigmaB);
    if (norm3(sum) == 0.0)
        return {};
    const Vector3 direction = unit3(sum);
    const auto varianceOf = [&](std::size_t p, std::size_t q)
    {
        const auto it = m_tangentVariance.find({p, q});
        return it == m_tangentVariance.end() ? 0.0 : it->second;
    };
    const double variance = varianceOf(a, b) + varianceOf(b, a);
    return endSlopeBulge(u, length, direction, dot3(sigmaA, direction), dot3(sigmaB, direction), variance);
}

// The control point of the conic is at the intersection of the two end tangents: x = L sb / (sa + sb) along the
// chord, at the height sa x. Written with ρ = (|sb| − |sa|) / (|sa| + |sb|): x = L (1 + ρ)/2 and
// height = (sa + sb) x (L − x) / L, which keeps the mean slope when the slopes have opposite signs (no conic then) and
// vanishes smoothly on a flat edge. The bulge is half the offset of the control point from the middle of the chord.
//
// `variance`: the variance of the two end slopes. Where the slopes are not well above their own uncertainty (an edge
// that is straight but for the noise of the coordinates), their difference means nothing and the control point stays
// over the middle of the chord: ρ is shrunk by s² / (s² + 200 var), s = |sa| + |sb|, the estimate of a quantity
// expected of the order of 0.1 and measured with the variance 2 var / s².
Vector3 endSlopeBulge(const Vector3& u, double length, const Vector3& direction, double sa, double sb, double variance)
{
    // the shrinkage of ρ: 2 var / s² against (0.1)², and a guard of a vanishing s
    constexpr double SHRINK = 200.0;
    constexpr double GUARD = 1e-18;
    const double s = std::abs(sa) + std::abs(sb);
    double rho = (std::abs(sb) - std::abs(sa)) * s / (s * s + SHRINK * variance + GUARD);
    // A conic tangent to its chord at one end is the chord: when one slope vanishes the control point would come on
    // the other grid and the curve would stop there. It is kept within the middle half of the chord (ρ within ±1/2);
    // equal slopes, and slopes that differ little, are unchanged to the second order.
    rho = rho / (1.0 + rho * rho);
    const double xc = 0.5 * length * (1.0 + rho);
    const double height = (sa + sb) * xc * (length - xc) / length;
    return add3(scl3(u, 0.5 * (xc - 0.5 * length)), scl3(direction, 0.5 * height));
}

} // namespace oos::elements::mixedHybrid::detail
