#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"

#include <cmath>
#include <map>

// The forces at the grid points of the outline of the shells read as tractions along its edges
// (ShellPart::forcesAlongEdges; docs/mixed-hybrid-formulation-spec.md sections 7 and 12).

namespace oos::elements::mixedHybrid
{

namespace
{

// The conjugate gradients stop at this share of the norm of the right-hand side, or after this many steps (the
// matrix scaled by its diagonal has a condition number under 3 on a line of edges: about 30 steps).
constexpr double SOLVE_TOLERANCE = 1e-15;
constexpr std::size_t SOLVE_STEP_LIMIT = 500;

// An edge of the outline that carries a traction.
struct TractionEdge
{
    std::size_t edge = 0;
    // Its two grids, the lower first, and the traction value of each end (a corner of the outline has one value per
    // side, any other grid one value).
    std::array<std::size_t, 2> grids{};
    std::array<std::size_t, 2> values{};
    // M = ∫ N_i N_j ds, N_0 = 1 − c and N_1 = c the linear functions of the chord fraction c.
    double mass00 = 0.0;
    double mass01 = 0.0;
    double mass11 = 0.0;
    // The unit tangent of the edge at each grid, away from the grid.
    std::array<SpaceVector, 2> outward{};

    [[nodiscard]] double length() const { return mass00 + 2.0 * mass01 + mass11; }
    // ∫ N_end ds: the force at the grid of `end` of a unit uniform traction on the edge.
    [[nodiscard]] double share(std::size_t end) const { return (end == 0 ? mass00 : mass11) + mass01; }
};

// y = M x on the traction values.
void multiply(std::span<const TractionEdge> edges, std::span<const double> x, std::span<double> y)
{
    std::fill(y.begin(), y.end(), 0.0);
    for (const TractionEdge& edge : edges)
    {
        const std::size_t a = edge.values[0];
        const std::size_t b = edge.values[1];
        y[a] += edge.mass00 * x[a] + edge.mass01 * x[b];
        y[b] += edge.mass01 * x[a] + edge.mass11 * x[b];
    }
}

// The solution of M q = f by conjugate gradients preconditioned by the diagonal of M (symmetric positive definite:
// the mass matrix of the linear functions on the edges).
std::vector<double> solve(std::span<const TractionEdge> edges, std::span<const double> diagonal,
                          std::span<const double> rhs)
{
    const std::size_t n = rhs.size();
    std::vector<double> q(n, 0.0);
    std::vector<double> residual(rhs.begin(), rhs.end());
    std::vector<double> scaled(n);
    std::vector<double> direction(n);
    std::vector<double> image(n);
    double rhsNorm = 0.0;
    for (const double value : rhs)
        rhsNorm += value * value;
    if (rhsNorm == 0.0)
        return q;
    double product = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        scaled[i] = residual[i] / diagonal[i];
        direction[i] = scaled[i];
        product += residual[i] * scaled[i];
    }
    for (std::size_t step = 0; step < SOLVE_STEP_LIMIT; ++step)
    {
        multiply(edges, direction, image);
        double curvature = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            curvature += direction[i] * image[i];
        if (!(curvature > 0.0))
            break;
        const double alpha = product / curvature;
        double residualNorm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            q[i] += alpha * direction[i];
            residual[i] -= alpha * image[i];
            residualNorm += residual[i] * residual[i];
        }
        if (residualNorm <= SOLVE_TOLERANCE * SOLVE_TOLERANCE * rhsNorm)
            break;
        double next = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            scaled[i] = residual[i] / diagonal[i];
            next += residual[i] * scaled[i];
        }
        const double beta = next / product;
        product = next;
        for (std::size_t i = 0; i < n; ++i)
            direction[i] = scaled[i] + beta * direction[i];
    }
    return q;
}

} // namespace

ShellEdgeForceReading ShellPart::forcesAlongEdges(std::span<const ShellGridForce> forces) const
{
    ShellEdgeForceReading out;
    out.isTaken.assign(forces.size(), false);
    // the force of each loaded grid (several forces of one grid add up)
    std::map<std::size_t, SpaceVector> forceOf;
    for (const ShellGridForce& force : forces)
        if (force.grid < gridCount())
        {
            const auto [place, isNew] = forceOf.try_emplace(force.grid, force.force);
            if (!isNew)
                place->second = add3(place->second, force.force);
        }
    std::erase_if(forceOf, [](const auto& entry) { return norm3(entry.second) == 0.0; });
    if (forceOf.size() < 2)
        return out;

    // the edges read: those one shell has, with a force at both their grids
    std::vector<TractionEdge> edges;
    std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>> sidesAt; // grid: (edge read, its end)
    for (std::size_t e = 0; e < m_topology.edges.size(); ++e)
    {
        const std::size_t ga = m_topology.edges[e].lower;
        const std::size_t gb = m_topology.edges[e].higher;
        if (m_topology.edgeElements[e].size() != 1 || !forceOf.contains(ga) || !forceOf.contains(gb))
            continue;
        const std::size_t ie = m_topology.edgeElements[e].front();
        const ShellElementData el = element(ie);
        const ShellReferenceElement& re = referenceElement(el);
        const ShellElementDefinition def = el.definition();
        const std::size_t kk = localEdgeOf(ie, e);
        const SpaceVector& xa = m_input.grids[ga].position;
        const SpaceVector chord = sub3(m_input.grids[gb].position, xa);
        const double chordSquared = dot3(chord, chord);
        TractionEdge read;
        read.edge = e;
        read.grids = {ga, gb};
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const PointGeometry g = edgeGeometry(def, re.h1Count, kk, gi);
            const auto tv = referenceEdgePointAndDirection(el.cornerCount, kk, re.edgeParameters[gi]).second;
            const double ds = re.edgeWeights[gi] * norm3(add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1])));
            const double c = dot3(sub3(g.x, xa), chord) / chordSquared;
            read.mass00 += ds * (1.0 - c) * (1.0 - c);
            read.mass01 += ds * (1.0 - c) * c;
            read.mass11 += ds * c * c;
        }
        // the tangent at each end, along the parameter of the shell, turned away from the grid
        for (std::size_t end = 0; end < 2; ++end)
        {
            const double s = end == 0 ? 0.0 : 1.0;
            const double tau = el.edgeSigns[kk] > 0.0 ? s : 1.0 - s;
            const auto [point, tv] = referenceEdgePointAndDirection(el.cornerCount, kk, tau);
            const PointGeometry g = geometryAt(re, def, point[0], point[1]);
            const SpaceVector along = unit3(add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1])));
            // `along` runs with tau: from the lower grid when the sign of the edge is positive
            const double toHigher = el.edgeSigns[kk] > 0.0 ? 1.0 : -1.0;
            read.outward[end] = scl3(along, end == 0 ? toHigher : -toHigher);
        }
        sidesAt[ga].emplace_back(edges.size(), 0);
        sidesAt[gb].emplace_back(edges.size(), 1);
        edges.push_back(read);
    }
    if (edges.empty())
        return out;
    out.edgeCount = edges.size();
    for (std::size_t i = 0; i < forces.size(); ++i)
        out.isTaken[i] = sidesAt.contains(forces[i].grid);

    // the traction values: one per grid where the outline goes on or ends, one per side at a corner
    std::size_t valueCount = 0;
    std::map<std::size_t, bool> isCorner;
    for (const auto& [grid, sides] : sidesAt)
    {
        bool corner = sides.size() > 2;
        if (sides.size() == 2)
        {
            // two tangents away from the grid: opposite where the outline goes on
            const double cosine = -dot3(edges[sides[0].first].outward[sides[0].second],
                                        edges[sides[1].first].outward[sides[1].second]);
            corner = cosine < std::cos(OUTLINE_CORNER_ANGLE);
        }
        isCorner[grid] = corner;
        for (const auto& [edge, end] : sides)
            edges[edge].values[end] = corner ? valueCount++ : valueCount;
        if (!corner)
            ++valueCount;
    }

    // the right-hand side: the force of a grid, or at a corner the share of each side
    std::array<std::vector<double>, 3> rhs;
    for (std::vector<double>& component : rhs)
        component.assign(valueCount, 0.0);
    // the uniform traction that the force of a grid without corner stands for: its force over half the length of
    // its edges
    const auto tractionAt = [&](std::size_t grid)
    {
        double length = 0.0;
        for (const auto& [edge, end] : sidesAt.at(grid))
            length += edges[edge].length();
        return scl3(forceOf.at(grid), 2.0 / length);
    };
    for (const auto& [grid, sides] : sidesAt)
    {
        const SpaceVector& force = forceOf.at(grid);
        if (!isCorner.at(grid))
        {
            for (std::size_t c = 0; c < 3; ++c)
                rhs[c][edges[sides.front().first].values[sides.front().second]] = force[c];
            continue;
        }
        // each side continues the traction of its other grid; what is left of the force is shared equally, among
        // the sides that have no such traction (their other grid is a corner too) when there are some
        std::vector<SpaceVector> shares(sides.size(), SpaceVector{});
        std::vector<bool> isContinued(sides.size(), false);
        SpaceVector left = force;
        std::size_t continuedCount = 0;
        for (std::size_t s = 0; s < sides.size(); ++s)
        {
            const TractionEdge& edge = edges[sides[s].first];
            const std::size_t other = edge.grids[1 - sides[s].second];
            if (isCorner.at(other))
                continue;
            shares[s] = scl3(tractionAt(other), edge.share(sides[s].second));
            isContinued[s] = true;
            ++continuedCount;
            left = sub3(left, shares[s]);
        }
        const bool toAll = continuedCount == 0 || continuedCount == sides.size();
        const double part = 1.0 / static_cast<double>(toAll ? sides.size() : sides.size() - continuedCount);
        for (std::size_t s = 0; s < sides.size(); ++s)
        {
            if (toAll || !isContinued[s])
                shares[s] = add3(shares[s], scl3(left, part));
            for (std::size_t c = 0; c < 3; ++c)
                rhs[c][edges[sides[s].first].values[sides[s].second]] = shares[s][c];
        }
    }

    // the traction values of each component
    std::vector<double> diagonal(valueCount, 0.0);
    for (const TractionEdge& edge : edges)
    {
        diagonal[edge.values[0]] += edge.mass00;
        diagonal[edge.values[1]] += edge.mass11;
    }
    std::array<std::vector<double>, 3> traction;
    for (std::size_t c = 0; c < 3; ++c)
        traction[c] = solve(edges, diagonal, rhs[c]);

    // the loads of the tractions, on the unknowns of the part
    std::map<std::size_t, double> loadOf;
    for (const TractionEdge& edge : edges)
    {
        const SpaceVector qa{traction[0][edge.values[0]], traction[1][edge.values[0]], traction[2][edge.values[0]]};
        const SpaceVector qb{traction[0][edge.values[1]], traction[1][edge.values[1]], traction[2][edge.values[1]]};
        const SpaceVector& xa = m_input.grids[edge.grids[0]].position;
        const SpaceVector chord = sub3(m_input.grids[edge.grids[1]].position, xa);
        const std::variant<ShellEdgeLoad, std::string> load = edgeLoadAt(
            edge.edge, edge.grids[0], edge.grids[1],
            [&](double c, const SpaceVector& position)
            {
                // the traction at the point of the chord, carried rigidly to the point of the edge
                const SpaceVector q = add3(scl3(qa, 1.0 - c), scl3(qb, c));
                const SpaceVector couple = cross3(sub3(add3(xa, scl3(chord, c)), position), q);
                return std::array<double, 6>{q[0], q[1], q[2], couple[0], couple[1], couple[2]};
            },
            true);
        const ShellEdgeLoad& local = std::get<ShellEdgeLoad>(load);
        const std::span<const std::size_t> unknowns = localUnknowns(local.shell);
        const std::span<const double> signs = localSigns(local.shell);
        for (std::size_t i = 0; i < local.local.size() && i < unknowns.size(); ++i)
            if (local.local[i] != 0.0)
                loadOf[unknowns[i]] += signs[i] * local.local[i];
    }
    out.loads.reserve(loadOf.size());
    for (const auto& [unknown, value] : loadOf)
        out.loads.push_back({unknown, value});
    return out;
}

} // namespace oos::elements::mixedHybrid
