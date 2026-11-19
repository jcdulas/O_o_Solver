#include "oos/input/tables/PiercingFootprint.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <tuple>

namespace oos::input
{

// ---- FootprintSamples

void FootprintSamples::add(std::int32_t elementId, double weight, std::span<const std::int32_t> grids,
                           std::span<const double> shapeValues)
{
    m_elementIds.push_back(elementId);
    m_weights.push_back(weight);
    m_grids.insert(m_grids.end(), grids.begin(), grids.end());
    m_shapeValues.insert(m_shapeValues.end(), shapeValues.begin(), shapeValues.end());
    m_gridOffsets.push_back(static_cast<std::int64_t>(m_grids.size()));
}

FootprintSample FootprintSamples::at(std::int64_t row, std::int64_t k) const
{
    const auto sample = static_cast<std::size_t>(m_rowOffsets[static_cast<std::size_t>(row)] + k);
    const auto first = static_cast<std::size_t>(m_gridOffsets[sample]);
    const auto count = static_cast<std::size_t>(m_gridOffsets[sample + 1]) - first;
    return {m_elementIds[sample], m_weights[sample], std::span<const std::int32_t>(m_grids).subspan(first, count),
            std::span<const double>(m_shapeValues).subspan(first, count)};
}

// ---- ShellMeshIndex

ShellMeshIndex::ShellMeshIndex(const NodeTable& nodes, std::span<const ElementBatch> batches)
    : m_batches(batches), m_positions(nodes.positionsInBasic())
{
    const std::span<const std::int32_t> nodeIds = nodes.ids();
    m_nodeRows.reserve(nodeIds.size());
    for (std::size_t r = 0; r < nodeIds.size(); ++r)
        m_nodeRows.emplace_back(nodeIds[r], static_cast<std::int64_t>(r));
    std::sort(m_nodeRows.begin(), m_nodeRows.end());
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        if (std::find(PIERCED_SHELL_CARDS.begin(), PIERCED_SHELL_CARDS.end(), batches[b].card()) ==
                PIERCED_SHELL_CARDS.end() ||
            !batches[b].hasNodes())
            continue;
        const auto batch = static_cast<std::uint32_t>(b);
        const ConnectivityView connectivity = batches[b].nodes();
        const std::span<const std::int32_t> ids = batches[b].ids();
        for (std::int64_t row = 0; row < connectivity.size(); ++row)
        {
            m_shellsById.push_back({ids[static_cast<std::size_t>(row)], batch, row});
            for (const std::int32_t id : connectivity[row])
                if (id != 0)
                    m_shellsAtGrid.push_back({id, batch, row});
        }
    }
    std::sort(m_shellsAtGrid.begin(), m_shellsAtGrid.end(), [](const ShellAtGrid& a, const ShellAtGrid& b)
              { return std::tie(a.gridId, a.batch, a.row) < std::tie(b.gridId, b.batch, b.row); });
    std::sort(m_shellsById.begin(), m_shellsById.end(), [](const ShellOfId& a, const ShellOfId& b)
              { return std::tie(a.id, a.batch, a.row) < std::tie(b.id, b.batch, b.row); });
}

std::optional<base::Vec3> ShellMeshIndex::positionOf(std::int32_t gridId) const
{
    const auto found =
        std::lower_bound(m_nodeRows.begin(), m_nodeRows.end(), std::pair<std::int32_t, std::int64_t>{gridId, 0});
    if (found == m_nodeRows.end() || found->first != gridId)
        return std::nullopt;
    const std::span<const double> x = m_positions[found->second];
    return base::Vec3{x[0], x[1], x[2]};
}

std::optional<ShellRef> ShellMeshIndex::find(std::int32_t elementId) const
{
    const auto found = std::lower_bound(m_shellsById.begin(), m_shellsById.end(), elementId,
                                        [](const ShellOfId& shell, std::int32_t id) { return shell.id < id; });
    if (found == m_shellsById.end() || found->id != elementId)
        return std::nullopt;
    return refOf(found->batch, found->row);
}

ShellRef ShellMeshIndex::refOf(std::uint32_t batch, std::int64_t row) const
{
    return {m_batches[batch].ids()[static_cast<std::size_t>(row)], m_batches[batch].nodes()[row]};
}

std::pair<std::size_t, std::size_t> ShellMeshIndex::shellsAt(std::int32_t gridId) const
{
    const auto range = std::equal_range(m_shellsAtGrid.begin(), m_shellsAtGrid.end(), ShellAtGrid{gridId, 0, 0},
                                        [](const ShellAtGrid& a, const ShellAtGrid& b) { return a.gridId < b.gridId; });
    return {static_cast<std::size_t>(range.first - m_shellsAtGrid.begin()),
            static_cast<std::size_t>(range.second - m_shellsAtGrid.begin())};
}

std::optional<base::Vec3> ShellMeshIndex::normalOf(std::span<const std::int32_t> grids) const
{
    const std::size_t corners = cornerCount(grids.size());
    if (grids.size() < corners)
        return std::nullopt;
    std::array<base::Vec3, 4> x{};
    for (std::size_t c = 0; c < corners; ++c)
    {
        const std::optional<base::Vec3> position = positionOf(grids[c]);
        if (!position)
            return std::nullopt;
        x[c] = *position;
    }
    const base::Vec3 n = corners == 3 ? base::cross(x[1] - x[0], x[2] - x[0]) : base::cross(x[2] - x[0], x[3] - x[1]);
    const double length = base::norm(n);
    if (!(length > 0.0))
        return std::nullopt;
    return n / length;
}

// ---- The footprint

namespace
{

constexpr std::size_t MAX_NODES = 8;
// The nearest point of an element is found when a step of the iteration is below this, in parent coordinates, or
// after this many steps (a point far from a curved element may have none: it is then not on the element).
constexpr double PROJECTION_TOLERANCE = 1.0e-12;
constexpr int PROJECTION_STEPS = 30;
// An element is looked at when the point is within this many times its radius (the largest distance of a corner to
// the mean of its corners) of the mean of its corners.
constexpr double REACH_PER_RADIUS = 1.5;

// The three-point Gauss rule on [−1, 1], its weights as shares of the whole.
struct Abscissa
{
    double at;
    double weight;
};
constexpr std::array<Abscissa, 3> GAUSS_RULE{Abscissa{-0.7745966692414834, 5.0 / 18.0}, Abscissa{0.0, 8.0 / 18.0},
                                             Abscissa{0.7745966692414834, 5.0 / 18.0}};
// The corners of the square, in units of its half side.
constexpr std::array<std::array<double, 2>, 4> SQUARE_CORNERS{
    std::array<double, 2>{-1.0, -1.0}, std::array<double, 2>{1.0, -1.0}, std::array<double, 2>{1.0, 1.0},
    std::array<double, 2>{-1.0, 1.0}};

// The shape functions of a shell card of `nodes` grid points at a point of its parent domain, with their parent
// derivatives. The parent coordinates are those of the piercing table: (ξ, η) in [−1, 1]² on a quadrilateral, the
// area coordinates (L2, L3) on a triangle.
struct ShapeAt
{
    std::array<double, MAX_NODES> n{};
    std::array<double, MAX_NODES> dXi{};
    std::array<double, MAX_NODES> dEta{};
};

constexpr bool isTriangle(std::size_t nodes) noexcept
{
    return nodes == 3 || nodes == 6;
}

// The triangles: the linear functions L1 = 1 − ξ − η, L2 = ξ, L3 = η of three grid points; of six, the quadratic
// functions N_i = L_i (2 L_i − 1) of the corners and 4 L_i L_j of the mid-side points G4 on G1-G2, G5 on G2-G3, G6
// on G3-G1 (Zienkiewicz & Taylor, The Finite Element Method, vol. 1, §8.8).
ShapeAt triangleShapeAt(std::size_t nodes, double xi, double eta)
{
    const double l1 = 1.0 - xi - eta;
    ShapeAt shape;
    if (nodes == 3)
    {
        shape.n = {l1, xi, eta};
        shape.dXi = {-1.0, 1.0, 0.0};
        shape.dEta = {-1.0, 0.0, 1.0};
        return shape;
    }
    shape.n = {l1 * (2.0 * l1 - 1.0), xi * (2.0 * xi - 1.0), eta * (2.0 * eta - 1.0),
               4.0 * l1 * xi,         4.0 * xi * eta,        4.0 * eta * l1};
    shape.dXi = {1.0 - 4.0 * l1, 4.0 * xi - 1.0, 0.0, 4.0 * (l1 - xi), 4.0 * eta, -4.0 * eta};
    shape.dEta = {1.0 - 4.0 * l1, 0.0, 4.0 * eta - 1.0, -4.0 * xi, 4.0 * xi, 4.0 * (l1 - eta)};
    return shape;
}

// The quadrilaterals: the bilinear functions N_i = ¼(1 + ξ ξ_i)(1 + η η_i) of four grid points; of eight, the
// serendipity functions, N_i = ¼(1 + ξ ξ_i)(1 + η η_i)(ξ ξ_i + η η_i − 1) at the corners and ½(1 − ξ²)(1 + η η_i)
// or ½(1 + ξ ξ_i)(1 − η²) at the mid-side points G5 on G1-G2, G6 on G2-G3, G7 on G3-G4, G8 on G4-G1 (Zienkiewicz &
// Taylor vol. 1, §8.5).
ShapeAt quadrilateralShapeAt(std::size_t nodes, double xi, double eta)
{
    constexpr std::array<double, 8> XI{-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0};
    constexpr std::array<double, 8> ETA{-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0};
    ShapeAt shape;
    for (std::size_t i = 0; i < nodes; ++i)
    {
        const double a = 1.0 + xi * XI[i];
        const double b = 1.0 + eta * ETA[i];
        if (nodes == 4)
        {
            shape.n[i] = 0.25 * a * b;
            shape.dXi[i] = 0.25 * XI[i] * b;
            shape.dEta[i] = 0.25 * ETA[i] * a;
        }
        else if (i < 4)
        {
            shape.n[i] = 0.25 * a * b * (xi * XI[i] + eta * ETA[i] - 1.0);
            shape.dXi[i] = 0.25 * XI[i] * b * (2.0 * xi * XI[i] + eta * ETA[i]);
            shape.dEta[i] = 0.25 * ETA[i] * a * (xi * XI[i] + 2.0 * eta * ETA[i]);
        }
        else if (XI[i] == 0.0)
        {
            shape.n[i] = 0.5 * (1.0 - xi * xi) * b;
            shape.dXi[i] = -xi * b;
            shape.dEta[i] = 0.5 * (1.0 - xi * xi) * ETA[i];
        }
        else
        {
            shape.n[i] = 0.5 * a * (1.0 - eta * eta);
            shape.dXi[i] = 0.5 * XI[i] * (1.0 - eta * eta);
            shape.dEta[i] = -eta * a;
        }
    }
    return shape;
}

ShapeAt shapeAt(std::size_t nodes, double xi, double eta)
{
    return isTriangle(nodes) ? triangleShapeAt(nodes, xi, eta) : quadrilateralShapeAt(nodes, xi, eta);
}

// A point of the parent domain of an element.
struct ParentPoint
{
    double xi = 0.0;
    double eta = 0.0;
};

// How far a point is beyond the sides of the parent domain (0: inside).
double excessOf(std::size_t nodes, ParentPoint p)
{
    return isTriangle(nodes) ? std::max({0.0, -p.xi, -p.eta, p.xi + p.eta - 1.0})
                             : std::max({0.0, std::abs(p.xi) - 1.0, std::abs(p.eta) - 1.0});
}

// The point brought back to the parent domain.
ParentPoint clamped(std::size_t nodes, ParentPoint p)
{
    if (!isTriangle(nodes))
        return {std::clamp(p.xi, -1.0, 1.0), std::clamp(p.eta, -1.0, 1.0)};
    p.xi = std::max(p.xi, 0.0);
    p.eta = std::max(p.eta, 0.0);
    const double sum = p.xi + p.eta;
    return sum > 1.0 ? ParentPoint{p.xi / sum, p.eta / sum} : p;
}

// A shell element the points of a footprint may be on: its grid points, their positions and its unit normal.
struct Candidate
{
    std::int32_t id = 0;
    std::span<const std::int32_t> grids;
    std::size_t nodes = 0;
    std::array<base::Vec3, MAX_NODES> x{};
    base::Vec3 normal;
    base::Vec3 centre;
    double radius = 0.0;
};

// Nothing for a shell with an absent mid-side point, with a grid point the model does not have, or whose corners
// are on a line.
std::optional<Candidate> candidateOf(const ShellMeshIndex& mesh, const ShellRef& shell)
{
    Candidate candidate;
    candidate.id = shell.id;
    candidate.grids = shell.grids;
    candidate.nodes = shell.grids.size();
    if (candidate.nodes != 3 && candidate.nodes != 4 && candidate.nodes != 6 && candidate.nodes != 8)
        return std::nullopt;
    const std::optional<base::Vec3> normal = mesh.normalOf(shell.grids);
    if (!normal)
        return std::nullopt;
    candidate.normal = *normal;
    for (std::size_t j = 0; j < candidate.nodes; ++j)
    {
        const std::optional<base::Vec3> position = shell.grids[j] != 0 ? mesh.positionOf(shell.grids[j]) : std::nullopt;
        if (!position)
            return std::nullopt;
        candidate.x[j] = *position;
    }
    const std::size_t corners = ShellMeshIndex::cornerCount(candidate.nodes);
    for (std::size_t c = 0; c < corners; ++c)
        candidate.centre = candidate.centre + candidate.x[c] / static_cast<double>(corners);
    for (std::size_t c = 0; c < corners; ++c)
        candidate.radius = std::max(candidate.radius, base::norm(candidate.x[c] - candidate.centre));
    return candidate;
}

// The point of the surface of an element nearest to q, by the Gauss-Newton iteration on its parent coordinates:
// with x(ξ, η) = Σ N_j x_j and its tangents a_ξ, a_η, each step solves (a_α · a_β) δ_β = a_α · (q − x). On a flat
// element this is the Newton iteration of the inverse of the isoparametric map.
ParentPoint nearestPoint(const Candidate& element, base::Vec3 q)
{
    ParentPoint p = isTriangle(element.nodes) ? ParentPoint{1.0 / 3.0, 1.0 / 3.0} : ParentPoint{};
    for (int step = 0; step < PROJECTION_STEPS; ++step)
    {
        const ShapeAt shape = shapeAt(element.nodes, p.xi, p.eta);
        base::Vec3 x;
        base::Vec3 alongXi;
        base::Vec3 alongEta;
        for (std::size_t j = 0; j < element.nodes; ++j)
        {
            x = x + shape.n[j] * element.x[j];
            alongXi = alongXi + shape.dXi[j] * element.x[j];
            alongEta = alongEta + shape.dEta[j] * element.x[j];
        }
        const double a11 = base::dot(alongXi, alongXi);
        const double a12 = base::dot(alongXi, alongEta);
        const double a22 = base::dot(alongEta, alongEta);
        const double b1 = base::dot(alongXi, q - x);
        const double b2 = base::dot(alongEta, q - x);
        const double determinant = a11 * a22 - a12 * a12;
        if (!(determinant > 0.0))
            break;
        const double stepXi = (a22 * b1 - a12 * b2) / determinant;
        const double stepEta = (a11 * b2 - a12 * b1) / determinant;
        p.xi += stepXi;
        p.eta += stepEta;
        if (!std::isfinite(p.xi) || !std::isfinite(p.eta))
            return {2.0 + FOOTPRINT_SIDE_TOLERANCE, 2.0 + FOOTPRINT_SIDE_TOLERANCE}; // not on the element
        if (std::abs(stepXi) + std::abs(stepEta) <= PROJECTION_TOLERANCE)
            break;
    }
    return p;
}

// A point located on an element of the sheet (the index of the element among those of SheetAround).
struct Located
{
    std::size_t element = 0;
    ParentPoint at;
};

// The shell elements around a pierced element that continue its sheet, ring after ring, and the points of space
// located on them.
class SheetAround
{
public:
    SheetAround(const ShellMeshIndex& mesh, const Candidate& pierced)
        : m_mesh(mesh), m_normal(pierced.normal),
          m_cosine(std::cos(PIERCING_PATCH_ANGLE_DEGREES * std::numbers::pi / 180.0))
    {
        // The first ring at once: a point on a side of the pierced element is on its neighbour too.
        m_elements.push_back(pierced);
        addRing();
    }

    [[nodiscard]] const Candidate& element(std::size_t index) const { return m_elements[index]; }

    // The elements that hold the point of the sheet nearest to q: every element it is in or on a side of; else,
    // once every ring has been looked at, the nearest element when the point is within FOOTPRINT_SIDE_TOLERANCE of
    // it; else none.
    [[nodiscard]] std::vector<Located> locate(base::Vec3 q)
    {
        for (;;)
        {
            std::vector<Located> holders; // at most the elements around one grid point
            std::optional<Located> nearest;
            double smallestExcess = FOOTPRINT_SIDE_TOLERANCE;
            for (std::size_t e = 0; e < m_elements.size(); ++e)
            {
                const Candidate& candidate = m_elements[e];
                if (base::norm(q - candidate.centre) > REACH_PER_RADIUS * candidate.radius)
                    continue;
                const ParentPoint at = nearestPoint(candidate, q);
                const double excess = excessOf(candidate.nodes, at);
                if (excess <= FOOTPRINT_ON_SIDE)
                    holders.push_back({e, clamped(candidate.nodes, at)});
                else if (excess <= smallestExcess)
                {
                    smallestExcess = excess;
                    nearest = Located{e, clamped(candidate.nodes, at)};
                }
            }
            if (!holders.empty())
                return holders;
            if (m_rings < FOOTPRINT_RINGS)
            {
                addRing();
                continue;
            }
            if (nearest)
                holders.push_back(*nearest);
            return holders;
        }
    }

private:
    // Adds the elements that hold a corner of an element of the last ring and continue the sheet.
    void addRing()
    {
        const std::size_t ringEnd = m_elements.size();
        for (std::size_t e = m_ringBegin; e < ringEnd; ++e)
        {
            const std::size_t corners = ShellMeshIndex::cornerCount(m_elements[e].nodes);
            for (std::size_t c = 0; c < corners; ++c)
                m_mesh.forEachShellAt(m_elements[e].grids[c], [&](const ShellRef& shell) { addElement(shell); });
        }
        m_ringBegin = ringEnd;
        ++m_rings;
    }

    void addElement(const ShellRef& shell)
    {
        const auto isSame = [&](const Candidate& element) { return element.id == shell.id; };
        if (std::any_of(m_elements.begin(), m_elements.end(), isSame))
            return;
        std::optional<Candidate> candidate = candidateOf(m_mesh, shell);
        if (!candidate || std::abs(base::dot(candidate->normal, m_normal)) < m_cosine)
            return;
        // every normal on the side of that of the pierced element
        if (base::dot(candidate->normal, m_normal) < 0.0)
            candidate->normal = -candidate->normal;
        m_elements.push_back(*candidate);
    }

    const ShellMeshIndex& m_mesh;
    base::Vec3 m_normal;
    double m_cosine;
    std::vector<Candidate> m_elements; // the pierced element, then ring after ring (a few tens of elements)
    std::size_t m_ringBegin = 0;
    int m_rings = 0;
};

void addSample(FootprintSamples& samples, const Candidate& element, ParentPoint at, double weight)
{
    const ShapeAt shape = shapeAt(element.nodes, at.xi, at.eta);
    samples.add(element.id, weight, element.grids, std::span<const double>(shape.n.data(), element.nodes));
}

} // namespace

void appendFootprint(const ShellMeshIndex& mesh, std::int32_t piercedElementId, std::span<const double> shapeValues,
                     double diameter, bool hasCorners, FootprintSamples& corners, FootprintSamples& mean)
{
    const auto endRows = [&]
    {
        corners.endRow();
        mean.endRow();
    };
    const std::optional<ShellRef> shell = diameter > 0.0 ? mesh.find(piercedElementId) : std::nullopt;
    const std::optional<Candidate> pierced = shell ? candidateOf(mesh, *shell) : std::nullopt;
    if (!pierced || shapeValues.size() != pierced->nodes)
        return endRows();

    // The square: its centre the point of the pierced element; its plane the tangent plane there, whose normal is
    // the mean of the normals of the elements that hold the point (on a grid point or a side of a curved skin the
    // pierced element is one of several, and the square must not lean to its side); its first side along the first
    // side of the pierced element brought to that plane, its second across.
    base::Vec3 centre;
    for (std::size_t j = 0; j < pierced->nodes; ++j)
        centre = centre + shapeValues[j] * pierced->x[j];
    SheetAround sheet(mesh, *pierced);
    base::Vec3 normal;
    for (const Located& holder : sheet.locate(centre))
        normal = normal + sheet.element(holder.element).normal;
    normal = base::norm(normal) > 0.0 ? base::normalized(normal) : pierced->normal;
    const base::Vec3 side = pierced->x[1] - pierced->x[0];
    const base::Vec3 inPlane = side - base::dot(side, normal) * normal;
    if (!(base::norm(inPlane) > 0.0))
        return endRows();
    const base::Vec3 first = base::normalized(inPlane);
    const base::Vec3 second = base::cross(normal, first);
    const double halfSide = 0.5 * FOOTPRINT_SIDE_PER_DIAMETER * diameter;
    const auto pointAt = [&](double s, double t, double half)
    { return centre + (half * s) * first + (half * t) * second; };

    // The corners: all four, on a square halved until the shells hold them.
    double half = halfSide;
    for (int halving = 0; hasCorners && halving <= FOOTPRINT_HALVINGS; ++halving, half *= 0.5)
    {
        std::array<Located, 4> found{};
        bool hasAll = true;
        for (std::size_t k = 0; k < SQUARE_CORNERS.size() && hasAll; ++k)
        {
            const base::Vec3 corner = pointAt(SQUARE_CORNERS[k][0], SQUARE_CORNERS[k][1], half);
            const std::vector<Located> holders = sheet.locate(corner);
            hasAll = !holders.empty();
            if (hasAll)
                found[k] = holders.front();
        }
        if (!hasAll)
            continue;
        for (const Located& corner : found)
            addSample(corners, sheet.element(corner.element), corner.at, 1.0 / static_cast<double>(found.size()));
        break;
    }
    // The Gauss points: those the shells hold, each shared by the elements it is on.
    for (const Abscissa& t : GAUSS_RULE)
        for (const Abscissa& s : GAUSS_RULE)
        {
            const std::vector<Located> holders = sheet.locate(pointAt(s.at, t.at, halfSide));
            for (const Located& holder : holders)
                addSample(mean, sheet.element(holder.element), holder.at,
                          s.weight * t.weight / static_cast<double>(holders.size()));
        }
    endRows();
}

} // namespace oos::input
