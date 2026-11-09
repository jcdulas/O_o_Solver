#include "oos/elements/core/SolidShapeFunctions.hpp"

#include <array>

namespace oos::elements
{

namespace
{

constexpr std::array<ParentPoint, 8> HEXA_CORNERS{{{-1.0, -1.0, -1.0},
                                                   {1.0, -1.0, -1.0},
                                                   {1.0, 1.0, -1.0},
                                                   {-1.0, 1.0, -1.0},
                                                   {-1.0, -1.0, 1.0},
                                                   {1.0, -1.0, 1.0},
                                                   {1.0, 1.0, 1.0},
                                                   {-1.0, 1.0, 1.0}}};
constexpr std::array<ParentPoint, 5> PYRAM_CORNERS{
    {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {0.0, 0.0, 1.0}}};
constexpr std::array<ParentPoint, 6> PENTA_CORNERS{
    {{0.0, 0.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}};
constexpr std::array<ParentPoint, 4> TETRA_CORNERS{
    {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};

constexpr std::array<SolidEdge, 6> TETRA_EDGES{{{0, 1, 4}, {1, 2, 5}, {2, 0, 6}, {0, 3, 7}, {1, 3, 8}, {2, 3, 9}}};
constexpr std::array<SolidEdge, 9> PENTA_EDGES{
    {{0, 1, 6}, {1, 2, 7}, {2, 0, 8}, {0, 3, 9}, {1, 4, 10}, {2, 5, 11}, {3, 4, 12}, {4, 5, 13}, {5, 3, 14}}};
constexpr std::array<SolidEdge, 8> PYRAM_EDGES{
    {{0, 1, 5}, {1, 2, 6}, {2, 3, 7}, {3, 0, 8}, {0, 4, 9}, {1, 4, 10}, {2, 4, 11}, {3, 4, 12}}};
constexpr std::array<SolidEdge, 12> HEXA_EDGES{{{0, 1, 8},
                                                {1, 2, 9},
                                                {2, 3, 10},
                                                {3, 0, 11},
                                                {0, 4, 12},
                                                {1, 5, 13},
                                                {2, 6, 14},
                                                {3, 7, 15},
                                                {4, 5, 16},
                                                {5, 6, 17},
                                                {6, 7, 18},
                                                {7, 4, 19}}};

constexpr std::array<SolidFace, 4> TETRA_FACES{
    {{3, {0, 2, 1, 0}}, {3, {0, 1, 3, 0}}, {3, {1, 2, 3, 0}}, {3, {0, 3, 2, 0}}}};
constexpr std::array<SolidFace, 5> PENTA_FACES{
    {{3, {0, 2, 1, 0}}, {3, {3, 4, 5, 0}}, {4, {0, 1, 4, 3}}, {4, {1, 2, 5, 4}}, {4, {2, 0, 3, 5}}}};
constexpr std::array<SolidFace, 5> PYRAM_FACES{
    {{4, {0, 3, 2, 1}}, {3, {0, 1, 4, 0}}, {3, {1, 2, 4, 0}}, {3, {2, 3, 4, 0}}, {3, {3, 0, 4, 0}}}};
constexpr std::array<SolidFace, 6> HEXA_FACES{
    {{4, {0, 3, 2, 1}}, {4, {4, 5, 6, 7}}, {4, {0, 1, 5, 4}}, {4, {1, 2, 6, 5}}, {4, {2, 3, 7, 6}}, {4, {3, 0, 4, 7}}}};

// The parent coordinates of the mid-side nodes of the 20-node hexahedron, G9 to G20 (the middles of HEXA_EDGES).
[[nodiscard]] ParentPoint hexaMiddle(std::size_t edge)
{
    const SolidEdge& e = HEXA_EDGES[edge];
    const ParentPoint& a = HEXA_CORNERS[e.first];
    const ParentPoint& b = HEXA_CORNERS[e.second];
    return {0.5 * (a.xi + b.xi), 0.5 * (a.eta + b.eta), 0.5 * (a.zeta + b.zeta)};
}

void setNode(SolidShapeValues& values, std::size_t node, double n, double dXi, double dEta, double dZeta)
{
    values.n[node] = n;
    values.dXi[node] = dXi;
    values.dEta[node] = dEta;
    values.dZeta[node] = dZeta;
}

void addNode(SolidShapeValues& values, std::size_t node, const SolidShapeValues& from, std::size_t source)
{
    values.n[node] += from.n[source];
    values.dXi[node] += from.dXi[source];
    values.dEta[node] += from.dEta[source];
    values.dZeta[node] += from.dZeta[source];
}

[[nodiscard]] SolidShapeValues hexa8At(ParentPoint p)
{
    SolidShapeValues v;
    v.count = 8;
    for (std::size_t i = 0; i < 8; ++i)
    {
        const ParentPoint& c = HEXA_CORNERS[i];
        const double a = 1.0 + c.xi * p.xi;
        const double b = 1.0 + c.eta * p.eta;
        const double d = 1.0 + c.zeta * p.zeta;
        setNode(v, i, 0.125 * a * b * d, 0.125 * c.xi * b * d, 0.125 * c.eta * a * d, 0.125 * c.zeta * a * b);
    }
    return v;
}

[[nodiscard]] SolidShapeValues hexa20At(ParentPoint p)
{
    SolidShapeValues v;
    v.count = 20;
    for (std::size_t i = 0; i < 8; ++i)
    {
        const ParentPoint& c = HEXA_CORNERS[i];
        const double a = 1.0 + c.xi * p.xi;
        const double b = 1.0 + c.eta * p.eta;
        const double d = 1.0 + c.zeta * p.zeta;
        const double s = c.xi * p.xi + c.eta * p.eta + c.zeta * p.zeta - 2.0;
        // N = ⅛ a b d s; ∂N/∂ξ = ⅛ ξ_i b d (s + a), and the same for η, ζ.
        setNode(v, i, 0.125 * a * b * d * s, 0.125 * c.xi * b * d * (s + a), 0.125 * c.eta * a * d * (s + b),
                0.125 * c.zeta * a * b * (s + d));
    }
    for (std::size_t edge = 0; edge < HEXA_EDGES.size(); ++edge)
    {
        const ParentPoint m = hexaMiddle(edge);
        const std::size_t node = 8 + edge;
        if (m.xi == 0.0)
        {
            const double b = 1.0 + m.eta * p.eta;
            const double d = 1.0 + m.zeta * p.zeta;
            const double q = 1.0 - p.xi * p.xi;
            setNode(v, node, 0.25 * q * b * d, -0.5 * p.xi * b * d, 0.25 * q * m.eta * d, 0.25 * q * b * m.zeta);
        }
        else if (m.eta == 0.0)
        {
            const double a = 1.0 + m.xi * p.xi;
            const double d = 1.0 + m.zeta * p.zeta;
            const double q = 1.0 - p.eta * p.eta;
            setNode(v, node, 0.25 * a * q * d, 0.25 * m.xi * q * d, -0.5 * p.eta * a * d, 0.25 * a * q * m.zeta);
        }
        else
        {
            const double a = 1.0 + m.xi * p.xi;
            const double b = 1.0 + m.eta * p.eta;
            const double q = 1.0 - p.zeta * p.zeta;
            setNode(v, node, 0.25 * a * b * q, 0.25 * m.xi * b * q, 0.25 * a * m.eta * q, -0.5 * p.zeta * a * b);
        }
    }
    return v;
}

// Pyramids: the hexahedron with its face ζ = 1 (corners G5-G8, and G17-G20 for 20 nodes) collapsed into the apex.
[[nodiscard]] SolidShapeValues pyram5At(ParentPoint p)
{
    const SolidShapeValues hexa = hexa8At(p);
    SolidShapeValues v;
    v.count = 5;
    for (std::size_t i = 0; i < 4; ++i)
        addNode(v, i, hexa, i);
    for (std::size_t i = 4; i < 8; ++i)
        addNode(v, 4, hexa, i);
    return v;
}

[[nodiscard]] SolidShapeValues pyram13At(ParentPoint p)
{
    const SolidShapeValues hexa = hexa20At(p);
    SolidShapeValues v;
    v.count = 13;
    for (std::size_t i = 0; i < 4; ++i)
        addNode(v, i, hexa, i);
    for (std::size_t i = 4; i < 8; ++i)
        addNode(v, 4, hexa, i);
    for (std::size_t i = 16; i < 20; ++i)
        addNode(v, 4, hexa, i);
    // Base mid-side nodes G6-G9 are the hexahedron's G9-G12; the edges to the apex G10-G13 its G13-G16.
    for (std::size_t i = 0; i < 8; ++i)
        addNode(v, 5 + i, hexa, 8 + i);
    return v;
}

[[nodiscard]] SolidShapeValues tetraAt(ParentPoint p, bool isQuadratic)
{
    // L1 = 1 − ξ − η − ζ, L2 = ξ, L3 = η, L4 = ζ, and their derivatives along ξ, η, ζ.
    const std::array<double, 4> l{1.0 - p.xi - p.eta - p.zeta, p.xi, p.eta, p.zeta};
    constexpr std::array<std::array<double, 3>, 4> D_L{
        {{-1.0, -1.0, -1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    SolidShapeValues v;
    if (!isQuadratic)
    {
        v.count = 4;
        for (std::size_t i = 0; i < 4; ++i)
            setNode(v, i, l[i], D_L[i][0], D_L[i][1], D_L[i][2]);
        return v;
    }
    v.count = 10;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const double factor = 4.0 * l[i] - 1.0;
        setNode(v, i, l[i] * (2.0 * l[i] - 1.0), factor * D_L[i][0], factor * D_L[i][1], factor * D_L[i][2]);
    }
    for (std::size_t e = 0; e < TETRA_EDGES.size(); ++e)
    {
        const std::size_t a = TETRA_EDGES[e].first;
        const std::size_t b = TETRA_EDGES[e].second;
        std::array<double, 3> d{};
        for (std::size_t k = 0; k < 3; ++k)
            d[k] = 4.0 * (D_L[a][k] * l[b] + l[a] * D_L[b][k]);
        setNode(v, TETRA_EDGES[e].middle, 4.0 * l[a] * l[b], d[0], d[1], d[2]);
    }
    return v;
}

[[nodiscard]] SolidShapeValues pentaAt(ParentPoint p, bool isQuadratic)
{
    const std::array<double, 3> l{1.0 - p.xi - p.eta, p.xi, p.eta};
    constexpr std::array<std::array<double, 2>, 3> D_L{{{-1.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}}};
    const double z = p.zeta;
    SolidShapeValues v;
    if (!isQuadratic)
    {
        v.count = 6;
        for (std::size_t i = 0; i < 3; ++i)
        {
            setNode(v, i, 0.5 * l[i] * (1.0 - z), 0.5 * D_L[i][0] * (1.0 - z), 0.5 * D_L[i][1] * (1.0 - z),
                    -0.5 * l[i]);
            setNode(v, i + 3, 0.5 * l[i] * (1.0 + z), 0.5 * D_L[i][0] * (1.0 + z), 0.5 * D_L[i][1] * (1.0 + z),
                    0.5 * l[i]);
        }
        return v;
    }
    v.count = 15;
    const double q = 1.0 - z * z;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (const double side : {-1.0, 1.0})
        {
            // ½ L(2L − 1)(1 + side ζ) − ½ L (1 − ζ²)
            const double h = 1.0 + side * z;
            const double dl = 0.5 * ((4.0 * l[i] - 1.0) * h - q);
            setNode(v, side < 0.0 ? i : i + 3, 0.5 * l[i] * (2.0 * l[i] - 1.0) * h - 0.5 * l[i] * q, dl * D_L[i][0],
                    dl * D_L[i][1], 0.5 * l[i] * (2.0 * l[i] - 1.0) * side + l[i] * z);
        }
    }
    // Triangle edges: G7-G9 at ζ = −1 and G13-G15 at ζ = 1, 2 L_a L_b (1 ± ζ).
    constexpr std::array<std::array<std::size_t, 2>, 3> TRIANGLE_EDGES{{{0, 1}, {1, 2}, {2, 0}}};
    for (std::size_t e = 0; e < 3; ++e)
    {
        const std::size_t a = TRIANGLE_EDGES[e][0];
        const std::size_t b = TRIANGLE_EDGES[e][1];
        const double lab = l[a] * l[b];
        const double dr = D_L[a][0] * l[b] + l[a] * D_L[b][0];
        const double ds = D_L[a][1] * l[b] + l[a] * D_L[b][1];
        setNode(v, 6 + e, 2.0 * lab * (1.0 - z), 2.0 * dr * (1.0 - z), 2.0 * ds * (1.0 - z), -2.0 * lab);
        setNode(v, 12 + e, 2.0 * lab * (1.0 + z), 2.0 * dr * (1.0 + z), 2.0 * ds * (1.0 + z), 2.0 * lab);
    }
    // Vertical edges G10-G12: L_i (1 − ζ²).
    for (std::size_t i = 0; i < 3; ++i)
        setNode(v, 9 + i, l[i] * q, D_L[i][0] * q, D_L[i][1] * q, -2.0 * z * l[i]);
    return v;
}

} // namespace

SolidShapeValues solidShapeAt(SolidShape shape, ParentPoint point)
{
    switch (shape)
    {
    case SolidShape::Tetra4:
        return tetraAt(point, false);
    case SolidShape::Tetra10:
        return tetraAt(point, true);
    case SolidShape::Penta6:
        return pentaAt(point, false);
    case SolidShape::Penta15:
        return pentaAt(point, true);
    case SolidShape::Pyram5:
        return pyram5At(point);
    case SolidShape::Pyram13:
        return pyram13At(point);
    case SolidShape::Hexa8:
        return hexa8At(point);
    case SolidShape::Hexa20:
        return hexa20At(point);
    }
    return {};
}

std::size_t solidNodeCount(SolidShape shape) noexcept
{
    constexpr std::array<std::size_t, 8> COUNTS{4, 10, 6, 15, 5, 13, 8, 20};
    return COUNTS[static_cast<std::size_t>(shape)];
}

std::size_t solidCornerCount(SolidShape shape) noexcept
{
    constexpr std::array<std::size_t, 8> COUNTS{4, 4, 6, 6, 5, 5, 8, 8};
    return COUNTS[static_cast<std::size_t>(shape)];
}

bool isQuadraticSolid(SolidShape shape) noexcept
{
    return shape == SolidShape::Tetra10 || shape == SolidShape::Penta15 || shape == SolidShape::Pyram13 ||
           shape == SolidShape::Hexa20;
}

std::span<const ParentPoint> solidCornerPoints(SolidShape shape) noexcept
{
    switch (shape)
    {
    case SolidShape::Tetra4:
    case SolidShape::Tetra10:
        return TETRA_CORNERS;
    case SolidShape::Penta6:
    case SolidShape::Penta15:
        return PENTA_CORNERS;
    case SolidShape::Pyram5:
    case SolidShape::Pyram13:
        return PYRAM_CORNERS;
    case SolidShape::Hexa8:
    case SolidShape::Hexa20:
        return HEXA_CORNERS;
    }
    return {};
}

ParentPoint solidCentre(SolidShape shape) noexcept
{
    switch (shape)
    {
    case SolidShape::Tetra4:
    case SolidShape::Tetra10:
        return {0.25, 0.25, 0.25};
    case SolidShape::Penta6:
    case SolidShape::Penta15:
        return {1.0 / 3.0, 1.0 / 3.0, 0.0};
    default:
        return {0.0, 0.0, 0.0};
    }
}

std::span<const SolidEdge> solidEdges(SolidShape shape) noexcept
{
    switch (shape)
    {
    case SolidShape::Tetra4:
    case SolidShape::Tetra10:
        return TETRA_EDGES;
    case SolidShape::Penta6:
    case SolidShape::Penta15:
        return PENTA_EDGES;
    case SolidShape::Pyram5:
    case SolidShape::Pyram13:
        return PYRAM_EDGES;
    case SolidShape::Hexa8:
    case SolidShape::Hexa20:
        return HEXA_EDGES;
    }
    return {};
}

std::span<const SolidFace> solidFaces(SolidShape shape) noexcept
{
    switch (shape)
    {
    case SolidShape::Tetra4:
    case SolidShape::Tetra10:
        return TETRA_FACES;
    case SolidShape::Penta6:
    case SolidShape::Penta15:
        return PENTA_FACES;
    case SolidShape::Pyram5:
    case SolidShape::Pyram13:
        return PYRAM_FACES;
    case SolidShape::Hexa8:
    case SolidShape::Hexa20:
        return HEXA_FACES;
    }
    return {};
}

} // namespace oos::elements
