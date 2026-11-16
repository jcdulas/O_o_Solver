#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

namespace oos::elements::mixedHybrid
{

namespace
{

// Reference: volume::TET_V, PYR_V, PEN_V, HEX_V
constexpr std::array<SolidPoint, 4> TETRAHEDRON_VERTICES{
    {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}}};
constexpr std::array<SolidPoint, 5> PYRAMID_VERTICES{
    {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {0.0, 0.0, 1.0}}};
constexpr std::array<SolidPoint, 6> PENTAHEDRON_VERTICES{
    {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {-1.0, 1.0, 1.0}}};
constexpr std::array<SolidPoint, 8> HEXAHEDRON_VERTICES{{{-1.0, -1.0, -1.0},
                                                         {1.0, -1.0, -1.0},
                                                         {1.0, 1.0, -1.0},
                                                         {-1.0, 1.0, -1.0},
                                                         {-1.0, -1.0, 1.0},
                                                         {1.0, -1.0, 1.0},
                                                         {1.0, 1.0, 1.0},
                                                         {-1.0, 1.0, 1.0}}};

// The edges in the order of the mid-side nodes of the cards (SolidTetra G5-G10, SolidPyramid G6-G13, SolidPenta
// G7-G15, SolidHexa G9-G20).
// Reference: volume::TET_E, PYR_E, PEN_E, HEX_E
constexpr std::array<ShapeEdge, 6> TETRAHEDRON_EDGES{{{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}}};
constexpr std::array<ShapeEdge, 8> PYRAMID_EDGES{{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}}};
constexpr std::array<ShapeEdge, 9> PENTAHEDRON_EDGES{
    {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 4}, {2, 5}, {3, 4}, {4, 5}, {5, 3}}};
constexpr std::array<ShapeEdge, 12> HEXAHEDRON_EDGES{
    {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 4}}};

// Reference: volume::TET_F, PYR_F, PEN_F, HEX_F
constexpr std::array<ShapeFace, 4> TETRAHEDRON_FACES{
    {{3, {1, 2, 3, 0}}, {3, {0, 2, 3, 0}}, {3, {0, 1, 3, 0}}, {3, {0, 1, 2, 0}}}};
constexpr std::array<ShapeFace, 5> PYRAMID_FACES{
    {{4, {0, 1, 2, 3}}, {3, {0, 1, 4, 0}}, {3, {1, 2, 4, 0}}, {3, {2, 3, 4, 0}}, {3, {3, 0, 4, 0}}}};
constexpr std::array<ShapeFace, 5> PENTAHEDRON_FACES{
    {{3, {0, 1, 2, 0}}, {3, {3, 4, 5, 0}}, {4, {0, 1, 4, 3}}, {4, {1, 2, 5, 4}}, {4, {2, 0, 3, 5}}}};
constexpr std::array<ShapeFace, 6> HEXAHEDRON_FACES{
    {{4, {0, 1, 2, 3}}, {4, {4, 5, 6, 7}}, {4, {0, 1, 5, 4}}, {4, {1, 2, 6, 5}}, {4, {2, 3, 7, 6}}, {4, {3, 0, 4, 7}}}};

constexpr std::size_t TETRAHEDRON_CORNERS = 4;
constexpr std::size_t PYRAMID_CORNERS = 5;
constexpr std::size_t PENTAHEDRON_CORNERS = 6;
constexpr std::size_t HEXAHEDRON_CORNERS = 8;

} // namespace

std::optional<VolumeShape> volumeShapeOfCornerCount(std::size_t cornerCount)
{
    switch (cornerCount)
    {
    case TETRAHEDRON_CORNERS:
        return VolumeShape::Tetrahedron;
    case PYRAMID_CORNERS:
        return VolumeShape::Pyramid;
    case PENTAHEDRON_CORNERS:
        return VolumeShape::Pentahedron;
    case HEXAHEDRON_CORNERS:
        return VolumeShape::Hexahedron;
    default:
        return std::nullopt;
    }
}

const char* shapeName(VolumeShape shape)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return "tetrahedron";
    case VolumeShape::Pyramid:
        return "pyramid";
    case VolumeShape::Pentahedron:
        return "pentahedron";
    case VolumeShape::Hexahedron:
        break;
    }
    return "hexahedron";
}

std::span<const SolidPoint> shapeVertices(VolumeShape shape)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return TETRAHEDRON_VERTICES;
    case VolumeShape::Pyramid:
        return PYRAMID_VERTICES;
    case VolumeShape::Pentahedron:
        return PENTAHEDRON_VERTICES;
    case VolumeShape::Hexahedron:
        break;
    }
    return HEXAHEDRON_VERTICES;
}

std::size_t shapeCornerCount(VolumeShape shape)
{
    return shapeVertices(shape).size();
}

std::span<const ShapeEdge> shapeEdges(VolumeShape shape)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return TETRAHEDRON_EDGES;
    case VolumeShape::Pyramid:
        return PYRAMID_EDGES;
    case VolumeShape::Pentahedron:
        return PENTAHEDRON_EDGES;
    case VolumeShape::Hexahedron:
        break;
    }
    return HEXAHEDRON_EDGES;
}

std::span<const ShapeFace> shapeFaces(VolumeShape shape)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return TETRAHEDRON_FACES;
    case VolumeShape::Pyramid:
        return PYRAMID_FACES;
    case VolumeShape::Pentahedron:
        return PENTAHEDRON_FACES;
    case VolumeShape::Hexahedron:
        break;
    }
    return HEXAHEDRON_FACES;
}

SolidPoint shapeCentre(VolumeShape shape)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return {-0.5, -0.5, -0.5};
    case VolumeShape::Pyramid:
        return {0.0, 0.0, -0.5};
    case VolumeShape::Pentahedron:
        return {-1.0 / 3.0, -1.0 / 3.0, 0.0};
    case VolumeShape::Hexahedron:
        break;
    }
    return {0.0, 0.0, 0.0};
}

std::optional<std::size_t> shapeEdgeIndex(VolumeShape shape, std::size_t a, std::size_t b)
{
    const std::span<const ShapeEdge> edges = shapeEdges(shape);
    for (std::size_t e = 0; e < edges.size(); ++e)
        if ((edges[e][0] == a && edges[e][1] == b) || (edges[e][0] == b && edges[e][1] == a))
            return e;
    return std::nullopt;
}

std::size_t edgeFunctionCount(std::size_t traceDegree)
{
    return traceDegree >= 2 ? traceDegree - 1 : 0;
}

std::size_t faceFunctionCount(std::size_t faceDegree, std::size_t cornerCount)
{
    if (cornerCount == 3)
        return faceDegree >= 3 ? (faceDegree - 1) * (faceDegree - 2) / 2 : 0;
    return faceDegree >= 2 ? (faceDegree - 1) * (faceDegree - 1) : 0;
}

std::size_t volumeSpaceDimension(VolumeShape shape, std::size_t n)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return tetrahedronSpaceDimension(n);
    case VolumeShape::Pyramid:
        return pyramidSpaceDimension(n);
    case VolumeShape::Pentahedron:
        return pentahedronSpaceDimension(n);
    case VolumeShape::Hexahedron:
        break;
    }
    return hexahedronSpaceDimension(n);
}

std::vector<std::size_t> volumeLevels(VolumeShape shape, std::size_t n)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return tetrahedronLevels(n);
    case VolumeShape::Pyramid:
        return pyramidLevels(n);
    case VolumeShape::Pentahedron:
        return pentahedronLevels(n);
    case VolumeShape::Hexahedron:
        break;
    }
    return hexahedronLevels(n);
}

std::vector<VolumeRulePoint> volumeRule(VolumeShape shape, std::size_t n)
{
    switch (shape)
    {
    case VolumeShape::Tetrahedron:
        return tetrahedronRule(n);
    case VolumeShape::Pyramid:
        return pyramidRule(n);
    case VolumeShape::Pentahedron:
        return pentahedronRule(n);
    case VolumeShape::Hexahedron:
        break;
    }
    return hexahedronRule(n);
}

} // namespace oos::elements::mixedHybrid
