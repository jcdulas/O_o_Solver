#pragma once

#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <functional>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The mesh whose surface is read (MeshReadingInput) and its topology (Topology): the edges of the shells and of the
// line elements, their owners, their common frames, and the curve of each edge read from the mesh (document 25 of the
// reference, 3.2, and document 26, 3.2).
//
// The reading has no angle criterion: only the declarations decide where the surface is not smooth. A raw edge is a
// straight line along which the surface may fold (the mesh lines that cross it end there); an element whose edges are
// all raw is a facet; every other edge is smooth: it follows its mesh line and the surface is smooth across it.
//
// Invariants: grids, shells and lines are given by index (0-based, into MeshReadingInput::grids). The edges of the
// shells are numbered in the order they are met (shells in order, edge k of a shell from its grid k to its grid
// k + 1), then the edges of the line elements that no shell has. Thread-safety: plain values; a Topology is immutable
// once built and safe to share between threads.
//
// Reference: src/model.rs (Model, Topology).

// An edge as the pair of its grids, lower first.
struct EdgeKey
{
    std::size_t lower = 0;
    std::size_t higher = 0;

    [[nodiscard]] static constexpr EdgeKey of(std::size_t a, std::size_t b)
    {
        return a < b ? EdgeKey{a, b} : EdgeKey{b, a};
    }
    friend constexpr auto operator<=>(const EdgeKey&, const EdgeKey&) = default;
};

struct EdgeKeyHash
{
    [[nodiscard]] std::size_t operator()(const EdgeKey& key) const noexcept
    {
        // a multiplicative mix of the two grids (Fibonacci hashing)
        constexpr std::size_t MIX = 0x9E3779B97F4A7C15ULL;
        return std::hash<std::size_t>{}(key.lower * MIX ^ key.higher);
    }
};

// The grids of one shell element, in order: four for a quadrilateral, three for a triangle.
// Reference: src/model.rs::Element (its grids).
struct ShellCorners
{
    std::array<std::size_t, 4> grids{};
    std::size_t count = 4;

    [[nodiscard]] static constexpr ShellCorners quadrilateral(std::size_t g0, std::size_t g1, std::size_t g2,
                                                              std::size_t g3)
    {
        return {{g0, g1, g2, g3}, 4};
    }
    [[nodiscard]] static constexpr ShellCorners triangle(std::size_t g0, std::size_t g1, std::size_t g2)
    {
        return {{g0, g1, g2, 0}, 3};
    }
    [[nodiscard]] std::span<const std::size_t> view() const { return {grids.data(), count}; }
    // Grid k, taken modulo the corner count.
    [[nodiscard]] std::size_t at(std::size_t k) const { return grids[k % count]; }
};

// The two grids of a line element, and the orientation vector of its section when it has one (a rod has none).
struct LineGrids
{
    std::array<std::size_t, 2> grids{};
    std::optional<Vector3> orientation;
};

// Everything the reading of the surface takes from the model.
struct MeshReadingInput
{
    // Grid positions in the basic system.
    std::vector<Vector3> grids;
    std::vector<ShellCorners> shells;
    // Line elements: their edges come after those of the shells.
    std::vector<LineGrids> lines;
    // Edges declared raw.
    std::set<EdgeKey> rawEdges;
    // Shells declared raw: all their edges are raw (a facet).
    std::vector<std::size_t> rawElements;
    // Edges of the skin of the solids declared sharp: the skin is not smooth across them (SolidSkin.hpp).
    std::set<EdgeKey> sharpEdges;
    // The grids of the boundary of the mesh through which the boundary line goes on, when that is known (the others
    // are corners): set on the skin of the solids, where the sharp edges say it. Empty: the rule of the shells, by
    // the elements at the grid.
    std::optional<std::set<std::size_t>> throughGrids;
    // The translation components (0, 1, 2 in the basic system) held at the grids, as (grid, component): they orient
    // the common frame of an external edge (Topology::frames). The integrator decides which supports to pass.
    std::set<std::pair<std::size_t, int>> heldTranslations;

    [[nodiscard]] bool isSharp(std::size_t a, std::size_t b) const { return sharpEdges.contains(EdgeKey::of(a, b)); }
};

// The declared raw edges and the edges of the raw elements.
// Reference: src/model.rs::Model::set_raw, Model::raw_element.
[[nodiscard]] std::set<EdgeKey> rawEdgeSet(const MeshReadingInput& input);

// Reference: src/model.rs::Topology.
struct Topology
{
    std::vector<EdgeKey> edges;
    std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> edgeIndex;
    // The shells of each edge, in the order of the input; none on the edge of a line element that no shell has (these
    // edges come last).
    std::vector<std::vector<std::size_t>> edgeElements;
    // The common frame (e, b, n) of each edge, e from the lower to the higher grid.
    std::vector<std::array<Vector3, 3>> frames;
    // The bulge of the conic of each edge (half the offset of its control point from the middle of its chord).
    std::vector<Vector3> bulges;
    // The weight of the conic of each edge where it is not that of its control polygon (NaN: the sine of half the
    // angle of the control polygon, the circular arc). An edge that is a plane section of the surface between two
    // grids takes the cosine of half the angle of the normals at its two grids: the section of a cylinder by a plane
    // that is not across its axis is an arc of an ellipse, the image of the arc of a circle whose half angle is that
    // one, and a conic keeps its weight in such a map. On a sphere and on a plane the two weights are the same.
    std::vector<double> weights;
    // The unit normal of each shell: the cross product of the diagonals (of two sides on a triangle).
    std::vector<Vector3> normals;
    // The raw edges (rawEdgeSet of the input).
    std::set<EdgeKey> rawEdges;

    // The shared edges of the mesh, their common frames and their curves.
    // Reference: src/model.rs::Topology::new.
    [[nodiscard]] static Topology fromMesh(const MeshReadingInput& input);

    // Reference: src/model.rs::Topology::is_external.
    [[nodiscard]] bool isExternal(std::size_t edge) const { return edgeElements[edge].size() == 1; }
    // The index of edge a − b; throws InternalError when the mesh has no such edge.
    [[nodiscard]] std::size_t indexOf(std::size_t a, std::size_t b) const;
    [[nodiscard]] bool isRaw(std::size_t a, std::size_t b) const { return rawEdges.contains(EdgeKey::of(a, b)); }
};

// The element axes (rows x_e, y_e, z_e in basic components) of a quadrilateral, and the direction of its side G1-G2
// projected on its mean plane: z_e along the cross product of the diagonals, x_e the bisector of the diagonals in the
// mean plane, y_e = z_e × x_e.
struct ElementAxes
{
    std::array<Vector3, 3> axes{};
    Vector3 side{};
};

// Reference: src/model.rs::element_axes.
[[nodiscard]] ElementAxes elementAxes(const std::array<Vector3, 4>& corners);

// The element axes of a triangle: x_e along its side G1-G2, z_e normal to the plane of its grids; the side is x_e.
// Any other pair of axes in that plane gives the same element (tangentFrameXy: the local frame turns by the same angle
// at every point, and the stress space of total degree is unchanged by a constant rotation).
// Reference: src/model.rs::element_axes_tri.
[[nodiscard]] ElementAxes elementAxesTriangle(const std::array<Vector3, 3>& corners);

} // namespace oos::elements::mixedHybrid
