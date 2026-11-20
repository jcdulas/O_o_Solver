#pragma once

#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <functional>
#include <map>
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
// Each edge of two shells or more has a state (EdgeState):
// - smooth: the surface keeps one tangent plane across it and the edge follows its mesh line;
// - sharp: the surface folds along it; the edge follows its own mesh line and may be curved (the rim of a cylinder
//   closed by a flat end), and the mesh lines that would cross it end there;
// - raw: the surface folds along it and the edge is straight; an element whose edges are all raw is a facet.
// The state is declared (smoothSides, sharpSides, rawEdges, rawElements) or, where nothing declares it, decided by
// one angle (foldAngle): the edge is sharp when the normals of its two elements differ by more than that angle at one
// of its grids. Without an angle every edge that nothing declares is smooth. The reading holds no other criterion:
// which mesh line goes on into which is declared too where its own rules are not wanted (declaredLines). Not in the
// reference, which has the raw edges and elements alone.
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
    // Edges declared smooth, and edges declared sharp: for each, the shells whose side of the edge is named
    // (WHOLE_EDGE: every side). A side matters at an edge of three shells or more: a sharp side is smooth with no
    // other shell across the edge, a smooth side is smooth with the shell that faces it. Where both name a side, it
    // is sharp.
    std::map<EdgeKey, std::set<std::size_t>> smoothSides;
    std::map<EdgeKey, std::set<std::size_t>> sharpSides;
    // The angle (radians) between the normals of two shells at a grid of their edge above which an edge that nothing
    // declares is sharp; zero or negative: every such edge is sharp. Empty: no angle criterion, every such edge is
    // smooth.
    std::optional<double> foldAngle;
    // The mesh lines declared: for edge a → b at grid a (the key {a, b}), the grids c of the edges a − c that it goes
    // on into; no grid: the line ends at a. A key replaces the rules of the reading for that edge at that grid
    // (SurfaceReconstruction::candidates); the other end of a continuation is another key.
    std::map<std::pair<std::size_t, std::size_t>, std::set<std::size_t>> declaredLines;
    // The grids of the boundary of the mesh through which the boundary line goes on, when that is known (the others
    // are corners): set on the skin of the solids, where the sharp edges say it. Empty: the rule of the shells, by
    // the elements at the grid.
    std::optional<std::set<std::size_t>> throughGrids;
    // The translation components (0, 1, 2 in the basic system) held at the grids, as (grid, component): they orient
    // the common frame of an external edge (Topology::frames). The integrator decides which supports to pass.
    std::set<std::pair<std::size_t, int>> heldTranslations;

    // In smoothSides and sharpSides: every side of the edge.
    static constexpr std::size_t WHOLE_EDGE = static_cast<std::size_t>(-1);
};

// What the surface does at an edge of two shells or more.
enum class EdgeState
{
    Smooth, // one tangent plane across the edge, which follows its mesh line
    Sharp,  // a fold along the edge, which follows its own mesh line (at a junction: one side at least ends there)
    Raw,    // a fold along the edge, which is straight
};

// Where the state of an edge comes from, the strongest reason first.
enum class EdgeStateOrigin
{
    Declared,   // a raw, sharp or smooth declaration of the edge or of one of its sides
    RawElement, // an edge of a shell declared raw
    Angle,      // the angle between the normals of its shells against MeshReadingInput::foldAngle
    Junction,   // an edge of three shells or more where one of them faces no other
    Unmeasured, // smooth because the reading was given no angle
    Outline,    // an edge of one shell, or of a line element alone: nothing to decide
};

// The state of an edge, where it comes from, and the angle measured (edgeAngle; zero where fewer than two shells
// face each other).
struct EdgeReading
{
    EdgeState state = EdgeState::Smooth;
    EdgeStateOrigin origin = EdgeStateOrigin::Outline;
    double angle = 0.0;
};

// The unit normal of shell `shell` at its grid `grid`: the cross product of its two edges there (the normal of the
// bilinear surface of a quadrilateral at that corner, which follows a warped element; a triangle has one normal),
// turned as the grids of the shell turn. `fallback` where the two edges are along one line.
[[nodiscard]] Vector3 cornerNormal(std::span<const Vector3> grids, const ShellCorners& shell, std::size_t grid,
                                   const Vector3& fallback);

// The angle (radians, 0 to π) between the surfaces of two shells at grid `grid` of their common edge `grid` − `other`:
// that of their corner normals there once both shells are turned the same way about the edge, so that the numbering
// sense of the shells plays no part (two shells that run along the edge in opposite senses turn the same way).
[[nodiscard]] double cornerAngle(std::span<const Vector3> grids, const ShellCorners& first, const ShellCorners& second,
                                 std::size_t grid, std::size_t other);

// The declared raw edges and the edges of the raw elements that nothing declares smooth or sharp.
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
    // The state of each edge (resolved from the declarations and the angle of the input).
    std::vector<EdgeReading> states;
    // For each edge, the shell that each of its shells is smooth with across it. A shell without one ends there: at a
    // sharp edge, or at a junction where no shell faces it. Across a raw edge the two shells keep each other: the
    // reading asks the raw edges apart (isRaw), as the reference does.
    std::vector<std::map<std::size_t, std::size_t>> partners;

    // The shared edges of the mesh, their common frames and their curves.
    // Reference: src/model.rs::Topology::new.
    [[nodiscard]] static Topology fromMesh(const MeshReadingInput& input);

    // Reference: src/model.rs::Topology::is_external.
    [[nodiscard]] bool isExternal(std::size_t edge) const { return edgeElements[edge].size() == 1; }
    // The index of edge a − b; throws InternalError when the mesh has no such edge.
    [[nodiscard]] std::size_t indexOf(std::size_t a, std::size_t b) const;
    [[nodiscard]] bool isRaw(std::size_t a, std::size_t b) const { return rawEdges.contains(EdgeKey::of(a, b)); }
    [[nodiscard]] bool isSharp(std::size_t a, std::size_t b) const
    {
        return states[indexOf(a, b)].state == EdgeState::Sharp;
    }
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
