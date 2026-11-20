#pragma once

// Internal to the geometry of the mixed hybrid formulation: the reading of the shared edge curves of a mesh
// (Topology::fromMesh). Document 25 of the reference, 3.2, and document 26, 3.2.
//
// The states of the edges are those of the topology (Topology::states and partners, from the declarations and the
// fold angle of the input). A raw edge is straight, and a mesh line does not go on across a raw edge. A sharp edge
// follows its own mesh line, which is geometry for the elements on both sides, and each side ends there as at a
// junction where no element faces it: the lines that reach it are prolonged from their other grids. Which line goes
// on into which follows the rules of candidates, or the lines declared (MeshReadingInput::declaredLines).
//
// Every smooth edge follows its mesh line (parallels and meridians of a sphere stay parallels and meridians): it is
// the conic tangent to the line at its two grids. The tangent of a mesh line at a grid is that of the circle through
// the grid and its two neighbours on the line, corrected for the variation of the curvature along the line. At the
// open end of a line the tangent is prolonged from the other grids of the line, and the boundary of the mesh tells
// whether the surface is symmetric there or cut: the tangent moves towards the normal of the plane of the boundary
// when that normal is within its uncertainty.
//
// The reading caches what it computes (tangents, normals, fits) as it goes: every quantity is a function of the mesh
// alone, so the order of the queries does not change the results. Thread-safety: none (one reading on one thread).
// Ownership: borrows the input and the topology, which must outlive it.
//
// Reference: src/model.rs::Reconstruction.

#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace oos::elements::mixedHybrid::detail
{

class SurfaceReconstruction
{
public:
    SurfaceReconstruction(const MeshReadingInput& input, const Topology& topology);

    // The bulge of the curve of edge a − b: half the offset of the control point of its conic from the middle of the
    // chord. The conic is tangent to the mesh line at both grids and lies in the plane of the sum of the parts of the
    // two tangents normal to the chord. Zero on a raw edge.
    // Reference: Reconstruction::bulge.
    [[nodiscard]] Vector3 bulge(std::size_t a, std::size_t b);

    // The normal at a of the edge a − b when that edge was read as a plane section of the surface (the edges of
    // triangles, Topology::weights), else null.
    [[nodiscard]] const Vector3* sectionNormal(std::size_t a, std::size_t b) const;

private:
    // An ordered pair of grids (a, b): the edge a → b seen from a. Or (element, grid).
    using GridPair = std::pair<std::size_t, std::size_t>;
    // An element at one of its grids, with the far grid of its other edge there.
    using RingStep = std::pair<std::size_t, std::size_t>;

    // What the mesh line of edge a → b does at a.
    struct Continuation
    {
        bool smooth = false;  // the line goes on through a across smooth edges
        Vector3 tangent{};    // its tangent at a (the chord direction when it does not go on)
        bool openEnd = false; // the line has no continuation at all at a
    };
    // The circle through three grids of a line.
    struct Circle
    {
        double curvature = 0.0;
        Vector3 centre{};
        Vector3 binormal{}; // T × N
    };
    // The quadric fitted at a grid: its normal, the weight of the fit against the mesh lines, the variance of the
    // normal and that of the position of a grid (the residual of the fit, as a squared length).
    struct QuadricFit
    {
        Vector3 normal{};
        double weight = 0.0;
        double normalVariance = 0.0;
        double positionVariance = 0.0;
    };
    // A normal from the edges of a fan and its uncertainty (a squared angle).
    struct FanNormal
    {
        Vector3 normal{};
        double uncertainty = 0.0;
    };
    // A condition n · direction = value of a fan normal, from the edge to grid `grid`.
    struct NormalCondition
    {
        std::size_t grid = 0;
        Vector3 direction{};
        double value = 0.0;
    };
    // The normal, the fit weight, and the variances of the fits at one grid of an edge, over its elements.
    struct EdgeGridNormal
    {
        Vector3 normal{};
        double weight = 0.0;
        double normalVariance = 0.0;
        double positionVariance = 0.0;
    };

    [[nodiscard]] const Vector3& position(std::size_t grid) const { return m_input.grids[grid]; }
    [[nodiscard]] const ShellCorners& shell(std::size_t element) const { return m_input.shells[element]; }
    [[nodiscard]] const std::vector<std::size_t>& elementsOf(std::size_t a, std::size_t b) const;
    [[nodiscard]] bool isRaw(std::size_t a, std::size_t b) const { return m_topology.isRaw(a, b); }

    [[nodiscard]] std::size_t other(std::size_t e, std::size_t a, std::size_t b) const;
    [[nodiscard]] std::optional<std::size_t> facing(std::size_t e, std::size_t a, std::size_t d) const;
    [[nodiscard]] bool hasTriangle(std::size_t a) const;
    [[nodiscard]] bool triangleEdge(std::size_t a, std::size_t b) const;
    [[nodiscard]] std::pair<std::vector<RingStep>, bool> ring(std::size_t e, std::size_t a, std::size_t b);
    [[nodiscard]] const std::map<std::size_t, bool>& candidates(std::size_t a, std::size_t b);
    [[nodiscard]] std::map<std::size_t, bool> candidatesAt(std::size_t a, std::size_t b);
    // An edge at a mesh transition on a boundary that goes straight on (transitionEdge): the grid of the edge it goes
    // on into (nothing: the edge ends there), and whether it does so without crossing a raw edge.
    struct TransitionEdge
    {
        std::optional<std::size_t> goesOnTo;
        bool smooth = false;
    };
    [[nodiscard]] std::optional<TransitionEdge> transitionEdge(std::size_t e, std::size_t a, std::size_t b);
    [[nodiscard]] double cornerAngle(std::size_t e, std::size_t a) const;
    [[nodiscard]] bool informed(std::size_t a, std::size_t b);
    [[nodiscard]] std::optional<std::size_t> previous(std::size_t a, std::size_t b);
    [[nodiscard]] static std::optional<Circle> circle(const Vector3& x0, const Vector3& x1, const Vector3& x2);
    [[nodiscard]] static Vector3 curvatureVariation(const std::vector<Vector3>& line, std::size_t it,
                                                    const std::array<Vector3, 3>& base, double product);
    [[nodiscard]] static double circleSpread(const std::vector<Vector3>& line, std::size_t mid);
    [[nodiscard]] std::optional<Vector3> boundaryPlane(std::size_t a, std::size_t b, const Vector3& t);
    [[nodiscard]] std::vector<Vector3> chain(std::size_t a, std::size_t b, std::size_t count);
    [[nodiscard]] Continuation continuation(std::size_t a, std::size_t b);
    [[nodiscard]] Vector3 lineTangent(std::size_t a, std::size_t b);
    [[nodiscard]] double tangentUncertaintySquared(std::size_t a, std::size_t b) const;
    [[nodiscard]] std::vector<std::size_t> fan(std::size_t e, std::size_t g);
    [[nodiscard]] std::optional<FanNormal> fanNormal(std::size_t e, std::size_t a, std::size_t level);
    [[nodiscard]] std::optional<FanNormal> fanNormalAt(std::size_t e, std::size_t a, std::size_t level);
    [[nodiscard]] Vector3 facetNormal(std::size_t e, std::size_t a, std::size_t b);
    [[nodiscard]] std::optional<Vector3> farNormal(std::size_t e, std::size_t a, std::size_t b, std::size_t level);
    [[nodiscard]] std::optional<FanNormal> triangleNormal(std::size_t e, std::size_t a);
    [[nodiscard]] Vector3 lineNormal(std::size_t e, std::size_t a);
    [[nodiscard]] std::optional<QuadricFit> fit(std::size_t e, std::size_t g);
    [[nodiscard]] std::optional<QuadricFit> fitAt(std::size_t e, std::size_t g);
    [[nodiscard]] Vector3 normal(std::size_t e, std::size_t a);
    [[nodiscard]] EdgeGridNormal edgeGridNormal(const std::vector<std::size_t>& owners, const Vector3& frameNormal,
                                                std::size_t g);
    [[nodiscard]] Vector3 tangent(std::size_t a, std::size_t b);

    const MeshReadingInput& m_input;
    const Topology& m_topology;
    // The edges (with elements) at each grid, in edge order.
    std::map<std::size_t, std::vector<std::size_t>> m_nodeEdges;
    // The continuations of each edge a → b at a (candidates; the reference computes them on each query).
    std::map<GridPair, std::map<std::size_t, bool>> m_candidates;
    std::map<GridPair, Continuation> m_continuation;
    // The tangent at a of the curve of edge a → b.
    std::map<GridPair, Vector3> m_tangent;
    // The tangent at a of the mesh line of edge a → b, before the local fit.
    std::map<GridPair, Vector3> m_lineTangent;
    // The spread of the tangents of the lines that continue edge a → b at a.
    std::map<GridPair, double> m_branch;
    // The uncertainty of the line tangent: how much it was corrected for the variation of the curvature, how far its
    // circle is from the neighbouring ones, how much the lines differ where the line branches (zero on a circle, of
    // the order of the kinks on a line that zigzags).
    std::map<GridPair, double> m_tangentUncertainty;
    // (element, grid) → the quadric fitted on the neighbourhood of the grid.
    std::map<GridPair, std::optional<QuadricFit>> m_fits;
    // The variance of the tangent of the curve of edge a → b at a (a squared angle).
    std::map<GridPair, double> m_tangentVariance;
    // The normal at a of the edge a − b that is a plane section of the surface.
    std::map<GridPair, Vector3> m_section;
    // (element, grid, level) → fanNormal.
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, std::optional<FanNormal>> m_fanNormals;
};

// The bulge of the conic of chord u (unit), of length `length`, in the plane (u, direction), whose end tangents are
// u + sa·direction at the first grid and −u + sb·direction at the second (see the source).
// Reference: src/model.rs::end_slope_bulge.
[[nodiscard]] Vector3 endSlopeBulge(const Vector3& u, double length, const Vector3& direction, double sa, double sb,
                                    double variance);

} // namespace oos::elements::mixedHybrid::detail
