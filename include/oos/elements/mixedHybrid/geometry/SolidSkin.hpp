#pragma once

#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"
#include "oos/elements/mixedHybrid/geometry/Patch.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The reading of the skin of the solids (document 28 of the reference, 3.1; document 35, 3.4, rule 4). The faces that
// one solid element only has are the triangles and quadrilaterals of a surface, read as the surface of the shells is
// (Topology): every edge follows its mesh line, the faces are rational patches fitted on the normals. Across an edge
// declared sharp the skin is not smooth: each smooth part is read on its own, the sharp edge being a boundary line of
// the parts it separates, and its conic is the mean of their readings. The boundary of a smooth part goes on through
// a grid that has two sharp edges and turns a corner where more meet (MeshReadingInput::throughGrids).
//
// What follows the reading (where a shell lies on the skin, the projection of the curves on the functions of the
// edges and faces of the solids) belongs to the solids.
//
// Invariants: the faces of the skin turn about their outward normal. Thread-safety: plain values and pure functions.

// A solid element as its corner grids and its faces, each a list of corner indices (local, 0-based) in the order of
// the shape.
struct SolidCell
{
    std::vector<std::size_t> corners;
    std::vector<std::vector<std::size_t>> faces;
};

// A face of the skin: its grids (3 or 4) turning about the outward normal, and where it comes from.
struct SkinFace
{
    std::array<std::size_t, 4> grids{};
    std::size_t count = 3;
    std::size_t solid = 0;
    std::size_t localFace = 0;

    [[nodiscard]] std::span<const std::size_t> view() const { return {grids.data(), count}; }
};

// The faces that one solid only has, solids and faces in order, each turned so that its grids turn about the outward
// normal: (x1 − x0) × (x2 − x0) points away from the centroid of the corners of its solid.
// Reference: src/tdnns/solid.rs::read_skin (the faces of the skin), as src/analysis.rs::solid_skin for the tetrahedra.
[[nodiscard]] std::vector<SkinFace> skinFaces(std::span<const Vector3> grids, std::span<const SolidCell> solids);

// The conic of an edge of the skin: its bulge and its weight (the mean of its readings).
struct SkinConic
{
    Vector3 bulge{};
    double weight = 0.0;
};

// The skin read: the shell mesh of its smooth parts (one grid per grid and part; element i is face i of the input),
// the grid of the model of each of its grids, its topology, and the conic of each edge of the skin by the grids of
// the model.
struct SkinReading
{
    MeshReadingInput mesh;
    std::vector<std::size_t> modelGrid;
    Topology topology;
    std::map<EdgeKey, SkinConic> conics;
};

// Reads the skin made of `faces` (skinFaces), with the sharp edges `sharpEdges` (by the grids of the model).
// Reference: src/analysis.rs::solid_skin (its reading: the smooth parts, their topology, the conics of the edges).
[[nodiscard]] SkinReading readSolidSkin(std::span<const Vector3> grids, std::span<const SkinFace> faces,
                                        const std::set<EdgeKey>& sharpEdges);

// The patch of face `face` of a skin read: the rational triangle with its interior fitted outside the rational form
// (TrianglePatch::fitInterior), or the rational quadrilateral with its centre fitted, on the grids of the face in its
// order and the bulges and weights of its edges.
// Reference: src/tdnns/solid.rs::read_skin (the patch of a face), as src/analysis.rs::solid_skin for the triangles.
[[nodiscard]] Patch skinFacePatch(const SkinReading& reading, std::size_t face);

} // namespace oos::elements::mixedHybrid
