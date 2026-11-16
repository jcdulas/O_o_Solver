#pragma once

#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The supports of the shells read along their edges (document 25 of the reference, 6.2; document 34, section 5):
// what an edge holds, as components of the displacement and of the rotation in the basic system. A support is stated
// at grids; an external edge holds what both its grids hold. A model may instead state supports on edges, each
// holding its components along its edge and at its two grids.
//
// Thread-safety: plain values and pure functions.

// What a support holds along an edge of shells and at its two grids.
// Reference: src/model.rs::EdgeSupport.
struct EdgeSupport
{
    std::array<bool, 3> translations{};
    std::array<bool, 3> rotations{};

    // The displacement and the rotation held.
    [[nodiscard]] static constexpr EdgeSupport clamped() { return {{true, true, true}, {true, true, true}}; }
    // The displacement held, the rotation free: the simple support of a shear-deformable shell that leaves the edge
    // free to twist (soft).
    [[nodiscard]] static constexpr EdgeSupport pinned() { return {{true, true, true}, {false, false, false}}; }
    // The simple support of an edge whose direction across it, in the surface, is the basic axis `across`: the
    // displacement held and the rotation about `across` (the twist of the edge) held (hard).
    [[nodiscard]] static EdgeSupport simple(std::size_t across);
    // Symmetry about the plane through the edge normal to the basic axis `axis`: the displacement along `axis` and the
    // rotations about the two other axes held.
    [[nodiscard]] static EdgeSupport symmetry(std::size_t axis);
    // Antisymmetry about the plane through the edge normal to the basic axis `axis`, which is also a rigid diaphragm
    // in that plane: the displacement in the plane and the rotation about `axis` held.
    [[nodiscard]] static EdgeSupport antisymmetry(std::size_t axis);

    // The components held, 0 to 2 the translations and 3 to 5 the rotations, increasing.
    [[nodiscard]] std::vector<int> components() const;

    friend constexpr bool operator==(const EdgeSupport&, const EdgeSupport&) = default;
};

// Adds a support stated on the edge between grids a and b to the supports stated on edges: several supports on one
// edge add up.
// Reference: src/model.rs::Model::support_edge (the part on the edge; the grids hold components()).
void addEdgeSupport(std::map<EdgeKey, EdgeSupport>& edgeSupports, std::size_t a, std::size_t b,
                    const EdgeSupport& support);

// What is held along an edge of the shells: the support stated on it when the model states supports on edges
// (`statedOnEdges` not empty; an edge without one is free); otherwise, on an external edge (one shell) with
// `readGridSupportsAlongEdges`, the components held at both its grids, `heldAtLower` and `heldAtHigher` (0 to 2 the
// translations, 3 to 5 the rotations, in the basic system).
// Reference: src/analysis.rs::edge_held.
[[nodiscard]] EdgeSupport edgeHeld(const std::map<EdgeKey, EdgeSupport>& statedOnEdges, const EdgeKey& edge,
                                   std::size_t shellCount, const std::array<bool, 6>& heldAtLower,
                                   const std::array<bool, 6>& heldAtHigher, bool readGridSupportsAlongEdges);

} // namespace oos::elements::mixedHybrid
