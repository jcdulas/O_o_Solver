#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace oos::elements
{

// The equivalent nodal loads of a pressure on one face of a solid: the nodes of the face (indices into the element's
// node list) and the force on each, in the basic system.
struct SolidFaceLoads
{
    std::size_t count = 0;
    std::array<std::size_t, 8> nodes{};
    std::array<base::Vec3, 8> forces{};
};

// The face of a solid whose corners are the grid points `faceCorners` (their IDs, 3 or 4), with the index of each in
// the element and the mid-side nodes of its edges; nothing when the element has no such face.
struct SolidFaceNodes
{
    std::size_t cornerCount = 0;
    std::size_t count = 0;              // corners, then the mid-side nodes of the edges c0-c1, c1-c2, ...
    std::array<std::size_t, 8> nodes{}; // indices into the element's nodes, corners in the order of faceCorners
};
[[nodiscard]] std::optional<SolidFaceNodes> findSolidFace(SolidShape shape, std::span<const std::int32_t> elementIds,
                                                          std::span<const std::int32_t> faceCorners);

// The consistent nodal loads F_i = −∫ N_i p n dA of a pressure on a face (PLOAD4: a positive pressure on a solid
// face acts into the element, along −n with n the outward normal: P = 1 on the face z = 0 of a SolidTetra lying in z > 0
// pushes it along +z). The pressure varies over
// the face with the linear (triangle) or bilinear (quadrilateral) functions of its corners, `cornerPressures[k]` at
// the k-th corner of `face` (P1-P4 of the card at the corners as pressureTable gives them); N_i are the face's own
// shape functions (3, 4, 6 or 8 nodes). Integration: 3 × 3 Gauss points on a quadrilateral, a degree-4 rule on a
// triangle. `positions` are the element's node positions in the basic system; the outward side is the one away from
// the element's centroid.
[[nodiscard]] SolidFaceLoads solidPressureLoads(SolidShape shape, std::span<const base::Vec3> positions,
                                                const SolidFaceNodes& face,
                                                const std::array<double, 4>& cornerPressures);

} // namespace oos::elements
