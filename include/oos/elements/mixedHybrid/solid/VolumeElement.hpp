#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// A solid element of the mixed hybrid formulation as the model gives it: the numbers (IDs) of its corner grid points
// in the order of the input format (VolumeShape.hpp), the numbers of its mid-side nodes in the order of the cards, and
// the index of its material. The shape follows from the number of corners. Any orientation of the corners is accepted
// (the measure of the element is the absolute value of its Jacobian); an element that folds is refused by its kernel.
//
// The numbers of the grid points orient the edges and faces of the element: the functions of an edge or a face are
// written from them (VolumeShape.hpp), so that the elements that share it see the same functions. They are the `order`
// of elementFunctions, VolumeGeometry::straight and VolumeKernel::build.
//
// Thread-safety: a plain value type.

// No mid-side node on that edge: the ID 0 of a blank field of the cards.
inline constexpr base::EntityId NO_MID_SIDE_NODE = 0;

// Reference: volume::VolumeElement
struct VolumeElement
{
    // `corners`: SolidTetra G1-G4; SolidPyramid G1-G4 the base, G5 the apex; SolidPenta G1-G3 one triangle, G4-G6 the
    // other (G4 opposite G1); SolidHexa G1-G4 one face, G5-G8 the opposite one (G5 opposite G1).
    std::vector<base::EntityId> corners;
    // `mid`: one per edge of the shape in the order of shapeEdges (SolidTetra G5-G10, SolidPyramid G6-G13, SolidPenta
    // G7-G15, SolidHexa G9-G20), NO_MID_SIDE_NODE where an edge has none; empty when the element has none at all.
    std::vector<base::EntityId> midSideNodes;
    // `prop`: the index of the material of the element.
    std::size_t property = 0;

    // Reference: volume::VolumeElement::new (and hexa, penta, pyramid, tetra)
    [[nodiscard]] static VolumeElement fromCorners(std::span<const base::EntityId> cornerIds, std::size_t propertyIndex)
    {
        return {{cornerIds.begin(), cornerIds.end()}, {}, propertyIndex};
    }

    // The same element with its mid-side nodes, in the order of the cards.
    // Reference: volume::VolumeElement::with_mid
    [[nodiscard]] VolumeElement withMidSideNodes(std::span<const base::EntityId> midSideIds) const
    {
        VolumeElement element = *this;
        element.midSideNodes.assign(midSideIds.begin(), midSideIds.end());
        return element;
    }

    // The shape of its corners; empty for a corner count that is no shape.
    // Reference: volume::VolumeElement::shape
    [[nodiscard]] std::optional<VolumeShape> shape() const { return volumeShapeOfCornerCount(corners.size()); }

    // The mid-side node of edge `edge` of its shape, if it has one.
    // Reference: volume::VolumeElement::mid_of
    [[nodiscard]] std::optional<base::EntityId> midSideNodeOf(std::size_t edge) const
    {
        if (edge >= midSideNodes.size() || midSideNodes[edge] == NO_MID_SIDE_NODE)
            return std::nullopt;
        return midSideNodes[edge];
    }
};

} // namespace oos::elements::mixedHybrid
