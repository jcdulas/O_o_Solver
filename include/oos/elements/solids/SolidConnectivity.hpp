#pragma once

#include "oos/base/arrow/ValidityView.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"
#include "oos/input/tables/ConnectivityView.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace oos::elements
{

// The grid points of a solid element: its shape (with or without its mid-side nodes), and for each node in the order
// of the card its ID, its row in the node table, its position in the basic system and the rotation of its
// displacement system CD to the basic system.
struct SolidNodes
{
    SolidShape shape = SolidShape::Hexa8;
    std::size_t count = 0;
    std::array<std::int32_t, MAX_SOLID_NODES> ids{};
    std::array<std::int32_t, MAX_SOLID_NODES> rows{};
    std::array<base::Vec3, MAX_SOLID_NODES> positions{};
    std::array<base::Mat3, MAX_SOLID_NODES> cdFrames{};
};

// The two shapes of one solid card: without and with its mid-side nodes (CHEXA: Hexa8, Hexa20).
struct SolidCardShapes
{
    SolidShape linear = SolidShape::Hexa8;
    SolidShape quadratic = SolidShape::Hexa20;
};

// The node columns of a solid batch (nodes and nodesRows of model.query("SolidHexa"): a fixed list of 20, 15, 13 or 10
// entries per element, the absent mid-side nodes given as ID 0 and row −1), read once per facet call.
struct SolidConnectivity
{
    input::ConnectivityView rows;
    input::ConnectivityView ids;
    base::ValidityView valid;

    explicit SolidConnectivity(const ElementBatchView& view);
};

// The grid points of the element of batch row `row`. The corners must be distinct grid points of the model; the
// mid-side nodes are either all present or all absent (the input format lets a SolidHexa, SolidPenta, SolidTetra or
// SolidPyramid leave out any of them, but the shape functions of an element with only some mid-side nodes are not
// implemented: such an element is refused, problem unsupported-field on the first absent node).
[[nodiscard]] Resolution<SolidNodes> resolveSolidNodes(const ElementBatchView& view,
                                                       const SolidConnectivity& connectivity, std::int64_t row,
                                                       SolidCardShapes shapes);

} // namespace oos::elements
