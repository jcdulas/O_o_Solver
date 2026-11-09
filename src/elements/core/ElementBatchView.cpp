#include "oos/elements/core/ElementBatchView.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <string>

namespace oos::elements
{

namespace
{

constexpr std::size_t ROTATION_TERMS = 9; // one 3 x 3 rotation per grid point, row-major

} // namespace

base::Vec3 ElementBatchView::positionInBasic(std::int32_t gridRow) const
{
    const std::span<const double> xyz = snapshot().nodes().positionsInBasic()[gridRow];
    return {xyz[0], xyz[1], xyz[2]};
}

base::Mat3 ElementBatchView::displacementFrame(std::int32_t gridRow) const
{
    const input::NodeTable& nodes = snapshot().nodes();
    if (!nodes.hasCdRotations())
    {
        const std::int32_t displacementFrameId = nodes.displacementFrameIds()[static_cast<std::size_t>(gridRow)];
        if (displacementFrameId == 0)
            return base::Mat3::identity();
        throw base::ContractError(base::concatenate(
            {"grid point ", std::to_string(nodes.ids()[static_cast<std::size_t>(gridRow)]),
             ": expected the rotation of its displacement system Node.displacementFrameId ",
             std::to_string(displacementFrameId),
             ", found none (the snapshot was taken without displacementRotations()); take the snapshot with "
             "the Node.displacementFrameId rotations"}));
    }
    base::Mat3 rotation;
    const std::span<const double> terms =
        nodes.cdRotations().subspan(static_cast<std::size_t>(gridRow) * ROTATION_TERMS, ROTATION_TERMS);
    for (std::size_t k = 0; k < ROTATION_TERMS; ++k)
        rotation.values[k] = terms[k];
    return rotation;
}

} // namespace oos::elements
