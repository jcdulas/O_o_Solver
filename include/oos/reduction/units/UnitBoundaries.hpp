#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/reduction/units/ClusterAbsorption.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::reduction
{

class SpiderAbsorption;

// Why a point is on a unit's boundary (bit flags of BoundaryNode::reasons).
enum class BoundaryReason : std::uint8_t
{
    Interface = 1,         // a connection of the connection system uses it
    SharedPoint = 2,       // another unit holds it too (an element list sharing a point with its neighbour)
    Retained = 4,          // the caller retains it (a measurement point, a design response)
    PartialConstraint = 8, // one of its DOFs is constrained in some subcases only
    Support = 16,          // a FreeBodySupport or SUPORT1 point (inertia relief is solved on the interface problem)
    AbsorbedSpider = 32    // the reference point of a spider absorbed into the unit, or the pierced element's grid
                           // point standing for the point of an absorbed fastener side
                           // (reduction/absorption/SpiderAbsorption)
};

struct BoundaryNode
{
    base::EntityId nodeId = 0;
    std::uint8_t reasons = 0; // BoundaryReason bits

    [[nodiscard]] bool has(BoundaryReason reason) const noexcept
    {
        return (reasons & static_cast<std::uint8_t>(reason)) != 0;
    }
};

// "interface, retained" for the report.
[[nodiscard]] std::string describeReasons(std::uint8_t reasons);

// What the analysis adds to the boundaries; every list sorted.
struct BoundaryInputs
{
    std::vector<base::EntityId> retainedNodes;
    // The points with a DOF constrained in some subcases only ("a DOF constrained in some subcases only
    // is a boundary DOF, its constraint being applied to the interface problem of those subcases, so that one
    // condensation serves every subcase").
    std::vector<base::EntityId> partiallyConstrainedNodes;
    std::vector<base::EntityId> supportNodes;
};

// The boundary of every unit after absorption: its interface points (the
// points a connection of the connection system uses), the points it shares with another unit, the points the caller
// retains, the points with a DOF constrained in some subcases only and the FreeBodySupport points. A point only
// constrained in every subcase or only requested for output is never on it; neither is a connection-only point a
// cluster absorption made interior, unless retained. With spider absorption, the legs of an absorbed spider are no
// longer used by the connection system (SpiderAbsorption::removedUses) and its reference point joins the unit's points
// as a boundary point (AbsorbedSpider).
//
// Granularity: the point. The interface problem is assembled at node granularity (6 x 6 blocks), so
// a boundary point brings all its free DOFs (those not constrained in every subcase and not dependent), the DOF
// constrained in some subcases among them.
//
// Thread-safety: immutable once built. Ownership: owns its lists.
class UnitBoundaries
{
public:
    // spiders: the spider absorption made on `absorption` (null: none).
    UnitBoundaries(const ClusterAbsorption& absorption, const BoundaryInputs& inputs,
                   const SpiderAbsorption* spiders = nullptr);

    // The boundary of the unit at an index of absorption.units(), sorted by point.
    [[nodiscard]] std::span<const BoundaryNode> at(std::size_t unitIndex) const noexcept
    {
        return m_boundaries[unitIndex];
    }
    [[nodiscard]] std::size_t size() const noexcept { return m_boundaries.size(); }

private:
    std::vector<std::vector<BoundaryNode>> m_boundaries;
};

} // namespace oos::reduction
