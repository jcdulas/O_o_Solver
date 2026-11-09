#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/reduction/absorption/Spider.hpp"
#include "oos/reduction/absorption/SpiderWeights.hpp"

#include <cstdint>
#include <vector>

namespace oos::reduction
{

// A spider absorbed into its unit by change of basis (never by a flexibility form or Lagrange multipliers): an
// RigidInterpolation, or a pierced side of a Fastener or Weld, with its relation u_ref = W·u_legs.
// - A RigidInterpolation: W from its rows of the Gm of the unit (SpiderWeights::rigidInterpolationRelation);
//   referenceDofs the REFC DOFs of REFGRID, dependent in that Gm.
// - A fastener side whose GA (GB) is a grid point of the model: W from that point's rows of the Gm of the unit
//   (assembly::FastenerConstraints, u_GA = A·M·u_grids); referenceDofs its six DOFs.
// - A fastener side whose point is generated (the generated GA/GB stands as a virtual boundary point): W = M
//   (FastenerPiercing::relation), referenceDofs empty; the unit gets a point of its own for it,
//   six DOFs in the basic system at the piercing point.
struct UnitSpider
{
    SpiderKey key;
    SpiderRelation relation;
    base::Vec3 generatedPointPosition{}; // the piercing point (basic system) of a generated point
};

// What a unit is made of, for its condensation (from its plan, reduction::UnitPlan, and the relations of its absorbed
// spiders). Every list sorted.
struct UnitDescription
{
    std::int64_t unitId = 0;
    // Its points: its own, the connection-only points its absorbed clusters made interior, the references of its
    // absorbed spiders.
    std::vector<base::EntityId> nodeIds;
    // Its elements and the connection elements it absorbed. An ID no element batch holds (a rigid element: a relation,
    // in the Gm of the units) is skipped.
    std::vector<base::EntityId> elementIds;
    // Its boundary points, among nodeIds: their DOFs (those not held in every subcase, not dependent inside the unit)
    // are kept in K*. A generated fastener point is on the boundary without being listed.
    std::vector<base::EntityId> boundaryNodeIds;
    // The spiders absorbed by change of basis (the RigidBodyStar absorbed into the unit need none: their own
    // elimination, rows of the Gm of the units, makes their legs dependent and GN stands on the boundary).
    std::vector<UnitSpider> spiders;
};

} // namespace oos::reduction
