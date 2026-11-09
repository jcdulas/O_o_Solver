#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/input/tables/FastenerPiercingTable.hpp"
#include "oos/reduction/absorption/Spider.hpp"
#include "oos/reduction/absorption/SpiderWeights.hpp"

#include <optional>
#include <string>
#include <variant>

namespace oos::reduction
{

// A pierced side of a Fastener or Weld as a spider (fastener sides are absorbed like spiders): on each side the
// fastener ties its point P (GA or GB, given or generated) to the grid points of the shell element it pierces by the
// interpolation of elements::interpolatePiercing (an RigidInterpolation weighted by the shape functions at P, through
// elements::interpolateSide),
//     (u_P, θ_P) = M·u_grids,
// M the 6 × 6g map of the side (every form of the card; O_o_Reader's piercing rows, through
// input::FastenerPiercingTable), u_grids the DOFs of the g grid points of the pierced element. That is a spider whose
// legs are the grid points of the pierced element and whose weighting matrix is W = M: the side is absorbed into
// its unit by the same change of basis (SpiderChangeOfBasis), only P stays on the boundary, and the fastener
// stiffness between P_A and P_B stays a connection. The rows of W are the motion of P in the basic system
// (translations, then rotations); its columns the DOFs of the pierced element's grid points M uses (the translations
// of the grid points only, or the six DOFs of a grid point connected directly), in increasing order; the
// characteristic length is the mean distance of the grid points of the pierced element to P.
//
// When the card gives GA (or GB) as a grid point of the model, the point's own DOFs follow the pierced element by
// u_GA = A·M·u_grids (assembly::FastenerConstraints, A the rigid offset from P to GA in the displacement system of GA,
// invertible): the condensation takes those rows of Gm as W, which has the conditioning of M up to A.
//
// Thread-safety: stateless functions. Ownership: none.
class FastenerPiercing
{
public:
    // The relation of side `side` of fastener `elementId`, or the message of a problem (no rows for the side, a grid
    // point of the pierced element the model does not define, shape functions that are not an interpolation:
    // elements::interpolatePiercing).
    [[nodiscard]] static std::variant<SpiderRelation, std::string> relation(
        const input::FastenerPiercingTable& piercings, const assembly::DofMap& dofMap, base::EntityId elementId,
        FastenerSide side);
};

} // namespace oos::reduction
