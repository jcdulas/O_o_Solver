#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"

namespace oos::elements
{

// The rigid offset of an element's reference point from its grid point (shell ZOFFS, bar and beam offsets): the six
// DOFs at the offset point, d away from the grid point, follow the grid point as a rigid body,
// u_offset = u + θ × d, θ_offset = θ, i.e. [u; θ]_offset = O·[u; θ] with O = [[I, −[d]×], [0, I]]
// ([d]× the cross-product matrix of d). d and the DOFs are in one frame (the element frame for a shell).
[[nodiscard]] base::MatN<6> rigidOffset(base::Vec3 offset);

} // namespace oos::elements
