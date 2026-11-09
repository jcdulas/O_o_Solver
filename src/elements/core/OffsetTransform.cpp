#include "oos/elements/core/OffsetTransform.hpp"

#include "oos/elements/core/MatrixKernels.hpp"

namespace oos::elements
{

base::MatN<6> rigidOffset(base::Vec3 offset)
{
    base::MatN<6> o = base::MatN<6>::identity();
    // θ × d = −d × θ = −[d]×·θ.
    const base::Mat3 cross = crossProductMatrix(offset);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            o(i, 3 + j) = -cross(i, j);
    return o;
}

} // namespace oos::elements
