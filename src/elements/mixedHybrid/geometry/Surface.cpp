#include "oos/elements/mixedHybrid/geometry/Surface.hpp"

#include <cmath>

namespace oos::elements::mixedHybrid
{

std::array<Vector3, 2> tangentFrame(const Vector3& normal, const Vector3& reference)
{
    const double p = dot3(reference, normal);
    const Vector3 e1 = unit3(sub3(reference, scl3(normal, p)));
    return {e1, cross3(normal, e1)};
}

PhysicalGeometry physicalGeometry(const SurfacePointGeometry& g, const Vector3& reference)
{
    return physicalGeometryInFrame(g, tangentFrame(g.normal, reference));
}

PhysicalGeometry physicalGeometryXy(const SurfacePointGeometry& g, const Vector3& x, const Vector3& y)
{
    return physicalGeometryInFrame(g, tangentFrameXy(g.normal, x, y));
}

PhysicalGeometry physicalGeometryInFrame(const SurfacePointGeometry& g, const std::array<Vector3, 2>& axes)
{
    PhysicalGeometry out;
    out.frame = axes;
    out.components = {{{dot3(axes[0], g.tangents[0]), dot3(axes[0], g.tangents[1])},
                       {dot3(axes[1], g.tangents[0]), dot3(axes[1], g.tangents[1])}}};
    const auto& cm = out.components;
    const double det = cm[0][0] * cm[1][1] - cm[0][1] * cm[1][0];
    out.inverse = {{{cm[1][1] / det, -cm[0][1] / det}, {-cm[1][0] / det, cm[0][0] / det}}};
    for (std::size_t j = 0; j < 2; ++j)
        for (std::size_t i = 0; i < 3; ++i)
            out.normalGradient[j][i] =
                g.normalDerivatives[0][i] * out.inverse[0][j] + g.normalDerivatives[1][i] * out.inverse[1][j];
    return out;
}

double controlPolygonWeight(const Vector3& chord, const Vector3& bulge)
{
    const Vector3 control = scl3(bulge, 2.0);
    const Vector3 a1 = add3(scl3(chord, 0.5), control);
    const Vector3 a2 = sub3(scl3(chord, 0.5), control);
    const double cosGamma = -dot3(a1, a2) / (norm3(a1) * norm3(a2));
    // fmax as the reference's max: a NaN gives 0
    return std::sqrt(std::fmax(0.5 * (1.0 - cosGamma), 0.0));
}

} // namespace oos::elements::mixedHybrid
