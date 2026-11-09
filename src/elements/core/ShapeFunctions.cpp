#include "oos/elements/core/ShapeFunctions.hpp"

namespace oos::elements
{

BilinearShape bilinearShapeAt(double xi, double eta)
{
    BilinearShape shape;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const double xiI = QUAD_CORNER_XI[i];
        const double etaI = QUAD_CORNER_ETA[i];
        shape.n[i] = 0.25 * (1.0 + xiI * xi) * (1.0 + etaI * eta);
        shape.dXi[i] = 0.25 * xiI * (1.0 + etaI * eta);
        shape.dEta[i] = 0.25 * etaI * (1.0 + xiI * xi);
    }
    return shape;
}

PlaneJacobian planeJacobianAt(const BilinearShape& shape, const std::array<double, 4>& x,
                              const std::array<double, 4>& y)
{
    PlaneJacobian jacobian;
    for (std::size_t i = 0; i < 4; ++i)
    {
        jacobian.j(0, 0) += shape.dXi[i] * x[i];
        jacobian.j(0, 1) += shape.dXi[i] * y[i];
        jacobian.j(1, 0) += shape.dEta[i] * x[i];
        jacobian.j(1, 1) += shape.dEta[i] * y[i];
    }
    const base::MatN<2>& j = jacobian.j;
    jacobian.determinant = j(0, 0) * j(1, 1) - j(0, 1) * j(1, 0);
    const double scale = 1.0 / jacobian.determinant;
    jacobian.inverse(0, 0) = j(1, 1) * scale;
    jacobian.inverse(0, 1) = -j(0, 1) * scale;
    jacobian.inverse(1, 0) = -j(1, 0) * scale;
    jacobian.inverse(1, 1) = j(0, 0) * scale;
    for (std::size_t i = 0; i < 4; ++i)
    {
        jacobian.dX[i] = jacobian.inverse(0, 0) * shape.dXi[i] + jacobian.inverse(0, 1) * shape.dEta[i];
        jacobian.dY[i] = jacobian.inverse(1, 0) * shape.dXi[i] + jacobian.inverse(1, 1) * shape.dEta[i];
    }
    return jacobian;
}

} // namespace oos::elements
