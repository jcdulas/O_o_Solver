#include "oos/elements/solids/SolidKinematics.hpp"

#include "oos/elements/core/MatrixKernels.hpp"

#include <algorithm>

namespace oos::elements
{

void SolidMatrix::reset(std::size_t newSize)
{
    size = newSize;
    std::fill_n(values.begin(), newSize * newSize, 0.0);
}

void SolidOperator::reset(std::size_t newColumns)
{
    columns = newColumns;
    std::fill_n(values.begin(), 6 * newColumns, 0.0);
}

Voigt6 SolidOperator::apply(std::span<const double> u) const
{
    Voigt6 result{};
    for (std::size_t i = 0; i < 6; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < columns; ++j)
            sum += (*this)(i, j) * u[j];
        result[i] = sum;
    }
    return result;
}

ShapeGradients shapeGradientsAt(SolidShape shape, std::span<const base::Vec3> positions, ParentPoint point)
{
    ShapeGradients g;
    g.shape = solidShapeAt(shape, point);
    const SolidShapeValues& s = g.shape;
    for (std::size_t n = 0; n < s.count; ++n)
        for (std::size_t k = 0; k < 3; ++k)
        {
            g.jacobian(0, k) += s.dXi[n] * positions[n][k];
            g.jacobian(1, k) += s.dEta[n] * positions[n][k];
            g.jacobian(2, k) += s.dZeta[n] * positions[n][k];
        }
    g.determinant = base::determinant(g.jacobian);
    if (g.determinant == 0.0)
        return g;
    const base::Mat3 inverse = base::inverse(g.jacobian);
    for (std::size_t n = 0; n < s.count; ++n)
        g.gradients[n] = inverse * base::Vec3{s.dXi[n], s.dEta[n], s.dZeta[n]};
    return g;
}

void fillStrainOperator(const ShapeGradients& gradients, SolidOperator& b)
{
    const std::size_t count = gradients.shape.count;
    b.reset(3 * count);
    for (std::size_t n = 0; n < count; ++n)
    {
        const base::Vec3& d = gradients.gradients[n];
        const std::size_t u = 3 * n;
        b(0, u) = d[0];
        b(1, u + 1) = d[1];
        b(2, u + 2) = d[2];
        b(3, u) = d[1];
        b(3, u + 1) = d[0];
        b(4, u + 1) = d[2];
        b(4, u + 2) = d[1];
        b(5, u) = d[2];
        b(5, u + 2) = d[0];
    }
}

Voigt6 voigtStrainOf(const base::Mat3& g)
{
    return {g(0, 0), g(1, 1), g(2, 2), g(0, 1) + g(1, 0), g(1, 2) + g(2, 1), g(0, 2) + g(2, 0)};
}

Mat6 complianceOf(const Mat6& elasticity)
{
    return inverted(elasticity);
}

} // namespace oos::elements
