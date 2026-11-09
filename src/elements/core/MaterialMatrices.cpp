#include "oos/elements/core/MaterialMatrices.hpp"

#include <cmath>

namespace oos::elements
{

base::Mat3 planeStressMatrix(double youngModulus, double poissonRatio)
{
    const double scale = youngModulus / (1.0 - poissonRatio * poissonRatio);
    return base::matrixFromRows({scale, scale * poissonRatio, 0.0}, {scale * poissonRatio, scale, 0.0},
                                {0.0, 0.0, scale * (1.0 - poissonRatio) / 2.0});
}

base::Mat3 strainRotation(double angleInRadians)
{
    const double c = std::cos(angleInRadians);
    const double s = std::sin(angleInRadians);
    return base::matrixFromRows({c * c, s * s, c * s}, {s * s, c * c, -c * s},
                                {-2.0 * c * s, 2.0 * c * s, c * c - s * s});
}

base::Mat3 rotateToElementFrame(const base::Mat3& materialMatrix, double angleInRadians)
{
    return base::congruence(materialMatrix, strainRotation(angleInRadians));
}

Mat6 isotropicSolidMatrix(double youngModulus, double poissonRatio)
{
    const double lambda = youngModulus * poissonRatio / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));
    const double mu = youngModulus / (2.0 * (1.0 + poissonRatio));
    Mat6 d;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            d(i, j) = lambda;
        d(i, i) = lambda + 2.0 * mu;
        d(i + 3, i + 3) = mu;
    }
    return d;
}

Mat6 solidStrainTransformation(const base::Mat3& axes)
{
    // The Voigt pairs (i, j) of the six components; a shear component is γ_ij = ε_ij + ε_ji.
    constexpr std::array<std::array<std::size_t, 2>, 6> PAIRS{{{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}}};
    Mat6 t;
    for (std::size_t row = 0; row < 6; ++row)
    {
        const std::size_t i = PAIRS[row][0];
        const std::size_t j = PAIRS[row][1];
        const bool isShearRow = i != j;
        for (std::size_t col = 0; col < 6; ++col)
        {
            const std::size_t k = PAIRS[col][0];
            const std::size_t l = PAIRS[col][1];
            // ε_local,ij = Σ_kl a_ki a_lj ε_kl: a global normal strain enters once, a global shear γ_kl = 2 ε_kl
            // through the terms (k, l) and (l, k), hence half their sum; a local shear row is γ_ij = 2 ε_ij.
            double term = axes(k, i) * axes(l, j);
            if (k != l)
                term = 0.5 * (term + axes(l, i) * axes(k, j));
            t(row, col) = (isShearRow ? 2.0 : 1.0) * term;
        }
    }
    return t;
}

} // namespace oos::elements
