#pragma once

// What the checks of the ShellQuad4 and ShellTria3 kernels share: the isotropic section of the formulation document
// (docs/shell-quad4-formulation.md, section 14), the eigenvalues of an element matrix, the DOFs of a rigid motion and
// the passage of an element matrix to the basic axes.

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/shells/PlateConstitutive.hpp"
#include "oos/elements/shells/ShellScalar.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace oos::checks::shells
{

inline constexpr double YOUNG = 1000.0;
inline constexpr double POISSON = 0.3;
inline constexpr double THICKNESS = 0.05;
inline constexpr double SHEAR_FACTOR = 5.0 / 6.0;

// The isotropic section of the document: the moduli of the membrane and of the bending, and κ G on the transverse
// shears (none for a section without transverse shear flexibility).
[[nodiscard]] inline elements::PlateConstitutive<double> isotropicSection(bool hasShearFlexibility,
                                                                          double poisson = POISSON)
{
    elements::PlateConstitutive<double> c;
    c.hasMembrane = true;
    c.hasBending = true;
    c.hasShearFlexibility = hasShearFlexibility;
    c.bendingRatio = 1.0;
    c.shearRatio = SHEAR_FACTOR;
    const double plane = YOUNG / (1.0 - poisson * poisson);
    for (elements::Matrix3<double>* modulus : {&c.membrane, &c.bending})
    {
        (*modulus)(0, 0) = plane;
        (*modulus)(1, 1) = plane;
        (*modulus)(0, 1) = poisson * plane;
        (*modulus)(1, 0) = poisson * plane;
        (*modulus)(2, 2) = 0.5 * (1.0 - poisson) * plane;
    }
    if (hasShearFlexibility)
    {
        const double shear = SHEAR_FACTOR * YOUNG / (2.0 * (1.0 + poisson));
        c.transverseShear(0, 0) = shear;
        c.transverseShear(1, 1) = shear;
    }
    return c;
}

template <std::size_t N>
[[nodiscard]] std::vector<double> sortedEigenvalues(const elements::ScalarMatrix<double, N>& k)
{
    elements::mixedHybrid::Matrix matrix(N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            matrix(i, j) = k(i, j);
    const elements::mixedHybrid::SymmetricEigen eigen = elements::mixedHybrid::jacobiEigen(matrix);
    std::vector<double> values(eigen.values.begin(), eigen.values.end());
    std::sort(values.begin(), values.end());
    return values;
}

// The local DOFs of a rigid motion of the nodes at `position` (element frame): the translation `shift` and the small
// rotation `turn` about the origin.
template <std::size_t NODES>
[[nodiscard]] std::array<double, 6 * NODES> rigidMotion(const std::array<std::array<double, 3>, NODES>& position,
                                                        const std::array<double, 3>& shift,
                                                        const std::array<double, 3>& turn)
{
    std::array<double, 6 * NODES> q{};
    for (std::size_t a = 0; a < NODES; ++a)
    {
        const std::array<double, 3>& x = position[a];
        q[6 * a + 0] = shift[0] + turn[1] * x[2] - turn[2] * x[1];
        q[6 * a + 1] = shift[1] + turn[2] * x[0] - turn[0] * x[2];
        q[6 * a + 2] = shift[2] + turn[0] * x[1] - turn[1] * x[0];
        for (std::size_t k = 0; k < 3; ++k)
            q[6 * a + 3 + k] = turn[k];
    }
    return q;
}

// Adds the element matrix `k` (element frame, `axes` its columns in basic) to the matrix of a model in basic axes,
// node a of the element being grid point `nodes[a]` (six DOFs each): K += Rᵀ k R, R the basic-to-element rotation of
// every triplet.
template <std::size_t N>
void addToModel(elements::mixedHybrid::Matrix& model, const elements::ScalarMatrix<double, N>& k,
                const base::Mat3& axes, const std::array<std::size_t, N / 6>& nodes)
{
    const auto dofOf = [&](std::size_t local) { return 6 * nodes[local / 6] + local % 6; };
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
            if (k(i, j) == 0.0)
                continue;
            // local DOF i = Σ_r axes(r, i % 3) · basic DOF (triplet of i, component r)
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t s = 0; s < 3; ++s)
                    model(dofOf(i - i % 3 + r), dofOf(j - j % 3 + s)) += axes(r, i % 3) * k(i, j) * axes(s, j % 3);
        }
}

} // namespace oos::checks::shells
