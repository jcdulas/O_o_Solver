#pragma once

#include "oos/elements/bars_beams/BeamMass.hpp"

namespace oos::elements::beam_mass
{
// Gauss-Legendre rule of 4 points on [−1, 1] (exact to degree 7).
constexpr std::array<double, 4> GAUSS_NODES{-0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
                                            0.8611363115940526};
constexpr std::array<double, 4> GAUSS_WEIGHTS{0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
                                              0.3478548451374538};

// N(ξ), the displacements (u, v, w) of the axis at ξ = x/L from the end DOFs: linear on u, Hermite cubics on (v, θz)
// and (w, −θy) (θy = −w', θz = v').
inline base::MatN<3, 12> shapeFunctions(double xi, double length)
{
    const double h1 = 1.0 - 3.0 * xi * xi + 2.0 * xi * xi * xi;
    const double h2 = length * (xi - 2.0 * xi * xi + xi * xi * xi);
    const double h3 = 3.0 * xi * xi - 2.0 * xi * xi * xi;
    const double h4 = length * (-xi * xi + xi * xi * xi);
    base::MatN<3, 12> n;
    n(0, 0) = 1.0 - xi;
    n(0, 6) = xi;
    n(1, 1) = h1;
    n(1, 5) = h2;
    n(1, 7) = h3;
    n(1, 11) = h4;
    n(2, 2) = h1;
    n(2, 4) = -h2;
    n(2, 8) = h3;
    n(2, 10) = -h4;
    return n;
}

// Integrate material and added mass at their own stations. The same interpolation also carries sizing derivatives.
template <typename Matrix, typename Scalar>
void addStream(Matrix& mass, const Scalar& weight, const std::array<Scalar, 2>& offset, double xi, double length,
               MassFormulation formulation)
{
    const std::array<double, 2> linear{1.0 - xi, xi};
    const bool lumped = formulation == MassFormulation::Lumped;
    const auto shape = shapeFunctions(xi, length);
    const auto& [y, z] = offset;
    for (std::size_t block = 0; block < (lumped ? 2u : 1u); ++block)
    {
        std::array<std::array<Scalar, 12>, 3> n{};
        for (std::size_t end = 0; end < 2; ++end)
        {
            if (lumped && end != block)
                continue;
            const auto at = end * 6;
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 6; ++c)
                    n[r][at + c] = lumped ? double(r == c) : shape(r, at + c);
            const double spin = lumped ? 1.0 : linear[end];
            n[0][at + 4] += spin * z;
            n[0][at + 5] -= spin * y;
            n[1][at + 3] -= spin * z;
            n[2][at + 3] += spin * y;
        }
        const Scalar w = weight * (lumped ? linear[block] : 1.0);
        const std::size_t begin = lumped ? 6 * block : 0;
        const std::size_t stop = lumped ? begin + 6 : 12;
        for (std::size_t i = begin; i < stop; ++i)
            for (std::size_t j = i; j < stop; ++j)
            {
                const Scalar entry = w * (n[0][i] * n[0][j] + n[1][i] * n[1][j] + n[2][i] * n[2][j]);
                mass(i, j) += entry;
                if (j != i)
                    mass(j, i) += entry;
            }
    }
}

template <typename Matrix, typename Scalar>
void addStation(Matrix& mass, double xi, double length, double weight, MassFormulation formulation,
                TorsionalInertia torsion, Scalar density, Scalar area, Scalar nsm, Scalar i1, Scalar i2, Scalar i12,
                const std::array<Scalar, 2>& neutral, const std::array<Scalar, 2>& centre)
{
    addStream(mass, weight * density * area, neutral, xi, length, formulation);
    addStream(mass, weight * nsm, centre, xi, length, formulation);
    const std::array<double, 2> n{1.0 - xi, xi};
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t b = 0; b < 2; ++b)
        {
            const auto w = density * weight * n[a] * (formulation == MassFormulation::Lumped ? double(a == b) : n[b]);
            const auto r = 6 * a + 3, c = 6 * b + 3;
            if (torsion == TorsionalInertia::Polar)
                mass(r, c) += w * (i1 + i2);
            mass(r + 1, c + 1) += w * i2;
            mass(r + 2, c + 2) += w * i1;
            mass(r + 1, c + 2) -= w * i12;
            mass(r + 2, c + 1) -= w * i12;
        }
}

template <typename Visitor>
void integrate(const BeamSection& section, double length, Visitor visit)
{
    for (std::size_t k = 1; k < section.count; ++k)
    {
        const double a = section.stations[k - 1].position;
        const double half = 0.5 * (section.stations[k].position - a);
        if (half > 0.0)
            for (std::size_t q = 0; q < GAUSS_NODES.size(); ++q)
                visit(a + half * (1.0 + GAUSS_NODES[q]), half * length * GAUSS_WEIGHTS[q]);
    }
}
} // namespace oos::elements::beam_mass
