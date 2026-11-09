#include "oos/elements/bars_beams/BeamMass.hpp"

#include <array>

namespace oos::elements
{

namespace
{

// Gauss-Legendre rule of 4 points on [−1, 1] (exact to degree 7).
constexpr std::array<double, 4> GAUSS_NODES{-0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
                                            0.8611363115940526};
constexpr std::array<double, 4> GAUSS_WEIGHTS{0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
                                              0.3478548451374538};

// N(ξ), the displacements (u, v, w) of the axis at ξ = x/L from the end DOFs: linear on u, Hermite cubics on (v, θz)
// and (w, −θy) (θy = −w', θz = v').
base::MatN<3, 12> shapeFunctions(double xi, double length)
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

} // namespace

base::MatN<12> beamMass(const BeamSection& section, double density, double addedMassPerLength, double length,
                        MassFormulation formulation, TorsionalInertia torsionalInertia)
{
    base::MatN<12> mass;
    double massA = 0.0;
    double massB = 0.0;
    double torsionA = 0.0;
    double torsionB = 0.0;
    const bool hasTorsionalInertia = torsionalInertia == TorsionalInertia::Polar;
    for (std::size_t k = 0; k + 1 < section.count; ++k)
    {
        const double a = section.stations[k].position;
        const double b = section.stations[k + 1].position;
        if (b <= a)
            continue;
        const double half = 0.5 * (b - a);
        for (std::size_t point = 0; point < GAUSS_NODES.size(); ++point)
        {
            const double xi = a + half * (1.0 + GAUSS_NODES[point]);
            const BeamStation station = section.at(xi);
            const double perLength = density * station.area + station.nonStructuralMass + addedMassPerLength;
            const double weight = perLength * half * GAUSS_WEIGHTS[point] * length;
            // The polar mass moment of inertia per unit length ρ·(I1 + I2) (the polar moment of the section about its
            // axis is I1 + I2 whatever the axes, I12 or not).
            const double torsionWeight = hasTorsionalInertia ? density * (station.inertia1 + station.inertia2) * half *
                                                                   GAUSS_WEIGHTS[point] * length
                                                             : 0.0;
            if (formulation == MassFormulation::Lumped)
            {
                massA += weight * (1.0 - xi);
                massB += weight * xi;
                torsionA += torsionWeight * (1.0 - xi);
                torsionB += torsionWeight * xi;
            }
            else
            {
                const base::MatN<3, 12> n = shapeFunctions(xi, length);
                mass += weight * (n.transposed() * n);
                // Consistent torsional inertia ∫ ρ·(I1 + I2)·Nᵀ·N dx with the linear shape functions of θx
                // (N = [1 − ξ, ξ] on the end DOFs 3 and 9): ρ·(I1 + I2)·L/6·[2 1; 1 2] for a uniform section.
                mass(3, 3) += torsionWeight * (1.0 - xi) * (1.0 - xi);
                mass(3, 9) += torsionWeight * (1.0 - xi) * xi;
                mass(9, 3) += torsionWeight * (1.0 - xi) * xi;
                mass(9, 9) += torsionWeight * xi * xi;
            }
        }
    }
    if (formulation == MassFormulation::Lumped)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            mass(i, i) = massA;
            mass(6 + i, 6 + i) = massB;
        }
        mass(3, 3) = torsionA;
        mass(9, 9) = torsionB;
    }
    return mass;
}

} // namespace oos::elements
