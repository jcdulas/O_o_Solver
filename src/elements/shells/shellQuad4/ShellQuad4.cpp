#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include "ShellQuad4Kernel.hpp"
#include "oos/elements/shells/SecondOrderScalar.hpp"

namespace oos::elements
{

std::optional<std::string> ShellQuad4::integrationProblem(const Geometry& geometry,
                                                          const std::array<double, NODES>& thickness)
{
    using namespace quad4_detail;
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
            for (const double zeta : {-1.0, 1.0})
                if (!(determinantOf(covariantAt(geometry, thickness, shapeAt(xi, eta), zeta)) > 0.0))
                    return std::string(
                        "expected a positive Jacobian at every integration point, found one not "
                        "positive (a folded corner, or a thickness large against a warped element); "
                        "reorder ShellQuad4.node1-ShellQuad4.node4 around the element or remesh it");
    return std::nullopt;
}

// Document section 9: with the 2 × 2 rule (weights 1),
//   K_dd = Σ j (Bᵀ C B + B_sᵀ H B_s + c_d A33 b_dᵀ b_d),  K = K_dd − K_dα K_αα⁻¹ K_dαᵀ,
// B the rows of (e0, κ), C = [[A, B], [B, D]], B_s the modified transverse shears, b_d the drilling row. The drilling
// term is present when the section has both a membrane and a bending stiffness.
template <typename T>
ScalarMatrix<T, ShellQuad4::DOFS> ShellQuad4::stiffness(const Geometry& geometry,
                                                        const PlateConstitutive<T>& constitutive,
                                                        const std::array<T, NODES>& thickness)
{
    using namespace quad4_detail;
    const std::array<PlanePoint<T>, POINT_COUNT> points = planePointsOf(geometry, constitutive, thickness);
    const Widths widths = widthsOf(geometry);
    ScalarMatrix<T, DOFS> k;
    TiedShears<T> tied;
    if (constitutive.hasBending)
        tied = tiedShearsOf(geometry, thickness);
    for (const PlanePoint<T>& point : points)
    {
        addCongruence(k, point.rows, planeMatrixOf(point.section), T(point.weight));
        if (constitutive.hasBending)
        {
            // The modified shears (document section 7): the part of γ1 even across η weighted by √s_x0 and its odd
            // part by √s_x1; γ2 likewise across ξ. Each γ is taken with the frame of its own point.
            const ShearFactors<T> factors = shearFactorsOf(point.section, widths.a, widths.b, TWIST_FACTOR);
            const std::array<Row<T>, 2> here = shearRowsAt(geometry, thickness, tied, point.xi, point.eta);
            const std::array<Row<T>, 2> acrossEta = shearRowsAt(geometry, thickness, tied, point.xi, -point.eta);
            const std::array<Row<T>, 2> acrossXi = shearRowsAt(geometry, thickness, tied, -point.xi, point.eta);
            const T evenX = 0.5 * squareRoot(factors.evenX);
            const T oddX = 0.5 * squareRoot(factors.oddX);
            const T evenY = 0.5 * squareRoot(factors.evenY);
            const T oddY = 0.5 * squareRoot(factors.oddY);
            ScalarMatrix<T, 2, DOFS> shear;
            for (std::size_t dof = 0; dof < DOFS; ++dof)
            {
                shear(0, dof) = evenX * (here[0][dof] + acrossEta[0][dof]) + oddX * (here[0][dof] - acrossEta[0][dof]);
                shear(1, dof) = evenY * (here[1][dof] + acrossXi[1][dof]) + oddY * (here[1][dof] - acrossXi[1][dof]);
            }
            addCongruence(k, shear, point.section.h, T(point.weight));
        }
        if (constitutive.hasMembrane && constitutive.hasBending && constitutive.drillingFactor > 0.0)
        {
            const Row<T> drilling = drillingRowAt(geometry, thickness, shapeAt(point.xi, point.eta), point.frame);
            const T scale = (constitutive.drillingFactor * point.weight) * point.section.a(2, 2);
            for (std::size_t i = 0; i < DOFS; ++i)
            {
                if (drilling[i] == T(0.0))
                    continue;
                const T left = scale * drilling[i];
                for (std::size_t j = 0; j < DOFS; ++j)
                    k(i, j) += left * drilling[j];
            }
        }
    }
    // The enhanced strains condensed: K −= K_dα K_αα⁻¹ K_dαᵀ.
    const EnhancedPart<T> enhanced = enhancedPartOf(constitutive, points);
    const ScalarMatrix<T, ENHANCED, DOFS> solved = solvedEnhanced(enhanced, transposeOf(enhanced.coupling));
    for (std::size_t i = 0; i < DOFS; ++i)
        for (std::size_t q = 0; q < ENHANCED; ++q)
        {
            const T left = enhanced.coupling(i, q);
            if (left == T(0.0))
                continue;
            for (std::size_t j = 0; j < DOFS; ++j)
                k(i, j) -= left * solved(q, j);
        }
    // K_ij = K_ji = ½ (K_ij + K_ji): the round-off of the two triangles is shared.
    for (std::size_t r = 0; r < DOFS; ++r)
        for (std::size_t c = r + 1; c < DOFS; ++c)
        {
            const T mean = 0.5 * (k(r, c) + k(c, r));
            k(r, c) = mean;
            k(c, r) = mean;
        }
    return k;
}

double ShellQuad4::drillingEnergy(const Geometry& geometry, const PlateConstitutive<double>& constitutive,
                                  const std::array<double, NODES>& thickness, const std::array<double, DOFS>& q)
{
    using namespace quad4_detail;
    if (!constitutive.hasMembrane || !constitutive.hasBending || !(constitutive.drillingFactor > 0.0))
        return 0.0;
    double energy = 0.0;
    for (const PlanePoint<double>& point : planePointsOf(geometry, constitutive, thickness))
    {
        const Row<double> drilling = drillingRowAt(geometry, thickness, shapeAt(point.xi, point.eta), point.frame);
        double rotation = 0.0; // r = ω − θ_n at the point
        for (std::size_t i = 0; i < DOFS; ++i)
            rotation += drilling[i] * q[i];
        energy += 0.5 * constitutive.drillingFactor * point.weight * point.section.a(2, 2) * rotation * rotation;
    }
    return energy;
}

template ScalarMatrix<double, ShellQuad4::DOFS> ShellQuad4::stiffness(const Geometry&, const PlateConstitutive<double>&,
                                                                      const std::array<double, NODES>&);
template ScalarMatrix<Complex, ShellQuad4::DOFS> ShellQuad4::stiffness(const Geometry&,
                                                                       const PlateConstitutive<Complex>&,
                                                                       const std::array<Complex, NODES>&);
template ScalarMatrix<SecondOrderScalar, ShellQuad4::DOFS> ShellQuad4::stiffness(
    const Geometry&, const PlateConstitutive<SecondOrderScalar>&, const std::array<SecondOrderScalar, NODES>&);

} // namespace oos::elements
