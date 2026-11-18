#include "ShellQuad4Kernel.hpp"
#include "oos/elements/shells/SecondOrderScalar.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

namespace oos::elements
{

namespace
{

constexpr std::size_t DOFS = ShellQuad4::DOFS;
template <typename T>
using Affine = AffineRow<T, DOFS>;

} // namespace

// The values at the element centre (ShellCentreRecovery.hpp), where the local frame is the element frame and the
// enhanced strains vanish (G = 0 at ξ = η = 0):
// - strains: e0 = B_m q and the curvature of the tables, κ = −B_b q (the tables write ε(z) = ε_m − z κ);
// - forces (document section 12): N = A (e0 − ε_th) + B (κ_d − κ_th) and M_d = B (e0 − ε_th) + D (κ_d − κ_th), κ_d
//   the curvature of the document (ε(z) = e0 + z κ_d) and ε_th + z κ_th the free thermal strain; the moments of the
//   tables are −M_d (M = −∫σ z dz);
// - transverse shear forces: Q = S H S γ with S = diag(√s_x0, √s_y0), the forces conjugate to the shears γ of the
//   centre in the energy γ'ᵀ H γ' of the modified shears (the part odd across the element vanishes at the centre);
// - stresses at the fibre Z (ζ = 2 Z / t): σ = (G1 − 6 ζ G4)(e0 − ε_th) + (G2 − 6 ζ G4) Z (κ_d − κ_th), the stress
//   whose resultants over the thickness are N and M_d above (G2 without 12I/T³: the fibre stress M z / I of a plate
//   of bending material G2).
template <typename T>
CentreRecoveryOperator<T, ShellQuad4::DOFS> ShellQuad4::recovery(const Geometry& geometry,
                                                                 const PlateConstitutive<T>& constitutive,
                                                                 const std::array<T, NODES>& thickness,
                                                                 const std::array<T, 2>& fibres,
                                                                 const PlateThermalStrain& thermal)
{
    using namespace quad4_detail;
    const PlateConstitutive<T>& c = constitutive;
    const Shape shape = shapeAt(0.0, 0.0);
    const Frame frame = frameAt(geometry, shape);
    const T t = interpolated(shape, thickness);
    const ScalarMatrix<T, 6, DOFS> rows = planeRowsAt(geometry, thickness, shape, frame);
    const PointSection<T> section = sectionAt(c, t, 0.0, 4.0 * frame.area);

    // The mechanical strains as affine rows: the thermal strain in the last column, subtracted. Its dilatation is
    // taken as the strains of the homothety of the element, as in the thermal loads (ShellQuad4Loads.cpp).
    const std::array<T, 6> homothety = dilatationStrainsOf(geometry, rows);
    const double dilatation = thermal.isZero ? 0.0 : dilatationOf(thermal);
    const std::array<double, DOFS> homothetyDofs = director_shell::homothetyOf(geometry.position);
    std::array<Affine<T>, 3> membrane{};
    std::array<Affine<T>, 3> curvature{};
    for (std::size_t r = 0; r < 3; ++r)
    {
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            membrane[r][k] = rows(r, k);
            curvature[r][k] = rows(3 + r, k);
        }
        membrane[r][DOFS] = T(-thermal.membrane[r]) - dilatation * (homothety[r] - (r < 2 ? 1.0 : 0.0));
        curvature[r][DOFS] = T(-thermal.gradient[r]) - dilatation * homothety[3 + r];
    }

    CentreRecoveryOperator<T, DOFS> recovery;
    for (std::size_t i = 0; i < 3; ++i)
    {
        Affine<T> force{};
        Affine<T> moment{};
        std::array<Affine<T>, 2> stress{};
        for (std::size_t j = 0; j < 3; ++j)
        {
            addScaled(force, membrane[j], section.a(i, j));
            addScaled(force, curvature[j], section.b(i, j));
            addScaled(moment, membrane[j], -section.b(i, j));
            addScaled(moment, curvature[j], -section.d(i, j));
            for (std::size_t f = 0; f < 2; ++f)
            {
                const T zeta = 2.0 * fibres[f] / t;
                T onMembrane{};
                T onCurvature{};
                if (c.hasMembrane)
                    onMembrane += c.membrane(i, j);
                if (c.hasBending)
                    onCurvature += c.bending(i, j);
                if (c.hasCoupling && c.hasMembrane && c.hasBending)
                {
                    onMembrane -= 6.0 * zeta * c.coupling(i, j);
                    onCurvature -= 6.0 * zeta * c.coupling(i, j);
                }
                addScaled(stress[f], membrane[j], onMembrane);
                addScaled(stress[f], curvature[j], onCurvature * fibres[f]);
            }
        }
        setRow(recovery, centre_row::MEMBRANE_FORCE + i, force);
        setRow(recovery, centre_row::MOMENT + i, moment);
        setRow(recovery, centre_row::STRESS_FIBER1 + i, stress[0]);
        setRow(recovery, centre_row::STRESS_FIBER2 + i, stress[1]);
        // the total strains: no thermal column
        Affine<T> totalMembrane{};
        Affine<T> totalCurvature{};
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            totalMembrane[k] = rows(i, k);
            totalCurvature[k] = -rows(3 + i, k);
        }
        setRow(recovery, centre_row::MEMBRANE_STRAIN + i, totalMembrane);
        setRow(recovery, centre_row::CURVATURE + i, totalCurvature);
    }
    if (c.hasBending)
    {
        const TiedShears<T> tied = tiedShearsOf(geometry, thickness);
        const std::array<quad4_detail::Row<T>, 2> shear = shearRowsAt(geometry, thickness, tied, 0.0, 0.0);
        const ShearFactors<T> factors = shearFactorsOf(section, geometry.meanWidthX, geometry.meanWidthY, TWIST_FACTOR);
        const std::array<T, 2> scale{squareRoot(factors.evenX), squareRoot(factors.evenY)};
        for (std::size_t i = 0; i < 2; ++i)
        {
            Affine<T> force{};
            for (std::size_t j = 0; j < 2; ++j)
            {
                const T weight = scale[i] * section.h(i, j) * scale[j];
                for (std::size_t k = 0; k < DOFS; ++k)
                    force[k] += weight * shear[j][k];
            }
            // the shear of the free dilatation (directors that are not normal to the element), subtracted
            for (std::size_t k = 0; k < DOFS; ++k)
                force[DOFS] -= dilatation * force[k] * homothetyDofs[k];
            setRow(recovery, centre_row::SHEAR_FORCE + i, force);
        }
    }
    return recovery;
}

template CentreRecoveryOperator<double, ShellQuad4::DOFS> ShellQuad4::recovery(const Geometry&,
                                                                               const PlateConstitutive<double>&,
                                                                               const std::array<double, NODES>&,
                                                                               const std::array<double, 2>&,
                                                                               const PlateThermalStrain&);
template CentreRecoveryOperator<Complex, ShellQuad4::DOFS> ShellQuad4::recovery(const Geometry&,
                                                                                const PlateConstitutive<Complex>&,
                                                                                const std::array<Complex, NODES>&,
                                                                                const std::array<Complex, 2>&,
                                                                                const PlateThermalStrain&);
template CentreRecoveryOperator<SecondOrderScalar, ShellQuad4::DOFS> ShellQuad4::recovery(
    const Geometry&, const PlateConstitutive<SecondOrderScalar>&, const std::array<SecondOrderScalar, NODES>&,
    const std::array<SecondOrderScalar, 2>&, const PlateThermalStrain&);

} // namespace oos::elements
