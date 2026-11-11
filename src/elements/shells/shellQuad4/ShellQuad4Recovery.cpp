#include "ShellQuad4Kinematics.hpp"
#include "oos/elements/shells/SecondOrderScalar.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

namespace oos::elements
{

namespace
{

using quad4_detail::GAUSS_POINT;
using quad4_detail::GAUSS_POINTS;

constexpr std::size_t DOFS = ShellQuad4::DOFS;
template <typename T>
using Row = AffineRow<T, DOFS>;

// Spec §7.2(2): the in-plane evaluation points, the four Gauss points then the centre.
constexpr std::size_t GAUSS_POINT_COUNT = 4;
constexpr std::array<std::array<double, 2>, 5> RECOVERY_POINTS{{{-GAUSS_POINT, -GAUSS_POINT},
                                                                {-GAUSS_POINT, GAUSS_POINT},
                                                                {GAUSS_POINT, -GAUSS_POINT},
                                                                {GAUSS_POINT, GAUSS_POINT},
                                                                {0.0, 0.0}}};
// Spec §7.2(8): the strains are the means over the five points.
constexpr double STRAIN_POINT_WEIGHT = 1.0 / 5.0;
// Spec §7.2(5): the linear extrapolation in ζ from ±c to the faces ±1: f(±1) = f(±c)(1 + e) − f(∓c) e.
constexpr double FACE_EXTRAPOLATION = (1.0 - GAUSS_POINT) / (2.0 * GAUSS_POINT);

// The stresses and transverse shear stresses at one in-plane point, at the two heights ζ = ∓c.
template <typename T>
struct PointStresses
{
    std::array<std::array<Row<T>, 3>, 2> stress{};
    std::array<std::array<Row<T>, 2>, 2> shear{};
};

// The value of a pair (at −c, at +c) extrapolated to the bottom (face −1) or top (face +1) face.
template <typename T>
[[nodiscard]] Row<T> extrapolated(const Row<T>& atBottom, const Row<T>& atTop, bool toTop)
{
    Row<T> face{};
    const Row<T>& near = toTop ? atTop : atBottom;
    const Row<T>& far = toTop ? atBottom : atTop;
    addScaled(face, near, T(1.0 + FACE_EXTRAPOLATION));
    addScaled(face, far, T(-FACE_EXTRAPOLATION));
    return face;
}

} // namespace

// Spec §7.2: the stresses at ζ = ±c of the 4 Gauss points and the centre, from the strain rows of §2.5 and §2.6 (the
// in-plane shears at the centre, the transverse shears at the point itself), σ = (G1 − 6ζ G4) ε_m + G2 ε_b (and
// −6ζ G4 ε_b, see below) and
// τ = G3_eff γ with the recovery factors f_x, f_y; extrapolated to the faces; the centre values:
// - stresses at Z1, Z2: the mean over the Gauss points of (½ − Z/t) σ_bottom + (½ + Z/t) σ_top;
// - membrane and shear forces: the mean over the Gauss points of (σ_bottom + σ_top) t/2, (τ_bottom + τ_top) t/2;
// - moments from the centre point: (σ_bottom − σ_top) I / t, I = (12I/T³) t³/12;
// - strains: the mean over the 5 points at ζ = −c of the membrane rows, and of the bending rows scaled to curvatures
//   (see below).
// G2 enters the stresses without 12I/T³ while the moments take I with it: NASTRAN-95's recovery, reproduced (the fibre
// stress M z/I of a plate of bending material G2).
template <typename T>
CentreRecoveryOperator<T, ShellQuad4::DOFS> ShellQuad4::recovery(const Geometry& geometry,
                                                                 const PlateConstitutive<T>& constitutive,
                                                                 const std::array<T, NODES>& thickness,
                                                                 const std::array<T, 2>& fibres,
                                                                 const PlateThermalStrain& thermal)
{
    using namespace quad4_detail;
    const PlateConstitutive<T>& c = constitutive;
    T meanThickness{};
    for (const T& t : thickness)
        meanThickness += 0.25 * t;
    const ShearCorrection<T> correction = shearCorrectionOf(geometry, c, meanThickness);
    std::array<ScalarMatrix<T, 6, 24>, 2> centreRows;
    for (std::size_t z = 0; z < 2; ++z)
        centreRows[z] = inPlaneStrainRows(
            geometry, thickness, kinematicsAt(geometry, thickness, 0.0, 0.0, GAUSS_POINTS[z]), GAUSS_POINTS[z]);

    std::array<Row<T>, 3> membraneForce{};
    std::array<Row<T>, 3> moment{};
    std::array<Row<T>, 2> shearForce{};
    std::array<std::array<Row<T>, 3>, 2> fibreStress{};
    std::array<Row<T>, 3> membraneStrain{};
    std::array<Row<T>, 3> bendingStrain{}; // the mean bending strain at ζ = −c
    for (std::size_t p = 0; p < RECOVERY_POINTS.size(); ++p)
    {
        const double xi = RECOVERY_POINTS[p][0];
        const double eta = RECOVERY_POINTS[p][1];
        const bool isCentre = p == GAUSS_POINT_COUNT;
        const T t = interpolated(bilinearShape(xi, eta), thickness);
        const T inertia = c.bendingRatio * t * t * t / 12.0;
        const std::array<T, 2> f = recoveryShearFactors(geometry, c, correction, t, xi, eta, isCentre);
        const T coupled = squareRoot(f[0] * f[1]);
        PointStresses<T> at;
        for (std::size_t z = 0; z < 2; ++z)
        {
            const double zeta = GAUSS_POINTS[z];
            const PointKinematics<T> point = kinematicsAt(geometry, thickness, xi, eta, zeta);
            const ScalarMatrix<T, 6, 24> plane = inPlaneStrainRows(geometry, thickness, point, zeta);
            const ScalarMatrix<T, 2, 24> shear = transverseShearRows(geometry, point);
            // The strains as affine rows, the thermal strain in the last column (subtracted).
            std::array<Row<T>, 3> em{};
            std::array<Row<T>, 3> eb{};
            std::array<Row<T>, 2> gamma{};
            const T height = 0.5 * zeta * t;
            for (std::size_t r = 0; r < 3; ++r)
            {
                for (std::size_t k = 0; k < DOFS; ++k)
                {
                    em[r][k] = r == 2 ? centreRows[z](2, k) : plane(r, k);
                    eb[r][k] = r == 2 ? centreRows[z](5, k) : plane(3 + r, k);
                }
                if (z == 0)
                {
                    addScaled(membraneStrain[r], em[r], T(STRAIN_POINT_WEIGHT));
                    addScaled(bendingStrain[r], eb[r], T(STRAIN_POINT_WEIGHT));
                }
                em[r][DOFS] = T(-thermal.membrane[r]);
                eb[r][DOFS] = -height * thermal.gradient[r];
            }
            for (std::size_t r = 0; r < 2; ++r)
                for (std::size_t k = 0; k < DOFS; ++k)
                    gamma[r][k] = shear(r, k);
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                {
                    if (c.hasMembrane)
                    {
                        T membrane = c.membrane(i, j);
                        if (c.hasCoupling)
                            membrane -= 6.0 * zeta * c.coupling(i, j);
                        addScaled(at.stress[z][i], em[j], membrane);
                    }
                    if (c.hasBending)
                        addScaled(at.stress[z][i], eb[j], c.bending(i, j));
                    // Deviation from NASTRAN-95 (deliberate): the membrane stress also takes the stiffness's cross
                    // term −6ζ G4 on the bending strain, which integrates to the coupling t² G4 κ of the membrane
                    // forces; without it the forces of a coupled section (MID4, an unsymmetric laminate) miss that term
                    // and do not balance the loads (the TRIA3 recovery has it).
                    if (c.hasCoupling && c.hasMembrane && c.hasBending)
                        addScaled(at.stress[z][i], eb[j], -6.0 * zeta * c.coupling(i, j));
                }
            if (c.hasBending)
            {
                // τ = G3_eff γ, G3_eff = [[f_x G3xx, √(f_x f_y) G3xy], [√(f_x f_y) G3xy, f_y G3yy]] (κ G in each).
                addScaled(at.shear[z][0], gamma[0], f[0] * c.transverseShear(0, 0));
                addScaled(at.shear[z][0], gamma[1], coupled * c.transverseShear(0, 1));
                addScaled(at.shear[z][1], gamma[0], coupled * c.transverseShear(1, 0));
                addScaled(at.shear[z][1], gamma[1], f[1] * c.transverseShear(1, 1));
            }
        }
        for (std::size_t i = 0; i < 3; ++i)
        {
            const Row<T> bottom = extrapolated(at.stress[0][i], at.stress[1][i], false);
            const Row<T> top = extrapolated(at.stress[0][i], at.stress[1][i], true);
            if (isCentre)
            {
                addScaled(moment[i], bottom, inertia / t);
                addScaled(moment[i], top, -inertia / t);
                continue;
            }
            const T quarter(1.0 / GAUSS_POINT_COUNT);
            for (std::size_t k = 0; k < 2; ++k)
            {
                addScaled(fibreStress[k][i], bottom, quarter * (0.5 - fibres[k] / t));
                addScaled(fibreStress[k][i], top, quarter * (0.5 + fibres[k] / t));
            }
            addScaled(membraneForce[i], bottom, quarter * 0.5 * t);
            addScaled(membraneForce[i], top, quarter * 0.5 * t);
        }
        if (isCentre)
            continue;
        for (std::size_t i = 0; i < 2; ++i)
        {
            const T scale = 0.5 * t / static_cast<double>(GAUSS_POINT_COUNT);
            addScaled(shearForce[i], extrapolated(at.shear[0][i], at.shear[1][i], false), scale);
            addScaled(shearForce[i], extrapolated(at.shear[0][i], at.shear[1][i], true), scale);
        }
    }

    CentreRecoveryOperator<T, DOFS> recovery;
    // Spec §7.2(8): the bending rows at ζ = −c are −c (t/2) κ_std = c (t/2) κ with the shell curvature κ = −κ_std of
    // this solver's tables (ε(z) = ε_m − z κ, M = D κ), so κ = mean / (c t̄/2). (The spec writes the NASTRAN-95 sign
    // as −mean/(c t̄/2); the curvature written here is the one conjugate to the moments of the force table.)
    const T curvatureScale = 1.0 / (GAUSS_POINT * 0.5 * meanThickness);
    for (std::size_t i = 0; i < 3; ++i)
    {
        setRow(recovery, centre_row::MEMBRANE_FORCE + i, membraneForce[i]);
        setRow(recovery, centre_row::MOMENT + i, moment[i]);
        setRow(recovery, centre_row::STRESS_FIBER1 + i, fibreStress[0][i]);
        setRow(recovery, centre_row::STRESS_FIBER2 + i, fibreStress[1][i]);
        setRow(recovery, centre_row::MEMBRANE_STRAIN + i, membraneStrain[i]);
        Row<T> curvature{};
        addScaled(curvature, bendingStrain[i], curvatureScale);
        setRow(recovery, centre_row::CURVATURE + i, curvature);
    }
    for (std::size_t i = 0; i < 2; ++i)
        setRow(recovery, centre_row::SHEAR_FORCE + i, shearForce[i]);
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
