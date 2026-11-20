#include "ShellQuad4Kernel.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include <cmath>

// The recovery of the ShellQuad4 at its corners (docs/shell-quad4-formulation.md, "the document", section 12).
//
// The resultants of the element are those of its 2 × 2 integration points, with the enhanced strains:
//   α = K_αα⁻¹ (f_α − K_dαᵀ d),   (N, M) = C (B d + G α) − S0.
// The values of a corner are read on the bilinear surface through the values of the four points (the local least
// squares smoothing of E. Hinton, J. S. Campbell, Local and global smoothing of discontinuous finite element functions
// using a least squares method, Int. J. Numer. Methods Eng. 8, 1974): the points of the 2 × 2 rule are those where the
// strains of a four-node element are the most accurate (J. Barlow, Optimal stress locations in finite element models,
// Int. J. Numer. Methods Eng. 10, 1976), which the strains taken at the corners themselves are not.
//
// Every value is an affine map of the local DOFs (ShellCentreRecovery.hpp), so the corner operators are the same
// combination of the operators of the points.

namespace oos::elements
{

namespace
{

using namespace quad4_detail;

using Affine = AffineRow<double, DOFS>;
using PointOperator = CentreRecoveryOperator<double, DOFS>;
template <std::size_t ROWS>
using AffineRows = std::array<Affine, ROWS>;
// α as an affine map of the local DOFs: one row per enhanced parameter.
using EnhancedParameters = ScalarMatrix<double, ENHANCED, DOFS + 1>;

// The weight of integration point (ξ_p, η_p) in the value at corner i of the bilinear surface through the four
// points: ¼ (1 + 3 ξ_i ξ_p)(1 + 3 η_i η_p), since ξ_p² = η_p² = 1/3.
[[nodiscard]] double extrapolationWeight(std::size_t corner, double xi, double eta)
{
    return 0.25 * (1.0 + 3.0 * NODE_XI[corner] * xi) * (1.0 + 3.0 * NODE_ETA[corner] * eta);
}

// The point across η of integration point p, and across ξ (planePointsOf: p = 2 · (index of ξ) + (index of η)).
[[nodiscard]] constexpr std::size_t acrossEtaOf(std::size_t p) noexcept
{
    return p ^ 1U;
}
[[nodiscard]] constexpr std::size_t acrossXiOf(std::size_t p) noexcept
{
    return p ^ 2U;
}

// The free thermal strain (e0, κ) of a point in its local frame, less its dilatation, which the element takes as
// its homothety (ShellQuad4::thermalLoads).
[[nodiscard]] std::array<double, 6> freeStrainAt(const PlanePoint<double>& point, const PlateThermalStrain& thermal)
{
    std::array<double, 3> membrane = thermal.membrane;
    std::array<double, 3> gradient = thermal.gradient;
    if (std::abs(point.angle) > NO_ROTATION)
    {
        const Matrix3<double> toLocal = engineeringStrainRotation(-point.angle);
        membrane = productOf(toLocal, thermal.membrane);
        gradient = productOf(toLocal, thermal.gradient);
    }
    const double dilatation = dilatationOf(thermal);
    return {membrane[0] - dilatation, membrane[1] - dilatation, membrane[2], gradient[0], gradient[1], gradient[2]};
}

// α = K_αα⁻¹ (f_α − K_dαᵀ d) as an affine map of d: the columns of −K_dαᵀ, then f_α = Σ j Gᵀ C ε_th of the free
// thermal strains of the points.
[[nodiscard]] EnhancedParameters enhancedParametersOf(const PlateConstitutive<double>& constitutive,
                                                      const std::array<PlanePoint<double>, POINT_COUNT>& points,
                                                      const PlateThermalStrain& thermal)
{
    const EnhancedPart<double> enhanced = enhancedPartOf(constitutive, points);
    EnhancedParameters rightHandSides;
    for (std::size_t q = 0; q < ENHANCED; ++q)
        for (std::size_t dof = 0; dof < DOFS; ++dof)
            rightHandSides(q, dof) = -enhanced.coupling(dof, q);
    if (!thermal.isZero)
        for (const PlanePoint<double>& point : points)
        {
            const std::array<double, 6> resultants =
                productOf(planeMatrixOf(point.section), freeStrainAt(point, thermal));
            for (std::size_t q = 0; q < ENHANCED; ++q)
            {
                const std::size_t part = q < ENHANCED_PER_PART ? 0 : 3;
                for (std::size_t r = 0; r < 3; ++r)
                    rightHandSides(q, DOFS) +=
                        point.weight * point.enhanced(r, q % ENHANCED_PER_PART) * resultants[part + r];
            }
        }
    return solvedEnhanced(enhanced, rightHandSides);
}

// The dilatation ε of the thermal strain is the homothety u = ε x of the element: a row written for the displacements
// d − ε x is brought to d by moving ε times its value on the homothety to its constant.
void bringToDisplacements(Affine& row, const std::array<double, DOFS>& homothety, double dilatation)
{
    if (dilatation == 0.0)
        return;
    double onHomothety = 0.0;
    for (std::size_t k = 0; k < DOFS; ++k)
        onHomothety += row[k] * homothety[k];
    row[DOFS] -= dilatation * onHomothety;
}

// Three rows of engineering strains turned by `rotation`.
[[nodiscard]] AffineRows<3> turned(const AffineRows<3>& rows, const Matrix3<double>& rotation)
{
    AffineRows<3> result{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            addScaled(result[i], rows[j], rotation(i, j));
    return result;
}

// The strains of one integration point in the axes of the element turned onto its tangent plane: the mechanical
// strains (the free thermal strain subtracted) and the total strains, membrane then curvature (ε(z) = e0 + z κ_d).
struct PointStrains
{
    AffineRows<3> membrane{};
    AffineRows<3> curvature{};
    AffineRows<3> totalMembrane{};
    AffineRows<3> totalCurvature{};
};

// B d + G α at a point (document sections 5 and 6), then turned from the local frame of the point to the element
// axes (ε_element = U3(angle) ε_local, frameAngleOf).
[[nodiscard]] PointStrains strainsAt(const PlanePoint<double>& point, const EnhancedParameters& parameters,
                                     const PlateThermalStrain& thermal, const std::array<double, DOFS>& homothety)
{
    const double dilatation = thermal.isZero ? 0.0 : dilatationOf(thermal);
    const std::array<double, 6> freeStrain = thermal.isZero ? std::array<double, 6>{} : freeStrainAt(point, thermal);
    AffineRows<6> mechanical{};
    AffineRows<6> total{};
    for (std::size_t r = 0; r < 6; ++r)
    {
        // G α: the membrane strains take the parameters 0-3, the curvatures 4-7.
        Affine enhanced{};
        const std::size_t part = r < 3 ? 0 : ENHANCED_PER_PART;
        for (std::size_t s = 0; s < ENHANCED_PER_PART; ++s)
            for (std::size_t k = 0; k <= DOFS; ++k)
                enhanced[k] += point.enhanced(r % 3, s) * parameters(part + s, k);
        for (std::size_t k = 0; k < DOFS; ++k)
            mechanical[r][k] = point.rows(r, k) + enhanced[k];
        mechanical[r][DOFS] = enhanced[DOFS] - freeStrain[r];
        bringToDisplacements(mechanical[r], homothety, dilatation);
        // The total strains: B d of the displacements themselves, with the enhanced strains of the element.
        bringToDisplacements(enhanced, homothety, dilatation);
        for (std::size_t k = 0; k < DOFS; ++k)
            total[r][k] = point.rows(r, k) + enhanced[k];
        total[r][DOFS] = enhanced[DOFS];
    }
    const Matrix3<double> toElement = engineeringStrainRotation(point.angle);
    PointStrains strains;
    strains.membrane = turned({mechanical[0], mechanical[1], mechanical[2]}, toElement);
    strains.curvature = turned({mechanical[3], mechanical[4], mechanical[5]}, toElement);
    strains.totalMembrane = turned({total[0], total[1], total[2]}, toElement);
    strains.totalCurvature = turned({total[3], total[4], total[5]}, toElement);
    return strains;
}

// The forces, moments, fibre stresses and strains of a point from its strains (the rows of ShellQuad4::recovery,
// with the section at the thickness t of the point): N = A e + B κ_d, the moments of the tables −(B e + D κ_d), the
// stress at the fibre Z (ζ = 2 Z / t) (G1 − 6 ζ G4) e + (G2 − 6 ζ G4) Z κ_d, the curvature of the tables −κ_d.
void setPlaneRows(PointOperator& recovery, const PointStrains& strains, const PlateConstitutive<double>& c,
                  const PointSection<double>& section, double t, const std::array<double, 2>& fibres)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        Affine force{};
        Affine moment{};
        std::array<Affine, 2> stress{};
        for (std::size_t j = 0; j < 3; ++j)
        {
            addScaled(force, strains.membrane[j], section.a(i, j));
            addScaled(force, strains.curvature[j], section.b(i, j));
            addScaled(moment, strains.membrane[j], -section.b(i, j));
            addScaled(moment, strains.curvature[j], -section.d(i, j));
            for (std::size_t f = 0; f < 2; ++f)
            {
                const double zeta = 2.0 * fibres[f] / t;
                double onMembrane = c.hasMembrane ? c.membrane(i, j) : 0.0;
                double onCurvature = c.hasBending ? c.bending(i, j) : 0.0;
                if (c.hasCoupling && c.hasMembrane && c.hasBending)
                {
                    onMembrane -= 6.0 * zeta * c.coupling(i, j);
                    onCurvature -= 6.0 * zeta * c.coupling(i, j);
                }
                addScaled(stress[f], strains.membrane[j], onMembrane);
                addScaled(stress[f], strains.curvature[j], onCurvature * fibres[f]);
            }
        }
        Affine tableCurvature{};
        addScaled(tableCurvature, strains.totalCurvature[i], -1.0);
        setRow(recovery, centre_row::MEMBRANE_FORCE + i, force);
        setRow(recovery, centre_row::MOMENT + i, moment);
        setRow(recovery, centre_row::STRESS_FIBER1 + i, stress[0]);
        setRow(recovery, centre_row::STRESS_FIBER2 + i, stress[1]);
        setRow(recovery, centre_row::MEMBRANE_STRAIN + i, strains.totalMembrane[i]);
        setRow(recovery, centre_row::CURVATURE + i, tableCurvature);
    }
}

// The transverse shear of the four points: Q' = H γ' of the modified shears γ' of each point (document section 7),
// in its local frame, and the square roots of its factors.
struct ModifiedShears
{
    std::array<std::array<Row<double>, 2>, POINT_COUNT> force{}; // Q'
    std::array<double, POINT_COUNT> evenX{};                     // √s_x0
    std::array<double, POINT_COUNT> oddX{};                      // √s_x1
    std::array<double, POINT_COUNT> evenY{};
    std::array<double, POINT_COUNT> oddY{};
};

[[nodiscard]] ModifiedShears modifiedShearsOf(const ShellQuad4Geometry& geometry,
                                              const std::array<double, NODES>& thickness,
                                              const std::array<PlanePoint<double>, POINT_COUNT>& points)
{
    const TiedShears<double> tied = tiedShearsOf(geometry, thickness);
    ModifiedShears shears;
    for (std::size_t p = 0; p < POINT_COUNT; ++p)
    {
        const PlanePoint<double>& point = points[p];
        const ShearFactors<double> factors =
            shearFactorsOf(point.section, geometry.meanWidthX, geometry.meanWidthY, TWIST_FACTOR);
        shears.evenX[p] = std::sqrt(factors.evenX);
        shears.oddX[p] = std::sqrt(factors.oddX);
        shears.evenY[p] = std::sqrt(factors.evenY);
        shears.oddY[p] = std::sqrt(factors.oddY);
        const std::array<Row<double>, 2> here = shearRowsAt(geometry, thickness, tied, point.xi, point.eta);
        const std::array<Row<double>, 2> acrossEta = shearRowsAt(geometry, thickness, tied, point.xi, -point.eta);
        const std::array<Row<double>, 2> acrossXi = shearRowsAt(geometry, thickness, tied, -point.xi, point.eta);
        for (std::size_t dof = 0; dof < DOFS; ++dof)
        {
            const double modifiedX = 0.5 * (shears.evenX[p] * (here[0][dof] + acrossEta[0][dof]) +
                                            shears.oddX[p] * (here[0][dof] - acrossEta[0][dof]));
            const double modifiedY = 0.5 * (shears.evenY[p] * (here[1][dof] + acrossXi[1][dof]) +
                                            shears.oddY[p] * (here[1][dof] - acrossXi[1][dof]));
            shears.force[p][0][dof] = point.section.h(0, 0) * modifiedX + point.section.h(0, 1) * modifiedY;
            shears.force[p][1][dof] = point.section.h(1, 0) * modifiedX + point.section.h(1, 1) * modifiedY;
        }
    }
    return shears;
}

// The transverse shear forces of point p: those conjugate to its tied shears γ(p) in the energy Σ j γ'ᵀ H γ' of the
// modified shears. γ1(p) enters γ'_x of p with ½ (√s_x0 + √s_x1) and γ'_x of the point across η with
// ½ (√s_x0 − √s_x1) of that point, so that
//   Q_x(p) = ½ (√s_x0 + √s_x1)_p Q'_x(p) + (j_q / j_p) ½ (√s_x0 − √s_x1)_q Q'_x(q),  q across η,
// and Q_y alike across ξ; at the centre of the element, the forces of ShellQuad4::recovery. Turned to the element
// axes as a vector.
void setShearRows(PointOperator& recovery, const ModifiedShears& shears,
                  const std::array<PlanePoint<double>, POINT_COUNT>& points, std::size_t p,
                  const std::array<double, DOFS>& homothety, double dilatation)
{
    const std::size_t qx = acrossEtaOf(p);
    const std::size_t qy = acrossXiOf(p);
    const double hereX = 0.5 * (shears.evenX[p] + shears.oddX[p]);
    const double hereY = 0.5 * (shears.evenY[p] + shears.oddY[p]);
    const double acrossX = 0.5 * (shears.evenX[qx] - shears.oddX[qx]) * points[qx].weight / points[p].weight;
    const double acrossY = 0.5 * (shears.evenY[qy] - shears.oddY[qy]) * points[qy].weight / points[p].weight;
    Affine localX{};
    Affine localY{};
    for (std::size_t dof = 0; dof < DOFS; ++dof)
    {
        localX[dof] = hereX * shears.force[p][0][dof] + acrossX * shears.force[qx][0][dof];
        localY[dof] = hereY * shears.force[p][1][dof] + acrossY * shears.force[qy][1][dof];
    }
    // the shear of the free dilatation (directors that are not normal to the element), subtracted
    bringToDisplacements(localX, homothety, dilatation);
    bringToDisplacements(localY, homothety, dilatation);
    const double cosine = std::cos(points[p].angle);
    const double sine = std::sin(points[p].angle);
    for (std::size_t k = 0; k <= DOFS; ++k)
    {
        recovery(centre_row::SHEAR_FORCE, k) = cosine * localX[k] + sine * localY[k];
        recovery(centre_row::SHEAR_FORCE + 1, k) = -sine * localX[k] + cosine * localY[k];
    }
}

} // namespace

std::array<CentreRecoveryOperator<double, ShellQuad4::DOFS>, ShellQuad4::NODES> ShellQuad4::cornerRecovery(
    const Geometry& geometry, const PlateConstitutive<double>& constitutive,
    const std::array<double, NODES>& thickness, const std::array<std::array<double, 2>, NODES>& fibres,
    const PlateThermalStrain& thermal)
{
    const std::array<PlanePoint<double>, POINT_COUNT> points = planePointsOf(geometry, constitutive, thickness);
    const EnhancedParameters parameters = enhancedParametersOf(constitutive, points, thermal);
    const std::array<double, DOFS> homothety = director_shell::homothetyOf(geometry.position);
    const double dilatation = thermal.isZero ? 0.0 : dilatationOf(thermal);
    ModifiedShears shears;
    if (constitutive.hasBending)
        shears = modifiedShearsOf(geometry, thickness, points);

    std::array<PointOperator, NODES> corners{};
    for (std::size_t p = 0; p < POINT_COUNT; ++p)
    {
        const PlanePoint<double>& point = points[p];
        const Shape shape = shapeAt(point.xi, point.eta);
        const double t = interpolated(shape, thickness);
        std::array<double, 2> fibresHere{};
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t f = 0; f < 2; ++f)
                fibresHere[f] += shape.value[i] * fibres[i][f];
        // The section in the element axes: the strains were turned to them.
        const PointSection<double> section = sectionAt(constitutive, t, 0.0, 4.0 * point.frame.area);
        PointOperator atPoint;
        setPlaneRows(atPoint, strainsAt(point, parameters, thermal, homothety), constitutive, section, t, fibresHere);
        if (constitutive.hasBending)
            setShearRows(atPoint, shears, points, p, homothety, dilatation);
        for (std::size_t corner = 0; corner < NODES; ++corner)
        {
            const double weight = extrapolationWeight(corner, point.xi, point.eta);
            for (std::size_t r = 0; r < CENTRE_RECOVERY_ROWS; ++r)
                for (std::size_t k = 0; k <= DOFS; ++k)
                    corners[corner](r, k) += weight * atPoint(r, k);
        }
    }
    return corners;
}

} // namespace oos::elements
