#include "oos/elements/shells/shellTria3/ShellTria3.hpp"

#include "oos/elements/shells/ShellDifferential.hpp"

#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

constexpr std::size_t NODES = ShellTria3::NODES;
constexpr std::size_t DOFS = ShellTria3::DOFS;
template <typename T>
using Row = AffineRow<T, DOFS>;

// Collinear or coincident grid points: ‖(x2 − x1) × (x3 − x1)‖ ≤ this factor · ‖x2 − x1‖·‖x3 − x1‖.
constexpr double DEGENERATE_FACTOR = 1.0e-12;
// Spec §3.2: the shortest edge is chosen with ties to the earlier edge; edges whose lengths differ by less than this
// share are a tie (the round-off of lengths computed in another order), so that the frame does not flip on it.
constexpr double EDGE_TIE_TOLERANCE = 1.0e-10;
// A singular edge fit (its determinant below this relative value) leaves the shear undefined.
constexpr double SINGULAR_FIT_FACTOR = 1.0e-12;
// Spec §3.5: the edge midpoints (½, 0), (½, ½), (0, ½), the stiffness, mass and thermal integration points (weight
// det/6 each), and the interior points (⅙, ⅙), (⅙, ⅔), (⅔, ⅙) of the pressure.
constexpr std::array<std::array<double, 2>, 3> EDGE_MIDPOINTS{{{0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}}};
constexpr std::array<std::array<double, 2>, 3> INTERIOR_POINTS{
    {{1.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 3.0}, {2.0 / 3.0, 1.0 / 6.0}}};
constexpr double POINT_WEIGHT = 1.0 / 6.0;

// The linear shapes N1 = 1 − ξ − η, N2 = ξ, N3 = η.
[[nodiscard]] std::array<double, 3> linearShape(double xi, double eta)
{
    return {1.0 - xi - eta, xi, eta};
}

template <typename T>
[[nodiscard]] T interpolated(const std::array<double, 3>& shape, const std::array<T, 3>& nodal)
{
    return shape[0] * nodal[0] + shape[1] * nodal[1] + shape[2] * nodal[2];
}

[[nodiscard]] Vector3<double> vectorOf(const base::Vec3& v)
{
    return {v[0], v[1], v[2]};
}

[[nodiscard]] Vector3<double> difference(const Vector3<double>& a, const Vector3<double>& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

[[nodiscard]] double lengthOf(const Vector3<double>& v)
{
    return std::sqrt(dotOf(v, v));
}

[[nodiscard]] ShellTria3GeometryResult refused(std::string field, std::string problem)
{
    ShellTria3GeometryResult result;
    result.problemField = std::move(field);
    result.problem = std::move(problem);
    return result;
}

// Spec §3.5, rows 1 to 6 of B (constant): the membrane strains (εx, εy, γxy) and the NASTRAN-95 curvatures
// κx = −N,x θy, κy = N,y θx, κxy = N,x θx − N,y θy (the shell curvature of ShellRecovery.hpp, ε(z) = ε_m − z κ).
[[nodiscard]] ScalarMatrix<double, 6, DOFS> inPlaneRows(const ShellTria3Geometry& geometry)
{
    ScalarMatrix<double, 6, DOFS> b;
    for (std::size_t a = 0; a < NODES; ++a)
    {
        const std::size_t u = 6 * a;
        const double dx = geometry.dX[a];
        const double dy = geometry.dY[a];
        b(0, u) = dx;
        b(1, u + 1) = dy;
        b(2, u) = dy;
        b(2, u + 1) = dx;
        b(3, u + 4) = -dx;
        b(4, u + 3) = dy;
        b(5, u + 3) = dx;
        b(5, u + 4) = -dy;
    }
    return b;
}

// Spec §3.6: the assumed shear rows [γxz⁰; γyz⁰; q] = edgeFit · [γ_t(1); γ_t(2); γ_t(3)], γ_t(p) the tangential shear
// c_p γxz + s_p γyz at the midpoint of edge p, with γxz = N,x w + N θy and γyz = N,y w − N θx (rows 7, 8 of §3.5).
[[nodiscard]] ScalarMatrix<double, 3, DOFS> assumedShearRows(const ShellTria3Geometry& geometry)
{
    ScalarMatrix<double, 3, DOFS> tangential;
    for (std::size_t p = 0; p < 3; ++p)
    {
        const std::array<double, 3> n = linearShape(EDGE_MIDPOINTS[p][0], EDGE_MIDPOINTS[p][1]);
        const double c = geometry.edgeCosine[p];
        const double s = geometry.edgeSine[p];
        for (std::size_t a = 0; a < NODES; ++a)
        {
            const std::size_t u = 6 * a;
            tangential(p, u + 2) = c * geometry.dX[a] + s * geometry.dY[a];
            tangential(p, u + 3) = -s * n[a];
            tangential(p, u + 4) = c * n[a];
        }
    }
    return productOf(geometry.edgeFit, tangential);
}

// Spec §3.7: the shear compliance Z on (γxz⁰, γyz⁰, q) at a point of thickness t, in this implementation's order (x
// first; NASTRAN-95 orders γyz first). The transverse shear compliance (κ G)⁻¹/t of MID3 adds to a bending relief
// compliance: L_x²/(24 I G2xx) on γxz, L_y²/(24 I G2yy) on γyz, and on q the mean of the shear compliances plus
// L_x L_y / (4 I √(G2xx G2yy)).
template <typename T>
[[nodiscard]] Matrix3<T> shearCompliance(const ShellTria3Geometry& geometry, const PlateConstitutive<T>& c, const T& t)
{
    const T inertia = c.bendingRatio * t * t * t / 12.0;
    const T& gxx = c.bending(0, 0);
    const T& gyy = c.bending(1, 1);
    const bool hasRelief = realPart(gxx) * realPart(gyy) != 0.0;
    const double lx = geometry.lengthX;
    const double ly = geometry.lengthY;
    const Matrix2<T>& shear = c.transverseShearCompliance;
    Matrix3<T> z;
    z(0, 0) = shear(0, 0) / t + (hasRelief ? lx * lx / (24.0 * inertia * gxx) : T(0.0));
    z(1, 1) = shear(1, 1) / t + (hasRelief ? ly * ly / (24.0 * inertia * gyy) : T(0.0));
    z(0, 1) = shear(0, 1) / t;
    z(1, 0) = z(0, 1);
    z(2, 2) = 0.5 * (shear(0, 0) + shear(1, 1)) / t +
              (hasRelief ? lx * ly / (4.0 * inertia * squareRoot(gxx * gyy)) : T(0.0));
    return z;
}

} // namespace

ShellTria3GeometryResult buildShellTria3Geometry(const std::array<base::Vec3, 3>& positions)
{
    std::array<Vector3<double>, 3> x{};
    for (std::size_t a = 0; a < 3; ++a)
        x[a] = vectorOf(positions[a]);
    const Vector3<double> e12 = difference(x[1], x[0]);
    const Vector3<double> e23 = difference(x[2], x[1]);
    const Vector3<double> e13 = difference(x[2], x[0]);
    const std::array<double, 3> length{lengthOf(e12), lengthOf(e23), lengthOf(e13)};
    const Vector3<double> normal = crossOf(e12, e13);
    const double twiceArea = lengthOf(normal);
    if (!(twiceArea > DEGENERATE_FACTOR * length[0] * length[2]))
        return refused("node3",
                       "expected a triangle, found ShellTria3.node1, ShellTria3.node2 and "
                       "ShellTria3.node3 on a line (or coincident); correct their "
                       "positions");
    const Vector3<double> z = scaledBy(normal, 1.0 / twiceArea);

    // Spec §3.2: x along the median from the vertex opposite the shortest edge (ties to the earlier edge 1-2, 2-3,
    // 3-1).
    std::size_t opposite = 2;
    std::array<std::size_t, 2> ends{0, 1};
    double shortest = length[0];
    if (length[1] < shortest * (1.0 - EDGE_TIE_TOLERANCE))
    {
        shortest = length[1];
        opposite = 0;
        ends = {1, 2};
    }
    if (length[2] < shortest * (1.0 - EDGE_TIE_TOLERANCE))
    {
        opposite = 1;
        ends = {0, 2};
    }
    const Vector3<double> median = difference(scaledBy(sumOf(x[ends[0]], x[ends[1]]), 0.5), x[opposite]);
    ShellTria3Geometry geometry;
    geometry.lengthX = lengthOf(median);
    geometry.lengthY = twiceArea / geometry.lengthX;
    const Vector3<double> elementX = scaledBy(median, 1.0 / geometry.lengthX);
    const Vector3<double> elementY = crossOf(z, elementX);
    for (std::size_t r = 0; r < 3; ++r)
    {
        geometry.axes(r, 0) = elementX[r];
        geometry.axes(r, 1) = elementY[r];
        geometry.axes(r, 2) = z[r];
    }
    Vector3<double> centre{};
    for (std::size_t a = 0; a < 3; ++a)
        centre = sumOf(centre, scaledBy(x[a], 1.0 / 3.0));
    geometry.centre = {centre[0], centre[1], centre[2]};
    for (std::size_t a = 0; a < 3; ++a)
    {
        const Vector3<double> fromCentre = difference(x[a], centre);
        geometry.x[a] = dotOf(fromCentre, elementX);
        geometry.y[a] = dotOf(fromCentre, elementY);
    }
    // The user frame: x along G1→G2.
    geometry.userAngle = std::atan2(dotOf(e12, elementY), dotOf(e12, elementX));

    // Spec §3.5: the plane Jacobian and the derivatives of the linear shapes.
    const double ax = geometry.x[1] - geometry.x[0];
    const double ay = geometry.y[1] - geometry.y[0];
    const double bx = geometry.x[2] - geometry.x[0];
    const double by = geometry.y[2] - geometry.y[0];
    geometry.determinant = ax * by - ay * bx;
    const std::array<double, 3> dXi{-1.0, 1.0, 0.0};
    const std::array<double, 3> dEta{-1.0, 0.0, 1.0};
    for (std::size_t a = 0; a < 3; ++a)
    {
        geometry.dX[a] = (by * dXi[a] - ay * dEta[a]) / geometry.determinant;
        geometry.dY[a] = (-bx * dXi[a] + ax * dEta[a]) / geometry.determinant;
    }

    // Spec §3.6: the edge directions (3-D lengths) and the fit of the constant shear and q to the edge strains, rows
    // [c_p, s_p, 1] on (γxz⁰, γyz⁰, q).
    Matrix3<double> fit;
    for (std::size_t p = 0; p < 3; ++p)
    {
        const std::size_t start = p;
        const std::size_t end = (p + 1) % 3;
        const double edgeLength = p == 0 ? length[0] : (p == 1 ? length[1] : length[2]);
        geometry.edgeCosine[p] = (geometry.x[end] - geometry.x[start]) / edgeLength;
        geometry.edgeSine[p] = (geometry.y[end] - geometry.y[start]) / edgeLength;
        fit(p, 0) = geometry.edgeCosine[p];
        fit(p, 1) = geometry.edgeSine[p];
        fit(p, 2) = 1.0;
    }
    if (!(std::abs(determinantOf(fit)) > SINGULAR_FIT_FACTOR))
        return refused("node3",
                       "expected a triangle whose edge shears determine its transverse shear, found a "
                       "degenerate one; correct the grid point positions");
    geometry.edgeFit = inverseOf(fit);
    ShellTria3GeometryResult result;
    result.geometry = geometry;
    return result;
}

// Spec §3.7: K = Σ over the 3 edge midpoints of Bᵀ G B, B = [rows 1-6; the assumed shear rows], G = [G1 t, G4 t²;
// G4 t², G2 I] w and Z⁻¹ w on the shear, w = det/6, t the thickness at the point and I = (12I/T³) t³/12.
template <typename T>
ScalarMatrix<T, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry& geometry,
                                                        const PlateConstitutive<T>& constitutive,
                                                        const std::array<T, NODES>& thickness)
{
    const PlateConstitutive<T>& c = constitutive;
    const ScalarMatrix<double, 6, DOFS> plane = inPlaneRows(geometry);
    const ScalarMatrix<double, 3, DOFS> shear = assumedShearRows(geometry);
    ScalarMatrix<T, 9, DOFS> b;
    for (std::size_t k = 0; k < DOFS; ++k)
    {
        for (std::size_t r = 0; r < 6; ++r)
            b(r, k) = plane(r, k);
        for (std::size_t r = 0; r < 3; ++r)
            b(6 + r, k) = shear(r, k);
    }
    const double weight = POINT_WEIGHT * geometry.determinant;
    ScalarMatrix<T, DOFS> k;
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const T t = interpolated(linearShape(point[0], point[1]), thickness);
        const T inertia = c.bendingRatio * t * t * t / 12.0;
        ScalarMatrix<T, 9> g;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
            {
                if (c.hasMembrane)
                    g(i, j) = t * c.membrane(i, j);
                if (c.hasBending)
                    g(3 + i, 3 + j) = inertia * c.bending(i, j);
                if (c.hasCoupling)
                {
                    g(i, 3 + j) = t * t * c.coupling(i, j);
                    g(3 + j, i) = g(i, 3 + j);
                }
            }
        if (c.hasBending)
        {
            const Matrix3<T> shearStiffness = inverseOf(shearCompliance(geometry, c, t));
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    g(6 + i, 6 + j) = shearStiffness(i, j);
        }
        addCongruence(k, b, g, T(weight));
    }
    return k;
}

template ScalarMatrix<double, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry&, const PlateConstitutive<double>&,
                                                                      const std::array<double, NODES>&);
template ScalarMatrix<Complex, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry&,
                                                                       const PlateConstitutive<Complex>&,
                                                                       const std::array<Complex, NODES>&);

// Spec §3.8: w_m = (ρ t + NSM) det/6 at the edge midpoints; lumped M_aa = Σ N_a w_m, coupled M_ab = Σ N_a N_b w_m.
// The mass is formed when ρ or NSM is positive (NASTRAN-95's TRIA3 ignores a negative NSM alone). Translations only.
template <typename T>
ScalarMatrix<T, ShellTria3::DOFS> ShellTria3::mass(const Geometry& geometry, const T& density,
                                                   const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                   MassFormulation formulation)
{
    ScalarMatrix<T, DOFS> result;
    if (!(realPart(density) > 0.0) && !(realPart(nonStructuralMass) > 0.0))
        return result;
    ScalarMatrix<T, NODES> scalar;
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const std::array<double, 3> n = linearShape(point[0], point[1]);
        const T areal = density * interpolated(n, thickness) + nonStructuralMass;
        const T weight = areal * (POINT_WEIGHT * geometry.determinant);
        for (std::size_t a = 0; a < NODES; ++a)
        {
            if (formulation == MassFormulation::Lumped)
            {
                scalar(a, a) += weight * n[a];
                continue;
            }
            for (std::size_t b = 0; b < NODES; ++b)
                scalar(a, b) += weight * n[a] * n[b];
        }
    }
    for (std::size_t a = 0; a < NODES; ++a)
        for (std::size_t b = 0; b < NODES; ++b)
            for (std::size_t k = 0; k < 3; ++k)
                result(6 * a + k, 6 * b + k) = scalar(a, b);
    return result;
}

template ScalarMatrix<double, ShellTria3::DOFS> ShellTria3::mass(const Geometry&, const double&, const double&,
                                                                 const std::array<double, NODES>&, MassFormulation);
template ScalarMatrix<Complex, ShellTria3::DOFS> ShellTria3::mass(const Geometry&, const Complex&, const Complex&,
                                                                  const std::array<Complex, NODES>&, MassFormulation);

// Spec §7.3, at the centroid: the strains B u with the constant assumed shear; Q = Z⁻¹ γ (Z without q);
// [N; M] = [[t G1, t² G4], [t² G4, I G2]] [ε_m; κ]; σ(Z) = [G1 − (12 Z/t) G4] ε_m − Z [(12I/T³) G2 − (12 Z/t) G4] κ,
// the thermal strain subtracted from ε_m and κ (deviation from NASTRAN-95, deliberate: its recovery never subtracts
// the thermal curvature, its two TEMPP flags are tested together; and its MID4 terms are read with a wrong stride
// there: the coupling is taken as written above).
template <typename T>
CentreRecoveryOperator<T, ShellTria3::DOFS> ShellTria3::recovery(const Geometry& geometry,
                                                                 const PlateConstitutive<T>& constitutive,
                                                                 const std::array<T, NODES>& thickness,
                                                                 const std::array<T, 2>& fibres,
                                                                 const PlateThermalStrain& thermal)
{
    const PlateConstitutive<T>& c = constitutive;
    const ScalarMatrix<double, 6, DOFS> plane = inPlaneRows(geometry);
    const ScalarMatrix<double, 3, DOFS> shear = assumedShearRows(geometry);
    const T t = (thickness[0] + thickness[1] + thickness[2]) / 3.0;
    const T inertia = c.bendingRatio * t * t * t / 12.0;
    std::array<Row<T>, 3> membraneStrain{};
    std::array<Row<T>, 3> curvature{};
    std::array<Row<T>, 2> gamma{};
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            membraneStrain[r][k] = T(plane(r, k));
            curvature[r][k] = T(plane(3 + r, k));
        }
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t k = 0; k < DOFS; ++k)
            gamma[r][k] = T(shear(r, k));
    // The mechanical strains: the thermal ones subtracted (the free curvature is −gradient in this sign).
    std::array<Row<T>, 3> elasticStrain = membraneStrain;
    std::array<Row<T>, 3> elasticCurvature = curvature;
    for (std::size_t r = 0; r < 3; ++r)
    {
        elasticStrain[r][DOFS] = T(-thermal.membrane[r]);
        elasticCurvature[r][DOFS] = T(thermal.gradient[r]);
    }

    CentreRecoveryOperator<T, DOFS> recovery;
    for (std::size_t i = 0; i < 3; ++i)
    {
        Row<T> force{};
        Row<T> moment{};
        std::array<Row<T>, 2> stress{};
        for (std::size_t j = 0; j < 3; ++j)
        {
            if (c.hasMembrane)
            {
                addScaled(force, elasticStrain[j], t * c.membrane(i, j));
                for (std::size_t f = 0; f < 2; ++f)
                    addScaled(stress[f], elasticStrain[j], c.membrane(i, j));
            }
            if (c.hasBending)
            {
                addScaled(moment, elasticCurvature[j], inertia * c.bending(i, j));
                for (std::size_t f = 0; f < 2; ++f)
                    addScaled(stress[f], elasticCurvature[j], -fibres[f] * c.bendingRatio * c.bending(i, j));
            }
            if (c.hasCoupling)
            {
                addScaled(force, elasticCurvature[j], t * t * c.coupling(i, j));
                addScaled(moment, elasticStrain[j], t * t * c.coupling(i, j));
                for (std::size_t f = 0; f < 2; ++f)
                {
                    const T scale = 12.0 * fibres[f] / t * c.coupling(i, j);
                    addScaled(stress[f], elasticStrain[j], -scale);
                    addScaled(stress[f], elasticCurvature[j], fibres[f] * scale);
                }
            }
        }
        setRow(recovery, centre_row::MEMBRANE_FORCE + i, force);
        setRow(recovery, centre_row::MOMENT + i, moment);
        setRow(recovery, centre_row::STRESS_FIBER1 + i, stress[0]);
        setRow(recovery, centre_row::STRESS_FIBER2 + i, stress[1]);
        setRow(recovery, centre_row::MEMBRANE_STRAIN + i, membraneStrain[i]);
        setRow(recovery, centre_row::CURVATURE + i, curvature[i]);
    }
    if (c.hasBending)
    {
        // Q = Z⁻¹ γ with the 2 × 2 compliance of the constant shear.
        const Matrix3<T> z = shearCompliance(geometry, c, t);
        Matrix2<T> z2;
        z2(0, 0) = z(0, 0);
        z2(0, 1) = z(0, 1);
        z2(1, 0) = z(1, 0);
        z2(1, 1) = z(1, 1);
        const Matrix2<T> stiffness = inverseOf(z2);
        for (std::size_t i = 0; i < 2; ++i)
        {
            Row<T> force{};
            for (std::size_t j = 0; j < 2; ++j)
                addScaled(force, gamma[j], stiffness(i, j));
            setRow(recovery, centre_row::SHEAR_FORCE + i, force);
        }
    }
    return recovery;
}

template CentreRecoveryOperator<double, ShellTria3::DOFS> ShellTria3::recovery(const Geometry&,
                                                                               const PlateConstitutive<double>&,
                                                                               const std::array<double, NODES>&,
                                                                               const std::array<double, 2>&,
                                                                               const PlateThermalStrain&);
template CentreRecoveryOperator<Complex, ShellTria3::DOFS> ShellTria3::recovery(const Geometry&,
                                                                                const PlateConstitutive<Complex>&,
                                                                                const std::array<Complex, NODES>&,
                                                                                const std::array<Complex, 2>&,
                                                                                const PlateThermalStrain&);

// Spec §5.2: F_a = Σ (1/6) p N_a d at the three interior points, d = det along the normal (or det times the unit
// direction of a FacePressureLoad with CID or N), p = Σ N_k P_k. P4 is not used.
// Deviation from NASTRAN-95 (deliberate): NASTRAN-95 pairs the corner pressures with the shape functions of its
// internal node order (sorted by internal DOF number), wrong whenever that order is not G1, G2, G3; the connectivity
// order is used here.
std::array<double, ShellTria3::DOFS> ShellTria3::pressureLoads(const Geometry& geometry,
                                                               const std::array<double, 4>& cornerPressures,
                                                               const std::optional<base::Vec3>& direction)
{
    std::array<double, DOFS> loads{};
    const Vector3<double> along =
        direction ? Vector3<double>{(*direction)[0], (*direction)[1], (*direction)[2]} : Vector3<double>{0.0, 0.0, 1.0};
    for (const std::array<double, 2>& point : INTERIOR_POINTS)
    {
        const std::array<double, 3> n = linearShape(point[0], point[1]);
        const double p = n[0] * cornerPressures[0] + n[1] * cornerPressures[1] + n[2] * cornerPressures[2];
        for (std::size_t a = 0; a < NODES; ++a)
            for (std::size_t k = 0; k < 3; ++k)
                loads[6 * a + k] += POINT_WEIGHT * p * n[a] * geometry.determinant * along[k];
    }
    return loads;
}

// Spec §6.3: P = Σ over the 3 edge midpoints of w Bᵀ G [ε_m; κ_th], w = det/6, the free curvature κ_th = −gradient
// in the NASTRAN-95 curvature (ε(z) = ε_m − z κ).
std::array<double, ShellTria3::DOFS> ShellTria3::thermalLoads(const Geometry& geometry,
                                                              const PlateConstitutive<double>& constitutive,
                                                              const std::array<double, NODES>& thickness,
                                                              const PlateThermalStrain& thermal)
{
    const PlateConstitutive<double>& c = constitutive;
    std::array<double, DOFS> loads{};
    if (thermal.isZero)
        return loads;
    const ScalarMatrix<double, 6, DOFS> plane = inPlaneRows(geometry);
    const double weight = POINT_WEIGHT * geometry.determinant;
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const double t = interpolated(linearShape(point[0], point[1]), thickness);
        const double inertia = c.bendingRatio * t * t * t / 12.0;
        std::array<double, 6> stress{};
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
            {
                const double strain = thermal.membrane[j];
                const double curvature = -thermal.gradient[j];
                if (c.hasMembrane)
                    stress[i] += t * c.membrane(i, j) * strain;
                if (c.hasBending)
                    stress[3 + i] += inertia * c.bending(i, j) * curvature;
                if (c.hasCoupling)
                {
                    stress[i] += t * t * c.coupling(i, j) * curvature;
                    stress[3 + i] += t * t * c.coupling(i, j) * strain;
                }
            }
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < 6; ++r)
                sum += plane(r, k) * stress[r];
            loads[k] += weight * sum;
        }
    }
    return loads;
}

base::MatN<ShellTria3::DOFS> ShellTria3::differentialStiffness(const Geometry& geometry,
                                                               const std::array<double, 3>& membraneForce)
{
    base::MatN<DOFS> k;
    addMembraneDifferentialStiffness<NODES, 6>(k, geometry.dX, geometry.dY, 0.5 * geometry.determinant, membraneForce);
    return k;
}

} // namespace oos::elements
