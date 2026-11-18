#include "oos/elements/shells/shellTria3/ShellTria3.hpp"

#include "../EdgeShearShellSection.hpp"
#include "oos/elements/shells/SecondOrderScalar.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"

#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

using namespace edge_shear_detail;

constexpr std::size_t NODES = ShellTria3::NODES;
constexpr std::size_t DOFS = ShellTria3::DOFS;
template <typename T>
using Row = AffineRow<T, DOFS>;

// Collinear or coincident grid points: ‖(x2 − x1) × (x3 − x1)‖ ≤ this factor · ‖x2 − x1‖·‖x3 − x1‖.
constexpr double DEGENERATE_FACTOR = 1.0e-12;
// The longest side is chosen with ties to the earlier side; sides whose lengths differ by less than this share are a
// tie (the round-off of lengths computed in another order), so that the frame does not flip on it.
constexpr double EDGE_TIE_TOLERANCE = 1.0e-10;
// A singular side fit (its determinant below this relative value) leaves the shear undefined.
constexpr double SINGULAR_FIT_FACTOR = 1.0e-12;
// c_t of the triangle (docs/shell-tria3-formulation.md, section 4): the twist constant of the factors of the
// rotational part of its transverse shear. That part is −½ (θx,x + θy,y), which the linear rotations of a coarse
// mesh cannot keep at zero: a large value (the 99 of the ShellQuad4) stiffens coarse meshes, a small one leaves a
// strip one element wide too soft in torsion.
constexpr double TWIST_FACTOR = 8.0;
// c_n (the same document, section 3): the rotation about the normal of a flat triangle is, for the neighbours it
// meets at an angle, a bending rotation. Its stiffness therefore has, beside the membrane term c_d A33 of the
// ShellQuad4, a term of the order of the bending stiffness of the element, c_n · 12 max(D11, D22) / S. Without it a
// warped surface meshed with triangles softens as the mesh is refined (the twisted beam of section 6).
constexpr double FACET_ROTATION_FACTOR = 0.1;
// 12 D / S: the bending stiffness D of a section over the area S of the element, as in the shear stiffness of a section
// without transverse shear flexibility (EdgeShearShellSection.hpp).
constexpr double BENDING_OVER_AREA = 12.0;
// The mid-sides (½, 0), (½, ½), (0, ½) in (ξ, η): the integration points of the stiffness, of the mass and of the
// thermal loads (weight det/6 each, exact for a polynomial of degree 2), and the interior points (⅙, ⅙), (⅙, ⅔),
// (⅔, ⅙) of the pressure.
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

[[nodiscard]] Vector3<double> unitOf(const Vector3<double>& v)
{
    return scaledBy(v, 1.0 / lengthOf(v));
}

[[nodiscard]] ShellTria3GeometryResult refused(std::string field, std::string problem)
{
    ShellTria3GeometryResult result;
    result.problemField = std::move(field);
    result.problem = std::move(problem);
    return result;
}

// The rows of the membrane strains (εx, εy, γxy) and of the curvatures of ε(z) = e0 + z κ (constant): with the
// rotations moving a fibre by z (θ × n), u(z) = u + z θy and v(z) = v − z θx, so
//   κx = θy,x,  κy = −θx,y,  κxy = θy,y − θx,x.
[[nodiscard]] ScalarMatrix<double, 6, DOFS> planeRows(const ShellTria3Geometry& geometry)
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
        b(3, u + 4) = dx;
        b(4, u + 3) = -dy;
        b(5, u + 3) = -dx;
        b(5, u + 4) = dy;
    }
    return b;
}

// The tied transverse shear: rows [γx; γy; q] = edgeFit · [γ_t(1); γ_t(2); γ_t(3)], γ_t(p) the tangential shear
// c_p γxz + s_p γyz at the middle of side p, with γxz = w,x + θy and γyz = w,y − θx. The field is
// γ(x, y) = (γx − q y, γy + q x), x and y from the centroid: γx, γy its mean, q its rotational part.
[[nodiscard]] ScalarMatrix<double, 3, DOFS> tiedShearRows(const ShellTria3Geometry& geometry)
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

// The plane coordinates of a point of shapes n, from the centroid.
struct PlanePosition
{
    double x = 0.0;
    double y = 0.0;
};

[[nodiscard]] PlanePosition positionOf(const ShellTria3Geometry& geometry, const std::array<double, 3>& n)
{
    return {n[0] * geometry.x[0] + n[1] * geometry.x[1] + n[2] * geometry.x[2],
            n[0] * geometry.y[0] + n[1] * geometry.y[1] + n[2] * geometry.y[2]};
}

// The factors of the transverse shear of a point, s(z) = z / (1 + z), on the widths a and b of the parallelogram:
// the mean shear takes those of the ShellQuad4, s_x0 = s(12 D*11 / (H11 a²)) and s_y0 = s(12 D*22 / (H22 b²)), which
// make a strip of right triangles bend as a beam along the sides of their right angle; the rotational part takes
// s_x1 = s(12 (D*11 + c_t D*33) / (H11 a²)) and s_y1 = s(12 (D*22 + c_t D*33) / (H22 b²)).
template <typename T>
[[nodiscard]] ShearFactors<T> shearFactorsOf(const ShellTria3Geometry& geometry, const PointSection<T>& section)
{
    const std::array<T, 3> strip = stripBendingOf(section);
    const T alongX = RESIDUAL_BENDING_FACTOR / (section.h(0, 0) * (geometry.widthX * geometry.widthX));
    const T alongY = RESIDUAL_BENDING_FACTOR / (section.h(1, 1) * (geometry.widthY * geometry.widthY));
    ShearFactors<T> factors;
    factors.evenX = saturatedOf(alongX * strip[0]);
    factors.oddX = saturatedOf(alongX * (strip[0] + TWIST_FACTOR * strip[2]));
    factors.evenY = saturatedOf(alongY * strip[1]);
    factors.oddY = saturatedOf(alongY * (strip[1] + TWIST_FACTOR * strip[2]));
    return factors;
}

// The drilling row of a point, r = ω − θz with ω = ½ (v,x − u,y) (the ShellQuad4's, on a flat element whose normals
// are z). Zero for a rigid motion.
[[nodiscard]] std::array<double, DOFS> drillingRow(const ShellTria3Geometry& geometry, const std::array<double, 3>& n)
{
    std::array<double, DOFS> row{};
    for (std::size_t a = 0; a < NODES; ++a)
    {
        row[6 * a] = -0.5 * geometry.dY[a];
        row[6 * a + 1] = 0.5 * geometry.dX[a];
        row[6 * a + 5] = -n[a];
    }
    return row;
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

    // The parallelogram the triangle is half of: its diagonal is the longest side (ties to the earlier side 1-2, 2-3,
    // 3-1), its corners O (the node opposite that side), A, O + (A − O) + (B − O), B, with A and B the ends of the side
    // in the order of the nodes. The element axes are those of the ShellQuad4 on these corners: x along the bisector
    // of its diagonals, unit(d13) − unit(d24) with d13 = (A − O) + (B − O) and d24 = B − A. On a right triangle x and
    // y are along the sides of the right angle.
    std::size_t longest = 0;
    for (std::size_t p = 1; p < 3; ++p)
        if (length[p] > length[longest] * (1.0 + EDGE_TIE_TOLERANCE))
            longest = p;
    const std::size_t opposite = (longest + 2) % 3;
    const Vector3<double> legA = difference(x[longest], x[opposite]);
    const Vector3<double> legB = difference(x[(longest + 1) % 3], x[opposite]);
    const Vector3<double> elementX = unitOf(difference(unitOf(sumOf(legA, legB)), unitOf(difference(legB, legA))));
    const Vector3<double> elementY = crossOf(z, elementX);
    ShellTria3Geometry geometry;
    // The widths of the parallelogram (ShellQuad4: a = ½ |x2 + x3 − x1 − x4|, b = ½ |y3 + y4 − y1 − y2|).
    geometry.widthX = std::abs(dotOf(legA, elementX));
    geometry.widthY = std::abs(dotOf(legB, elementY));
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
    // The reference direction: G1→G2.
    geometry.userAngle = std::atan2(dotOf(e12, elementY), dotOf(e12, elementX));

    // The plane Jacobian and the derivatives of the linear shapes.
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

    // The side directions and the fit of the shear field γ(x, y) = (γx − q y, γy + q x) to the tangential shears of
    // the mid-sides: rows [c_p, s_p, x_m s_p − y_m c_p] on (γx, γy, q), (x_m, y_m) the middle of side p. The last
    // term is the distance from the centroid to the side, det / (3 L_p).
    Matrix3<double> fit;
    for (std::size_t p = 0; p < 3; ++p)
    {
        const std::size_t start = p;
        const std::size_t end = (p + 1) % 3;
        geometry.edgeCosine[p] = (geometry.x[end] - geometry.x[start]) / length[p];
        geometry.edgeSine[p] = (geometry.y[end] - geometry.y[start]) / length[p];
        const double middleX = 0.5 * (geometry.x[start] + geometry.x[end]);
        const double middleY = 0.5 * (geometry.y[start] + geometry.y[end]);
        fit(p, 0) = geometry.edgeCosine[p];
        fit(p, 1) = geometry.edgeSine[p];
        fit(p, 2) = middleX * geometry.edgeSine[p] - middleY * geometry.edgeCosine[p];
    }
    const double scale = std::sqrt(geometry.determinant);
    if (!(std::abs(determinantOf(fit)) > SINGULAR_FIT_FACTOR * scale))
        return refused("node3",
                       "expected a triangle whose side shears determine its transverse shear, found a "
                       "degenerate one; correct the grid point positions");
    geometry.edgeFit = inverseOf(fit);
    ShellTria3GeometryResult result;
    result.geometry = geometry;
    return result;
}

// K = Σ over the three mid-sides of w (Bᵀ C B + B_sᵀ H B_s + k_d b_dᵀ b_d), w = det/6, as the ShellQuad4
// (docs/shell-quad4-formulation.md, section 9) without enhanced strains, which a constant strain has no use for:
// - B the rows of (e0, κ) and C = [[A, B], [B, D]] at the thickness of the point;
// - B_s the modified transverse shears (section 7 of that document): the mean shear weighted by √s_x0 and √s_y0, its
//   rotational part by √s_x1 and √s_y1,
//     γ'x = √s_x0 γx − √s_x1 q y,  γ'y = √s_y0 γy + √s_y1 q x,
//   with the factors of shearFactorsOf above;
// - b_d the drilling row and k_d = c_d A33 + c_n · 12 max(D11, D22) / S its stiffness, present when the section has
//   both a membrane and a bending stiffness.
template <typename T>
ScalarMatrix<T, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry& geometry,
                                                        const PlateConstitutive<T>& constitutive,
                                                        const std::array<T, NODES>& thickness)
{
    const PlateConstitutive<T>& c = constitutive;
    const ScalarMatrix<double, 6, DOFS> planeReal = planeRows(geometry);
    const ScalarMatrix<double, 3, DOFS> tied = tiedShearRows(geometry);
    ScalarMatrix<T, 6, DOFS> plane;
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t k = 0; k < DOFS; ++k)
            plane(r, k) = planeReal(r, k);
    const double weight = POINT_WEIGHT * geometry.determinant;
    ScalarMatrix<T, DOFS> k;
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const std::array<double, 3> n = linearShape(point[0], point[1]);
        const PointSection<T> section = sectionAt(c, interpolated(n, thickness), 0.0, geometry.determinant);
        addCongruence(k, plane, planeMatrixOf(section), T(weight));
        if (c.hasBending)
        {
            const ShearFactors<T> factors = shearFactorsOf(geometry, section);
            const PlanePosition position = positionOf(geometry, n);
            const T meanX = squareRoot(factors.evenX);
            const T meanY = squareRoot(factors.evenY);
            const T rotationalX = -position.y * squareRoot(factors.oddX);
            const T rotationalY = position.x * squareRoot(factors.oddY);
            ScalarMatrix<T, 2, DOFS> shear;
            for (std::size_t dof = 0; dof < DOFS; ++dof)
            {
                shear(0, dof) = meanX * tied(0, dof) + rotationalX * tied(2, dof);
                shear(1, dof) = meanY * tied(1, dof) + rotationalY * tied(2, dof);
            }
            addCongruence(k, shear, section.h, T(weight));
        }
        if (c.hasMembrane && c.hasBending)
        {
            const std::array<double, DOFS> drilling = drillingRow(geometry, n);
            const T onRotation =
                DRILLING_FACTOR * section.a(2, 2) + (FACET_ROTATION_FACTOR * BENDING_OVER_AREA / geometry.determinant) *
                                                        largerOf(section.d(0, 0), section.d(1, 1));
            const T scale = weight * onRotation;
            for (std::size_t i = 0; i < DOFS; ++i)
            {
                if (drilling[i] == 0.0)
                    continue;
                const T left = scale * drilling[i];
                for (std::size_t j = 0; j < DOFS; ++j)
                    k(i, j) += left * drilling[j];
            }
        }
    }
    return k;
}

template ScalarMatrix<double, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry&, const PlateConstitutive<double>&,
                                                                      const std::array<double, NODES>&);
template ScalarMatrix<Complex, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry&,
                                                                       const PlateConstitutive<Complex>&,
                                                                       const std::array<Complex, NODES>&);
template ScalarMatrix<SecondOrderScalar, ShellTria3::DOFS> ShellTria3::stiffness(
    const Geometry&, const PlateConstitutive<SecondOrderScalar>&, const std::array<SecondOrderScalar, NODES>&);

// As the ShellQuad4 (docs/shell-quad4-formulation.md, section 10), at the mid-sides with w = det/6: the mass moments
// per unit area I0 = ρ t + NSM and I2 = ρ t³/12 of a homogeneous section,
//   M = Σ w (I0 N_uᵀ N_u + I2 N_βᵀ N_β),  β = Σ N_i θ_i × z = Σ N_i (θy, −θx, 0)_i.
// Lumped: m_i = Σ w I0 N_i on the three translations of node i, nothing on the rotations.
template <typename T>
ScalarMatrix<T, ShellTria3::DOFS> ShellTria3::mass(const Geometry& geometry, const T& density,
                                                   const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                   MassFormulation formulation)
{
    ScalarMatrix<T, DOFS> result;
    if (realPart(density) == 0.0 && realPart(nonStructuralMass) == 0.0)
        return result;
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const std::array<double, 3> n = linearShape(point[0], point[1]);
        const T t = interpolated(n, thickness);
        const double weight = POINT_WEIGHT * geometry.determinant;
        const T translational = weight * (density * t + nonStructuralMass);
        const T rotary = weight * (density * t * t * t / 12.0);
        for (std::size_t a = 0; a < NODES; ++a)
        {
            if (formulation == MassFormulation::Lumped)
            {
                for (std::size_t k = 0; k < 3; ++k)
                    result(6 * a + k, 6 * a + k) += translational * n[a];
                continue;
            }
            for (std::size_t b = 0; b < NODES; ++b)
            {
                const double shapes = n[a] * n[b];
                for (std::size_t k = 0; k < 3; ++k)
                    result(6 * a + k, 6 * b + k) += translational * shapes;
                result(6 * a + 3, 6 * b + 3) += rotary * shapes;
                result(6 * a + 4, 6 * b + 4) += rotary * shapes;
            }
        }
    }
    return result;
}

template ScalarMatrix<double, ShellTria3::DOFS> ShellTria3::mass(const Geometry&, const double&, const double&,
                                                                 const std::array<double, NODES>&, MassFormulation);
template ScalarMatrix<Complex, ShellTria3::DOFS> ShellTria3::mass(const Geometry&, const Complex&, const Complex&,
                                                                  const std::array<Complex, NODES>&, MassFormulation);
template ScalarMatrix<SecondOrderScalar, ShellTria3::DOFS> ShellTria3::mass(
    const Geometry&, const SecondOrderScalar&, const SecondOrderScalar&,
    const std::array<SecondOrderScalar, NODES>&, MassFormulation);

// The values at the centroid (ShellCentreRecovery.hpp), with the conventions of the ShellQuad4 (ShellQuad4Recovery):
// - strains: e0 = B_m q and the curvature of the tables, κ = −B_b q (the tables write ε(z) = ε_m − z κ);
// - forces: N = A (e0 − ε_th) + B (κ_d − κ_th) and M_d = B (e0 − ε_th) + D (κ_d − κ_th), κ_d the curvature of
//   ε(z) = e0 + z κ_d and ε_th + z κ_th the free thermal strain; the moments of the tables are −M_d (M = −∫σ z dz);
// - transverse shear forces: Q = S H S γ with S = diag(√s_x0, √s_y0) and γ the mean shear, the forces conjugate to γ
//   in the energy of the modified shears;
// - stresses at the fibre Z (ζ = 2 Z / t): σ = (G1 − 6 ζ G4)(e0 − ε_th) + (G2 − 6 ζ G4) Z (κ_d − κ_th).
// The section is taken at the mean thickness of the three nodes.
template <typename T>
CentreRecoveryOperator<T, ShellTria3::DOFS> ShellTria3::recovery(const Geometry& geometry,
                                                                 const PlateConstitutive<T>& constitutive,
                                                                 const std::array<T, NODES>& thickness,
                                                                 const std::array<T, 2>& fibres,
                                                                 const PlateThermalStrain& thermal)
{
    const PlateConstitutive<T>& c = constitutive;
    const ScalarMatrix<double, 6, DOFS> rows = planeRows(geometry);
    const T t = (thickness[0] + thickness[1] + thickness[2]) / 3.0;
    const PointSection<T> section = sectionAt(c, t, 0.0, geometry.determinant);

    // The mechanical strains as affine rows: the thermal strain in the last column, subtracted.
    std::array<Row<T>, 3> membrane{};
    std::array<Row<T>, 3> curvature{};
    for (std::size_t r = 0; r < 3; ++r)
    {
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            membrane[r][k] = T(rows(r, k));
            curvature[r][k] = T(rows(3 + r, k));
        }
        membrane[r][DOFS] = T(-thermal.membrane[r]);
        curvature[r][DOFS] = T(-thermal.gradient[r]);
    }

    CentreRecoveryOperator<T, DOFS> recovery;
    for (std::size_t i = 0; i < 3; ++i)
    {
        Row<T> force{};
        Row<T> moment{};
        std::array<Row<T>, 2> stress{};
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
        Row<T> totalMembrane{};
        Row<T> totalCurvature{};
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            totalMembrane[k] = T(rows(i, k));
            totalCurvature[k] = T(-rows(3 + i, k));
        }
        setRow(recovery, centre_row::MEMBRANE_STRAIN + i, totalMembrane);
        setRow(recovery, centre_row::CURVATURE + i, totalCurvature);
    }
    if (c.hasBending)
    {
        const ScalarMatrix<double, 3, DOFS> tied = tiedShearRows(geometry);
        const ShearFactors<T> factors = shearFactorsOf(geometry, section);
        const std::array<T, 2> scale{squareRoot(factors.evenX), squareRoot(factors.evenY)};
        for (std::size_t i = 0; i < 2; ++i)
        {
            Row<T> force{};
            for (std::size_t j = 0; j < 2; ++j)
            {
                const T onShear = scale[i] * section.h(i, j) * scale[j];
                for (std::size_t k = 0; k < DOFS; ++k)
                    force[k] += onShear * tied(j, k);
            }
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
template CentreRecoveryOperator<SecondOrderScalar, ShellTria3::DOFS> ShellTria3::recovery(
    const Geometry&, const PlateConstitutive<SecondOrderScalar>&, const std::array<SecondOrderScalar, NODES>&,
    const std::array<SecondOrderScalar, 2>&, const PlateThermalStrain&);

// F_a = Σ (1/6) p N_a d at the three interior points, d = det along the normal (or det times the unit direction of a
// FacePressureLoad with CID or N), p = Σ N_k P_k in the order of the connectivity. P4 is not used.
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

// f = Σ over the three mid-sides of w Bᵀ S0, w = det/6, S0 = C ε_th the initial resultants of the free thermal
// strain ε_th = (ε_m, gradient), ε(z) = ε_m + z · gradient (docs/shell-quad4-formulation.md, section 12): a free
// thermal expansion leaves the element without stress.
std::array<double, ShellTria3::DOFS> ShellTria3::thermalLoads(const Geometry& geometry,
                                                              const PlateConstitutive<double>& constitutive,
                                                              const std::array<double, NODES>& thickness,
                                                              const PlateThermalStrain& thermal)
{
    std::array<double, DOFS> loads{};
    if (thermal.isZero)
        return loads;
    const ScalarMatrix<double, 6, DOFS> rows = planeRows(geometry);
    const double weight = POINT_WEIGHT * geometry.determinant;
    std::array<double, 6> strain{};
    for (std::size_t r = 0; r < 3; ++r)
    {
        strain[r] = thermal.membrane[r];
        strain[3 + r] = thermal.gradient[r];
    }
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const double t = interpolated(linearShape(point[0], point[1]), thickness);
        const PointSection<double> section = sectionAt(constitutive, t, 0.0, geometry.determinant);
        const std::array<double, 6> resultants = productOf(planeMatrixOf(section), strain);
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < 6; ++r)
                sum += rows(r, k) * resultants[r];
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
