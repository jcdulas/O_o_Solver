#include "oos/elements/shells/shellTria3/ShellTria3.hpp"

#include "../DirectorShellKinematics.hpp"
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
using director_shell::Frame;

constexpr std::size_t NODES = ShellTria3::NODES;
constexpr std::size_t DOFS = ShellTria3::DOFS;
template <typename T>
using Affine = AffineRow<T, DOFS>;
template <typename T>
using Row = director_shell::Row<T, NODES>;
using Shape = director_shell::Shape<NODES>;

// Collinear or coincident grid points: ‖(x2 − x1) × (x3 − x1)‖ ≤ this factor · ‖x2 − x1‖·‖x3 − x1‖.
constexpr double DEGENERATE_FACTOR = 1.0e-12;
// The longest side is chosen with ties to the earlier side; sides whose lengths differ by less than this share are a
// tie (the round-off of lengths computed in another order), so that the frame does not flip on it.
constexpr double EDGE_TIE_TOLERANCE = 1.0e-10;
// c_t of the triangle (docs/shell-tria3-formulation.md, section 4): the twist constant of the factors of the
// rotational part of its transverse shear. That part is −½ (θx,x + θy,y), which the linear rotations of a coarse
// mesh cannot keep at zero: a large value (the 99 of the ShellQuad4) stiffens coarse meshes, a small one leaves a
// strip one element wide too soft in torsion.
constexpr double TWIST_FACTOR = 8.0;
// The mid-sides (½, 0), (½, ½), (0, ½) in (ξ, η): the integration points of the stiffness, of the mass and of the
// thermal loads (weight 1/6 of the area Jacobian each, exact for a polynomial of degree 2), and the interior points
// (⅙, ⅙), (⅙, ⅔), (⅔, ⅙) of the pressure.
constexpr std::array<std::array<double, 2>, 3> EDGE_MIDPOINTS{{{0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}}};
constexpr std::array<std::array<double, 2>, 3> INTERIOR_POINTS{
    {{1.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 3.0}, {2.0 / 3.0, 1.0 / 6.0}}};
constexpr double POINT_WEIGHT = 1.0 / 6.0;
constexpr double CENTROID = 1.0 / 3.0;

// The linear shapes N1 = 1 − ξ − η, N2 = ξ, N3 = η and their derivatives.
[[nodiscard]] Shape shapeAt(double xi, double eta)
{
    Shape shape;
    shape.value = {1.0 - xi - eta, xi, eta};
    shape.dXi = {-1.0, 1.0, 0.0};
    shape.dEta = {-1.0, 0.0, 1.0};
    return shape;
}

template <typename T>
[[nodiscard]] T interpolated(const Shape& shape, const std::array<T, NODES>& nodal)
{
    return shape.value[0] * nodal[0] + shape.value[1] * nodal[1] + shape.value[2] * nodal[2];
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

[[nodiscard]] Frame frameAt(const ShellTria3Geometry& geometry, const Shape& shape)
{
    return director_shell::frameAt(geometry.position, shape);
}

// The rows of (e0, κ) at a point (DirectorShellKinematics.hpp): constant over a triangle whose directors are its
// normal, linear where they turn across it.
template <typename T>
[[nodiscard]] ScalarMatrix<T, 6, DOFS> planeRowsAt(const ShellTria3Geometry& geometry,
                                                   const std::array<T, NODES>& thickness, const Shape& shape,
                                                   const Frame& frame)
{
    return director_shell::planeRowsAt(geometry.position, geometry.normal, thickness, shape, frame);
}

// The covariant transverse shears tied at the mid-sides: γ̃_ξ = a + c η and γ̃_η = b − c ξ, the field of lowest order
// whose component along each side is constant on it (Lee and Bathe, 2004). a is γ̃_ξ at the middle of side 1-2, b is
// γ̃_η at the middle of side 3-1, and c follows from the shear along side 2-3 at its middle, γ̃_η − γ̃_ξ = b − a − c.
template <typename T>
struct TiedShears
{
    Row<T> a{};
    Row<T> b{};
    Row<T> c{};
};

template <typename T>
[[nodiscard]] TiedShears<T> tiedShearsOf(const ShellTria3Geometry& geometry, const std::array<T, NODES>& thickness)
{
    const auto covariantAtPoint = [&](double xi, double eta)
    { return director_shell::covariantShearAt(geometry.position, geometry.normal, thickness, shapeAt(xi, eta)); };
    TiedShears<T> tied;
    tied.a = covariantAtPoint(0.5, 0.0)[0];
    tied.b = covariantAtPoint(0.0, 0.5)[1];
    const std::array<Row<T>, 2> oblique = covariantAtPoint(0.5, 0.5);
    for (std::size_t dof = 0; dof < DOFS; ++dof)
        tied.c[dof] = tied.b[dof] - tied.a[dof] - (oblique[1][dof] - oblique[0][dof]);
    return tied;
}

// The local transverse shears γx, γy of a point from the tied strains (document section 7 of the ShellQuad4).
template <typename T>
[[nodiscard]] std::array<Row<T>, 2> shearRowsAt(const ShellTria3Geometry& geometry,
                                               const std::array<T, NODES>& thickness, const TiedShears<T>& tied,
                                               double xi, double eta)
{
    std::array<Row<T>, 2> covariant{};
    for (std::size_t dof = 0; dof < DOFS; ++dof)
    {
        covariant[0][dof] = tied.a[dof] + eta * tied.c[dof];
        covariant[1][dof] = tied.b[dof] - xi * tied.c[dof];
    }
    const Shape shape = shapeAt(xi, eta);
    return director_shell::localShearsOf(geometry.position, geometry.normal, thickness, shape,
                                         frameAt(geometry, shape), covariant);
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

// The mean in-plane strain of a free thermal strain: its dilatation.
[[nodiscard]] double dilatationOf(const PlateThermalStrain& thermal)
{
    return 0.5 * (thermal.membrane[0] + thermal.membrane[1]);
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
    const Vector3<double> elementX = director_shell::unitOf(
        difference(director_shell::unitOf(sumOf(legA, legB)), director_shell::unitOf(difference(legB, legA))));
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
        geometry.position[a] = {dotOf(fromCentre, elementX), dotOf(fromCentre, elementY), 0.0};
        geometry.normal[a] = {0.0, 0.0, 1.0};
    }
    // The reference direction: G1→G2.
    geometry.userAngle = std::atan2(dotOf(e12, elementY), dotOf(e12, elementX));

    // The plane Jacobian and the derivatives of the linear shapes.
    const double ax = geometry.position[1][0] - geometry.position[0][0];
    const double ay = geometry.position[1][1] - geometry.position[0][1];
    const double bx = geometry.position[2][0] - geometry.position[0][0];
    const double by = geometry.position[2][1] - geometry.position[0][1];
    geometry.determinant = ax * by - ay * bx;
    const std::array<double, 3> dXi{-1.0, 1.0, 0.0};
    const std::array<double, 3> dEta{-1.0, 0.0, 1.0};
    for (std::size_t a = 0; a < 3; ++a)
    {
        geometry.dX[a] = (by * dXi[a] - ay * dEta[a]) / geometry.determinant;
        geometry.dY[a] = (-bx * dXi[a] + ax * dEta[a]) / geometry.determinant;
    }
    ShellTria3GeometryResult result;
    result.geometry = geometry;
    return result;
}

std::optional<std::string> ShellTria3::integrationProblem(const Geometry& geometry,
                                                          const std::array<double, NODES>& thickness)
{
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
        for (const double zeta : {-1.0, 1.0})
        {
            const Matrix3<double> j = director_shell::covariantAt(geometry.position, geometry.normal, thickness,
                                                                  shapeAt(point[0], point[1]), zeta);
            if (!(determinantOf(j) > 0.0))
                return std::string("expected a positive Jacobian at every integration point, found one not positive "
                                   "(a thickness large against the element where the normals of its grid points "
                                   "turn); remesh the element or reduce PARAM ShellNormalAngle");
        }
    return std::nullopt;
}

// K = Σ over the three mid-sides of w (Bᵀ C B + B_sᵀ H B_s + c_d A33 b_dᵀ b_d), w = j/6, as the ShellQuad4
// (docs/shell-quad4-formulation.md, section 9) without enhanced strains, which a constant strain has no use for:
// - B the rows of (e0, κ) and C = [[A, B], [B, D]] at the thickness of the point;
// - B_s the modified transverse shears (section 7 of that document): the mean shear of the element weighted by
//   √s_x0 and √s_y0, the rest (its rotational part, which vanishes at the centroid) by √s_x1 and √s_y1,
//     γ'x = √s_x0 γ̄x + √s_x1 (γx − γ̄x),  γ'y = √s_y0 γ̄y + √s_y1 (γy − γ̄y),
//   with the factors of shearFactorsOf above;
// - b_d the drilling row, present when the section has both a membrane and a bending stiffness.
template <typename T>
ScalarMatrix<T, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry& geometry,
                                                        const PlateConstitutive<T>& constitutive,
                                                        const std::array<T, NODES>& thickness)
{
    const PlateConstitutive<T>& c = constitutive;
    std::array<std::array<Row<T>, 2>, 3> shears{};
    std::array<Row<T>, 2> meanShear{};
    if (c.hasBending)
    {
        const TiedShears<T> tied = tiedShearsOf(geometry, thickness);
        for (std::size_t p = 0; p < 3; ++p)
        {
            shears[p] = shearRowsAt(geometry, thickness, tied, EDGE_MIDPOINTS[p][0], EDGE_MIDPOINTS[p][1]);
            for (std::size_t a = 0; a < 2; ++a)
                for (std::size_t dof = 0; dof < DOFS; ++dof)
                    meanShear[a][dof] += shears[p][a][dof] / 3.0;
        }
    }
    ScalarMatrix<T, DOFS> k;
    for (std::size_t p = 0; p < 3; ++p)
    {
        const Shape shape = shapeAt(EDGE_MIDPOINTS[p][0], EDGE_MIDPOINTS[p][1]);
        const Frame frame = frameAt(geometry, shape);
        const double weight = POINT_WEIGHT * frame.area;
        const PointSection<T> section = sectionAt(c, interpolated(shape, thickness), 0.0, frame.area);
        addCongruence(k, planeRowsAt(geometry, thickness, shape, frame), planeMatrixOf(section), T(weight));
        if (c.hasBending)
        {
            const ShearFactors<T> factors = shearFactorsOf(geometry, section);
            const std::array<T, 2> onMean{squareRoot(factors.evenX), squareRoot(factors.evenY)};
            const std::array<T, 2> onRest{squareRoot(factors.oddX), squareRoot(factors.oddY)};
            ScalarMatrix<T, 2, DOFS> shear;
            for (std::size_t a = 0; a < 2; ++a)
                for (std::size_t dof = 0; dof < DOFS; ++dof)
                    shear(a, dof) =
                        onMean[a] * meanShear[a][dof] + onRest[a] * (shears[p][a][dof] - meanShear[a][dof]);
            addCongruence(k, shear, section.h, T(weight));
        }
        if (c.hasMembrane && c.hasBending && c.drillingFactor > 0.0)
        {
            const Row<T> drilling =
                director_shell::drillingRowAt(geometry.position, geometry.normal, thickness, shape, frame);
            const T scale = (c.drillingFactor * weight) * section.a(2, 2);
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
    return k;
}

double ShellTria3::drillingEnergy(const Geometry& geometry, const PlateConstitutive<double>& constitutive,
                                  const std::array<double, NODES>& thickness, const std::array<double, DOFS>& q)
{
    if (!constitutive.hasMembrane || !constitutive.hasBending || !(constitutive.drillingFactor > 0.0))
        return 0.0;
    double energy = 0.0;
    for (std::size_t p = 0; p < 3; ++p)
    {
        const Shape shape = shapeAt(EDGE_MIDPOINTS[p][0], EDGE_MIDPOINTS[p][1]);
        const Frame frame = frameAt(geometry, shape);
        const PointSection<double> section =
            sectionAt(constitutive, interpolated(shape, thickness), 0.0, frame.area);
        const Row<double> drilling =
            director_shell::drillingRowAt(geometry.position, geometry.normal, thickness, shape, frame);
        double rotation = 0.0; // r = ω − θ_n at the point
        for (std::size_t i = 0; i < DOFS; ++i)
            rotation += drilling[i] * q[i];
        energy += 0.5 * constitutive.drillingFactor * POINT_WEIGHT * frame.area * section.a(2, 2) * rotation * rotation;
    }
    return energy;
}

template ScalarMatrix<double, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry&, const PlateConstitutive<double>&,
                                                                      const std::array<double, NODES>&);
template ScalarMatrix<Complex, ShellTria3::DOFS> ShellTria3::stiffness(const Geometry&,
                                                                       const PlateConstitutive<Complex>&,
                                                                       const std::array<Complex, NODES>&);
template ScalarMatrix<SecondOrderScalar, ShellTria3::DOFS> ShellTria3::stiffness(
    const Geometry&, const PlateConstitutive<SecondOrderScalar>&, const std::array<SecondOrderScalar, NODES>&);

// As the ShellQuad4 (docs/shell-quad4-formulation.md, section 10), at the mid-sides with w = j/6: the mass moments
// per unit area I0 = ρ t + NSM and I2 = ρ t³/12 of a homogeneous section,
//   M = Σ w (I0 N_uᵀ N_u + I2 N_βᵀ N_β),  β = Σ N_i θ_i × n_i the director increment.
// Lumped: m_i = Σ w I0 N_i on the three translations of node i, nothing on the rotations.
template <typename T>
ScalarMatrix<T, ShellTria3::DOFS> ShellTria3::mass(const Geometry& geometry, const T& density,
                                                   const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                   MassFormulation formulation)
{
    ScalarMatrix<T, DOFS> result;
    if (realPart(density) == 0.0 && realPart(nonStructuralMass) == 0.0)
        return result;
    // θ × n = S θ
    std::array<std::array<std::array<double, 3>, 3>, NODES> skew{};
    for (std::size_t a = 0; a < NODES; ++a)
    {
        const Vector3<double>& n = geometry.normal[a];
        skew[a] = {{{0.0, n[2], -n[1]}, {-n[2], 0.0, n[0]}, {n[1], -n[0], 0.0}}};
    }
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const Shape shape = shapeAt(point[0], point[1]);
        const T t = interpolated(shape, thickness);
        const double weight = POINT_WEIGHT * geometry.determinant;
        const T translational = weight * (density * t + nonStructuralMass);
        const T rotary = weight * (density * t * t * t / 12.0);
        for (std::size_t a = 0; a < NODES; ++a)
        {
            if (formulation == MassFormulation::Lumped)
            {
                for (std::size_t k = 0; k < 3; ++k)
                    result(6 * a + k, 6 * a + k) += translational * shape.value[a];
                continue;
            }
            for (std::size_t b = 0; b < NODES; ++b)
            {
                const double shapes = shape.value[a] * shape.value[b];
                for (std::size_t k = 0; k < 3; ++k)
                    result(6 * a + k, 6 * b + k) += translational * shapes;
                for (std::size_t r = 0; r < 3; ++r)
                    for (std::size_t s = 0; s < 3; ++s)
                    {
                        double product = 0.0;
                        for (std::size_t m = 0; m < 3; ++m)
                            product += skew[a][m][r] * skew[b][m][s];
                        if (product != 0.0)
                            result(6 * a + 3 + r, 6 * b + 3 + s) += rotary * (shapes * product);
                    }
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
//   ε(z) = e0 + z κ_d and ε_th + z κ_th the free thermal strain, its dilatation taken as the strains of the homothety
//   of the element (DirectorShellKinematics.hpp); the moments of the tables are −M_d (M = −∫σ z dz);
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
    const Shape shape = shapeAt(CENTROID, CENTROID);
    const Frame frame = frameAt(geometry, shape);
    const ScalarMatrix<T, 6, DOFS> rows = planeRowsAt(geometry, thickness, shape, frame);
    const T t = interpolated(shape, thickness);
    const PointSection<T> section = sectionAt(c, t, 0.0, frame.area);

    // The mechanical strains as affine rows: the thermal strain in the last column, subtracted.
    const std::array<T, 6> homothety = director_shell::dilatationStrainsOf(geometry.position, rows);
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
        // the tied field is linear: its value at the centroid is its mean
        const TiedShears<T> tied = tiedShearsOf(geometry, thickness);
        const std::array<Row<T>, 2> shear = shearRowsAt(geometry, thickness, tied, CENTROID, CENTROID);
        const ShearFactors<T> factors = shearFactorsOf(geometry, section);
        const std::array<T, 2> scale{squareRoot(factors.evenX), squareRoot(factors.evenY)};
        for (std::size_t i = 0; i < 2; ++i)
        {
            Affine<T> force{};
            for (std::size_t j = 0; j < 2; ++j)
            {
                const T onShear = scale[i] * section.h(i, j) * scale[j];
                for (std::size_t k = 0; k < DOFS; ++k)
                    force[k] += onShear * shear[j][k];
            }
            // the shear of the free dilatation (directors that are not normal to the element), subtracted
            for (std::size_t k = 0; k < DOFS; ++k)
                force[DOFS] -= dilatation * force[k] * homothetyDofs[k];
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
        const Shape shape = shapeAt(point[0], point[1]);
        const double p = shape.value[0] * cornerPressures[0] + shape.value[1] * cornerPressures[1] +
                         shape.value[2] * cornerPressures[2];
        for (std::size_t a = 0; a < NODES; ++a)
            for (std::size_t k = 0; k < 3; ++k)
                loads[6 * a + k] += POINT_WEIGHT * p * shape.value[a] * geometry.determinant * along[k];
    }
    return loads;
}

// f = Σ over the three mid-sides of w Bᵀ S0, w = j/6, S0 = C ε_th the initial resultants of the free thermal strain
// ε_th = (ε_m, gradient), ε(z) = ε_m + z · gradient (docs/shell-quad4-formulation.md, section 12), less its dilatation
// (its mean in-plane strain), whose loads are K q for the homothety q of the element (addDilatationLoads): a free
// thermal expansion leaves the element without stress, whatever its directors.
std::array<double, ShellTria3::DOFS> ShellTria3::thermalLoads(const Geometry& geometry,
                                                              const PlateConstitutive<double>& constitutive,
                                                              const std::array<double, NODES>& thickness,
                                                              const PlateThermalStrain& thermal)
{
    std::array<double, DOFS> loads{};
    if (thermal.isZero)
        return loads;
    const double dilatation = dilatationOf(thermal);
    for (const std::array<double, 2>& point : EDGE_MIDPOINTS)
    {
        const Shape shape = shapeAt(point[0], point[1]);
        const Frame frame = frameAt(geometry, shape);
        const ScalarMatrix<double, 6, DOFS> rows = planeRowsAt(geometry, thickness, shape, frame);
        std::array<double, 6> strain{};
        for (std::size_t r = 0; r < 3; ++r)
        {
            strain[r] = thermal.membrane[r] - (r < 2 ? dilatation : 0.0);
            strain[3 + r] = thermal.gradient[r];
        }
        const PointSection<double> section = sectionAt(constitutive, interpolated(shape, thickness), 0.0, frame.area);
        const std::array<double, 6> resultants = productOf(planeMatrixOf(section), strain);
        const double weight = POINT_WEIGHT * frame.area;
        for (std::size_t k = 0; k < DOFS; ++k)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < 6; ++r)
                sum += rows(r, k) * resultants[r];
            loads[k] += weight * sum;
        }
    }
    director_shell::addDilatationLoads(loads, stiffness(geometry, constitutive, thickness), geometry.position,
                                       dilatation);
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
