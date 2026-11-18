#pragma once

// The pieces of the ShellQuad4 kernel that its stiffness, loads and recovery share
// (docs/shell-quad4-formulation.md, "the document" below, sections 3 to 9): the shape functions, the covariant
// vectors and the local frame at a point, the displacement gradient, the membrane strains and curvatures, the tied
// transverse shears with their factors, the drilling term, the section at a point and the enhanced strains.
//
// Every function is a template on the scalar (ShellScalar.hpp): the corner thicknesses and the section may be complex
// or second-order numbers, the grid point positions are real. A row of 24 values is a linear form of the local DOFs
// (u, v, w, θx, θy, θz of each node, node by node, in the element frame).

#include "../EdgeShearShellSection.hpp"
#include "oos/elements/shells/PlateConstitutive.hpp"
#include "oos/elements/shells/ShellScalar.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace oos::elements::quad4_detail
{

using namespace edge_shear_detail;

inline constexpr std::size_t NODES = ShellQuad4::NODES;
inline constexpr std::size_t DOFS = ShellQuad4::DOFS;
// The 2 × 2 Gauss rule: points ±1/√3, weights 1.
inline constexpr double GAUSS_POINT = 0.57735026918962576451;
inline constexpr std::array<double, 2> GAUSS_POINTS{-GAUSS_POINT, GAUSS_POINT};
inline constexpr std::size_t POINT_COUNT = 4;
// The natural coordinates of the nodes.
inline constexpr std::array<double, NODES> NODE_XI{-1.0, 1.0, 1.0, -1.0};
inline constexpr std::array<double, NODES> NODE_ETA{-1.0, -1.0, 1.0, 1.0};
// The parameters of the enhanced strains: four on the membrane strains, four on the curvatures (document section 6).
inline constexpr std::size_t ENHANCED_PER_PART = 4;
inline constexpr std::size_t ENHANCED = 2 * ENHANCED_PER_PART;

// The constants of the document, section 13.
// c_t: the linear part of the transverse shear may relieve the twisting curvature; a thin strip one element wide
// keeps μ / (1 + μ) of its torsional rigidity, μ = c_t + (D11 / D33)(b / a)². c_t = 1 / ε − 1 with ε = 1 %.
inline constexpr double TWIST_FACTOR = 99.0;

template <typename T>
using Row = std::array<T, DOFS>;

struct Shape
{
    std::array<double, NODES> value{};
    std::array<double, NODES> dXi{};
    std::array<double, NODES> dEta{};
};

// N_i = ¼ (1 + ξ ξ_i)(1 + η η_i) and its derivatives.
[[nodiscard]] inline Shape shapeAt(double xi, double eta)
{
    Shape shape;
    for (std::size_t i = 0; i < NODES; ++i)
    {
        shape.value[i] = 0.25 * (1.0 + xi * NODE_XI[i]) * (1.0 + eta * NODE_ETA[i]);
        shape.dXi[i] = 0.25 * NODE_XI[i] * (1.0 + eta * NODE_ETA[i]);
        shape.dEta[i] = 0.25 * NODE_ETA[i] * (1.0 + xi * NODE_XI[i]);
    }
    return shape;
}

template <typename T>
[[nodiscard]] T interpolated(const Shape& shape, const std::array<T, NODES>& values)
{
    T sum{};
    for (std::size_t i = 0; i < NODES; ++i)
        sum += shape.value[i] * values[i];
    return sum;
}

// The local frame of a point of the reference surface (document section 3): e3 the normal, e1 = e_y × e3 made
// unit, e2 = e3 × e1 (rows of E), and j = ‖g1 × g2‖ at ζ = 0.
struct Frame
{
    std::array<Vector3<double>, 3> axis{};
    double area = 0.0; // j
};

[[nodiscard]] inline Frame frameAt(const ShellQuad4Geometry& geometry, const Shape& shape)
{
    Vector3<double> g1{};
    Vector3<double> g2{};
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t k = 0; k < 3; ++k)
        {
            g1[k] += shape.dXi[i] * geometry.position[i][k];
            g2[k] += shape.dEta[i] * geometry.position[i][k];
        }
    const Vector3<double> normal = crossOf(g1, g2);
    Frame frame;
    frame.area = std::sqrt(dotOf(normal, normal));
    frame.axis[2] = scaledBy(normal, 1.0 / frame.area);
    frame.axis[0] = unitOf(crossOf(Vector3<double>{0.0, 1.0, 0.0}, frame.axis[2]));
    frame.axis[1] = crossOf(frame.axis[2], frame.axis[0]);
    return frame;
}

// The covariant vectors g1 = X,ξ, g2 = X,η, g3 = X,ζ at (ξ, η, ζ), rows of J(ζ):
// X = Σ N_i (x_i + ζ (t_i / 2) n_i).
template <typename T>
[[nodiscard]] Matrix3<T> covariantAt(const ShellQuad4Geometry& geometry, const std::array<T, NODES>& thickness,
                                     const Shape& shape, double zeta)
{
    Matrix3<T> j;
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t k = 0; k < 3; ++k)
        {
            const T fibre = 0.5 * thickness[i] * geometry.normal[i][k];
            const T point = geometry.position[i][k] + zeta * fibre;
            j(0, k) += shape.dXi[i] * point;
            j(1, k) += shape.dEta[i] * point;
            j(2, k) += shape.value[i] * fibre;
        }
    return j;
}

// u,ξ, u,η and u,ζ at (ξ, η, ζ) as rows of the local DOFs (component c of derivative k at [k](c, dof)):
// u = Σ N_i (u_i + ζ (t_i / 2) θ_i × n_i).
template <typename T>
[[nodiscard]] std::array<ScalarMatrix<T, 3, DOFS>, 3> displacementRowsAt(const ShellQuad4Geometry& geometry,
                                                                        const std::array<T, NODES>& thickness,
                                                                        const Shape& shape, double zeta)
{
    std::array<ScalarMatrix<T, 3, DOFS>, 3> rows;
    for (std::size_t i = 0; i < NODES; ++i)
    {
        const Vector3<double>& n = geometry.normal[i];
        // θ × n = S θ with S = [[0, n_z, −n_y], [−n_z, 0, n_x], [n_y, −n_x, 0]]
        const std::array<std::array<double, 3>, 3> skew{{{0.0, n[2], -n[1]}, {-n[2], 0.0, n[0]}, {n[1], -n[0], 0.0}}};
        const T half = 0.5 * thickness[i];
        const std::array<double, 2> derivative{shape.dXi[i], shape.dEta[i]};
        for (std::size_t c = 0; c < 3; ++c)
        {
            for (std::size_t k = 0; k < 2; ++k)
                rows[k](c, 6 * i + c) += T(derivative[k]);
            for (std::size_t r = 0; r < 3; ++r)
            {
                if (skew[c][r] == 0.0)
                    continue;
                for (std::size_t k = 0; k < 2; ++k)
                    rows[k](c, 6 * i + 3 + r) += (derivative[k] * zeta * skew[c][r]) * half;
                rows[2](c, 6 * i + 3 + r) += (shape.value[i] * skew[c][r]) * half;
            }
        }
    }
    return rows;
}

// The displacement gradient in the local frame at (ξ, η, ζ): L[a][b] = ∂u_b/∂x_a = (E J(ζ)⁻¹) [u,ξ; u,η; u,ζ] Eᵀ
// (document section 3; the three rows u,ξ, u,η, u,ζ are kept).
template <typename T>
using Gradient = std::array<std::array<Row<T>, 3>, 3>;

template <typename T>
[[nodiscard]] Gradient<T> gradientAt(const ShellQuad4Geometry& geometry, const std::array<T, NODES>& thickness,
                                     const Shape& shape, double zeta, const Frame& frame)
{
    const Matrix3<T> inverse = inverseOf(covariantAt(geometry, thickness, shape, zeta));
    const std::array<ScalarMatrix<T, 3, DOFS>, 3> rows = displacementRowsAt(geometry, thickness, shape, zeta);
    Gradient<T> gradient{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        // P_ak = Σ_m E_am (J⁻¹)_mk: the derivative along local axis a in terms of ξ, η, ζ
        std::array<T, 3> p{};
        for (std::size_t k = 0; k < 3; ++k)
            for (std::size_t m = 0; m < 3; ++m)
                p[k] += frame.axis[a][m] * inverse(m, k);
        for (std::size_t b = 0; b < 3; ++b)
            for (std::size_t dof = 0; dof < DOFS; ++dof)
            {
                T sum{};
                for (std::size_t k = 0; k < 3; ++k)
                    for (std::size_t c = 0; c < 3; ++c)
                        sum += p[k] * frame.axis[b][c] * rows[k](c, dof);
                gradient[a][b][dof] = sum;
            }
    }
    return gradient;
}

// The membrane strains e0 and the curvatures κ of a point (document section 5), rows 0-2 and 3-5:
// ε = (L11, L22, L12 + L21) at ζ = ±1, e0 = ½ (ε⁺ + ε⁻), κ = (ε⁺ − ε⁻) / (2 h), h = g3 · e3.
template <typename T>
[[nodiscard]] ScalarMatrix<T, 6, DOFS> planeRowsAt(const ShellQuad4Geometry& geometry,
                                                   const std::array<T, NODES>& thickness, const Shape& shape,
                                                   const Frame& frame)
{
    const Gradient<T> top = gradientAt(geometry, thickness, shape, 1.0, frame);
    const Gradient<T> bottom = gradientAt(geometry, thickness, shape, -1.0, frame);
    const Matrix3<T> middle = covariantAt(geometry, thickness, shape, 0.0);
    T height{};
    for (std::size_t k = 0; k < 3; ++k)
        height += middle(2, k) * frame.axis[2][k];
    const T curvatureScale = T(0.5) / height;
    ScalarMatrix<T, 6, DOFS> rows;
    for (std::size_t dof = 0; dof < DOFS; ++dof)
    {
        const std::array<T, 3> above{top[0][0][dof], top[1][1][dof], top[0][1][dof] + top[1][0][dof]};
        const std::array<T, 3> below{bottom[0][0][dof], bottom[1][1][dof], bottom[0][1][dof] + bottom[1][0][dof]};
        for (std::size_t r = 0; r < 3; ++r)
        {
            rows(r, dof) = 0.5 * (above[r] + below[r]);
            rows(3 + r, dof) = (above[r] - below[r]) * curvatureScale;
        }
    }
    return rows;
}

// The covariant transverse shears at ζ = 0 (document section 7): γ̃_α = g3 · u,α + g_α · Σ N_i (t_i / 2) θ_i × n_i.
template <typename T>
[[nodiscard]] std::array<Row<T>, 2> covariantShearAt(const ShellQuad4Geometry& geometry,
                                                    const std::array<T, NODES>& thickness, double xi, double eta)
{
    const Shape shape = shapeAt(xi, eta);
    const Matrix3<T> j = covariantAt(geometry, thickness, shape, 0.0);
    const std::array<ScalarMatrix<T, 3, DOFS>, 3> rows = displacementRowsAt(geometry, thickness, shape, 0.0);
    std::array<Row<T>, 2> shear{};
    for (std::size_t alpha = 0; alpha < 2; ++alpha)
        for (std::size_t dof = 0; dof < DOFS; ++dof)
        {
            T sum{};
            for (std::size_t c = 0; c < 3; ++c)
                sum += j(2, c) * rows[alpha](c, dof) + j(alpha, c) * rows[2](c, dof);
            shear[alpha][dof] = sum;
        }
    return shear;
}

// The covariant shears tied at the mid-sides (Dvorkin and Bathe, 1984): γ̃_ξ at (0, −1) and (0, 1), γ̃_η at (−1, 0)
// and (1, 0).
template <typename T>
struct TiedShears
{
    std::array<Row<T>, 2> alongXi{};  // at η = −1, η = 1
    std::array<Row<T>, 2> alongEta{}; // at ξ = −1, ξ = 1
};

template <typename T>
[[nodiscard]] TiedShears<T> tiedShearsOf(const ShellQuad4Geometry& geometry, const std::array<T, NODES>& thickness)
{
    TiedShears<T> tied;
    tied.alongXi[0] = covariantShearAt(geometry, thickness, 0.0, -1.0)[0];
    tied.alongXi[1] = covariantShearAt(geometry, thickness, 0.0, 1.0)[0];
    tied.alongEta[0] = covariantShearAt(geometry, thickness, -1.0, 0.0)[1];
    tied.alongEta[1] = covariantShearAt(geometry, thickness, 1.0, 0.0)[1];
    return tied;
}

// The local transverse shears γ1, γ2 of a point from the tied strains, interpolated linearly across the element and
// brought to the local frame of the point with the contravariant vectors g^k (the columns of J(0)⁻¹):
// γ_a = Σ_α γ̃_α [(g^α · e_a)(g^3 · e_3) + (g^3 · e_a)(g^α · e_3)].
template <typename T>
[[nodiscard]] std::array<Row<T>, 2> shearRowsAt(const ShellQuad4Geometry& geometry,
                                               const std::array<T, NODES>& thickness, const TiedShears<T>& tied,
                                               double xi, double eta)
{
    const Shape shape = shapeAt(xi, eta);
    const Frame frame = frameAt(geometry, shape);
    const Matrix3<T> inverse = inverseOf(covariantAt(geometry, thickness, shape, 0.0));
    // g^k · e_a = Σ_m (J⁻¹)_mk e_a[m]
    const auto contravariant = [&](std::size_t k, std::size_t a)
    {
        T sum{};
        for (std::size_t m = 0; m < 3; ++m)
            sum += inverse(m, k) * frame.axis[a][m];
        return sum;
    };
    std::array<Row<T>, 2> rows{};
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t alpha = 0; alpha < 2; ++alpha)
        {
            const T weight =
                contravariant(alpha, a) * contravariant(2, 2) + contravariant(2, a) * contravariant(alpha, 2);
            const double low = alpha == 0 ? 0.5 * (1.0 - eta) : 0.5 * (1.0 - xi);
            const double high = alpha == 0 ? 0.5 * (1.0 + eta) : 0.5 * (1.0 + xi);
            const std::array<Row<T>, 2>& sides = alpha == 0 ? tied.alongXi : tied.alongEta;
            for (std::size_t dof = 0; dof < DOFS; ++dof)
                rows[a][dof] += weight * (low * sides[0][dof] + high * sides[1][dof]);
        }
    return rows;
}

// The drilling term of a point (document section 8), at ζ = 0: r = ω − θ_n, ω = ½ (L12 − L21),
// θ_n = Σ N_i (θ_i − θ̄) · n_i + θ̄ · e3, θ̄ the mean rotation of the nodes. Zero for a rigid motion.
template <typename T>
[[nodiscard]] Row<T> drillingRowAt(const ShellQuad4Geometry& geometry, const std::array<T, NODES>& thickness,
                                   const Shape& shape, const Frame& frame)
{
    const Gradient<T> gradient = gradientAt(geometry, thickness, shape, 0.0, frame);
    Row<T> row{};
    for (std::size_t dof = 0; dof < DOFS; ++dof)
        row[dof] = 0.5 * (gradient[0][1][dof] - gradient[1][0][dof]);
    // the coefficient of θ̄ in θ_n: e3 − Σ N_i n_i
    Vector3<double> mean = frame.axis[2];
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            mean[k] -= shape.value[i] * geometry.normal[i][k];
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            row[6 * i + 3 + k] -= T(shape.value[i] * geometry.normal[i][k] + 0.25 * mean[k]);
    return row;
}

// The angle from the element x axis to the local e1 of a point, about the normal: the section, given in the element
// frame, is turned by its opposite. Zero on a flat element.
[[nodiscard]] inline double frameAngleOf(const Frame& frame, double materialAngle)
{
    // the material direction m (in the plane of the element axes) seen in the local frame, less its angle in the
    // element axes: φ_p − φ_E
    const Vector3<double> m{std::cos(materialAngle), std::sin(materialAngle), 0.0};
    const double local = std::atan2(dotOf(m, frame.axis[1]), dotOf(m, frame.axis[0]));
    double angle = local - materialAngle;
    const double pi = 3.14159265358979323846;
    while (angle > pi)
        angle -= 2.0 * pi;
    while (angle < -pi)
        angle += 2.0 * pi;
    return angle;
}

// What the enhanced strains need of the element centre (document section 6): G = (j0 / j) T0 M(ξ, η) at a point, T0
// from the inverse of J0 = [[g1·e1, g1·e2], [g2·e1, g2·e2]] at the centre (Simo and Rifai, 1990; the five-parameter
// stress field of Pian and Sumihara, 1984, in its enhanced-strain form).
struct EnhancedCentre
{
    Matrix3<double> t0;
    double j0 = 0.0;
};

[[nodiscard]] inline EnhancedCentre enhancedCentreOf(const ShellQuad4Geometry& geometry)
{
    const Shape shape = shapeAt(0.0, 0.0);
    const Frame frame = frameAt(geometry, shape);
    Matrix2<double> j;
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t c = 0; c < 2; ++c)
        {
            j(0, c) += shape.dXi[i] * dotOf(geometry.position[i], frame.axis[c]);
            j(1, c) += shape.dEta[i] * dotOf(geometry.position[i], frame.axis[c]);
        }
    const Matrix2<double> inverse = inverseOf(j);
    EnhancedCentre centre;
    centre.j0 = determinantOf(j);
    centre.t0(0, 0) = inverse(0, 0) * inverse(0, 0);
    centre.t0(0, 1) = inverse(0, 1) * inverse(0, 1);
    centre.t0(0, 2) = inverse(0, 0) * inverse(0, 1);
    centre.t0(1, 0) = inverse(1, 0) * inverse(1, 0);
    centre.t0(1, 1) = inverse(1, 1) * inverse(1, 1);
    centre.t0(1, 2) = inverse(1, 0) * inverse(1, 1);
    centre.t0(2, 0) = 2.0 * inverse(0, 0) * inverse(1, 0);
    centre.t0(2, 1) = 2.0 * inverse(0, 1) * inverse(1, 1);
    centre.t0(2, 2) = inverse(0, 0) * inverse(1, 1) + inverse(0, 1) * inverse(1, 0);
    return centre;
}

// G (3 × 4) at a point: (j0 / j) T0 [[ξ, 0, 0, 0], [0, η, 0, 0], [0, 0, ξ, η]].
[[nodiscard]] inline ScalarMatrix<double, 3, ENHANCED_PER_PART> enhancedAt(const EnhancedCentre& centre, double xi,
                                                                           double eta, double area)
{
    ScalarMatrix<double, 3, ENHANCED_PER_PART> g;
    const double scale = centre.j0 / area;
    for (std::size_t r = 0; r < 3; ++r)
    {
        g(r, 0) = scale * centre.t0(r, 0) * xi;
        g(r, 1) = scale * centre.t0(r, 1) * eta;
        g(r, 2) = scale * centre.t0(r, 2) * xi;
        g(r, 3) = scale * centre.t0(r, 2) * eta;
    }
    return g;
}

// One integration point of the membrane and bending part: its weight j, its strain rows (e0, κ), its section and its
// enhanced strains.
template <typename T>
struct PlanePoint
{
    double weight = 0.0;
    double xi = 0.0;
    double eta = 0.0;
    double angle = 0.0; // frameAngleOf
    Frame frame;
    ScalarMatrix<T, 6, DOFS> rows;
    PointSection<T> section;
    ScalarMatrix<double, 3, ENHANCED_PER_PART> enhanced;
};

template <typename T>
[[nodiscard]] std::array<PlanePoint<T>, POINT_COUNT> planePointsOf(const ShellQuad4Geometry& geometry,
                                                                  const PlateConstitutive<T>& constitutive,
                                                                  const std::array<T, NODES>& thickness)
{
    const EnhancedCentre centre = enhancedCentreOf(geometry);
    std::array<PlanePoint<T>, POINT_COUNT> points;
    std::size_t p = 0;
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            PlanePoint<T>& point = points[p++];
            const Shape shape = shapeAt(xi, eta);
            point.xi = xi;
            point.eta = eta;
            point.frame = frameAt(geometry, shape);
            point.weight = point.frame.area;
            point.angle = frameAngleOf(point.frame, constitutive.materialAngle);
            point.rows = planeRowsAt(geometry, thickness, shape, point.frame);
            point.section =
                sectionAt(constitutive, interpolated(shape, thickness), point.angle, 4.0 * point.frame.area);
            point.enhanced = enhancedAt(centre, xi, eta, point.frame.area);
        }
    return points;
}

// The enhanced part of the stiffness (document section 9): K_dα = Σ j Bᵀ C G and K_αα = Σ j Gᵀ C G, G applied to the
// membrane strains (parameters 0-3) and to the curvatures (4-7). `isActive`: a section without membrane stiffness, or
// without bending stiffness, drops the corresponding four parameters.
template <typename T>
struct EnhancedPart
{
    ScalarMatrix<T, DOFS, ENHANCED> coupling; // K_dα
    ScalarMatrix<T, ENHANCED> stiffness;      // K_αα
    std::array<bool, ENHANCED> isActive{};
};

template <typename T>
[[nodiscard]] EnhancedPart<T> enhancedPartOf(const PlateConstitutive<T>& constitutive,
                                             const std::array<PlanePoint<T>, POINT_COUNT>& points)
{
    EnhancedPart<T> part;
    for (std::size_t q = 0; q < ENHANCED; ++q)
        part.isActive[q] = q < ENHANCED_PER_PART ? constitutive.hasMembrane : constitutive.hasBending;
    for (const PlanePoint<T>& point : points)
    {
        const ScalarMatrix<T, 6> c = planeMatrixOf(point.section);
        // C G: column q of the 6 × 8 matrix [[G, 0], [0, G]] times C
        ScalarMatrix<T, 6, ENHANCED> cg;
        for (std::size_t r = 0; r < 6; ++r)
            for (std::size_t q = 0; q < ENHANCED; ++q)
            {
                const std::size_t part0 = q < ENHANCED_PER_PART ? 0 : 3;
                const std::size_t column = q % ENHANCED_PER_PART;
                T sum{};
                for (std::size_t k = 0; k < 3; ++k)
                    sum += c(r, part0 + k) * point.enhanced(k, column);
                cg(r, q) = sum;
            }
        for (std::size_t q = 0; q < ENHANCED; ++q)
        {
            for (std::size_t dof = 0; dof < DOFS; ++dof)
            {
                T sum{};
                for (std::size_t r = 0; r < 6; ++r)
                    sum += point.rows(r, dof) * cg(r, q);
                part.coupling(dof, q) += point.weight * sum;
            }
            const std::size_t part0 = q < ENHANCED_PER_PART ? 0 : 3;
            const std::size_t column = q % ENHANCED_PER_PART;
            for (std::size_t s = 0; s < ENHANCED; ++s)
            {
                T sum{};
                for (std::size_t k = 0; k < 3; ++k)
                    sum += point.enhanced(k, column) * cg(part0 + k, s);
                part.stiffness(q, s) += point.weight * sum;
            }
        }
    }
    return part;
}

// X = K_αα⁻¹ R for the active parameters (zero rows for the others), by Gauss elimination with the pivot chosen on the
// real parts: K_αα is symmetric positive definite on the active parameters.
template <typename T, std::size_t C>
[[nodiscard]] ScalarMatrix<T, ENHANCED, C> solvedEnhanced(const EnhancedPart<T>& part,
                                                         const ScalarMatrix<T, ENHANCED, C>& rightHandSides)
{
    std::array<std::size_t, ENHANCED> active{};
    std::size_t count = 0;
    for (std::size_t q = 0; q < ENHANCED; ++q)
        if (part.isActive[q])
            active[count++] = q;
    ScalarMatrix<T, ENHANCED> a;
    ScalarMatrix<T, ENHANCED, C> x;
    for (std::size_t i = 0; i < count; ++i)
    {
        for (std::size_t k = 0; k < count; ++k)
            a(i, k) = part.stiffness(active[i], active[k]);
        for (std::size_t c = 0; c < C; ++c)
            x(i, c) = rightHandSides(active[i], c);
    }
    for (std::size_t k = 0; k < count; ++k)
    {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < count; ++i)
            if (std::abs(realPart(a(i, k))) > std::abs(realPart(a(pivot, k))))
                pivot = i;
        if (pivot != k)
        {
            for (std::size_t c = 0; c < count; ++c)
                std::swap(a(k, c), a(pivot, c));
            for (std::size_t c = 0; c < C; ++c)
                std::swap(x(k, c), x(pivot, c));
        }
        const T inverse = T(1.0) / a(k, k);
        for (std::size_t i = k + 1; i < count; ++i)
        {
            const T factor = a(i, k) * inverse;
            if (realPart(factor) == 0.0)
                continue;
            for (std::size_t c = k; c < count; ++c)
                a(i, c) -= factor * a(k, c);
            for (std::size_t c = 0; c < C; ++c)
                x(i, c) -= factor * x(k, c);
        }
    }
    for (std::size_t k = count; k-- > 0;)
    {
        const T inverse = T(1.0) / a(k, k);
        for (std::size_t c = 0; c < C; ++c)
        {
            T sum = x(k, c);
            for (std::size_t i = k + 1; i < count; ++i)
                sum -= a(k, i) * x(i, c);
            x(k, c) = sum * inverse;
        }
    }
    // back to the numbering of the eight parameters
    ScalarMatrix<T, ENHANCED, C> result;
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t c = 0; c < C; ++c)
            result(active[i], c) = x(i, c);
    return result;
}

// The widths of the element along its axes (document section 7): a = ½ |x2 + x3 − x1 − x4|,
// b = ½ |y3 + y4 − y1 − y2| (the centre frame is the element frame).
struct Widths
{
    double a = 0.0;
    double b = 0.0;
};

[[nodiscard]] inline Widths widthsOf(const ShellQuad4Geometry& geometry)
{
    return {geometry.meanWidthX, geometry.meanWidthY};
}

} // namespace oos::elements::quad4_detail
