#pragma once

// The kinematics of a shell with a director at each node (docs/shell-quad4-formulation.md, "the document", sections 3,
// 5, 7 and 8), for the ShellQuad4 and the ShellTria3: the covariant vectors and the local frame of a point, the
// displacement gradient, the membrane strains and the curvatures, the covariant transverse shears and their local
// components, the drilling term. N is the number of nodes; a row of 6 N values is a linear form of the local DOFs
// (u, v, w, θx, θy, θz of each node, node by node, in the element frame).
//
// `position` are the nodes in the element frame, `normal` their unit directors there: the normal of the element at
// the node, or the normal its grid point shares with the neighbouring elements (ShellDirectors.hpp). The
// rotation of a node about its director moves nothing: it is the drilling rotation, held by the drilling term alone.

#include "oos/elements/shells/ShellScalar.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace oos::elements::director_shell
{

template <std::size_t N>
using Points = std::array<Vector3<double>, N>;

template <typename T, std::size_t N>
using Row = std::array<T, 6 * N>;

// The shape functions of a point and their derivatives along the natural coordinates.
template <std::size_t N>
struct Shape
{
    std::array<double, N> value{};
    std::array<double, N> dXi{};
    std::array<double, N> dEta{};
};

[[nodiscard]] inline Vector3<double> unitOf(const Vector3<double>& v)
{
    return scaledBy(v, 1.0 / std::sqrt(dotOf(v, v)));
}

// The local frame of a point of the reference surface (document section 3): e3 the normal, e1 = e_y × e3 made
// unit, e2 = e3 × e1 (rows of E), and j = ‖g1 × g2‖ at ζ = 0.
struct Frame
{
    std::array<Vector3<double>, 3> axis{};
    double area = 0.0; // j
};

template <std::size_t N>
[[nodiscard]] Frame frameAt(const Points<N>& position, const Shape<N>& shape)
{
    Vector3<double> g1{};
    Vector3<double> g2{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t k = 0; k < 3; ++k)
        {
            g1[k] += shape.dXi[i] * position[i][k];
            g2[k] += shape.dEta[i] * position[i][k];
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
template <typename T, std::size_t N>
[[nodiscard]] Matrix3<T> covariantAt(const Points<N>& position, const Points<N>& normal,
                                     const std::array<T, N>& thickness, const Shape<N>& shape, double zeta)
{
    Matrix3<T> j;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t k = 0; k < 3; ++k)
        {
            const T fibre = 0.5 * thickness[i] * normal[i][k];
            const T point = position[i][k] + zeta * fibre;
            j(0, k) += shape.dXi[i] * point;
            j(1, k) += shape.dEta[i] * point;
            j(2, k) += shape.value[i] * fibre;
        }
    return j;
}

// u,ξ, u,η and u,ζ at (ξ, η, ζ) as rows of the local DOFs (component c of derivative k at [k](c, dof)):
// u = Σ N_i (u_i + ζ (t_i / 2) θ_i × n_i).
template <typename T, std::size_t N>
[[nodiscard]] std::array<ScalarMatrix<T, 3, 6 * N>, 3> displacementRowsAt(const Points<N>& normal,
                                                                          const std::array<T, N>& thickness,
                                                                          const Shape<N>& shape, double zeta)
{
    std::array<ScalarMatrix<T, 3, 6 * N>, 3> rows;
    for (std::size_t i = 0; i < N; ++i)
    {
        const Vector3<double>& n = normal[i];
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
template <typename T, std::size_t N>
using Gradient = std::array<std::array<Row<T, N>, 3>, 3>;

template <typename T, std::size_t N>
[[nodiscard]] Gradient<T, N> gradientAt(const Points<N>& position, const Points<N>& normal,
                                        const std::array<T, N>& thickness, const Shape<N>& shape, double zeta,
                                        const Frame& frame)
{
    const Matrix3<T> inverse = inverseOf(covariantAt(position, normal, thickness, shape, zeta));
    const std::array<ScalarMatrix<T, 3, 6 * N>, 3> rows = displacementRowsAt(normal, thickness, shape, zeta);
    Gradient<T, N> gradient{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        // P_ak = Σ_m E_am (J⁻¹)_mk: the derivative along local axis a in terms of ξ, η, ζ
        std::array<T, 3> p{};
        for (std::size_t k = 0; k < 3; ++k)
            for (std::size_t m = 0; m < 3; ++m)
                p[k] += frame.axis[a][m] * inverse(m, k);
        for (std::size_t b = 0; b < 3; ++b)
            for (std::size_t dof = 0; dof < 6 * N; ++dof)
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
template <typename T, std::size_t N>
[[nodiscard]] ScalarMatrix<T, 6, 6 * N> planeRowsAt(const Points<N>& position, const Points<N>& normal,
                                                   const std::array<T, N>& thickness, const Shape<N>& shape,
                                                   const Frame& frame)
{
    const Gradient<T, N> top = gradientAt(position, normal, thickness, shape, 1.0, frame);
    const Gradient<T, N> bottom = gradientAt(position, normal, thickness, shape, -1.0, frame);
    const Matrix3<T> middle = covariantAt(position, normal, thickness, shape, 0.0);
    T height{};
    for (std::size_t k = 0; k < 3; ++k)
        height += middle(2, k) * frame.axis[2][k];
    const T curvatureScale = T(0.5) / height;
    ScalarMatrix<T, 6, 6 * N> rows;
    for (std::size_t dof = 0; dof < 6 * N; ++dof)
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
template <typename T, std::size_t N>
[[nodiscard]] std::array<Row<T, N>, 2> covariantShearAt(const Points<N>& position, const Points<N>& normal,
                                                    const std::array<T, N>& thickness, const Shape<N>& shape)
{
    const Matrix3<T> j = covariantAt(position, normal, thickness, shape, 0.0);
    const std::array<ScalarMatrix<T, 3, 6 * N>, 3> rows = displacementRowsAt(normal, thickness, shape, 0.0);
    std::array<Row<T, N>, 2> shear{};
    for (std::size_t alpha = 0; alpha < 2; ++alpha)
        for (std::size_t dof = 0; dof < 6 * N; ++dof)
        {
            T sum{};
            for (std::size_t c = 0; c < 3; ++c)
                sum += j(2, c) * rows[alpha](c, dof) + j(alpha, c) * rows[2](c, dof);
            shear[alpha][dof] = sum;
        }
    return shear;
}

// The drilling term of a point (document section 8), at ζ = 0: r = ω − θ_n, ω = ½ (L12 − L21),
// θ_n = Σ N_i (θ_i − θ̄) · n_i + θ̄ · e3, θ̄ the mean rotation of the nodes. Zero for a rigid motion.
template <typename T, std::size_t N>
[[nodiscard]] Row<T, N> drillingRowAt(const Points<N>& position, const Points<N>& normal,
                                      const std::array<T, N>& thickness, const Shape<N>& shape, const Frame& frame)
{
    const Gradient<T, N> gradient = gradientAt(position, normal, thickness, shape, 0.0, frame);
    Row<T, N> row{};
    for (std::size_t dof = 0; dof < 6 * N; ++dof)
        row[dof] = 0.5 * (gradient[0][1][dof] - gradient[1][0][dof]);
    // the coefficient of θ̄ in θ_n: e3 − Σ N_i n_i
    Vector3<double> mean = frame.axis[2];
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            mean[k] -= shape.value[i] * normal[i][k];
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            row[6 * i + 3 + k] -= T(shape.value[i] * normal[i][k] + mean[k] / static_cast<double>(N));
    return row;
}

// The local transverse shears γ1, γ2 of a point from its covariant shears γ̃_ξ, γ̃_η (document section 7), with the
// contravariant vectors g^k (the columns of J(0)⁻¹):
// γ_a = Σ_α γ̃_α [(g^α · e_a)(g^3 · e_3) + (g^3 · e_a)(g^α · e_3)].
template <typename T, std::size_t N>
[[nodiscard]] std::array<Row<T, N>, 2> localShearsOf(const Points<N>& position, const Points<N>& normal,
                                                    const std::array<T, N>& thickness, const Shape<N>& shape,
                                                    const Frame& frame, const std::array<Row<T, N>, 2>& covariant)
{
    const Matrix3<T> inverse = inverseOf(covariantAt(position, normal, thickness, shape, 0.0));
    // g^k · e_a = Σ_m (J⁻¹)_mk e_a[m]
    const auto contravariant = [&](std::size_t k, std::size_t a)
    {
        T sum{};
        for (std::size_t m = 0; m < 3; ++m)
            sum += inverse(m, k) * frame.axis[a][m];
        return sum;
    };
    std::array<Row<T, N>, 2> rows{};
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t alpha = 0; alpha < 2; ++alpha)
        {
            const T weight =
                contravariant(alpha, a) * contravariant(2, 2) + contravariant(2, a) * contravariant(alpha, 2);
            for (std::size_t dof = 0; dof < 6 * N; ++dof)
                rows[a][dof] += weight * covariant[alpha][dof];
        }
    return rows;
}

// The strains (e0, κ) at a point of the homothety u = x of the element (unit dilatation, no rotation): `rows` the
// rows of (e0, κ) at the point. On a flat element whose directors are its normal they are (1, 1, 0, 0, 0, 0). Where
// the directors turn across the element (a warped element, directors shared with the neighbours) the homothety,
// which moves each fibre without stretching it, has a curvature (minus that of the surface of the directors) and a
// membrane strain off (1, 1, 0) by the square of thickness times curvature. A free thermal dilatation ε is the
// homothety u = ε x, without stress (a uniform dilatation bends nothing: J. L. Sanders, An improved first-approximation
// theory for thin shells, NASA TR R-24, 1959): its free strains are ε times these.
template <typename T, std::size_t N>
[[nodiscard]] std::array<T, 6> dilatationStrainsOf(const Points<N>& position, const ScalarMatrix<T, 6, 6 * N>& rows)
{
    std::array<T, 6> strains{};
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t k = 0; k < 3; ++k)
                strains[r] += rows(r, 6 * i + k) * position[i][k];
    return strains;
}

// The local DOFs of the homothety u = x of the element: the positions on the translations, no rotation.
template <std::size_t N>
[[nodiscard]] std::array<double, 6 * N> homothetyOf(const Points<N>& position)
{
    std::array<double, 6 * N> q{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            q[6 * i + k] = position[i][k];
    return q;
}

// The loads K q that hold the element in the homothety u = ε x: the thermal loads of a free dilatation ε, which
// leave the element without stress in every one of its strain measures (the transverse shear of directors that are
// not normal to the element included).
template <std::size_t N>
void addDilatationLoads(std::array<double, 6 * N>& loads, const ScalarMatrix<double, 6 * N>& stiffness,
                        const Points<N>& position, double dilatation)
{
    const std::array<double, 6 * N> q = homothetyOf(position);
    for (std::size_t i = 0; i < 6 * N; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < 6 * N; ++j)
            sum += stiffness(i, j) * q[j];
        loads[i] += dilatation * sum;
    }
}

} // namespace oos::elements::director_shell
