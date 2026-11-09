#pragma once

// The kinematics of the NASTRAN-95 ShellQuad4 shared by its stiffness, recovery and thermal loads (ShellQuad4.hpp):
// the bilinear shapes, the degenerated-shell Jacobian and lamina frame at a point (spec §2.2(5), §2.5), the strain rows
// of the membrane and of the bending (spec §2.5) and the assumed transverse shear rows of the edge strains (spec §2.6),
// and the shear-correction factors (spec §2.4, §2.7, §7.2). Private to the ShellQuad4 sources.

#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include <array>
#include <numbers>

namespace oos::elements::quad4_detail
{

// The 2-point Gauss abscissa ±1/√3 (weight 1), in ξ, η and ζ alike.
inline constexpr double GAUSS_POINT = std::numbers::inv_sqrt3;
inline constexpr std::array<double, 2> GAUSS_POINTS{-GAUSS_POINT, GAUSS_POINT};
// The natural coordinates of the nodes, in connectivity order.
inline constexpr std::array<double, 4> NODE_XI{-1.0, 1.0, 1.0, -1.0};
inline constexpr std::array<double, 4> NODE_ETA{-1.0, -1.0, 1.0, 1.0};

// N_a = ¼ (1 + ξ ξ_a)(1 + η η_a) and its derivatives (spec §2.5).
struct BilinearShape
{
    std::array<double, 4> value{};
    std::array<double, 4> dXi{};
    std::array<double, 4> dEta{};
};

[[nodiscard]] inline BilinearShape bilinearShape(double xi, double eta)
{
    BilinearShape shape;
    for (std::size_t a = 0; a < 4; ++a)
    {
        shape.value[a] = 0.25 * (1.0 + xi * NODE_XI[a]) * (1.0 + eta * NODE_ETA[a]);
        shape.dXi[a] = 0.25 * NODE_XI[a] * (1.0 + eta * NODE_ETA[a]);
        shape.dEta[a] = 0.25 * NODE_ETA[a] * (1.0 + xi * NODE_XI[a]);
    }
    return shape;
}

template <typename T>
[[nodiscard]] T interpolated(const BilinearShape& shape, const std::array<T, 4>& nodal)
{
    T sum{};
    for (std::size_t a = 0; a < 4; ++a)
        sum += shape.value[a] * nodal[a];
    return sum;
}

// The kinematics at a point (ξ, η, ζ): det J of the 3-D map, the lamina frame (v1, v2, v3) and the derivatives of
// the shapes along v1 and v2.
template <typename T>
struct PointKinematics
{
    BilinearShape shape;
    T jacobian{};                   // det J
    std::array<Vector3<T>, 3> lamina{}; // v1, v2, v3 (spec §2.2(5): the rows of Ψ)
    std::array<T, 4> dX{};          // N_a,x' along v1
    std::array<T, 4> dY{};          // N_a,y' along v2
};

// Spec §2.5: X(ξ, η, ζ) = Σ N_a [X_a + (ζ/2) t_a ν_a]; J has rows ∂X/∂ξ, ∂X/∂η, ∂X/∂ζ. The lamina frame: v3 normal
// to ∂X/∂ξ and ∂X/∂η, v1 = ŷ × v3 normalised (ŷ the element y axis), v2 = v3 × v1. Φ = Ψ·J⁻¹ gives the derivatives
// along the lamina axes: N_a,x' = Φ11 N_a,ξ + Φ12 N_a,η, N_a,y' = Φ21 N_a,ξ + Φ22 N_a,η.
template <typename T>
[[nodiscard]] PointKinematics<T> kinematicsAt(const ShellQuad4Geometry& geometry, const std::array<T, 4>& thickness,
                                              double xi, double eta, double zeta)
{
    PointKinematics<T> point;
    point.shape = bilinearShape(xi, eta);
    const BilinearShape& shape = point.shape;
    Matrix3<T> j;
    for (std::size_t a = 0; a < 4; ++a)
    {
        const T fibre = 0.5 * zeta * thickness[a];
        for (std::size_t k = 0; k < 3; ++k)
        {
            const T coordinate = geometry.position[a][k] + fibre * geometry.normal[a][k];
            j(0, k) += shape.dXi[a] * coordinate;
            j(1, k) += shape.dEta[a] * coordinate;
            j(2, k) += 0.5 * thickness[a] * shape.value[a] * geometry.normal[a][k];
        }
    }
    point.jacobian = determinantOf(j);
    const Matrix3<T> inverse = inverseOf(j);
    const Vector3<T> alongXi{j(0, 0), j(0, 1), j(0, 2)};
    const Vector3<T> alongEta{j(1, 0), j(1, 1), j(1, 2)};
    const Vector3<T> v3 = unitOf(crossOf(alongXi, alongEta));
    const Vector3<T> elementY{T(0.0), T(1.0), T(0.0)};
    const Vector3<T> v1 = unitOf(crossOf(elementY, v3));
    const Vector3<T> v2 = crossOf(v3, v1);
    point.lamina = {v1, v2, v3};
    // Φ(r, i) = Σ_k Ψ(r, k) J⁻¹(k, i), for r = 1, 2 and i = ξ, η.
    std::array<std::array<T, 2>, 2> phi{};
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t k = 0; k < 3; ++k)
                phi[r][i] += point.lamina[r][k] * inverse(k, i);
    for (std::size_t a = 0; a < 4; ++a)
    {
        point.dX[a] = phi[0][0] * shape.dXi[a] + phi[0][1] * shape.dEta[a];
        point.dY[a] = phi[1][0] * shape.dXi[a] + phi[1][1] * shape.dEta[a];
    }
    return point;
}

// Spec §2.5, rows 1 to 6 of B at a point of height ζ: the membrane strains (εx', εy', γx'y') from the translations,
// the bending strains at the fibre z = ζ t/2 from the rotations. With r1_a = ν_a × v1, r2_a = ν_a × v2:
//   1: N_a,x' v1·u_a   2: N_a,y' v2·u_a   3: (N_a,y' v1 + N_a,x' v2)·u_a
//   4: (ζ/2) t_a N_a,x' r1_a·θ_a   5: (ζ/2) t_a N_a,y' r2_a·θ_a   6: (ζ/2) t_a (N_a,y' r1_a + N_a,x' r2_a)·θ_a
template <typename T>
[[nodiscard]] ScalarMatrix<T, 6, 24> inPlaneStrainRows(const ShellQuad4Geometry& geometry,
                                                       const std::array<T, 4>& thickness,
                                                       const PointKinematics<T>& point, double zeta)
{
    ScalarMatrix<T, 6, 24> b;
    const Vector3<T>& v1 = point.lamina[0];
    const Vector3<T>& v2 = point.lamina[1];
    for (std::size_t a = 0; a < 4; ++a)
    {
        Vector3<T> normal{T(geometry.normal[a][0]), T(geometry.normal[a][1]), T(geometry.normal[a][2])};
        const Vector3<T> r1 = crossOf(normal, v1);
        const Vector3<T> r2 = crossOf(normal, v2);
        const T fibre = 0.5 * zeta * thickness[a];
        const std::size_t u = 6 * a;
        const std::size_t theta = u + 3;
        for (std::size_t k = 0; k < 3; ++k)
        {
            b(0, u + k) = point.dX[a] * v1[k];
            b(1, u + k) = point.dY[a] * v2[k];
            b(2, u + k) = point.dY[a] * v1[k] + point.dX[a] * v2[k];
            b(3, theta + k) = fibre * point.dX[a] * r1[k];
            b(4, theta + k) = fibre * point.dY[a] * r2[k];
            b(5, theta + k) = fibre * (point.dY[a] * r1[k] + point.dX[a] * r2[k]);
        }
    }
    return b;
}

// Spec §2.6, the assumed transverse shear at a point, rows (γx'z, γy'z) in that order (NASTRAN-95 stores γy'z first).
//
// The tangential shear strain of edge e (node e to node e + 1) at the mid-surface,
//   γ_e = (u_(e+1) − u_e)·μ_e / L_e + ½ (β_e + β_(e+1))·ê_e,   β_a = θ_a × ν_a,
// is interpolated to the point through the corner dual vectors: the shear vector at node a is the in-plane vector whose
// components along its two edges are their strains, and Σ_a N_a of these regroups per edge as Σ_e S_e γ_e with
//   S_e(ξ, η) = N_e (ê_e + c_e ê_(e−1)) / (1 − c_e²) + N_(e+1) (ê_e + c_(e+1) ê_(e+1)) / (1 − c_(e+1)²);
// the shear along a lamina axis v is Σ_e (v·S_e) γ_e.
template <typename T>
[[nodiscard]] ScalarMatrix<T, 2, 24> transverseShearRows(const ShellQuad4Geometry& geometry,
                                                         const PointKinematics<T>& point)
{
    std::array<Vector3<double>, 4> s{};
    for (std::size_t e = 0; e < 4; ++e)
    {
        const std::size_t next = (e + 1) % 4;
        const std::size_t previous = (e + 3) % 4;
        const std::array<double, 4>& c = geometry.cornerCosine;
        const std::array<Vector3<double>, 4>& edge = geometry.edgeDirection;
        const double start = point.shape.value[e] / (1.0 - c[e] * c[e]);
        const double end = point.shape.value[next] / (1.0 - c[next] * c[next]);
        for (std::size_t k = 0; k < 3; ++k)
            s[e][k] = start * (edge[e][k] + c[e] * edge[previous][k]) + end * (edge[e][k] + c[next] * edge[next][k]);
    }
    ScalarMatrix<T, 2, 24> b;
    for (std::size_t direction = 0; direction < 2; ++direction)
    {
        const Vector3<T>& v = point.lamina[direction];
        std::array<T, 4> along{}; // v·S_e
        for (std::size_t e = 0; e < 4; ++e)
            along[e] = v[0] * s[e][0] + v[1] * s[e][1] + v[2] * s[e][2];
        for (std::size_t a = 0; a < 4; ++a)
        {
            const std::size_t outgoing = a;
            const std::size_t incoming = (a + 3) % 4;
            const T fromOutgoing = along[outgoing] / geometry.edgeLength[outgoing];
            const T fromIncoming = along[incoming] / geometry.edgeLength[incoming];
            // The rotation part: wᵀ [ν_a]× θ_a, with w = −½ [(v·S_j) ê_j + (v·S_i) ê_i] (so that wᵀ [ν]× θ =
            // ½ Σ (v·S) ê·(θ × ν), the β terms of the two edges at node a).
            Vector3<T> w{};
            for (std::size_t k = 0; k < 3; ++k)
                w[k] = -0.5 * (along[outgoing] * geometry.edgeDirection[outgoing][k] +
                               along[incoming] * geometry.edgeDirection[incoming][k]);
            const Vector3<double>& n = geometry.normal[a];
            const std::size_t u = 6 * a;
            for (std::size_t k = 0; k < 3; ++k)
                b(direction, u + k) =
                    -fromOutgoing * geometry.edgeNormal[outgoing][k] + fromIncoming * geometry.edgeNormal[incoming][k];
            // [ν]× = [[0, −ν3, ν2], [ν3, 0, −ν1], [−ν2, ν1, 0]]; (wᵀ [ν]×)_k = Σ_m w_m [ν]×(m, k).
            b(direction, u + 3) = w[1] * n[2] - w[2] * n[1];
            b(direction, u + 4) = -w[0] * n[2] + w[2] * n[0];
            b(direction, u + 5) = w[0] * n[1] - w[1] * n[0];
        }
    }
    return b;
}

// The parts of the shear-correction factors that are constant over the element (spec §2.4, §2.7).
template <typename T>
struct ShearCorrection
{
    bool isDefined = false;  // bending present and no zero modulus (NASTRAN-95's NOCSUB false)
    bool isFlexible = false; // MID3 given: the factors in series with the shear compliance
    T poissonX{};            // ν_x, ν_y of the stiffness (capped below 1, ν' when not positive)
    T poissonY{};
    T recoveryPoissonX{}; // ν_x, ν_y of the recovery (not capped, ν' when negative)
    T recoveryPoissonY{};
    T torsionX{}; // C_tx = C_t (A/B)², C_ty = C_t (B/A)²
    T torsionY{};
    T anisotropy{}; // D of the anisotropic factor, clamped
};

template <typename T>
[[nodiscard]] ShearCorrection<T> shearCorrectionOf(const ShellQuad4Geometry& geometry,
                                                   const PlateConstitutive<T>& constitutive, const T& meanThickness);

// The stiffness factors at an in-plane Gauss point: S1 = s1 + s2 on the even part of a shear, S2 = s1 − s2 on the odd
// part (x for γxz, y for γyz), and the anisotropic factors f_x, f_y.
template <typename T>
struct PointShearFactors
{
    T evenX{};
    T oddX{};
    T evenY{};
    T oddY{};
    T anisotropicX{T(1.0)};
    T anisotropicY{T(1.0)};
};

template <typename T>
[[nodiscard]] PointShearFactors<T> stiffnessShearFactors(const ShellQuad4Geometry& geometry,
                                                         const PlateConstitutive<T>& constitutive,
                                                         const ShearCorrection<T>& correction, const T& thickness,
                                                         double xi, double eta);

// The recovery factors f_x, f_y at an in-plane point (spec §7.2(3)); `isCentre` takes A and B as the lengths.
template <typename T>
[[nodiscard]] std::array<T, 2> recoveryShearFactors(const ShellQuad4Geometry& geometry,
                                                    const PlateConstitutive<T>& constitutive,
                                                    const ShearCorrection<T>& correction, const T& thickness,
                                                    double xi, double eta, bool isCentre);

} // namespace oos::elements::quad4_detail
