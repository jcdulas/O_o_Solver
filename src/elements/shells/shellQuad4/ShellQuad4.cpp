#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"
#include "oos/elements/shells/SecondOrderScalar.hpp"

#include "ShellQuad4Kinematics.hpp"

#include <string>

namespace oos::elements
{

namespace quad4_detail
{

namespace
{

// Spec §2.7, the shear-correction constants of NASTRAN-95's QUAD4 (routine quad4d).
constexpr double BENDING_FACTOR = 1.6;  // C_b, on the part of a shear even across the element
constexpr double TORSION_FACTOR = 71.0; // C_t = 71 α (1.6 / C_b)(1 + 415 α λ²), on the odd part
constexpr double TORSION_THICKNESS_FACTOR = 415.0;
constexpr double STIFFNESS_SHEAR_COEFFICIENT = 20.0; // c = 20 I / (t_s (1 − ν) L²)
constexpr double RECOVERY_SHEAR_COEFFICIENT = 32.0;  // ψ = 32 I / ((1 − ν) t_s a²) of the recovery (spec §7.2)
constexpr double POISSON_CAP = 0.999999;             // the effective Poisson ratio of the stiffness stays below 1
// Spec §2.7, the anisotropic factor ("based on T.J.R. Hughes"): W1 = 1 + 4400 λ⁴, D = (18.375 − 11.875 α) W1 for
// α ≥ 0.2, else (159.85 α − 15.97) W1, clamped to [0.01, 2000].
constexpr double ANISOTROPY_THICKNESS_FACTOR = 4400.0;
constexpr double ANISOTROPY_ASPECT_LIMIT = 0.2;
constexpr double ANISOTROPY_SQUARE_BASE = 18.375;
constexpr double ANISOTROPY_SQUARE_SLOPE = 11.875;
constexpr double ANISOTROPY_SLENDER_SLOPE = 159.85;
constexpr double ANISOTROPY_SLENDER_BASE = 15.97;
constexpr double ANISOTROPY_MIN = 0.01;
constexpr double ANISOTROPY_MAX = 2000.0;
constexpr double ANISOTROPY_FLOOR = 1.0e-7; // a factor not positive is replaced by this

// The effective Poisson ratio of spec §2.4: ν = m E / (2 G) − 1 (exactly ν for a IsoMat with 12I/T³ = 1); the stiffness
// caps it at 0.999999 and falls back to ν' (from the rotated bending matrix) when it is not positive, the recovery
// does not cap it and falls back when it is negative.
template <typename T>
[[nodiscard]] T effectivePoisson(const T& ratio, const T& young, const T& shear, const T& fallback, bool isStiffness)
{
    T nu = ratio * young / (2.0 * shear) - 1.0;
    if (isStiffness)
    {
        nu = smallerOf(nu, T(POISSON_CAP));
        return realPart(nu) <= 0.0 ? fallback : nu;
    }
    return realPart(nu) < 0.0 ? fallback : nu;
}

// Spec §2.7(4): the length of the element along x on the Gauss row of η (A at the centre row), and along y on the
// Gauss column of ξ.
[[nodiscard]] double lengthAlongX(const ShellQuad4Geometry& geometry, double eta)
{
    const std::array<Vector3<double>, 4>& p = geometry.position;
    const double edge = eta < 0.0 ? p[1][0] - p[0][0] : p[2][0] - p[3][0];
    return geometry.meanWidthX + GAUSS_POINT * (edge - geometry.meanWidthX);
}

[[nodiscard]] double lengthAlongY(const ShellQuad4Geometry& geometry, double xi)
{
    const std::array<Vector3<double>, 4>& p = geometry.position;
    const double edge = xi < 0.0 ? p[3][1] - p[0][1] : p[2][1] - p[1][1];
    return geometry.meanWidthY + GAUSS_POINT * (edge - geometry.meanWidthY);
}

// a/(1 + a) in series with the shear compliance (MID3 given), a alone otherwise (spec §2.7).
template <typename T>
[[nodiscard]] T saturated(const T& factor, bool isFlexible)
{
    return isFlexible ? factor / (1.0 + factor) : factor;
}

} // namespace

template <typename T>
ShearCorrection<T> shearCorrectionOf(const ShellQuad4Geometry& geometry, const PlateConstitutive<T>& constitutive,
                                     const T& meanThickness)
{
    ShearCorrection<T> correction;
    correction.isFlexible = constitutive.hasShearFlexibility;
    if (!constitutive.hasBending)
        return correction;
    const double a = geometry.meanWidthX;
    const double b = geometry.meanWidthY;
    const double alpha = geometry.aspect;
    const T lambda = meanThickness / std::max(a, b); // λ = t̄ / max(A, B)
    const T torsion =
        TORSION_FACTOR * alpha * (1.6 / BENDING_FACTOR) * (1.0 + TORSION_THICKNESS_FACTOR * alpha * lambda * lambda);
    correction.torsionX = torsion * (a / b) * (a / b);
    correction.torsionY = torsion * (b / a) * (b / a);

    const PlateConstitutive<T>& c = constitutive;
    const T fallbackX = c.bending(0, 1) / c.bending(0, 0); // ν'_x of the rotated bending matrix
    const T fallbackY = c.bending(1, 0) / c.bending(1, 1);
    const bool hasZeroModulus = realPart(c.youngX) == 0.0 || realPart(c.youngY) == 0.0 ||
                                realPart(c.shearModulusX) == 0.0 || realPart(c.shearModulusY) == 0.0 ||
                                realPart(c.bendingRatio) == 0.0;
    correction.isDefined = !hasZeroModulus;
    if (hasZeroModulus)
    {
        // Deviation from NASTRAN-95 (deliberate): with a zero modulus NASTRAN-95 keeps the ν_x, ν_y of the element it
        // processed before (static storage), so the result depends on the element order; ν' is taken instead.
        correction.poissonX = correction.recoveryPoissonX = fallbackX;
        correction.poissonY = correction.recoveryPoissonY = fallbackY;
    }
    else
    {
        correction.poissonX = effectivePoisson(c.bendingRatio, c.youngX, c.shearModulusX, fallbackX, true);
        correction.poissonY = effectivePoisson(c.bendingRatio, c.youngY, c.shearModulusY, fallbackY, true);
        correction.recoveryPoissonX = effectivePoisson(c.bendingRatio, c.youngX, c.shearModulusX, fallbackX, false);
        correction.recoveryPoissonY = effectivePoisson(c.bendingRatio, c.youngY, c.shearModulusY, fallbackY, false);
    }

    // Spec §2.7(2-3): D of the anisotropic factor.
    const T lambda2 = lambda * lambda;
    const T thicknessWeight = 1.0 + ANISOTROPY_THICKNESS_FACTOR * lambda2 * lambda2;
    T d = alpha >= ANISOTROPY_ASPECT_LIMIT
              ? (ANISOTROPY_SQUARE_BASE - ANISOTROPY_SQUARE_SLOPE * alpha) * thicknessWeight
              : (ANISOTROPY_SLENDER_SLOPE * alpha - ANISOTROPY_SLENDER_BASE) * thicknessWeight;
    correction.anisotropy = largerOf(smallerOf(d, T(ANISOTROPY_MAX)), T(ANISOTROPY_MIN));
    return correction;
}

template <typename T>
PointShearFactors<T> stiffnessShearFactors(const ShellQuad4Geometry& geometry, const PlateConstitutive<T>& constitutive,
                                           const ShearCorrection<T>& correction, const T& thickness, double xi,
                                           double eta)
{
    PointShearFactors<T> factors;
    const double a = geometry.meanWidthX;
    const double b = geometry.meanWidthY;
    const T inertia = constitutive.bendingRatio * thickness * thickness * thickness / 12.0;
    const T shearThickness = constitutive.shearRatio * thickness;
    // Spec §2.7: c_x = 20 I / (t_s (1 − ν_x) A²), c_y = 20 I / (t_s (1 − ν_y) B²) with the mean widths.
    const T cX = STIFFNESS_SHEAR_COEFFICIENT * inertia / (shearThickness * (1.0 - correction.poissonX) * a * a);
    const T cY = STIFFNESS_SHEAR_COEFFICIENT * inertia / (shearThickness * (1.0 - correction.poissonY) * b * b);
    const bool flexible = correction.isFlexible;
    const T bendingX = saturated(BENDING_FACTOR * cX, flexible);
    const T torsionX = saturated(correction.torsionX * cX, flexible);
    const T bendingY = saturated(BENDING_FACTOR * cY, flexible);
    const T torsionY = saturated(correction.torsionY * cY, flexible);
    factors.evenX = bendingX + torsionX;
    factors.oddX = bendingX - torsionX;
    factors.evenY = bendingY + torsionY;
    factors.oddY = bendingY - torsionY;
    if (!correction.isDefined)
        return factors;
    // Spec §2.7(4-6): ψ = 20 D I S_w (1 + α²) / (t_s (1 − ν) a²) with the lengths along the Gauss row and column.
    const double alpha = geometry.aspect;
    const double ax = lengthAlongX(geometry, eta);
    const double by = lengthAlongY(geometry, xi);
    const T common = STIFFNESS_SHEAR_COEFFICIENT * correction.anisotropy * inertia * geometry.warpFactor *
                     (1.0 + alpha * alpha) / shearThickness;
    const T psiX = common / ((1.0 - correction.poissonX) * ax * ax);
    const T psiY = common / ((1.0 - correction.poissonY) * by * by);
    const T fX = saturated(psiX, flexible);
    const T fY = saturated(psiY, flexible);
    factors.anisotropicX = realPart(fX) <= 0.0 ? T(ANISOTROPY_FLOOR) : fX;
    factors.anisotropicY = realPart(fY) <= 0.0 ? T(ANISOTROPY_FLOOR) : fY;
    return factors;
}

template <typename T>
std::array<T, 2> recoveryShearFactors(const ShellQuad4Geometry& geometry, const PlateConstitutive<T>& constitutive,
                                      const ShearCorrection<T>& correction, const T& thickness, double xi, double eta,
                                      bool isCentre)
{
    if (!constitutive.hasBending || !correction.isDefined)
        return {T(1.0), T(1.0)};
    // Spec §7.2(3): ψ = 32 I / ((1 − ν) t_s a²); with MID3 f = min(ψ, 1), without f = ψ / (1 − ψ) below ψ = 1,
    // capped at 1. These differ from the stiffness factors: NASTRAN-95's recovery, reproduced.
    const T inertia = constitutive.bendingRatio * thickness * thickness * thickness / 12.0;
    const T shearThickness = constitutive.shearRatio * thickness;
    const double ax = isCentre ? geometry.meanWidthX : lengthAlongX(geometry, eta);
    const double by = isCentre ? geometry.meanWidthY : lengthAlongY(geometry, xi);
    const std::array<T, 2> psi{
        RECOVERY_SHEAR_COEFFICIENT * inertia / ((1.0 - correction.recoveryPoissonX) * shearThickness * ax * ax),
        RECOVERY_SHEAR_COEFFICIENT * inertia / ((1.0 - correction.recoveryPoissonY) * shearThickness * by * by)};
    std::array<T, 2> factors{};
    for (std::size_t k = 0; k < 2; ++k)
    {
        if (correction.isFlexible)
            factors[k] = smallerOf(psi[k], T(1.0));
        else
            factors[k] = realPart(psi[k]) < 1.0 ? smallerOf(psi[k] / (1.0 - psi[k]), T(1.0)) : T(1.0);
    }
    return factors;
}

template ShearCorrection<double> shearCorrectionOf(const ShellQuad4Geometry&, const PlateConstitutive<double>&,
                                                   const double&);
template ShearCorrection<Complex> shearCorrectionOf(const ShellQuad4Geometry&, const PlateConstitutive<Complex>&,
                                                    const Complex&);
template std::array<double, 2> recoveryShearFactors(const ShellQuad4Geometry&, const PlateConstitutive<double>&,
                                                    const ShearCorrection<double>&, const double&, double, double,
                                                    bool);
template std::array<Complex, 2> recoveryShearFactors(const ShellQuad4Geometry&, const PlateConstitutive<Complex>&,
                                                     const ShearCorrection<Complex>&, const Complex&, double, double,
                                                     bool);

namespace
{

// The 10 strain rows of spec §2.8 at the Gauss point (ξ_i, η_j, ζ_k), in this implementation's order:
//   0-5  rows 1-6 of §2.5, rows 3 and 6 (the in-plane shears) taken at the centre of the same ζ (reduced shear);
//   6, 7 γxz at the point and at the point mirrored in η;  8, 9 γyz at the point and mirrored in ξ.
template <typename T>
struct GaussRows
{
    std::array<ScalarMatrix<T, 6, 24>, 2> centre; // per ζ
    std::array<ScalarMatrix<T, 2, 24>, 8> shear;  // per (ξ, η, ζ) index 4 i + 2 j + k
    [[nodiscard]] static std::size_t index(std::size_t i, std::size_t j, std::size_t k) { return 4 * i + 2 * j + k; }
};

template <typename T>
[[nodiscard]] GaussRows<T> gaussRowsOf(const ShellQuad4Geometry& geometry, const std::array<T, 4>& thickness)
{
    GaussRows<T> rows;
    for (std::size_t k = 0; k < 2; ++k)
    {
        const double zeta = GAUSS_POINTS[k];
        const PointKinematics<T> centre = kinematicsAt(geometry, thickness, 0.0, 0.0, zeta);
        rows.centre[k] = inPlaneStrainRows(geometry, thickness, centre, zeta);
    }
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            for (std::size_t k = 0; k < 2; ++k)
            {
                const PointKinematics<T> point =
                    kinematicsAt(geometry, thickness, GAUSS_POINTS[i], GAUSS_POINTS[j], GAUSS_POINTS[k]);
                rows.shear[GaussRows<T>::index(i, j, k)] = transverseShearRows(geometry, point);
            }
    return rows;
}

// Spec §2.8: the generalised constitutive matrix at an in-plane Gauss point and a height ζ.
template <typename T>
[[nodiscard]] ScalarMatrix<T, 10> constitutiveAt(const PlateConstitutive<T>& c, const PointShearFactors<T>& factors,
                                                 double zeta)
{
    ScalarMatrix<T, 10> g;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            if (c.hasMembrane)
                g(i, j) = c.membrane(i, j);
            if (c.hasBending)
                g(3 + i, 3 + j) = c.bendingRatio * c.bending(i, j);
            if (c.hasMembrane && c.hasBending)
            {
                // The membrane-bending cross term: G1, and G1 − 6 ζ G4 with MID4 (integrated over the thickness, the
                // −6 ζ G4 term gives the coupling t² G4 against the NASTRAN-95 curvature).
                T cross = c.membrane(i, j);
                if (c.hasCoupling)
                    cross -= 6.0 * zeta * c.coupling(i, j);
                g(i, 3 + j) = cross;
                g(3 + i, j) = cross;
            }
        }
    if (!c.hasBending)
        return g;
    // The transverse shear: ¼ S1 on the diagonal and ¼ S2 between a shear and its mirror (the means of the even and
    // odd parts squared), and the anisotropic coupling √(f_x f_y) G3_xy between γxz and γyz at the point.
    const T& shearX = c.transverseShear(0, 0);
    const T& shearY = c.transverseShear(1, 1);
    g(6, 6) = 0.25 * factors.evenX * shearX;
    g(7, 7) = g(6, 6);
    g(6, 7) = 0.25 * factors.oddX * shearX;
    g(7, 6) = g(6, 7);
    g(8, 8) = 0.25 * factors.evenY * shearY;
    g(9, 9) = g(8, 8);
    g(8, 9) = 0.25 * factors.oddY * shearY;
    g(9, 8) = g(8, 9);
    g(6, 8) = squareRoot(factors.anisotropicX * factors.anisotropicY) * c.transverseShear(0, 1);
    g(8, 6) = g(6, 8);
    return g;
}

} // namespace

} // namespace quad4_detail

std::optional<std::string> ShellQuad4::integrationProblem(const Geometry& geometry,
                                                          const std::array<double, NODES>& thickness)
{
    using namespace quad4_detail;
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
            for (const double zeta : GAUSS_POINTS)
                if (!(kinematicsAt(geometry, thickness, xi, eta, zeta).jacobian > 0.0))
                    return std::string(
                        "expected a positive Jacobian at every integration point, found one not "
                        "positive (a folded corner, or a thickness large against a warped element); "
                        "reorder ShellQuad4.node1-ShellQuad4.node4 around the element or remesh it");
    return std::nullopt;
}

// Spec §2.9: K = Σ over the 2 × 2 × 2 points of det J · Bᵀ G B (weights 1), then symmetrised.
template <typename T>
ScalarMatrix<T, ShellQuad4::DOFS> ShellQuad4::stiffness(const Geometry& geometry,
                                                        const PlateConstitutive<T>& constitutive,
                                                        const std::array<T, NODES>& thickness)
{
    using namespace quad4_detail;
    T meanThickness{};
    for (const T& t : thickness)
        meanThickness += 0.25 * t;
    const ShearCorrection<T> correction = shearCorrectionOf(geometry, constitutive, meanThickness);
    const GaussRows<T> rows = gaussRowsOf(geometry, thickness);
    ScalarMatrix<T, DOFS> k;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
        {
            const double xi = GAUSS_POINTS[i];
            const double eta = GAUSS_POINTS[j];
            const T t = interpolated(bilinearShape(xi, eta), thickness);
            const PointShearFactors<T> factors =
                constitutive.hasBending ? stiffnessShearFactors(geometry, constitutive, correction, t, xi, eta)
                                        : PointShearFactors<T>{};
            for (std::size_t z = 0; z < 2; ++z)
            {
                const double zeta = GAUSS_POINTS[z];
                const PointKinematics<T> point = kinematicsAt(geometry, thickness, xi, eta, zeta);
                const ScalarMatrix<T, 6, 24> plane = inPlaneStrainRows(geometry, thickness, point, zeta);
                ScalarMatrix<T, 10, 24> b;
                for (std::size_t c = 0; c < 24; ++c)
                {
                    for (std::size_t r = 0; r < 6; ++r)
                        b(r, c) = (r == 2 || r == 5) ? rows.centre[z](r, c) : plane(r, c);
                    b(6, c) = rows.shear[GaussRows<T>::index(i, j, z)](0, c);
                    b(7, c) = rows.shear[GaussRows<T>::index(i, 1 - j, z)](0, c);
                    b(8, c) = rows.shear[GaussRows<T>::index(i, j, z)](1, c);
                    b(9, c) = rows.shear[GaussRows<T>::index(1 - i, j, z)](1, c);
                }
                addCongruence(k, b, constitutiveAt(constitutive, factors, zeta), point.jacobian);
            }
        }
    // Spec §2.9: K_ij = K_ji = ½ (K_ij + K_ji) (NASTRAN-95 leaves it unsymmetrised with a damping GE; the damping is
    // not formed from this matrix here, so it is always symmetrised).
    for (std::size_t r = 0; r < DOFS; ++r)
        for (std::size_t c = r + 1; c < DOFS; ++c)
        {
            const T mean = 0.5 * (k(r, c) + k(c, r));
            k(r, c) = mean;
            k(c, r) = mean;
        }
    return k;
}

template ScalarMatrix<double, ShellQuad4::DOFS> ShellQuad4::stiffness(const Geometry&, const PlateConstitutive<double>&,
                                                                      const std::array<double, NODES>&);
template ScalarMatrix<Complex, ShellQuad4::DOFS> ShellQuad4::stiffness(const Geometry&,
                                                                       const PlateConstitutive<Complex>&,
                                                                       const std::array<Complex, NODES>&);

template ScalarMatrix<SecondOrderScalar, ShellQuad4::DOFS> ShellQuad4::stiffness(
    const Geometry&, const PlateConstitutive<SecondOrderScalar>&, const std::array<SecondOrderScalar, NODES>&);
template std::array<SecondOrderScalar, 2> quad4_detail::recoveryShearFactors(
    const ShellQuad4Geometry&, const PlateConstitutive<SecondOrderScalar>&,
    const quad4_detail::ShearCorrection<SecondOrderScalar>&, const SecondOrderScalar&, double, double, bool);
template quad4_detail::ShearCorrection<SecondOrderScalar> quad4_detail::shearCorrectionOf(
    const ShellQuad4Geometry&, const PlateConstitutive<SecondOrderScalar>&, const SecondOrderScalar&);

} // namespace oos::elements
