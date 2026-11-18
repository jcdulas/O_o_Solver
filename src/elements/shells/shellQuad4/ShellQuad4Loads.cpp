#include "ShellQuad4Kernel.hpp"
#include "oos/elements/shells/SecondOrderScalar.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include <cmath>

namespace oos::elements
{

namespace
{

using quad4_detail::GAUSS_POINTS;
using quad4_detail::Shape;
using quad4_detail::shapeAt;

// The tangent vectors ∂X/∂ξ and ∂X/∂η of the surface of the grid points at a point (element frame).
struct SurfaceTangents
{
    Vector3<double> alongXi{};
    Vector3<double> alongEta{};
};

[[nodiscard]] SurfaceTangents tangentsAt(const ShellQuad4Geometry& geometry, const Shape& shape)
{
    SurfaceTangents tangents;
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t k = 0; k < 3; ++k)
        {
            tangents.alongXi[k] += shape.dXi[a] * geometry.position[a][k];
            tangents.alongEta[k] += shape.dEta[a] * geometry.position[a][k];
        }
    return tangents;
}

// θ × n = S θ.
[[nodiscard]] std::array<std::array<double, 3>, 3> skewOf(const Vector3<double>& n)
{
    return {{{0.0, n[2], -n[1]}, {-n[2], 0.0, n[0]}, {n[1], -n[0], 0.0}}};
}

} // namespace

// Document section 10, with the 2 × 2 rule: the mass moments per unit area I0 = ρ t + NSM and I2 = ρ t³/12 of a
// homogeneous section about its reference surface (I1 = 0),
//   M = Σ j (I0 N_uᵀ N_u + I2 N_βᵀ N_β),  β = Σ N_i θ_i × n_i the director increment.
// Lumped: m_i = Σ j I0 N_i on the three translations of node i, nothing on the rotations.
// A laminate is taken as a homogeneous section of its mass per unit area (the family gives ρ = mass per area / t).
template <typename T>
ScalarMatrix<T, ShellQuad4::DOFS> ShellQuad4::mass(const Geometry& geometry, const T& density,
                                                   const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                   MassFormulation formulation)
{
    ScalarMatrix<T, DOFS> result;
    if (realPart(density) == 0.0 && realPart(nonStructuralMass) == 0.0)
        return result;
    std::array<std::array<std::array<double, 3>, 3>, NODES> skew{};
    for (std::size_t a = 0; a < NODES; ++a)
        skew[a] = skewOf(geometry.normal[a]);
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            const Shape shape = shapeAt(xi, eta);
            const double area = quad4_detail::frameAt(geometry, shape).area;
            const T t = quad4_detail::interpolated(shape, thickness);
            const T translational = area * (density * t + nonStructuralMass);
            const T rotary = area * (density * t * t * t / 12.0);
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
                    const double weight = shape.value[a] * shape.value[b];
                    for (std::size_t k = 0; k < 3; ++k)
                        result(6 * a + k, 6 * b + k) += translational * weight;
                    for (std::size_t r = 0; r < 3; ++r)
                        for (std::size_t s = 0; s < 3; ++s)
                        {
                            double product = 0.0;
                            for (std::size_t c = 0; c < 3; ++c)
                                product += skew[a][c][r] * skew[b][c][s];
                            if (product != 0.0)
                                result(6 * a + 3 + r, 6 * b + 3 + s) += rotary * (weight * product);
                        }
                }
            }
        }
    return result;
}

template ScalarMatrix<double, ShellQuad4::DOFS> ShellQuad4::mass(const Geometry&, const double&, const double&,
                                                                 const std::array<double, NODES>&, MassFormulation);
template ScalarMatrix<Complex, ShellQuad4::DOFS> ShellQuad4::mass(const Geometry&, const Complex&, const Complex&,
                                                                  const std::array<Complex, NODES>&,
                                                                  MassFormulation);
template ScalarMatrix<SecondOrderScalar, ShellQuad4::DOFS> ShellQuad4::mass(
    const Geometry&, const SecondOrderScalar&, const SecondOrderScalar&,
    const std::array<SecondOrderScalar, NODES>&, MassFormulation);

// Document section 12: F_a = Σ p N_a d with the 2 × 2 rule on the surface of the grid points, p = Σ N_a P_a,
// d = ∂X/∂ξ × ∂X/∂η (the surface element, not normalised: its length is the area Jacobian), or ‖d‖ times the unit
// direction of a FacePressureLoad with CID or N. Translations only.
std::array<double, ShellQuad4::DOFS> ShellQuad4::pressureLoads(const Geometry& geometry,
                                                               const std::array<double, 4>& cornerPressures,
                                                               const std::optional<base::Vec3>& direction)
{
    std::array<double, DOFS> loads{};
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            const Shape shape = shapeAt(xi, eta);
            const SurfaceTangents tangents = tangentsAt(geometry, shape);
            Vector3<double> d = crossOf(tangents.alongXi, tangents.alongEta);
            if (direction)
                d = scaledBy(Vector3<double>{(*direction)[0], (*direction)[1], (*direction)[2]},
                             std::sqrt(dotOf(d, d)));
            double p = 0.0;
            for (std::size_t a = 0; a < NODES; ++a)
                p += shape.value[a] * cornerPressures[a];
            for (std::size_t a = 0; a < NODES; ++a)
                for (std::size_t k = 0; k < 3; ++k)
                    loads[6 * a + k] += p * shape.value[a] * d[k];
        }
    return loads;
}

// Document section 12: the initial resultants S0 = C ε_th of the free thermal strain ε_th = (ε_m, gradient) at each
// point (ε(z) = ε_m + z · gradient, given in the element frame and turned to the local frame of the point),
//   f_d = Σ j Bᵀ S0,  f_α = Σ j Gᵀ S0,  f = f_d − K_dα K_αα⁻¹ f_α,
// so that a free thermal expansion leaves the element without stress. The dilatation of ε_th (its mean in-plane
// strain) is taken apart: its loads are K q for the homothety q of the element (addDilatationLoads), so that a warped
// element, or one whose directors are the normals of its grid points, expands freely under a uniform temperature
// rise as a flat one does; the rest of ε_th is taken as given.
std::array<double, ShellQuad4::DOFS> ShellQuad4::thermalLoads(const Geometry& geometry,
                                                              const PlateConstitutive<double>& constitutive,
                                                              const std::array<double, NODES>& thickness,
                                                              const PlateThermalStrain& thermal)
{
    using namespace quad4_detail;
    std::array<double, DOFS> loads{};
    if (thermal.isZero)
        return loads;
    const std::array<PlanePoint<double>, POINT_COUNT> points = planePointsOf(geometry, constitutive, thickness);
    ScalarMatrix<double, ENHANCED, 1> onParameters;
    for (const PlanePoint<double>& point : points)
    {
        std::array<double, 6> strain{};
        for (std::size_t r = 0; r < 3; ++r)
        {
            strain[r] = thermal.membrane[r];
            strain[3 + r] = thermal.gradient[r];
        }
        if (std::abs(point.angle) > NO_ROTATION)
        {
            // ε_element = U3(angle) ε_local
            const Matrix3<double> toLocal = engineeringStrainRotation(-point.angle);
            const std::array<double, 3> membrane = productOf(toLocal, thermal.membrane);
            const std::array<double, 3> gradient = productOf(toLocal, thermal.gradient);
            for (std::size_t r = 0; r < 3; ++r)
            {
                strain[r] = membrane[r];
                strain[3 + r] = gradient[r];
            }
        }
        // the dilatation is taken apart (below)
        strain[0] -= dilatationOf(thermal);
        strain[1] -= dilatationOf(thermal);
        const std::array<double, 6> resultants = productOf(planeMatrixOf(point.section), strain);
        for (std::size_t dof = 0; dof < DOFS; ++dof)
            for (std::size_t r = 0; r < 6; ++r)
                loads[dof] += point.weight * point.rows(r, dof) * resultants[r];
        for (std::size_t q = 0; q < ENHANCED; ++q)
        {
            const std::size_t part = q < ENHANCED_PER_PART ? 0 : 3;
            for (std::size_t r = 0; r < 3; ++r)
                onParameters(q, 0) += point.weight * point.enhanced(r, q % ENHANCED_PER_PART) * resultants[part + r];
        }
    }
    const EnhancedPart<double> enhanced = enhancedPartOf(constitutive, points);
    const ScalarMatrix<double, ENHANCED, 1> parameters = solvedEnhanced(enhanced, onParameters);
    for (std::size_t dof = 0; dof < DOFS; ++dof)
        for (std::size_t q = 0; q < ENHANCED; ++q)
            loads[dof] -= enhanced.coupling(dof, q) * parameters(q, 0);
    director_shell::addDilatationLoads(loads, stiffness(geometry, constitutive, thickness), geometry.position,
                                       dilatationOf(thermal));
    return loads;
}

// Document section 12: the differential stiffness of the membrane forces (N11, N22, N12), constant over the element
// (ShellDifferential), with the local gradients of the translations of the reference surface at the 2 × 2 points,
//   K_σ = Σ j (N11 G1ᵀ G1 + N22 G2ᵀ G2 + N12 (G1ᵀ G2 + G2ᵀ G1)).
// The gradient along local axis a is Σ_k (e_a · g^k) ∂N/∂ξ_k, g^k the contravariant vectors of (g1, g2, the director):
// only the direction of the director enters, taken as Σ N_i n_i (the corner thicknesses being equal).
base::MatN<ShellQuad4::DOFS> ShellQuad4::differentialStiffness(const Geometry& geometry,
                                                               const std::array<double, 3>& membraneForce)
{
    using namespace quad4_detail;
    const std::array<double, NODES> unitThickness{2.0, 2.0, 2.0, 2.0};
    base::MatN<DOFS> k;
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            const Shape shape = shapeAt(xi, eta);
            const Frame frame = frameAt(geometry, shape);
            const Matrix3<double> inverse = inverseOf(covariantAt(geometry, unitThickness, shape, 0.0));
            std::array<double, NODES> dX{};
            std::array<double, NODES> dY{};
            for (std::size_t a = 0; a < NODES; ++a)
                for (std::size_t m = 0; m < 3; ++m)
                {
                    const double alongNatural = inverse(m, 0) * shape.dXi[a] + inverse(m, 1) * shape.dEta[a];
                    dX[a] += frame.axis[0][m] * alongNatural;
                    dY[a] += frame.axis[1][m] * alongNatural;
                }
            addMembraneDifferentialStiffness<NODES, 6>(k, dX, dY, frame.area, membraneForce);
        }
    return k;
}

} // namespace oos::elements
