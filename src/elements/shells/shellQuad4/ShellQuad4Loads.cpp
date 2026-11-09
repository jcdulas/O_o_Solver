#include "ShellQuad4Kinematics.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include <cmath>

namespace oos::elements
{

namespace
{

using quad4_detail::BilinearShape;
using quad4_detail::GAUSS_POINTS;
using quad4_detail::bilinearShape;

// The tangent vectors ∂X/∂ξ and ∂X/∂η of the mid-surface at a point (element frame).
struct SurfaceTangents
{
    Vector3<double> alongXi{};
    Vector3<double> alongEta{};
};

[[nodiscard]] SurfaceTangents tangentsAt(const ShellQuad4Geometry& geometry, const BilinearShape& shape)
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

} // namespace

// Spec §2.10: the 2 × 2 rule on the mid-surface (weights 1), dA = ‖∂X/∂ξ × ∂X/∂η‖, t_g the thickness at the point;
// lumped M_aa = Σ_g ρ t_g dA N_a, coupled M_ab = Σ_g ρ t_g dA N_a N_b (ρ taken as 1 when zero), then scaled by
// (V ρ + NSM A)/(V ρ) when there is a non-structural mass (V = Σ dA t_g, A = Σ dA): the NSM distributed like the
// structural mass (a negative NSM accepted, as NASTRAN-95's QUAD4 does). Translations only: no rotary inertia.
template <typename T>
ScalarMatrix<T, ShellQuad4::DOFS> ShellQuad4::mass(const Geometry& geometry, const T& density,
                                                   const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                   MassFormulation formulation)
{
    ScalarMatrix<T, DOFS> result;
    if (realPart(density) == 0.0 && realPart(nonStructuralMass) == 0.0)
        return result;
    const T weightDensity = realPart(density) != 0.0 ? density : T(1.0);
    ScalarMatrix<T, NODES> scalar;
    T area{};
    T volume{};
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            const BilinearShape shape = bilinearShape(xi, eta);
            const SurfaceTangents tangents = tangentsAt(geometry, shape);
            const Vector3<double> normal = crossOf(tangents.alongXi, tangents.alongEta);
            const double dA = std::sqrt(dotOf(normal, normal));
            const T t = quad4_detail::interpolated(shape, thickness);
            area += dA;
            volume += dA * t;
            const T weight = weightDensity * t * dA;
            for (std::size_t a = 0; a < NODES; ++a)
            {
                if (formulation == MassFormulation::Lumped)
                {
                    scalar(a, a) += weight * shape.value[a];
                    continue;
                }
                for (std::size_t b = 0; b < NODES; ++b)
                    scalar(a, b) += weight * shape.value[a] * shape.value[b];
            }
        }
    T scale(1.0);
    if (realPart(nonStructuralMass) != 0.0)
        scale = (volume * density + nonStructuralMass * area) / (volume * weightDensity);
    for (std::size_t a = 0; a < NODES; ++a)
        for (std::size_t b = 0; b < NODES; ++b)
            for (std::size_t k = 0; k < 3; ++k)
                result(6 * a + k, 6 * b + k) = scale * scalar(a, b);
    return result;
}

template ScalarMatrix<double, ShellQuad4::DOFS> ShellQuad4::mass(const Geometry&, const double&, const double&,
                                                                 const std::array<double, NODES>&, MassFormulation);
template ScalarMatrix<Complex, ShellQuad4::DOFS> ShellQuad4::mass(const Geometry&, const Complex&, const Complex&,
                                                                  const std::array<Complex, NODES>&,
                                                                  MassFormulation);

// Spec §5.1: F_a = Σ_g p(g) N_a(g) d(g) with the 2 × 2 rule, p = Σ N_a P_a, d = ∂X/∂ξ × ∂X/∂η (the surface element,
// not normalised: its length is the area Jacobian), or ‖d‖ times the unit direction of a FacePressureLoad with CID or
// N. Translations only.
std::array<double, ShellQuad4::DOFS> ShellQuad4::pressureLoads(const Geometry& geometry,
                                                               const std::array<double, 4>& cornerPressures,
                                                               const std::optional<base::Vec3>& direction)
{
    std::array<double, DOFS> loads{};
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            const BilinearShape shape = bilinearShape(xi, eta);
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

// Spec §6.2: P = Σ over the 2 × 2 × 2 points of det J Bᵀ G ε_th, with the membrane and bending rows of the stiffness
// (the in-plane shears at the centre) and G its membrane-bending block, ε_th = [ε_m; z·gradient], z = ζ t/2.
// Deviation from NASTRAN-95 (deliberate): NASTRAN-95 flattens the element (each node at the mean height) for its
// thermal loads; the stiffness geometry is kept here, so that a free thermal expansion gives zero stress exactly.
std::array<double, ShellQuad4::DOFS> ShellQuad4::thermalLoads(const Geometry& geometry,
                                                              const PlateConstitutive<double>& constitutive,
                                                              const std::array<double, NODES>& thickness,
                                                              const PlateThermalStrain& thermal)
{
    using namespace quad4_detail;
    const PlateConstitutive<double>& c = constitutive;
    std::array<double, DOFS> loads{};
    if (thermal.isZero)
        return loads;
    std::array<ScalarMatrix<double, 6, 24>, 2> centreRows;
    for (std::size_t z = 0; z < 2; ++z)
        centreRows[z] = inPlaneStrainRows(geometry, thickness,
                                          kinematicsAt(geometry, thickness, 0.0, 0.0, GAUSS_POINTS[z]),
                                          GAUSS_POINTS[z]);
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            const double t = interpolated(bilinearShape(xi, eta), thickness);
            for (std::size_t z = 0; z < 2; ++z)
            {
                const double zeta = GAUSS_POINTS[z];
                const PointKinematics<double> point = kinematicsAt(geometry, thickness, xi, eta, zeta);
                const ScalarMatrix<double, 6, 24> plane = inPlaneStrainRows(geometry, thickness, point, zeta);
                std::array<double, 6> strain{};
                for (std::size_t r = 0; r < 3; ++r)
                {
                    strain[r] = thermal.membrane[r];
                    strain[3 + r] = 0.5 * zeta * t * thermal.gradient[r];
                }
                // σ = G ε_th with the blocks of the stiffness (spec §2.8, rows 1 to 6).
                std::array<double, 6> stress{};
                for (std::size_t i = 0; i < 3; ++i)
                    for (std::size_t j = 0; j < 3; ++j)
                    {
                        if (c.hasMembrane)
                            stress[i] += c.membrane(i, j) * strain[j];
                        if (c.hasBending)
                            stress[3 + i] += c.bendingRatio * c.bending(i, j) * strain[3 + j];
                        if (c.hasMembrane && c.hasBending)
                        {
                            const double cross =
                                c.membrane(i, j) - (c.hasCoupling ? 6.0 * zeta * c.coupling(i, j) : 0.0);
                            stress[i] += cross * strain[3 + j];
                            stress[3 + i] += cross * strain[j];
                        }
                    }
                for (std::size_t k = 0; k < DOFS; ++k)
                {
                    double sum = 0.0;
                    for (std::size_t r = 0; r < 6; ++r)
                    {
                        const double b = (r == 2 || r == 5) ? centreRows[z](r, k) : plane(r, k);
                        sum += b * stress[r];
                    }
                    loads[k] += point.jacobian * sum;
                }
            }
        }
    return loads;
}

// The membrane-stress differential stiffness (ShellDifferential) on the mean plane: the shape derivatives of the plane
// coordinates at the 2 × 2 Gauss points, weight det J.
base::MatN<ShellQuad4::DOFS> ShellQuad4::differentialStiffness(const Geometry& geometry,
                                                               const std::array<double, 3>& membraneForce)
{
    base::MatN<DOFS> k;
    for (const double xi : GAUSS_POINTS)
        for (const double eta : GAUSS_POINTS)
        {
            const BilinearShape shape = bilinearShape(xi, eta);
            double j11 = 0.0;
            double j12 = 0.0;
            double j21 = 0.0;
            double j22 = 0.0;
            for (std::size_t a = 0; a < NODES; ++a)
            {
                j11 += shape.dXi[a] * geometry.position[a][0];
                j12 += shape.dXi[a] * geometry.position[a][1];
                j21 += shape.dEta[a] * geometry.position[a][0];
                j22 += shape.dEta[a] * geometry.position[a][1];
            }
            const double determinant = j11 * j22 - j12 * j21;
            std::array<double, NODES> dX{};
            std::array<double, NODES> dY{};
            for (std::size_t a = 0; a < NODES; ++a)
            {
                dX[a] = (j22 * shape.dXi[a] - j12 * shape.dEta[a]) / determinant;
                dY[a] = (-j21 * shape.dXi[a] + j11 * shape.dEta[a]) / determinant;
            }
            addMembraneDifferentialStiffness<NODES, 6>(k, dX, dY, determinant, membraneForce);
        }
    return k;
}

} // namespace oos::elements
