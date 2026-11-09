#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingPlate.hpp"

#include "oos/elements/core/IntegrationRules.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"

#include <cmath>

namespace oos::elements
{

namespace
{

// The plate takes no membrane-bending coupling (the family refuses one): the resultant rates carry none.
constexpr bool IS_COUPLED = false;

// The bilinear extrapolation from the 2 × 2 Gauss points (ordered as the corners) to corner i: the bilinear function
// through the four point values, evaluated at the corner, whose coordinates are (±√3, ±√3) in the frame where the
// points are at (±1, ±1).
[[nodiscard]] std::array<double, 4> extrapolationWeights(std::size_t corner)
{
    const double scale = std::sqrt(3.0);
    std::array<double, 4> weights{};
    for (std::size_t k = 0; k < 4; ++k)
        weights[k] = 0.25 * (1.0 + QUAD_CORNER_XI[k] * QUAD_CORNER_XI[corner] * scale) *
                     (1.0 + QUAD_CORNER_ETA[k] * QUAD_CORNER_ETA[corner] * scale);
    return weights;
}

[[nodiscard]] std::array<double, 3> extrapolatedToCorner(const std::array<std::array<double, 3>, 4>& atPoints,
                                                         const std::array<double, 4>& weights)
{
    std::array<double, 3> value{};
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t c = 0; c < 3; ++c)
            value[c] += weights[k] * atPoints[k][c];
    return value;
}

// The temperature change at (ξ, η): the bilinear interpolation of the corner values, or the uniform value.
[[nodiscard]] double deltaTAt(const ShellTemperatureChange& temperature, double xi, double eta)
{
    if (!temperature.hasCornerValues)
        return temperature.deltaT;
    const BilinearShape shape = bilinearShapeAt(xi, eta);
    double value = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        value += shape.n[i] * temperature.atCorners[i];
    return value;
}

// The local DOF of DOF a of the membrane (u, v, θn per corner).
[[nodiscard]] constexpr std::size_t membraneDof(std::size_t a)
{
    constexpr std::array<std::size_t, 3> OF_NODE{0, 1, 5};
    return 6 * (a / 3) + OF_NODE[a % 3];
}

// The local DOF of DOF a of the bending (w, θx, θy per corner).
[[nodiscard]] constexpr std::size_t bendingDof(std::size_t a)
{
    return 6 * (a / 3) + 2 + a % 3;
}

// Adds a matrix over the bending DOFs (12 × 12) into the 24 local DOFs.
void addOnBendingDofs(base::MatN<24>& k, const base::MatN<12>& bending)
{
    for (std::size_t a = 0; a < 12; ++a)
        for (std::size_t b = 0; b < 12; ++b)
            k(bendingDof(a), bendingDof(b)) += bending(a, b);
}

// Adds a matrix over the membrane DOFs (12 × 12) into the 24 local DOFs.
void addOnMembraneDofs(base::MatN<24>& k, const base::MatN<12>& membrane)
{
    for (std::size_t a = 0; a < 12; ++a)
        for (std::size_t b = 0; b < 12; ++b)
            k(membraneDof(a), membraneDof(b)) += membrane(a, b);
}

// The bending DOFs u12 (w, θx, θy per corner) of the local displacements q.
[[nodiscard]] std::array<double, 12> bendingDofsOf(const std::array<double, 24>& q)
{
    std::array<double, 12> u12{};
    for (std::size_t a = 0; a < 12; ++a)
        u12[a] = q[bendingDof(a)];
    return u12;
}

// The membrane DOFs (u, v, θn per corner) of the local displacements q.
[[nodiscard]] std::array<double, 12> membraneDofsOf(const std::array<double, 24>& q)
{
    std::array<double, 12> u12{};
    for (std::size_t a = 0; a < 12; ++a)
        u12[a] = q[membraneDof(a)];
    return u12;
}

// The moment resultant −M_h of the hybrid moments M_h (ShellQuad4DrillingBending's curvature sign, see ShellRecovery.hpp),
// written 0 − M_h so that a zero moment is +0 in the result tables, not −0.
[[nodiscard]] double momentResultant(double hybridMoment)
{
    return 0.0 - hybridMoment;
}

} // namespace

ShellQuad4DrillingPlate::ShellQuad4DrillingPlate(const ShellQuad4DrillingGeometry& geometry, const ShellSection& section,
                               const std::array<double, 4>& thickness)
    : m_geometry(geometry), m_section(section), m_thickness(thickness)
{
    // A warped element is formed on its plane coordinates (flat in its mean plane) and joined to its grid points by
    // the rigid offsets of its heights.
    if (geometry.isWarped)
        m_offsets = ShellQuad4DrillingWarpOffsets{geometry.zDeviation};
    if (section.hasMembrane)
        m_membrane.emplace(geometry, section.membraneMatrix, thickness, section.membraneShearModulus);
    if (section.hasBending)
        m_bending.emplace(geometry, section.bendingMatrix, section.bendingRatio, thickness,
                          TransverseShear{section.transverseShearStiffnessAlongAxes()});
}

double ShellQuad4DrillingPlate::thicknessAt(double xi, double eta) const
{
    const BilinearShape shape = bilinearShapeAt(xi, eta);
    double t = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        t += shape.n[i] * m_thickness[i];
    return t;
}

base::MatN<24> ShellQuad4DrillingPlate::onGridPoints(base::MatN<24> k) const
{
    if (m_offsets)
        k = m_offsets->matrixOnGridPoints(k);
    return k;
}

std::array<double, 24> ShellQuad4DrillingPlate::loadsOnGridPoints(const std::array<double, 24>& f) const
{
    return m_offsets ? m_offsets->loadsOnGridPoints(f) : f;
}

std::array<double, 24> ShellQuad4DrillingPlate::onProjections(const std::array<double, 24>& q) const
{
    return m_offsets ? m_offsets->projected(q) : q;
}

base::MatN<24> ShellQuad4DrillingPlate::stiffness() const
{
    base::MatN<24> k;
    if (m_bending)
        addOnBendingDofs(k, m_bending->stiffness());
    if (m_membrane)
        addOnMembraneDofs(k, m_membrane->stiffness());
    return onGridPoints(k);
}

ShellQuad4DrillingPlate::ThermalAtGaussPoints ShellQuad4DrillingPlate::thermalAtGaussPoints(
    const ShellTemperatureChange& temperature) const
{
    ThermalAtGaussPoints thermal;
    const std::array<QuadraturePoint, 4> points = gaussRule2x2();
    for (std::size_t g = 0; g < 4; ++g)
    {
        const double t = thicknessAt(points[g].xi, points[g].eta);
        const ThermalResultants resultants =
            m_section.thermalResultants(deltaTAt(temperature, points[g].xi, points[g].eta), temperature.gradient, t);
        thermal.force[g] = resultants.force;
        if (m_bending)
        {
            // κ_T = −D⁻¹ M_T: the thermal curvature in the sign of ShellQuad4DrillingBending (κ = −κ' of the lamination theory).
            const std::array<double, 3> curvature =
                inverted(m_bending->bendingStiffnessAt(points[g].xi, points[g].eta)) * resultants.moment;
            for (std::size_t c = 0; c < 3; ++c)
                thermal.curvature[g][c] = -curvature[c];
        }
    }
    return thermal;
}

std::array<double, 24> ShellQuad4DrillingPlate::thermalLoads(const ShellTemperatureChange& temperature) const
{
    std::array<double, 24> loads{};
    if (temperature.isZero())
        return loads;
    const ThermalAtGaussPoints thermal = thermalAtGaussPoints(temperature);
    if (m_bending)
    {
        const std::array<double, 12> nodal = m_bending->initialCurvatureLoads(thermal.curvature).nodal;
        for (std::size_t a = 0; a < 12; ++a)
            loads[bendingDof(a)] += nodal[a];
    }
    if (m_membrane)
    {
        // f = Σ_g w det J Bᵀ N_T on (u, v, θn).
        const std::array<QuadraturePoint, 4> points = gaussRule2x2();
        for (std::size_t g = 0; g < 4; ++g)
        {
            const double scale = points[g].weight * m_geometry.jacobianAt(points[g].xi, points[g].eta).determinant;
            const std::array<double, 12> nodal =
                m_membrane->strainMatrix(points[g].xi, points[g].eta).transposed() * thermal.force[g];
            for (std::size_t a = 0; a < 12; ++a)
                loads[membraneDof(a)] += scale * nodal[a];
        }
    }
    return loadsOnGridPoints(loads);
}

std::array<ShellPlaceState, 5> ShellQuad4DrillingPlate::placeStates(const std::array<double, 24>& q,
                                                           const std::array<std::int32_t, 4>& gridIds,
                                                           const ShellTemperatureChange& temperature) const
{
    const PlaceOperators operators = placeOperators();
    return placeStates(operators, placeThermal(operators, temperature), q, gridIds);
}

ShellQuad4DrillingPlate::PlaceOperators ShellQuad4DrillingPlate::placeOperators() const
{
    PlaceOperators operators;
    const std::array<QuadraturePoint, 4> points = gaussRule2x2();
    for (std::size_t k = 0; k < 5; ++k)
    {
        // Index 0 the centre, 1-4 the Gauss points.
        const double xi = k == 0 ? 0.0 : points[k - 1].xi;
        const double eta = k == 0 ? 0.0 : points[k - 1].eta;
        if (m_membrane)
            operators.membrane[k] = m_membrane->strainMatrix(xi, eta);
        if (m_bending)
            operators.moment[k] = m_bending->momentInterpolationAt(xi, eta);
    }
    for (std::size_t place = 0; place < 5; ++place)
    {
        // Index 0 the centre, 1-4 the corners.
        const bool isCentre = place == 0;
        const std::size_t corner = isCentre ? 0 : place - 1;
        const double thickness = isCentre ? thicknessAt(0.0, 0.0) : m_thickness[corner];
        operators.thickness[place] = thickness;
        if (m_bending)
            operators.flexibility[place] = inverted(m_section.bendingStiffness(thickness));
        else
            operators.jacobian[place] = m_geometry.jacobianAt(isCentre ? 0.0 : QUAD_CORNER_XI[corner],
                                                              isCentre ? 0.0 : QUAD_CORNER_ETA[corner]);
        if (!isCentre)
            operators.extrapolation[corner] = extrapolationWeights(corner);
    }
    return operators;
}

ShellQuad4DrillingPlate::PlaceThermal ShellQuad4DrillingPlate::placeThermal(const PlaceOperators& operators,
                                                          const ShellTemperatureChange& temperature) const
{
    PlaceThermal thermal;
    if (!temperature.isZero() && m_bending)
    {
        const ThermalAtGaussPoints atPoints = thermalAtGaussPoints(temperature);
        thermal.initialMoments = m_bending->initialCurvatureLoads(atPoints.curvature).parameters;
    }
    for (std::size_t place = 0; place < 5; ++place)
    {
        const bool isCentre = place == 0;
        const std::size_t corner = isCentre ? 0 : place - 1;
        const double xi = isCentre ? 0.0 : QUAD_CORNER_XI[corner];
        const double eta = isCentre ? 0.0 : QUAD_CORNER_ETA[corner];
        thermal.temperature[place] = temperature;
        thermal.temperature[place].deltaT = deltaTAt(temperature, xi, eta);
        thermal.resultants[place] = m_section.thermalResultants(thermal.temperature[place].deltaT, temperature.gradient,
                                                                operators.thickness[place]);
    }
    return thermal;
}

// PERFORMANCE: the state of the places from the parts formed once (PlaceOperators, PlaceThermal): the recovery slopes
// evaluate 25 states per element, which formed the strain matrices, the moment interpolation and the flexibility of
// each point and place again for each; the operations on the displacements are those of before, in the same order
// (the moment parameters, the same for every point, made once), so their bits.
// Measured on the recovery of a plate meshed with this element.
std::array<ShellPlaceState, 5> ShellQuad4DrillingPlate::placeStates(const PlaceOperators& operators, const PlaceThermal& thermal,
                                                           const std::array<double, 24>& q,
                                                           const std::array<std::int32_t, 4>& gridIds) const
{
    const std::array<double, 24> qPlate = onProjections(q);
    const std::array<double, 12> u12 = bendingDofsOf(qPlate);

    // Membrane strains and bending moments (of the hybrid field, M = D·κ_mechanical) at the centre and the Gauss
    // points; without bending, the compatible curvatures of the bilinear rotations take the place of the moments'
    // curvatures.
    std::array<std::array<double, 3>, 5> strains{}; // the centre, then the Gauss points
    std::array<std::array<double, 3>, 5> moments{};
    if (m_membrane)
    {
        const std::array<double, 12> membraneDofs = membraneDofsOf(qPlate);
        for (std::size_t k = 0; k < 5; ++k)
            strains[k] = operators.membrane[k] * membraneDofs;
    }
    if (m_bending)
    {
        const std::array<double, 5> parameters = m_bending->momentParameters(u12, thermal.initialMoments);
        for (std::size_t k = 0; k < 5; ++k)
            moments[k] = operators.moment[k] * parameters;
    }
    const std::array<double, 2> shear = m_bending ? m_bending->shearForces(u12) : std::array<double, 2>{};
    const std::array<std::array<double, 3>, 4> strainAtPoints{strains[1], strains[2], strains[3], strains[4]};
    const std::array<std::array<double, 3>, 4> momentAtPoints{moments[1], moments[2], moments[3], moments[4]};

    std::array<ShellPlaceState, 5> states{};
    for (std::size_t place = 0; place < 5; ++place)
    {
        ShellPlaceState& state = states[place];
        const bool isCentre = place == 0;
        const std::size_t corner = isCentre ? 0 : place - 1;
        state.nodeId = isCentre ? 0 : gridIds[corner];
        state.thickness = operators.thickness[place];
        state.temperature = thermal.temperature[place];
        state.membraneStrain =
            isCentre ? strains[0] : extrapolatedToCorner(strainAtPoints, operators.extrapolation[corner]);
        const ThermalResultants& resultants = thermal.resultants[place];
        if (m_bending)
        {
            const std::array<double, 3> moment =
                isCentre ? moments[0] : extrapolatedToCorner(momentAtPoints, operators.extrapolation[corner]);
            // The total curvature (ShellQuad4DrillingBending's sign) is D⁻¹ (M_h − M_T): the mechanical part of the hybrid
            // moments plus the thermal curvature −D⁻¹ M_T.
            std::array<double, 3> total{};
            for (std::size_t c = 0; c < 3; ++c)
                total[c] = moment[c] - resultants.moment[c];
            state.curvature = operators.flexibility[place] * total;
            // The resultant ∫σ·z dz is −M_h (ShellQuad4DrillingBending's curvature sign, see ShellRecovery.hpp).
            for (std::size_t c = 0; c < 3; ++c)
                state.moment[c] = momentResultant(moment[c]);
            state.shearForce = shear;
        }
        else
        {
            // No bending stiffness: the curvatures of the bilinear rotations, for the strains at the fibres.
            const PlaneJacobian& jacobian = operators.jacobian[place];
            for (std::size_t i = 0; i < 4; ++i)
            {
                const double thetaX = u12[3 * i + 1];
                const double thetaY = u12[3 * i + 2];
                state.curvature[0] -= jacobian.dX[i] * thetaY;
                state.curvature[1] += jacobian.dY[i] * thetaX;
                state.curvature[2] += jacobian.dX[i] * thetaX - jacobian.dY[i] * thetaY;
            }
        }
        if (m_membrane)
        {
            // N = t G1 ε − N_T.
            const std::array<double, 3> force = m_section.membraneMatrix * state.membraneStrain;
            for (std::size_t c = 0; c < 3; ++c)
                state.membraneForce[c] = state.thickness * force[c] - resultants.force[c];
        }
    }
    return states;
}

base::MatN<24> ShellQuad4DrillingPlate::mass(MassFormulation formulation, double areaMass) const
{
    return massOf(m_geometry, m_section, m_thickness, formulation, areaMass);
}

base::MatN<24> ShellQuad4DrillingPlate::massThicknessRate(MassFormulation formulation, double thickness) const
{
    return massThicknessRateOf(m_geometry, m_section, m_thickness, formulation, thickness);
}

base::MatN<24> ShellQuad4DrillingPlate::massOf(const ShellQuad4DrillingGeometry& geometry, const ShellSection& section,
                                      const std::array<double, 4>& thickness, MassFormulation formulation,
                                      double areaMass)
{
    // The consistent mass of the bilinear functions, M_ij = ∫ m N_i N_j dA with m = ρ t(ξ, η) + NSM, Gauss 2 × 2, and
    // the rotary inertia ∫ ρ t³/12 N_i N_j dA.
    const double nonStructural = section.nonStructuralMass + areaMass;
    return massWith(
        geometry, thickness, formulation, [&](double t) { return section.density * t + nonStructural; },
        [&](double t) { return section.density * t * t * t / 12.0; });
}

base::MatN<24> ShellQuad4DrillingPlate::massThicknessRateOf(const ShellQuad4DrillingGeometry& geometry, const ShellSection& section,
                                                   const std::array<double, 4>& thickness, MassFormulation formulation,
                                                   double nominalThickness)
{
    // t(ξ, η) = r(ξ, η)·T: d(ρ t)/dT = ρ t/T, d(ρ t³/12)/dT = ρ t³/(4 T).
    return massWith(
        geometry, thickness, formulation, [&](double t) { return section.density * t / nominalThickness; },
        [&](double t) { return section.density * t * t * t / (4.0 * nominalThickness); });
}

base::MatN<24> ShellQuad4DrillingPlate::massThicknessSecondRateOf(const ShellQuad4DrillingGeometry& geometry, const ShellSection& section,
                                                         const std::array<double, 4>& thickness,
                                                         MassFormulation formulation, double nominalThickness)
{
    // t(ξ, η) = r(ξ, η)·T: d²(ρ t)/dT² = 0, d²(ρ t³/12)/dT² = ρ t³/(2 T²).
    return massWith(
        geometry, thickness, formulation, [](double) { return 0.0; },
        [&](double t) { return section.density * t * t * t / (2.0 * nominalThickness * nominalThickness); });
}

template <typename Areal, typename Inertia>
base::MatN<24> ShellQuad4DrillingPlate::massWith(const ShellQuad4DrillingGeometry& geometry, const std::array<double, 4>& thickness,
                                        MassFormulation formulation, const Areal& arealOf, const Inertia& inertiaOf)
{
    base::MatN<4> translational;
    base::MatN<4> rotary;
    std::array<double, 4> rowSums{};
    for (const QuadraturePoint& point : gaussRule2x2())
    {
        const BilinearShape shape = bilinearShapeAt(point.xi, point.eta);
        const double detJ = geometry.jacobianAt(point.xi, point.eta).determinant;
        double t = 0.0;
        for (std::size_t i = 0; i < 4; ++i)
            t += shape.n[i] * thickness[i];
        const double areal = arealOf(t);
        const double inertia = inertiaOf(t);
        for (std::size_t i = 0; i < 4; ++i)
        {
            rowSums[i] += point.weight * areal * shape.n[i] * detJ;
            for (std::size_t j = 0; j < 4; ++j)
            {
                translational(i, j) += point.weight * areal * shape.n[i] * shape.n[j] * detJ;
                rotary(i, j) += point.weight * inertia * shape.n[i] * shape.n[j] * detJ;
            }
        }
    }
    base::MatN<24> m;
    if (formulation == MassFormulation::Lumped)
    {
        // The lumped mass: the row sums of the consistent translational mass, m_i = ∫ m N_i dA (positive for the
        // bilinear functions; A/4 each on a parallelogram); no rotary inertia. Not the HRZ diagonal scaled to the total
        // (Hinton, Rock, Zienkiewicz), which a distorted element tells apart: the row sums keep the first moments of
        // the consistent mass (Σ N_i x_i = x), so the rigid-body mass matrix of a mesh is exact, where HRZ moves them.
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                m(6 * i + d, 6 * i + d) = rowSums[i];
        return m;
    }
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
        {
            for (std::size_t d = 0; d < 3; ++d)
                m(6 * i + d, 6 * j + d) = translational(i, j);
            for (std::size_t d = 3; d < 5; ++d)
                m(6 * i + d, 6 * j + d) = rotary(i, j);
        }
    return m;
}

std::array<ShellPlaceState, 5> ShellQuad4DrillingPlate::placeStateRates(const std::array<ShellPlaceState, 5>& states,
                                                               const std::array<double, 24>& q, double thickness) const
{
    const std::array<double, 2> shearRates =
        m_bending ? m_bending->shearForceRates(bendingDofsOf(onProjections(q)), thickness) : std::array<double, 2>{};
    std::array<ShellPlaceState, 5> rates;
    for (std::size_t place = 0; place < states.size(); ++place)
        rates[place] = shellResultantRates(states[place], m_section, IS_COUPLED, shearRates, thickness);
    return rates;
}

std::array<ShellPlaceState, 5> ShellQuad4DrillingPlate::placeStateSecondRates(const std::array<ShellPlaceState, 5>& states,
                                                                     const std::array<double, 24>& q,
                                                                     double thickness) const
{
    const std::array<double, 2> shearSecondRates =
        m_bending ? m_bending->shearForceSecondRates(bendingDofsOf(onProjections(q)), thickness)
                  : std::array<double, 2>{};
    std::array<ShellPlaceState, 5> rates;
    for (std::size_t place = 0; place < states.size(); ++place)
        rates[place] = shellResultantSecondRates(states[place], m_section, IS_COUPLED, shearSecondRates, thickness);
    return rates;
}

base::MatN<24> ShellQuad4DrillingPlate::thicknessSecondRate(double thickness) const
{
    base::MatN<24> rate;
    if (m_bending)
        addOnBendingDofs(rate, m_bending->thicknessSecondRate(thickness));
    return onGridPoints(rate);
}

base::MatN<24> ShellQuad4DrillingPlate::thicknessRate(double thickness) const
{
    base::MatN<24> rate;
    if (m_bending)
        addOnBendingDofs(rate, m_bending->thicknessRate(thickness));
    if (m_membrane)
    {
        const base::MatN<12>& membrane = m_membrane->stiffness();
        for (std::size_t a = 0; a < 12; ++a)
            for (std::size_t b = 0; b < 12; ++b)
                rate(membraneDof(a), membraneDof(b)) += membrane(a, b) / thickness;
    }
    return onGridPoints(rate);
}

std::array<double, 24> ShellQuad4DrillingPlate::pressureLoads(const std::array<double, 4>& cornerPressures,
                                                     const std::optional<base::Vec3>& direction) const
{
    // F_i = ∫ N_i p dA, p = Σ_j N_j p_j (Gauss 2 × 2, exact for the bilinear products on a parallelogram), along +n or
    // along the given unit vector.
    std::array<double, 4> nodal{};
    for (const QuadraturePoint& point : gaussRule2x2())
    {
        const BilinearShape shape = bilinearShapeAt(point.xi, point.eta);
        const double detJ = m_geometry.jacobianAt(point.xi, point.eta).determinant;
        double pressure = 0.0;
        for (std::size_t j = 0; j < 4; ++j)
            pressure += shape.n[j] * cornerPressures[j];
        for (std::size_t i = 0; i < 4; ++i)
            nodal[i] += point.weight * pressure * shape.n[i] * detJ;
    }
    std::array<double, 24> loads{};
    const base::Vec3 along = direction.value_or(base::Vec3{0.0, 0.0, 1.0});
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t d = 0; d < 3; ++d)
            loads[6 * i + d] = nodal[i] * along[d];
    return loadsOnGridPoints(loads);
}

ShellQuad4DrillingPlate::SectionRate ShellQuad4DrillingPlate::sectionRate(const ShellSectionRate& rate, bool withStiffness) const
{
    SectionRate out;
    if (m_bending && (rate.movesBending || rate.movesShear))
    {
        const std::array<double, 2> shearRate =
            rate.movesShear ? transverseShearStiffnessAlongAxesRate(m_section, rate) : std::array<double, 2>{};
        out.bending = m_bending->sectionRate(rate.bendingMatrix, TransverseShear{shearRate}, withStiffness);
        if (withStiffness)
            addOnBendingDofs(out.stiffness, out.bending->stiffness);
    }
    if (!withStiffness)
        return out;
    if (m_membrane && rate.movesMembrane)
    {
        // K = Σ_g w det J t (Bᵀ C B + γ hᵀ h), γ = G: linear in (C, G) together.
        const ShellQuad4DrillingMembrane membrane(m_geometry, rate.membraneMatrix, m_thickness,
                                                rate.membraneShearModulus);
        addOnMembraneDofs(out.stiffness, membrane.stiffness());
    }
    out.stiffness = onGridPoints(out.stiffness);
    return out;
}

std::array<ShellPlaceState, 5> ShellQuad4DrillingPlate::placeStateSectionRates(const std::array<ShellPlaceState, 5>& states,
                                                                      const std::array<double, 24>& q,
                                                                      const ShellSectionRate& rate,
                                                                      const SectionRate& plateRate) const
{
    const std::array<double, 12> u12 = bendingDofsOf(onProjections(q));
    const std::array<QuadraturePoint, 4> points = gaussRule2x2();
    // The rates of the hybrid moments at the centre and the Gauss points (placeStates).
    std::array<std::array<double, 3>, 5> momentRates{};
    if (m_bending && plateRate.bending)
    {
        const std::array<double, 5> parameterRates = plateRate.bending->momentParameters * u12;
        for (std::size_t k = 0; k < 5; ++k)
        {
            const double xi = k == 0 ? 0.0 : points[k - 1].xi;
            const double eta = k == 0 ? 0.0 : points[k - 1].eta;
            momentRates[k] = m_bending->momentInterpolationAt(xi, eta) * parameterRates;
        }
    }
    const std::array<double, 2> shearRates = m_bending && plateRate.bending
                                                 ? m_bending->shearForceRatesOf(plateRate.bending->coefficients, u12)
                                                 : std::array<double, 2>{};
    const std::array<std::array<double, 3>, 4> momentAtPoints{momentRates[1], momentRates[2], momentRates[3],
                                                              momentRates[4]};
    std::array<ShellPlaceState, 5> rates{};
    for (std::size_t place = 0; place < 5; ++place)
    {
        const ShellPlaceState& state = states[place];
        ShellPlaceState& out = rates[place];
        const bool isCentre = place == 0;
        const std::size_t corner = isCentre ? 0 : place - 1;
        out.nodeId = state.nodeId;
        const double t = state.thickness;
        // The membrane strains have no rate (out.membraneStrain stays zero): the strain matrix is kinematic.
        if (m_bending)
        {
            const std::array<double, 3> momentRate =
                isCentre ? momentRates[0] : extrapolatedToCorner(momentAtPoints, extrapolationWeights(corner));
            // κ = D⁻¹ M_h: dκ = D⁻¹ (dM_h − dD κ).
            const base::Mat3 flexibility = inverted(m_section.bendingStiffness(t));
            const std::array<double, 3> stiffnessTerm =
                (m_section.bendingRatio * t * t * t / 12.0 * rate.bendingMatrix) * state.curvature;
            std::array<double, 3> net{};
            for (std::size_t c = 0; c < 3; ++c)
                net[c] = momentRate[c] - stiffnessTerm[c];
            out.curvature = flexibility * net;
            // M = −M_h.
            for (std::size_t c = 0; c < 3; ++c)
                out.moment[c] = momentResultant(momentRate[c]);
            out.shearForce = shearRates;
        }
        if (m_membrane)
        {
            // N = t G1 ε at fixed ε: dN = t dG1 ε.
            const std::array<double, 3> fromMatrix = rate.membraneMatrix * state.membraneStrain;
            for (std::size_t c = 0; c < 3; ++c)
                out.membraneForce[c] = t * fromMatrix[c];
        }
    }
    return rates;
}

base::MatN<24> ShellQuad4DrillingPlate::differentialStiffness(const std::array<double, 3>& membraneForce) const
{
    base::MatN<24> k;
    for (const QuadraturePoint& point : gaussRule2x2())
    {
        const PlaneJacobian jacobian = m_geometry.jacobianAt(point.xi, point.eta);
        addMembraneDifferentialStiffness<4, 6>(k, jacobian.dX, jacobian.dY, point.weight * jacobian.determinant,
                                               membraneForce);
    }
    return onGridPoints(k);
}

} // namespace oos::elements
