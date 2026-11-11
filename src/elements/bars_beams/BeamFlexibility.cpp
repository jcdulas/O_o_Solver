#include "oos/elements/bars_beams/BeamFlexibility.hpp"

#include "oos/elements/core/MatrixKernels.hpp"

#include <algorithm>
#include <vector>

namespace oos::elements
{

namespace
{

// The stress resultants and end-B components, in this order (the order of q and P).
constexpr std::size_t AXIAL = 0;
constexpr std::size_t SHEAR_Y = 1;
constexpr std::size_t SHEAR_Z = 2;
constexpr std::size_t TORSION = 3;
constexpr std::size_t MOMENT_Y = 4;
constexpr std::size_t MOMENT_Z = 5;

// Gauss-Legendre rule of 8 points on [−1, 1] (Abramowitz and Stegun, table 25.4).
constexpr std::array<double, 8> GAUSS_NODES{-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                                            -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,
                                            0.7966664774136267,  0.9602898564975363};
constexpr std::array<double, 8> GAUSS_WEIGHTS{0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                                              0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                                              0.2223810344533745, 0.1012285362903763};

// S(x): the resultants at x of the end-B loads.
base::MatN<6> resultantMap(double length, double x)
{
    base::MatN<6> s = base::MatN<6>::identity();
    s(MOMENT_Y, SHEAR_Z) = -(length - x);
    s(MOMENT_Z, SHEAR_Y) = length - x;
    return s;
}

bool isPositive(const BeamSection& section, double BeamStation::* member)
{
    return section.stations[0].*member > 0.0; // findSectionProblem: zero everywhere or positive everywhere
}

} // namespace

BeamFlexibility::BeamFlexibility(const BeamSection& section, const IsotropicMaterial& material, double length)
    : m_section(&section), m_material(material), m_length(length)
{
    const double e = material.youngModulus;
    const double g = material.shearModulus;
    m_isActive[AXIAL] = e > 0.0 && isPositive(section, &BeamStation::area);
    m_isActive[TORSION] = g > 0.0 && isPositive(section, &BeamStation::torsionalConstant);
    const bool isPlane1Active = e > 0.0 && isPositive(section, &BeamStation::inertia1);
    const bool isPlane2Active = e > 0.0 && isPositive(section, &BeamStation::inertia2);
    m_isActive[SHEAR_Y] = m_isActive[MOMENT_Z] = isPlane1Active;
    m_isActive[SHEAR_Z] = m_isActive[MOMENT_Y] = isPlane2Active;

    integrate({},
              [&](double x, double weight)
              {
                  const base::MatN<6> s = resultantMap(m_length, x);
                  m_flexibility += weight * congruence(complianceAt(x), s);
              });

    // k = F⁻¹ on the active components, zero elsewhere.
    base::MatN<6> active = m_flexibility;
    for (std::size_t i = 0; i < 6; ++i)
    {
        if (m_isActive[i])
            continue;
        for (std::size_t j = 0; j < 6; ++j)
            active(i, j) = active(j, i) = 0.0;
        active(i, i) = 1.0;
    }
    m_naturalStiffness = inverted(active);
    for (std::size_t i = 0; i < 6; ++i)
    {
        if (m_isActive[i])
            continue;
        for (std::size_t j = 0; j < 6; ++j)
            m_naturalStiffness(i, j) = m_naturalStiffness(j, i) = 0.0;
    }
    // Symmetric by construction up to the round-off of the inversion: made exactly symmetric, so that K is.
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = i + 1; j < 6; ++j)
            m_naturalStiffness(i, j) = m_naturalStiffness(j, i) =
                0.5 * (m_naturalStiffness(i, j) + m_naturalStiffness(j, i));
}

base::MatN<6> BeamFlexibility::complianceAt(double x) const
{
    const BeamStation station = m_section->at(x / m_length);
    const double e = m_material.youngModulus;
    const double g = m_material.shearModulus;
    base::MatN<6> c;
    if (m_isActive[AXIAL])
        c(AXIAL, AXIAL) = 1.0 / (e * station.area);
    if (m_isActive[TORSION])
        c(TORSION, TORSION) = 1.0 / (g * station.torsionalConstant);
    const double shearStiffness1 = m_section->shearFactor1 * station.area * g;
    const double shearStiffness2 = m_section->shearFactor2 * station.area * g;
    if (m_isActive[SHEAR_Y] && shearStiffness1 > 0.0)
        c(SHEAR_Y, SHEAR_Y) = 1.0 / shearStiffness1;
    if (m_isActive[SHEAR_Z] && shearStiffness2 > 0.0)
        c(SHEAR_Z, SHEAR_Z) = 1.0 / shearStiffness2;
    if (station.productOfInertia != 0.0)
    {
        // [χy; χz] = (1 / (E·(I1·I2 − I12²)))·[[I1, I12], [I12, I2]]·[My; Mz].
        const double determinant =
            e * (station.inertia1 * station.inertia2 - station.productOfInertia * station.productOfInertia);
        c(MOMENT_Y, MOMENT_Y) = station.inertia1 / determinant;
        c(MOMENT_Y, MOMENT_Z) = c(MOMENT_Z, MOMENT_Y) = station.productOfInertia / determinant;
        c(MOMENT_Z, MOMENT_Z) = station.inertia2 / determinant;
    }
    else
    {
        if (m_isActive[MOMENT_Y])
            c(MOMENT_Y, MOMENT_Y) = 1.0 / (e * station.inertia2);
        if (m_isActive[MOMENT_Z])
            c(MOMENT_Z, MOMENT_Z) = 1.0 / (e * station.inertia1);
    }
    if (!station.hasNeutralAxisOffset())
        return c;
    // The resultants s act on the element axis, the axis of the shear centres; the axial and bending compliances are
    // those of the neutral axis at (y_n, z_n) from it (the axial force at the neutral axis,
    // the shears and the torque at the shear centre). The moments about the neutral axis are those about the axis
    // less the moment of the axial force: My_n = My − z_n·N, Mz_n = Mz + y_n·N (the shears' moment is the torque's,
    // which twists about the shear centre). With s_n = E·s, the compliance about the axis is Eᵀ·c·E.
    base::MatN<6> toNeutralAxis = base::MatN<6>::identity();
    toNeutralAxis(MOMENT_Y, AXIAL) = -station.neutralAxis[1];
    toNeutralAxis(MOMENT_Z, AXIAL) = station.neutralAxis[0];
    return congruence(c, toNeutralAxis);
}

template <typename Visit>
void BeamFlexibility::integrate(std::span<const double> breaks, Visit&& visit) const
{
    // The segment ends: the stations, then the breaks, sorted (at most 11 stations and a few loads).
    std::vector<double> ends;
    ends.reserve(m_section->count + breaks.size());
    for (std::size_t s = 0; s < m_section->count; ++s)
        ends.push_back(m_section->stations[s].position * m_length);
    for (const double position : breaks)
        ends.push_back(std::clamp(position, 0.0, m_length));
    std::sort(ends.begin(), ends.end());
    const bool isUniform = m_section->isUniform();
    for (std::size_t k = 0; k + 1 < ends.size(); ++k)
    {
        const double a = ends[k];
        const double b = ends[k + 1];
        if (b <= a)
            continue;
        const int parts = isUniform ? 1 : BEAM_TAPER_SUBINTERVALS;
        const double partLength = (b - a) / parts;
        for (int part = 0; part < parts; ++part)
        {
            const double start = a + part * partLength;
            const double half = 0.5 * partLength;
            for (std::size_t point = 0; point < GAUSS_NODES.size(); ++point)
                visit(start + half * (1.0 + GAUSS_NODES[point]), half * GAUSS_WEIGHTS[point]);
        }
    }
}

base::MatN<12> BeamFlexibility::stiffness() const
{
    return congruence(m_naturalStiffness, cantileverDeformation(m_length));
}

void BeamFlexibility::visitIntegrationPoints(const std::function<void(double, double)>& visit,
                                            std::span<const double> breaks) const
{
    integrate(breaks, visit);
}

std::array<double, 12> BeamFlexibility::equivalentLoads(std::span<const BeamLineLoad> loads) const
{
    std::vector<double> breaks;
    breaks.reserve(2 * loads.size());
    for (const BeamLineLoad& load : loads)
        breaks.insert(breaks.end(), {load.start, load.end});
    // q0 = ∫ Sᵀ·c·s₀ dx, s₀(x) the resultants of the loads on [x, L] about x.
    std::array<double, 6> q0{};
    integrate(breaks,
              [&](double x, double weight)
              {
                  const LoadResultant right = loadResultant(loads, x, m_length, x, true);
                  const std::array<double, 6> s0{right.force[0],  right.force[1],  right.force[2],
                                                 right.moment[0], right.moment[1], right.moment[2]};
                  const std::array<double, 6> cs = complianceAt(x) * s0;
                  const std::array<double, 6> part = resultantMap(m_length, x).transposed() * cs;
                  for (std::size_t i = 0; i < 6; ++i)
                      q0[i] += weight * part[i];
              });
    const std::array<double, 6> endB = m_naturalStiffness * q0;
    std::array<double, 12> loadsAtEnds = cantileverDeformation(m_length).transposed() * endB;
    const LoadResultant total = loadResultant(loads, 0.0, m_length, 0.0, true);
    for (std::size_t i = 0; i < 3; ++i)
    {
        loadsAtEnds[i] += total.force[i];
        loadsAtEnds[3 + i] += total.moment[i];
    }
    return loadsAtEnds;
}

base::MatN<6, 12> cantileverDeformation(double length)
{
    base::MatN<6, 12> b;
    for (std::size_t i = 0; i < 6; ++i)
    {
        b(i, i) = -1.0;
        b(i, 6 + i) = 1.0;
    }
    b(SHEAR_Y, MOMENT_Z) = -length; // Δv = v_B − v_A − L·θz_A
    b(SHEAR_Z, MOMENT_Y) = length;  // Δw = w_B − w_A + L·θy_A
    return b;
}

} // namespace oos::elements
