#include "BeamSizing.hpp"

namespace oos::elements::beam_sizing
{
namespace
{

struct Forces
{
    MixedScalar axial, shear1, shear2, torque, moment1, moment2;
};

Forces forcesAt(const std::array<MixedScalar, 12>& ends, const Station& station, double x,
                const LoadResultant& left = {})
{
    Forces result;
    result.axial = -ends[0] - left.force[0];
    result.shear1 = -ends[1] - left.force[1];
    result.shear2 = -ends[2] - left.force[2];
    result.torque = -ends[3] - left.moment[0];
    result.moment1 = -ends[5] + x * ends[1] - left.moment[2] + station.neutral[0] * result.axial;
    result.moment2 = ends[4] + x * ends[2] + left.moment[1] + station.neutral[1] * result.axial;
    return result;
}

std::array<MixedScalar, 4> bendingStress(const Station& station, const Forces& forces)
{
    std::array<MixedScalar, 4> result{};
    for (std::size_t k = 0; k < result.size(); ++k)
    {
        const auto y = station.points[k][0];
        const auto z = station.points[k][1];
        if (station.inertia1.value > 0.0 && station.inertia2.value > 0.0)
        {
            const auto determinant = station.inertia1 * station.inertia2 -
                                     station.productInertia * station.productInertia;
            result[k] = (z * (-station.inertia1 * forces.moment2 + station.productInertia * forces.moment1) -
                         y * (-station.productInertia * forces.moment2 + station.inertia2 * forces.moment1)) /
                        determinant;
        }
        else
        {
            if (station.inertia2.value > 0.0) result[k] -= z * forces.moment2 / station.inertia2;
            if (station.inertia1.value > 0.0) result[k] -= y * forces.moment1 / station.inertia1;
        }
    }
    return result;
}

MixedScalar axialStress(const Station& station, const Forces& forces)
{
    return station.area.value > 0.0 ? forces.axial / station.area : MixedScalar{};
}

} // namespace

std::vector<RecoveredRow> Kernel::recover(const ElementBatchView& view, ElementOutput output,
                                         std::span<const double> grid, MixedScalar thermalStrain,
                                         std::span<const BeamLineLoad> lineLoads) const
{
    const auto ends = endForces(grid, thermalStrain, lineLoads);
    const auto e = MixedScalar(m_beam.material.youngModulus, m_first.material.youngModulus,
                                m_other.material.youngModulus);
    const bool strains = output == ElementOutput::Strains;
    const double length = m_beam.geometry.length;
    if (view.card() == "Bar2")
    {
        const auto a = at(0.0), b = at(1.0);
        const auto forceA = forcesAt(ends, a, 0.0, loadResultant(lineLoads, 0.0, 0.0, 0.0, false));
        const auto forceB = forcesAt(ends, b, length, loadResultant(lineLoads, 0.0, length, length, true));
        RecoveredRow row;
        if (output == ElementOutput::Forces)
        {
            row.count = 8;
            row.values = {forceA.moment1, forceA.moment2, forceB.moment1, forceB.moment2,
                           forceA.shear1, forceA.shear2, forceA.axial, forceA.torque};
        }
        else
        {
            row.count = 15;
            const auto stressA = bendingStress(a, forceA), stressB = bendingStress(b, forceB);
            row.values[4] = axialStress(a, forceA);
            for (std::size_t k = 0; k < 4; ++k)
            {
                row.values[k] = stressA[k];
                row.values[8 + k] = stressB[k];
            }
            if (strains)
                for (auto& value : row.values)
                    value = e.value > 0.0 ? value / e : MixedScalar{};
        }
        return {row};
    }
    std::vector<RecoveredRow> rows;
    std::int32_t layer = 0;
    const auto ids = view.snapshot().nodes().ids();
    for (std::size_t s = 0; s < m_beam.section.count; ++s)
    {
        const auto& base = m_beam.section.stations[s];
        if (!base.hasOutput)
            continue;
        RecoveredRow row;
        row.place = {s == 0 ? ids[static_cast<std::size_t>(m_beam.geometry.rowA)]
                     : s + 1 == m_beam.section.count ? ids[static_cast<std::size_t>(m_beam.geometry.rowB)] : 0,
                     ++layer, base.position};
        const auto station = at(base.position);
        const double x = base.position * length;
        const auto left = loadResultant(lineLoads, 0.0, x, x, x >= length);
        const auto forces = forcesAt(ends, station, x, left);
        if (output == ElementOutput::Forces)
        {
            row.count = 7;
            row.values = {forces.moment1, forces.moment2, forces.shear1, forces.shear2,
                           forces.axial, forces.torque, 0.0};
        }
        else
        {
            row.count = 8;
            const auto bending = bendingStress(station, forces);
            const auto axial = axialStress(station, forces);
            for (std::size_t k = 0; k < 4; ++k)
                row.values[k] = strains ? (e.value > 0.0 ? (axial + bending[k]) / e : MixedScalar{})
                                        : axial + bending[k];
        }
        rows.push_back(row);
    }
    return rows;
}

} // namespace oos::elements::beam_sizing
