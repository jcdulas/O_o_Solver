#include "BeamSizing.hpp"

namespace oos::elements::beam_sizing
{

// q0 = integral(S^T c s0 dx), f_end = B^T k q0 + [F_total; M_total; 0; 0].
// Load values and geometry stay fixed; only the section compliance and natural stiffness carry derivatives.
std::array<MixedScalar, 12> Kernel::lineEndLoads(std::span<const BeamLineLoad> loads) const
{
    std::array<MixedScalar, 12> result{};
    if (loads.empty())
        return result;
    std::vector<double> breaks;
    breaks.reserve(loads.size() * 2);
    for (const auto& load : loads)
        breaks.insert(breaks.end(), {load.start, load.end});
    const double length = m_beam.geometry.length;
    std::array<MixedScalar, 6> deformation{};
    m_flexibility.visitIntegrationPoints([&](double x, double weight)
    {
        const auto right = loadResultant(loads, x, length, x, true);
        const std::array<double, 6> resultant{right.force[0], right.force[1], right.force[2],
                                             right.moment[0], right.moment[1], right.moment[2]};
        const auto section = complianceAt(x / length);
        std::array<MixedScalar, 6> part{};
        for (std::size_t i = 0; i < part.size(); ++i)
            for (std::size_t j = 0; j < part.size(); ++j)
                part[i] += section(i, j) * resultant[j];
        // S^T maps section strains to the cantilever's end deformation (BeamFlexibility).
        part[2] -= (length - x) * part[4];
        part[1] += (length - x) * part[5];
        for (std::size_t i = 0; i < part.size(); ++i)
            deformation[i] += weight * part[i];
    }, breaks);
    std::array<MixedScalar, 6> force{};
    for (std::size_t i = 0; i < force.size(); ++i)
        for (std::size_t j = 0; j < force.size(); ++j)
            force[i] += m_natural(i, j) * deformation[j];
    const auto b = cantileverDeformation(length);
    for (std::size_t i = 0; i < result.size(); ++i)
        for (std::size_t j = 0; j < force.size(); ++j)
            result[i] += b(j, i) * force[j];
    const auto total = loadResultant(loads, 0.0, length, 0.0, true);
    for (std::size_t i = 0; i < 3; ++i)
    {
        result[i] += total.force[i];
        result[i + 3] += total.moment[i];
    }
    return result;
}

std::array<MixedScalar, 12> Kernel::loads(MixedScalar thermalStrain,
                                        std::span<const BeamLineLoad> lineLoads, bool onOffsets) const
{
    const auto raw = lineEndLoads(lineLoads);
    std::array<MixedScalar, 12> end{};
    for (std::size_t i = 0; i < end.size(); ++i)
    {
        // C^T condenses applied loads with the same differentiated release map as elastic stiffness.
        for (std::size_t j = 0; j < end.size(); ++j)
            end[i] += m_completion(j, i) * raw[j];
        end[i] += m_end(i, 6) * thermalStrain * m_beam.geometry.length;
    }
    const auto transform = onOffsets ? m_beam.geometry.endTransform() : m_beam.geometry.rotationTransform();
    std::array<MixedScalar, 12> result{};
    for (std::size_t i = 0; i < result.size(); ++i)
        for (std::size_t j = 0; j < result.size(); ++j)
            result[i] += transform(j, i) * end[j];
    return result;
}

} // namespace oos::elements::beam_sizing
