#include "oos/aero/outputs/HingeMoments.hpp"

namespace oos::aero
{

HingeMoments::HingeMoments(const AeroBoxes& boxes, const TrimVariables& variables)
{
    for (const TrimVariables::ControlSurface& source : variables.controlSurfaces())
    {
        Surface surface;
        surface.label = source.label;
        surface.referenceProduct = source.referenceChord * source.referenceArea;
        surface.boxes = source.boxes;
        for (const std::size_t j : source.boxes)
            surface.factors.push_back(
                boxes[j].area *
                base::dot(base::cross(boxes[j].forcePoint - source.hingeOrigin, boxes[j].normal), source.hinge));
        m_surfaces.push_back(std::move(surface));
    }
}

std::vector<double> HingeMoments::momentsOf(std::span<const double> pressures, double dynamicPressure) const
{
    std::vector<double> moments;
    moments.reserve(m_surfaces.size());
    for (const Surface& surface : m_surfaces)
    {
        double moment = 0.0;
        for (std::size_t k = 0; k < surface.boxes.size(); ++k)
            moment += surface.factors[k] * pressures[surface.boxes[k]];
        moments.push_back(dynamicPressure * moment);
    }
    return moments;
}

std::vector<double> HingeMoments::coefficientsOf(std::span<const double> pressures) const
{
    std::vector<double> coefficients = momentsOf(pressures, 1.0);
    for (std::size_t s = 0; s < m_surfaces.size(); ++s)
        coefficients[s] /= m_surfaces[s].referenceProduct;
    return coefficients;
}

} // namespace oos::aero
