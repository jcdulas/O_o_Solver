#include "oos/api/AeroTransfer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace oos::api
{
namespace
{
void checkField(std::span<const double> values, std::size_t count, const char* field)
{
    if (values.size() != count)
        throw std::invalid_argument(std::string(field) + " expected " + std::to_string(count) +
                                    " values, received " + std::to_string(values.size()));
    if (std::any_of(values.begin(), values.end(), [](double value) { return !std::isfinite(value); }))
        throw std::invalid_argument(std::string(field) + " requires finite values");
}
} // namespace

AeroTransfer::AeroTransfer(const input::ModelSnapshot& snapshot)
    : m_reference(aero::AeroReference::read(snapshot)), m_boxes(snapshot, m_reference, m_budget),
      m_splines(snapshot, m_reference, m_boxes, m_budget)
{
    for (const auto& box : m_boxes.boxes())
        m_boxIds.push_back(box.id);
    for (const auto& spline : m_splines.splines())
        m_nodeIds.insert(m_nodeIds.end(), spline.nodes.begin(), spline.nodes.end());
    std::sort(m_nodeIds.begin(), m_nodeIds.end());
    m_nodeIds.erase(std::unique(m_nodeIds.begin(), m_nodeIds.end()), m_nodeIds.end());
    for (std::size_t i = 0; i < m_nodeIds.size(); ++i)
        m_nodeRows.emplace(m_nodeIds[i], i);
}

aero::BoxMotion AeroTransfer::boxMotion(std::span<const double> nodalMotion) const
{
    checkField(nodalMotion, 6 * m_nodeIds.size(), "nodal motion");
    return m_splines.boxMotion([&](std::int32_t id)
    {
        const auto u = nodalMotion.subspan(6 * m_nodeRows.at(id), 6);
        return aero::PointMotion{{u[0], u[1], u[2]}, {u[3], u[4], u[5]}};
    });
}

std::vector<double> AeroTransfer::structuralLoads(std::span<const double> boxForces) const
{
    checkField(boxForces, m_boxIds.size(), "box forces");
    std::vector<double> loads(6 * m_nodeIds.size());
    for (const auto& point : m_splines.structuralLoads(boxForces))
    {
        const auto offset = 6 * m_nodeRows.at(point.nodeId);
        for (std::size_t i = 0; i < 3; ++i)
        {
            loads[offset + i] += point.force[i];
            loads[offset + 3 + i] += point.moment[i];
        }
    }
    return loads;
}

} // namespace oos::api
