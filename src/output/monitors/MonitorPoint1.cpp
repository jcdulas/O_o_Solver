#include "oos/output/monitors/MonitorPoint1.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"

#include <string>

namespace oos::output
{

namespace
{

// The frame of CP or CD: a rectangular coordinate system of the snapshot (the basic system for 0).
input::CoordinateFrame rectangularFrame(const input::ModelSnapshot& snapshot, std::int32_t frameId,
                                        const MonitorPoint1Definition& definition, const char* field)
{
    const input::CoordinateFrame frame = aero::frameOf(snapshot, frameId, "MonitorIntegratedLoad1");
    if (frame.kind != input::CoordinateFrame::Kind::Rectangular)
        throw base::SolverError(base::concatenate(
            {"MonitorIntegratedLoad1 ", definition.name, ": expected a rectangular ", field,
             " (a RectFrame1 or RectFrame2), found the ", "cylindrical or spherical system ", std::to_string(frameId),
             "; give the point and the axes in a rectangular "
             "system"}));
    return frame;
}

} // namespace

MonitorPoint1::MonitorPoint1(const MonitorPoint1Definition& definition, const input::ModelSnapshot& snapshot,
                             const assembly::DofMap& dofMap, const aero::AeroReference& reference,
                             const aero::AeroBoxes& boxes)
    : m_definition(definition), m_dofMap(&dofMap), m_boxes(&boxes)
{
    const input::CoordinateFrame positionFrame =
        rectangularFrame(snapshot, definition.positionFrameId, definition, "MonitorIntegratedLoad1.positionFrameId");
    m_point = positionFrame.origin + positionFrame.axes * definition.pointInPositionFrame;
    m_cdAxes = rectangularFrame(snapshot, definition.displacementFrameId, definition,
                                "MonitorIntegratedLoad1.displacementFrameId")
                   .axes;
    m_aeroAxes = reference.aeroFrame.axes;
    m_aeroOrigin = reference.aeroFrame.origin;
    // AXES: the digits 1 to 6 of the components reported.
    for (std::int32_t digits = definition.axes; digits > 0; digits /= 10)
    {
        const std::int32_t component = digits % 10;
        if (component >= 1 && component <= 6)
            m_isReported[static_cast<std::size_t>(component - 1)] = true;
    }
    for (const std::int32_t id : definition.ids)
    {
        if (definition.isAerodynamic)
        {
            const std::optional<std::size_t> box = boxes.find(id);
            if (!box)
                throw base::SolverError(base::concatenate(
                    {"MonitorIntegratedLoad1 ", definition.name, ": its component ", definition.component,
                     " names the box ", std::to_string(id),
                     ", which no AeroLiftingPanel1 of the model has; correct the AeroBoxList or CAERO list of "
                     "the AeroComponent"}));
            m_boxIndices.push_back(*box);
        }
        else if (const std::optional<std::int32_t> row = dofMap.points().findGrid(id))
            m_rows.push_back(*row);
        else
            throw base::SolverError(
                base::concatenate({"MonitorIntegratedLoad1 ", definition.name, ": its component ", definition.component,
                                   " names the grid point ", std::to_string(id),
                                   ", which the model does not have; correct the IdListSet1 of the AeroComponent"}));
    }
}

std::array<double, 6> MonitorPoint1::loadsOfStructure(std::span<const double> loadsG) const
{
    base::Vec3 force;
    base::Vec3 moment;
    for (const std::int32_t row : m_rows)
    {
        const base::Mat3 cdToBasic = m_dofMap->cdToBasic(row);
        base::Vec3 rowForce;
        base::Vec3 rowMoment;
        for (int c = 0; c < 3; ++c)
        {
            rowForce[static_cast<std::size_t>(c)] = loadsG[static_cast<std::size_t>(m_dofMap->gridDof(row, 1 + c))];
            rowMoment[static_cast<std::size_t>(c)] = loadsG[static_cast<std::size_t>(m_dofMap->gridDof(row, 4 + c))];
        }
        const base::Vec3 forceInBasic = cdToBasic * rowForce;
        force = force + forceInBasic;
        moment = moment + base::cross(m_dofMap->positionInBasic(row) - m_point, forceInBasic) + cdToBasic * rowMoment;
    }
    const base::Mat3 basicToCd = m_cdAxes.transposed();
    const base::Vec3 f = basicToCd * force;
    const base::Vec3 m = basicToCd * moment;
    return {f[0], f[1], f[2], m[0], m[1], m[2]};
}

std::array<double, 6> MonitorPoint1::loadsOfBoxes(std::span<const double> boxForces) const
{
    base::Vec3 force;
    base::Vec3 moment;
    for (const std::size_t j : m_boxIndices)
    {
        const aero::AeroBox& box = (*m_boxes)[j];
        const base::Vec3 boxForce = m_aeroAxes * (boxForces[j] * box.normal);
        const base::Vec3 point = m_aeroOrigin + m_aeroAxes * box.forcePoint;
        force = force + boxForce;
        moment = moment + base::cross(point - m_point, boxForce);
    }
    const base::Mat3 basicToCd = m_cdAxes.transposed();
    const base::Vec3 f = basicToCd * force;
    const base::Vec3 m = basicToCd * moment;
    return {f[0], f[1], f[2], m[0], m[1], m[2]};
}

} // namespace oos::output
