#include "oos/output/monitors/MonitorPoint3.hpp"

#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"

#include <algorithm>
#include <string>

namespace oos::output
{

namespace
{

// The frame of CID or CD: a rectangular coordinate system of the snapshot (the basic system for 0).
input::CoordinateFrame rectangularFrame(const input::ModelSnapshot& snapshot, std::int32_t frameId,
                                        const MonitorPoint3Definition& definition, const char* field)
{
    const input::CoordinateFrame frame = aero::frameOf(snapshot, frameId, "MonitorNodeForceSum3");
    if (frame.kind != input::CoordinateFrame::Kind::Rectangular)
        throw base::SolverError(
            base::concatenate({"MonitorNodeForceSum3 ", definition.name, ": expected a rectangular ", field,
                               " (a RectFrame1 or RectFrame2), found the cylindrical or spherical system ",
                               std::to_string(frameId), "; give the point and the axes in a rectangular system"}));
    return frame;
}

bool isExcluded(const MonitorPoint3Definition& definition, std::string_view letters)
{
    return std::any_of(letters.begin(), letters.end(),
                       [&](char letter) { return definition.excluded.find(letter) != std::string::npos; });
}

} // namespace

MonitorPoint3::MonitorPoint3(MonitorPoint3Definition definition, const input::ModelSnapshot& snapshot,
                             const assembly::ElementSet& set, const assembly::DofMap& dofMap)
    : m_definition(std::move(definition)), m_dofMap(&dofMap)
{
    if (m_definition.grids.empty())
        throw base::SolverError(
            base::concatenate({"MonitorNodeForceSum3 ", m_definition.name,
                               ": expected the grid points of its MonitorNodeForceSum3.nodeSetId, found "
                               "none; correct the IdListSet1 its MonitorNodeForceSum3.nodeSetId names"}));
    const input::CoordinateFrame positionFrame =
        rectangularFrame(snapshot, m_definition.positionFrameId, m_definition, "positionFrameId");
    m_point = positionFrame.origin + positionFrame.axes * m_definition.pointInPositionFrame;
    m_cdAxes = rectangularFrame(snapshot, m_definition.displacementFrameId, m_definition,
                                "MonitorNodeForceSum3.displacementFrameId")
                   .axes;
    // AXES: the digits 1 to 6 of the components reported.
    for (std::int32_t digits = m_definition.axes; digits > 0; digits /= 10)
    {
        const std::int32_t component = digits % 10;
        if (component >= 1 && component <= 6)
            m_isReported[static_cast<std::size_t>(component - 1)] = true;
    }
    m_addsApplied = !isExcluded(m_definition, "ALP");
    m_addsSpc = !isExcluded(m_definition, "S");
    m_addsMpc = !isExcluded(m_definition, "M");
    for (const std::int32_t grid : m_definition.grids)
    {
        const std::optional<std::int32_t> row = dofMap.points().findGrid(grid);
        if (!row)
            throw base::SolverError(
                base::concatenate({"MonitorNodeForceSum3 ", m_definition.name,
                                   ": its MonitorNodeForceSum3.nodeSetId names the grid point ", std::to_string(grid),
                                   ", which the model does not have; correct the IdListSet1"}));
        m_rows.push_back(*row);
    }
    std::sort(m_rows.begin(), m_rows.end());
    m_rows.erase(std::unique(m_rows.begin(), m_rows.end()), m_rows.end());
    if (m_definition.elements.empty())
        return;
    CutDefinition cut;
    cut.id = 1;
    cut.name = base::concatenate({"MonitorNodeForceSum3 ", m_definition.name});
    cut.grids = m_definition.grids;
    cut.elements = m_definition.elements;
    cut.point = m_definition.pointInPositionFrame;
    cut.positionFrameId = m_definition.positionFrameId;
    cut.displacementFrameId = m_definition.displacementFrameId;
    base::DiagnosticSink sink;
    m_cut = std::make_unique<CutLoads>(std::vector<CutDefinition>{std::move(cut)}, snapshot, set, dofMap, sink);
    if (sink.hasErrors())
    {
        std::string problems;
        for (const base::Diagnostic& diagnostic : sink.diagnostics())
            problems.append(problems.empty() ? "" : "; ").append(diagnostic.message);
        // The remedy given by each diagnostic, in `problems`.
        throw base::SolverError(base::concatenate({"MonitorNodeForceSum3 ", m_definition.name, ": ", problems}));
    }
}

std::array<double, 6> MonitorPoint3::loadsOf(const assembly::ElementSet& set, std::span<const double> displacementsG,
                                             const MonitorNodalLoads& nodal,
                                             const elements::ElementTemperatures* temperatures) const
{
    std::array<double, 6> loads{};
    if (m_cut)
        loads = m_cut->evaluateCut(0, set, displacementsG, temperatures);
    base::Vec3 force;
    base::Vec3 moment;
    for (const std::int32_t row : m_rows)
    {
        base::Vec3 rowForce;
        base::Vec3 rowMoment;
        for (const auto& [values, isAdded] :
             {std::make_pair(nodal.applied, m_addsApplied), std::make_pair(nodal.spc, m_addsSpc),
              std::make_pair(nodal.mpc, m_addsMpc)})
        {
            if (!isAdded || values.empty())
                continue;
            for (int c = 0; c < 3; ++c)
            {
                rowForce[static_cast<std::size_t>(c)] +=
                    values[static_cast<std::size_t>(m_dofMap->gridDof(row, 1 + c))];
                rowMoment[static_cast<std::size_t>(c)] +=
                    values[static_cast<std::size_t>(m_dofMap->gridDof(row, 4 + c))];
            }
        }
        const base::Mat3 cdToBasic = m_dofMap->cdToBasic(row);
        const base::Vec3 forceInBasic = cdToBasic * rowForce;
        force = force + forceInBasic;
        moment = moment + base::cross(m_dofMap->positionInBasic(row) - m_point, forceInBasic) + cdToBasic * rowMoment;
    }
    const base::Mat3 basicToCd = m_cdAxes.transposed();
    const base::Vec3 f = basicToCd * force;
    const base::Vec3 m = basicToCd * moment;
    for (std::size_t c = 0; c < 3; ++c)
    {
        loads[c] += f[c];
        loads[3 + c] += m[c];
    }
    return loads;
}

} // namespace oos::output
