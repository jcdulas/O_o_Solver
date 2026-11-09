#include "oos/aero/trim/TrimVariables.hpp"

#include "oos/aero/trim/TrimCase.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace oos::aero
{

namespace
{

// The rigid-body variables of AERODYNAMIC_RIGID_BODY_LABELS, one by one.
constexpr std::string_view ANGLE_OF_ATTACK = AERODYNAMIC_RIGID_BODY_LABELS[0];
constexpr std::string_view SIDE_SLIP = AERODYNAMIC_RIGID_BODY_LABELS[1];
constexpr std::string_view ROLL_RATE = AERODYNAMIC_RIGID_BODY_LABELS[2];
constexpr std::string_view PITCH_RATE = AERODYNAMIC_RIGID_BODY_LABELS[3];
constexpr std::string_view YAW_RATE = AERODYNAMIC_RIGID_BODY_LABELS[4];
const base::Vec3 FLOW{1.0, 0.0, 0.0};

bool isAerodynamicRigidBody(std::string_view label)
{
    return std::find(AERODYNAMIC_RIGID_BODY_LABELS.begin(), AERODYNAMIC_RIGID_BODY_LABELS.end(), label) !=
           AERODYNAMIC_RIGID_BODY_LABELS.end();
}

} // namespace

TrimVariables::TrimVariables(const input::ModelSnapshot& snapshot, const AeroReference& reference,
                             const AeroBoxes& boxes)
    : m_boxes(&boxes), m_reference(reference)
{
    if (snapshot.hasTable(input::TableRole::ControlSurfaces))
    {
        const base::ContractReader& table = snapshot.aeroTable(input::TableRole::ControlSurfaces);
        const base::Utf8View labels = table.texts("label");
        const std::span<const std::int32_t> frameIds = table.values<std::int32_t>("frameId1");
        const std::span<const double> effectiveness = table.values<double>("effectiveness", base::NullPolicy::Allow);
        const base::ValidityView hasEffectiveness = table.validity("effectiveness");
        const bool hasReferences = table.hasColumn("refChord") && table.hasColumn("refArea");
        const std::span<const double> chords =
            hasReferences ? table.values<double>("refChord", base::NullPolicy::Allow) : std::span<const double>{};
        const std::span<const double> areas =
            hasReferences ? table.values<double>("refArea", base::NullPolicy::Allow) : std::span<const double>{};
        const base::ValidityView hasChord = hasReferences ? table.validity("refChord") : base::ValidityView{};
        const base::ValidityView hasArea = hasReferences ? table.validity("refArea") : base::ValidityView{};
        const std::span<const std::int32_t> ids = table.values<std::int32_t>("id");
        // A limit column, or nothing for a table without it.
        const auto limitColumn = [&](const char* name)
        {
            return table.hasColumn(name) ? table.values<double>(name, base::NullPolicy::Allow)
                                         : std::span<const double>{};
        };
        const auto limitValidity = [&](const char* name)
        { return table.hasColumn(name) ? table.validity(name) : base::ValidityView{}; };
        const std::span<const double> lowerPositions = limitColumn("positionLower");
        const std::span<const double> upperPositions = limitColumn("positionUpper");
        const std::span<const double> lowerMoments = limitColumn("hingeMomentLower");
        const std::span<const double> upperMoments = limitColumn("hingeMomentUpper");
        const base::ValidityView hasLowerPosition = limitValidity("positionLower");
        const base::ValidityView hasUpperPosition = limitValidity("positionUpper");
        const base::ValidityView hasLowerMoment = limitValidity("hingeMomentLower");
        const base::ValidityView hasUpperMoment = limitValidity("hingeMomentUpper");
        // A value of a limit column when the row has one (a NaN, O_o_Reader's blank real, counts as none).
        const auto limitOf = [](std::span<const double> column, const base::ValidityView& validity,
                                std::int64_t row) -> std::optional<double>
        {
            if (column.empty() || !validity.isValid(row) || std::isnan(column[static_cast<std::size_t>(row)]))
                return std::nullopt;
            return column[static_cast<std::size_t>(row)];
        };
        const base::ValidityView hasSecond = table.validity("frameId2");
        const base::ListView<std::int32_t> boxIds = table.lists<std::int32_t>("boxes", base::NullPolicy::Allow);
        const base::ValidityView hasBoxes = table.validity("boxes");
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            const auto index = static_cast<std::size_t>(row);
            ControlSurface surface;
            surface.id = ids[index];
            surface.frameId1 = frameIds[index];
            if (table.hasColumn("aeroListId1"))
                surface.aeroListId1 = table.values<std::int32_t>("aeroListId1", base::NullPolicy::Allow)[index];
            surface.label = std::string(labels[row]);
            surface.positionLower = limitOf(lowerPositions, hasLowerPosition, row).value_or(-DEFAULT_POSITION_LIMIT);
            surface.positionUpper = limitOf(upperPositions, hasUpperPosition, row).value_or(DEFAULT_POSITION_LIMIT);
            surface.hingeMomentLower = limitOf(lowerMoments, hasLowerMoment, row);
            surface.hingeMomentUpper = limitOf(upperMoments, hasUpperMoment, row);
            if (hasSecond.isValid(row))
                throw base::SolverError(base::concatenate({"AeroControlSurface ", surface.label,
                                                           ": a second component (AeroControlSurface.frameId2, "
                                                           "AeroControlSurface.boxListId2) is not supported yet; "
                                                           "split the surface into two AeroControlSurface"}));
            const input::CoordinateFrame hingeFrame =
                frameOf(snapshot, frameIds[index], "AeroControlSurface AeroControlSurface.frameId1");
            if (hingeFrame.kind != input::CoordinateFrame::Kind::Rectangular)
                throw base::SolverError(base::concatenate(
                    {"AeroControlSurface ", surface.label,
                     ": expected a rectangular AeroControlSurface.frameId1 (its y axis is the hinge line), "
                     "found a cylindrical or spherical system; use a RectFrame2"}));
            surface.hinge = base::normalized(reference.directionInAeroFrame(
                base::Vec3{hingeFrame.axes(0, 1), hingeFrame.axes(1, 1), hingeFrame.axes(2, 1)}));
            surface.hingeOrigin = reference.pointInAeroFrame(hingeFrame.origin);
            surface.effectiveness = hasEffectiveness.isValid(row) ? effectiveness[index] : 1.0;
            // AeroControlSurface: CREFC and CREFS default to 1.0.
            if (hasReferences && hasChord.isValid(row))
                surface.referenceChord = chords[index];
            if (hasReferences && hasArea.isValid(row))
                surface.referenceArea = areas[index];
            if (hasBoxes.isValid(row))
                for (const std::int32_t id : boxIds[row])
                    if (const std::optional<std::size_t> box = boxes.find(id))
                        surface.boxes.push_back(*box);
            m_variables.push_back({surface.label, true});
            m_surfaces.push_back(std::move(surface));
        }
    }
    // The rigid-body variables the model declares (AESTAT, in the trim cases), all of them without a TRIM.
    std::vector<std::string> declared;
    if (snapshot.hasTable(input::TableRole::TrimCases))
    {
        const base::ContractReader& table = snapshot.aeroTable(input::TableRole::TrimCases);
        const base::Utf8View labels = table.texts("variable");
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            std::string label = std::string(labels[row]);
            if (isAerodynamicRigidBody(label) && std::find(declared.begin(), declared.end(), label) == declared.end())
                declared.push_back(std::move(label));
        }
    }
    for (const std::string_view label : AERODYNAMIC_RIGID_BODY_LABELS)
        if (declared.empty() || std::find(declared.begin(), declared.end(), label) != declared.end())
            m_variables.push_back({std::string(label), false});
    // The AeroRigidTrimVariable IDs, for the results (they give the trim variables by ID).
    if (const input::CardTable* statistics = snapshot.findAeroCardTable("AeroRigidTrimVariable"))
    {
        const base::ContractReader& reader = statistics->reader();
        if (reader.hasColumn("id") && reader.hasColumn("label"))
        {
            const std::span<const std::int32_t> ids = reader.values<std::int32_t>("id");
            const base::Utf8View labels = reader.texts("label");
            for (std::int64_t row = 0; row < reader.rowCount(); ++row)
                m_statisticIds.emplace_back(std::string(labels[row]), ids[static_cast<std::size_t>(row)]);
        }
    }
}

std::int32_t TrimVariables::idOf(std::string_view label) const
{
    for (const ControlSurface& surface : m_surfaces)
        if (surface.label == label)
            return surface.id;
    for (const auto& [name, id] : m_statisticIds)
        if (name == label)
            return id;
    return 0;
}

void TrimVariables::normalwashOf(std::string_view labelName, std::span<double> normalwash) const
{
    const std::string label = std::string(labelName);
    const AeroBoxes& boxes = *m_boxes;
    if (normalwash.size() != boxes.size())
        throw std::invalid_argument(
            base::concatenate({"TrimVariables::normalwashOf: expected one value per box (",
                               std::to_string(boxes.size()), "), found ", std::to_string(normalwash.size())}));
    std::fill(normalwash.begin(), normalwash.end(), 0.0);
    for (const ControlSurface& surface : m_surfaces)
    {
        if (surface.label != label)
            continue;
        const base::Vec3 axis = base::cross(FLOW, surface.hinge);
        for (const std::size_t box : surface.boxes)
            normalwash[box] = -surface.effectiveness * base::dot(axis, boxes[box].normal);
        return;
    }
    if (label == ANGLE_OF_ATTACK || label == SIDE_SLIP)
    {
        for (std::size_t j = 0; j < boxes.size(); ++j)
            normalwash[j] = label == ANGLE_OF_ATTACK ? -boxes[j].normal.z() : boxes[j].normal.y();
        return;
    }
    // ω / V of a unit rate about the reference axes, in the aerodynamic frame.
    base::Vec3 rateInReference;
    if (label == ROLL_RATE)
        rateInReference = {2.0 / m_reference.span, 0.0, 0.0};
    else if (label == PITCH_RATE)
        rateInReference = {0.0, 2.0 / m_reference.chord, 0.0};
    else if (label == YAW_RATE)
        rateInReference = {0.0, 0.0, 2.0 / m_reference.span};
    else
        throw std::invalid_argument(base::concatenate(
            {"TrimVariables::normalwashOf: expected AngleOfAttack, SideSlip, RollRate, PitchRate, YawRate or an "
             "AeroControlSurface label of the model, "
             "found '",
             labelName, "'"}));
    const base::Vec3 rate = m_reference.directionInAeroFrame(m_reference.referenceFrame.axes * rateInReference);
    const base::Vec3 origin = m_reference.referenceOriginInAeroFrame();
    for (std::size_t j = 0; j < boxes.size(); ++j)
        normalwash[j] = base::dot(base::cross(rate, boxes[j].controlPoint - origin), boxes[j].normal);
}

} // namespace oos::aero
