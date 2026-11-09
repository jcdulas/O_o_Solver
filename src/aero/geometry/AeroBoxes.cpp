#include "oos/aero/geometry/AeroBoxes.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

namespace oos::aero
{

namespace
{

// The panels the doublet-lattice boxes come from (CAERO1); the others are refused.
constexpr std::string_view LIFTING_SURFACE_CARD = "AeroLiftingPanel1";
// The quarter chord, where the doublet line (the bound vortex) of a box lies (Albano & Rodden 1969).
constexpr double DOUBLET_LINE_CHORD_FRACTION = 0.25;

void checkPanels(const input::ModelSnapshot& snapshot)
{
    if (!snapshot.hasTable(input::TableRole::AeroPanels))
        return;
    const base::ContractReader& panels = snapshot.aeroTable(input::TableRole::AeroPanels);
    const std::span<const std::int32_t> elementIds = panels.values<std::int32_t>("elementId");
    const base::DictionaryView types = panels.dictionaryTexts("type");
    const base::ValidityView hasBoxes = panels.validity("boxCount");
    for (std::int64_t row = 0; row < panels.rowCount(); ++row)
    {
        const std::string elementId = std::to_string(elementIds[static_cast<std::size_t>(row)]);
        if (types[row] != LIFTING_SURFACE_CARD)
            throw base::UnsupportedCardError(base::concatenate(
                {std::string(types[row]), " ", elementId,
                 ": expected AeroLiftingPanel1 lifting surfaces (the doublet-lattice method; bodies, Mach boxes "
                 "and strip theory come later), found this panel; model the surface with AeroLiftingPanel1 panels"}));
        if (!hasBoxes.isValid(row))
            throw base::SolverError(
                base::concatenate({"AeroLiftingPanel1 ", elementId,
                                   ": expected its boxes, found none (an AeroFactorList of its "
                                   "divisions is missing or has fewer than two "
                                   "points); add the AeroFactorList or give AeroLiftingPanel1.spanBoxCount and "
                                   "AeroLiftingPanel1.chordBoxCount"}));
    }
}

} // namespace

AeroBoxes::AeroBoxes(const input::ModelSnapshot& snapshot, const AeroReference& reference, base::MemoryBudget& budget)
{
    if (!snapshot.hasTable(input::TableRole::AeroBoxes))
        throw base::SolverError(
            "a static aeroelastic analysis needs the aerodynamic boxes (aeroBoxesTable of "
            "O_o_Reader), and the snapshot has none; add AeroLiftingPanel1 panels to the "
            "model or take the snapshot with its aero tables");
    checkPanels(snapshot);
    const base::ContractReader& table = snapshot.aeroTable(input::TableRole::AeroBoxes);
    const auto column = [&](const char* name) { return table.values<double>(name); };
    const std::span<const std::int32_t> ids = table.values<std::int32_t>("boxId");
    const auto pointOf = [&](const char* x, const char* y, const char* z, std::size_t row)
    { return base::Vec3{column(x)[row], column(y)[row], column(z)[row]}; };
    m_boxes = base::BudgetedBuffer<AeroBox>(budget, static_cast<std::size_t>(table.rowCount()), "aerodynamic boxes");
    const base::Vec3 flow{1.0, 0.0, 0.0};
    for (std::size_t row = 0; row < m_boxes.size(); ++row)
    {
        AeroBox box;
        box.id = ids[row];
        // Corners 1 and 2 on the inboard edge (leading, trailing), 3 and 4 on the outboard edge (trailing, leading).
        const base::Vec3 corner1 = reference.pointInAeroFrame(pointOf("corner1X", "corner1Y", "corner1Z", row));
        const base::Vec3 corner2 = reference.pointInAeroFrame(pointOf("corner2X", "corner2Y", "corner2Z", row));
        const base::Vec3 corner3 = reference.pointInAeroFrame(pointOf("corner3X", "corner3Y", "corner3Z", row));
        const base::Vec3 corner4 = reference.pointInAeroFrame(pointOf("corner4X", "corner4Y", "corner4Z", row));
        box.boundStart = corner1 + DOUBLET_LINE_CHORD_FRACTION * (corner2 - corner1);
        box.boundEnd = corner4 + DOUBLET_LINE_CHORD_FRACTION * (corner3 - corner4);
        box.controlPoint = reference.pointInAeroFrame(pointOf("controlX", "controlY", "controlZ", row));
        box.normal = base::normalized(reference.directionInAeroFrame(pointOf("normalX", "normalY", "normalZ", row)));
        box.forcePoint = 0.5 * (box.boundStart + box.boundEnd);
        box.center = reference.pointInAeroFrame(pointOf("centerX", "centerY", "centerZ", row));
        box.area = column("area")[row];
        box.chord = column("chord")[row];
        // The quarter-chord line along n × x: a positive Δcp then lifts the box along n (Kutta-Joukowski, F = ρ V × Γ).
        if (base::dot(box.boundEnd - box.boundStart, base::cross(box.normal, flow)) < 0.0)
            std::swap(box.boundStart, box.boundEnd);
        m_boxes[row] = box;
    }
}

AeroBoxes::AeroBoxes(std::span<const AeroBox> boxes, base::MemoryBudget& budget)
    : m_boxes(budget, boxes.size(), "aerodynamic boxes")
{
    std::copy(boxes.begin(), boxes.end(), m_boxes.data());
}

std::optional<std::size_t> AeroBoxes::find(std::int32_t id) const
{
    for (std::size_t i = 0; i < m_boxes.size(); ++i)
        if (m_boxes[i].id == id)
            return i;
    return std::nullopt;
}

} // namespace oos::aero
