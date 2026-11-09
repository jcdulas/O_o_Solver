#pragma once

#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace oos::aero
{

// One box of the doublet-lattice method in the aerodynamic frame (Albano & Rodden, AIAA J. 7 (1969) 279): the lifting
// element is a line of doublets on the quarter chord of the box, from `boundStart` to `boundEnd` (at zero frequency the
// horseshoe vortex of the vortex-lattice method, Hedman 1966), the boundary condition is met at the control point at
// three quarters of the chord at mid span, and the pressure acts at the force point, the middle of the quarter-chord
// line. The quarter-chord line runs in the direction of n × x (x the flow), so that a positive pressure coefficient Δcp
// (lower minus upper surface) lifts the box along its normal n.
struct AeroBox
{
    std::int32_t id = 0;
    base::Vec3 boundStart;
    base::Vec3 boundEnd;
    base::Vec3 controlPoint;
    base::Vec3 forcePoint;
    base::Vec3 center; // the aerodynamic grid point of the box (the average of its corners), where the splines move it
    base::Vec3 normal;
    double area = 0.0;
    double chord = 0.0; // along the flow at mid span
};

// The boxes of the AeroLiftingPanel1 panels of a model (aeroBoxesTable, aeroPanelsTable), in the
// aerodynamic frame of the AeroStaticParams and in the order O_o_Reader gives them (by panel, then span strip, then
// chordwise), the order of the aerodynamic j set.
//
// Thread-safety: immutable. Ownership: owns its boxes (budgeted: one per box of the model).
class AeroBoxes
{
public:
    // Throws SolverError when the snapshot has no boxes table or a panel has no boxes (an AeroFactorList missing), and
    // UnsupportedCardError for a panel other than AeroLiftingPanel1 (CAERO2 bodies, AeroMachBoxPanel3 Mach boxes,
    // AeroStripPanel4 and AeroPistonPanel5 strips: not supported yet).
    AeroBoxes(const input::ModelSnapshot& snapshot, const AeroReference& reference, base::MemoryBudget& budget);
    // From given boxes (tests, and the image models of the symmetry checks).
    AeroBoxes(std::span<const AeroBox> boxes, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t size() const noexcept { return m_boxes.size(); }
    [[nodiscard]] const AeroBox& operator[](std::size_t i) const noexcept { return m_boxes[i]; }
    [[nodiscard]] std::span<const AeroBox> boxes() const noexcept { return m_boxes.span(); }
    // The place of box `id`, or nothing.
    [[nodiscard]] std::optional<std::size_t> find(std::int32_t id) const;

private:
    base::BudgetedBuffer<AeroBox> m_boxes;
};

} // namespace oos::aero
