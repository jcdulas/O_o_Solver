#pragma once

#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/aero/trim/TrimVariables.hpp"

#include <span>
#include <string>
#include <vector>

namespace oos::aero
{

// The hinge moments of the control surfaces (their derivatives with respect to the trim variables, and their values at
// the trimmed state): the moment about the hinge line of an AeroControlSurface (the y axis of
// its CID1) of the forces q·Δcp_j·A_j·n_j of its boxes at their force points,
//     H = q Σ_j Δcp_j A_j ((p_j − o) × n_j)·e,
// e the hinge axis, o its origin: the virtual work of the box forces in a unit rotation of the surface about e, so that
// H > 0 tends to turn it the way its positive deflection does (the lift of a trailing-edge flap turned down gives
// H < 0). The coefficient is H / (q·CREFC·CREFS). Each surface sums its own boxes; the images of a half model do not
// add to it.
//
// Thread-safety: immutable. Ownership: owns its factors (one per box of each surface).
class HingeMoments
{
public:
    HingeMoments(const AeroBoxes& boxes, const TrimVariables& variables);

    [[nodiscard]] std::size_t surfaceCount() const noexcept { return m_surfaces.size(); }
    [[nodiscard]] const std::string& labelOf(std::size_t surface) const { return m_surfaces[surface].label; }

    // The hinge moment coefficient of each surface, in the order of TrimVariables::controlSurfaces(), for the pressure
    // coefficients `pressures` of the boxes.
    [[nodiscard]] std::vector<double> coefficientsOf(std::span<const double> pressures) const;
    // The hinge moment of each surface (force × length) at dynamic pressure q.
    [[nodiscard]] std::vector<double> momentsOf(std::span<const double> pressures, double dynamicPressure) const;

private:
    struct Surface
    {
        std::string label;
        double referenceProduct = 1.0;  // CREFC·CREFS
        std::vector<std::size_t> boxes; // bounded by the boxes of the surface
        std::vector<double> factors;    // A_j ((p_j − o) × n_j)·e, one per box
    };

    std::vector<Surface> m_surfaces; // bounded by the AeroControlSurface cards
};

} // namespace oos::aero
