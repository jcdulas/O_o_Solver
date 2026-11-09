#pragma once

#include "oos/base/math/MatN.hpp"

#include <array>
#include <optional>

namespace oos::elements
{

// The pin flags of a bar or beam (CBAR, Beam2 fields PA, PB: the end DOFs, in the element system, that the bar does
// not connect to its grid point), applied by static condensation of the released end DOFs r out of the element
// stiffness, the others a keeping their connection (they carry no force through r):
//
//   K_c = K_aa − K_ar·K_rr⁻¹·K_ra,   f_c = f_a − K_ar·K_rr⁻¹·f_r,   d_r = K_rr⁻¹·(f_r − K_ra·d_a),
//
// the rows and columns of r in K_c and f_c set to zero (the grid point DOF is no longer connected through them). A
// valid deck gives the bar stiffness in each released DOF; K_rr must be positive definite, which make() checks.
//
// Thread-safety: immutable.
class PinRelease
{
public:
    // The release of the flagged end DOFs of a 12 × 12 end stiffness, or nothing when K_rr is not positive definite
    // (a released DOF without stiffness, or releases that leave a mechanism of the element, e.g. the twist released at
    // both ends): a Cholesky pivot below PIN_RELEASE_PIVOT_RATIO times the diagonal term it started from.
    [[nodiscard]] static std::optional<PinRelease> make(const base::MatN<12>& stiffness,
                                                        const std::array<bool, 12>& isReleased);

    [[nodiscard]] const base::MatN<12>& condensedStiffness() const noexcept { return m_condensed; }
    [[nodiscard]] std::array<double, 12> condenseLoads(const std::array<double, 12>& loads) const;
    // The end DOFs with the released ones computed from the kept ones and the loads.
    [[nodiscard]] std::array<double, 12> completeDisplacements(const std::array<double, 12>& kept,
                                                               const std::array<double, 12>& loads) const;

private:
    PinRelease() = default;

    std::array<bool, 12> m_isReleased{};
    base::MatN<12> m_stiffness; // K, not condensed
    base::MatN<12> m_inverse;   // K_rr⁻¹ in the rows and columns of r, zero elsewhere
    base::MatN<12> m_condensed; // K_c
};

// See PinRelease::make: a pivot below this ratio of its original diagonal term is a released DOF without stiffness.
inline constexpr double PIN_RELEASE_PIVOT_RATIO = 1.0e-10;

} // namespace oos::elements
