#pragma once

#include "oos/design/constraints/ConstraintDefinitions.hpp"

#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::design
{

// The group of a constraint for the NSTR limit of DesignConstrScreen: constraints of the same response type, region and
// subcase compete for NSTR places.
//   Region:   the REGION field of the response (the same for every response of that type and REGION);
//   Response: one region per response ID (a grid response without REGION; a DesignResp2 or DesignResp3 without REGION);
//   Property: one region per property of the element (an element response without REGION: each property ID the
//             response reads makes its own region), the property card distinguishing equal IDs of other cards;
//   Unlimited: no NSTR (WEIGHT, VOLUME, FRMASS: one value each, so only TRS applies).
struct ScreeningRegion
{
    enum class Kind : std::uint8_t
    {
        Unlimited,
        Region,
        Response,
        Property
    };
    Kind kind = Kind::Unlimited;
    std::int32_t id = 0;
    std::string propertyCard;

    friend auto operator<=>(const ScreeningRegion&, const ScreeningRegion&) = default;
};

// One constraint as the screening sees it: its DesignConstrScreen type ("Stress", "Displacement", "Equation" for a
// DesignResp2, "DesignResp3"), its region, its subcase (0 for a global response), its normalised value, and whether the
// explicit group keeps it.
struct ScreeningEntry
{
    std::string responseType;
    ScreeningRegion region;
    std::int32_t subcase = 0;
    double normalised = 0.0;
    bool isExplicit = false;
};

// What the screening did with a constraint.
enum class ScreeningStatus : std::uint8_t
{
    KeptExplicitly,      // in the explicit group: never screened, does not take an NSTR place
    RetainedByScreening, // above TRS and among the NSTR largest of its region and subcase
    ScreenedOut
};

[[nodiscard]] std::string_view screeningStatusName(ScreeningStatus status) noexcept;

// The constraint screening of DesignConstrScreen: a constraint is retained when its normalised value is above the
// threshold TRS of its response type and among the NSTR largest of its region and subcase; a type without a
// DesignConstrScreen entry takes the defaults TRS = −0.5 and NSTR = 20; RTYPE = ALL applies to every type and the other
// entries are ignored. The constraints of the explicit group (an extension of O_o_Solver, ExplicitConstraintGroup) are
// kept whatever their value and take no NSTR place. Equal values keep the earlier constraint (the order of the
// entries), so that the result is deterministic.
//
// Thread-safety: immutable. Ownership: owns its rules.
class ConstraintScreening
{
public:
    static constexpr double DEFAULT_THRESHOLD = -0.5;
    static constexpr std::int32_t DEFAULT_MAXIMUM_RETAINED = 20;

    // Throws OptimisationError for a DesignConstrScreen of an RTYPE that is not a response type of DesignResp1 (nor
    // EQUA, DesignResp3, ALL), an NSTR below 1, a type given twice.
    explicit ConstraintScreening(std::vector<ScreeningRule> rules);

    // The rule of a response type (ALL, its own, or the defaults).
    [[nodiscard]] ScreeningRule ruleFor(std::string_view responseType) const;
    [[nodiscard]] const std::vector<ScreeningRule>& rules() const noexcept { return m_rules; }

    // The status of each entry, in their order.
    [[nodiscard]] std::vector<ScreeningStatus> screen(std::span<const ScreeningEntry> entries) const;

private:
    std::vector<ScreeningRule> m_rules;
};

} // namespace oos::design
