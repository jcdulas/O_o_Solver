#pragma once

#include "oos/base/ids/EntityId.hpp"

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace oos::reduction
{

// What a spider is (a rigid element or a pierced side of a fastener, absorbed alike).
enum class SpiderKind : std::uint8_t
{
    RigidInterpolation, // legs: the weighted points Gi; reference: REFGRID (its REFC DOFs dependent)
    RigidBodyStar,      // legs: the dependent points GMi; reference: GN (independent)
    FastenerSide // legs: the grid points of the element one side of a Fastener or Weld pierces; reference: GA
                 // or GB, or the point generated there when the card gives none
};

// The side of a fastener a spider stands for (None for a rigid element).
enum class FastenerSide : std::uint8_t
{
    None,
    A,
    B
};

// A spider is named by its card's ID and, for a fastener, its side.
struct SpiderKey
{
    base::EntityId elementId = 0;
    FastenerSide side = FastenerSide::None;

    [[nodiscard]] auto operator<=>(const SpiderKey&) const = default;
};

// One spider as O_o_Reader describes it (its spiders table; fastenerPiercings for the fastener sides), before the
// analysis-dependent rules of O_o_Solver (AbsorptionEligibility).
//
// Invariant: legNodeIds sorted and distinct (the card's order is not kept: the change of basis takes the leg DOFs in
// increasing order of point and component).
struct Spider
{
    SpiderKind kind = SpiderKind::RigidInterpolation;
    SpiderKey key;
    std::string card; // "RigidInterpolation", "RigidBodyStar", "Fastener", "Weld"
    // REFGRID, GN, or the GA/GB the fastener card gives; 0 for a generated fastener point (no model point).
    base::EntityId referenceNodeId = 0;
    std::string referenceDofs; // the components as written ("123456"); empty for a fastener side
    std::vector<base::EntityId> legNodeIds;
    // The component of the legs O_o_Reader gives (null when they are on several components or one on none).
    std::optional<std::int64_t> componentId;
    // O_o_Reader's topology: a card whose legs are all on one component, with its reason when it is not
    // ("legsOnSeveralUnits", "um", "referenceIsLeg", "referenceDependentElsewhere").
    bool isSpider = false;
    std::string notSpiderReason;
    // No leg touched by another connection ("the rule is computed by O_o_Reader from the connectivity
    // alone"), with what touches them: the legs (blockingNodeIds) and the connections (an MPC equation by its set ID).
    bool isEligible = false;
    bool hasEligibility = false; // false when the table does not say (a piercing table without is_eligible)
    std::vector<base::EntityId> blockingNodeIds;
    std::vector<base::EntityId> blockingConnections;
    // O_o_Reader's revision stamp of the card and of what it refers to; for a fastener side, the stamp of its piercing
    // rows (SpiderCatalog: the card, the side, the grid points and the bits of the weights).
    std::int64_t revision = 0;
};

} // namespace oos::reduction
