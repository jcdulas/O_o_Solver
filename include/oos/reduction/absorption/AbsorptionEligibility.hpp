#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/reduction/absorption/AbsorptionSafeguard.hpp"
#include "oos/reduction/absorption/Spider.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::reduction
{

// What becomes of a spider in a plan, and why (each decision is in the plan report). The first
// reason that applies, in this order, is the one given.
enum class AbsorptionDecision : std::uint8_t
{
    Absorbed,                // absorbed into its unit: its reference point stands for its legs on the boundary
    AbsorbedWithCluster,     // its cluster touches one unit only: absorbed whole by the cluster rule
    PartOfUnit,              // not a connection: an element list holds it
    NotConnection,           // the connection system does not know it (nothing links it to a unit)
    NotSpider,               // O_o_Reader: legs on several components, UM, reference a leg or dependent elsewhere
    LegsOnSeveralUnits,      // its legs are on several units of the plan, or one on none (after a regrouping)
    LegTouched,              // a leg is touched by another connection (O_o_Reader's topological rule)
    NoEligibility,           // the table does not say whether a leg is touched (a piercing table without is_eligible)
    LegRetained,             // a leg is retained by the caller
    LegOnSplineNode,         // a leg is a spline node (SOL 144)
    LegOnSupport,            // a leg is a FreeBodySupport or SUPORT1 point
    LegPartiallyConstrained, // a leg has a DOF constrained in some subcases only
    LegConstrained,          // a leg DOF of the relation is constrained in every subcase
    NotRequested,            // the plan asked for no spider absorption
    NoRelation,              // the relation could not be built (AbsorptionSafeguard)
    RankDeficient,           // W does not fix every reference DOF
    IllConditioned           // the change of basis is above SPIDER_ABSORPTION_MAX_CONDITION
};

// "absorbed", "a leg is touched by another connection", ...
[[nodiscard]] std::string_view describeDecision(AbsorptionDecision decision) noexcept;
// A short code for tables and the caller: "absorbed", "legTouched", ...
[[nodiscard]] std::string_view decisionCode(AbsorptionDecision decision) noexcept;

// A connection that touches a leg and so keeps a spider on its legs (from connectionSharing when the snapshot has it).
struct BlockingConnection
{
    base::EntityId leg = 0;
    std::string kind; // "rigid", "rigidInterpolation", "mpc", "fastener", "connector"; empty when only the ID is known
    std::string card;
    base::EntityId id = 0; // the element ID, the set ID of an MPC equation
};

// The decision on one spider, with what the report and solver.absorption() show.
struct SpiderDecision
{
    SpiderKey key;
    SpiderKind kind = SpiderKind::RigidInterpolation;
    std::string card;
    base::EntityId referenceNodeId = 0; // REFGRID, GN, the given GA or GB; 0 for a generated fastener point
    // The point the plan puts on the boundary for an absorbed spider: the reference point, or for a fastener side the
    // smallest grid point of the pierced element standing for P (the plan's estimates are made on model points; the
    // condensation puts P's six DOFs there).
    base::EntityId boundaryNodeId = 0;
    std::int64_t unitId = 0; // the unit of its legs; 0 when none
    AbsorptionDecision decision = AbsorptionDecision::NotSpider;
    std::string detail; // O_o_Reader's reason for NotSpider; free text otherwise
    double conditionNumber = 0.0;
    std::vector<base::EntityId> legNodeIds;
    std::vector<base::EntityId> concernedNodeIds;        // the legs the decision is about (touched, retained, ...)
    std::vector<BlockingConnection> blockingConnections; // LegTouched: what touches them
    std::int64_t revision = 0;
};

// A grid point of a contact between bodies (glued contact): a secondary grid point of a node pair or a grid
// point of the primary face it meets, with the contact table that pairs the bodies. Trivially copyable (a model-sized
// list is budgeted); `table` views the text of O_o_Reader's contact table ("ContactTable 5"; empty for a pairing
// without table), valid while the snapshot lives.
struct GluedPoint
{
    base::EntityId nodeId = 0;
    std::string_view table;
};

// What the analysis keeps on the boundary on top of O_o_Reader's topological rule (nor is a spider absorbed when one
// of its legs must stay in the boundary for the analysis). Every list sorted.
struct AbsorptionExclusions
{
    std::vector<base::EntityId> retainedNodes;
    std::vector<base::EntityId> splineNodes; // SOL 144 (filled by the aeroelastic planner, step 7)
    std::vector<base::EntityId> supportNodes;
    std::vector<base::EntityId> partiallyConstrainedNodes;
    // The points of the contacts between bodies, sorted by point (a point may come once per contact). A glue equation
    // is a connection like an MPC: a leg that is a secondary point of a glue or a
    // corner of a glued face is touched by another connection, so its spider stays on its legs. O_o_Reader's
    // connectionSharing and spiders see the contact only when called with contact="connection"; the units are taken
    // with contact="merge" (as long as the glue does not cross units), so the solver applies this part of the rule.
    std::span<const GluedPoint> gluedPoints;
};

// The eligibility rule of one spider (a topological rule, decided per spider): O_o_Reader's topological decision (no
// leg touched by another connection: another rigid element or
// RigidInterpolation, an MPC equation, a Fastener or Weld through the grid points of the element it pierces, a
// connection element; a leg shared by two spiders makes both ineligible), then the exclusions of the analysis (a leg
// retained, a spline node, a FreeBodySupport point, constrained in some subcases only, or constrained in every subcase
// on a DOF of the relation), then the numerical safeguard (AbsorptionSafeguard). The connection system and the units
// are SpiderAbsorption's.
//
// Thread-safety: stateless. Ownership: none.
class AbsorptionEligibility
{
public:
    // The topological decision of O_o_Reader: Absorbed when eligible, else NotSpider, NoEligibility or LegTouched;
    // `concerned` gets the touched legs.
    [[nodiscard]] static AbsorptionDecision topological(const Spider& spider, std::vector<base::EntityId>& concerned);

    // The legs on a contact between bodies (AbsorptionExclusions::gluedPoints): LegTouched with the contacts as the
    // blocking connections (kind "contact"), else Absorbed.
    [[nodiscard]] static AbsorptionDecision glued(const Spider& spider, const AbsorptionExclusions& exclusions,
                                                  std::vector<base::EntityId>& concerned,
                                                  std::vector<BlockingConnection>& blocking);

    // The exclusions of the analysis on an eligible spider: Absorbed when none applies; `concerned` gets the legs.
    [[nodiscard]] static AbsorptionDecision analysis(const Spider& spider, const AbsorptionExclusions& exclusions,
                                                     const SpiderMeasurement& measurement,
                                                     std::vector<base::EntityId>& concerned);

    // The safeguard's verdict as a decision.
    [[nodiscard]] static AbsorptionDecision numerical(const SpiderMeasurement& measurement,
                                                      double maxCondition = SPIDER_ABSORPTION_MAX_CONDITION) noexcept;
};

} // namespace oos::reduction
