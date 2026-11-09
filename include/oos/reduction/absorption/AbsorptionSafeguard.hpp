#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/reduction/absorption/SpiderChangeOfBasis.hpp"

#include <cstdint>

namespace oos::reduction
{

// The largest condition number of a spider's change of basis that is absorbed (the numerical safeguard, per spider),
// the condition number of the scaled weighting matrix (SpiderChangeOfBasis::conditionNumber). CALIBRATED on shell
// models: two plates joined through an RigidInterpolation of 3 to 8 legs, uneven weights, its reference raised up to
// 128 element sizes, legs brought within 0.001 of a line, and Fastener and Weld piercing points off their
// element centres, condensed with the spider absorbed and on its legs against the full path:
// - the disagreement with the full path grows with the condition number on both paths alike (the model's own
//   conditioning: the lever of a raised reference, the elements distorted under legs nearly in line), the absorbed one
//   between 0.1 and 12.5 times the one on legs with no trend up to 1633: absorption adds no error that grows with it;
// - the absorbed path stays within the 1e-9 of the condensed-path tests up to 48 on every shape and crosses it
//   between 48 and 64 (raised references) and 34 and 68 (raised triangles), where on legs it is still 2 to 5 times
//   below;
// - compact spiders, fastener sides and the generated assemblies with varied shapes measure 1 to 8.7.
// 50 absorbs every shape of practice and keeps on its legs (exact, a few more boundary DOFs) only the shapes that make
// the model itself ill-conditioned. Spiders on a plane membrane sharing legs (condition number 430, a reduced matrix
// 50 000 times worse conditioned) are the case O_o_Reader's topological rule excludes.
inline constexpr double SPIDER_ABSORPTION_MAX_CONDITION = 50.0;

// What the model says about one spider's relation, measured once per snapshot (solutions::CondensationPlanner): the
// conditioning of its change of basis and whether a leg DOF of the relation is constrained in every subcase.
// Trivially copyable: kept in a budgeted array parallel to the spider catalog.
struct SpiderMeasurement
{
    bool hasRelation = false;     // the weighting matrix could be built (Gm rows found, piercing point interpolated)
    bool isRankDeficient = false; // SpiderChangeOfBasis::isRankDeficient
    double conditionNumber = 0.0; // SpiderChangeOfBasis::conditionNumber; 1 for an RigidBodyStar (no change of basis)
    bool hasLegConstrainedEverywhere = false; // a leg DOF the relation uses is constrained in every subcase
    std::int32_t referenceDofCount = 0;       // the reference DOFs the absorption makes boundary DOFs
};

// The verdict of the numerical safeguard on one spider.
enum class SafeguardVerdict : std::uint8_t
{
    Accepted,
    NoRelation,    // its relation could not be built: the spider stays on its legs
    RankDeficient, // W does not fix every reference DOF (collinear or coincident legs)
    IllConditioned // conditionNumber above SPIDER_ABSORPTION_MAX_CONDITION
};

// The numerical safeguard of spider absorption: the conditioning of each eligible spider's change of
// basis is computed, and a spider whose W is rank-deficient or whose condition number exceeds
// SPIDER_ABSORPTION_MAX_CONDITION is not absorbed: locally, the condensation is done on its legs.
//
// Thread-safety: stateless. Ownership: none.
class AbsorptionSafeguard
{
public:
    // The conditioning of the change of basis of `weights` (the relation exists). MemoryBudgetError when the
    // reflectors do not fit; std::invalid_argument for inconsistent weights.
    [[nodiscard]] static SpiderMeasurement measure(const SpiderWeightMatrix& weights, base::MemoryBudget& budget);

    // The measurement of an RigidBodyStar: it is absorbed by its own elimination (dependent legs on the unit, GN on the
    // boundary), with no change of basis to condition.
    [[nodiscard]] static SpiderMeasurement rigidBodyStarMeasurement(std::int32_t referenceDofCount) noexcept;

    // `maxCondition`: the threshold, SPIDER_ABSORPTION_MAX_CONDITION but in the calibration studies (PlanRequest).
    [[nodiscard]] static SafeguardVerdict judge(const SpiderMeasurement& measurement,
                                                double maxCondition = SPIDER_ABSORPTION_MAX_CONDITION) noexcept;
};

} // namespace oos::reduction
