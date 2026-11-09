#pragma once

#include "oos/design/constraints/DesignConstraints.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace oos::design
{

// The constraints kept always, whatever the screening (an extension of O_o_Solver: the input format has no field for
// it; the caller's keep_constraints(...), a deck run's keep_constraints= option). A
// constraint belongs to the group when any of these selects it:
//   constraintSets: its DesignConstr set (DCID; a DesignConstrUnion set stands for its members, expanded by the
//   caller); responses:      its response (DRESP1, DesignResp2 or DesignResp3 ID); regions:        the REGION of its
//   response; elements:       the element of an element response's item (STRESS ... CSTRAT); grids:          the grid
//   point of a grid response's item (DISP, SPCFORCE).
// The group's constraints are never screened, always evaluated, differentiated and handed to the optimiser, and they
// take no NSTR place of the others (ConstraintScreening).
//
// Thread-safety: immutable once built (a plain carrier). Ownership: owns its lists.
struct ExplicitConstraintGroup
{
    std::vector<std::int32_t> constraintSets;
    std::vector<std::int32_t> responses;
    std::vector<std::int32_t> regions;
    std::vector<std::int32_t> elements;
    std::vector<std::int32_t> grids;

    [[nodiscard]] bool isEmpty() const noexcept
    {
        return constraintSets.empty() && responses.empty() && regions.empty() && elements.empty() && grids.empty();
    }

    // Whether the group keeps `constraint`. `region`: the REGION of its response (nothing when blank);
    // `isElementResponse`, `isGridResponse`: what the entity of its item is.
    [[nodiscard]] bool contains(const ConstraintValue& constraint, std::optional<std::int32_t> region,
                                bool isElementResponse, bool isGridResponse) const noexcept;
};

} // namespace oos::design
