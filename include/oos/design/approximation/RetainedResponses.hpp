#pragma once

#include "oos/design/approximation/ConstraintScreening.hpp"
#include "oos/design/approximation/ExplicitConstraintGroup.hpp"
#include "oos/design/approximation/ScreeningReport.hpp"
#include "oos/design/constraints/ConstraintNormalisation.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/sensitivity/DesignGradients.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::design
{

// The retained responses of a design point: the objective ∪ the explicit group ∪ the constraints
// retained by screening (DSCREEN of the problem's constraints, ConstraintScreening), counted in items.
//
// Each constraint value of the evaluation is renormalised with the GSCAL of the problem (ConstraintNormalisation),
// given its DesignConstrScreen type (the DesignResp1 RTYPE, EQUA for a DesignResp2, DesignResp3) and its region (the
// REGION of its response; without one, the response for a grid response or an equation, the property of the item's
// element for an element response; none for WEIGHT, VOLUME and FRMASS), then screened. The items of the retained
// constraints, and the objective's item, are the retained items: gradientOptions() asks the sensitivities for them only
// (item masks), so that MethodChoice counts and the solves cover the retained items only (an equation item still has
// its arguments linearised).
//
// Thread-safety: immutable once built. Ownership: owns its results; keeps no reference to the problem.
class RetainedResponses
{
public:
    // Throws as ConstraintScreening (the DesignConstrScreen rules of the problem).
    RetainedResponses(const DesignProblem& problem, const DesignEvaluation& evaluation,
                      const ExplicitConstraintGroup& group,
                      double constraintScale = ConstraintNormalisation::DEFAULT_CONSTRAINT_SCALE);

    // What the screening made of every constraint of the evaluation (its order).
    [[nodiscard]] const std::vector<ScreeningStatus>& statuses() const noexcept { return m_statuses; }
    // The constraints retained (explicitly or by screening), indices into the evaluation's constraints, in order.
    [[nodiscard]] const std::vector<std::size_t>& retainedConstraints() const noexcept { return m_retained; }
    // The masks of the retained items, one per response that has a retained item, in the evaluation's order.
    [[nodiscard]] const std::vector<ItemMask>& itemMasks() const noexcept { return m_masks; }
    // The number of retained items (the objective's included).
    [[nodiscard]] std::size_t retainedItemCount() const noexcept;
    // The gradient options of the retained items (all-zero masks when nothing is retained: no item differentiated).
    [[nodiscard]] GradientOptions gradientOptions(SensitivityMethod method = SensitivityMethod::Automatic) const;
    [[nodiscard]] const ScreeningReport& report() const noexcept { return m_report; }

private:
    std::vector<ScreeningStatus> m_statuses;
    std::vector<std::size_t> m_retained;
    std::vector<ItemMask> m_masks;
    std::vector<ItemMask> m_empty; // an all-zero mask of every evaluated response: nothing retained
    ScreeningReport m_report;
};

} // namespace oos::design
