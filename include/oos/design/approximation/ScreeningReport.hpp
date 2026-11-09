#pragma once

#include "oos/design/approximation/ConstraintScreening.hpp"
#include "oos/design/constraints/ConstraintNormalisation.hpp"
#include "oos/design/responses/ResponseValues.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::design
{

// One constraint of an evaluation in the screening report: what it is, its value and bound, and what the screening
// made of it.
struct ScreeningReportRow
{
    std::size_t constraint = 0; // its index in DesignEvaluation::constraints
    std::int32_t responseId = 0;
    std::string label;
    std::int32_t setId = 0;
    ResponseItem item;
    BoundSide side = BoundSide::Upper;
    double bound = 0.0;
    double value = 0.0;
    double normalised = 0.0;
    std::string responseType;
    ScreeningRegion region;
    ScreeningStatus status = ScreeningStatus::ScreenedOut;
};

// The screening report: for every constraint, "kept explicitly", "retained by screening" or "screened out", with its
// value and bound; the rules applied and the counts; it states that the explicit group is an extension of the
// screening when one is used.
struct ScreeningReport
{
    std::vector<ScreeningReportRow> rows;
    std::vector<ScreeningRule> rules; // the rule applied to each response type present, in order of first use
    std::size_t keptExplicitly = 0;
    std::size_t retainedByScreening = 0;
    std::size_t screenedOut = 0;

    // A text of the report: the counts, the rules and one line per constraint. It names the cards and fields by their
    // neutral model names ("DesignConstrScreen.maxRetained").
    [[nodiscard]] std::string text() const;
};

} // namespace oos::design
