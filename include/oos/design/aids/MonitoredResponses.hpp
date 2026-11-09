#pragma once

#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/responses/DesignResponses.hpp"
#include "oos/design/responses/ResponseValues.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oos::design
{

// The summary of one monitored response at a design point: its extreme items.
struct MonitoredSummary
{
    std::int32_t responseId = 0;
    std::string label;
    std::string type; // its DesignConstrScreen type ("Stress", "Displacement", "Equation" ...)
    std::size_t itemCount = 0;
    double smallest = 0.0;
    double largest = 0.0;
    ResponseItem largestItem; // the item of the largest |value|
    double largestMagnitude = 0.0;
};

// The monitored responses of a design problem (an optimisation aid beyond gradients: responses evaluated and
// reported at every call, such as stresses, buckling factors and frequencies, without being constraints; their
// gradients on request). A monitored response is a response with no constraint, evaluated in every analysed
// subcase; this class names the
// ones a program monitors and summarises them at every evaluation: their extremes and where. The buckling factors,
// eigenvalues and frequencies of the MODES and BUCK analyses and the trim values and stability derivatives
// of the SAERO analysis are monitored as any response.
//
// Thread-safety: immutable once built. Ownership: owns its IDs.
class MonitoredResponses
{
public:
    // Throws OptimisationError naming a response ID the problem does not have.
    MonitoredResponses(const DesignResponses& responses, std::vector<std::int32_t> ids);

    // The version that brings a response kind of a later analysis, or nothing for a kind this version evaluates
    // (every kind is evaluated now).
    [[nodiscard]] static std::optional<std::string> laterStepOf(std::string_view kind);

    [[nodiscard]] const std::vector<std::int32_t>& ids() const noexcept { return m_ids; }

    // The summaries of the monitored responses an evaluation holds, in the order of ids().
    [[nodiscard]] std::vector<MonitoredSummary> summarise(const std::vector<ResponseValues>& values) const;

private:
    std::vector<std::int32_t> m_ids;
    std::vector<std::string> m_types;
};

} // namespace oos::design
