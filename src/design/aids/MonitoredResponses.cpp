#include "oos/design/aids/MonitoredResponses.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace oos::design
{

namespace
{

// The kinds of the later analyses and the version that brings them: none left (the frequencies, eigenvalues and
// buckling factors are evaluated in SOL 200 with MODES and BUCK, the trim values and stability derivatives in SOL 200
// with SAERO).
constexpr std::array<std::pair<std::string_view, std::string_view>, 0> LATER_KINDS{};

} // namespace

MonitoredResponses::MonitoredResponses(const DesignResponses& responses, std::vector<std::int32_t> ids)
    : m_ids(std::move(ids))
{
    for (const std::int32_t id : m_ids)
    {
        if (!responses.hasResponse(id))
            throw base::OptimisationError(base::concatenate(
                {"monitored responses: expected a response of the problem, found ID ", std::to_string(id),
                 ", which it does not have; declare it with "
                 "DesignProblem.monitor before the evaluation"}));
        m_types.push_back(responses.screeningType(id));
    }
}

std::optional<std::string> MonitoredResponses::laterStepOf(std::string_view kind)
{
    for (const auto& [name, step] : LATER_KINDS)
        if (name == kind)
            return std::string(step);
    return std::nullopt;
}

std::vector<MonitoredSummary> MonitoredResponses::summarise(const std::vector<ResponseValues>& values) const
{
    std::vector<MonitoredSummary> summaries;
    for (std::size_t m = 0; m < m_ids.size(); ++m)
        for (const ResponseValues& response : values)
        {
            if (response.id != m_ids[m] || response.itemCount() == 0)
                continue;
            MonitoredSummary summary;
            summary.responseId = response.id;
            summary.label = response.label;
            summary.type = m_types[m];
            summary.itemCount = response.itemCount();
            summary.smallest = response.values.front();
            summary.largest = response.values.front();
            summary.largestItem = response.items.front();
            summary.largestMagnitude = std::abs(response.values.front());
            for (std::size_t i = 1; i < response.itemCount(); ++i)
            {
                const double value = response.values[i];
                summary.smallest = std::min(summary.smallest, value);
                summary.largest = std::max(summary.largest, value);
                if (std::abs(value) > summary.largestMagnitude)
                {
                    summary.largestMagnitude = std::abs(value);
                    summary.largestItem = response.items[i];
                }
            }
            summaries.push_back(std::move(summary));
        }
    return summaries;
}

} // namespace oos::design
