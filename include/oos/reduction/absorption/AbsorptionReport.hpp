#pragma once

#include "oos/reduction/absorption/AbsorptionEligibility.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace oos::reduction
{

// The most spiders the text report lists one by one (the others are counted; solver.absorption() gives every one).
inline constexpr std::size_t ABSORPTION_REPORT_MAX_LINES = 200;

// The absorption part of the plan report (per spider, its decision and the reason: the legs touched by another
// connection, with those connections: kind, card, ID): the count of each decision, then one line per spider of the
// connection system that is not absorbed, with its legs concerned and the connections that touch them, and the
// condition number of its change of basis. The text names cards and fields by their neutral model names.
//
// Thread-safety: immutable. Ownership: owns its text.
class AbsorptionReport
{
public:
    // hasRigidSpiders: the snapshot had the spiders table (else the report says that no RigidBodyStar or
    // RigidInterpolation was decided).
    AbsorptionReport(std::span<const SpiderDecision> decisions, bool hasRigidSpiders);

    [[nodiscard]] const std::string& text() const noexcept { return m_text; }
    [[nodiscard]] std::int64_t absorbedCount() const noexcept { return m_absorbedCount; }
    // The spiders of the connection system kept on their legs.
    [[nodiscard]] std::int64_t keptOnLegsCount() const noexcept { return m_keptOnLegsCount; }

    // "RigidInterpolation 501", "Fastener 601 side A".
    [[nodiscard]] static std::string spiderName(const SpiderDecision& decision);
    // One line for a decision: name, unit, reason, the legs concerned and what touches them.
    [[nodiscard]] static std::string decisionLine(const SpiderDecision& decision);

private:
    std::string m_text;
    std::int64_t m_absorbedCount = 0;
    std::int64_t m_keptOnLegsCount = 0;
};

} // namespace oos::reduction
