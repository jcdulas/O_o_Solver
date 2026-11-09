#include "oos/reduction/absorption/AbsorptionReport.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace oos::reduction
{

namespace
{

// Decisions of spiders that are not candidates of the spider absorption (another rule decided them).
bool isOutsideConnectionSystem(AbsorptionDecision decision) noexcept
{
    return decision == AbsorptionDecision::AbsorbedWithCluster || decision == AbsorptionDecision::PartOfUnit ||
           decision == AbsorptionDecision::NotConnection;
}

std::string idList(const std::vector<base::EntityId>& ids)
{
    std::string text;
    for (const base::EntityId id : ids)
    {
        if (!text.empty())
            text.append(", ");
        text.append(std::to_string(id));
    }
    return text;
}

std::string conditionText(double condition)
{
    if (std::isinf(condition))
        return "infinite";
    char buffer[32];
    const int length = std::snprintf(buffer, sizeof(buffer), "%.3g", condition);
    return std::string(buffer, static_cast<std::size_t>(length > 0 ? length : 0));
}

} // namespace

std::string AbsorptionReport::spiderName(const SpiderDecision& decision)
{
    std::string name = decision.card + " " + std::to_string(decision.key.elementId);
    if (decision.key.side == FastenerSide::A)
        name.append(" side A");
    else if (decision.key.side == FastenerSide::B)
        name.append(" side B");
    return name;
}

std::string AbsorptionReport::decisionLine(const SpiderDecision& decision)
{
    std::string line = spiderName(decision);
    if (decision.unitId != 0)
        line.append(" (unit ").append(std::to_string(decision.unitId)).append(")");
    line.append(": ").append(describeDecision(decision.decision));
    if (!decision.detail.empty())
        line.append(" (").append(decision.detail).append(")");
    if (!decision.concernedNodeIds.empty())
        line.append("; legs ").append(idList(decision.concernedNodeIds));
    if (!decision.blockingConnections.empty())
    {
        line.append(", touched by ");
        bool isFirst = true;
        for (const BlockingConnection& blocking : decision.blockingConnections)
        {
            if (!isFirst)
                line.append(", ");
            isFirst = false;
            if (!blocking.card.empty())
                line.append(blocking.card).append(" ");
            else
                line.append("connection ");
            line.append(std::to_string(blocking.id));
            if (blocking.leg != 0)
                line.append(" at ").append(std::to_string(blocking.leg));
        }
    }
    if (decision.decision == AbsorptionDecision::Absorbed || decision.decision == AbsorptionDecision::IllConditioned ||
        decision.decision == AbsorptionDecision::RankDeficient)
        line.append("; condition number ").append(conditionText(decision.conditionNumber));
    if (decision.decision == AbsorptionDecision::Absorbed)
        line.append("; boundary point ").append(std::to_string(decision.boundaryNodeId));
    return line;
}

AbsorptionReport::AbsorptionReport(std::span<const SpiderDecision> decisions, bool hasRigidSpiders)
{
    std::map<AbsorptionDecision, std::int64_t> counts;
    for (const SpiderDecision& decision : decisions)
        ++counts[decision.decision];
    m_absorbedCount = counts[AbsorptionDecision::Absorbed];
    for (const auto& [decision, count] : counts)
        if (decision != AbsorptionDecision::Absorbed && !isOutsideConnectionSystem(decision) &&
            decision != AbsorptionDecision::NotSpider)
            m_keptOnLegsCount += count;

    m_text.append("spiders (spider and fastener side absorption): ")
        .append(std::to_string(m_absorbedCount))
        .append(" absorbed, ")
        .append(std::to_string(m_keptOnLegsCount))
        .append(" kept on their legs");
    for (const auto& [decision, count] : counts)
        if (isOutsideConnectionSystem(decision) || decision == AbsorptionDecision::NotSpider)
            m_text.append(", ").append(std::to_string(count)).append(" ").append(decisionCode(decision));
    m_text.append("\n");
    if (!hasRigidSpiders)
        m_text.append("  no spiders table: no RigidBodyStar or RigidInterpolation absorbed, only fastener sides\n");

    std::size_t lines = 0;
    std::size_t omitted = 0;
    for (const SpiderDecision& decision : decisions)
    {
        if (decision.decision == AbsorptionDecision::Absorbed || isOutsideConnectionSystem(decision.decision))
            continue;
        if (lines == ABSORPTION_REPORT_MAX_LINES)
        {
            ++omitted;
            continue;
        }
        m_text.append("  ").append(decisionLine(decision)).append("\n");
        ++lines;
    }
    if (omitted > 0)
        m_text.append("  ... and ")
            .append(std::to_string(omitted))
            .append(" more spiders kept on their legs (solver.absorption() lists every spider)\n");
}

} // namespace oos::reduction
