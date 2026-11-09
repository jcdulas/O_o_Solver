#include "oos/reduction/plan/PlanReport.hpp"

#include "oos/reduction/absorption/AbsorptionReport.hpp"

#include <cstdio>
#include <string>

namespace oos::reduction
{

namespace
{

constexpr double BYTES_PER_VALUE = 8.0;
constexpr double BYTES_PER_MB = 1024.0 * 1024.0;
constexpr double FLOPS_PER_GFLOP = 1.0e9;

double megabytesOfValues(std::int64_t values)
{
    return static_cast<double>(values) * BYTES_PER_VALUE / BYTES_PER_MB;
}

double megabytes(std::uint64_t bytes)
{
    return static_cast<double>(bytes) / BYTES_PER_MB;
}

std::string formatted(const char* format, auto... values)
{
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), format, values...);
    return buffer;
}

const char* originName(UnitOrigin origin)
{
    switch (origin)
    {
    case UnitOrigin::Component:
        return "component";
    case UnitOrigin::MergedComponents:
        return "merged";
    case UnitOrigin::ElementList:
        return "elements";
    }
    return "";
}

std::string idList(const std::vector<std::int64_t>& ids)
{
    std::string text;
    for (const std::int64_t id : ids)
    {
        if (!text.empty())
            text.append(", ");
        text.append(std::to_string(id));
    }
    return text;
}

std::string interfaceLine(const char* title, const InterfaceEstimate& estimate)
{
    return formatted("%s: %lld points, %lld DOFs; factor %.3f MB, memory peak %.3f MB, %.4f GFLOP, %.4f s\n", title,
                     static_cast<long long>(estimate.nodeCount), static_cast<long long>(estimate.dofCount),
                     megabytesOfValues(estimate.factorValueCount), megabytes(estimate.memoryPeakBytes),
                     estimate.flopCount / FLOPS_PER_GFLOP, estimate.estimatedSeconds);
}

} // namespace

PlanReport::PlanReport(const CondensationPlan& plan)
{
    std::string& text = m_text;
    std::int64_t condensedCount = 0;
    for (const UnitPlan& unit : plan.units())
        condensedCount += unit.decision == UnitDecision::Condensed ? 1 : 0;
    text.append(formatted("CONDENSATION PLAN (%s)\n",
                          plan.reduction() == ReductionKind::Static ? "static condensation" : "Craig-Bampton"));
    text.append(formatted("units: %lld condensed, %lld kept full\n", static_cast<long long>(condensedCount),
                          static_cast<long long>(plan.units().size()) - static_cast<long long>(condensedCount)));
    if (!plan.note().empty())
        text.append("note: ").append(plan.note()).append("\n");
    text.append("\n");
    text.append(
        " unit  origin     elements  points  boundary points  boundary  interior  interior factor"
        "       K*    work    time  memory peak  decision   recommendation\n");
    text.append(
        "                                     before   after      DOFs      DOFs               MB"
        "       MB   GFLOP       s           MB\n");
    for (const UnitPlan& unit : plan.units())
    {
        const UnitCostEstimate& cost = unit.cost;
        std::string recommendation;
        if (cost.isKeptFullRecommended && unit.decision == UnitDecision::Condensed)
        {
            recommendation = "keep full: K* larger than its interior factor";
            m_keptFullRecommendations.push_back(unit.unitId);
        }
        text.append(formatted(
            "%5lld  %-9s  %8lld  %6lld  %6lld  %6lld  %8lld  %8lld  %15.3f  %7.3f  %6.3f  %6.3f  "
            "%11.3f  %-9s  %s\n",
            static_cast<long long>(unit.unitId), originName(unit.origin),
            static_cast<long long>(unit.elementIds.size()), static_cast<long long>(unit.nodeIds.size()),
            static_cast<long long>(unit.boundaryNodeCountBefore), static_cast<long long>(unit.boundary.size()),
            static_cast<long long>(cost.boundaryDofCount), static_cast<long long>(cost.interiorDofCount),
            megabytesOfValues(cost.interiorFactorValueCount), megabytesOfValues(cost.schurValueCount),
            cost.flopCount / FLOPS_PER_GFLOP, cost.estimatedSeconds, megabytes(cost.memoryPeakBytes),
            unit.decision == UnitDecision::Condensed ? "condensed" : "full", recommendation.c_str()));
    }

    bool hasAbsorbed = false;
    for (const UnitPlan& unit : plan.units())
    {
        if (unit.absorbedClusterCount == 0 && unit.memberUnits.size() <= 1)
            continue;
        if (!hasAbsorbed)
            text.append("\nabsorbed into their unit (clusters touching one unit):\n");
        hasAbsorbed = true;
        text.append(formatted("  unit %lld: %lld clusters, %lld connection-only points made interior",
                              static_cast<long long>(unit.unitId), static_cast<long long>(unit.absorbedClusterCount),
                              static_cast<long long>(unit.absorbedNodes.size())));
        if (unit.memberUnits.size() > 1)
            text.append(", 1D/0D-only units absorbed: ")
                .append(idList(std::vector<std::int64_t>(unit.memberUnits.begin() + 1, unit.memberUnits.end())));
        text.append("\n");
    }
    for (const UnitPlan& unit : plan.units())
    {
        std::int64_t retained = 0;
        std::int64_t partial = 0;
        for (const BoundaryNode& node : unit.boundary)
        {
            retained += node.has(BoundaryReason::Retained) ? 1 : 0;
            partial += node.has(BoundaryReason::PartialConstraint) ? 1 : 0;
        }
        if (retained + partial > 0)
            text.append(
                formatted("  unit %lld boundary: %lld retained points, %lld points constrained in some "
                          "subcases only\n",
                          static_cast<long long>(unit.unitId), static_cast<long long>(retained),
                          static_cast<long long>(partial)));
    }

    if (plan.hasSpiderAbsorption())
    {
        // Spider absorption: per unit the spiders absorbed, then every spider kept on its legs and why.
        text.append("\n");
        for (const UnitPlan& unit : plan.units())
            if (!unit.absorbedSpiders.empty())
                text.append(formatted("  unit %lld: %lld spiders absorbed, their reference points on the boundary\n",
                                      static_cast<long long>(unit.unitId),
                                      static_cast<long long>(unit.absorbedSpiders.size())));
        text.append(AbsorptionReport(plan.spiderDecisions(), plan.hasRigidSpiders()).text());
    }

    const ConnectionSummary& connections = plan.connectionSystem();
    text.append(
        formatted("\nconnection system: %lld connections in %lld clusters, %lld connection-only points; "
                  "%lld clusters absorbed into units",
                  static_cast<long long>(connections.connectionCount), static_cast<long long>(connections.clusterCount),
                  static_cast<long long>(connections.connectionNodeIds.size()),
                  static_cast<long long>(connections.absorbedClusterCount)));
    if (!connections.movedUnits.empty())
        text.append("; 1D/0D-only units in it: ").append(idList(connections.movedUnits));
    text.append("\n");
    text.append(interfaceLine("interface problem", plan.interfaceEstimate()));
    text.append(interfaceLine("full path (no unit condensed)", plan.fullPathEstimate()));
    if (!m_keptFullRecommendations.empty())
        text.append("recommendation: keep full the units ")
            .append(idList(m_keptFullRecommendations))
            .append(" (solver.plan(..., full=[...])): their K* would cost more than their interior\n");
}

} // namespace oos::reduction
