#include "oos/reduction/units/UnitBoundaries.hpp"

#include "oos/reduction/absorption/SpiderAbsorption.hpp"

#include <algorithm>
#include <map>

namespace oos::reduction
{

std::string describeReasons(std::uint8_t reasons)
{
    static constexpr std::pair<BoundaryReason, const char*> NAMES[] = {
        {BoundaryReason::Interface, "interface"},
        {BoundaryReason::SharedPoint, "shared point"},
        {BoundaryReason::Retained, "retained"},
        {BoundaryReason::PartialConstraint, "constrained in some subcases"},
        {BoundaryReason::Support, "FreeBodySupport"},
        {BoundaryReason::AbsorbedSpider, "spider reference"}};
    std::string text;
    for (const auto& [reason, name] : NAMES)
    {
        if ((reasons & static_cast<std::uint8_t>(reason)) == 0)
            continue;
        if (!text.empty())
            text.append(", ");
        text.append(name);
    }
    return text;
}

UnitBoundaries::UnitBoundaries(const ClusterAbsorption& absorption, const BoundaryInputs& inputs,
                               const SpiderAbsorption* spiders)
{
    const ConnectionSystem& system = absorption.system();
    std::vector<base::EntityId> interfaceUse;
    for (const std::size_t c : absorption.remainingClusters())
        for (const std::size_t connection : system.clusters()[c].connections)
        {
            const std::vector<base::EntityId>& nodes = system.connections()[connection].nodes;
            const std::span<const base::EntityId> removed =
                spiders != nullptr ? spiders->removedUses(connection) : std::span<const base::EntityId>{};
            for (const base::EntityId nodeId : nodes)
                if (!std::binary_search(removed.begin(), removed.end(), nodeId))
                    interfaceUse.push_back(nodeId);
        }
    std::sort(interfaceUse.begin(), interfaceUse.end());
    interfaceUse.erase(std::unique(interfaceUse.begin(), interfaceUse.end()), interfaceUse.end());

    // The points of each unit: its own, and the points its absorbed spiders put on its boundary (an RigidInterpolation
    // reference that was a connection-only point becomes a point of the unit).
    std::vector<std::vector<base::EntityId>> unitNodes(absorption.units().size());
    std::map<base::EntityId, int> holders;
    for (std::size_t index = 0; index < absorption.units().size(); ++index)
    {
        std::vector<base::EntityId>& nodes = unitNodes[index];
        nodes = absorption.units()[index].nodeIds;
        if (spiders != nullptr)
        {
            const std::span<const base::EntityId> references = spiders->referenceNodes(index);
            nodes.insert(nodes.end(), references.begin(), references.end());
            std::sort(nodes.begin(), nodes.end());
            nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        }
        for (const base::EntityId nodeId : nodes)
            ++holders[nodeId];
    }

    const auto contains = [](const std::vector<base::EntityId>& sorted, base::EntityId nodeId)
    { return std::binary_search(sorted.begin(), sorted.end(), nodeId); };
    m_boundaries.resize(absorption.units().size());
    for (std::size_t index = 0; index < absorption.units().size(); ++index)
    {
        const std::span<const base::EntityId> references =
            spiders != nullptr ? spiders->referenceNodes(index) : std::span<const base::EntityId>{};
        for (const base::EntityId nodeId : unitNodes[index])
        {
            std::uint8_t reasons = 0;
            const auto add = [&](bool condition, BoundaryReason reason)
            {
                if (condition)
                    reasons |= static_cast<std::uint8_t>(reason);
            };
            add(contains(interfaceUse, nodeId), BoundaryReason::Interface);
            add(holders[nodeId] > 1, BoundaryReason::SharedPoint);
            add(contains(inputs.retainedNodes, nodeId), BoundaryReason::Retained);
            add(contains(inputs.partiallyConstrainedNodes, nodeId), BoundaryReason::PartialConstraint);
            add(contains(inputs.supportNodes, nodeId), BoundaryReason::Support);
            add(std::binary_search(references.begin(), references.end(), nodeId), BoundaryReason::AbsorbedSpider);
            if (reasons != 0)
                m_boundaries[index].push_back(BoundaryNode{nodeId, reasons});
        }
    }
}

} // namespace oos::reduction
