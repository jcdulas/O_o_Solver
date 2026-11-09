#include "oos/design/approximation/RetainedResponses.hpp"

#include "oos/design/responses/ElementCatalog.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace oos::design
{

namespace
{

// What the screening needs of a response: its DesignConstrScreen type, its REGION and the kind of entity of its items.
struct ResponseScreening
{
    std::string type;
    std::optional<std::int32_t> region;
    bool isGlobal = false;
    bool isElement = false;
    bool isGrid = false;
};

ResponseScreening screeningOf(const DesignResponses& responses, std::int32_t id)
{
    ResponseScreening result;
    if (const DesignResp1Response* response = responses.findDesignResp1(id))
    {
        const DesignResp1Definition& definition = response->definition();
        result.type = std::string(responseTypeName(definition.type));
        result.region = definition.region;
        result.isGlobal = isGlobalType(definition.type);
        result.isElement = isElementType(definition.type);
        result.isGrid = definition.type == ResponseType::Displacement || definition.type == ResponseType::SpcForce ||
                        definition.type == ResponseType::DeflectionToThickness ||
                        definition.type == ResponseType::Rotation;
    }
    else if (const EquationResponse* equation = responses.findEquation(id))
    {
        // DesignConstrScreen: RTYPE = EQUA screens the DesignResp2 constraints, DesignResp3 those of the DesignResp3.
        result.type = equation->definition().card == "DesignResp3" ? "DesignResp3" : "Equation";
        result.region = equation->definition().region;
    }
    return result;
}

ScreeningRegion regionOf(const ResponseScreening& response, std::int32_t responseId, const ResponseItem& item,
                         const ElementCatalog& catalog)
{
    if (response.isGlobal)
        return {ScreeningRegion::Kind::Unlimited, 0, ""};
    if (response.region)
        return {ScreeningRegion::Kind::Region, *response.region, ""};
    if (response.isElement)
        if (const ElementEntry* element = catalog.find(item.entity))
            return {ScreeningRegion::Kind::Property, element->propId, element->propertyCard};
    return {ScreeningRegion::Kind::Response, responseId, ""};
}

} // namespace

RetainedResponses::RetainedResponses(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                     const ExplicitConstraintGroup& group, double constraintScale)
{
    const ConstraintScreening screening(problem.constraints().screening());
    const ConstraintNormalisation normalisation(constraintScale);
    const ElementCatalog catalog(problem.snapshot());
    std::map<std::int32_t, ResponseScreening> byResponse;
    std::vector<ScreeningEntry> entries;
    std::set<std::string> typesSeen;
    for (std::size_t k = 0; k < evaluation.constraints.size(); ++k)
    {
        const ConstraintValue& constraint = evaluation.constraints[k];
        auto found = byResponse.find(constraint.responseId);
        if (found == byResponse.end())
            found = byResponse.emplace(constraint.responseId, screeningOf(problem.responses(), constraint.responseId))
                        .first;
        const ResponseScreening& response = found->second;
        ScreeningEntry entry;
        entry.responseType = response.type;
        entry.region = regionOf(response, constraint.responseId, constraint.label, catalog);
        entry.subcase = constraint.label.subcase;
        entry.normalised = normalisation.normalise(constraint.side, constraint.bound, constraint.value);
        entry.isExplicit = group.contains(constraint, response.region, response.isElement, response.isGrid);
        entries.push_back(std::move(entry));
        if (typesSeen.insert(response.type).second)
            m_report.rules.push_back(screening.ruleFor(response.type));
    }
    m_statuses = screening.screen(entries);

    // The retained items: those of the retained constraints, and the objective's.
    std::map<std::int32_t, std::size_t> positionOf;
    for (std::size_t p = 0; p < evaluation.responses.size(); ++p)
        positionOf[evaluation.responses[p].id] = p;
    std::vector<std::vector<std::uint8_t>> retainedItems(evaluation.responses.size());
    const auto retain = [&](std::int32_t responseId, std::size_t item)
    {
        const auto position = positionOf.find(responseId);
        if (position == positionOf.end())
            return;
        std::vector<std::uint8_t>& mask = retainedItems[position->second];
        mask.resize(evaluation.responses[position->second].itemCount(), 0);
        mask[item] = 1;
    };
    for (std::size_t k = 0; k < evaluation.constraints.size(); ++k)
    {
        const ConstraintValue& constraint = evaluation.constraints[k];
        ScreeningReportRow row;
        row.constraint = k;
        row.responseId = constraint.responseId;
        row.label = problem.responses().label(constraint.responseId);
        row.setId = constraint.setId;
        row.item = constraint.label;
        row.side = constraint.side;
        row.bound = constraint.bound;
        row.value = constraint.value;
        row.normalised = entries[k].normalised;
        row.responseType = entries[k].responseType;
        row.region = entries[k].region;
        row.status = m_statuses[k];
        m_report.rows.push_back(std::move(row));
        switch (m_statuses[k])
        {
        case ScreeningStatus::KeptExplicitly:
            ++m_report.keptExplicitly;
            break;
        case ScreeningStatus::RetainedByScreening:
            ++m_report.retainedByScreening;
            break;
        case ScreeningStatus::ScreenedOut:
            ++m_report.screenedOut;
            continue;
        }
        m_retained.push_back(k);
        retain(constraint.responseId, constraint.item);
    }
    if (const std::optional<std::int32_t> objective = problem.constraints().objective())
        if (const auto position = positionOf.find(*objective); position != positionOf.end())
            for (std::size_t i = 0; i < evaluation.responses[position->second].itemCount(); ++i)
                retain(*objective, i);
    for (std::size_t p = 0; p < evaluation.responses.size(); ++p)
    {
        if (!retainedItems[p].empty())
            m_masks.push_back({evaluation.responses[p].id, std::move(retainedItems[p])});
        m_empty.push_back(
            {evaluation.responses[p].id, std::vector<std::uint8_t>(evaluation.responses[p].itemCount(), 0)});
    }
}

std::size_t RetainedResponses::retainedItemCount() const noexcept
{
    std::size_t count = 0;
    for (const ItemMask& mask : m_masks)
        count += static_cast<std::size_t>(std::count(mask.isRetained.begin(), mask.isRetained.end(), 1));
    return count;
}

GradientOptions RetainedResponses::gradientOptions(SensitivityMethod method) const
{
    GradientOptions options;
    options.method = method;
    options.itemMasks = m_masks.empty() ? m_empty : m_masks;
    for (const ItemMask& mask : options.itemMasks)
        options.responseIds.push_back(mask.responseId);
    return options;
}

} // namespace oos::design
