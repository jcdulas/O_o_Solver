#include "oos/design/constraints/DesignConstraints.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace oos::design
{

namespace
{

// DesignConstr: LALLOW defaults to -1.0E20 and UALLOW to 1.0E20. A DesignConstr with both blank (for instance a FREQ
// kept in the design model for monitoring) is a constraint that never acts, g = -1.
constexpr double DEFAULT_LOWER_BOUND = -1.0e20;
constexpr double DEFAULT_UPPER_BOUND = 1.0e20;

bool contains(const std::vector<std::int32_t>& list, std::int32_t value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

} // namespace

DesignConstraints::DesignConstraints(ConstraintParts parts, const DesignResponses& responses)
    : m_parts(std::move(parts))
{
    std::vector<base::Diagnostic> problems;
    std::set<std::int32_t> sets;
    for (ConstraintDefinition& constraint : m_parts.constraints)
    {
        sets.insert(constraint.setId);
        if (!responses.hasResponse(constraint.responseId))
            problems.push_back(
                designProblem(design_code::MISSING_REFERENCE, "DesignConstr", constraint.setId, "responseId",
                              base::concatenate({"expected a DesignResp1, DesignResp2 or DesignResp3 of ID ",
                                                 std::to_string(constraint.responseId), ", found none"})));
        if (!constraint.lower && !constraint.upper)
        {
            // Both bounds blank: the defaults of the entry, not a refusal.
            constraint.lower = DEFAULT_LOWER_BOUND;
            constraint.upper = DEFAULT_UPPER_BOUND;
        }
        if (constraint.lower && constraint.upper && *constraint.lower > *constraint.upper)
            problems.push_back(designProblem(
                design_code::INVALID_CONSTRAINT, "DesignConstr", constraint.setId, "upperAllowable",
                base::concatenate({"expected DesignConstr.lowerAllowable ≤ DesignConstr.upperAllowable on response ",
                                   std::to_string(constraint.responseId), ", found ", std::to_string(*constraint.lower),
                                   " > ", std::to_string(*constraint.upper)})));
    }
    std::map<std::int32_t, std::size_t> unionOf;
    for (std::size_t u = 0; u < m_parts.unions.size(); ++u)
    {
        if (!unionOf.emplace(m_parts.unions[u].setId, u).second || sets.count(m_parts.unions[u].setId) != 0)
            problems.push_back(designProblem(
                design_code::INVALID_CONSTRAINT, "DesignConstrUnion", m_parts.unions[u].setId, "constraintSetId",
                "expected a DesignConstrUnion ID that no DesignConstr and no other DesignConstrUnion uses, found "
                "it used twice"));
    }
    for (const ConstraintSetUnion& setUnion : m_parts.unions)
        for (const std::int32_t member : setUnion.members)
            if (sets.count(member) == 0 && unionOf.count(member) == 0)
                problems.push_back(designProblem(
                    design_code::MISSING_REFERENCE, "DesignConstrUnion", setUnion.setId, "memberSetId",
                    base::concatenate({"expected a DesignConstr set of ID ", std::to_string(member), ", found none"})));
    // A union that contains itself, directly or through others.
    for (const ConstraintSetUnion& setUnion : m_parts.unions)
    {
        std::set<std::int32_t> seen;
        std::vector<std::int32_t> pending = setUnion.members;
        while (!pending.empty())
        {
            const std::int32_t member = pending.back();
            pending.pop_back();
            if (member == setUnion.setId)
            {
                problems.push_back(designProblem(design_code::INVALID_CONSTRAINT, "DesignConstrUnion", setUnion.setId,
                                                 "memberSetId",
                                                 "expected constraint set unions without a cycle, found one through "
                                                 "this DesignConstrUnion"));
                break;
            }
            if (!seen.insert(member).second)
                continue;
            if (const auto found = unionOf.find(member); found != unionOf.end())
                pending.insert(pending.end(), m_parts.unions[found->second].members.begin(),
                               m_parts.unions[found->second].members.end());
        }
    }
    const auto checkSelected = [&](std::optional<std::int32_t> set, const char* command)
    {
        if (set && sets.count(*set) == 0 && unionOf.count(*set) == 0)
            problems.push_back(
                designProblem(design_code::MISSING_REFERENCE, command, *set, command,
                              base::concatenate({"expected a DesignConstr or DesignConstrUnion set of ID ",
                                                 std::to_string(*set), ", found none"})));
    };
    for (const DesignSelection::Subcase& subcase : m_parts.selection.subcases)
        checkSelected(subcase.dessub, "SubcaseDesignConstrSelection");
    checkSelected(m_parts.selection.desglb, "GlobalDesignConstrSelection");
    if (m_parts.selection.objective && !responses.hasResponse(*m_parts.selection.objective))
        problems.push_back(
            designProblem(design_code::MISSING_REFERENCE, "DesignObjectiveSelection", *m_parts.selection.objective,
                          "DesignObjectiveSelection",
                          base::concatenate({"expected a DesignResp1, DesignResp2 or DesignResp3 of ID ",
                                             std::to_string(*m_parts.selection.objective), ", found none"})));
    if (m_parts.selection.objective && responses.hasResponse(*m_parts.selection.objective) &&
        !responses.isGlobal(*m_parts.selection.objective) && !m_parts.selection.objectiveSubcase)
        problems.push_back(
            designProblem(design_code::INVALID_CONSTRAINT, "DesignObjectiveSelection", *m_parts.selection.objective,
                          "DesignObjectiveSelection",
                          "expected a global response (WEIGHT, VOLUME, an equation of global "
                          "responses) for a DesignObjectiveSelection above the subcases, found a response of a "
                          "subcase; give DesignObjectiveSelection inside the subcase"));
    throwDesignDiagnostics("the design constraints cannot be used", problems);
}

std::set<std::int32_t> DesignConstraints::expandSet(std::int32_t setId) const
{
    std::set<std::int32_t> result;
    std::vector<std::int32_t> pending{setId};
    while (!pending.empty())
    {
        const std::int32_t set = pending.back();
        pending.pop_back();
        if (!result.insert(set).second)
            continue;
        for (const ConstraintSetUnion& setUnion : m_parts.unions)
            if (setUnion.setId == set)
                pending.insert(pending.end(), setUnion.members.begin(), setUnion.members.end());
    }
    return result;
}

std::optional<std::int32_t> DesignConstraints::selectedSet(std::int32_t subcase) const
{
    if (subcase == 0)
        return m_parts.selection.desglb;
    std::optional<std::int32_t> global;
    for (const DesignSelection::Subcase& entry : m_parts.selection.subcases)
    {
        if (entry.subcase == subcase && entry.dessub)
            return entry.dessub;
        if (entry.subcase == 0)
            global = entry.dessub;
    }
    return global;
}

bool DesignConstraints::actsIn(const ConstraintDefinition& constraint, std::int32_t subcase,
                               const DesignResponses& responses) const
{
    if (!constraint.subcases.empty() && !contains(constraint.subcases, subcase))
        return false;
    const bool isGlobal = responses.isGlobal(constraint.responseId);
    if (constraint.setId == 0)
        return isGlobal == (subcase == 0);
    if (!isGlobal)
    {
        const std::optional<std::int32_t> set = subcase == 0 ? std::nullopt : selectedSet(subcase);
        return set && expandSet(*set).count(constraint.setId) != 0;
    }
    // A global response acts once, at subcase 0, whichever selection names its set.
    return false;
}

ResponseSubcases DesignConstraints::subcasesOfResponses(std::span<const std::int32_t> analysed,
                                                        const DesignResponses& responses) const
{
    ResponseSubcases subcases;
    const auto add = [&subcases](std::int32_t id, std::int32_t subcase)
    {
        std::vector<std::int32_t>& list = subcases[id];
        if (!contains(list, subcase))
            list.push_back(subcase);
    };
    std::vector<std::int32_t> all{0};
    all.insert(all.end(), analysed.begin(), analysed.end());
    for (const ConstraintDefinition& constraint : m_parts.constraints)
    {
        if (responses.isGlobal(constraint.responseId))
        {
            add(constraint.responseId, 0);
            continue;
        }
        for (const std::int32_t subcase : analysed)
            if (actsIn(constraint, subcase, responses))
                add(constraint.responseId, subcase);
    }
    if (const std::optional<std::int32_t> objective = m_parts.selection.objective)
        add(*objective, responses.isGlobal(*objective) ? 0 : m_parts.selection.objectiveSubcase.value_or(0));
    for (const std::int32_t id : responses.ids())
    {
        const std::vector<std::int32_t> declared = responses.declaredSubcases(id);
        if (responses.isGlobal(id))
        {
            add(id, 0);
            continue;
        }
        for (const std::int32_t subcase : declared)
            if (contains(std::vector<std::int32_t>(analysed.begin(), analysed.end()), subcase))
                add(id, subcase);
        if (subcases.count(id) == 0 && declared.empty())
            for (const std::int32_t subcase : analysed)
                add(id, subcase); // a monitored response: every analysed subcase
    }
    // In the order of the analysis.
    for (auto& [id, list] : subcases)
        std::stable_sort(list.begin(), list.end(), [&all](std::int32_t a, std::int32_t b)
                         { return std::find(all.begin(), all.end(), a) < std::find(all.begin(), all.end(), b); });
    return subcases;
}

std::vector<ConstraintValue> DesignConstraints::evaluate(const std::vector<ResponseValues>& values,
                                                         std::span<const std::int32_t> analysed,
                                                         const DesignResponses& responses, double constraintScale) const
{
    std::map<std::int32_t, const ResponseValues*> byId;
    for (const ResponseValues& response : values)
        byId[response.id] = &response;
    std::vector<ConstraintValue> result;
    const ConstraintNormalisation normalisation(constraintScale);
    std::vector<std::int32_t> subcases{0};
    subcases.insert(subcases.end(), analysed.begin(), analysed.end());
    for (const ConstraintDefinition& constraint : m_parts.constraints)
    {
        const auto found = byId.find(constraint.responseId);
        if (found == byId.end())
            continue;
        const ResponseValues& response = *found->second;
        const bool isGlobal = responses.isGlobal(constraint.responseId);
        for (const std::int32_t subcase : subcases)
        {
            bool acts = actsIn(constraint, subcase, responses);
            if (isGlobal && subcase == 0 && constraint.setId != 0)
            {
                // A constraint on a global response acts once when its set is selected anywhere.
                acts = false;
                for (const std::int32_t candidate : subcases)
                {
                    const std::optional<std::int32_t> set = selectedSet(candidate);
                    acts = acts || (set && expandSet(*set).count(constraint.setId) != 0);
                }
            }
            if (!acts)
                continue;
            for (std::size_t i = 0; i < response.items.size(); ++i)
            {
                // An item that failed or was not evaluated (an external response) is no constraint value: such items
                // are excluded from the constraints.
                if (response.items[i].subcase != subcase || !response.isOk(i))
                    continue;
                const double r = response.values[i];
                if (constraint.lower)
                {
                    const double l = *constraint.lower;
                    result.push_back({constraint.setId, constraint.responseId, i, response.items[i], BoundSide::Lower,
                                      l, r, normalisation.normalise(BoundSide::Lower, l, r)});
                }
                if (constraint.upper)
                {
                    const double u = *constraint.upper;
                    result.push_back({constraint.setId, constraint.responseId, i, response.items[i], BoundSide::Upper,
                                      u, r, normalisation.normalise(BoundSide::Upper, u, r)});
                }
            }
        }
    }
    return result;
}

std::optional<double> DesignConstraints::objectiveValue(const std::vector<ResponseValues>& values) const
{
    const std::optional<std::int32_t> objective = m_parts.selection.objective;
    if (!objective)
        return std::nullopt;
    for (const ResponseValues& response : values)
    {
        if (response.id != *objective)
            continue;
        const std::int32_t subcase = m_parts.selection.objectiveSubcase.value_or(0);
        std::vector<double> found;
        for (std::size_t i = 0; i < response.items.size(); ++i)
            if (response.items[i].subcase == subcase || response.items[i].subcase == 0)
                found.push_back(response.values[i]);
        if (found.size() != 1)
            throw base::OptimisationError(base::concatenate(
                {"DesignObjectiveSelection ", std::to_string(*objective),
                 ": expected an objective response of one item, found ", std::to_string(found.size()), " items (",
                 response.label, "); an objective is one value: use an equation (MAX, SUM...) of the items"}));
        return found.front();
    }
    return std::nullopt;
}

} // namespace oos::design
