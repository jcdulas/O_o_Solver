#include "oos/design/responses/DesignResponses.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <algorithm>
#include <set>
#include <utility>

namespace oos::design
{

DesignResponses::DesignResponses(ResponseParts parts, const DesignModel& model, const input::ModelSnapshot& snapshot)
    : m_parts(std::move(parts))
{
    std::vector<base::Diagnostic> problems;
    const ElementCatalog catalog(snapshot);
    for (const DesignResp1Definition& definition : m_parts.designResp1)
    {
        if (!m_kinds.emplace(definition.id, std::make_pair(Kind::DesignResp1, m_designResp1.size())).second)
            problems.push_back(
                designProblem(design_code::INVALID_RESPONSE, "DesignResp1", definition.id, "id",
                              base::concatenate({"expected unique response IDs across DesignResp1, DesignResp2 "
                                                 "and DesignResp3, found ",
                                                 std::to_string(definition.id), " twice"})));
        m_designResp1.emplace_back(definition, snapshot, catalog, problems);
        m_isGlobal[definition.id] = isGlobalType(definition.type);
        // A static type in the STATICS and SAERO subcases (the trimmed state), the others in their analysis'.
        const SubcaseAnalysis analysis = analysisOfType(definition.type);
        m_analyses[definition.id] =
            isGlobalType(definition.type) ? EVERY_ANALYSIS
            : analysis == SubcaseAnalysis::Statics
                ? static_cast<std::uint8_t>(analysisBit(analysis) | analysisBit(SubcaseAnalysis::Aeroelastic))
                : analysisBit(analysis);
    }
    // The external responses: their IDs unique too, their zones among the problem's.
    std::set<std::int32_t> externalIds;
    for (const ExternalResponseDefinition& definition : m_parts.externals)
    {
        if (!m_kinds.emplace(definition.id, std::make_pair(Kind::External, m_externals.size())).second)
            problems.push_back(designProblem(
                design_code::INVALID_RESPONSE, "ExternalResponse", definition.id, "id",
                base::concatenate({"expected unique response IDs, found ", std::to_string(definition.id), " twice"})));
        m_externals.emplace_back(definition, model, m_parts.zones, problems);
        m_isGlobal[definition.id] = false;
        m_analyses[definition.id] = analysisBit(SubcaseAnalysis::Statics);
        externalIds.insert(definition.id);
    }
    for (const DesignResp1Definition& definition : m_parts.designResp1)
        for (const std::int32_t zone : definition.zones)
            if (std::none_of(m_parts.zones.begin(), m_parts.zones.end(),
                             [&](const output::ZoneDefinition& z) { return z.id == zone; }))
                problems.push_back(designProblem(design_code::INVALID_RESPONSE, "DesignResp1", definition.id, "zone",
                                                 base::concatenate({"names zone ", std::to_string(zone),
                                                                    ", which the problem does not have; declare it "
                                                                    "with add_zone"})));
    // The equations, each after the equations it reads (a depth-first order; a cycle is refused).
    std::map<std::int32_t, std::size_t> equationOf;
    for (std::size_t e = 0; e < m_parts.equations.size(); ++e)
    {
        const EquationResponseDefinition& definition = m_parts.equations[e];
        if (m_kinds.count(definition.id) != 0 || !equationOf.emplace(definition.id, e).second)
            problems.push_back(
                designProblem(design_code::INVALID_RESPONSE, definition.card, definition.id, "id",
                              base::concatenate({"expected unique response IDs across DesignResp1, DesignResp2 "
                                                 "and DesignResp3, found ",
                                                 std::to_string(definition.id), " twice"})));
    }
    for (const EquationResponseDefinition& definition : m_parts.equations)
        for (const ResponseArgument& argument : definition.arguments)
        {
            const bool isDesignResp1 = argument.kind == ArgumentKind::DesignResp1;
            const bool isDesignResp2 = argument.kind == ArgumentKind::DesignResp2;
            if ((isDesignResp1 && m_kinds.count(argument.id) == 0) ||
                (isDesignResp2 && equationOf.count(argument.id) == 0 && externalIds.count(argument.id) == 0))
                problems.push_back(
                    designProblem(design_code::MISSING_REFERENCE, definition.card, definition.id,
                                  isDesignResp1 ? "DesignResp1" : "DesignResp2",
                                  base::concatenate({"expected a ", isDesignResp1 ? "DesignResp1" : "DesignResp2",
                                                     " of ID ", std::to_string(argument.id), ", found none"})));
        }
    std::vector<std::uint8_t> state(m_parts.equations.size(), 0); // 0 unvisited, 1 visiting, 2 done
    std::vector<std::size_t> order;
    const auto visit = [&](auto&& self, std::size_t e) -> void
    {
        if (state[e] == 2)
            return;
        const EquationResponseDefinition& definition = m_parts.equations[e];
        if (state[e] == 1)
        {
            problems.push_back(designProblem(design_code::INVALID_RESPONSE, definition.card, definition.id,
                                             "DesignResp2",
                                             "expected equation responses that do not read themselves, "
                                             "found a cycle through this one"));
            return;
        }
        state[e] = 1;
        for (const ResponseArgument& argument : definition.arguments)
            if (argument.kind == ArgumentKind::DesignResp2)
                if (const auto found = equationOf.find(argument.id); found != equationOf.end())
                    self(self, found->second);
        state[e] = 2;
        order.push_back(e);
    };
    for (std::size_t e = 0; e < m_parts.equations.size(); ++e)
        visit(visit, e);
    throwDesignDiagnostics("the design responses cannot be used", problems);
    for (const std::size_t e : order)
    {
        const EquationResponseDefinition& definition = m_parts.equations[e];
        m_kinds[definition.id] = {Kind::Equation, m_equations.size()};
        m_equations.emplace_back(definition, model, snapshot, problems);
        // By the card rules an equation is global when every response it reads is (by label it follows its labels).
        bool isGlobal = !definition.byLabel;
        bool readsExternal = false;
        // An equation is evaluated in the subcases where every response it reads is (a FREQ in a MODES
        // subcase, a stress in a STATICS one; one of each is evaluated nowhere).
        std::uint8_t analyses = EVERY_ANALYSIS;
        for (const ResponseArgument& argument : definition.arguments)
            if (argument.kind == ArgumentKind::DesignResp1 || argument.kind == ArgumentKind::DesignResp2)
            {
                isGlobal = isGlobal && m_isGlobal[argument.id];
                analyses = static_cast<std::uint8_t>(analyses & m_analyses[argument.id]);
                if (externalIds.count(argument.id) != 0)
                    readsExternal = true;
                else if (const auto found = m_kinds.find(argument.id);
                         found != m_kinds.end() && found->second.first == Kind::Equation)
                    readsExternal = readsExternal || m_readsExternal[found->second.second] != 0;
            }
        m_isGlobal[definition.id] = isGlobal;
        m_analyses[definition.id] = analyses;
        m_readsExternal.push_back(readsExternal ? 1 : 0);
    }
    throwDesignDiagnostics("the design responses cannot be used", problems);
}

std::vector<std::int32_t> DesignResponses::ids() const
{
    std::vector<std::int32_t> ids;
    for (const DesignResp1Definition& definition : m_parts.designResp1)
        ids.push_back(definition.id);
    for (const EquationResponseDefinition& definition : m_parts.equations)
        ids.push_back(definition.id);
    for (const ExternalResponseDefinition& definition : m_parts.externals)
        ids.push_back(definition.id);
    return ids;
}

bool DesignResponses::isEvaluatedIn(std::int32_t id, SubcaseAnalysis analysis) const
{
    const auto found = m_analyses.find(id);
    return found != m_analyses.end() && (found->second & analysisBit(analysis)) != 0;
}

bool DesignResponses::isGlobal(std::int32_t id) const
{
    const auto found = m_isGlobal.find(id);
    return found != m_isGlobal.end() && found->second;
}

std::string DesignResponses::label(std::int32_t id) const
{
    const auto found = m_kinds.find(id);
    if (found == m_kinds.end())
        return "";
    switch (found->second.first)
    {
    case Kind::DesignResp1:
        return m_designResp1[found->second.second].definition().label;
    case Kind::Equation:
        return m_equations[found->second.second].definition().label;
    case Kind::External:
        return m_externals[found->second.second].definition().label;
    }
    return "";
}

std::vector<std::int32_t> DesignResponses::declaredSubcases(std::int32_t id) const
{
    const auto found = m_kinds.find(id);
    if (found == m_kinds.end())
        return {};
    switch (found->second.first)
    {
    case Kind::DesignResp1:
        return m_designResp1[found->second.second].definition().subcases;
    case Kind::Equation:
        return m_equations[found->second.second].definition().subcases;
    case Kind::External:
        return m_externals[found->second.second].definition().subcases;
    }
    return {};
}

std::string DesignResponses::screeningType(std::int32_t id) const
{
    const auto found = m_kinds.find(id);
    if (found == m_kinds.end())
        return "";
    if (found->second.first == Kind::DesignResp1)
        return std::string(responseTypeName(m_designResp1[found->second.second].definition().type));
    // An external response screens as a DesignResp3 (the nearest entry of the input format: a user-function response).
    if (found->second.first == Kind::External)
        return "DesignResp3";
    // The screening type of an equation response is the keyword of its card.
    return m_equations[found->second.second].definition().card == "DesignResp3" ? "DesignResp3" : "DesignResp2";
}

const DesignResp1Response* DesignResponses::findDesignResp1(std::int32_t id) const
{
    const auto found = m_kinds.find(id);
    if (found == m_kinds.end() || found->second.first != Kind::DesignResp1)
        return nullptr;
    return &m_designResp1[found->second.second];
}

const EquationResponse* DesignResponses::findEquation(std::int32_t id) const
{
    const auto found = m_kinds.find(id);
    if (found == m_kinds.end() || found->second.first != Kind::Equation)
        return nullptr;
    return &m_equations[found->second.second];
}

const ExternalResponse* DesignResponses::findExternal(std::int32_t id) const
{
    const auto found = m_kinds.find(id);
    if (found == m_kinds.end() || found->second.first != Kind::External)
        return nullptr;
    return &m_externals[found->second.second];
}

std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> DesignResponses::requiredOutputs() const
{
    std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> outputs{};
    for (const DesignResp1Response& response : m_designResp1)
        response.addRequiredOutputs(outputs);
    return outputs;
}

output::SectionLoadRequest DesignResponses::sectionLoadRequest() const
{
    output::SectionLoadRequest request;
    for (const DesignResp1Response& response : m_designResp1)
    {
        const DesignResp1Definition& definition = response.definition();
        for (output::SectionDefinition section : definition.sections)
        {
            section.group = definition.id;
            request.sections.push_back(std::move(section));
        }
        for (output::CutDefinition cut : definition.cuts)
        {
            cut.group = definition.id;
            request.cuts.push_back(std::move(cut));
        }
    }
    return request;
}

output::ZoneRequest DesignResponses::zoneRequest() const
{
    output::ZoneRequest request;
    std::set<std::int32_t> used;
    bool needsValues = false;
    bool needsBasis = false;
    for (const DesignResp1Response& response : m_designResp1)
        for (const std::int32_t zone : response.definition().zones)
        {
            used.insert(zone);
            needsValues = true;
        }
    for (const ExternalResponse& response : m_externals)
    {
        const ExternalResponseDefinition& definition = response.definition();
        used.insert(definition.zones.begin(), definition.zones.end());
        const bool onBasis = definition.isLinear && definition.coupling == ExternalCoupling::Callback;
        needsBasis = needsBasis || definition.isLinear;
        needsValues = needsValues || !onBasis;
    }
    for (output::ZoneDefinition zone : m_parts.zones)
        if (used.count(zone.id) != 0)
        {
            zone.group = zone.id;
            request.zones.push_back(std::move(zone));
        }
    request.contourSubcases = m_parts.zoneSubcases;
    request.keepsValues = needsValues;
    request.keepsBasis = needsBasis;
    return request;
}

ResponseSubcases DesignResponses::closeSubcases(ResponseSubcases subcases) const
{
    const auto add = [&subcases](std::int32_t id, const std::vector<std::int32_t>& more)
    {
        std::vector<std::int32_t>& list = subcases[id];
        for (const std::int32_t subcase : more)
            if (std::find(list.begin(), list.end(), subcase) == list.end())
                list.push_back(subcase);
    };
    for (auto& [id, list] : subcases)
        if (isGlobal(id))
            list = {0};
    // The equations from the last (the readers) to the first, so that a chain of equations passes its subcases down.
    for (auto equation = m_equations.rbegin(); equation != m_equations.rend(); ++equation)
    {
        const auto found = subcases.find(equation->definition().id);
        if (found == subcases.end())
            continue;
        const std::vector<std::int32_t> own = found->second;
        for (const std::int32_t argument : equation->responseArguments())
            add(argument, isGlobal(argument) ? std::vector<std::int32_t>{0} : own);
    }
    for (auto& [id, list] : subcases)
        if (isGlobal(id))
            list = {0};
    return subcases;
}

std::vector<ResponseValues> DesignResponses::evaluate(const ResponseSubcases& subcases, ResponseContext& context,
                                                      const CallableRegistry& callables, std::size_t blockItems,
                                                      const ExternalRegistry& externals,
                                                      std::map<std::int32_t, ItemSelection>* selections) const
{
    std::map<std::int32_t, ResponseValues> computed;
    KnownResponses known;
    for (const DesignResp1Response& response : m_designResp1)
    {
        const std::int32_t id = response.definition().id;
        const auto found = subcases.find(id);
        if (found == subcases.end())
            continue;
        ResponseValues& values = computed[id];
        values.id = id;
        values.label = response.definition().label;
        for (const std::int32_t subcase : found->second)
            response.evaluate(subcase, context, values);
        known[id] = &values;
    }
    // The equations in two passes: those that read no external response, then (after the external responses) those
    // that read one.
    const auto evaluateEquations = [&](bool readingExternals)
    {
        for (std::size_t e = 0; e < m_equations.size(); ++e)
        {
            if ((m_readsExternal[e] != 0) != readingExternals)
                continue;
            const EquationResponse& equation = m_equations[e];
            const std::int32_t id = equation.definition().id;
            const auto found = subcases.find(id);
            if (found == subcases.end())
                continue;
            for (const std::int32_t argument : equation.responseArguments())
                if (known.count(argument) == 0)
                    // no remedy: closeSubcases evaluates every argument before its equation; a skipped one is a bug.
                    throw base::OptimisationError(
                        base::concatenate({equation.definition().card, " ", std::to_string(id), ": expected response ",
                                           std::to_string(argument),
                                           " to be evaluated before it, found it skipped (closeSubcases was not "
                                           "applied)"}));
            computed[id] = equation.evaluate(found->second, known, context, callables, blockItems);
            known[id] = &computed[id];
        }
    };
    evaluateEquations(false);
    for (const ExternalResponse& response : m_externals)
    {
        const std::int32_t id = response.definition().id;
        const auto found = subcases.find(id);
        if (found == subcases.end())
            continue;
        ExternalValues values = response.evaluate(found->second, known, context, externals, blockItems, false);
        if (selections != nullptr)
            (*selections)[id] = std::move(values.evaluated);
        computed[id] = std::move(values.values);
        known[id] = &computed[id];
    }
    evaluateEquations(true);
    std::vector<ResponseValues> ordered;
    for (const std::int32_t id : ids())
        if (const auto found = computed.find(id); found != computed.end())
            ordered.push_back(std::move(found->second));
    return ordered;
}

} // namespace oos::design
