#include "oos/design/external/ExternalResponse.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/model/DesignCodes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace oos::design
{

namespace
{

constexpr double NOT_A_VALUE = std::numeric_limits<double>::quiet_NaN();

bool isZoneData(ExternalArgumentKind kind)
{
    return kind != ExternalArgumentKind::DesignVariable;
}

// What one zone gives per evaluated case: its outputs, their statuses and, with partials, the terms of each output.
struct CaseOutputs
{
    std::vector<double> values;
    std::vector<ItemStatus> statuses;
    std::vector<std::vector<ExternalTerm>> partials; // per output
};

std::size_t componentsOf(ExternalArgumentKind kind, const output::ZoneTable& table)
{
    switch (kind)
    {
    case ExternalArgumentKind::ContourDisplacement:
    case ExternalArgumentKind::ContourForce:
        return table.contourDofCount();
    case ExternalArgumentKind::ZoneLoads:
        return table.interiorDofCount();
    case ExternalArgumentKind::DesignVariable:
        return 1;
    }
    return 0;
}

} // namespace

ExternalResponse::ExternalResponse(ExternalResponseDefinition definition, const DesignModel& model,
                                   std::span<const output::ZoneDefinition> zones,
                                   std::vector<base::Diagnostic>& problems)
    : m_definition(std::move(definition))
{
    const auto report = [&](std::string field, std::string message)
    {
        problems.push_back(designProblem(design_code::INVALID_RESPONSE, "ExternalResponse", m_definition.id,
                                         std::move(field), std::move(message)));
    };
    if (m_definition.zones.empty())
        report("zones",
               "expected the zones of the external response (its labels), found none; give the zones of the "
               "plugin");
    for (const std::int32_t zone : m_definition.zones)
        if (std::none_of(zones.begin(), zones.end(), [&](const output::ZoneDefinition& z) { return z.id == zone; }))
            report("zones", base::concatenate({"names zone ", std::to_string(zone),
                                               ", which the problem does not have; declare it with add_zone"}));
    if (!m_definition.outputs.empty() && m_definition.outputs.size() != m_definition.zones.size())
        report("outputs",
               base::concatenate({"expected one output count per zone (", std::to_string(m_definition.zones.size()),
                                  "), found ", std::to_string(m_definition.outputs.size())}));
    if (m_definition.evaluator.empty())
        report("evaluator", "expected the name of the evaluator of the response, found none");
    if (m_definition.hasPost && !m_definition.isLinear)
        report("post",
               "post= combines the outputs of a linear plugin on the load basis: give linear=True, or compute "
               "the outputs of each case in the plugin");
    m_variables.resize(m_definition.arguments.size());
    for (std::size_t a = 0; a < m_definition.arguments.size(); ++a)
    {
        const ExternalArgument& argument = m_definition.arguments[a];
        if (argument.kind != ExternalArgumentKind::DesignVariable)
            continue;
        if (argument.variableByLabel.empty())
        {
            const std::optional<std::size_t> index = model.variables().findIndex(argument.variable);
            if (!index)
                report(argument.name, base::concatenate({"expected a DesignVar of ID ",
                                                         std::to_string(argument.variable), ", found none"}));
            else
                for (const std::int32_t zone : m_definition.zones)
                    m_variables[a][zone] = *index;
            continue;
        }
        for (const auto& [zone, variable] : argument.variableByLabel)
        {
            const std::optional<std::size_t> index = model.variables().findIndex(variable);
            if (!index)
                report(argument.name,
                       base::concatenate({"expected a design variable for zone ", std::to_string(zone), ", found ",
                                          std::to_string(variable), ", which is not defined"}));
            else
                m_variables[a][zone] = *index;
        }
        for (const std::int32_t zone : m_definition.zones)
            if (m_variables[a].count(zone) == 0)
                report(argument.name, base::concatenate({"expected a design variable for every zone, found none for "
                                                         "zone ",
                                                         std::to_string(zone)}));
    }
}

std::string ExternalResponse::name() const
{
    return base::concatenate({"external response ", std::to_string(m_definition.id), " (", m_definition.label, ")"});
}

std::size_t ExternalResponse::variableOf(std::size_t argument, std::int32_t zone) const
{
    return m_variables[argument].at(zone);
}

const output::ZoneTable& ExternalResponse::tableOf(std::int32_t zone, const ResponseContext& context) const
{
    for (const output::ZoneTable& table : context.zones)
        if (table.id == zone)
            return table;
    throw base::OptimisationError(
        base::concatenate({name(), ": expected the contour data of zone ", std::to_string(zone),
                           " in the analysis (DesignProblem::evaluateStatics computes them), found none; evaluate "
                           "the response on an analysis of the design problem that declares the zone"}));
}

std::vector<BatchArgument> ExternalResponse::gather(const output::ZoneTable& table, std::int32_t zone,
                                                    ExternalRows kind, std::size_t set,
                                                    std::span<const std::size_t> rows, ResponseContext& context) const
{
    std::vector<BatchArgument> arguments;
    for (std::size_t a = 0; a < m_definition.arguments.size(); ++a)
    {
        const ExternalArgument& argument = m_definition.arguments[a];
        BatchArgument slot{argument.name, componentsOf(argument.kind, table), {}};
        slot.values.reserve(rows.size() * slot.components);
        if (argument.kind == ExternalArgumentKind::DesignVariable)
        {
            const double value = context.design->variables.values[variableOf(a, zone)];
            slot.values.assign(rows.size(), value);
            arguments.push_back(std::move(slot));
            continue;
        }
        const bool isContour = argument.kind != ExternalArgumentKind::ZoneLoads;
        for (const std::size_t row : rows)
        {
            if (kind == ExternalRows::Basis)
            {
                const output::ZoneBasisSet& basis = table.basis[set];
                const linalg::DenseMatrix& matrix = argument.kind == ExternalArgumentKind::ContourDisplacement
                                                        ? basis.displacements
                                                        : (isContour ? basis.forces : basis.loads);
                for (std::size_t r = 0; r < slot.components; ++r)
                    slot.values.push_back(matrix(r, row));
                continue;
            }
            const base::BudgetedBuffer<double>& values = argument.kind == ExternalArgumentKind::ContourDisplacement
                                                             ? table.displacements
                                                             : (isContour ? table.forces : table.loads);
            const double* first = values.data() + row * slot.components;
            slot.values.insert(slot.values.end(), first, first + slot.components);
        }
        arguments.push_back(std::move(slot));
    }
    return arguments;
}

ExternalValues ExternalResponse::evaluate(std::span<const std::int32_t> subcases, const KnownResponses& known,
                                          ResponseContext& context, const ExternalRegistry& evaluators,
                                          std::size_t blockItems, bool withPartials) const
{
    if (blockItems == 0)
        throw std::invalid_argument("ExternalResponse::evaluate: expected a block of at least one item, found 0");
    const auto found = evaluators.find(m_definition.evaluator);
    if (found == evaluators.end() || found->second == nullptr)
        throw base::OptimisationError(base::concatenate(
            {name(), ": expected its plugin or supplied values to be registered (evaluator '", m_definition.evaluator,
             "'), found none; register the plugin "
             "(DesignProblem.add_external_model) or supply the values before the evaluation"}));
    ExternalEvaluator& evaluator = *found->second;
    ExternalValues result;
    result.values.id = m_definition.id;
    result.values.label = m_definition.label;
    ItemSelection selection;
    if (m_definition.hasSelection)
        selection = evaluator.select(m_definition, subcases, known);
    const std::size_t argumentCount = m_definition.arguments.size();
    bool hasStatus = false;
    for (std::size_t zi = 0; zi < m_definition.zones.size(); ++zi)
    {
        const std::int32_t zone = m_definition.zones[zi];
        const output::ZoneTable& table = tableOf(zone, context);
        // The selected subcases of the zone, in the order of `subcases`, that the analysis gave.
        std::vector<std::int32_t> selected;
        const auto chosen = selection.find(zone);
        for (const std::int32_t subcase : subcases)
        {
            if (chosen != selection.end() &&
                std::find(chosen->second.begin(), chosen->second.end(), subcase) == chosen->second.end())
                continue;
            if (std::find(table.cases.begin(), table.cases.end(), subcase) == table.cases.end())
                throw base::OptimisationError(
                    base::concatenate({name(), ": the analysis did not solve subcase ", std::to_string(subcase),
                                       " for zone ", std::to_string(zone),
                                       "; analyse every subcase the response reads (DesignProblem(subcases=...))"}));
            selected.push_back(subcase);
        }
        result.evaluated[zone] = selected;
        std::size_t outputs = m_definition.outputs.empty() ? 0 : static_cast<std::size_t>(m_definition.outputs[zi]);
        std::map<std::int32_t, CaseOutputs> byCase;
        // The answer of a call: `rows` rows of `expected` outputs (0: the first answer sets it).
        const auto checkOutputs =
            [&](const ExternalResult& answer, std::size_t rows, const char* what, std::size_t& expected)
        {
            if (answer.outputs == 0 || answer.values.size() != rows * answer.outputs ||
                (expected != 0 && answer.outputs != expected))
                throw base::OptimisationError(base::concatenate(
                    {name(), ", zone ", std::to_string(zone), ": expected ", what, " of ", std::to_string(rows),
                     " rows and ", expected == 0 ? std::string("the same number of") : std::to_string(expected),
                     " outputs, found ", std::to_string(answer.values.size()), " values of ",
                     std::to_string(answer.outputs),
                     " outputs; make the plugin return one row per row of its arguments, with the outputs it declares "
                     "(DesignProblem.check_external finds such a plugin)"}));
            if (!answer.statuses.empty() && answer.statuses.size() != answer.values.size())
                throw base::OptimisationError(
                    base::concatenate({name(), ", zone ", std::to_string(zone), ": expected a status per value, found ",
                                       std::to_string(answer.statuses.size()),
                                       "; make the plugin return one status per value, or none"}));
            expected = answer.outputs;
        };
        const auto statusOf = [](const ExternalResult& answer, std::size_t k)
        {
            ItemStatus status = answer.statuses.empty() ? ItemStatus::Ok : answer.statuses[k];
            if (!std::isfinite(answer.values[k]) && status == ItemStatus::Ok)
                status = ItemStatus::Failed;
            return status;
        };

        if (!m_definition.isLinear || m_definition.coupling == ExternalCoupling::Supplied)
        {
            const ExternalRows kind =
                m_definition.coupling == ExternalCoupling::Supplied ? ExternalRows::Supplied : ExternalRows::Cases;
            if (kind == ExternalRows::Cases && !selected.empty() && !table.hasValues)
                throw base::OptimisationError(base::concatenate(
                    {name(), ": expected the contour data of every case of zone ", std::to_string(zone),
                     " in the analysis, found none (they were not kept); "
                     "evaluate the response on an analysis of the design problem, which keeps "
                     "them for the external responses"}));
            for (std::size_t first = 0; first < selected.size(); first += blockItems)
            {
                const std::size_t count = std::min(blockItems, selected.size() - first);
                ExternalCall call;
                call.response = &m_definition;
                call.zone = zone;
                call.kind = kind;
                call.cases.assign(selected.begin() + static_cast<std::ptrdiff_t>(first),
                                  selected.begin() + static_cast<std::ptrdiff_t>(first + count));
                call.rows = count;
                if (kind == ExternalRows::Cases)
                {
                    std::vector<std::size_t> rows;
                    for (const std::int32_t subcase : call.cases)
                        rows.push_back(static_cast<std::size_t>(
                            std::find(table.cases.begin(), table.cases.end(), subcase) - table.cases.begin()));
                    call.arguments = gather(table, zone, kind, 0, rows, context);
                }
                const ExternalResult answer = evaluator.evaluate(call, withPartials);
                checkOutputs(answer, count, "values", outputs);
                if (withPartials && answer.partials.size() != argumentCount)
                    throw base::OptimisationError(base::concatenate(
                        {name(), ", zone ", std::to_string(zone), ": expected the partials of ",
                         std::to_string(argumentCount), " arguments, found ", std::to_string(answer.partials.size()),
                         "; make the plugin return one partial per argument (DesignProblem.check_external checks "
                         "them)"}));
                for (std::size_t i = 0; i < count; ++i)
                {
                    CaseOutputs& entry = byCase[call.cases[i]];
                    entry.partials.resize(outputs);
                    for (std::size_t o = 0; o < outputs; ++o)
                    {
                        const std::size_t k = i * outputs + o;
                        entry.values.push_back(answer.values[k]);
                        entry.statuses.push_back(statusOf(answer, k));
                        if (!withPartials)
                            continue;
                        for (std::size_t a = 0; a < argumentCount; ++a)
                        {
                            const std::size_t components = componentsOf(m_definition.arguments[a].kind, table);
                            const std::vector<double>& partial = answer.partials[a];
                            if (partial.size() != count * outputs * components)
                                throw base::OptimisationError(base::concatenate(
                                    {name(), ", zone ", std::to_string(zone), ": expected the partials of argument '",
                                     m_definition.arguments[a].name, "' of shape (", std::to_string(count), ", ",
                                     std::to_string(outputs), ", ", std::to_string(components), "), found ",
                                     std::to_string(partial.size()),
                                     " values; make the plugin return partials of that shape "
                                     "(DesignProblem.check_external checks them)"}));
                            for (std::size_t c = 0; c < components; ++c)
                                if (const double weight = partial[(i * outputs + o) * components + c]; weight != 0.0)
                                    entry.partials[o].push_back({a, c, weight});
                        }
                    }
                }
            }
        }
        else
        {
            // A linear plugin on the load basis of each constraint set: its outputs on the k basis
            // vectors, combined for the selected cases of the set, then post= of each case.
            if (!selected.empty() && table.basis.empty())
                throw base::OptimisationError(
                    base::concatenate({name(), ": expected the contour data of zone ", std::to_string(zone),
                                       " on the load basis (a linear plugin works on them), found none; evaluate the "
                                       "response on an analysis of the design problem, which keeps them for the "
                                       "linear plugins"}));
            std::size_t linearOutputs = 0;
            for (std::size_t s = 0; s < table.basis.size(); ++s)
            {
                const output::ZoneBasisSet& basis = table.basis[s];
                std::vector<std::int32_t> cases;
                std::vector<std::size_t> columns;
                for (const std::int32_t subcase : selected)
                    if (const auto column = std::find(basis.cases.begin(), basis.cases.end(), subcase);
                        column != basis.cases.end())
                    {
                        cases.push_back(subcase);
                        columns.push_back(static_cast<std::size_t>(column - basis.cases.begin()));
                    }
                if (cases.empty())
                    continue;
                if (basis.hasThermalCases)
                    throw base::OptimisationError(base::concatenate(
                        {name(), ", zone ", std::to_string(zone), ": constraint set ",
                         std::to_string(basis.constraintSet),
                         " has a thermal load, whose contour forces are not the combination of the load basis' in this "
                         "version; declare the plugin linear=False (it then gets the data of each case)"}));
                const std::size_t k = basis.basisSize();
                std::vector<double> linear;                                    // k × linearOutputs
                std::vector<ItemStatus> linearStatuses;                        // k × linearOutputs
                std::vector<std::vector<double>> operators(argumentCount);     // zone data: linearOutputs × components
                std::vector<std::vector<double>> variableRates(argumentCount); // variables: k × linearOutputs
                for (std::size_t first = 0; first < k; first += blockItems)
                {
                    const std::size_t count = std::min(blockItems, k - first);
                    std::vector<std::size_t> rows(count);
                    for (std::size_t j = 0; j < count; ++j)
                        rows[j] = first + j;
                    ExternalCall call;
                    call.response = &m_definition;
                    call.zone = zone;
                    call.kind = ExternalRows::Basis;
                    call.cases = cases;
                    call.rows = count;
                    call.constraintSet = basis.constraintSet;
                    call.arguments = gather(table, zone, ExternalRows::Basis, s, rows, context);
                    const ExternalResult answer = evaluator.evaluate(call, withPartials);
                    checkOutputs(answer, count, "linear outputs", linearOutputs);
                    linear.insert(linear.end(), answer.values.begin(), answer.values.end());
                    for (std::size_t q = 0; q < answer.values.size(); ++q)
                        linearStatuses.push_back(statusOf(answer, q));
                    if (!withPartials)
                        continue;
                    if (answer.partials.size() != argumentCount)
                        throw base::OptimisationError(
                            base::concatenate({name(), ", zone ", std::to_string(zone), ": expected the partials of ",
                                               std::to_string(argumentCount), " arguments, found ",
                                               std::to_string(answer.partials.size()),
                                               "; make the plugin return one linear operator per argument "
                                               "(DesignProblem.check_external checks them)"}));
                    for (std::size_t a = 0; a < argumentCount; ++a)
                    {
                        const ExternalArgument& argument = m_definition.arguments[a];
                        const std::vector<double>& partial = answer.partials[a];
                        if (isZoneData(argument.kind))
                        {
                            const std::size_t components = componentsOf(argument.kind, table);
                            if (partial.size() != linearOutputs * components)
                                throw base::OptimisationError(base::concatenate(
                                    {name(), ", zone ", std::to_string(zone),
                                     ": expected the linear operator of "
                                     "argument '",
                                     argument.name, "' of shape (", std::to_string(linearOutputs), ", ",
                                     std::to_string(components), "), found ", std::to_string(partial.size()),
                                     " values; make the plugin return operators of that shape "
                                     "(DesignProblem.check_external checks them)"}));
                            operators[a] = partial;
                        }
                        else
                        {
                            if (partial.size() != count * linearOutputs)
                                throw base::OptimisationError(base::concatenate(
                                    {name(), ", zone ", std::to_string(zone),
                                     ": expected the derivatives of argument '", argument.name, "' of shape (",
                                     std::to_string(count), ", ", std::to_string(linearOutputs), ", 1), found ",
                                     std::to_string(partial.size()),
                                     " values; make the plugin return derivatives of that shape "
                                     "(DesignProblem.check_external checks them)"}));
                            variableRates[a].insert(variableRates[a].end(), partial.begin(), partial.end());
                        }
                    }
                }
                // Y_c = Σ_j α(j, c)·Y_j, the basis vectors in order (one chain per value: the same bits whatever the
                // batches).
                const std::size_t n = cases.size();
                std::vector<double> combined(n * linearOutputs, 0.0);
                std::vector<ItemStatus> combinedStatuses(n * linearOutputs, ItemStatus::Ok);
                for (std::size_t c = 0; c < n; ++c)
                    for (std::size_t j = 0; j < k; ++j)
                    {
                        const double alpha = basis.coefficients(j, columns[c]);
                        if (alpha == 0.0)
                            continue;
                        for (std::size_t q = 0; q < linearOutputs; ++q)
                        {
                            combined[c * linearOutputs + q] += alpha * linear[j * linearOutputs + q];
                            combinedStatuses[c * linearOutputs + q] = combinedStatus(
                                combinedStatuses[c * linearOutputs + q], linearStatuses[j * linearOutputs + q]);
                        }
                    }
                std::size_t finalOutputs = linearOutputs;
                std::vector<double> finals = combined;
                std::vector<ItemStatus> finalStatuses = combinedStatuses;
                std::vector<double> postRates; // n × final outputs × linear outputs
                if (m_definition.hasPost)
                {
                    finals.clear();
                    finalStatuses.clear();
                    finalOutputs = 0;
                    for (std::size_t first = 0; first < n; first += blockItems)
                    {
                        const std::size_t count = std::min(blockItems, n - first);
                        ExternalPostCall call;
                        call.response = &m_definition;
                        call.zone = zone;
                        call.cases.assign(cases.begin() + static_cast<std::ptrdiff_t>(first),
                                          cases.begin() + static_cast<std::ptrdiff_t>(first + count));
                        call.outputs = linearOutputs;
                        call.values.assign(
                            combined.begin() + static_cast<std::ptrdiff_t>(first * linearOutputs),
                            combined.begin() + static_cast<std::ptrdiff_t>((first + count) * linearOutputs));
                        const ExternalResult answer = evaluator.post(call, withPartials);
                        if (answer.outputs == 0 || answer.values.size() != count * answer.outputs ||
                            (finalOutputs != 0 && answer.outputs != finalOutputs))
                            throw base::OptimisationError(base::concatenate(
                                {name(), ", zone ", std::to_string(zone), ": expected post= values of ",
                                 std::to_string(count), " rows and the same number of outputs, found ",
                                 std::to_string(answer.values.size()), " values of ", std::to_string(answer.outputs),
                                 " outputs; make post= return one row per case with the outputs it declares"}));
                        finalOutputs = answer.outputs;
                        for (std::size_t q = 0; q < answer.values.size(); ++q)
                        {
                            finals.push_back(answer.values[q]);
                            const std::size_t row = first + q / finalOutputs;
                            ItemStatus status = statusOf(answer, q);
                            for (std::size_t p = 0; p < linearOutputs; ++p)
                                status = combinedStatus(status, combinedStatuses[row * linearOutputs + p]);
                            finalStatuses.push_back(status);
                        }
                        if (withPartials)
                        {
                            if (answer.partials.size() != 1 ||
                                answer.partials[0].size() != count * finalOutputs * linearOutputs)
                                throw base::OptimisationError(base::concatenate(
                                    {name(), ", zone ", std::to_string(zone),
                                     ": expected the derivatives of post= of shape (", std::to_string(count), ", ",
                                     std::to_string(finalOutputs), ", ", std::to_string(linearOutputs),
                                     "); make post= return derivatives of that shape"}));
                            postRates.insert(postRates.end(), answer.partials[0].begin(), answer.partials[0].end());
                        }
                    }
                }
                for (std::size_t c = 0; c < n; ++c)
                {
                    CaseOutputs& entry = byCase[cases[c]];
                    entry.partials.resize(finalOutputs);
                    for (std::size_t o = 0; o < finalOutputs; ++o)
                    {
                        entry.values.push_back(finals[c * finalOutputs + o]);
                        ItemStatus status = finalStatuses[c * finalOutputs + o];
                        if (!std::isfinite(entry.values.back()) && status == ItemStatus::Ok)
                            status = ItemStatus::Failed;
                        entry.statuses.push_back(status);
                        if (!withPartials || status != ItemStatus::Ok)
                            continue;
                        // ∂h_o/∂Y_q: post's rates, or the identity.
                        const auto rate = [&](std::size_t q)
                        {
                            if (!m_definition.hasPost)
                                return q == o ? 1.0 : 0.0;
                            return postRates[(c * finalOutputs + o) * linearOutputs + q];
                        };
                        for (std::size_t a = 0; a < argumentCount; ++a)
                        {
                            const ExternalArgument& argument = m_definition.arguments[a];
                            if (isZoneData(argument.kind))
                            {
                                const std::size_t components = componentsOf(argument.kind, table);
                                for (std::size_t r = 0; r < components; ++r)
                                {
                                    double weight = 0.0;
                                    for (std::size_t q = 0; q < linearOutputs; ++q)
                                        weight += rate(q) * operators[a][q * components + r];
                                    if (weight != 0.0)
                                        entry.partials[o].push_back({a, r, weight});
                                }
                                continue;
                            }
                            double weight = 0.0;
                            for (std::size_t q = 0; q < linearOutputs; ++q)
                            {
                                double sum = 0.0;
                                for (std::size_t j = 0; j < k; ++j)
                                    sum += basis.coefficients(j, columns[c]) * variableRates[a][j * linearOutputs + q];
                                weight += rate(q) * sum;
                            }
                            if (weight != 0.0)
                                entry.partials[o].push_back({a, 0, weight});
                        }
                    }
                }
                if (outputs != 0 && finalOutputs != outputs)
                    throw base::OptimisationError(
                        base::concatenate({name(), ", zone ", std::to_string(zone), ": expected ",
                                           std::to_string(outputs), " outputs, found ", std::to_string(finalOutputs),
                                           "; make every zone of the response give the same outputs"}));
                outputs = finalOutputs;
            }
        }

        // The items of the zone: per subcase and output; the subcases not evaluated are NaN and not_evaluated.
        for (const std::int32_t subcase : subcases)
        {
            const auto entry = byCase.find(subcase);
            for (std::size_t o = 0; o < outputs; ++o)
            {
                result.values.items.push_back({subcase, zone, static_cast<std::int32_t>(o + 1), 0});
                if (entry == byCase.end())
                {
                    result.values.values.push_back(NOT_A_VALUE);
                    result.values.statuses.push_back(ItemStatus::NotEvaluated);
                    hasStatus = true;
                    if (withPartials)
                        result.partials.emplace_back();
                    continue;
                }
                const ItemStatus status = entry->second.statuses[o];
                result.values.values.push_back(status == ItemStatus::Ok ? entry->second.values[o] : NOT_A_VALUE);
                result.values.statuses.push_back(status);
                hasStatus = hasStatus || status != ItemStatus::Ok;
                if (withPartials)
                    result.partials.push_back(status == ItemStatus::Ok ? entry->second.partials[o]
                                                                       : std::vector<ExternalTerm>{});
            }
        }
    }
    if (!hasStatus)
        result.values.statuses.clear();
    return result;
}

std::vector<ExternalArgumentBlock> ExternalResponse::arguments(std::span<const std::int32_t> subcases,
                                                               const ItemSelection& selection,
                                                               ResponseContext& context) const
{
    std::vector<ExternalArgumentBlock> blocks;
    for (const std::int32_t zone : m_definition.zones)
    {
        const output::ZoneTable& table = tableOf(zone, context);
        std::vector<std::int32_t> selected;
        const auto chosen = selection.find(zone);
        for (const std::int32_t subcase : subcases)
            if ((chosen == selection.end() ||
                 std::find(chosen->second.begin(), chosen->second.end(), subcase) != chosen->second.end()) &&
                std::find(table.cases.begin(), table.cases.end(), subcase) != table.cases.end())
                selected.push_back(subcase);
        std::vector<std::pair<std::int32_t, std::int32_t>> contourDofs;
        for (std::size_t r = 0; r < table.contourDofCount(); ++r)
            contourDofs.emplace_back(table.contourGrids[r], table.contourComponents[r]);
        std::vector<std::pair<std::int32_t, std::int32_t>> interiorDofs;
        for (std::size_t r = 0; r < table.interiorDofCount(); ++r)
            interiorDofs.emplace_back(table.interiorGrids[r], table.interiorComponents[r]);
        if (!m_definition.isLinear)
        {
            ExternalArgumentBlock block;
            block.zone = zone;
            block.kind = ExternalRows::Cases;
            block.cases = selected;
            block.rows = selected.size();
            std::vector<std::size_t> rows;
            for (const std::int32_t subcase : selected)
                rows.push_back(static_cast<std::size_t>(std::find(table.cases.begin(), table.cases.end(), subcase) -
                                                        table.cases.begin()));
            if (!selected.empty() && !table.hasValues)
                throw base::OptimisationError(base::concatenate(
                    {name(), ": expected the contour data of every case of zone ", std::to_string(zone),
                     " in the analysis, found none (they were not kept); "
                     "evaluate the response on an analysis of the design problem, which keeps "
                     "them for the external responses"}));
            block.arguments = gather(table, zone, ExternalRows::Cases, 0, rows, context);
            block.contourDofs = contourDofs;
            block.interiorDofs = interiorDofs;
            blocks.push_back(std::move(block));
            continue;
        }
        for (std::size_t s = 0; s < table.basis.size(); ++s)
        {
            const output::ZoneBasisSet& basis = table.basis[s];
            std::vector<std::int32_t> cases;
            std::vector<std::size_t> columns;
            for (const std::int32_t subcase : selected)
                if (const auto column = std::find(basis.cases.begin(), basis.cases.end(), subcase);
                    column != basis.cases.end())
                {
                    cases.push_back(subcase);
                    columns.push_back(static_cast<std::size_t>(column - basis.cases.begin()));
                }
            if (cases.empty())
                continue;
            ExternalArgumentBlock block;
            block.zone = zone;
            block.kind = ExternalRows::Basis;
            block.constraintSet = basis.constraintSet;
            block.cases = cases;
            block.rows = basis.basisSize();
            std::vector<std::size_t> rows(block.rows);
            for (std::size_t j = 0; j < block.rows; ++j)
                rows[j] = j;
            block.arguments = gather(table, zone, ExternalRows::Basis, s, rows, context);
            for (std::size_t j = 0; j < block.rows; ++j)
                for (const std::size_t column : columns)
                    block.coefficients.push_back(basis.coefficients(j, column));
            block.contourDofs = contourDofs;
            block.interiorDofs = interiorDofs;
            blocks.push_back(std::move(block));
        }
    }
    return blocks;
}

} // namespace oos::design
