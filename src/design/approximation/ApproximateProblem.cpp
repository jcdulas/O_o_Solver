#include "oos/design/approximation/ApproximateProblem.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.14159265358979323846;
// DESVAR: XLB and XUB default to ∓1.0E+20, "no bound"; a bound at or beyond this magnitude constrains nothing.
constexpr double NO_BOUND = 1.0e20;

bool isBound(double value) noexcept
{
    return std::isfinite(value) && std::abs(value) < NO_BOUND;
}

std::string screeningTypeOf(const DesignResponses& responses, std::int32_t id)
{
    std::string type = responses.screeningType(id);
    return type == "DesignResp2" ? std::string("Equation")
                                 : type; // DesignConstrScreen screens the DesignResp2 constraints as EQUA
}

// The eigenvalue form of an EIGN, FREQ or LAMA item (EigenApproximation): under APRCOD = 1 the ATTB of its DesignResp1
// chooses (EIGN, FREQ: blank the Rayleigh quotient, 1 direct, 2 inverse; LAMA: blank or 1 direct, 2 inverse); under
// the mixed method (APRCOD = 2) EIGN and FREQ take the Rayleigh quotient and LAMA the direct form, the quotient being
// the default and usually the better form; the other kinds (convex, reciprocal, the hints) treat them as any response.
// A root of several modes has no Rayleigh quotient (its derivative is that of a
// subspace): it takes the direct form. Nothing for another response.
std::optional<EigenApproximation> eigenFormOf(const DesignProblem& problem, std::int32_t responseId, std::size_t item,
                                              const DesignGradients& gradients, ApproximationKind kind)
{
    const DesignResp1Response* response = problem.responses().findDesignResp1(responseId);
    if (response == nullptr || !isEigenType(response->definition().type))
        return std::nullopt;
    if (kind != ApproximationKind::Direct && kind != ApproximationKind::Mixed)
        return std::nullopt;
    const ResponseType type = response->definition().type;
    const std::optional<double> code = response->definition().attributeB;
    EigenApproximation form;
    form.isFrequency = type == ResponseType::Frequency;
    // DesignResp1 ATTB of an eigenvalue response: 1 direct linearisation, 2 inverse (the reciprocal variables).
    if (kind == ApproximationKind::Direct && code && *code == 2.0)
    {
        form.kind = ApproximationKind::Reciprocal;
        return form;
    }
    if (type == ResponseType::BucklingFactor || (kind == ApproximationKind::Direct && code && *code == 1.0))
    {
        form.kind = ApproximationKind::Direct;
        return form;
    }
    for (const RayleighGradient& part : gradients.rayleigh)
        if (part.responseId == responseId && part.item == item)
        {
            form.usesRayleighQuotient = true;
            form.root = part.root;
            form.numerator = part.numerator;
            form.denominator = part.denominator;
            return form;
        }
    form.kind = ApproximationKind::Direct;
    return form;
}

// λ̃ and the denominator 1 + ∇T·(x − x0) of the Rayleigh quotient approximation (EigenApproximation).
std::pair<double, double> rayleighQuotient(const EigenApproximation& form, std::span<const double> x0,
                                           std::span<const double> x)
{
    double numerator = form.root;
    double denominator = 1.0;
    for (std::size_t i = 0; i < x0.size(); ++i)
    {
        numerator += form.numerator[i] * (x[i] - x0[i]);
        denominator += form.denominator[i] * (x[i] - x0[i]);
    }
    return {numerator / denominator, denominator};
}

// f = √|λ| / 2π, signed as λ (a negative root, an approximation gone below zero).
double frequencyOf(double root)
{
    const double magnitude = std::sqrt(std::abs(root)) / (2.0 * PI);
    return root < 0.0 ? -magnitude : magnitude;
}

std::vector<double> denseGradient(const std::vector<std::pair<std::size_t, double>>& sparse, std::size_t n)
{
    std::vector<double> dense(n, 0.0);
    for (const auto& [column, derivative] : sparse)
        dense[column] += derivative;
    return dense;
}

} // namespace

ApproximateProblem::ApproximateProblem(ApproximationInputs inputs, ApproximationSettings settings,
                                       base::MemoryBudget& budget)
    : m_inputs(std::move(inputs)), m_settings(settings), m_normalisation(settings.constraintScale)
{
    const std::size_t n = m_inputs.variables.size();
    if (n == 0)
        throw base::OptimisationError(
            "approximate problem: expected at least one design variable, found none; declare the variables of the "
            "problem (DesignVar, DesignProblem.add_variable)");
    const auto checkGradient = [n](const std::vector<double>& gradient, const std::string& what)
    {
        if (gradient.size() != n)
            throw base::OptimisationError(base::concatenate(
                {"approximate problem: expected a gradient of ", std::to_string(n), " values for ", what, ", found ",
                 std::to_string(gradient.size()), "; give one value per independent design variable"}));
        for (const double value : gradient)
            if (!std::isfinite(value))
                throw base::OptimisationError(
                    base::concatenate({"approximate problem: expected a finite gradient for ", what,
                                       ", found NaN or infinity; compute the gradients of every retained item "
                                       "(RetainedResponses::gradientOptions)"}));
    };
    const MoveLimits limits(settings.moveLimits);
    m_x0.resize(n);
    m_lower.resize(n);
    m_upper.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const ApproximationVariable& variable = m_inputs.variables[i];
        const Range range = limits.variableRange(variable.value, variable.lower, variable.upper, variable.moveLimit);
        m_x0[i] = variable.value;
        m_lower[i] = range.lower;
        m_upper[i] = range.upper;
    }

    // Designed properties: the limits of a property one variable drives linearly cut that variable's bounds; the
    // others are constraints on the linearised property.
    for (std::size_t f = 0; f < m_inputs.properties.size(); ++f)
    {
        const ApproximationProperty& property = m_inputs.properties[f];
        checkGradient(property.gradient, base::concatenate({"property ", property.label}));
        const Range range = limits.propertyRange(property.value, property.lower, property.upper);
        std::size_t driving = 0;
        std::size_t drivers = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (property.gradient[i] != 0.0)
            {
                driving = i;
                ++drivers;
            }
        if (drivers == 0)
            continue;
        if (drivers == 1 && property.isLinear)
        {
            const double slope = property.gradient[driving];
            const double fromLower = m_x0[driving] + (range.lower - property.value) / slope;
            const double fromUpper = m_x0[driving] + (range.upper - property.value) / slope;
            m_lower[driving] = std::max(m_lower[driving], slope > 0.0 ? fromLower : fromUpper);
            m_upper[driving] = std::min(m_upper[driving], slope > 0.0 ? fromUpper : fromLower);
            continue;
        }
        if (isBound(range.lower))
            m_constraints.push_back({ApproximateConstraintKind::Property, f, BoundSide::Lower, range.lower});
        if (isBound(range.upper))
            m_constraints.push_back({ApproximateConstraintKind::Property, f, BoundSide::Upper, range.upper});
    }
    for (std::size_t i = 0; i < n; ++i)
        if (!(m_lower[i] <= m_upper[i]))
            throw base::OptimisationError(
                base::concatenate({"approximate problem: the bounds of design variable ", m_inputs.variables[i].label,
                                   " cut by the move limits and the property limits are empty ([",
                                   std::to_string(m_lower[i]), ", ", std::to_string(m_upper[i]),
                                   "]); check its DesignVar.lowerBound, DesignVar.upperBound and the "
                                   "DesignPropRelation1.minValue, DesignPropRelation1.maxValue of the "
                                   "properties it drives"}));

    // Response constraints first (the order of the listing: the retained responses, then the dependent variables).
    std::vector<ApproximateConstraintInfo> constraints;
    for (std::size_t c = 0; c < m_inputs.constraints.size(); ++c)
    {
        const ApproximationConstraint& constraint = m_inputs.constraints[c];
        if (constraint.item >= m_inputs.items.size())
            throw base::OptimisationError(base::concatenate(
                {"approximate problem: constraint ", std::to_string(c), " names item ", std::to_string(constraint.item),
                 ", expected one of the ", std::to_string(m_inputs.items.size()),
                 " items; give the index of an item of the problem"}));
        constraints.push_back({ApproximateConstraintKind::Response, c, constraint.side, constraint.bound});
    }
    for (std::size_t d = 0; d < m_inputs.linkedVariables.size(); ++d)
    {
        const ApproximationLinkedVariable& linked = m_inputs.linkedVariables[d];
        checkGradient(linked.gradient, base::concatenate({"dependent variable ", linked.label}));
        const Range range = limits.variableRange(linked.value, linked.lower, linked.upper, linked.moveLimit);
        if (isBound(range.lower))
            constraints.push_back({ApproximateConstraintKind::LinkedVariable, d, BoundSide::Lower, range.lower});
        if (isBound(range.upper))
            constraints.push_back({ApproximateConstraintKind::LinkedVariable, d, BoundSide::Upper, range.upper});
    }
    constraints.insert(constraints.end(), m_constraints.begin(), m_constraints.end());
    m_constraints = std::move(constraints);

    m_approximation = ResponseApproximation(m_x0, m_lower);
    m_itemGradients = base::BudgetedBuffer<double>(budget, m_inputs.items.size() * n, "approximate problem gradients");
    for (std::size_t k = 0; k < m_inputs.items.size(); ++k)
    {
        ApproximationItem& item = m_inputs.items[k];
        checkGradient(item.gradient, base::concatenate({"response ", std::to_string(item.responseId), " item ",
                                                        std::to_string(item.item)}));
        std::copy(item.gradient.begin(), item.gradient.end(), m_itemGradients.data() + k * n);
        item.gradient.clear(); // kept once, in the budgeted buffer
    }
    if (m_inputs.objective.isPresent)
        checkGradient(m_inputs.objective.gradient, "the objective");
}

ApproximateProblem ApproximateProblem::fromDesign(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                  const DesignGradients& gradients, const RetainedResponses& retained,
                                                  const ApproximationSettings& settings, base::MemoryBudget& budget,
                                                  const ApproximationHintsResult* hints)
{
    if (settings.kind == ApproximationKind::Hints && hints == nullptr)
        throw base::OptimisationError(
            "approximate problem: the hints approximation needs the approximation hints of the design point "
            "(ApproximationHints::compute); compute them first");
    const auto powersOf = [&](std::int32_t responseId) -> std::vector<int>
    {
        if (settings.kind != ApproximationKind::Hints)
            return {};
        const ResponseHint* hint = hints->find(responseId);
        return hint != nullptr ? hint->exponents : std::vector<int>{};
    };
    const DesignModel& model = problem.model();
    const DesignValues& values = evaluation.analysis.point.values();
    const std::vector<double>& x0 = evaluation.analysis.point.x();
    const std::size_t n = x0.size();
    ApproximationInputs inputs;
    const std::vector<std::size_t>& independent = model.independentIndices();
    std::vector<bool> isIndependent(model.variables().size(), false);
    for (std::size_t k = 0; k < independent.size(); ++k)
    {
        const DesignVariable& variable = model.variables()[independent[k]];
        isIndependent[independent[k]] = true;
        inputs.variables.push_back({variable.label, x0[k], variable.lower, variable.upper, variable.moveLimit});
    }
    for (std::size_t v = 0; v < model.variables().size(); ++v)
    {
        if (isIndependent[v])
            continue;
        const DesignVariable& variable = model.variables()[v];
        inputs.linkedVariables.push_back({variable.label, values.variables.values[v], variable.lower, variable.upper,
                                          variable.moveLimit, denseGradient(values.variables.derivatives[v], n)});
    }
    for (std::size_t f = 0; f < model.relations().size(); ++f)
    {
        const DesignRelation& relation = model.relations()[f];
        inputs.properties.push_back({relation.label(), values.fields[f].value,
                                     denseGradient(values.fields[f].gradient, n), relation.lowerBound,
                                     relation.upperBound, relation.form == RelationForm::Linear});
    }

    std::map<std::int32_t, const ResponseValues*> valuesOf;
    for (const ResponseValues& response : evaluation.responses)
        valuesOf[response.id] = &response;
    std::map<std::int32_t, const ResponseGradient*> gradientOf;
    for (const ResponseGradient& gradient : gradients.responses)
        gradientOf[gradient.id] = &gradient;
    std::map<std::pair<std::int32_t, std::size_t>, std::size_t> itemIndex;
    for (const std::size_t index : retained.retainedConstraints())
    {
        const ConstraintValue& constraint = evaluation.constraints[index];
        const auto key = std::make_pair(constraint.responseId, constraint.item);
        auto found = itemIndex.find(key);
        if (found == itemIndex.end())
        {
            const auto gradient = gradientOf.find(constraint.responseId);
            if (gradient == gradientOf.end())
                throw base::OptimisationError(base::concatenate(
                    {"approximate problem: the retained constraint on response ", std::to_string(constraint.responseId),
                     " has no gradient; compute the gradients with the retained responses' options"}));
            const std::vector<double>& rows = gradient->second->values;
            ApproximationItem item;
            item.responseId = constraint.responseId;
            item.item = constraint.item;
            item.label = constraint.label;
            item.responseType = screeningTypeOf(problem.responses(), constraint.responseId);
            item.value = valuesOf.at(constraint.responseId)->values[constraint.item];
            item.gradient.assign(rows.begin() + static_cast<std::ptrdiff_t>(constraint.item * n),
                                 rows.begin() + static_cast<std::ptrdiff_t>((constraint.item + 1) * n));
            item.powers = powersOf(constraint.responseId);
            item.eigen = eigenFormOf(problem, constraint.responseId, constraint.item, gradients, settings.kind);
            found = itemIndex.emplace(key, inputs.items.size()).first;
            inputs.items.push_back(std::move(item));
        }
        inputs.constraints.push_back({found->second, constraint.side, constraint.bound, constraint.setId, index});
    }
    if (const std::optional<std::int32_t> objective = problem.constraints().objective();
        objective && evaluation.objective)
    {
        if (!gradients.objective)
            throw base::OptimisationError(
                base::concatenate({"approximate problem: the objective (response ", std::to_string(*objective),
                                   ") has no gradient; compute the gradients with the retained responses' options"}));
        inputs.objective = {true,
                            *objective,
                            screeningTypeOf(problem.responses(), *objective),
                            problem.constraints().isMaximised(),
                            *evaluation.objective,
                            *gradients.objective,
                            powersOf(*objective),
                            eigenFormOf(problem, *objective, 0, gradients, settings.kind)};
    }
    return {std::move(inputs), settings, budget};
}

void ApproximateProblem::checkPoint(std::span<const double> x) const
{
    if (x.size() != m_x0.size())
        throw base::OptimisationError(
            base::concatenate({"approximate problem: expected x of ", std::to_string(m_x0.size()),
                               " values (the independent design variables), found ", std::to_string(x.size()),
                               "; give one value per independent design variable"}));
}

ApproximationKind ApproximateProblem::itemKind(const ApproximationItem& item) const noexcept
{
    if (item.eigen && item.eigen->kind)
        return *item.eigen->kind;
    if (m_settings.kind != ApproximationKind::Mixed)
        return m_settings.kind;
    return isReciprocalType(item.responseType) ? ApproximationKind::Reciprocal : ApproximationKind::Direct;
}

double ApproximateProblem::approximatedValue(const ApproximationItem& item, double value0,
                                             std::span<const double> gradient0, std::span<const double> x) const
{
    if (!item.eigen || !item.eigen->usesRayleighQuotient)
        return m_approximation.value(itemKind(item), value0, gradient0, x, item.powers);
    const double root = rayleighQuotient(*item.eigen, m_x0, x).first;
    return item.eigen->isFrequency ? frequencyOf(root) : root;
}

void ApproximateProblem::approximatedGradient(const ApproximationItem& item, std::span<const double> gradient0,
                                              std::span<const double> x, std::span<double> gradient) const
{
    if (!item.eigen || !item.eigen->usesRayleighQuotient)
    {
        m_approximation.gradient(itemKind(item), gradient0, x, gradient, item.powers);
        return;
    }
    // dλ̃/dx = (∇U − λ̃·∇T) / (1 + ∇T·(x − x0)); df̃/dλ̃ = 1 / (4π·√|λ̃|).
    const auto [root, denominator] = rayleighQuotient(*item.eigen, m_x0, x);
    double factor = 1.0 / denominator;
    if (item.eigen->isFrequency)
        factor *= root == 0.0 ? 0.0 : 1.0 / (4.0 * PI * std::sqrt(std::abs(root)));
    for (std::size_t i = 0; i < m_x0.size(); ++i)
        gradient[i] = factor * (item.eigen->numerator[i] - root * item.eigen->denominator[i]);
}

std::span<const double> ApproximateProblem::itemGradient(std::size_t item) const noexcept
{
    const std::size_t n = m_x0.size();
    return {m_itemGradients.data() + item * n, n};
}

double ApproximateProblem::linearValue(double value0, std::span<const double> gradient0,
                                       std::span<const double> x) const noexcept
{
    double value = value0;
    for (std::size_t i = 0; i < m_x0.size(); ++i)
        value += gradient0[i] * (x[i] - m_x0[i]);
    return value;
}

double ApproximateProblem::constraintValue(std::size_t index, std::span<const double> x) const
{
    const ApproximateConstraintInfo& info = m_constraints[index];
    switch (info.kind)
    {
    case ApproximateConstraintKind::LinkedVariable:
    {
        const ApproximationLinkedVariable& linked = m_inputs.linkedVariables[info.source];
        return m_normalisation.normalise(info.side, info.bound, linearValue(linked.value, linked.gradient, x));
    }
    case ApproximateConstraintKind::Property:
    {
        const ApproximationProperty& property = m_inputs.properties[info.source];
        return m_normalisation.normalise(info.side, info.bound, linearValue(property.value, property.gradient, x));
    }
    case ApproximateConstraintKind::Response:
        break;
    }
    const ApproximationConstraint& constraint = m_inputs.constraints[info.source];
    const ApproximationItem& item = m_inputs.items[constraint.item];
    const ApproximationKind kind = itemKind(item);
    const std::span<const double> gradient = itemGradient(constraint.item);
    if (kind != ApproximationKind::Convex)
        return m_normalisation.normalise(constraint.side, constraint.bound,
                                         approximatedValue(item, item.value, gradient, x));
    // The convex linearisation of g itself: 1/x where ∂g/∂x_i < 0.
    const double slope = m_normalisation.derivative(constraint.side, constraint.bound);
    std::vector<double> constraintGradient(gradient.begin(), gradient.end());
    for (double& value : constraintGradient)
        value *= slope;
    return m_approximation.value(kind, m_normalisation.normalise(constraint.side, constraint.bound, item.value),
                                 constraintGradient, x);
}

void ApproximateProblem::constraintGradient(std::size_t index, std::span<const double> x,
                                            std::span<double> gradient) const
{
    const ApproximateConstraintInfo& info = m_constraints[index];
    if (info.kind != ApproximateConstraintKind::Response)
    {
        const std::vector<double>& linear = info.kind == ApproximateConstraintKind::LinkedVariable
                                                ? m_inputs.linkedVariables[info.source].gradient
                                                : m_inputs.properties[info.source].gradient;
        const double slope = m_normalisation.derivative(info.side, info.bound);
        for (std::size_t i = 0; i < m_x0.size(); ++i)
            gradient[i] = slope * linear[i];
        return;
    }
    const ApproximationConstraint& constraint = m_inputs.constraints[info.source];
    const ApproximationItem& item = m_inputs.items[constraint.item];
    const ApproximationKind kind = itemKind(item);
    const double slope = m_normalisation.derivative(constraint.side, constraint.bound);
    std::vector<double> constraintGradient(itemGradient(constraint.item).begin(), itemGradient(constraint.item).end());
    for (double& value : constraintGradient)
        value *= slope;
    // g is affine in r: its approximation in the variables of r is that of r times dg/dr, except the convex one,
    // which chooses the variables by the sign of ∂g/∂x_i.
    if (item.eigen && item.eigen->usesRayleighQuotient)
    {
        approximatedGradient(item, itemGradient(constraint.item), x, gradient);
        for (double& value : gradient)
            value *= slope;
        return;
    }
    m_approximation.gradient(kind, constraintGradient, x, gradient, item.powers);
}

double ApproximateProblem::objective(std::span<const double> x) const
{
    checkPoint(x);
    const ApproximationObjective& objective = m_inputs.objective;
    if (!objective.isPresent)
        return 0.0;
    ApproximationItem asItem;
    asItem.responseType = objective.responseType;
    asItem.powers = objective.powers;
    asItem.eigen = objective.eigen;
    const ApproximationKind kind = itemKind(asItem);
    if (kind != ApproximationKind::Convex || !objective.isMaximised)
        return approximatedValue(asItem, objective.value, objective.gradient, x);
    // A maximised objective: the convex linearisation of −f (the minimised function).
    std::vector<double> negated(objective.gradient);
    for (double& value : negated)
        value = -value;
    return -m_approximation.value(kind, -objective.value, negated, x);
}

void ApproximateProblem::objectiveGradient(std::span<const double> x, std::span<double> gradient) const
{
    checkPoint(x);
    const ApproximationObjective& objective = m_inputs.objective;
    if (!objective.isPresent)
    {
        std::fill(gradient.begin(), gradient.end(), 0.0);
        return;
    }
    ApproximationItem asItem;
    asItem.responseType = objective.responseType;
    asItem.powers = objective.powers;
    asItem.eigen = objective.eigen;
    if (asItem.eigen && asItem.eigen->usesRayleighQuotient)
    {
        approximatedGradient(asItem, objective.gradient, x, gradient);
        return;
    }
    const ApproximationKind kind = itemKind(asItem);
    std::vector<double> sensed(objective.gradient);
    if (objective.isMaximised)
        for (double& value : sensed)
            value = -value;
    m_approximation.gradient(kind, sensed, x, gradient, objective.powers);
    if (objective.isMaximised)
        for (double& value : gradient)
            value = -value;
}

std::vector<double> ApproximateProblem::itemValues(std::span<const double> x) const
{
    checkPoint(x);
    std::vector<double> values(m_inputs.items.size());
    for (std::size_t k = 0; k < m_inputs.items.size(); ++k)
        values[k] = approximatedValue(m_inputs.items[k], m_inputs.items[k].value, itemGradient(k), x);
    return values;
}

std::vector<double> ApproximateProblem::itemGradients(std::span<const double> x) const
{
    checkPoint(x);
    const std::size_t n = m_x0.size();
    std::vector<double> gradients(m_inputs.items.size() * n);
    for (std::size_t k = 0; k < m_inputs.items.size(); ++k)
        approximatedGradient(m_inputs.items[k], itemGradient(k), x, std::span<double>(gradients.data() + k * n, n));
    return gradients;
}

std::vector<double> ApproximateProblem::constraintValues(std::span<const double> x, bool isNormalised) const
{
    checkPoint(x);
    std::vector<double> values(m_constraints.size());
    for (std::size_t c = 0; c < m_constraints.size(); ++c)
    {
        const double g = constraintValue(c, x);
        values[c] = isNormalised ? g : m_normalisation.value(m_constraints[c].side, m_constraints[c].bound, g);
    }
    return values;
}

double ApproximateProblem::maxViolation(std::span<const double> x) const
{
    double largest = -INF;
    for (const double g : constraintValues(x, true))
        largest = std::max(largest, g);
    return largest;
}

double ApproximateProblem::evaluate(std::span<const double> x, std::span<double> constraints) const
{
    checkPoint(x);
    for (std::size_t c = 0; c < m_constraints.size(); ++c)
        constraints[c] = constraintValue(c, x);
    const double value = objective(x);
    return m_inputs.objective.isMaximised ? -value : value;
}

void ApproximateProblem::gradients(std::span<const double> x, std::span<double> objectiveGradient,
                                   std::span<const std::size_t> rows, std::span<double> constraintGradients) const
{
    checkPoint(x);
    this->objectiveGradient(x, objectiveGradient);
    if (m_inputs.objective.isMaximised)
        for (double& value : objectiveGradient)
            value = -value;
    const std::size_t n = m_x0.size();
    for (std::size_t k = 0; k < rows.size(); ++k)
        constraintGradient(rows[k], x, constraintGradients.subspan(k * n, n));
}

} // namespace oos::design
