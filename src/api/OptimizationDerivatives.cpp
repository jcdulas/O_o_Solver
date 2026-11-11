#include "OptimizationDerivatives.hpp"

#include "oos/design/constraints/ConstraintNormalisation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace oos::api::detail
{
namespace
{
constexpr double HESSIAN_GRADIENT_STEP_RELATIVE = 1e-4;
constexpr double TRIM_SHAPE_STEP_RELATIVE = 1e-5;

design::ModeReferences modeReferences(const design::DesignEvaluation& evaluation)
{
    design::ModeReferences references;
    for (const auto& mode : evaluation.trackedModes)
        if (!mode.shape.empty())
            references[{mode.responseId, mode.subcase}] = mode.shape;
    return references;
}

// A weighted response gradient is the gradient of the scalar function whose Hessian is requested.
// The response IDs and item counts must remain fixed; mode references preserve modal identity.
std::vector<double> weightedGradient(const design::ProductWeights& weights,
                                     const design::DesignEvaluation& evaluation,
                                     const design::DesignGradients& gradients)
{
    const std::size_t n = gradients.variableCount;
    std::vector<double> result(n, 0.0);
    std::set<std::int32_t> found;
    for (const auto& response : gradients.responses)
    {
        const auto original = std::find_if(evaluation.responses.begin(), evaluation.responses.end(),
                                           [&](const auto& item) { return item.id == response.id; });
        if (original == evaluation.responses.end() || response.itemCount != original->itemCount() ||
            response.values.size() != response.itemCount * n)
            throw std::runtime_error("response items changed during Hessian evaluation");
        const auto dense = weights.weights.find(response.id);
        const auto sparse = weights.sparseWeights.find(response.id);
        if (dense == weights.weights.end() && sparse == weights.sparseWeights.end())
            continue;
        if (dense != weights.weights.end() && sparse != weights.sparseWeights.end())
            throw std::invalid_argument("a response cannot have both dense and sparse Hessian weights");
        found.insert(response.id);
        const auto addItem = [&](std::size_t item, double weight)
        {
            if (item >= response.itemCount)
                throw std::invalid_argument("Hessian weight item is out of range");
            if (weight == 0.0)
                return;
            for (std::size_t column = 0; column < n; ++column)
                result[column] += weight * response.values[item * n + column];
        };
        if (dense != weights.weights.end())
        {
            if (dense->second.size() != response.itemCount)
                throw std::invalid_argument("Hessian weights do not match the response item count");
            for (std::size_t item = 0; item < response.itemCount; ++item)
                addItem(item, dense->second[item]);
        }
        else
        {
            const auto& row = sparse->second;
            if (row.rowStart.size() != 2 || row.rowStart[0] != 0 || row.rowStart[1] != row.items.size() ||
                row.values.size() != row.items.size())
                throw std::invalid_argument("Hessian sparse weights require one valid row");
            for (std::size_t entry = 0; entry < row.items.size(); ++entry)
                addItem(row.items[entry], row.values[entry]);
        }
    }
    for (const auto& [id, row] : weights.weights)
        if (!found.count(id) && std::any_of(row.begin(), row.end(), [](double value) { return value != 0.0; }))
            throw std::invalid_argument("a weighted response has no native gradient");
    for (const auto& [id, row] : weights.sparseWeights)
        if (!found.count(id) && std::any_of(row.values.begin(), row.values.end(),
                                            [](double value) { return value != 0.0; }))
            throw std::invalid_argument("a weighted response has no native gradient");
    return result;
}

double feasibleStep(std::span<const double> x, std::span<const double> direction,
                    const design::DesignModel& model, int side)
{
    double maximum = std::numeric_limits<double>::infinity();
    for (std::size_t column = 0; column < x.size(); ++column)
    {
        const auto& variable = model.variables()[model.independentIndices()[column]];
        const double rate = side * direction[column];
        if (rate > 0.0)
            maximum = std::min(maximum, (variable.upper - x[column]) / rate);
        else if (rate < 0.0)
            maximum = std::min(maximum, (variable.lower - x[column]) / rate);
    }
    return std::max(0.0, maximum);
}

} // namespace

design::ProductReport nativeGradientHessian(OptimizationEvaluation& point, const design::ProductWeights& weights,
                                            std::span<const double> directions, std::size_t count,
                                            std::vector<double>& values)
{
    if (weights.rows != 1 || count == 0 || directions.size() != point.x().size() * count)
        throw std::invalid_argument(
            "Hessian-vector products require one weight row and one value per variable and direction");
    if (!point.evaluation().trackingWarnings.empty())
        throw std::invalid_argument("Hessian-vector products require unambiguous mode tracking");

    const auto start = std::chrono::steady_clock::now();
    const auto& x = point.x();
    const auto& model = point.owner().problem().model();
    const auto references = modeReferences(point.evaluation());
    design::GradientOptions options;
    for (const auto& [id, row] : weights.weights)
        options.responseIds.push_back(id);
    for (const auto& [id, row] : weights.sparseWeights)
        options.responseIds.push_back(id);
    std::sort(options.responseIds.begin(), options.responseIds.end());
    options.responseIds.erase(std::unique(options.responseIds.begin(), options.responseIds.end()),
                              options.responseIds.end());
    for (const auto id : options.responseIds)
    {
        if (weights.weights.count(id) && weights.sparseWeights.count(id))
            throw std::invalid_argument("a response cannot have both dense and sparse Hessian weights");
        const auto response = std::find_if(point.evaluation().responses.begin(), point.evaluation().responses.end(),
                                           [&](const auto& item) { return item.id == id; });
        if (response == point.evaluation().responses.end())
            throw std::invalid_argument("Hessian weights refer to an unknown response");
        design::ItemMask mask;
        mask.responseId = id;
        mask.isRetained.assign(response->itemCount(), 0);
        if (const auto dense = weights.weights.find(id); dense != weights.weights.end())
        {
            if (dense->second.size() != mask.isRetained.size())
                throw std::invalid_argument("Hessian weights do not match the response item count");
            for (std::size_t item = 0; item < mask.isRetained.size(); ++item)
                mask.isRetained[item] = dense->second[item] != 0.0;
        }
        else
        {
            const auto& sparse = weights.sparseWeights.at(id);
            if (sparse.rowStart.size() != 2 || sparse.rowStart[0] != 0 ||
                sparse.rowStart[1] != sparse.items.size() || sparse.values.size() != sparse.items.size())
                throw std::invalid_argument("Hessian sparse weights require one valid row");
            for (const auto item : sparse.items)
            {
                if (item >= mask.isRetained.size())
                    throw std::invalid_argument("Hessian weight item is out of range");
                mask.isRetained[item] = 1;
            }
        }
        options.itemMasks.push_back(std::move(mask));
    }
    const std::size_t n = x.size();
    values.assign(n * count, 0.0);
    design::ProductReport report;
    report.hessianBlocksReason = "differences of native first-order gradients";
    std::vector<double> baseGradient;
    const auto gradientAt = [&](std::span<const double> moved)
    {
        auto evaluation = point.evaluateAt(std::vector<double>(moved.begin(), moved.end()), references);
        if (!evaluation->evaluation().trackingWarnings.empty())
            throw std::invalid_argument(
                "Hessian-vector products require unambiguous mode tracking at perturbed points");
        const auto gradients = evaluation->gradients(options);
        report.solves += gradients.report.solves;
        report.factorisations += gradients.report.factorisations;
        ++report.perturbedLinearisations;
        return weightedGradient(weights, evaluation->evaluation(), gradients);
    };
    for (std::size_t column = 0; column < count; ++column)
    {
        std::vector<double> direction(n);
        double desired = std::numeric_limits<double>::infinity();
        for (std::size_t variable = 0; variable < n; ++variable)
        {
            direction[variable] = directions[variable * count + column];
            if (direction[variable] != 0.0)
                desired = std::min(desired, HESSIAN_GRADIENT_STEP_RELATIVE *
                                               std::max(1.0, std::abs(x[variable])) /
                                               std::abs(direction[variable]));
        }
        if (!std::isfinite(desired))
            continue;
        const double positive = feasibleStep(x, direction, model, +1);
        const double negative = feasibleStep(x, direction, model, -1);
        const bool central = positive > 0.0 && negative > 0.0;
        const int side = positive > 0.0 ? +1 : -1;
        const double step = std::min(desired, central ? std::min(positive, negative) * 0.5
                                                      : (side > 0 ? positive : negative) * 0.5);
        if (!(step > 0.0))
            throw std::invalid_argument("Hessian direction has no feasible perturbation within variable bounds");
        std::vector<double> moved = x;
        for (std::size_t variable = 0; variable < n; ++variable)
            moved[variable] += side * step * direction[variable];
        const auto first = gradientAt(moved);
        moved = x;
        if (central)
            for (std::size_t variable = 0; variable < n; ++variable)
                moved[variable] -= step * direction[variable];
        else if (baseGradient.empty())
        {
            const auto gradients = point.gradients(options);
            report.solves += gradients.report.solves;
            report.factorisations += gradients.report.factorisations;
            baseGradient = weightedGradient(weights, point.evaluation(), gradients);
        }
        const auto second = central ? gradientAt(moved) : baseGradient;
        const double denominator = central ? 2.0 * step : side * step;
        for (std::size_t variable = 0; variable < n; ++variable)
            values[variable * count + column] = (first[variable] - second[variable]) / denominator;
    }
    report.productSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return report;
}

// A moved grid changes trim splines and the rigid-body support modes of inertia relief. The structural
// linearisation holds these fixed, so complete analyses provide the affected shape column.
FullShapeColumn fullShapeColumn(OptimizationEvaluation& point, std::size_t column)
{
    const auto& x = point.x();
    const auto& original = point.evaluation();
    const auto& model = point.owner().problem().model();
    const auto references = modeReferences(original);
    if (column >= x.size())
        throw std::invalid_argument("shape column is outside the independent variables");
    FullShapeColumn result;
    const auto responseValuesAt = [&](std::span<const double> moved)
    {
        auto evaluation = point.evaluateAt(std::vector<double>(moved.begin(), moved.end()), references);
        ++result.analysisPerturbations;
        if (!evaluation->evaluation().trackingWarnings.empty())
            throw std::invalid_argument("shape derivative requires unambiguous mode tracking");
        const auto& responses = evaluation->evaluation().responses;
        if (responses.size() != original.responses.size())
            throw std::runtime_error("response count changed during shape differentiation");
        std::vector<double> values;
        for (std::size_t index = 0; index < responses.size(); ++index)
        {
            if (responses[index].id != original.responses[index].id ||
                responses[index].items != original.responses[index].items)
                throw std::runtime_error("response items changed during shape differentiation");
            values.insert(values.end(), responses[index].values.begin(), responses[index].values.end());
        }
        return values;
    };
    std::vector<double> direction(x.size(), 0.0);
    direction[column] = 1.0;
    const double positive = feasibleStep(x, direction, model, +1);
    const double negative = feasibleStep(x, direction, model, -1);
    const bool central = positive > 0.0 && negative > 0.0;
    const int side = positive > 0.0 ? +1 : -1;
    const double desired = TRIM_SHAPE_STEP_RELATIVE * std::max(1.0, std::abs(x[column]));
    const double step = std::min(desired, (central ? std::min(positive, negative)
                                                  : (side > 0 ? positive : negative)) * 0.5);
    if (!(step > 0.0))
        throw std::invalid_argument("shape variable has no feasible perturbation within its bounds");
    std::vector<double> moved = x;
    moved[column] += side * step;
    const auto first = responseValuesAt(moved);
    moved[column] = x[column] - step;
    const auto second = central ? responseValuesAt(moved) : std::vector<double>{};
    result.values.resize(first.size());
    std::size_t row = 0;
    for (const auto& response : original.responses)
        for (const double base : response.values)
        {
            const double previous = central ? second[row] : base;
            result.values[row] = (first[row] - previous) / (central ? 2.0 * step : side * step);
            ++row;
        }
    return result;
}

void completeShapeGradients(OptimizationEvaluation& point, const design::GradientOptions& options,
                            design::DesignGradients& gradients, const std::vector<std::size_t>& shapeColumns)
{
    const auto& original = point.evaluation();
    const std::size_t n = point.x().size();
    std::map<std::int32_t, std::size_t> firstRow;
    std::size_t rows = 0;
    for (const auto& response : original.responses)
    {
        firstRow[response.id] = rows;
        rows += response.itemCount();
    }
    for (const std::size_t column : shapeColumns)
    {
        const FullShapeColumn shape = fullShapeColumn(point, column);
        gradients.report.analysisPerturbations += shape.analysisPerturbations;
        for (auto& gradient : gradients.responses)
        {
            for (std::size_t item = 0; item < gradient.itemCount; ++item)
            {
                if (!design::isItemDifferentiated(options, gradient.id, item))
                    continue;
                gradient.values[item * n + column] = shape.values[firstRow.at(gradient.id) + item];
            }
        }
    }
    const design::ConstraintNormalisation normalisation;
    for (std::size_t index = 0; index < original.constraints.size(); ++index)
    {
        const auto& constraint = original.constraints[index];
        auto& row = gradients.constraints[index];
        if (row.empty())
            continue;
        const auto response = std::find_if(gradients.responses.begin(), gradients.responses.end(),
                                           [&](const auto& entry) { return entry.id == constraint.responseId; });
        if (response == gradients.responses.end())
            continue;
        const double scale = normalisation.derivative(constraint.side, constraint.bound);
        for (const auto column : shapeColumns)
            row[column] = response->values[constraint.item * n + column] * scale;
    }
    if (gradients.objective)
    {
        const auto id = point.owner().problem().constraints().objective();
        if (id)
        {
            const auto response = std::find_if(gradients.responses.begin(), gradients.responses.end(),
                                               [&](const auto& entry) { return entry.id == *id; });
            if (response != gradients.responses.end() && response->itemCount == 1)
                for (const auto column : shapeColumns)
                    (*gradients.objective)[column] = response->values[column];
        }
    }
}
} // namespace oos::api::detail
