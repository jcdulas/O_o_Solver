#include "OptimizationProducts.hpp"
#include "OptimizationDerivatives.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace oos::api::detail
{
namespace
{
using Clock = std::chrono::steady_clock;

struct SelectedResponse
{
    const design::ResponseValues* response = nullptr;
    std::vector<std::uint8_t> items;
};

std::vector<SelectedResponse> selectedResponses(const design::DesignEvaluation& evaluation,
                                                const design::ProductSelection& selection)
{
    std::map<std::int32_t, const design::ResponseValues*> byId;
    for (const auto& response : evaluation.responses)
        byId.emplace(response.id, &response);
    std::set<std::int32_t> ids(selection.responseIds.begin(), selection.responseIds.end());
    std::map<std::int32_t, const design::ItemMask*> masks;
    for (const auto& mask : selection.itemMasks)
    {
        if (!masks.emplace(mask.responseId, &mask).second)
            throw std::invalid_argument("jvp: expected one item mask per response");
        if (!selection.responseIds.empty() && !ids.count(mask.responseId))
            throw std::invalid_argument("jvp: an item mask names a response outside the selection");
    }
    if (ids.empty())
    {
        if (!masks.empty())
            for (const auto& entry : masks)
                ids.insert(entry.first);
        else
            for (const auto& response : evaluation.responses)
                ids.insert(response.id);
    }
    for (const auto id : ids)
        if (!byId.count(id))
            throw std::invalid_argument("jvp: a selected response is not in the evaluation");
    const std::set<std::int32_t> subcases(selection.subcases.begin(), selection.subcases.end());
    for (const auto subcase : subcases)
        if (std::find(evaluation.subcases.begin(), evaluation.subcases.end(), subcase) ==
            evaluation.subcases.end())
            throw std::invalid_argument("jvp: a selected subcase was not analysed");

    std::vector<SelectedResponse> result;
    for (const auto& response : evaluation.responses)
    {
        if (!ids.count(response.id))
            continue;
        std::vector<std::uint8_t> flags(response.itemCount(), 1);
        if (const auto mask = masks.find(response.id); mask != masks.end())
        {
            if (mask->second->isRetained.size() != flags.size())
                throw std::invalid_argument("jvp: an item mask has the wrong item count");
            flags = mask->second->isRetained;
        }
        if (!subcases.empty())
            for (std::size_t item = 0; item < flags.size(); ++item)
                if (response.items[item].subcase != 0 && !subcases.count(response.items[item].subcase))
                    flags[item] = 0;
        result.push_back({&response, std::move(flags)});
    }
    return result;
}

std::map<std::int32_t, const design::ResponseGradient*> gradientsById(const design::DesignGradients& gradients)
{
    std::map<std::int32_t, const design::ResponseGradient*> result;
    for (const auto& response : gradients.responses)
        result.emplace(response.id, &response);
    return result;
}

design::ProductReport productReport(const design::DesignGradients& gradients, double lineariseSeconds)
{
    design::ProductReport report;
    report.solves = gradients.report.solves;
    report.factorisations = gradients.report.factorisations;
    report.lineariseSeconds = lineariseSeconds;
    report.lineariseItemSeconds = gradients.report.itemSeconds;
    report.lineariseFieldSeconds = gradients.report.fieldSeconds;
    return report;
}

double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void validateWeights(const design::DesignEvaluation& evaluation, const design::ProductWeights& weights)
{
    if (weights.rows == 0)
        throw std::invalid_argument("vjp: expected at least one row of weights");
    std::map<std::int32_t, std::size_t> counts;
    for (const auto& response : evaluation.responses)
        counts.emplace(response.id, response.itemCount());
    for (const auto& [id, dense] : weights.weights)
    {
        const auto found = counts.find(id);
        if (found == counts.end())
            throw std::invalid_argument("vjp: a weighted response is not in the evaluation");
        if (weights.sparseWeights.count(id))
            throw std::invalid_argument("vjp: a response has both dense and sparse weights");
        if (dense.size() != weights.rows * found->second)
            throw std::invalid_argument("vjp: dense weights have the wrong item count");
    }
    for (const auto& [id, sparse] : weights.sparseWeights)
    {
        const auto found = counts.find(id);
        if (found == counts.end())
            throw std::invalid_argument("vjp: a weighted response is not in the evaluation");
        if (sparse.rowStart.size() != weights.rows + 1 || sparse.rowStart.front() != 0 ||
            sparse.rowStart.back() != sparse.items.size() || sparse.values.size() != sparse.items.size() ||
            !std::is_sorted(sparse.rowStart.begin(), sparse.rowStart.end()))
            throw std::invalid_argument("vjp: sparse row pointers are invalid");
        for (const auto item : sparse.items)
            if (item >= found->second)
                throw std::invalid_argument("vjp: a sparse weight item is out of range");
    }
}

} // namespace

design::ProductReport fullShapeJvp(OptimizationEvaluation& point, std::span<const double> directions,
                                   std::size_t count, std::size_t block, design::ProductSink& sink,
                                   const design::ProductSelection& selection)
{
    const std::size_t n = point.x().size();
    if (count == 0 || directions.size() != n * count || block == 0)
        throw std::invalid_argument("jvp: expected nonempty directions of shape (variables, count) and block > 0");
    const auto selected = selectedResponses(point.evaluation(), selection);
    design::GradientOptions options;
    options.isVariableWanted.assign(n, 0);
    std::size_t rows = 0;
    for (const auto& response : selected)
    {
        options.responseIds.push_back(response.response->id);
        options.itemMasks.push_back({response.response->id, response.items});
        rows += response.items.size();
    }
    for (std::size_t column = 0; column < n; ++column)
        for (std::size_t direction = 0; direction < count; ++direction)
            if (directions[column * count + direction] != 0.0)
                options.isVariableWanted[column] = 1;
    const auto started = Clock::now();
    const design::DesignGradients gradients = point.gradients(options);
    design::ProductReport report = productReport(gradients, secondsSince(started));
    const auto byId = gradientsById(gradients);
    const auto productsStarted = Clock::now();
    for (std::size_t first = 0; first < count; first += block)
    {
        const std::size_t columns = std::min(block, count - first);
        std::vector<double> values(rows * columns, 0.0);
        std::size_t row = 0;
        for (const auto& response : selected)
        {
            const auto& gradient = *byId.at(response.response->id);
            for (std::size_t item = 0; item < response.items.size(); ++item, ++row)
            {
                if (!response.items[item])
                {
                    std::fill_n(values.data() + row * columns, columns, std::numeric_limits<double>::quiet_NaN());
                    continue;
                }
                for (std::size_t direction = 0; direction < columns; ++direction)
                    for (std::size_t variable = 0; variable < n; ++variable)
                        if (directions[variable * count + first + direction] != 0.0)
                            values[row * columns + direction] +=
                                gradient.values[item * n + variable] *
                                directions[variable * count + first + direction];
            }
        }
        sink.consume({first, columns, rows, columns, std::span<const double>(values.data(), values.size())});
        ++report.blocks;
    }
    report.productSeconds = secondsSince(productsStarted);
    return report;
}

design::ProductReport fullShapeVjp(OptimizationEvaluation& point, const design::ProductWeights& weights,
                                   std::size_t block, design::ProductSink& sink)
{
    if (block == 0)
        throw std::invalid_argument("vjp: expected block > 0");
    validateWeights(point.evaluation(), weights);
    const std::size_t n = point.x().size();
    design::GradientOptions options;
    for (const auto& response : point.evaluation().responses)
    {
        const auto dense = weights.weights.find(response.id);
        const auto sparse = weights.sparseWeights.find(response.id);
        if (dense == weights.weights.end() && sparse == weights.sparseWeights.end())
            continue;
        options.responseIds.push_back(response.id);
        std::vector<std::uint8_t> wanted(response.itemCount(), 0);
        if (dense != weights.weights.end())
            for (std::size_t item = 0; item < wanted.size(); ++item)
                for (std::size_t row = 0; row < weights.rows; ++row)
                    wanted[item] |= dense->second[row * wanted.size() + item] != 0.0;
        else
            for (const auto item : sparse->second.items)
                wanted[item] = 1;
        options.itemMasks.push_back({response.id, std::move(wanted)});
    }
    const auto started = Clock::now();
    const design::DesignGradients gradients = options.responseIds.empty() ? design::DesignGradients{} :
                                               point.gradients(options);
    design::ProductReport report = productReport(gradients, secondsSince(started));
    const auto byId = gradientsById(gradients);
    const auto productsStarted = Clock::now();
    for (std::size_t first = 0; first < weights.rows; first += block)
    {
        const std::size_t rows = std::min(block, weights.rows - first);
        std::vector<double> values(rows * n, 0.0);
        for (const auto& response : point.evaluation().responses)
        {
            const auto gradient = byId.find(response.id);
            if (gradient == byId.end())
                continue;
            const auto dense = weights.weights.find(response.id);
            const auto sparse = weights.sparseWeights.find(response.id);
            for (std::size_t row = 0; row < rows; ++row)
            {
                const auto add = [&](std::size_t item, double weight)
                {
                    if (weight == 0.0)
                        return;
                    for (std::size_t variable = 0; variable < n; ++variable)
                        values[row * n + variable] += weight * gradient->second->values[item * n + variable];
                };
                if (dense != weights.weights.end())
                    for (std::size_t item = 0; item < response.itemCount(); ++item)
                    {
                        add(item, dense->second[(first + row) * response.itemCount() + item]);
                        ++report.weightsRead;
                    }
                else if (sparse != weights.sparseWeights.end())
                    for (std::size_t entry = sparse->second.rowStart[first + row];
                         entry < sparse->second.rowStart[first + row + 1]; ++entry)
                    {
                        add(sparse->second.items[entry], sparse->second.values[entry]);
                        ++report.weightsRead;
                    }
            }
        }
        sink.consume({first, rows, rows, n, std::span<const double>(values.data(), values.size())});
        ++report.blocks;
    }
    report.productSeconds = secondsSince(productsStarted);
    return report;
}

class FullShapeGradientSink final : public design::GradientSink
{
public:
    FullShapeGradientSink(OptimizationEvaluation& point, design::GradientSink& downstream,
                          std::vector<std::size_t> shapeColumns)
        : m_point(point), m_downstream(downstream), m_shapeColumns(std::move(shapeColumns))
    {
    }

    void consume(const design::GradientBlock& block) override
    {
        const auto first = std::lower_bound(m_shapeColumns.begin(), m_shapeColumns.end(), block.columnBegin);
        if (first == m_shapeColumns.end() || *first >= block.columnBegin + block.columnCount)
        {
            m_downstream.consume(block);
            return;
        }
        std::vector<double> values(block.values.begin(), block.values.end());
        for (auto column = first; column != m_shapeColumns.end() &&
                                  *column < block.columnBegin + block.columnCount; ++column)
        {
            // The direct stream emits every row of a column in one block. Release each full-analysis
            // column before the next block so memory does not grow with the number of shape variables.
            const FullShapeColumn shape = fullShapeColumn(m_point, *column);
            m_analysisPerturbations += shape.analysisPerturbations;
            m_peakBytes = std::max(m_peakBytes,
                                   static_cast<std::uint64_t>((shape.values.size() + values.size()) * sizeof(double)));
            for (std::size_t row = 0; row < block.rowCount; ++row)
                values[row * block.columnCount + *column - block.columnBegin] =
                    shape.values.at(block.rowBegin + row);
        }
        m_downstream.consume({block.rowBegin, block.rowCount, block.columnBegin, block.columnCount,
                              std::span<const double>(values.data(), values.size())});
    }

    [[nodiscard]] std::int64_t analysisPerturbations() const noexcept { return m_analysisPerturbations; }
    [[nodiscard]] std::uint64_t peakBytes() const noexcept { return m_peakBytes; }

private:
    OptimizationEvaluation& m_point;
    design::GradientSink& m_downstream;
    std::vector<std::size_t> m_shapeColumns;
    std::int64_t m_analysisPerturbations = 0;
    std::uint64_t m_peakBytes = 0;
};

design::GradientStreamReport fullShapeStream(OptimizationEvaluation& point, base::MemoryBudget& budget,
                                              design::GradientSink& sink, std::size_t block)
{
    if (block == 0)
        throw std::invalid_argument("stream_gradients: expected block > 0");
    const std::size_t n = point.x().size();
    std::vector<std::size_t> shapeColumns;
    for (const auto& shape : point.evaluation().analysis.point.values().shapes)
        for (const auto& [column, rate] : shape.gradient)
            if (rate != 0.0 && column < n)
                shapeColumns.push_back(column);
    std::sort(shapeColumns.begin(), shapeColumns.end());
    shapeColumns.erase(std::unique(shapeColumns.begin(), shapeColumns.end()), shapeColumns.end());
    design::GradientStreamOptions options;
    options.block = block;
    options.method = design::SensitivityMethod::Direct; // by columns: each full-analysis shape column is corrected once
    options.isVariableWanted.assign(n, 1);
    for (const auto column : shapeColumns)
        options.isVariableWanted[column] = 0;
    FullShapeGradientSink corrected(point, sink, std::move(shapeColumns));
    const auto& owner = point.owner();
    design::GradientStreamReport report = owner.problem().streamGradients(
        point.evaluation(), owner.callables(), budget, options, corrected, owner.settings().analysisOptions,
        owner.settings().blockItems);
    report.linearisation.analysisPerturbations += corrected.analysisPerturbations();
    report.measuredPeakBytes += corrected.peakBytes();
    return report;
}

} // namespace oos::api::detail
