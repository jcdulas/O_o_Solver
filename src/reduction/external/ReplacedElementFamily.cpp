#include "oos/reduction/external/ReplacedElementFamily.hpp"

#include <algorithm>

namespace oos::reduction
{

ReplacedElementFamily::ReplacedElementFamily(const elements::ElementFamily& inner,
                                             std::span<const base::EntityId> replaced)
    : m_inner(&inner), m_replaced(replaced)
{
}

bool ReplacedElementFamily::isReplaced(base::EntityId elementId) const noexcept
{
    return std::binary_search(m_replaced.begin(), m_replaced.end(), elementId);
}

std::vector<elements::ElementRange> ReplacedElementFamily::keptRuns(const elements::ElementBatchView& view,
                                                                    elements::ElementRange range) const
{
    std::vector<elements::ElementRange> runs; // bounded by the range
    std::int64_t first = range.first;
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        if (!isReplaced(view.elementId(row)))
            continue;
        if (row > first)
            runs.push_back({first, row});
        first = row + 1;
    }
    if (range.last > first)
        runs.push_back({first, range.last});
    return runs;
}

void ReplacedElementFamily::appendRows(const elements::ElementResultBlock& from, elements::ElementResultBlock& to)
{
    for (std::size_t i = 0; i < from.size(); ++i)
        to.append(from.elementRow(i), from.elementId(i), from.place(i), from.values(i));
}

void ReplacedElementFamily::check(const elements::ElementBatchView& view, base::DiagnosticSink& sink) const
{
    m_inner->check(view, sink);
}

void ReplacedElementFamily::computeStiffness(const elements::ElementBatchView& view, elements::ElementRange range,
                                             elements::ElementMatrixBlock& block) const
{
    for (const elements::ElementRange run : keptRuns(view, range))
        m_inner->computeStiffness(view, run, block);
}

void ReplacedElementFamily::computeMass(const elements::ElementBatchView& view, elements::ElementRange range,
                                        const elements::MassOptions& options, elements::ElementMatrixBlock& block) const
{
    for (const elements::ElementRange run : keptRuns(view, range))
        m_inner->computeMass(view, run, options, block);
}

std::vector<std::string> ReplacedElementFamily::outputColumns(elements::ElementOutput output) const
{
    return m_inner->outputColumns(output);
}

void ReplacedElementFamily::recover(const elements::ElementBatchView& view, elements::ElementRange range,
                                    elements::ElementOutput output, const elements::NodalDisplacements& displacements,
                                    elements::ElementResultBlock& block) const
{
    block.reset(m_inner->outputColumns(output));
    elements::ElementResultBlock run;
    for (const elements::ElementRange kept : keptRuns(view, range))
    {
        m_inner->recover(view, kept, output, displacements, run);
        appendRows(run, block);
    }
}

void ReplacedElementFamily::checkPressureLoads(const elements::ElementBatchView& view,
                                               std::span<const elements::ElementPressure> pressures,
                                               base::DiagnosticSink& sink) const
{
    m_inner->checkPressureLoads(view, pressures, sink);
}

void ReplacedElementFamily::computePressureLoads(const elements::ElementBatchView& view,
                                                 std::span<const elements::ElementPressure> pressures,
                                                 elements::ElementVectorBlock& block) const
{
    m_inner->computePressureLoads(view, pressures, block);
}

void ReplacedElementFamily::checkLineLoads(const elements::ElementBatchView& view,
                                           std::span<const elements::ElementLineLoad> loads,
                                           base::DiagnosticSink& sink) const
{
    m_inner->checkLineLoads(view, loads, sink);
}

void ReplacedElementFamily::computeLineLoads(const elements::ElementBatchView& view,
                                             std::span<const elements::ElementLineLoad> loads,
                                             elements::ElementVectorBlock& block) const
{
    m_inner->computeLineLoads(view, loads, block);
}

void ReplacedElementFamily::checkThermalLoads(const elements::ElementBatchView& view,
                                              const elements::ElementTemperatures& temperatures,
                                              base::DiagnosticSink& sink) const
{
    m_inner->checkThermalLoads(view, temperatures, sink);
}

void ReplacedElementFamily::computeThermalLoads(const elements::ElementBatchView& view, elements::ElementRange range,
                                                const elements::ElementTemperatures& temperatures,
                                                elements::ElementVectorBlock& block) const
{
    for (const elements::ElementRange run : keptRuns(view, range))
        m_inner->computeThermalLoads(view, run, temperatures, block);
}

void ReplacedElementFamily::checkDifferentialStiffness(const elements::ElementBatchView& view,
                                                       base::DiagnosticSink& sink) const
{
    m_inner->checkDifferentialStiffness(view, sink);
}

void ReplacedElementFamily::computeDifferentialStiffness(const elements::ElementBatchView& view,
                                                         elements::ElementRange range,
                                                         const elements::NodalDisplacements& preload,
                                                         elements::ElementMatrixBlock& block) const
{
    for (const elements::ElementRange run : keptRuns(view, range))
        m_inner->computeDifferentialStiffness(view, run, preload, block);
}

void ReplacedElementFamily::recoverUnderLineLoads(const elements::ElementBatchView& view,
                                                  std::span<const elements::ElementLineLoad> loads,
                                                  elements::ElementOutput output,
                                                  const elements::NodalDisplacements& displacements,
                                                  elements::ElementResultBlock& block) const
{
    std::vector<elements::ElementLineLoad> kept; // bounded by the loads of the batch
    for (const elements::ElementLineLoad& load : loads)
        if (!isReplaced(view.elementId(load.elementRow)))
            kept.push_back(load);
    if (kept.empty())
    {
        block.reset(m_inner->outputColumnsUnderLineLoads(output));
        return;
    }
    m_inner->recoverUnderLineLoads(view, kept, output, displacements, block);
}

void ReplacedElementFamily::recoverLayers(const elements::ElementBatchView& view, elements::ElementRange range,
                                          elements::LayerOutput output,
                                          const elements::NodalDisplacements& displacements,
                                          elements::ElementResultBlock& block) const
{
    block.reset(m_inner->layerOutputColumns(output));
    elements::ElementResultBlock run;
    for (const elements::ElementRange kept : keptRuns(view, range))
    {
        m_inner->recoverLayers(view, kept, output, displacements, run);
        appendRows(run, block);
    }
}

} // namespace oos::reduction
