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

bool ReplacedElementFamily::isReplacedRow(const elements::ElementBatchView& view, std::int64_t row) const
{
    return isReplaced(view.elementId(row));
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

void ReplacedElementFamily::checkRows(const elements::ElementBatchView& view, elements::ElementRange range,
                                      base::DiagnosticSink& sink) const
{
    m_inner->checkRows(view, range, sink);
}

bool ReplacedElementFamily::computeVolumes(const elements::ElementBatchView& view, elements::ElementRange range,
                                           std::span<double> volumes) const
{
    // Whether the card gives volumes is the wrapped family's answer, the same for every range: asked on an empty
    // range, as assembly::ElementVolumes asks it (a range of replaced elements alone must answer as the others).
    if (!m_inner->computeVolumes(view, {range.first, range.first}, {}))
        return false;
    std::fill(volumes.begin(), volumes.end(), 0.0);
    for (const elements::ElementRange run : keptRuns(view, range))
    {
        const std::span<double> ofRun = volumes.subspan(static_cast<std::size_t>(run.first - range.first),
                                                        static_cast<std::size_t>(run.last - run.first));
        if (!m_inner->computeVolumes(view, run, ofRun))
            return false;
    }
    return true;
}

bool ReplacedElementFamily::computeStabilisationEnergies(const elements::ElementBatchView& view,
                                                         elements::ElementRange range,
                                                         const elements::NodalDisplacements& displacements,
                                                         std::span<double> energies) const
{
    // As computeVolumes: the card's answer first, so that a range of replaced elements alone does not read as a
    // family without a stabilisation term (the caller would then skip the rest of the batch).
    if (!m_inner->computeStabilisationEnergies(view, {range.first, range.first}, displacements, {}))
        return false;
    std::fill(energies.begin(), energies.end(), 0.0);
    for (const elements::ElementRange run : keptRuns(view, range))
    {
        const std::span<double> ofRun = energies.subspan(static_cast<std::size_t>(run.first - range.first),
                                                         static_cast<std::size_t>(run.last - run.first));
        if (!m_inner->computeStabilisationEnergies(view, run, displacements, ofRun))
            return false;
    }
    return true;
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

std::vector<std::int64_t> ReplacedElementFamily::stationRows(const elements::ElementBatchView& view) const
{
    std::vector<std::int64_t> kept; // bounded by the station rows of the batch
    for (const std::int64_t row : m_inner->stationRows(view))
        if (!isReplaced(view.elementId(row)))
            kept.push_back(row);
    return kept;
}

void ReplacedElementFamily::recoverStationRows(const elements::ElementBatchView& view,
                                               std::span<const std::int64_t> rows,
                                               std::span<const elements::ElementLineLoad> loads,
                                               elements::ElementOutput output,
                                               const elements::NodalDisplacements& displacements,
                                               elements::ElementResultBlock& block) const
{
    std::vector<std::int64_t> keptRows; // bounded by the rows given
    for (const std::int64_t row : rows)
        if (!isReplaced(view.elementId(row)))
            keptRows.push_back(row);
    std::vector<elements::ElementLineLoad> keptLoads; // bounded by the loads of the batch
    for (const elements::ElementLineLoad& load : loads)
        if (!isReplaced(view.elementId(load.elementRow)))
            keptLoads.push_back(load);
    if (keptRows.empty())
    {
        block.reset(m_inner->outputColumnsUnderLineLoads(output));
        return;
    }
    m_inner->recoverStationRows(view, keptRows, keptLoads, output, displacements, block);
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

void ReplacedElementFamily::recoverOutputs(const elements::ElementBatchView& view, elements::ElementRange range,
                                           std::span<const elements::RecoveryOutput> outputs,
                                           const elements::NodalDisplacements& displacements,
                                           std::span<elements::ElementResultBlock> blocks) const
{
    // As recover() and recoverLayers(): each block reset to its columns, then the rows of the runs without a replaced
    // element; a run is recovered by the wrapped family for every output at once, each with what it asks (its corner
    // rows: the base class would recover each output by recover(), which gives the centre rows only).
    for (std::size_t k = 0; k < outputs.size(); ++k)
        blocks[k].reset(outputs[k].isLayer ? m_inner->layerOutputColumns(outputs[k].layerOutput)
                                           : m_inner->outputColumns(outputs[k].output));
    std::vector<elements::ElementResultBlock> run(outputs.size()); // one block per output
    for (const elements::ElementRange kept : keptRuns(view, range))
    {
        m_inner->recoverOutputs(view, kept, outputs, displacements, run);
        for (std::size_t k = 0; k < outputs.size(); ++k)
            appendRows(run[k], blocks[k]);
    }
}

bool ReplacedElementFamily::recoverAtUnitDisplacements(const elements::ElementBatchView& view, std::int64_t row,
                                                       elements::ElementOutput output,
                                                       std::vector<elements::ElementDof>& dofs,
                                                       std::vector<elements::ElementResultBlock>& blocks) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::recoverAtUnitDisplacements(view, row, output, dofs, blocks);
    return m_inner->recoverAtUnitDisplacements(view, row, output, dofs, blocks);
}

bool ReplacedElementFamily::recoverySlopes(const elements::ElementBatchView& view, std::int64_t row,
                                           elements::ElementOutput output, elements::RecoverySlopes& slopes) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::recoverySlopes(view, row, output, slopes);
    return m_inner->recoverySlopes(view, row, output, slopes);
}

bool ReplacedElementFamily::layerRecoverySlopes(const elements::ElementBatchView& view, std::int64_t row,
                                                elements::LayerOutput output, elements::RecoverySlopes& slopes) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::layerRecoverySlopes(view, row, output, slopes);
    return m_inner->layerRecoverySlopes(view, row, output, slopes);
}

bool ReplacedElementFamily::recoverySlopesOfOutputs(const elements::ElementBatchView& view, std::int64_t row,
                                                    std::span<const elements::RecoveryOutput> outputs,
                                                    std::span<elements::RecoverySlopes> slopes) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::recoverySlopesOfOutputs(view, row, outputs, slopes);
    return m_inner->recoverySlopesOfOutputs(view, row, outputs, slopes);
}

void ReplacedElementFamily::completeRecoveredRows(const elements::ElementBatchView& view,
                                                  elements::ElementOutput output,
                                                  std::span<elements::ElementResultBlock> blocks) const
{
    m_inner->completeRecoveredRows(view, output, blocks);
}

void ReplacedElementFamily::completeRecoveredRowsFromSource(const elements::ElementBatchView& view,
                                                            elements::ElementOutput output,
                                                            std::span<const elements::ElementResultBlock> sources,
                                                            std::span<elements::ElementResultBlock> blocks) const
{
    m_inner->completeRecoveredRowsFromSource(view, output, sources, blocks);
}

void ReplacedElementFamily::completeRecoveredLayerRows(const elements::ElementBatchView& view,
                                                       elements::LayerOutput output,
                                                       std::span<const elements::ElementResultBlock> sources,
                                                       std::span<elements::ElementResultBlock> blocks) const
{
    m_inner->completeRecoveredLayerRows(view, output, sources, blocks);
}

bool ReplacedElementFamily::stiffnessFieldDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                     const elements::DesignedField& field,
                                                     elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::stiffnessFieldDerivative(view, row, field, block);
    return m_inner->stiffnessFieldDerivative(view, row, field, block);
}

void ReplacedElementFamily::stiffnessFieldDerivatives(const elements::ElementBatchView& view, std::int64_t row,
                                                      std::span<const elements::DesignedField> fields,
                                                      std::span<elements::ElementMatrixBlock> blocks,
                                                      std::span<std::uint8_t> isClosed) const
{
    if (isReplacedRow(view, row))
        ElementFamily::stiffnessFieldDerivatives(view, row, fields, blocks, isClosed);
    else
        m_inner->stiffnessFieldDerivatives(view, row, fields, blocks, isClosed);
}

bool ReplacedElementFamily::massFieldDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                const elements::DesignedField& field,
                                                const elements::MassOptions& options,
                                                elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::massFieldDerivative(view, row, field, options, block);
    return m_inner->massFieldDerivative(view, row, field, options, block);
}

namespace
{

// The loads of a replaced element that the analysis keeps: its pressure and line loads (computePressureLoads and
// computeLineLoads pass them through), not its thermal load (computeThermalLoads leaves it out).
elements::ElementLoadContext withoutTemperatures(const elements::ElementLoadContext& loads)
{
    elements::ElementLoadContext kept = loads;
    kept.temperatures = nullptr;
    return kept;
}

} // namespace

bool ReplacedElementFamily::loadFieldDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                const elements::DesignedField& field,
                                                const elements::ElementLoadContext& loads,
                                                elements::ElementVectorBlock& block) const
{
    if (isReplacedRow(view, row))
        return m_inner->loadFieldDerivative(view, row, field, withoutTemperatures(loads), block);
    return m_inner->loadFieldDerivative(view, row, field, loads, block);
}

bool ReplacedElementFamily::loadFieldMixedDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                     const elements::DesignedField& first,
                                                     const elements::DesignedField& second,
                                                     const elements::ElementLoadContext& loads,
                                                     elements::ElementVectorBlock& block) const
{
    if (isReplacedRow(view, row))
        return m_inner->loadFieldMixedDerivative(view, row, first, second, withoutTemperatures(loads), block);
    return m_inner->loadFieldMixedDerivative(view, row, first, second, loads, block);
}

bool ReplacedElementFamily::differentialStiffnessFieldDerivative(const elements::ElementBatchView& view,
                                                                 std::int64_t row,
                                                                 const elements::DesignedField& field,
                                                                 const elements::NodalDisplacements& preload,
                                                                 elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::differentialStiffnessFieldDerivative(view, row, field, preload, block);
    return m_inner->differentialStiffnessFieldDerivative(view, row, field, preload, block);
}

bool ReplacedElementFamily::differentialStiffnessPreloadGradient(const elements::ElementBatchView& view,
                                                                 std::int64_t row,
                                                                 const elements::NodalDisplacements& shape,
                                                                 elements::ElementVectorBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::differentialStiffnessPreloadGradient(view, row, shape, block);
    return m_inner->differentialStiffnessPreloadGradient(view, row, shape, block);
}

bool ReplacedElementFamily::differentialStiffnessFieldMixedDerivative(const elements::ElementBatchView& view,
                                                                      std::int64_t row,
                                                                      const elements::DesignedField& first,
                                                                      const elements::DesignedField& second,
                                                                      const elements::NodalDisplacements& preload,
                                                                      elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::differentialStiffnessFieldMixedDerivative(view, row, first, second, preload, block);
    return m_inner->differentialStiffnessFieldMixedDerivative(view, row, first, second, preload, block);
}

bool ReplacedElementFamily::recoveryFieldRates(const elements::ElementBatchView& view, std::int64_t row,
                                               const elements::DesignedField& field, elements::ElementOutput output,
                                               const elements::NodalDisplacements& displacements,
                                               elements::ElementResultBlock& rates) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::recoveryFieldRates(view, row, field, output, displacements, rates);
    return m_inner->recoveryFieldRates(view, row, field, output, displacements, rates);
}

void ReplacedElementFamily::recoveryFieldRatesOfFields(const elements::ElementBatchView& view, std::int64_t row,
                                                       std::span<const elements::DesignedField> fields,
                                                       elements::ElementOutput output,
                                                       const elements::NodalDisplacements& displacements,
                                                       std::span<elements::ElementResultBlock> rates,
                                                       std::span<std::uint8_t> isClosed) const
{
    if (isReplacedRow(view, row))
        ElementFamily::recoveryFieldRatesOfFields(view, row, fields, output, displacements, rates, isClosed);
    else
        m_inner->recoveryFieldRatesOfFields(view, row, fields, output, displacements, rates, isClosed);
}

bool ReplacedElementFamily::layerRecoveryFieldRates(const elements::ElementBatchView& view, std::int64_t row,
                                                    const elements::DesignedField& field,
                                                    elements::LayerOutput output,
                                                    const elements::NodalDisplacements& displacements,
                                                    elements::ElementResultBlock& rates) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::layerRecoveryFieldRates(view, row, field, output, displacements, rates);
    return m_inner->layerRecoveryFieldRates(view, row, field, output, displacements, rates);
}

void ReplacedElementFamily::layerRecoveryFieldRatesOfFields(const elements::ElementBatchView& view, std::int64_t row,
                                                            std::span<const elements::DesignedField> fields,
                                                            elements::LayerOutput output,
                                                            const elements::NodalDisplacements& displacements,
                                                            std::span<elements::ElementResultBlock> rates,
                                                            std::span<std::uint8_t> isClosed) const
{
    if (isReplacedRow(view, row))
        ElementFamily::layerRecoveryFieldRatesOfFields(view, row, fields, output, displacements, rates, isClosed);
    else
        m_inner->layerRecoveryFieldRatesOfFields(view, row, fields, output, displacements, rates, isClosed);
}

bool ReplacedElementFamily::stiffnessFieldSecondDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                           const elements::DesignedField& field,
                                                           elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::stiffnessFieldSecondDerivative(view, row, field, block);
    return m_inner->stiffnessFieldSecondDerivative(view, row, field, block);
}

bool ReplacedElementFamily::massFieldSecondDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                      const elements::DesignedField& field,
                                                      const elements::MassOptions& options,
                                                      elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::massFieldSecondDerivative(view, row, field, options, block);
    return m_inner->massFieldSecondDerivative(view, row, field, options, block);
}

bool ReplacedElementFamily::stiffnessFieldMixedDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                          const elements::DesignedField& first,
                                                          const elements::DesignedField& second,
                                                          elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::stiffnessFieldMixedDerivative(view, row, first, second, block);
    return m_inner->stiffnessFieldMixedDerivative(view, row, first, second, block);
}

bool ReplacedElementFamily::massFieldMixedDerivative(const elements::ElementBatchView& view, std::int64_t row,
                                                     const elements::DesignedField& first,
                                                     const elements::DesignedField& second,
                                                     const elements::MassOptions& options,
                                                     elements::ElementMatrixBlock& block) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::massFieldMixedDerivative(view, row, first, second, options, block);
    return m_inner->massFieldMixedDerivative(view, row, first, second, options, block);
}

bool ReplacedElementFamily::recoverySlopeRates(const elements::ElementBatchView& view, std::int64_t row,
                                               const elements::DesignedField& field, elements::ElementOutput output,
                                               elements::RecoverySlopes& rates) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::recoverySlopeRates(view, row, field, output, rates);
    return m_inner->recoverySlopeRates(view, row, field, output, rates);
}

bool ReplacedElementFamily::recoveryFieldSecondRates(const elements::ElementBatchView& view, std::int64_t row,
                                                     const elements::DesignedField& field,
                                                     elements::ElementOutput output,
                                                     const elements::NodalDisplacements& displacements,
                                                     elements::ElementResultBlock& rates) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::recoveryFieldSecondRates(view, row, field, output, displacements, rates);
    return m_inner->recoveryFieldSecondRates(view, row, field, output, displacements, rates);
}

bool ReplacedElementFamily::recoveryFieldMixedRates(const elements::ElementBatchView& view, std::int64_t row,
                                                    const elements::DesignedField& first,
                                                    const elements::DesignedField& second,
                                                    elements::ElementOutput output,
                                                    const elements::NodalDisplacements& displacements,
                                                    elements::ElementResultBlock& rates) const
{
    if (isReplacedRow(view, row))
        return ElementFamily::recoveryFieldMixedRates(view, row, first, second, output, displacements, rates);
    return m_inner->recoveryFieldMixedRates(view, row, first, second, output, displacements, rates);
}

} // namespace oos::reduction
