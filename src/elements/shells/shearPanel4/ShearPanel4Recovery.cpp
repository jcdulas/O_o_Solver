#include "oos/elements/core/MarginOfSafety.hpp"
#include "oos/elements/shells/shearPanel4/ShearPanel4Element.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace oos::elements
{

// The forms of the columns: the corner and kick forces and the shear flows Linear (the flows are c_e·q̄),
// the average shear Linear, the largest |τ_e| and the margin Other.
std::vector<ColumnForm> ShearPanel4Element::outputColumnForms(ElementOutput output) const
{
    const ColumnForm linear = ColumnForm::linear();
    const ColumnForm other = ColumnForm::other();
    switch (output)
    {
    case ElementOutput::Forces:
        return std::vector<ColumnForm>(16, linear);
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {other, linear, other};
    case ElementOutput::StrainCurvatures:
        break;
    }
    return {};
}

std::vector<std::string> ShearPanel4Element::outputColumns(ElementOutput output) const
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"force41",    "force21", "force12",    "force32", "force23",    "force43", "force34",    "force14",
                "kickForce1", "shear12", "kickForce2", "shear23", "kickForce3", "shear34", "kickForce4", "shear41"};
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {"maxShear", "avgShear", "margin"};
    case ElementOutput::StrainCurvatures:
        break;
    }
    return {};
}

namespace
{

// The vectorised recovery of the ShearPanel4: the fewest cases of a block from which its rows are made from its
// slopes (recoverySlopesOfOutputs: one resolution of the panel). Calibrated on a ShearPanel4 panel model (22 500
// ShearPanel4 framed by 45 300 Rod2, six threads): the per-case recovery takes about 0.038 s per case, the
// recovery by blocks 0.20 s once and 0.0075 s per case: even from about 5.7 cases; 8.
constexpr std::size_t SHEAR_PANEL_VECTORISED_RECOVERY_MIN_CASES = 8;

// The scale from a shear flow to the stress (1/T) or to the strain (1/(T·G)) of the panel.
double flowScale(const ShearPanel4Element::Data& data, ElementOutput output)
{
    return output == ElementOutput::Stresses ? 1.0 / data.section.thickness
                                             : 1.0 / (data.section.thickness * data.section.shearModulus);
}

} // namespace

std::size_t ShearPanel4Element::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SHEAR_PANEL_VECTORISED_RECOVERY_MIN_CASES;
}

// The largest |τ_e| and the margin from the average shear AVG = ¼·Σ c_e·q̄·scale (recover()): the mean flow
// q̄ = AVG / (¼·scale·Σ c_e) (Σ c_e = 4 to round-off: ShearPanel4Panel normalises the pattern to a mean of 1), then
// as recover(): the flows c_e·q̄, the largest of their magnitudes times the scale, its margin; equal to recover()'s to
// round-off. The forces are all Linear.
void ShearPanel4Element::completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                                                  std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    if (output == ElementOutput::Forces || blocks.empty())
        return;
    const Columns columns(view);
    const ElementResultBlock& structure = blocks.front();
    for (std::size_t i = 0; i < structure.size(); ++i)
    {
        const Data data = resolve(view, columns, structure.elementRow(i)).required();
        const std::array<double, 4>& pattern = data.panel.flowPattern();
        const double scale = flowScale(data, output);
        double patternSum = 0.0;
        for (const double c : pattern)
            patternSum += c;
        for (ElementResultBlock& block : blocks)
        {
            const std::span<double> values = block.mutableValues(i);
            const double mean = values[1] / (0.25 * scale * patternSum);
            double largest = 0.0;
            for (const double c : pattern)
                largest = std::max(largest, std::abs(c * mean));
            values[0] = largest * scale;
            values[2] =
                output == ElementOutput::Stresses ? shearStressMargin(values[0], data.shearLimit) : UNDEFINED_MARGIN;
        }
    }
}

namespace
{

// The row of a panel of mean flow q̄ (recover()): the shear flows q_e = c_e·q̄ of the panel (ShearPanel4Panel), the
// stresses τ_e = q_e / T and their strains τ_e / G.
struct ShearPanel4Row
{
    std::array<double, 16> values{};
    std::size_t count = 0;
};

ShearPanel4Row shearPanel4RowOf(const ShearPanel4Element::Data& data, double mean, ElementOutput output)
{
    const std::array<double, 4>& pattern = data.panel.flowPattern();
    std::array<double, 4> flows{};
    for (std::size_t e = 0; e < 4; ++e)
        flows[e] = pattern[e] * mean;
    ShearPanel4Row row;
    if (output == ElementOutput::Forces)
    {
        // F_ij at corner i along edge ij: half the force q_e L_e of edge e = ij; the kick
        // forces of a flat panel are zero.
        const std::array<double, 4>& lengths = data.panel.edgeLengths();
        std::array<double, 4> half{};
        for (std::size_t e = 0; e < 4; ++e)
            half[e] = 0.5 * flows[e] * lengths[e];
        row.values = {half[3], half[0],  half[0], half[1],  half[1], half[2],  half[2], half[3],
                      0.0,     flows[0], 0.0,     flows[1], 0.0,     flows[2], 0.0,     flows[3]};
        row.count = 16;
        return row;
    }
    // MAX SHEAR = max(|τ_e|) (written positive), AVG SHEAR =
    // (τ1 + τ2 + τ3 + τ4)/4 with its sign (Garvey panel).
    const double scale = flowScale(data, output);
    double largest = 0.0;
    double sum = 0.0;
    for (const double flow : flows)
    {
        largest = std::max(largest, std::abs(flow));
        sum += flow;
    }
    const double maximum = largest * scale;
    const double margin =
        output == ElementOutput::Stresses ? shearStressMargin(maximum, data.shearLimit) : UNDEFINED_MARGIN;
    row.values[0] = maximum;
    row.values[1] = 0.25 * sum * scale;
    row.values[2] = margin;
    row.count = 3;
    return row;
}

} // namespace

void ShearPanel4Element::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                    const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Data data = resolve(view, columns, row).required();
        const ShearPanel4Row values =
            shearPanel4RowOf(data, data.panel.meanFlow(data.localDisplacements(displacements)), output);
        block.append(row, view.elementId(row), std::span<const double>(values.values.data(), values.count));
    }
}

// The recovery matrix of a ShearPanel4 (RecoverySlopes): its rows are linear in its
// mean flow q̄ = (c·q)/f (ShearPanel4Panel::meanFlow), itself linear in the translations of its grid points (q =
// Eᵀ·R_i·u_i: Data::localDisplacements), so the slope of DOF d is rows(e_d) − rows(0); every output from one resolution
// of the panel and one mean flow per DOF. PERFORMANCE: the slopes a ShearPanel4 gave by 13 recoveries at unit
// displacements per output (each resolving the panel again) from one resolution for all its outputs. Measured: the
// recovery of a ShearPanel4 panel model.
bool ShearPanel4Element::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                                    std::span<const RecoveryOutput> outputs,
                                                    std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
    {
        if (output.isLayer)
            return false;
        requireOutput(output.output);
    }
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const std::array<ElementDof, 12> dofs = data.dofs();
    std::vector<std::vector<ColumnForm>> forms; // per output
    std::vector<ShearPanel4Row> atRest;      // per output
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        forms.push_back(outputColumnForms(outputs[k].output));
        atRest.push_back(shearPanel4RowOf(data, 0.0, outputs[k].output));
        slopes[k].reset(dofs, atRest[k].count);
        slopes[k].appendRow({});
    }
    const base::Mat3& axes = data.panel.axes();
    for (std::size_t i = 0; i < 4; ++i)
    {
        const base::Mat3 toLocal = axes.transposed() * data.cdFrames[i];
        for (std::size_t j = 0; j < 3; ++j)
        {
            // The local translations of a unit displacement of component j + 1 of grid point i.
            std::array<double, 12> q{};
            for (std::size_t m = 0; m < 3; ++m)
                q[3 * i + m] = toLocal(m, j);
            const double mean = data.panel.meanFlow(q);
            for (std::size_t k = 0; k < outputs.size(); ++k)
            {
                const ShearPanel4Row moved = shearPanel4RowOf(data, mean, outputs[k].output);
                for (std::size_t c = 0; c < atRest[k].count; ++c)
                    if (forms[k][c].kind == ColumnForm::Kind::Linear)
                        slopes[k].slopes(0, c)[3 * i + j] = moved.values[c] - atRest[k].values[c];
            }
        }
    }
    return true;
}

bool ShearPanel4Element::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                           RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

} // namespace oos::elements
