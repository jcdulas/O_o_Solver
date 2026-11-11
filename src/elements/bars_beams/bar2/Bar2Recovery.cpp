#include "oos/elements/bars_beams/BeamResponse.hpp"
#include "oos/elements/bars_beams/BeamThermal.hpp"
#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"
#include "oos/elements/core/MarginOfSafety.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace oos::elements
{

// The forms of the columns: the forces and the stresses or strains at the recovery points and the axial
// one Linear (P/A ± M·c/I, affine under a thermal strain); the extremes over the points (piecewise linear) and the
// margins Other.
std::vector<ColumnForm> Bar2Element::outputColumnForms(ElementOutput output) const
{
    const ColumnForm linear = ColumnForm::linear();
    const ColumnForm other = ColumnForm::other();
    switch (output)
    {
    case ElementOutput::Forces:
        return std::vector<ColumnForm>(8, linear);
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {linear, linear, linear, linear, linear, other, other, other,
                linear, linear, linear, linear, other,  other, other};
    default:
        return {};
    }
}

std::vector<std::string> Bar2Element::outputColumns(ElementOutput output) const
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"bendingMomentA1", "bendingMomentA2", "bendingMomentB1", "bendingMomentB2",
                "shear1",          "shear2",          "axial",           "torque"};
    case ElementOutput::Stresses:
        return {"s1a", "s2a", "s3a", "s4a", "axial", "smaxa", "smina",        "msTension",
                "s1b", "s2b", "s3b", "s4b", "smaxb", "sminb", "msCompression"};
    case ElementOutput::Strains:
        return {"e1a", "e2a", "e3a", "e4a", "axial", "emaxa", "emina",        "msTension",
                "e1b", "e2b", "e3b", "e4b", "emaxb", "eminb", "msCompression"};
    default:
        return {};
    }
}

std::vector<std::string> Bar2Element::outputColumnsUnderLineLoads(ElementOutput output) const
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"bendingMoment1", "bendingMoment2", "shear1", "shear2", "axial", "torque"};
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {"sxc", "sxd", "sxe", "sxf", "axial", "smax", "smin", "ms"};
    default:
        return {};
    }
}

std::string Bar2Element::resultElementType(bool underLineLoads) const
{
    return underLineLoads ? "Bar2LoadStations" : "Bar2";
}

// The vectorised recovery of the Bar2: the fewest cases of a block from
// which its rows are made from its slopes. Calibrated on a 40-Bar2 chain in a 6 000-case deck
// with 8 to 96 cases, six threads: the per-case recovery takes about 0.28 ms per
// case, the vectorised one 1.4 ms once and 0.04 ms per case (8 cases 2.4 ms against 1.7 ms): even from about 6 cases;
// 8 as the shells.
constexpr std::size_t BAR_VECTORISED_RECOVERY_MIN_CASES = 8;

std::size_t Bar2Element::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : BAR_VECTORISED_RECOVERY_MIN_CASES;
}

// The extremes of each end, axial + the largest and smallest of its four point stresses (stationStresses: the axial
// stress of an unloaded bar the same at both ends), and the margins of the largest and smallest of them
// (tensionMargin, compressionMargin), as recover() makes them from its stresses; the strains' from the strains of the
// points (their extremes ε = σ/E exactly as recover()'s when σ/E is rounded alike: to round-off), their margins from
// the stresses σ = E·ε; the forces are all Linear.
void Bar2Element::completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                                           std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    if (output == ElementOutput::Forces || blocks.empty())
        return;
    const Columns columns(view);
    const ElementResultBlock& first = blocks.front();
    for (std::size_t i = 0; i < first.size(); ++i)
    {
        const BeamData beam = resolve(view, columns, first.elementRow(i)).required();
        const IsotropicMaterial& material = beam.material;
        const double toStress = output == ElementOutput::Strains ? material.youngModulus : 1.0;
        for (ElementResultBlock& block : blocks)
        {
            const std::span<double> values = block.mutableValues(i);
            const double axial = values[4];
            values[5] = axial + std::max({values[0], values[1], values[2], values[3]});
            values[6] = axial + std::min({values[0], values[1], values[2], values[3]});
            values[12] = axial + std::max({values[8], values[9], values[10], values[11]});
            values[13] = axial + std::min({values[8], values[9], values[10], values[11]});
            values[7] = tensionMargin(toStress * std::max(values[5], values[12]), material);
            values[14] = compressionMargin(toStress * std::min(values[6], values[13]), material);
        }
    }
}

namespace
{

// The row of one Bar2 under `response` (recover()): the forces at ends A and B (the shears and the axial force and
// torque are constant along an unloaded bar), 8 values, or the stresses (strains) of both ends, 15.
struct Bar2Row
{
    std::array<double, 15> values{};
    std::size_t count = 0;
};

Bar2Row bar2RowOf(const BeamData& beam, const BeamResponse& response, ElementOutput output)
{
    const BeamForces atA = response.forcesAt(0.0);
    const BeamForces atB = response.forcesAt(beam.geometry.length);
    Bar2Row row;
    if (output == ElementOutput::Forces)
    {
        row.values = {atA.moment1, atA.moment2, atB.moment1, atB.moment2,
                      atA.shear1,  atA.shear2,  atA.axial,   atA.torque};
        row.count = 8;
        return row;
    }
    const StationStresses endA = stationStresses(beam.section.stations[0], atA);
    const StationStresses endB = stationStresses(beam.section.stations[1], atB);
    const double tension = tensionMargin(std::max(endA.maximum, endB.maximum), beam.material);
    const double compression = compressionMargin(std::min(endA.minimum, endB.minimum), beam.material);
    row.values = {endA.bending[0], endA.bending[1], endA.bending[2], endA.bending[3], endA.axial,
                  endA.maximum,    endA.minimum,    tension,         endB.bending[0], endB.bending[1],
                  endB.bending[2], endB.bending[3], endB.maximum,    endB.minimum,    compression};
    if (output == ElementOutput::Strains)
        for (const std::size_t k : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 8U, 9U, 10U, 11U, 12U, 13U})
            row.values[k] = uniaxialStrain(row.values[k], beam.material);
    row.count = 15;
    return row;
}

} // namespace

// One row per Bar2 (bar2RowOf).
void Bar2Element::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                             const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const BeamData beam = resolve(view, columns, row).required();
        const auto loads = resolveLineLoadsOrThrow(view, displacements.lineLoads, beam.geometry, row);
        const BeamResponse response(beam, displacements, loads, beamThermalStrain(view, row, beam, displacements));
        const Bar2Row values = bar2RowOf(beam, response, output);
        block.append(row, view.elementId(row), std::span<const double>(values.values.data(), values.count));
    }
}

// The recovery matrix of a Bar2 (RecoverySlopes): its rows are linear in
// the displacements of its grid points in their Linear columns (no line load, no temperature), so the slope of DOF d is
// rows(e_d) − rows(0), the responses made from one end-force map (BeamEndForces: the stiffness, the end transform and
// the pin release built once), exact as a recovery at unit displacements and with its bits; every output from one
// resolution, one map and one response per DOF.
// PERFORMANCE: the slopes a Bar2 gave by 13 recoveries at unit displacements (each resolving the element, reading
// the batch's columns and building its stiffness again: about 90 µs per element and output) from one resolution, and
// then one for all its outputs.
// Measured: a 6 000-case deck of 40 Bar2; a line-loaded grillage.
bool Bar2Element::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
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
    const BeamData beam = resolve(view, columns, row).required();
    const BeamEndForces endForces(beam, {});
    const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
    std::array<double, 12> grid{};
    std::vector<Bar2Row> atRest; // per output
    {
        const BeamResponse response(beam, endForces.of(grid), {});
        for (const RecoveryOutput& output : outputs)
            atRest.push_back(bar2RowOf(beam, response, output.output));
    }
    std::vector<std::vector<ColumnForm>> forms; // per output
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        forms.push_back(outputColumnForms(outputs[k].output));
        slopes[k].reset(dofs, atRest[k].count);
        slopes[k].appendRow({});
    }
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
        grid[d] = 1.0;
        const BeamResponse response(beam, endForces.of(grid), {});
        grid[d] = 0.0;
        for (std::size_t k = 0; k < outputs.size(); ++k)
        {
            const Bar2Row moved = bar2RowOf(beam, response, outputs[k].output);
            for (std::size_t c = 0; c < atRest[k].count; ++c)
                if (forms[k][c].kind == ColumnForm::Kind::Linear)
                    slopes[k].slopes(0, c)[d] = moved.values[c] - atRest[k].values[c];
        }
    }
    return true;
}

// The rows of recover() at each unit displacement from one end-force map: recover() makes the end forces of a Bar2
// without temperatures as BeamEndForces(beam, {}, 0).of(grid), grid its displacements at its DOFs, so the map built
// once gives the same bits for each unit displacement.
// PERFORMANCE: output::SectionLoads recovered a Bar2 once per DOF (12), each resolving it and building its
// stiffness; here one resolution and one map (measured on a stiffened barrel panel).
bool Bar2Element::recoverAtUnitDisplacements(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                                std::vector<ElementDof>& dofs,
                                                std::vector<ElementResultBlock>& blocks) const
{
    requireOutput(output);
    const Columns columns(view);
    const BeamData beam = resolve(view, columns, row).required();
    const BeamEndForces endForces(beam, {});
    const std::array<ElementDof, 12> elementDofs = beam.geometry.dofs();
    dofs.assign(elementDofs.begin(), elementDofs.end());
    blocks.resize(dofs.size());
    const std::vector<std::string> columnNames = outputColumns(output);
    std::array<double, 12> grid{};
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
        grid[d] = 1.0;
        const BeamResponse response(beam, endForces.of(grid), {});
        grid[d] = 0.0;
        const Bar2Row values = bar2RowOf(beam, response, output);
        blocks[d].reset(columnNames);
        blocks[d].append(row, view.elementId(row), std::span<const double>(values.values.data(), values.count));
    }
    return true;
}

bool Bar2Element::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                    RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// Two rows per loaded Bar2, at its ends A (position 0) and B (position 1), in the station layout of a Bar2
// under BeamDistributedLoad; the forces include the loads between the ends.
void Bar2Element::recoverUnderLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                           ElementOutput output, const NodalDisplacements& displacements,
                                           ElementResultBlock& block) const
{
    if (outputColumnsUnderLineLoads(output).empty())
        throw std::invalid_argument(
            "Bar2: no output of this kind under line loads; ask outputColumnsUnderLineLoads()");
    block.reset(outputColumnsUnderLineLoads(output));
    const Columns columns(view);
    forEachLoadedElement(loads,
                         [&](std::int64_t row, std::span<const ElementLineLoad> elementLoads)
                         {
                             const BeamData beam = resolve(view, columns, row).required();
                             const std::vector<BeamLineLoad> resolved =
                                 resolveLineLoadsOrThrow(view, elementLoads, beam.geometry);
                             const BeamResponse response(beam, displacements, resolved,
                                                         beamThermalStrain(view, row, beam, displacements));
                             for (std::size_t end = 0; end < 2; ++end)
                             {
                                 const double position = static_cast<double>(end);
                                 const BeamForces forces = response.forcesAt(position * beam.geometry.length);
                                 const ResultPlace place{0, 1, position};
                                 if (output == ElementOutput::Forces)
                                 {
                                     const std::array<double, 6> values{forces.moment1, forces.moment2, forces.shear1,
                                                                        forces.shear2,  forces.axial,   forces.torque};
                                     block.append(row, view.elementId(row), place, values);
                                     continue;
                                 }
                                 const StationStresses stresses = stationStresses(beam.section.stations[end], forces);
                                 const double tension = tensionMargin(stresses.maximum, beam.material);
                                 const double compression = compressionMargin(stresses.minimum, beam.material);
                                 double margin = UNDEFINED_MARGIN;
                                 if (tension != UNDEFINED_MARGIN && compression != UNDEFINED_MARGIN)
                                     margin = std::min(tension, compression);
                                 else if (tension != UNDEFINED_MARGIN)
                                     margin = tension;
                                 else
                                     margin = compression;
                                 std::array<double, 8> values{
                                     stresses.bending[0], stresses.bending[1], stresses.bending[2], stresses.bending[3],
                                     stresses.axial,      stresses.maximum,    stresses.minimum,    margin};
                                 if (output == ElementOutput::Strains)
                                     for (std::size_t k = 0; k < 7; ++k)
                                         values[k] = uniaxialStrain(values[k], beam.material);
                                 block.append(row, view.elementId(row), place, values);
                             }
                         });
}

} // namespace oos::elements
