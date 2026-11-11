#include "oos/elements/bars_beams/BeamResponse.hpp"
#include "oos/elements/bars_beams/BeamThermal.hpp"
#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"
#include "oos/elements/core/MarginOfSafety.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace oos::elements
{

namespace
{

// The share of the largest stress of an element within which two stations tie for its margins of safety: far above the
// round-off of a stress (1e-15 of it, a few orders more through the section's terms) and far below a difference of
// stress that should decide the station (in practice equal stresses, or differences of 1e-3 and more).
constexpr double MARGIN_STATION_TIE = 1.0e-9;

// The station of the margins of an element (appendResponseRows' comment), from the larger of |SMAX| and |SMIN| of each
// station: the first within MARGIN_STATION_TIE of the largest, so that stations of equal stress in exact arithmetic (an
// unloaded span, whose ends carry the same stress) give the first station whatever the round-off; none (the count)
// when the element carries no stress.
std::size_t marginStationOf(std::span<const double> magnitudes)
{
    double largest = 0.0;
    for (const double magnitude : magnitudes)
        largest = std::max(largest, magnitude);
    for (std::size_t k = 0; k < magnitudes.size() && largest > 0.0; ++k)
        if (magnitudes[k] >= (1.0 - MARGIN_STATION_TIE) * largest)
            return k;
    return magnitudes.size();
}

// The vectorised recovery of the Beam2: the fewest cases of a block from which its rows are made from its slopes.
// Calibrated on a Beam2 grillage (45 300 Beam2, six threads): the per-case recovery takes about 0.093 s per case,
// the recovery by blocks 0.39 s once and 0.009 s per case: even from about 4.5 cases; 8, as the Bar2.
constexpr std::size_t BEAM_VECTORISED_RECOVERY_MIN_CASES = 8;

} // namespace

// The forms of the columns: the forces and the stresses or strains at the recovery points Linear; the
// extremes over the points and the margins Other.
std::vector<ColumnForm> Beam2Element::outputColumnForms(ElementOutput output) const
{
    const ColumnForm linear = ColumnForm::linear();
    const ColumnForm other = ColumnForm::other();
    switch (output)
    {
    case ElementOutput::Forces:
        return std::vector<ColumnForm>(7, linear);
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {linear, linear, linear, linear, other, other, other, other};
    default:
        return {};
    }
}

std::vector<std::string> Beam2Element::outputColumns(ElementOutput output) const
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"bendingMoment1", "bendingMoment2", "shear1", "shear2", "axialForce", "totalTorque", "warpingTorque"};
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {"sxc", "sxd", "sxe", "sxf", "smax", "smin", "msTension", "msCompression"};
    default:
        return {};
    }
}

// One row per station of output: end A, the intermediate stations whose SO is not NO, end B. The forces at a station
// come from the equilibrium of the part between end A and the station (BeamResponse); the stresses use the section of
// the station. Without warping the warping torque is 0 and the total torque is the Saint-Venant torque.
//
// The margins of safety (in tension and compression) of a Beam2 are written at one station only, the station of the
// larger stress, the larger of |SMAX| and |SMIN| (the first station on a tie), and are undefined at the others
// (UNDEFINED_MARGIN, blank in a printed report), and at every station when the element carries no stress.
void Beam2Element::appendStationRows(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                        std::span<const BeamLineLoad> loads, ElementOutput output,
                                        const NodalDisplacements& displacements, ElementResultBlock& block)
{
    appendResponseRows(view, row, beam,
                       BeamResponse(beam, displacements, loads, beamThermalStrain(view, row, beam, displacements)),
                       output, block);
}

void Beam2Element::appendResponseRows(const ElementBatchView& view, std::int64_t row, const BeamData& beam,
                                         const BeamResponse& response, ElementOutput output, ElementResultBlock& block)
{
    const BeamSection& section = beam.section;
    const input::NodeTable& nodes = view.snapshot().nodes();
    // The stress rows of the element, kept until the station of its margins is known.
    struct StressRow
    {
        ResultPlace place;
        std::array<double, 8> values{};
        double magnitude = 0.0; // max(|SMAX|, |SMIN|)
    };
    std::vector<StressRow> stressRows; // bounded by the stations of the element
    std::int32_t layer = 0;
    for (std::size_t s = 0; s < section.count; ++s)
    {
        const BeamStation& station = section.stations[s];
        if (!station.hasOutput)
            continue;
        ++layer;
        std::int32_t nodeId = 0;
        if (s == 0)
            nodeId = nodes.ids()[static_cast<std::size_t>(beam.geometry.rowA)];
        else if (s + 1 == section.count)
            nodeId = nodes.ids()[static_cast<std::size_t>(beam.geometry.rowB)];
        const ResultPlace place{nodeId, layer, station.position};
        const BeamForces forces = response.forcesAt(station.position * beam.geometry.length);
        if (output == ElementOutput::Forces)
        {
            const std::array<double, 7> values{
                forces.moment1, forces.moment2, forces.shear1, forces.shear2, forces.axial, forces.torque, 0.0};
            block.append(row, view.elementId(row), place, values);
            continue;
        }
        const StationStresses stresses = stationStresses(station, forces);
        std::array<double, 8> values{stresses.axial + stresses.bending[0],
                                     stresses.axial + stresses.bending[1],
                                     stresses.axial + stresses.bending[2],
                                     stresses.axial + stresses.bending[3],
                                     stresses.maximum,
                                     stresses.minimum,
                                     tensionMargin(stresses.maximum, beam.material),
                                     compressionMargin(stresses.minimum, beam.material)};
        if (output == ElementOutput::Strains)
            for (std::size_t k = 0; k < 6; ++k)
                values[k] = uniaxialStrain(values[k], beam.material);
        stressRows.push_back({place, values, std::max(std::abs(stresses.maximum), std::abs(stresses.minimum))});
    }
    std::vector<double> magnitudes; // bounded by the stations of the element
    for (const StressRow& stressRow : stressRows)
        magnitudes.push_back(stressRow.magnitude);
    const std::size_t marginStation = marginStationOf(magnitudes);
    for (std::size_t k = 0; k < stressRows.size(); ++k)
    {
        StressRow& stressRow = stressRows[k];
        if (k != marginStation)
        {
            stressRow.values[6] = UNDEFINED_MARGIN;
            stressRow.values[7] = UNDEFINED_MARGIN;
        }
        block.append(row, view.elementId(row), stressRow.place, stressRow.values);
    }
}

// The recovery matrix of a Beam2 (RecoverySlopes): its rows are linear in the
// displacements of its grid points in their Linear columns (no line load, no temperature), so the slope of DOF d is
// rows(e_d) − rows(0), the responses made from one end-force map (BeamEndForces), the same bits as recoveries at unit
// displacements (as the Bar2's); every output from one resolution, one map and one response per DOF.
// PERFORMANCE: the slopes a Beam2 gave by 13 recoveries at unit displacements per output (each resolving the element
// and building its tapered stiffness again) from one resolution for all its outputs.
// Measured: a Beam2 grillage of 45 300 elements.
bool Beam2Element::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
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
    // The rows of every output at rest, then at each unit displacement, one after the other in one block per output.
    std::vector<ElementResultBlock> blocks(outputs.size()); // bounded by the outputs
    for (std::size_t k = 0; k < outputs.size(); ++k)
        blocks[k].reset(outputColumns(outputs[k].output));
    std::array<double, 12> grid{};
    const auto appendRows = [&]
    {
        const BeamResponse response(beam, endForces.of(grid), {});
        for (std::size_t k = 0; k < outputs.size(); ++k)
            appendResponseRows(view, row, beam, response, outputs[k].output, blocks[k]);
    };
    appendRows();
    std::vector<std::size_t> rowCounts; // bounded by the outputs
    for (const ElementResultBlock& block : blocks)
        rowCounts.push_back(block.size());
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
        grid[d] = 1.0;
        appendRows();
        grid[d] = 0.0;
    }
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        const std::vector<ColumnForm> forms = outputColumnForms(outputs[k].output);
        const ElementResultBlock& block = blocks[k];
        const std::size_t rowCount = rowCounts[k];
        slopes[k].reset(dofs, forms.size());
        for (std::size_t r = 0; r < rowCount; ++r)
            slopes[k].appendRow(block.place(r));
        for (std::size_t d = 0; d < dofs.size(); ++d)
            for (std::size_t r = 0; r < rowCount; ++r)
                for (std::size_t c = 0; c < forms.size(); ++c)
                    if (forms[c].kind == ColumnForm::Kind::Linear)
                        slopes[k].slopes(r, c)[d] = block.values((d + 1) * rowCount + r)[c] - block.values(r)[c];
    }
    return true;
}

// The rows of recover() at each unit displacement from one end-force map, as the Bar2's: recover() makes the station
// rows of a Beam2 without temperatures from BeamEndForces(beam, {}, 0).of(grid) (appendStationRows), so the map
// built once gives the same bits for each unit displacement. PERFORMANCE: output::SectionLoads recovered a Beam2
// once per DOF (12), each resolving it and building its stiffness; here one resolution and one map (measured on a
// stiffened barrel panel).
bool Beam2Element::recoverAtUnitDisplacements(const ElementBatchView& view, std::int64_t row, ElementOutput output,
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
        blocks[d].reset(columnNames);
        appendResponseRows(view, row, beam, response, output, blocks[d]);
    }
    return true;
}

bool Beam2Element::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                     RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

std::size_t Beam2Element::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : BEAM_VECTORISED_RECOVERY_MIN_CASES;
}

// The extremes of each station, the largest and smallest of its stresses at C, D, E, F (each the axial stress plus a
// bending stress, so their extreme is the axial stress plus the extreme bending stress, as stationStresses gives it,
// rounding being monotonic), and the margins at the station of the larger stress only (appendResponseRows): the
// strains' extremes from the strains (ε = σ/E, monotonic), their margins and station from the stresses σ = E·ε (to
// round-off); the forces are all Linear.
void Beam2Element::completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                                            std::span<ElementResultBlock> blocks) const
{
    requireOutput(output);
    if (output == ElementOutput::Forces || blocks.empty())
        return;
    const Columns columns(view);
    const ElementResultBlock& structure = blocks.front();
    std::vector<double> magnitudes; // bounded by the stations of an element
    std::size_t first = 0;
    while (first < structure.size())
    {
        std::size_t last = first + 1;
        while (last < structure.size() && structure.elementRow(last) == structure.elementRow(first))
            ++last;
        const BeamData beam = resolve(view, columns, structure.elementRow(first)).required();
        const IsotropicMaterial& material = beam.material;
        const double toStress = output == ElementOutput::Strains ? material.youngModulus : 1.0;
        for (ElementResultBlock& block : blocks)
        {
            magnitudes.clear();
            for (std::size_t i = first; i < last; ++i)
            {
                const std::span<double> values = block.mutableValues(i);
                values[4] = std::max({values[0], values[1], values[2], values[3]});
                values[5] = std::min({values[0], values[1], values[2], values[3]});
                magnitudes.push_back(std::max(std::abs(toStress * values[4]), std::abs(toStress * values[5])));
            }
            const std::size_t marginStation = marginStationOf(magnitudes);
            for (std::size_t i = first; i < last; ++i)
            {
                const std::span<double> values = block.mutableValues(i);
                const bool isMarginStation = i - first == marginStation;
                values[6] = isMarginStation ? tensionMargin(toStress * values[4], material) : UNDEFINED_MARGIN;
                values[7] = isMarginStation ? compressionMargin(toStress * values[5], material) : UNDEFINED_MARGIN;
            }
        }
        first = last;
    }
}

void Beam2Element::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                              const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const BeamData beam = resolve(view, columns, row).required();
        const auto loads = resolveLineLoadsOrThrow(view, displacements.lineLoads, beam.geometry, row);
        appendStationRows(view, row, beam, loads, output, displacements, block);
    }
}

void Beam2Element::recoverUnderLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                            ElementOutput output, const NodalDisplacements& displacements,
                                            ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    forEachLoadedElement(loads,
                         [&](std::int64_t row, std::span<const ElementLineLoad> elementLoads)
                         {
                             const BeamData beam = resolve(view, columns, row).required();
                             const std::vector<BeamLineLoad> resolved =
                                 resolveLineLoadsOrThrow(view, elementLoads, beam.geometry);
                             appendStationRows(view, row, beam, resolved, output, displacements, block);
                         });
}

} // namespace oos::elements
