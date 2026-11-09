#include "oos/elements/shells/HigherOrderShell.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/shells/ShellRecoverySlopes.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::elements
{

namespace
{

// O_o_Reader gives MCID −1 when the field holds THETA.
constexpr std::int32_t NO_MAT_FRAME_ID = -1;
// An MCID whose x axis projects on the element plane to less than this has no direction there.
constexpr double MATERIAL_FRAME_MIN_PROJECTION = 1.0e-8;
// The director of a flat element's nodes: its normal n.
constexpr base::Vec3 ELEMENT_NORMAL{0.0, 0.0, 1.0};

// The directors of the element of batch row `row` with geometry `geometry`, in its element frame: at each node the
// director its grid point shares (HigherOrderShellDirectors), turned to the element's own normal there, or that normal.
template <std::size_t N>
[[nodiscard]] std::array<base::Vec3, N> directorsOf(const ElementBatchView& view,
                                                    const HigherOrderShellColumns& columns, std::int64_t row,
                                                    const HigherOrderShellGeometry<N>& geometry)
{
    std::array<base::Vec3, N> directors = geometry.ownDirector;
    for (std::size_t i = 0; i < N; ++i)
    {
        const std::optional<base::Vec3> shared = columns.directors->sharedDirector(view.card(), row, i);
        if (!shared)
            continue;
        const base::Mat3& e = geometry.axes;
        const base::Vec3& v = *shared;
        const base::Vec3 local{v[0] * e(0, 0) + v[1] * e(1, 0) + v[2] * e(2, 0),
                               v[0] * e(0, 1) + v[1] * e(1, 1) + v[2] * e(2, 1),
                               v[0] * e(0, 2) + v[1] * e(1, 2) + v[2] * e(2, 2)};
        directors[i] = base::dot(local, geometry.ownDirector[i]) >= 0.0 ? local : -local;
    }
    return directors;
}

} // namespace

HigherOrderShellColumns::HigherOrderShellColumns(const ElementBatchView& view)
    : connectivity(view),
      sections(view),
      theta(view.columns().values<double>("materialAngle")),
      matFrameId(view.columns().values<std::int32_t>("matFrameId")),
      offset(view.columns().values<double>("offset")),
      thicknessFlag(view.columns().values<std::int32_t>("thicknessFlag")),
      cornerThickness(view.columns().fixedLists<double>("cornerThickness")),
      directors(&higherOrderShellDirectors(view.index()))
{
}

template <typename Shape>
Resolution<HigherOrderShellData<Shape>> resolveHigherOrderShell(const ElementBatchView& view,
                                                                const HigherOrderShellColumns& columns,
                                                                std::int64_t row)
{
    using Data = HigherOrderShellData<Shape>;
    using Result = Resolution<Data>;
    constexpr std::size_t N = Shape::NODES;
    constexpr std::size_t C = Shape::CORNERS;
    const auto r = static_cast<std::size_t>(row);
    Resolution<HigherOrderShellGeometry<N>> geometry =
        resolveHigherOrderShellGeometry<Shape>(view, columns.connectivity, row);
    if (!geometry)
        return Result::failure(std::move(geometry.problem));
    Resolution<ShellSection> section = resolveShellSection(view, columns.sections, row);
    if (!section)
        return Result::failure(std::move(section.problem));
    const ShellCorners<N>& nodes = geometry.value->nodes;
    Data data;
    data.axes = geometry.value->axes;
    data.gridIds = nodes.ids;
    data.isCurved = geometry.value->isCurved;
    data.position = geometry.value->position;
    data.director = directorsOf<N>(view, columns, row, *geometry.value);
    Resolution<std::array<double, C>> thickness =
        resolveCornerThicknesses<C>(view, row, *section.value, columns.thicknessFlag[r], columns.cornerThickness[row]);
    if (!thickness)
        return Result::failure(std::move(thickness.problem));
    data.thickness = *thickness.value;

    if (columns.matFrameId[r] == NO_MAT_FRAME_ID)
    {
        data.materialAngle = geometry.value->edgeAngle + columns.theta[r] * std::numbers::pi / 180.0;
    }
    else
    {
        Resolution<input::CoordinateFrame> system =
            resolveRectangularFrame(view, row, columns.matFrameId[r], "matFrameId");
        if (!system)
            return Result::failure(std::move(system.problem));
        const base::Mat3& e = data.axes;
        const base::Mat3& axes = system.value->axes;
        const double alongX = axes(0, 0) * e(0, 0) + axes(1, 0) * e(1, 0) + axes(2, 0) * e(2, 0);
        const double alongY = axes(0, 0) * e(0, 1) + axes(1, 0) * e(1, 1) + axes(2, 0) * e(2, 1);
        if (std::hypot(alongX, alongY) <= MATERIAL_FRAME_MIN_PROJECTION)
            return Result::failure(elementProblem(
                view, row, problem_code::DEGENERATE_GEOMETRY, "matFrameId",
                base::concatenate({"expected the x axis of coordinate system ", std::to_string(columns.matFrameId[r]),
                                   " to cross the element plane at an angle, found it normal to the element; give "
                                   "another ShellQuad4.matFrameId or a ShellQuad4.materialAngle"})));
        data.materialAngle = std::atan2(alongY, alongX);
    }
    data.section = section.value->rotatedToElementFrame(data.materialAngle);
    if (std::optional<std::string> problem = offsetSectionProblem(data.section, columns.offset[r]))
        return Result::failure(elementProblem(view, row, problem_code::INVALID_FIELD, "offset", std::move(*problem)));
    if (std::optional<std::string> problem = offsetDefinitionProblem(view.snapshot(), columns.offset[r]))
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, "offset", std::move(*problem)));
    // ZOFFS along each node's director: n for all (ShellNodeFrames::build, the bits of a flat element), or ν_i, the
    // reference surface moving with it.
    const double offset = columns.offset[r];
    const bool isEveryDirectorNormal = std::all_of(data.director.begin(), data.director.end(),
                                                   [](const base::Vec3& d) { return d == ELEMENT_NORMAL; });
    if (isEveryDirectorNormal)
    {
        data.frames = ShellNodeFrames<N>::build(nodes.rows, nodes.cdFrames, data.axes, offset);
        return Result::success(std::move(data));
    }
    std::array<base::Vec3, N> offsets{};
    for (std::size_t i = 0; i < N; ++i)
    {
        offsets[i] = offset * data.director[i];
        data.position[i] = data.position[i] + offsets[i];
    }
    data.frames = ShellNodeFrames<N>::buildWithNodeOffsets(nodes.rows, nodes.cdFrames, data.axes, offsets);
    return Result::success(std::move(data));
}

template <typename Shape>
void checkHigherOrderShells(const ElementBatchView& view, base::DiagnosticSink& sink)
{
    const HigherOrderShellColumns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolveHigherOrderShell<Shape>(view, columns, row); });
}

template <typename Shape>
void appendHigherOrderShellStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block)
{
    const HigherOrderShellColumns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
        appendMatrix(block, row, data.frames.dofs(), data.frames.toDisplacementSystems(data.shell().stiffness()));
    }
}

template <typename Shape>
void appendHigherOrderShellMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                ElementMatrixBlock& block)
{
    const HigherOrderShellColumns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.massToDisplacementSystems(data.shell().mass(options.formulation, added),
                                                           view.isMassOnOffsets()));
    }
}

template <typename Shape>
void appendHigherOrderShellPressures(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                     ElementVectorBlock& block)
{
    const HigherOrderShellColumns columns(view);
    for (const ElementPressure& pressure : pressures)
    {
        const HigherOrderShellData<Shape> data =
            resolveHigherOrderShell<Shape>(view, columns, pressure.elementRow).required();
        const PressureDirection direction = pressureDirectionOf(view, pressure, data.axes);
        if (!direction.problem.empty())
            throw std::invalid_argument(base::concatenate(
                {view.card(), " ", std::to_string(view.elementId(pressure.elementRow)), ": ", direction.problem,
                 " (the loads facet was called on a pressure checkPressureLoads() refuses: a bug of O_o_Solver, report "
                 "it)"}));
        std::array<double, Shape::CORNERS> corners{};
        for (std::size_t i = 0; i < Shape::CORNERS; ++i)
            corners[i] = pressure.cornerPressures[i];
        block.append(pressure.elementRow, data.frames.dofs(),
                     data.frames.loadsToDisplacementSystems(
                         data.shell().pressureLoads(corners, direction.inElementFrame), view.areLoadsOnOffsets()));
    }
}

template <typename Shape>
void checkHigherOrderShellThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                       base::DiagnosticSink& sink)
{
    const HigherOrderShellColumns columns(view);
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        Resolution<HigherOrderShellData<Shape>> data = resolveHigherOrderShell<Shape>(view, columns, row);
        if (!data)
            continue; // check() reports it
        Resolution<ShellTemperatureChange> change =
            resolveShellTemperature(view, row, temperatures, data.value->frames.gridRows, data.value->section);
        if (!change)
            sink.report(std::move(change.problem));
    }
}

template <typename Shape>
void appendHigherOrderShellThermalLoads(const ElementBatchView& view, ElementRange range,
                                        const ElementTemperatures& temperatures, ElementVectorBlock& block)
{
    const HigherOrderShellColumns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
        const ShellTemperatureChange change =
            resolveShellTemperature(view, row, temperatures, data.frames.gridRows, data.section).required();
        if (change.isZero())
            continue;
        block.append(
            row, data.frames.dofs(),
            data.frames.loadsToDisplacementSystems(data.shell().thermalLoads(change), view.areLoadsOnOffsets()));
    }
}

namespace
{

template <typename Shape>
[[nodiscard]] ShellTemperatureChange temperatureOf(const ElementBatchView& view, std::int64_t row,
                                                   const HigherOrderShellData<Shape>& data,
                                                   const NodalDisplacements& displacements)
{
    if (displacements.temperatures == nullptr)
        return {};
    return resolveShellTemperature(view, row, *displacements.temperatures, data.frames.gridRows, data.section)
        .required();
}

} // namespace

// The membrane resultants of the preload at the centre (MindlinShell::placeStates, the thermal resultant
// subtracted), then MindlinShell::differentialStiffness, carried to the displacement systems through ZOFFS.
template <typename Shape>
void appendHigherOrderShellDifferential(const ElementBatchView& view, ElementRange range,
                                        const NodalDisplacements& preload, ElementMatrixBlock& block)
{
    const HigherOrderShellColumns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
        const MindlinShell<Shape> shell = data.shell();
        const auto q = data.frames.localDisplacements(preload);
        const ShellPlaceState centre = shell.placeStates(q, data.gridIds, temperatureOf(view, row, data, preload))[0];
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.toDisplacementSystems(shell.differentialStiffness(centre.membraneForce)));
    }
}

template <typename Shape>
void recoverHigherOrderShells(const ElementBatchView& view, ElementRange range, ElementOutput output,
                              const NodalDisplacements& displacements, ElementResultBlock& block)
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    recoverHigherOrderShellOutputs<Shape>(view, range, outputs, displacements,
                                          std::span<ElementResultBlock>(&block, 1));
}

template <typename Shape>
void recoverHigherOrderShellLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                   const NodalDisplacements& displacements, ElementResultBlock& block)
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::ofLayers(output)};
    recoverHigherOrderShellOutputs<Shape>(view, range, outputs, displacements,
                                          std::span<ElementResultBlock>(&block, 1));
}

// Every output from one resolution, one shell and one set of place states per element, as the ShellQuad4's.
template <typename Shape>
void recoverHigherOrderShellOutputs(const ElementBatchView& view, ElementRange range,
                                    std::span<const RecoveryOutput> outputs, const NodalDisplacements& displacements,
                                    std::span<ElementResultBlock> blocks)
{
    const bool hasElementOutput = resetShellRecoveryBlocks(outputs, blocks);
    const HigherOrderShellColumns columns(view);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
        if (!hasElementOutput && !data.section.isLaminate())
            continue;
        const auto q = data.frames.localDisplacements(displacements);
        const auto states = data.shell().placeStates(q, data.gridIds, temperatureOf(view, row, data, displacements));
        for (std::size_t k = 0; k < outputs.size(); ++k)
            appendShellOutputRows(blocks[k], row, view.elementId(row), states, data.section, outputs[k], composite,
                                  ShellRowValues::All);
    }
}

// PERFORMANCE: the slopes of the higher-order shells' rows from one shell and its place states per unit displacement
// (1 + 6N place states for every output asked), where recoveries at unit displacements resolved the element and built
// its shell again for each DOF and output (49 recoveries per output of a ShellQuad8).
// Measured: the recovery of ShellQuad8 and ShellTria6 plates.
template <typename Shape>
void higherOrderShellSlopes(const ElementBatchView& view, std::int64_t row, std::span<const RecoveryOutput> outputs,
                            std::span<RecoverySlopes> slopes)
{
    std::vector<std::vector<ColumnForm>> forms; // bounded by the outputs
    std::vector<std::vector<std::string>> columns;
    for (const RecoveryOutput& output : outputs)
    {
        forms.push_back(shellRecoveryOutputForms(output));
        columns.push_back(shellRecoveryOutputColumns(output));
    }
    const HigherOrderShellColumns snapshotColumns(view);
    const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, snapshotColumns, row).required();
    const MindlinShell<Shape> shell = data.shell();
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    // The place states' parts that do not depend on the displacements, once for every unit displacement.
    const typename MindlinShell<Shape>::PlaceOperators operators = shell.placeOperators();
    const typename MindlinShell<Shape>::PlaceThermal atRest = shell.placeThermal(operators, {});
    shellRecoverySlopesOfOutputs<Shape::NODES>(
        data.frames, forms, columns,
        [&](const std::array<double, 6 * Shape::NODES>& q, std::span<ElementResultBlock> blocks)
        {
            const auto states = shell.placeStates(operators, atRest, q, data.gridIds);
            for (std::size_t k = 0; k < outputs.size(); ++k)
                appendShellOutputRows(blocks[k], row, elementId, states, data.section, outputs[k], composite);
        },
        slopes);
}

template <typename Shape>
void completeHigherOrderShellFailureRows(const ElementBatchView& view, std::span<const ElementResultBlock> sources,
                                         std::span<ElementResultBlock> blocks)
{
    const HigherOrderShellColumns columns(view);
    completeLaminateFailureRows(sources, blocks, [&](std::int64_t row, const auto& visit)
                                { visit(resolveHigherOrderShell<Shape>(view, columns, row).required().section); });
}

namespace
{

// The section rate the Mindlin shell takes for `field`: a AnisoMat2D term's (shellSectionRate), or a ply thickness's
// rates of the products (LaminateSectionRate::ofProducts: MindlinShell is linear in t·G1, t³/12·G2, t²·B/t², t·κG and
// t·G at the corner thickness T of a laminate) when no direction is rigid in transverse shear (the shell then takes
// TS/T·G2_33·t, not a product); `laminate` gets the laminate's rate of a ply thickness.
template <typename Shape>
[[nodiscard]] std::optional<ShellSectionRate> mindlinSectionRate(const HigherOrderShellData<Shape>& data,
                                                                 const DesignedField& field,
                                                                 LaminateRate* laminate = nullptr)
{
    if (isAnisoMat2DStiffnessTerm(field))
        return shellSectionRate(data.section, field, data.materialAngle);
    if (!isPlyThickness(field) || !(data.section.transverseShearStiffness[0] > 0.0) ||
        !(data.section.transverseShearStiffness[1] > 0.0))
        return std::nullopt;
    std::optional<LaminateSectionRate> rate = laminateSectionRate(data.section, field, data.materialAngle);
    if (!rate)
        return std::nullopt;
    if (laminate != nullptr)
        *laminate = std::move(rate->laminate);
    return rate->ofProducts;
}

} // namespace

template <typename Shape>
bool higherOrderShellStiffnessRate(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   ElementMatrixBlock& block)
{
    if (!isAnisoMat2DStiffnessTerm(field) && !isPlyThickness(field))
        return false;
    const HigherOrderShellColumns columns(view);
    const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
    const std::optional<ShellSectionRate> rate = mindlinSectionRate(data, field);
    if (!rate)
        return false;
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(data.shell().stiffnessSectionRate(*rate)));
    return true;
}

template <typename Shape>
bool higherOrderShellDifferentialRate(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block)
{
    if ((!isAnisoMat2DStiffnessTerm(field) && !isPlyThickness(field)) || preload.temperatures != nullptr)
        return false;
    const HigherOrderShellColumns columns(view);
    const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
    const std::optional<ShellSectionRate> rate = mindlinSectionRate(data, field);
    if (!rate)
        return false;
    const MindlinShell<Shape> shell = data.shell();
    const auto q = data.frames.localDisplacements(preload);
    const auto states = shell.placeStates(q, data.gridIds);
    const ShellPlaceState centre = shell.placeStateSectionRates(states, q, *rate)[0];
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(shell.differentialStiffness(centre.membraneForce)));
    return true;
}

template <typename Shape>
bool higherOrderShellRecoveryRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   ElementOutput output, const NodalDisplacements& displacements,
                                   ElementResultBlock& rates)
{
    // A ply thickness: the forces only (a laminate's stresses and strains are its plies', higherOrderShellLayerRates).
    if (isPlyThickness(field) ? output != ElementOutput::Forces : !isAnisoMat2DStiffnessTerm(field))
        return false;
    if (displacements.temperatures != nullptr)
        return false;
    const HigherOrderShellColumns columns(view);
    const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
    const std::optional<ShellSectionRate> rate = mindlinSectionRate(data, field);
    if (!rate)
        return false;
    const MindlinShell<Shape> shell = data.shell();
    const auto q = data.frames.localDisplacements(displacements);
    const auto states = shell.placeStates(q, data.gridIds);
    const auto stateRates = shell.placeStateSectionRates(states, q, *rate);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    rates.reset(shellOutputColumns(output));
    for (std::size_t place = 0; place < states.size(); ++place)
        appendShellRowRates(rates, row, view.elementId(row), states[place], stateRates[place], data.section, output,
                            composite);
    return true;
}

template <typename Shape>
bool higherOrderShellLayerRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                LayerOutput output, const NodalDisplacements& displacements, ElementResultBlock& rates)
{
    if (!isPlyThickness(field) || displacements.temperatures != nullptr)
        return false;
    const HigherOrderShellColumns columns(view);
    const HigherOrderShellData<Shape> data = resolveHigherOrderShell<Shape>(view, columns, row).required();
    LaminateRate laminate;
    const std::optional<ShellSectionRate> rate = mindlinSectionRate(data, field, &laminate);
    if (!rate)
        return false;
    const MindlinShell<Shape> shell = data.shell();
    const auto q = data.frames.localDisplacements(displacements);
    const auto states = shell.placeStates(q, data.gridIds);
    const auto stateRates = shell.placeStateSectionRates(states, q, *rate);
    rates.reset(laminateOutputColumns(output));
    appendLaminateRowRates(rates, row, view.elementId(row), states[0], stateRates[0], data.section, laminate, output,
                           compositeOutputOf(view.snapshot()));
    return true;
}

#define OOS_HIGHER_ORDER_SHELL(Shape)                                                                                  \
    template Resolution<HigherOrderShellData<Shape>> resolveHigherOrderShell<Shape>(                                   \
        const ElementBatchView&, const HigherOrderShellColumns&, std::int64_t);                                        \
    template void checkHigherOrderShells<Shape>(const ElementBatchView&, base::DiagnosticSink&);                       \
    template void appendHigherOrderShellStiffness<Shape>(const ElementBatchView&, ElementRange, ElementMatrixBlock&);  \
    template void appendHigherOrderShellMass<Shape>(const ElementBatchView&, ElementRange, const MassOptions&,         \
                                                    ElementMatrixBlock&);                                              \
    template void appendHigherOrderShellPressures<Shape>(const ElementBatchView&, std::span<const ElementPressure>,    \
                                                         ElementVectorBlock&);                                         \
    template void checkHigherOrderShellThermalLoads<Shape>(const ElementBatchView&, const ElementTemperatures&,        \
                                                           base::DiagnosticSink&);                                     \
    template void appendHigherOrderShellThermalLoads<Shape>(const ElementBatchView&, ElementRange,                     \
                                                            const ElementTemperatures&, ElementVectorBlock&);          \
    template void appendHigherOrderShellDifferential<Shape>(const ElementBatchView&, ElementRange,                     \
                                                            const NodalDisplacements&, ElementMatrixBlock&);           \
    template void recoverHigherOrderShells<Shape>(const ElementBatchView&, ElementRange, ElementOutput,                \
                                                  const NodalDisplacements&, ElementResultBlock&);                     \
    template void recoverHigherOrderShellLayers<Shape>(const ElementBatchView&, ElementRange, LayerOutput,             \
                                                       const NodalDisplacements&, ElementResultBlock&);                \
    template void recoverHigherOrderShellOutputs<Shape>(const ElementBatchView&, ElementRange,                         \
                                                        std::span<const RecoveryOutput>, const NodalDisplacements&,    \
                                                        std::span<ElementResultBlock>);                                \
    template void higherOrderShellSlopes<Shape>(const ElementBatchView&, std::int64_t,                                 \
                                                std::span<const RecoveryOutput>, std::span<RecoverySlopes>);           \
    template void completeHigherOrderShellFailureRows<Shape>(                                                          \
        const ElementBatchView&, std::span<const ElementResultBlock>, std::span<ElementResultBlock>);                  \
    template bool higherOrderShellStiffnessRate<Shape>(const ElementBatchView&, std::int64_t, const DesignedField&,    \
                                                       ElementMatrixBlock&);                                           \
    template bool higherOrderShellDifferentialRate<Shape>(const ElementBatchView&, std::int64_t, const DesignedField&, \
                                                          const NodalDisplacements&, ElementMatrixBlock&);             \
    template bool higherOrderShellRecoveryRates<Shape>(const ElementBatchView&, std::int64_t, const DesignedField&,    \
                                                       ElementOutput, const NodalDisplacements&, ElementResultBlock&); \
    template bool higherOrderShellLayerRates<Shape>(const ElementBatchView&, std::int64_t, const DesignedField&,       \
                                                    LayerOutput, const NodalDisplacements&, ElementResultBlock&);

OOS_HIGHER_ORDER_SHELL(SerendipityQuadShape)
OOS_HIGHER_ORDER_SHELL(QuadraticTriangleShape)

#undef OOS_HIGHER_ORDER_SHELL

} // namespace oos::elements
