#include "oos/elements/mixedHybrid/families/MixedHybridSolidFamily.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"
#include "oos/elements/mixedHybrid/families/LocalMap.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/solids/SolidKinematics.hpp"
#include "oos/elements/solids/SolidRecovery.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The model's solid of a batch row, which check() accepted (a row it refused is a bug of the caller).
const MixedSolid& requiredSolid(const MixedHybridModel& model, const ElementBatchView& view, std::int64_t row)
{
    const MixedSolid* solid = model.solidOf(view.card(), row);
    if (solid == nullptr)
        throw std::logic_error(
            base::concatenate({"MixedHybridSolidFamily: element ", std::to_string(view.elementId(row)), " of ",
                               view.card(), " was not accepted by check() (a bug of O_o_Solver: report it)"}));
    return *solid;
}

// The kind of analysis of a view: its layout's (the interior fields of a modal analysis).
EntityLayout::Kind kindOf(const ElementBatchView& view)
{
    return view.entities()->kind();
}

// The kernel of a solid without loads (kept by the SolidPart).
SolidPart::SharedKernel kernelOf(const MixedHybridModel& model, const MixedSolid& solid, EntityLayout::Kind kind,
                                 bool withMass)
{
    std::variant<SolidPart::SharedKernel, std::string> kernel =
        model.solidPart()->kernelOf(solid.partIndex, kind, withMass);
    if (const std::string* message = std::get_if<std::string>(&kernel))
        throw std::logic_error(base::concatenate({"MixedHybridSolidFamily: element ", std::to_string(solid.elementId),
                                                  ": ", *message, " (check() accepted it: a bug of O_o_Solver)"}));
    return std::get<SolidPart::SharedKernel>(std::move(kernel));
}

// The map of the local unknowns of a solid to its DOFs.
LocalMap solidMapOf(const ElementBatchView& view, const MixedHybridModel& model, const MixedSolid& solid,
                    EntityLayout::Kind kind)
{
    return localMapOf(view, model.solidDofs(solid, kind));
}

// The temperature change of each corner of a solid from the grid point temperatures of the load set less the initial
// ones (or TREF of the material), as the card's other families read them; zero without a temperature load (a DEFORM
// alone).
Resolution<std::vector<double>> cornerChangesOf(const ElementBatchView& view, std::int64_t row, const MixedSolid& solid,
                                                const ElementTemperatures& temperatures)
{
    using Result = Resolution<std::vector<double>>;
    std::vector<double> changes(solid.cornerRows.size(), 0.0);
    if (!temperatures.hasTemperatureLoad)
        return Result::success(changes);
    const double reference = solid.classicSection.referenceTemperature;
    for (std::size_t a = 0; a < solid.cornerRows.size(); ++a)
    {
        const std::int32_t node = solid.cornerRows[a];
        const std::optional<double> load = temperatures.load.atGridRow(node);
        if (!load)
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD, "TemperatureSelection(LOAD)",
                base::concatenate({"expected a temperature at grid point ",
                                   std::to_string(view.snapshot().nodes().ids()[static_cast<std::size_t>(node)]),
                                   " of the solid in the thermal load set, found none (no NodeTemperatureLoad, no "
                                   "DefaultTemperatureLoad); give the grid point a NodeTemperatureLoad or the set a "
                                   "DefaultTemperatureLoad"})));
        const double initial =
            temperatures.initial.isEmpty() ? reference : temperatures.initial.atGridRow(node).value_or(reference);
        changes[a] = *load - initial;
    }
    return Result::success(changes);
}

} // namespace

MixedHybridSolidFamily::MixedHybridSolidFamily(std::unique_ptr<ElementFamily> classic) : m_classic(std::move(classic))
{
}

void MixedHybridSolidFamily::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    if (view.rowCount() == 0)
        return;
    if (view.entities() == nullptr)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "formulation-layout-missing";
        diagnostic.card = view.card();
        diagnostic.message = base::concatenate(
            {"expected the ", view.card(), " elements of ", MIXED_HYBRID_FORMULATION,
             " to be computed with the entities of the model's edges and faces, found an element view without them "
             "(a path that does not carry the formulation yet: run the model on the full path)"});
        sink.report(std::move(diagnostic));
        return;
    }
    for (base::Diagnostic& problem : MixedHybridModel::of(view.index()).elementProblems(view.card()))
        sink.report(std::move(problem));
}

void MixedHybridSolidFamily::computeStiffness(const ElementBatchView& view, ElementRange range,
                                              ElementMatrixBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const EntityLayout::Kind kind = kindOf(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedSolid& solid = requiredSolid(model, view, row);
        const SolidPart::SharedKernel kernel = kernelOf(model, solid, kind, false);
        const LocalMap map = solidMapOf(view, model, solid, kind);
        block.append(row, map.dofs, congruent(model.solidPart()->localStiffness(solid.partIndex, *kernel, kind), map));
    }
}

void MixedHybridSolidFamily::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                         ElementMatrixBlock& block) const
{
    // The consistent mass whatever options.formulation asks (spec section 8); the NSM cards name no solid property.
    static_cast<void>(options);
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const EntityLayout::Kind kind = kindOf(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedSolid& solid = requiredSolid(model, view, row);
        const SolidPart::SharedKernel kernel = kernelOf(model, solid, kind, true);
        const LocalMap map = solidMapOf(view, model, solid, kind);
        block.append(row, map.dofs, congruent(model.solidPart()->localMass(solid.partIndex, *kernel, kind), map));
    }
}

void MixedHybridSolidFamily::checkPressureLoads(const ElementBatchView& view,
                                                std::span<const ElementPressure> pressures,
                                                base::DiagnosticSink& sink) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (const ElementPressure& pressure : pressures)
    {
        const auto refuse = [&](std::string field, std::string message)
        {
            base::Diagnostic diagnostic;
            diagnostic.code = "unsupported-load";
            diagnostic.card = std::string(pressure.card);
            diagnostic.id = pressure.setId;
            diagnostic.field = std::move(field);
            diagnostic.message = std::move(message);
            sink.report(std::move(diagnostic));
        };
        const MixedSolid* solid = model.solidOf(view.card(), pressure.elementRow);
        if (solid == nullptr)
            continue; // the element's own problem is reported by check()
        std::vector<std::size_t> grids;
        for (const std::int32_t id : pressure.faceNodes)
            if (const std::optional<std::int32_t> row = view.index().points().findGrid(id))
                if (const std::int32_t grid = model.gridIndexOfRow(*row); grid >= 0)
                    grids.push_back(static_cast<std::size_t>(grid));
        if (grids.size() < 3 || grids.size() != pressure.faceNodes.size() ||
            !model.solidPart()->localFaceOf(solid->partIndex, grids))
        {
            refuse("faceNodes", base::concatenate({"expected the pressure on a face of ", view.card(), " ",
                                                   std::to_string(solid->elementId),
                                                   ", found grid points that are not the corners of one of its faces; "
                                                   "correct G1 and G3 or G4 of the card"}));
            continue;
        }
        if (pressure.hasDirection)
        {
            const PressureDirection direction = pressureDirectionOf(view, pressure, base::Mat3::identity());
            if (!direction.inElementFrame)
                refuse("cid", direction.problem);
            for (std::size_t k = 1; k < pressure.faceNodes.size(); ++k)
                if (pressure.cornerPressures[k] != pressure.cornerPressures[0])
                {
                    refuse("p2", base::concatenate({"expected the same pressure at every corner of the face of a "
                                                    "pressure along a direction on a solid of ",
                                                    MIXED_HYBRID_FORMULATION, ", found different ones on element ",
                                                    std::to_string(solid->elementId),
                                                    "; give one pressure or leave the direction out"}));
                    break;
                }
        }
    }
}

void MixedHybridSolidFamily::computePressureLoads(const ElementBatchView& view,
                                                  std::span<const ElementPressure> pressures,
                                                  ElementVectorBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const SolidPart& part = *model.solidPart();
    const EntityLayout::Kind kind = kindOf(view);
    for (const ElementPressure& pressure : pressures)
    {
        const MixedSolid& solid = requiredSolid(model, view, pressure.elementRow);
        std::vector<std::size_t> grids;
        for (const std::int32_t id : pressure.faceNodes)
            grids.push_back(static_cast<std::size_t>(model.gridIndexOfRow(*view.index().points().findGrid(id))));
        const std::size_t face = *part.localFaceOf(solid.partIndex, grids);
        const SolidPart::SharedKernel kernel = kernelOf(model, solid, kind, false);
        Vector onParameters;
        if (!pressure.hasDirection)
            onParameters =
                part.faceCornerPressureLoad(*kernel, solid.partIndex, face, grids,
                                            std::span<const double>(pressure.cornerPressures.data(), grids.size()));
        else
        {
            // The unit direction in the basic system (the element frame of pressureDirectionOf taken as the basic one).
            const base::Vec3 unit = *pressureDirectionOf(view, pressure, base::Mat3::identity()).inElementFrame;
            const double p = pressure.cornerPressures[0];
            onParameters = part.faceTractionLoad(
                *kernel, face, [&](const SolidPoint&) { return SolidPoint{p * unit[0], p * unit[1], p * unit[2]}; });
        }
        const LocalMap map = solidMapOf(view, model, solid, kind);
        Vector local(map.toLocal.rows(), 0.0);
        for (std::size_t i = 0; i < onParameters.size(); ++i)
            local[i] = onParameters[i];
        block.append(pressure.elementRow, map.dofs, onDofs(local, map));
    }
}

void MixedHybridSolidFamily::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                               base::DiagnosticSink& sink) const
{
    if (view.rowCount() == 0 || !temperatures.hasTemperatureLoad)
        return;
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        const MixedSolid* solid = model.solidOf(view.card(), row);
        if (solid == nullptr)
            continue;
        Resolution<std::vector<double>> changes = cornerChangesOf(view, row, *solid, temperatures);
        if (!changes)
        {
            sink.report(std::move(changes.problem));
            continue;
        }
        if (!solid->isotropicExpansion && MixedHybridModel::takesThermalStrain(*solid, *changes.value))
        {
            base::Diagnostic diagnostic;
            diagnostic.code = "unsupported-load";
            diagnostic.card = std::string(view.card());
            diagnostic.id = solid->elementId;
            diagnostic.field = "TemperatureSelection(LOAD)";
            diagnostic.message = base::concatenate(
                {"expected the expansion of a solid of ", MIXED_HYBRID_FORMULATION,
                 " to be the same along every direction under a temperature load, found different coefficients on "
                 "element ",
                 std::to_string(solid->elementId), " (AnisoMat3D A1-A6); use ClassicFormulation for it"});
            sink.report(std::move(diagnostic));
        }
    }
}

void MixedHybridSolidFamily::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                                 const ElementTemperatures& temperatures,
                                                 ElementVectorBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const EntityLayout::Kind kind = kindOf(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedSolid& solid = requiredSolid(model, view, row);
        const std::vector<double> changes = model.solidCornerChanges(solid, &temperatures);
        if (!MixedHybridModel::takesThermalStrain(solid, changes))
            continue;
        const VolumeKernel kernel = model.solidThermalKernel(solid, kind, changes);
        const Vector& onParameters = kernel.loads().front().force;
        const LocalMap map = solidMapOf(view, model, solid, kind);
        Vector local(map.toLocal.rows(), 0.0);
        for (std::size_t i = 0; i < onParameters.size(); ++i)
            local[i] = onParameters[i];
        block.append(row, map.dofs, onDofs(local, map));
    }
}

void MixedHybridSolidFamily::checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    static_cast<void>(view);
    static_cast<void>(sink);
}

void MixedHybridSolidFamily::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                          const NodalDisplacements& preload,
                                                          ElementMatrixBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const EntityLayout::Kind kind = kindOf(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedSolid& solid = requiredSolid(model, view, row);
        const MixedHybridModel::SolidCaseStresses stresses = model.solidStresses(solid, kind, preload);
        SolidResult<Matrix> geometric = SolidPart::geometricStiffness(*stresses.kernel, stresses.beta);
        if (!geometric.ok())
            throw std::logic_error(base::concatenate(
                {"MixedHybridSolidFamily: element ", std::to_string(solid.elementId), ": ", geometric.refusal()}));
        // On the parameters: the interior fields of a modal layout take none.
        const LocalMap map = solidMapOf(view, model, solid, kind);
        Matrix local(map.toLocal.rows(), map.toLocal.rows());
        local.setBlock(0, 0, geometric.value());
        block.append(row, map.dofs, congruent(local, map));
    }
}

void MixedHybridSolidFamily::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                     const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    block.reset(outputColumns(output));
    if (block.columns().empty())
        return;
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const SolidPart& part = *model.solidPart();
    const std::span<const std::int32_t> ids = view.snapshot().nodes().ids();
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedSolid& solid = requiredSolid(model, view, row);
        const MixedHybridModel::SolidCaseStresses stresses = model.solidStresses(solid, kindOf(view), displacements);
        const SolidPointStresses points = SolidPart::pointStresses(*stresses.kernel, stresses.beta);
        // The places: the centre, then the corners; the temperature change there for the strains (the functions of
        // the corners at the centre of the card's other families).
        const SolidShape shape = solid.cornerRows.size() == 4   ? SolidShape::Tetra4
                                 : solid.cornerRows.size() == 5 ? SolidShape::Pyram5
                                 : solid.cornerRows.size() == 6 ? SolidShape::Penta6
                                                                : SolidShape::Hexa8;
        const Mat6 compliance =
            output == ElementOutput::Strains ? complianceOf(solid.classicSection.elasticity) : Mat6{};
        const auto appendPlace = [&](std::int32_t nodeId, const VoigtVector& basic, double change)
        {
            const VoigtVector stress = part.toMaterialAxes(solid.partIndex, basic);
            ResultPlace place;
            place.nodeId = nodeId;
            if (output == ElementOutput::Stresses)
            {
                block.append(row, view.elementId(row), place, solidStressRow(stress));
                return;
            }
            // ε = C σ + α ΔT on the direct strains (the total strain).
            Voigt6 strain{};
            for (std::size_t i = 0; i < 6; ++i)
            {
                for (std::size_t j = 0; j < 6; ++j)
                    strain[i] += compliance(i, j) * stress[j];
                if (i < 3)
                    strain[i] += solid.part.material.thermalExpansion * change;
            }
            block.append(row, view.elementId(row), place, solidStrainRow(strain));
        };
        double centreChange = 0.0;
        if (!stresses.cornerChanges.empty())
        {
            const SolidShapeValues centre = solidShapeAt(shape, solidCentre(shape));
            for (std::size_t a = 0; a < stresses.cornerChanges.size(); ++a)
                centreChange += centre.n[a] * stresses.cornerChanges[a];
        }
        appendPlace(0, points.centre, centreChange);
        for (std::size_t a = 0; a < points.corners.size(); ++a)
            appendPlace(ids[static_cast<std::size_t>(solid.cornerRows[a])], points.corners[a],
                        stresses.cornerChanges.empty() ? 0.0 : stresses.cornerChanges[a]);
    }
}

} // namespace oos::elements::mixedHybrid
