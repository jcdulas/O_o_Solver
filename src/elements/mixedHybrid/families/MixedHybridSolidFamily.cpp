#include "oos/elements/mixedHybrid/families/MixedHybridSolidFamily.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
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

// The kernel a call got, or the error of the solid it is refused for. check() builds the kernels the SolidPart keeps
// whole; in a larger model it checks the geometry of the others, and their matrices may refuse them here.
SolidPart::SharedKernel acceptedKernel(std::variant<SolidPart::SharedKernel, std::string> kernel,
                                       const MixedSolid& solid)
{
    if (const std::string* message = std::get_if<std::string>(&kernel))
        throw base::SolverError(base::concatenate(
            {"solid element ", std::to_string(solid.elementId), " of ", MIXED_HYBRID_FORMULATION, ": ", *message,
             "; its geometry or its material makes its matrices singular: check its grid points and its material"}));
    return std::get<SolidPart::SharedKernel>(std::move(kernel));
}

// What the facets ask of the kernel of a solid: its stiffness, its mass, or neither (its volume, its functions and
// the loads of its faces: the kernel as the SolidPart keeps it, with or without its stiffness).
constexpr SolidPart::SolidKernelRequest WITH_STIFFNESS{true, false};
constexpr SolidPart::SolidKernelRequest WITH_MASS{false, true};
constexpr SolidPart::SolidKernelRequest AS_KEPT{false, false};

// The kernel of a solid without loads, with what `request` asks.
SolidPart::SharedKernel kernelOf(const MixedHybridModel& model, const MixedSolid& solid, EntityLayout::Kind kind,
                                 const SolidPart::SolidKernelRequest& request)
{
    return acceptedKernel(model.solidPart()->kernelOf(solid.partIndex, kind, request), solid);
}

// The map of the local unknowns of a solid to its DOFs.
LocalMap solidMapOf(const ElementBatchView& view, const MixedHybridModel& model, const MixedSolid& solid,
                    EntityLayout::Kind kind)
{
    return localMapOf(view, model.solidDofs(solid, kind));
}

// Thermal loads require a temperature at every corner. Mid-side values are optional.
std::optional<base::Diagnostic> missingCornerTemperature(const ElementBatchView& view, std::int64_t row,
                                                         const MixedSolid& solid,
                                                         const ElementTemperatures& temperatures)
{
    if (temperatures.hasTemperatureLoad)
        for (const auto node : solid.cornerRows)
            if (!temperatures.load.atGridRow(node))
                return elementProblem(
                    view, row, problem_code::INVALID_FIELD, "TemperatureSelection(LOAD)",
                    base::concatenate(
                        {"expected a temperature at grid point ",
                         std::to_string(view.snapshot().nodes().ids()[static_cast<std::size_t>(node)]),
                         " of the solid in the thermal load set, found none (no NodeTemperatureLoad, no "
                         "DefaultTemperatureLoad); give the grid point a NodeTemperatureLoad or the set a "
                         "DefaultTemperatureLoad"}));
    return std::nullopt;
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
        const SolidPart::SharedKernel kernel = kernelOf(model, solid, kind, WITH_STIFFNESS);
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
        const SolidPart::SharedKernel kernel = kernelOf(model, solid, kind, WITH_MASS);
        const LocalMap map = solidMapOf(view, model, solid, kind);
        block.append(row, map.dofs, congruent(model.solidPart()->localMass(solid.partIndex, *kernel, kind), map));
    }
}

namespace
{
bool isSolidDensity(const DesignedField& field)
{
    return (field.card == "IsoMat" || field.card == "AnisoMat3D") && field.column == "density";
}
// At stationary bubble displacement the condensed stiffness rate is -Sigma^T (dS/dnu) Sigma.
// The scalar stress basis is orthonormal on the solid. In engineering Voigt notation, E*dS/dnu
// has -1 in the off-diagonal normal entries and 2 on the three shear diagonals.
Matrix poissonStiffnessRate(const VolumeKernel& kernel, double young)
{
    const auto& stress = kernel.stressOperator();
    const auto count = kernel.stressFunctionCount();
    Matrix weighted(stress.rows(), stress.columns());
    for (std::size_t k = 0; k < count; ++k)
        for (std::size_t j = 0; j < stress.columns(); ++j)
            for (std::size_t c = 0; c < 3; ++c)
            {
                weighted(c * count + k, j) =
                    stress(((c + 1) % 3) * count + k, j) + stress(((c + 2) % 3) * count + k, j);
                weighted((c + 3) * count + k, j) = -2.0 * stress((c + 3) * count + k, j);
            }
    return symmetricPart(transposeTimes(stress, weighted)) * (1.0 / young);
}
} // namespace

bool MixedHybridSolidFamily::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                      const DesignedField& field, ElementMatrixBlock& block) const
{
    const bool young = field.card == "IsoMat" && field.column == "youngsModulus";
    const bool poisson = field.card == "IsoMat" && field.column == "poissonRatio";
    if (!young && !poisson && !isSolidDensity(field))
        return false;
    const auto& model = MixedHybridModel::of(view.index());
    const auto& solid = requiredSolid(model, view, row);
    const auto& section = solid.classicSection;
    if (field.id != section.materialId || ((young || poisson) && !(section.youngModulus > 0.0)))
        return false;
    const auto kind = kindOf(view);
    const bool interior = kind == EntityLayout::Kind::Modal && model.solidPart()->interiorEntity(solid.partIndex);
    if (poisson && interior)
        return false; // a change in nu also changes the retained modal interior basis
    if (!young && !poisson && !interior)
        return true;
    const auto kernel = kernelOf(model, solid, kind, poisson ? AS_KEPT : WITH_STIFFNESS);
    Matrix rate;
    if (poisson)
        rate = poissonStiffnessRate(*kernel, section.youngModulus);
    else if (young)
        rate = model.solidPart()->localStiffness(solid.partIndex, *kernel, kind) * (1.0 / section.youngModulus);
    else
    {
        rate = model.solidPart()->localStiffness(solid.partIndex, *kernel, kind);
        // At fixed stiffness the boundary entries are independent of rho. Each retained, mass-normalised
        // interior eigenvector scales as rho^-1/2, so its diagonal stiffness scales as rho^-1.
        const auto boundary = kernel->parameterCount();
        if (rate.rows() == boundary)
            return true;
        if (!(section.density > 0.0))
            return false;
        for (std::size_t i = 0; i < rate.rows(); ++i)
            for (std::size_t j = 0; j < rate.columns(); ++j)
                rate(i, j) = i == j && i >= boundary ? -rate(i, j) / section.density : 0.0;
    }
    const auto map = solidMapOf(view, model, solid, kind);
    block.append(row, map.dofs, congruent(rate, map));
    return true;
}

bool MixedHybridSolidFamily::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                 const DesignedField& field, const MassOptions&,
                                                 ElementMatrixBlock& block) const
{
    if (field.card == "IsoMat" && field.column == "youngsModulus")
        return true; // uniform elasticity scaling leaves static bubble extension and mass-normalised fields invariant
    if (!isSolidDensity(field))
        return false;
    const auto& model = MixedHybridModel::of(view.index());
    const auto& solid = requiredSolid(model, view, row);
    const auto& section = solid.classicSection;
    if (field.id != section.materialId)
        return false;
    const auto kind = kindOf(view);
    const auto kernel = kernelOf(model, solid, kind, WITH_MASS);
    Matrix rate;
    if (section.density == 0.0)
    {
        if (kind == EntityLayout::Kind::Modal)
            return false; // mass-normalised interior fields require positive density
        auto unitMass = kernel->withMass(1.0);
        if (!unitMass.ok())
            throw base::SolverError(unitMass.refusal());
        rate = model.solidPart()->localMass(solid.partIndex, unitMass.value(), kind);
    }
    else
    {
        rate = model.solidPart()->localMass(solid.partIndex, *kernel, kind);
        rate *= 1.0 / section.density;
        const auto boundary = kernel->parameterCount();
        for (std::size_t i = boundary; i < rate.rows(); ++i)
        {
            // Boundary/interior coupling scales as sqrt(rho); the interior mass is the identity.
            rate(i, i) = 0.0;
            for (std::size_t j = 0; j < boundary; ++j)
                rate(i, j) = rate(j, i) = 0.5 * rate(i, j);
        }
    }
    const auto map = solidMapOf(view, model, solid, kind);
    block.append(row, map.dofs, congruent(rate, map));
    return true;
}

bool MixedHybridSolidFamily::computeVolumes(const ElementBatchView& view, ElementRange range,
                                            std::span<double> volumes) const
{
    // A view without the entities of the analysis names no kernel (check() refuses it): no volume from here.
    if (view.entities() == nullptr)
        return false;
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const EntityLayout::Kind kind = kindOf(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        // The kernel the stiffness of the element is made from, as the SolidPart keeps it.
        const SolidPart::SharedKernel kernel = kernelOf(model, requiredSolid(model, view, row), kind, AS_KEPT);
        volumes[static_cast<std::size_t>(row - range.first)] = kernel->volume();
    }
    return true;
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
        const SolidPart::SharedKernel kernel = kernelOf(model, solid, kind, AS_KEPT);
        Vector onParameters;
        if (!pressure.hasDirection)
            onParameters = part.facePressureLoad(
                *kernel, face,
                model.solidFacePressure(solid, grids,
                                        std::span<const double>(pressure.cornerPressures.data(), grids.size())));
        else
        {
            // The unit direction in the basic system (the element frame of pressureDirectionOf taken as the basic one).
            const base::Vec3 unit = *pressureDirectionOf(view, pressure, base::Mat3::identity()).inElementFrame;
            const double p = pressure.cornerPressures[0];
            onParameters = part.faceTractionLoad(*kernel, face,
                                                 [&](const SolidPoint&) {
                                                     return SolidPoint{p * unit[0], p * unit[1], p * unit[2]};
                                                 });
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
        if (auto problem = missingCornerTemperature(view, row, *solid, temperatures))
        {
            sink.report(std::move(*problem));
            continue;
        }
        if (!solid->isotropicExpansion &&
            MixedHybridModel::takesThermalStrain(*solid, model.solidNodalChanges(*solid, &temperatures)))
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
        const std::vector<double> changes = model.solidNodalChanges(solid, &temperatures);
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

bool MixedHybridSolidFamily::differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                                  const DesignedField& field,
                                                                  const NodalDisplacements& preload,
                                                                  ElementMatrixBlock& block) const
{
    const bool density = isSolidDensity(field);
    const bool poisson = field.card == "IsoMat" && field.column == "poissonRatio";
    if (!density && !poisson && !(field.card == "IsoMat" && field.column == "youngsModulus"))
        return false;
    const auto& model = MixedHybridModel::of(view.index());
    const auto& solid = requiredSolid(model, view, row);
    if (solid.classicSection.materialId != field.id)
        return false;
    if (density && (preload.bodyAcceleration == nullptr || preload.bodyAcceleration->isZero()))
        return true;
    const auto kind = kindOf(view);
    auto homogeneous = preload;
    homogeneous.bodyAcceleration = nullptr;
    auto stresses = model.solidStresses(solid, kind, poisson ? preload : homogeneous);
    if (density)
    {
        std::fill(stresses.beta.begin(), stresses.beta.end(), 0.0);
        MixedHybridModel::addBodyForceStresses(1.0, preload.bodyAcceleration, *stresses.kernel, stresses.beta);
    }
    else if (!poisson)
    {
        if (!(solid.classicSection.youngModulus > 0.0))
            return false;
        for (double& coefficient : stresses.beta)
            coefficient /= solid.classicSection.youngModulus;
    }
    const auto result = poisson ? stresses.kernel->geometricPoissonRate(stresses.beta, solid.part.material,
                                                                        solid.classicSection.youngModulus)
                                : SolidPart::geometricStiffness(*stresses.kernel, stresses.beta);
    if (!result.ok())
        throw std::logic_error(result.refusal());
    const auto map = solidMapOf(view, model, solid, kind);
    Matrix local(map.toLocal.rows(), map.toLocal.rows());
    local.setBlock(0, 0, result.value());
    block.append(row, map.dofs, congruent(local, map));
    return true;
}

bool MixedHybridSolidFamily::differentialStiffnessPreloadGradient(const ElementBatchView& view, std::int64_t row,
                                                                  const NodalDisplacements& shape,
                                                                  ElementVectorBlock& block) const
{
    // K_e(u) = Tᵀ K_g(β(T u)) T on the parameters (computeDifferentialStiffness): the gradient on the local unknowns,
    // zero on the interior fields of a modal layout, carried to the DOFs by Tᵀ.
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const EntityLayout::Kind kind = kindOf(view);
    const MixedSolid& solid = requiredSolid(model, view, row);
    const MixedHybridModel::SolidCaseStresses stresses = model.solidStresses(solid, kind, shape);
    const LocalMap map = solidMapOf(view, model, solid, kind);
    const Vector phi = localUnknowns(map, shape);
    SolidResult<Vector> gradient = stresses.kernel->geometricGradient(phi);
    if (!gradient.ok())
        throw std::logic_error(base::concatenate(
            {"MixedHybridSolidFamily: element ", std::to_string(solid.elementId), ": ", gradient.refusal()}));
    Vector local(map.toLocal.rows(), 0.0);
    std::copy(gradient.value().begin(), gradient.value().end(), local.begin());
    block.append(row, map.dofs, onDofs(local, map));
    return true;
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
        // PERFORMANCE: the stresses read the kernel as the SolidPart keeps it, without its stiffness beyond the first
        // 80 hexahedra or so: before, each of the others was built again here. Measured on 216 hexahedra of degree 4:
        // the recovery from 6.5 s to 0.1 s, the stresses at the grid points (MixedHybridModel::gridPointStresses)
        // the same.
        const MixedHybridModel::SolidCaseStresses stresses = model.solidStresses(solid, kindOf(view), displacements);
        const SolidPointStresses points = SolidPart::pointStresses(*stresses.kernel, stresses.beta);
        // The places: the centre, then the corners; the temperature change there for the strains (the functions of
        // the corners at the centre of the card's other families).
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
        if (output == ElementOutput::Strains && !stresses.nodalChanges.empty())
        {
            const auto& geometry = stresses.kernel->geometry();
            centreChange =
                model.solidTemperatureField(solid, stresses.nodalChanges)(geometry.at(shapeCentre(geometry.shape)));
        }
        appendPlace(0, points.centre, centreChange);
        for (std::size_t a = 0; a < points.corners.size(); ++a)
            appendPlace(ids[static_cast<std::size_t>(solid.cornerRows[a])], points.corners[a],
                        stresses.nodalChanges.empty() ? 0.0 : stresses.nodalChanges[a]);
    }
}

} // namespace oos::elements::mixedHybrid
