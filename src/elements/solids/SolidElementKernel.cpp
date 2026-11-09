#include "oos/elements/solids/SolidElementKernel.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/solids/SolidIsoparametric.hpp"
#include "oos/elements/solids/SolidPressure.hpp"
#include "oos/elements/solids/SolidRecovery.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::elements
{

namespace
{

// An element whose |det J| falls below this fraction of its mean somewhere (at the mass rule's points, and at the
// corners of a linear element other than a pyramid) is degenerate: flat, folded or with coincident corners.
constexpr double SOLID_JACOBIAN_TOLERANCE = 1.0e-6;

// The response of one element by its formulation, the thermal modes cleared first (a formulation without internal
// modes leaves them so; the response is reused across the elements of a range, whose PSOLIDs may differ).
void formulate(SolidFormulation formulation, const SolidElementData& data, SolidResponseParts parts,
               SolidResponse& response)
{
    response.hasThermalModes = false;
    formulation(data, parts, response);
}

// D·ε for a strain in the material system.
[[nodiscard]] Voigt6 stressOf(const SolidSection& section, const Voigt6& strain)
{
    Voigt6 stress{};
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            stress[i] += section.elasticity(i, j) * strain[j];
    return stress;
}

// T_a = axesᵀ·R_a, the map from the displacement system CD of node a to the material system.
[[nodiscard]] base::Mat3 toMaterialSystem(const SolidElementData& data, std::size_t node)
{
    return data.section.axes.transposed() * data.nodes.cdFrames[node];
}

// The three translation DOFs of every node.
[[nodiscard]] std::vector<ElementDof> translationDofs(const SolidNodes& nodes)
{
    std::vector<ElementDof> dofs;
    dofs.reserve(3 * nodes.count);
    for (std::size_t a = 0; a < nodes.count; ++a)
        for (int c = 1; c <= 3; ++c)
            dofs.push_back(ElementDof::grid(nodes.rows[a], c));
    return dofs;
}

// K_CD = Tᵀ·K·T, block by block, into `values` (row-major, 3n × 3n).
void toDisplacementSystems(const SolidElementData& data, const SolidMatrix& k, std::vector<double>& values)
{
    const std::size_t n = data.nodes.count;
    const std::size_t size = 3 * n;
    values.assign(size * size, 0.0);
    std::array<base::Mat3, MAX_SOLID_NODES> t{};
    for (std::size_t a = 0; a < n; ++a)
        t[a] = toMaterialSystem(data, a);
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b)
        {
            base::Mat3 block;
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    block(i, j) = k(3 * a + i, 3 * b + j);
            const base::Mat3 carried = t[a].transposed() * block * t[b];
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    values[(3 * a + i) * size + 3 * b + j] = carried(i, j);
        }
}

// The temperature change T − T0 of each node of an element (SolidElementKernel.hpp, thermal loads).
using NodeTemperatureChanges = std::array<double, MAX_SOLID_NODES>;

[[nodiscard]] Resolution<NodeTemperatureChanges> resolveTemperatureChanges(const ElementBatchView& view,
                                                                           std::int64_t row,
                                                                           const SolidElementData& data,
                                                                           const ElementTemperatures& temperatures)
{
    using Result = Resolution<NodeTemperatureChanges>;
    NodeTemperatureChanges changes{};
    // A subcase with element deformations and no temperature load: no thermal strain (DEFORM loads no solid).
    if (!temperatures.hasTemperatureLoad)
        return Result::success(changes);
    for (std::size_t a = 0; a < data.nodes.count; ++a)
    {
        const std::int32_t node = data.nodes.rows[a];
        const std::optional<double> load = temperatures.load.atGridRow(node);
        if (!load)
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD, "TemperatureSelection(LOAD)",
                base::concatenate({"expected a temperature at grid point ", std::to_string(data.nodes.ids[a]),
                                   " of the solid in the thermal load set, found none (no NodeTemperatureLoad, no "
                                   "DefaultTemperatureLoad); "
                                   "give the grid point a NodeTemperatureLoad or the set a DefaultTemperatureLoad"})));
        const double initial = temperatures.initial.isEmpty()
                                   ? data.section.referenceTemperature
                                   : temperatures.initial.atGridRow(node).value_or(data.section.referenceTemperature);
        changes[a] = *load - initial;
    }
    return Result::success(changes);
}

// D·ε_T for a temperature change ΔT (ε_T = a·ΔT, SolidSection::expansion), in the material system.
[[nodiscard]] Voigt6 thermalStressOf(const SolidSection& section, double change)
{
    Voigt6 stress{};
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            stress[i] += section.elasticity(i, j) * section.expansion[j] * change;
    return stress;
}

// The temperature change at a parent point: Σ N_a(point)·ΔT_a.
[[nodiscard]] double changeAt(const SolidElementData& data, const NodeTemperatureChanges& changes, ParentPoint point)
{
    const SolidShapeValues values = solidShapeAt(data.nodes.shape, point);
    double change = 0.0;
    for (std::size_t a = 0; a < data.nodes.count; ++a)
        change += values.n[a] * changes[a];
    return change;
}

// The thermal stress at each recovery place of the formulation (SolidResponse: the centre, then the corners, whose
// temperature change is that of their node).
[[nodiscard]] std::array<Voigt6, MAX_SOLID_RECOVERY_POINTS> placeThermalStresses(const SolidElementData& data,
                                                                                 const NodeTemperatureChanges& changes)
{
    std::array<Voigt6, MAX_SOLID_RECOVERY_POINTS> stresses{};
    stresses[0] = thermalStressOf(data.section, changeAt(data, changes, solidCentre(data.nodes.shape)));
    const std::size_t corners = solidCornerCount(data.nodes.shape);
    for (std::size_t c = 0; c < corners; ++c)
        stresses[1 + c] = thermalStressOf(data.section, changes[c]);
    return stresses;
}

[[nodiscard]] std::string unsupportedPressure(const ElementBatchView& view, const ElementPressure& pressure,
                                              const SolidElementData* data, const char*& field)
{
    field = "frameId";
    if (pressure.hasDirection)
        return "expected a pressure normal to the face, found a direction (FacePressureLoad.frameId or "
               "FacePressureLoad.directionX-FacePressureLoad.directionZ); loads along a vector are not "
               "supported on solids yet";
    field = "node1";
    if (pressure.faceNodes.empty())
        return "expected the face of the solid (face_nodes of pressureTable, from FacePressureLoad.node1 and "
               "FacePressureLoad.node3 or "
               "FacePressureLoad.node4), found none; a FacePressureLoad on a solid needs "
               "FacePressureLoad.node1 and "
               "FacePressureLoad.node3 (FacePressureLoad.node4 for a tetrahedron)";
    if (data != nullptr && !findSolidFace(data->nodes.shape, data->nodes.ids, pressure.faceNodes))
    {
        std::string corners;
        for (const std::int32_t id : pressure.faceNodes)
            corners.append(corners.empty() ? "" : ", ").append(std::to_string(id));
        return base::concatenate({"expected the corners of a face of ", view.card(), " ",
                                  std::to_string(view.elementId(pressure.elementRow)), ", found grid points ", corners,
                                  "; give FacePressureLoad.node1 and FacePressureLoad.node3 "
                                  "(FacePressureLoad.node4) of one of its faces"});
    }
    return {};
}

} // namespace

const SolidRule& solidMassRule(SolidShape shape)
{
    switch (shape)
    {
    case SolidShape::Tetra4:
        return tetrahedronRule(2);
    case SolidShape::Tetra10:
        return tetrahedronRule(4);
    case SolidShape::Penta6:
        return pentahedronRule(2, 2);
    case SolidShape::Penta15:
        return pentahedronRule(4, 3);
    case SolidShape::Pyram5:
        return hexahedronRule(3);
    case SolidShape::Pyram13:
        return hexahedronRule(4);
    case SolidShape::Hexa8:
        return hexahedronRule(2);
    case SolidShape::Hexa20:
        return hexahedronRule(3);
    }
    return hexahedronRule(2);
}

SolidElementKernel::Columns::Columns(const ElementBatchView& view) : connectivity(view), sections(view) {}

Resolution<SolidElementData> SolidElementKernel::resolve(const ElementBatchView& view, const Columns& columns,
                                                         std::int64_t row) const
{
    using Result = Resolution<SolidElementData>;
    Resolution<SolidNodes> nodes = resolveSolidNodes(view, columns.connectivity, row, m_shapes);
    if (!nodes)
        return Result::failure(std::move(nodes.problem));
    Resolution<SolidSection> section = resolveSolidSection(view, columns.sections, row, nodes.value->shape);
    if (!section)
        return Result::failure(std::move(section.problem));
    SolidElementData data;
    data.nodes = *nodes.value;
    data.section = *section.value;
    const base::Mat3 axesT = data.section.axes.transposed();
    for (std::size_t a = 0; a < data.nodes.count; ++a)
        data.materialPositions[a] = axesT * (data.nodes.positions[a] - data.section.origin);

    const SolidShape shape = data.nodes.shape;
    const bool withCorners = !isQuadraticSolid(shape) && shape != SolidShape::Pyram5;
    if (solidJacobianSign(shape, data.positionsInMaterialSystem(), solidMassRule(shape), withCorners,
                          SOLID_JACOBIAN_TOLERANCE) == 0)
        return Result::failure(elementProblem(
            view, row, problem_code::DEGENERATE_GEOMETRY, "node1",
            base::concatenate({"expected a ", view.card(),
                               " whose volume map keeps one orientation (a Jacobian of one sign, away from zero), "
                               "found it flat, folded or with its corners out of the card's order; correct the grid "
                               "points or their order"})));
    return Result::success(std::move(data));
}

void SolidElementKernel::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

void SolidElementKernel::computeStiffness(const ElementBatchView& view, ElementRange range,
                                          ElementMatrixBlock& block) const
{
    const Columns columns(view);
    const auto response = std::make_unique<SolidResponse>();
    std::vector<double> values;
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const SolidElementData data = resolve(view, columns, row).required();
        formulate(m_formulation, data, SolidResponseParts::Stiffness, *response);
        toDisplacementSystems(data, response->stiffness, values);
        block.append(row, translationDofs(data.nodes), values);
    }
}

void SolidElementKernel::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                     ElementMatrixBlock& block) const
{
    const Columns columns(view);
    const auto scalar = std::make_unique<SolidMatrix>();
    const auto mass = std::make_unique<SolidMatrix>();
    std::vector<double> values;
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const SolidElementData data = resolve(view, columns, row).required();
        const std::size_t n = data.nodes.count;
        buildScalarMass(data.nodes.shape, data.positionsInMaterialSystem(), data.section.density,
                        solidMassRule(data.nodes.shape), *scalar);
        if (options.formulation == MassFormulation::Lumped)
        {
            // HRZ: m_a = M_aa · (Σ_bc M_bc) / Σ_b M_bb.
            double total = 0.0;
            double diagonal = 0.0;
            for (std::size_t a = 0; a < n; ++a)
            {
                diagonal += (*scalar)(a, a);
                for (std::size_t b = 0; b < n; ++b)
                    total += (*scalar)(a, b);
            }
            const double scale = diagonal > 0.0 ? total / diagonal : 0.0;
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t b = 0; b < n; ++b)
                    if (a != b)
                        (*scalar)(a, b) = 0.0;
            for (std::size_t a = 0; a < n; ++a)
                (*scalar)(a, a) *= scale;
        }
        mass->reset(3 * n);
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = 0; b < n; ++b)
                for (std::size_t i = 0; i < 3; ++i)
                    (*mass)(3 * a + i, 3 * b + i) = (*scalar)(a, b);
        toDisplacementSystems(data, *mass, values);
        block.append(row, translationDofs(data.nodes), values);
    }
}

void SolidElementKernel::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                 const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    recoverOutputs(view, range, outputs, displacements, std::span<ElementResultBlock>(&block, 1));
}

// PERFORMANCE: the rows at rest of a block of cases formulated every element again for its stresses and for its
// strains; here once for both, each output's rows those of its recovery alone (the same response and displacements).
void SolidElementKernel::recoverOutputs(const ElementBatchView& view, ElementRange range,
                                        std::span<const RecoveryOutput> outputs,
                                        const NodalDisplacements& displacements,
                                        std::span<ElementResultBlock> blocks) const
{
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        if (outputs[k].isLayer || solidOutputColumns(outputs[k].output).empty())
            throw std::invalid_argument(
                "solid recovery: expected stresses or strains, found another output kind (a bug of O_o_Solver: "
                "report it)");
        blocks[k].reset(solidOutputColumns(outputs[k].output));
    }
    const Columns columns(view);
    const auto response = std::make_unique<SolidResponse>();
    std::array<double, MAX_SOLID_DOFS> u{};
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const SolidElementData data = resolve(view, columns, row).required();
        formulate(m_formulation, data, SolidResponseParts::Recovery, *response);
        for (std::size_t a = 0; a < data.nodes.count; ++a)
        {
            base::Vec3 cd{};
            for (std::size_t i = 0; i < 3; ++i)
                cd[i] = displacements.valueAt(ElementDof::grid(data.nodes.rows[a], static_cast<int>(i) + 1));
            const base::Vec3 material = toMaterialSystem(data, a) * cd;
            for (std::size_t i = 0; i < 3; ++i)
                u[3 * a + i] = material[i];
        }
        std::array<Voigt6, MAX_SOLID_RECOVERY_POINTS> thermal{};
        std::array<Voigt6, MAX_SOLID_RECOVERY_POINTS> modeStrains{};
        std::span<const Voigt6> thermalStresses;
        std::span<const Voigt6> thermalModeStrains;
        if (displacements.temperatures != nullptr)
        {
            const NodeTemperatureChanges changes =
                resolveTemperatureChanges(view, row, data, *displacements.temperatures).required();
            thermal = placeThermalStresses(data, changes);
            thermalStresses = std::span<const Voigt6>(thermal.data(), response->pointCount);
            if (response->hasThermalModes)
            {
                // The internal modes' strain under the thermal load (SolidResponse): part of the total strain, and
                // of the stress D·(ε − ε_T).
                for (std::size_t p = 0; p < response->pointCount; ++p)
                {
                    modeStrains[p] =
                        response->thermalStrain[p].apply(std::span<const double>(changes.data(), data.nodes.count));
                    const Voigt6 modeStress = stressOf(data.section, modeStrains[p]);
                    for (std::size_t k = 0; k < 6; ++k)
                        thermal[p][k] -= modeStress[k];
                }
                thermalModeStrains = std::span<const Voigt6>(modeStrains.data(), response->pointCount);
            }
        }
        for (std::size_t k = 0; k < outputs.size(); ++k)
            appendSolidRows(blocks[k], row, view.elementId(row),
                            std::span<const std::int32_t>(data.nodes.ids.data(), solidCornerCount(data.nodes.shape)),
                            *response, std::span<const double>(u.data(), 3 * data.nodes.count), outputs[k].output,
                            thermalStresses, thermalModeStrains);
    }
}

// PERFORMANCE: the recovery matrices of a solid from the operators of one formulation, where the vectorised recovery
// would take 1 + 3n recoveries at unit displacements (each building the formulation again: 25 for a CHEXA8, 31 for a
// CTETRA10) per output.
bool SolidElementKernel::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                                 std::span<const RecoveryOutput> outputs,
                                                 std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (output.isLayer || solidOutputColumns(output.output).empty())
            throw std::invalid_argument(
                "solid recovery slopes: expected stresses or strains, found another output kind (a bug of "
                "O_o_Solver: report it)");
    const Columns columns(view);
    const SolidElementData data = resolve(view, columns, row).required();
    const auto response = std::make_unique<SolidResponse>();
    formulate(m_formulation, data, SolidResponseParts::Recovery, *response);
    const std::vector<ElementDof> dofs = translationDofs(data.nodes);
    std::array<base::Mat3, MAX_SOLID_NODES> toMaterial{};
    for (std::size_t a = 0; a < data.nodes.count; ++a)
        toMaterial[a] = toMaterialSystem(data, a);
    const std::size_t corners = solidCornerCount(data.nodes.shape);
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        const bool isStress = outputs[k].output == ElementOutput::Stresses;
        RecoverySlopes& outputSlopes = slopes[k];
        outputSlopes.reset(dofs, solidOutputColumns(outputs[k].output).size());
        for (std::size_t point = 0; point < response->pointCount; ++point)
        {
            ResultPlace place;
            place.nodeId = point == 0 || point > corners ? 0 : data.nodes.ids[point - 1];
            const std::size_t r = outputSlopes.appendRow(place);
            const SolidOperator& op = isStress ? response->stress[point] : response->strain[point];
            for (std::size_t c = 0; c < 6; ++c)
            {
                const std::span<double> slope = outputSlopes.slopes(r, c);
                for (std::size_t a = 0; a < data.nodes.count; ++a)
                    for (std::size_t j = 0; j < 3; ++j)
                    {
                        double sum = 0.0;
                        for (std::size_t i = 0; i < 3; ++i)
                            sum += op(c, 3 * a + i) * toMaterial[a](i, j);
                        slope[3 * a + j] = sum;
                    }
            }
        }
    }
    return true;
}

void SolidElementKernel::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                      const NodalDisplacements& preload,
                                                      ElementMatrixBlock& block) const
{
    const Columns columns(view);
    const auto response = std::make_unique<SolidResponse>();
    const auto differential = std::make_unique<SolidMatrix>();
    std::array<double, MAX_SOLID_DOFS> u{};
    std::vector<double> values;
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const SolidElementData data = resolve(view, columns, row).required();
        formulate(m_formulation, data, SolidResponseParts::Recovery, *response);
        for (std::size_t a = 0; a < data.nodes.count; ++a)
        {
            base::Vec3 cd{};
            for (std::size_t i = 0; i < 3; ++i)
                cd[i] = preload.valueAt(ElementDof::grid(data.nodes.rows[a], static_cast<int>(i) + 1));
            const base::Vec3 material = toMaterialSystem(data, a) * cd;
            for (std::size_t i = 0; i < 3; ++i)
                u[3 * a + i] = material[i];
        }
        // Point 0 of the recovery is the centre (SolidResponse); the thermal stress of a preload with temperatures
        // subtracted, as the recovery does.
        Voigt6 stress = response->stress[0].apply(std::span<const double>(u.data(), 3 * data.nodes.count));
        if (preload.temperatures != nullptr)
        {
            const NodeTemperatureChanges changes =
                resolveTemperatureChanges(view, row, data, *preload.temperatures).required();
            const Voigt6 thermal =
                thermalStressOf(data.section, changeAt(data, changes, solidCentre(data.nodes.shape)));
            for (std::size_t k = 0; k < 6; ++k)
                stress[k] -= thermal[k];
            if (response->hasThermalModes)
            {
                const Voigt6 modeStress = stressOf(
                    data.section,
                    response->thermalStrain[0].apply(std::span<const double>(changes.data(), data.nodes.count)));
                for (std::size_t k = 0; k < 6; ++k)
                    stress[k] += modeStress[k];
            }
        }
        buildInitialStressStiffness(data.nodes.shape, data.positionsInMaterialSystem(), stress,
                                    solidMassRule(data.nodes.shape), *differential);
        toDisplacementSystems(data, *differential, values);
        block.append(row, translationDofs(data.nodes), values);
    }
}

void SolidElementKernel::checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                            base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    for (const ElementPressure& pressure : pressures)
    {
        const Resolution<SolidElementData> data = resolve(view, columns, pressure.elementRow);
        const char* field = "node1";
        std::string reason = unsupportedPressure(view, pressure, data ? &*data.value : nullptr, field);
        if (reason.empty())
            continue;
        base::Diagnostic diagnostic;
        diagnostic.code = "unsupported-load";
        diagnostic.card = std::string(pressure.card);
        diagnostic.id = pressure.setId;
        diagnostic.field = field;
        diagnostic.message = base::concatenate(
            {std::move(reason), " (on ", view.card(), " ", std::to_string(view.elementId(pressure.elementRow)), ")"});
        sink.report(std::move(diagnostic));
    }
}

void SolidElementKernel::computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                              ElementVectorBlock& block) const
{
    const Columns columns(view);
    std::vector<ElementDof> dofs;
    std::vector<double> values;
    for (const ElementPressure& pressure : pressures)
    {
        const SolidElementData data = resolve(view, columns, pressure.elementRow).required();
        const char* field = "node1";
        const std::string reason = unsupportedPressure(view, pressure, &data, field);
        if (!reason.empty())
            throw std::invalid_argument(
                base::concatenate({view.card(), " ", std::to_string(view.elementId(pressure.elementRow)), ": ", reason,
                                   " (the loads facet was called on a pressure checkPressureLoads() refuses: a bug "
                                   "of O_o_Solver, report it)"}));
        const SolidFaceNodes face = *findSolidFace(data.nodes.shape, data.nodes.ids, pressure.faceNodes);
        const SolidFaceLoads loads = solidPressureLoads(
            data.nodes.shape, std::span<const base::Vec3>(data.nodes.positions.data(), data.nodes.count), face,
            pressure.cornerPressures);
        dofs.clear();
        values.clear();
        for (std::size_t k = 0; k < loads.count; ++k)
        {
            const std::size_t node = loads.nodes[k];
            const base::Vec3 cd = data.nodes.cdFrames[node].transposed() * loads.forces[k];
            for (int c = 1; c <= 3; ++c)
            {
                dofs.push_back(ElementDof::grid(data.nodes.rows[node], c));
                values.push_back(cd[static_cast<std::size_t>(c - 1)]);
            }
        }
        block.append(pressure.elementRow, dofs, values);
    }
}

void SolidElementKernel::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                           base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        const Resolution<SolidElementData> data = resolve(view, columns, row);
        if (!data)
            continue; // check() reports the element itself
        Resolution<NodeTemperatureChanges> changes = resolveTemperatureChanges(view, row, *data.value, temperatures);
        if (!changes)
            sink.report(std::move(changes.problem));
    }
}

void SolidElementKernel::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                             const ElementTemperatures& temperatures, ElementVectorBlock& block) const
{
    const Columns columns(view);
    const auto response = std::make_unique<SolidResponse>();
    SolidOperator b;
    std::vector<double> values;
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const SolidElementData data = resolve(view, columns, row).required();
        const NodeTemperatureChanges changes = resolveTemperatureChanges(view, row, data, temperatures).required();
        const std::size_t n = data.nodes.count;
        const SolidShape shape = data.nodes.shape;
        const std::span<const base::Vec3> positions = data.positionsInMaterialSystem();
        const SolidRule& rule = solidMassRule(shape);
        // The mean change over the volume, ΔT̄ = ∫ ΔT dV / V.
        double volume = 0.0;
        double integral = 0.0;
        for (const SolidQuadraturePoint& point : rule.all())
        {
            const ShapeGradients gradients = shapeGradientsAt(shape, positions, point.point);
            const double weight = std::abs(gradients.determinant) * point.weight;
            double change = 0.0;
            for (std::size_t a = 0; a < n; ++a)
                change += gradients.shape.n[a] * changes[a];
            volume += weight;
            integral += weight * change;
        }
        const double mean = volume > 0.0 ? integral / volume : 0.0;
        // f = K·u_T + Σ_g Bᵀ·(D·a·(ΔT(g) − ΔT̄))·|det J|·w_g in the material system (initial strains, Zienkiewicz &
        // Taylor vol. 1 §6.2): the constant part through the element's own stiffness and the free expansion u_T of
        // the strain a·ΔT̄ (u_T(x) = ε̄·x, ε̄ the strain tensor), which every formulation reproduces exactly, so that a
        // uniform change loads the element as its own constant strain does; the variation inside the element through
        // the isoparametric B, plus, for a formulation with internal modes, their condensed share
        // (SolidResponse::thermalLoad; a uniform change does no work on the incompatible modes of the SolidHexa).
        std::array<double, MAX_SOLID_DOFS> material{};
        const bool isLoaded = std::any_of(changes.begin(), changes.begin() + static_cast<std::ptrdiff_t>(n),
                                          [](double change) { return change != 0.0; });
        if (isLoaded)
            formulate(m_formulation, data, SolidResponseParts::Stiffness, *response);
        if (isLoaded && mean != 0.0)
        {
            const Voigt6& a = data.section.expansion;
            const base::Mat3 strain = base::matrixFromRows(
                {a[0], 0.5 * a[3], 0.5 * a[5]}, {0.5 * a[3], a[1], 0.5 * a[4]}, {0.5 * a[5], 0.5 * a[4], a[2]});
            std::array<double, MAX_SOLID_DOFS> expansion{};
            for (std::size_t node = 0; node < n; ++node)
            {
                const base::Vec3 u = mean * (strain * positions[node]);
                for (std::size_t i = 0; i < 3; ++i)
                    expansion[3 * node + i] = u[i];
            }
            for (std::size_t i = 0; i < 3 * n; ++i)
                for (std::size_t j = 0; j < 3 * n; ++j)
                    material[i] += response->stiffness(i, j) * expansion[j];
        }
        if (isLoaded && response->hasThermalModes)
            for (std::size_t i = 0; i < 3 * n; ++i)
                for (std::size_t node = 0; node < n; ++node)
                    material[i] += response->thermalLoad[i * n + node] * changes[node];
        for (const SolidQuadraturePoint& point : rule.all())
        {
            const ShapeGradients gradients = shapeGradientsAt(shape, positions, point.point);
            double change = -mean;
            for (std::size_t a = 0; a < n; ++a)
                change += gradients.shape.n[a] * changes[a];
            if (change == 0.0)
                continue;
            fillStrainOperator(gradients, b);
            const Voigt6 stress = thermalStressOf(data.section, change);
            const double weight = std::abs(gradients.determinant) * point.weight;
            for (std::size_t j = 0; j < 3 * n; ++j)
            {
                double sum = 0.0;
                for (std::size_t r = 0; r < 6; ++r)
                    sum += b(r, j) * stress[r];
                material[j] += weight * sum;
            }
        }
        // F_CD(a) = T_aᵀ·F_material(a).
        values.assign(3 * n, 0.0);
        for (std::size_t a = 0; a < n; ++a)
        {
            const base::Vec3 cd = toMaterialSystem(data, a).transposed() *
                                  base::Vec3{material[3 * a], material[3 * a + 1], material[3 * a + 2]};
            for (std::size_t i = 0; i < 3; ++i)
                values[3 * a + i] = cd[i];
        }
        block.append(row, translationDofs(data.nodes), values);
    }
}

} // namespace oos::elements
