#include "oos/elements/rods/RodFormulation.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MarginOfSafety.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

constexpr std::size_t AXIAL = 0;
constexpr std::size_t TWIST = 1;

[[nodiscard]] double dotOfComponents(base::Vec3 axis, const NodalDisplacements& displacements, std::int32_t gridRow,
                                     int firstComponent)
{
    double sum = 0.0;
    for (int k = 0; k < 3; ++k)
        sum += axis[static_cast<std::size_t>(k)] * displacements.valueAt(ElementDof::grid(gridRow, firstComponent + k));
    return sum;
}

} // namespace

RodConnectivity::RodConnectivity(const ElementBatchView& view)
    : rows(view.batch().nodeRows()), ids(view.batch().nodes()), valid(view.columns().validity("nodesRows"))
{
}

Resolution<RodGeometry> resolveRodGeometry(const ElementBatchView& view, const RodConnectivity& connectivity,
                                           std::int64_t row)
{
    using Result = Resolution<RodGeometry>;
    const std::int64_t nodeCount = view.snapshot().nodes().rowCount();
    const std::span<const std::int32_t> ends = connectivity.rows[row];
    const bool isValid = connectivity.valid.isValid(row) && ends.size() == 2;
    for (std::size_t end = 0; end < 2; ++end)
    {
        // A null row of nodesRows does not say which grid point is unknown: the IDs tell it.
        const std::span<const std::int32_t> ids = connectivity.ids[row];
        const bool isKnown = isValid ? ends[end] >= 0 && ends[end] < nodeCount
                                     : end < ids.size() && view.index().points().findGrid(ids[end]).has_value();
        if (!isKnown)
        {
            const std::string id = end < ids.size() ? std::to_string(ids[end]) : std::string("blank");
            return Result::failure(elementProblem(
                view, row, problem_code::MISSING_POINT, end == 0 ? "node1" : "node2",
                base::concatenate({"expected the ID of a grid point, found ", id,
                                   ", which the model does not define; define it or correct the card"})));
        }
    }
    if (!isValid)
        throw base::ContractError(base::concatenate(
            {view.card(), " ", std::to_string(view.elementId(row)),
             ": nodesRows is null although both grid points exist; the element table and the node table come from "
             "different snapshots: take every table of the snapshot from one model"}));
    RodGeometry geometry;
    const std::span<const std::int32_t> endIds = connectivity.ids[row];
    geometry.rowA = checkedGridRow(view, ends[0], endIds[0]);
    geometry.rowB = checkedGridRow(view, ends[1], endIds[1]);
    const base::Vec3 positionA = view.positionInBasic(geometry.rowA);
    const base::Vec3 positionB = view.positionInBasic(geometry.rowB);
    const base::Vec3 span = positionB - positionA;
    geometry.length = base::norm(span);
    const double scale = std::max({1.0, base::norm(positionA), base::norm(positionB)});
    if (geometry.length <= ROD_ZERO_LENGTH_FACTOR * scale)
        return Result::failure(elementProblem(
            view, row, problem_code::DEGENERATE_GEOMETRY, "node2",
            base::concatenate({"expected two distinct grid points, found Rod2.node1 and Rod2.node2 at the same "
                               "position (length ",
                               std::to_string(geometry.length),
                               "); a rod needs a length to define its axis: move a grid point or remove the rod"})));
    const base::Vec3 axis = span / geometry.length;
    geometry.frameA = view.displacementFrame(geometry.rowA);
    geometry.frameB = view.displacementFrame(geometry.rowB);
    geometry.axisAtA = geometry.frameA.transposed() * axis;
    geometry.axisAtB = geometry.frameB.transposed() * axis;
    return Result::success(geometry);
}

base::MatN<12> rodStiffness(const RodData& rod)
{
    const RodGeometry& g = rod.geometry;
    base::MatN<2, 12> h;
    for (std::size_t k = 0; k < 3; ++k)
    {
        h(AXIAL, k) = -g.axisAtA[k];
        h(AXIAL, 6 + k) = g.axisAtB[k];
        h(TWIST, 3 + k) = -g.axisAtA[k];
        h(TWIST, 9 + k) = g.axisAtB[k];
    }
    const std::array<double, 2> stiffness{rod.material.youngModulus * rod.section.area / g.length,
                                          rod.material.shearModulus * rod.section.torsionalConstant / g.length};
    return diagonalCongruence(stiffness, h);
}

base::MatN<12> rodDifferentialStiffness(const RodData& rod, double axialForce)
{
    const RodGeometry& g = rod.geometry;
    const double scale = axialForce / g.length;
    const std::array<const base::Mat3*, 2> frames{&g.frameA, &g.frameB};
    const std::array<base::Vec3, 2> axes{g.axisAtA, g.axisAtB};
    base::MatN<12> result;
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t b = 0; b < 2; ++b)
        {
            // D_ab = R_aᵀ·R_b − x_a·x_bᵀ, with the sign of [D −D; −D D].
            const base::Mat3 rotation = frames[a]->transposed() * *frames[b];
            const double sign = a == b ? scale : -scale;
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    result(6 * a + i, 6 * b + j) = sign * (rotation(i, j) - axes[a][i] * axes[b][j]);
        }
    return result;
}

base::MatN<6> rodMass(const RodData& rod, double massPerLength, MassFormulation formulation)
{
    const double totalMass = massPerLength * rod.geometry.length;
    base::MatN<6> massInBasic;
    if (formulation == MassFormulation::Lumped)
    {
        // Half the mass at each end, the same in every direction: the matrix is m·L/2·I in any frame.
        for (std::size_t k = 0; k < 6; ++k)
            massInBasic(k, k) = totalMass / 2.0;
        return massInBasic;
    }
    // The coupled mass of a rod (COUPMASS): along the axis x, the mean of the lumped and the consistent matrices,
    // (m/2·I + m/6·[2 1; 1 2])/2 = m/12·[5 1; 1 5]; across it, the lumped m/2·I. So M = m/2·I₆ + m/12·[−1 1; 1 −1] ⊗
    // x·xᵀ. Not the consistent ∫ ρ Nᵀ N dx in every direction: the transverse mass stays lumped by design (on a
    // ten-bar truss with MassDofProperty masses, the fully consistent matrix gives eigenvalues 2.6 % to 19 % higher).
    const base::Vec3 axis = rod.geometry.frameA * rod.geometry.axisAtA;
    for (std::size_t k = 0; k < 6; ++k)
        massInBasic(k, k) = totalMass / 2.0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            const double axial = totalMass / 12.0 * axis[i] * axis[j];
            massInBasic(i, j) -= axial;
            massInBasic(3 + i, 3 + j) -= axial;
            massInBasic(i, 3 + j) += axial;
            massInBasic(3 + i, j) += axial;
        }
    return toDisplacementFrames<6>(massInBasic, {rod.geometry.frameA, rod.geometry.frameB});
}

std::array<ElementDof, 12> rodStiffnessDofs(const RodGeometry& geometry)
{
    std::array<ElementDof, 12> dofs;
    for (int k = 0; k < 6; ++k)
    {
        dofs[static_cast<std::size_t>(k)] = ElementDof::grid(geometry.rowA, k + 1);
        dofs[static_cast<std::size_t>(6 + k)] = ElementDof::grid(geometry.rowB, k + 1);
    }
    return dofs;
}

std::array<ElementDof, 6> rodMassDofs(const RodGeometry& geometry)
{
    return {ElementDof::grid(geometry.rowA, 1), ElementDof::grid(geometry.rowA, 2), ElementDof::grid(geometry.rowA, 3),
            ElementDof::grid(geometry.rowB, 1), ElementDof::grid(geometry.rowB, 2), ElementDof::grid(geometry.rowB, 3)};
}

Resolution<double> resolveRodTemperatureChange(const ElementBatchView& view, std::int64_t row, const RodData& rod,
                                               const ElementTemperatures& temperatures)
{
    using Result = Resolution<double>;
    const std::array<std::int32_t, 2> ends{rod.geometry.rowA, rod.geometry.rowB};
    double load = 0.0;
    double initial = 0.0;
    for (const std::int32_t end : ends)
    {
        const std::optional<double> value = temperatures.load.atGridRow(end);
        if (!value)
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD, "TemperatureSelection(LOAD)",
                base::concatenate({"expected a temperature at grid point ",
                                   std::to_string(view.snapshot().nodes().ids()[static_cast<std::size_t>(end)]),
                                   " of the rod in the thermal load set, found none (no NodeTemperatureLoad, no "
                                   "DefaultTemperatureLoad); give "
                                   "the grid point a NodeTemperatureLoad or the set a DefaultTemperatureLoad"})));
        load += 0.5 * *value;
        initial += 0.5 * (temperatures.initial.isEmpty()
                              ? rod.material.referenceTemperature
                              : temperatures.initial.atGridRow(end).value_or(rod.material.referenceTemperature));
    }
    return Result::success(load - initial);
}

Resolution<RodInitialStrain> resolveRodInitialStrain(const ElementBatchView& view, std::int64_t row, const RodData& rod,
                                                     const ElementTemperatures& temperatures)
{
    using Result = Resolution<RodInitialStrain>;
    RodInitialStrain initial;
    if (temperatures.hasTemperatureLoad)
    {
        Resolution<double> change = resolveRodTemperatureChange(view, row, rod, temperatures);
        if (!change)
            return Result::failure(std::move(change.problem));
        initial.deltaT = *change.value;
    }
    initial.elongation = temperatures.elongationOf(view.elementId(row));
    return Result::success(initial);
}

double rodThermalForce(const RodData& rod, const RodInitialStrain& initial)
{
    // E A α ΔT first, then the elongation term (the same bits for a thermal load alone: adding the 0 of no elongation
    // keeps them).
    const double thermal = rod.material.youngModulus * rod.section.area * rod.material.expansion * initial.deltaT;
    return thermal + rod.material.youngModulus * rod.section.area / rod.geometry.length * initial.elongation;
}

RodForces rodForces(const RodData& rod, const NodalDisplacements& displacements, const RodInitialStrain& initial)
{
    const RodGeometry& g = rod.geometry;
    const double stretch =
        dotOfComponents(g.axisAtB, displacements, g.rowB, 1) - dotOfComponents(g.axisAtA, displacements, g.rowA, 1);
    const double twist =
        dotOfComponents(g.axisAtB, displacements, g.rowB, 4) - dotOfComponents(g.axisAtA, displacements, g.rowA, 4);
    return {rod.material.youngModulus * rod.section.area / g.length * stretch - rodThermalForce(rod, initial),
            rod.material.shearModulus * rod.section.torsionalConstant / g.length * twist};
}

std::vector<std::string> rodOutputColumns(ElementOutput output)
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"axial", "torsion"};
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {"axial", "sma", "torsion", "smt"};
    case ElementOutput::StrainCurvatures:
        break;
    }
    return {};
}

std::vector<ColumnForm> rodOutputColumnForms(ElementOutput output)
{
    const ColumnForm linear = ColumnForm::linear();
    const ColumnForm other = ColumnForm::other();
    switch (output)
    {
    case ElementOutput::Forces:
        return {linear, linear};
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {linear, other, linear, other};
    case ElementOutput::StrainCurvatures:
        break;
    }
    return {};
}

void rodRecoverySlopes(const RodData& rod, ElementOutput output, RecoverySlopes& slopes)
{
    const std::vector<ColumnForm> forms = rodOutputColumnForms(output);
    const std::size_t columnCount = forms.size();
    const std::array<ElementDof, 12> dofs = rodStiffnessDofs(rod.geometry);
    slopes.reset(dofs, columnCount);
    slopes.appendRow({});
    // The row per unit P and per unit T: rodOutputRow is linear in (P, T) in its Linear columns.
    const std::array<double, 4> atRest = rodOutputRow(rod, {0.0, 0.0}, output);
    const std::array<double, 4> perForce = rodOutputRow(rod, {1.0, 0.0}, output);
    const std::array<double, 4> perTorque = rodOutputRow(rod, {0.0, 1.0}, output);
    // ∂P/∂u and ∂T/∂u over (A 1-6, B 1-6).
    const RodGeometry& g = rod.geometry;
    const double axial = rod.material.youngModulus * rod.section.area / g.length;
    const double torsion = rod.material.shearModulus * rod.section.torsionalConstant / g.length;
    std::array<double, 12> forceRate{};
    std::array<double, 12> torqueRate{};
    for (std::size_t k = 0; k < 3; ++k)
    {
        forceRate[k] = -axial * g.axisAtA[k];
        forceRate[6 + k] = axial * g.axisAtB[k];
        torqueRate[3 + k] = -torsion * g.axisAtA[k];
        torqueRate[9 + k] = torsion * g.axisAtB[k];
    }
    for (std::size_t c = 0; c < columnCount; ++c)
    {
        if (forms[c].kind != ColumnForm::Kind::Linear)
            continue;
        const double perP = perForce[c] - atRest[c];
        const double perT = perTorque[c] - atRest[c];
        const std::span<double> slope = slopes.slopes(0, c);
        for (std::size_t d = 0; d < dofs.size(); ++d)
            slope[d] = perP * forceRate[d] + perT * torqueRate[d];
    }
}

base::MatN<12> rodStiffnessRate(const RodData& rod, bool isArea)
{
    RodData unit = rod;
    unit.section.area = isArea ? 1.0 : 0.0;
    unit.section.torsionalConstant = isArea ? 0.0 : 1.0;
    return rodStiffness(unit);
}

std::array<double, 4> rodOutputRow(const RodData& rod, const RodForces& forces, ElementOutput output)
{
    if (output == ElementOutput::Forces)
        return {forces.axial, forces.torque, 0.0, 0.0};
    const RodSection& section = rod.section;
    const double axialStress = section.area > 0.0 ? forces.axial / section.area : 0.0;
    const double torsionalStress =
        section.torsionalConstant > 0.0 ? section.stressCoefficient * forces.torque / section.torsionalConstant : 0.0;
    const double axialMargin =
        normalStressMargin(axialStress, rod.material.tensionLimit, rod.material.compressionLimit);
    const double torsionalMargin = shearStressMargin(torsionalStress, rod.material.shearLimit);
    if (output == ElementOutput::Strains)
    {
        // Hooke's law on the uniaxial and the torsional stress: ε = σ/E, γ = τ/G. The margins are those of the
        // stresses: the strain and the stress tables of a linear analysis carry the same margins, which is the margin
        // of the strain against the limits divided by the moduli (ST/E, SC/E, SS/G).
        const double axialStrain = uniaxialStrain(axialStress, rod.material);
        const double torsionalStrain =
            rod.material.shearModulus > 0.0 ? torsionalStress / rod.material.shearModulus : 0.0;
        return {axialStrain, axialMargin, torsionalStrain, torsionalMargin};
    }
    return {axialStress, axialMargin, torsionalStress, torsionalMargin};
}

} // namespace oos::elements
