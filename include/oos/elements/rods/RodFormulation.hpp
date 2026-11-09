#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"
#include "oos/elements/core/MarginOfSafety.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace oos::elements
{

// The formulation shared by the rod elements Rod2, Rod2InlineSection and Tube2: a two-node element with an axial
// stiffness EA/L and a torsional stiffness GJ/L along the axis x = (x_B − x_A)/L, and no bending or shear stiffness.
// The three cards differ only in where the section comes from (PROD, the Rod2InlineSection card itself, TubeProp), so
// each element's facet files gather a RodSection and call this.

// The section of a rod, from its property (or the Rod2InlineSection card).
struct RodSection
{
    double area = 0.0;              // A
    double torsionalConstant = 0.0; // J
    double stressCoefficient = 0.0; // C: torsional stress τ = C·T/J (PROD field C)
    double nonStructuralMass = 0.0; // NSM, mass per unit length
};

// The geometry of a rod: its grid point rows, axis and length in the basic system, and the axis written in the
// displacement system of each end (x_CD = Rᵀ·x, R the CD-to-basic rotation of that grid point).
struct RodGeometry
{
    std::int32_t rowA = 0;
    std::int32_t rowB = 0;
    double length = 0.0;
    base::Vec3 axisAtA; // the unit axis in the displacement system of A
    base::Vec3 axisAtB; // the unit axis in the displacement system of B
    base::Mat3 frameA = base::Mat3::identity();
    base::Mat3 frameB = base::Mat3::identity();
};

// Everything a rod facet needs about one element.
struct RodData
{
    RodGeometry geometry;
    RodSection section;
    IsotropicMaterial material;
};

// A rod is refused as degenerate when its length is at the round-off level of its coordinates: L ≤ this factor times
// max(1, |x_A|, |x_B|). Two coincident grid points give no axis, so such a rod is refused.
inline constexpr double ROD_ZERO_LENGTH_FACTOR = 1.0e-12;

// The grid point columns of a rod batch, read once per facet call: nodesRows (O_o_Reader: the rows of GA and GB in
// the node table, null for a grid point the model does not define) and nodes (their IDs, for the messages).
struct RodConnectivity
{
    input::ConnectivityView rows;
    input::ConnectivityView ids;
    base::ValidityView valid;

    explicit RodConnectivity(const ElementBatchView& view);
};

// The geometry of the rod of batch row `row`; an unknown grid point or a zero length is a problem.
[[nodiscard]] Resolution<RodGeometry> resolveRodGeometry(const ElementBatchView& view,
                                                         const RodConnectivity& connectivity, std::int64_t row);

// The 12 × 12 stiffness over the DOFs (A 1-6, B 1-6) in the displacement systems of A and B:
// K = Hᵀ·diag(EA/L, GJ/L)·H with the axial stretch δ = x·(u_B − u_A) and the twist φ = x·(θ_B − θ_A), each end's
// motion written in its own CD system: H = [−x_Aᵀ 0 x_Bᵀ 0; 0 −x_Aᵀ 0 x_Bᵀ] (x_A, x_B the axis in the CD systems).
[[nodiscard]] base::MatN<12> rodStiffness(const RodData& rod);

// The 6 × 6 mass over the translations (A 1-3, B 1-3) in the displacement systems of A and B, with the mass per unit
// length m = ρA + NSM (+ the NSM cards' share): lumped, m·L/2 on each translation of each end; coupled,
// m·L/12·[5 1; 1 5] along the axis (the mean of the lumped and the consistent matrices) and m·L/2 across it. A rod
// carries no rotational inertia in either form.
[[nodiscard]] base::MatN<6> rodMass(const RodData& rod, double massPerLength, MassFormulation formulation);

// The DOFs of rodStiffness (A 1-6, B 1-6) and of rodMass (A 1-3, B 1-3).
[[nodiscard]] std::array<ElementDof, 12> rodStiffnessDofs(const RodGeometry& geometry);
[[nodiscard]] std::array<ElementDof, 6> rodMassDofs(const RodGeometry& geometry);

// The free axial deformation of a rod under the initial strains of a subcase (ElementTemperatures): the
// temperature change ΔT of its thermal load (0 without TEMPERATURE(LOAD)) and the enforced elongation D of its DEFORM
// (0 when the set gives the rod none).
struct RodInitialStrain
{
    double deltaT = 0.0;
    double elongation = 0.0;
};

// The element forces of a rod (the force table of Rod2): the axial force P = EA/L·δ − P_0 (positive in tension; P_0
// the force of the free deformation of a subcase with initial strains, rodThermalForce) and the torque T = GJ/L·φ.
struct RodForces
{
    double axial = 0.0;
    double torque = 0.0;
};
[[nodiscard]] RodForces rodForces(const RodData& rod, const NodalDisplacements& displacements,
                                  const RodInitialStrain& initial = {});

// The temperature change of a rod (linear statics: α (T − T0)): T the mean of the
// temperatures of its two grid points in the load set (TEMP, DefaultTemperatureLoad), T0 the mean of those of the
// TEMPERATURE(INITIAL) set, or TREF of its IsoMat when the subcase has none (a grid point the initial set leaves out
// takes TREF, as the shells take their grid point temperatures). A grid point without a load temperature is a problem.
[[nodiscard]] Resolution<double> resolveRodTemperatureChange(const ElementBatchView& view, std::int64_t row,
                                                             const RodData& rod,
                                                             const ElementTemperatures& temperatures);

// The free deformation of the rod of batch row `row` under the initial strains of a subcase: ΔT (as
// resolveRodTemperatureChange, 0 when the subcase has no temperature load) and the D of its DEFORM.
[[nodiscard]] Resolution<RodInitialStrain> resolveRodInitialStrain(const ElementBatchView& view, std::int64_t row,
                                                                   const RodData& rod,
                                                                   const ElementTemperatures& temperatures);

// The 12 × 12 differential stiffness of a rod under the axial force P (positive in tension), over the DOFs of
// rodStiffness (Cook, Malkus, Plesha & Witt,
// Concepts and Applications of FEA, 4th ed., §18.2): the force P along the axis turns with it, δF_B = −δF_A =
// (P/L)·(I − x xᵀ)·(u_B − u_A) in basic, so that
//
//   K_d = (P/L)·[D −D; −D D] on the translations, D = I − x xᵀ,
//
// each block carried to the displacement systems of its ends, D_ab = R_aᵀ·R_b − x_a·x_bᵀ (x_a the axis in the CD
// system of end a). The torque gives no differential stiffness.
[[nodiscard]] base::MatN<12> rodDifferentialStiffness(const RodData& rod, double axialForce);

// The axial force the free deformation would carry in a rod held at both ends, P_0 = E A α ΔT + (E A / L)·D (an
// enforced elongation D of a DEFORM is the free strain D/L);
// its equivalent nodal loads are −P_0 x_A at A and +P_0 x_B at B.
[[nodiscard]] double rodThermalForce(const RodData& rod, const RodInitialStrain& initial);

// The output columns of the rod elements in the O_o_Reader data contract: forces (axial, torsion), stresses (axial,
// sma, torsion, smt: the axial stress P/A and its margin, the torsional stress C·T/J and its margin), strains (the
// same columns: P/(A·E) and C·T/(J·G), with the margins of the stresses).
[[nodiscard]] std::vector<std::string> rodOutputColumns(ElementOutput output);
// The row of an output kind for one rod: P/A is 0 for A = 0 and C·T/J is 0 for J = 0 (the stress is not defined).
[[nodiscard]] std::array<double, 4> rodOutputRow(const RodData& rod, const RodForces& forces, ElementOutput output);

// The form of each column of rodOutputColumns(output) (ColumnForm): the forces and the axial and
// torsional stresses and strains Linear, the margins Other.
[[nodiscard]] std::vector<ColumnForm> rodOutputColumnForms(ElementOutput output);

// The slopes of the row of a rod (RecoverySlopes, its DOFs those of rodStiffness) in closed form: the axial force
// P = EA/L·(x_B·u_B − x_A·u_A) − P_T and the torque T = GJ/L·(x_B·θ_B − x_A·θ_A) (rodForces), each Linear column of
// rodOutputRow linear in (P, T): its slope is its change per unit P and per unit T, chained with ∂P/∂u and ∂T/∂u.
void rodRecoverySlopes(const RodData& rod, ElementOutput output, RecoverySlopes& slopes);

// The derivatives of a rod's matrices with respect to its area A or its torsion constant J, in closed
// form: K = Hᵀ·diag(EA/L, GJ/L)·H is linear in each (dK/dA = Hᵀ·diag(E/L, 0)·H, dK/dJ = Hᵀ·diag(0, G/L)·H), and the
// mass m·L of rodMass is linear in m = ρA + NSM (dM/dA = rodMass at m = ρ; J does not enter it).
[[nodiscard]] base::MatN<12> rodStiffnessRate(const RodData& rod, bool isArea);

// The analytic derivatives of the rod families for their area and torsion constant fields: `isArea` and `isTorsion`
// say which field it is (neither: no closed form, false); dK/dv and dM/dv appended to `block` as the facets append
// them.
template <typename Resolve>
bool appendRodStiffnessRate(std::int64_t row, bool isArea, bool isTorsion, ElementMatrixBlock& block, Resolve&& resolve)
{
    if (!isArea && !isTorsion)
        return false;
    const RodData rod = resolve(row);
    const std::array<ElementDof, 12> dofs = rodStiffnessDofs(rod.geometry);
    const base::MatN<12> rate = rodStiffnessRate(rod, isArea);
    block.append(row, dofs, rate.values);
    return true;
}

template <typename Resolve>
bool appendRodMassRate(std::int64_t row, bool isArea, bool isTorsion, const MassOptions& options,
                       ElementMatrixBlock& block, Resolve&& resolve)
{
    if (!isArea && !isTorsion)
        return false;
    if (isTorsion)
        return true; // J is not in the mass: no matrix, a zero derivative
    const RodData rod = resolve(row);
    const std::array<ElementDof, 6> dofs = rodMassDofs(rod.geometry);
    const base::MatN<6> rate = rodMass(rod, rod.material.density, options.formulation);
    block.append(row, dofs, rate.values);
    return true;
}

// The rates of a rod's row with respect to its area A or torsion constant J at fixed displacements: the
// axial force P = A·(E·δ/L − E·α·ΔT) and the torque T = J·G·φ/L are linear in them, so their rates are the forces of
// the rod at a unit area or torsion constant (rodForces); the stresses P/A and C·T/J, their strains and margins do not
// move. Appended to `rates` in the layout of recoverRods; false for a field at zero (P/A is not defined there).
template <typename Resolve>
bool appendRodRowRates(const ElementBatchView& view, std::int64_t row, bool isArea, bool isTorsion,
                       ElementOutput output, const NodalDisplacements& displacements, ElementResultBlock& rates,
                       Resolve&& resolve)
{
    if (!isArea && !isTorsion)
        return false;
    const RodData rod = resolve(row);
    if (!((isArea ? rod.section.area : rod.section.torsionalConstant) > 0.0))
        return false;
    rates.reset(rodOutputColumns(output));
    if (output != ElementOutput::Forces)
    {
        rates.append(row, view.elementId(row), std::array<double, 4>{});
        return true;
    }
    RodData unit = rod;
    unit.section.area = isArea ? 1.0 : 0.0;
    unit.section.torsionalConstant = isArea ? 0.0 : 1.0;
    const RodInitialStrain initial =
        displacements.temperatures == nullptr
            ? RodInitialStrain{}
            : resolveRodInitialStrain(view, row, rod, *displacements.temperatures).required();
    const RodForces forces = rodForces(unit, displacements, initial);
    rates.append(row, view.elementId(row), std::array<double, 2>{forces.axial, forces.torque});
    return true;
}

// The second derivatives of a rod with respect to its area A or its torsion constant J: K, the mass and the rows of
// recoverRods are linear in each (the forces P ∝ A and T ∝ J, the stresses P/A and C·T/J independent of them), so
// d²K/dv², d²M/dv² and the rows' second rates are zero, and the slopes of the rows' rates are those of the forces of a
// rod of unit A or J (the stresses' none). False for another field, or for a field at zero (as appendRodRowRates).
template <typename Resolve>
bool hasRodSecondDerivative(std::int64_t row, bool isArea, bool isTorsion, Resolve&& resolve)
{
    if (!isArea && !isTorsion)
        return false;
    const RodData rod = resolve(row);
    return (isArea ? rod.section.area : rod.section.torsionalConstant) > 0.0;
}

template <typename Resolve>
bool appendRodSlopeRates(std::int64_t row, bool isArea, bool isTorsion, ElementOutput output, RecoverySlopes& rates,
                         Resolve&& resolve)
{
    if (!hasRodSecondDerivative(row, isArea, isTorsion, resolve))
        return false;
    RodData rod = resolve(row);
    if (output == ElementOutput::Forces)
    {
        rod.section.area = isArea ? 1.0 : 0.0;
        rod.section.torsionalConstant = isArea ? 0.0 : 1.0;
        rodRecoverySlopes(rod, output, rates);
        return true;
    }
    // The stresses and strains do not move with A or J at fixed displacements: the layout of the slopes, zero.
    const std::array<ElementDof, 12> dofs = rodStiffnessDofs(rod.geometry);
    rates.reset(dofs, rodOutputColumnForms(output).size());
    rates.appendRow({});
    return true;
}

template <typename Resolve>
bool appendRodRowSecondRates(const ElementBatchView& view, std::int64_t row, bool isArea, bool isTorsion,
                             ElementOutput output, ElementResultBlock& rates, Resolve&& resolve)
{
    if (!hasRodSecondDerivative(row, isArea, isTorsion, resolve))
        return false;
    constexpr double NONE = std::numeric_limits<double>::quiet_NaN();
    rates.reset(rodOutputColumns(output));
    if (output == ElementOutput::Forces)
        rates.append(row, view.elementId(row), std::array<double, 2>{});
    else
        rates.append(row, view.elementId(row), std::array<double, 4>{0.0, NONE, 0.0, NONE});
    return true;
}

// The batch loops of the three rod families: `resolve(row)` returns the RodData of a row (or throws for an element
// check() refuses).
template <typename Resolve>
void appendRodStiffness(ElementRange range, ElementMatrixBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const RodData rod = resolve(row);
        const std::array<ElementDof, 12> dofs = rodStiffnessDofs(rod.geometry);
        const base::MatN<12> stiffness = rodStiffness(rod);
        block.append(row, dofs, stiffness.values);
    }
}

template <typename Resolve>
void appendRodMass(ElementRange range, const MassOptions& options, ElementMatrixBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const RodData rod = resolve(row);
        double massPerLength = rod.material.density * rod.section.area + rod.section.nonStructuralMass;
        if (!options.addedNonStructuralMass.empty())
            massPerLength += options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        const std::array<ElementDof, 6> dofs = rodMassDofs(rod.geometry);
        const base::MatN<6> mass = rodMass(rod, massPerLength, options.formulation);
        block.append(row, dofs, mass.values);
    }
}

// The vectorised recovery of the rods (ElementFamily::vectorisedRecoveryMinCases): the fewest cases of a block from
// which their rows are made from their slopes (closed form, rodRecoverySlopes: about one recovery). Calibrated on a
// 6 000-case model's measure of the bars: the slopes,
// the rows at rest and the completion of an element are about three of its recoveries.
inline constexpr std::size_t ROD_VECTORISED_RECOVERY_MIN_CASES = 4;

// The columns of the rod rows (rodOutputRow) that are not Linear, the margins, from the Linear ones in every row of
// every block (ElementFamily::completeRecoveredRows; one row per rod, `resolve(row)` its data, read once for all the
// blocks): the stress margins by normalStressMargin and shearStressMargin of the stresses, as rodOutputRow; the strain
// margins, which rodOutputRow takes from the stresses, of the stresses σ = E·ε and τ = G·γ (to the round-off of
// those products); the forces are all Linear.
template <typename Resolve>
void completeRodRows(ElementOutput output, std::span<ElementResultBlock> blocks, Resolve&& resolve)
{
    if (output == ElementOutput::Forces || blocks.empty())
        return;
    const ElementResultBlock& first = blocks.front();
    for (std::size_t i = 0; i < first.size(); ++i)
    {
        const RodData rod = resolve(first.elementRow(i));
        const IsotropicMaterial& material = rod.material;
        for (ElementResultBlock& block : blocks)
        {
            const std::span<double> values = block.mutableValues(i);
            const double axial = output == ElementOutput::Strains ? values[0] * material.youngModulus : values[0];
            const double torsion = output == ElementOutput::Strains ? values[2] * material.shearModulus : values[2];
            values[1] = normalStressMargin(axial, material.tensionLimit, material.compressionLimit);
            values[3] = shearStressMargin(torsion, material.shearLimit);
        }
    }
}

template <typename Resolve>
void recoverRods(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block, Resolve&& resolve)
{
    block.reset(rodOutputColumns(output));
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const RodData rod = resolve(row);
        const RodInitialStrain initial =
            displacements.temperatures == nullptr
                ? RodInitialStrain{}
                : resolveRodInitialStrain(view, row, rod, *displacements.temperatures).required();
        const std::array<double, 4> values = rodOutputRow(rod, rodForces(rod, displacements, initial), output);
        const std::size_t count = output == ElementOutput::Forces ? 2 : 4;
        block.append(row, view.elementId(row), std::span<const double>(values.data(), count));
    }
}

// The differential stiffness facet of the three rod families: the axial force of the preload (the force of its
// initial strains subtracted, as the recovery does), then rodDifferentialStiffness.
template <typename Resolve>
void appendRodDifferentialStiffness(const ElementBatchView& view, ElementRange range, const NodalDisplacements& preload,
                                    ElementMatrixBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const RodData rod = resolve(row);
        const RodInitialStrain initial =
            preload.temperatures == nullptr ? RodInitialStrain{}
                                            : resolveRodInitialStrain(view, row, rod, *preload.temperatures).required();
        const double axialForce = rodForces(rod, preload, initial).axial;
        const std::array<ElementDof, 12> dofs = rodStiffnessDofs(rod.geometry);
        const base::MatN<12> differential = rodDifferentialStiffness(rod, axialForce);
        block.append(row, dofs, differential.values);
    }
}

// The thermal facets of the three rod families: the loads of their initial strains, the temperatures' and the
// enforced elongations.
template <typename Resolve>
void checkRodThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                          base::DiagnosticSink& sink, Resolve&& resolve)
{
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        auto rod = resolve(row);
        if (!rod)
            continue; // check() reports it
        Resolution<RodInitialStrain> initial = resolveRodInitialStrain(view, row, *rod.value, temperatures);
        if (!initial)
            sink.report(std::move(initial.problem));
    }
}

template <typename Resolve>
void appendRodThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                           ElementVectorBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const RodData rod = resolve(row);
        const double force = rodThermalForce(rod, resolveRodInitialStrain(view, row, rod, temperatures).required());
        if (force == 0.0)
            continue;
        const std::array<ElementDof, 6> dofs = rodMassDofs(rod.geometry);
        const std::array<double, 6> loads{-force * rod.geometry.axisAtA[0], -force * rod.geometry.axisAtA[1],
                                          -force * rod.geometry.axisAtA[2], force * rod.geometry.axisAtB[0],
                                          force * rod.geometry.axisAtB[1],  force * rod.geometry.axisAtB[2]};
        block.append(row, dofs, loads);
    }
}

} // namespace oos::elements
