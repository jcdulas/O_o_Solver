#include "oos/elements/shells/ShellRecoverySlopes.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingPlate.hpp"

namespace oos::elements
{

namespace
{

// What a stiffness term of a AnisoMat2D changes in a shell: the stiffness, the thermal loads and the recovery; not the
// mass, which reads ρ and the thickness only (ShellTria3DrillingPlate::mass).
constexpr FieldUse ANISO_MAT2D_TERM_USE{true, false, true, true, false};

// dK18 along a ply thickness, as ShellQuad4DrillingElement's: sectionRate(atFixedThickness) + dT · thicknessRate,
// the plate's thicknessRate(T) given.
[[nodiscard]] base::MatN<18> plyThicknessStiffness(const ShellTria3DrillingPlate& plate, const LaminateSectionRate& rate,
                                                   const base::MatN<18>& thicknessRate)
{
    base::MatN<18> k = plate.sectionRate(rate.atFixedThickness).stiffness;
    k += rate.thickness * thicknessRate;
    return k;
}

// The rate of the centre's state along a ply thickness at fixed q: the section's at fixed T and T's (`thicknessRates`,
// the plate's centreStateRates(T)).
[[nodiscard]] ShellPlaceState plyThicknessState(const ShellTria3DrillingPlate& plate, const ShellPlaceState& state,
                                                const std::array<double, 18>& q, const LaminateSectionRate& rate,
                                                const ShellPlaceState& thicknessRates)
{
    return combinedStateRates(
        plate.centreStateSectionRates(state, q, rate.atFixedThickness, plate.sectionRate(rate.atFixedThickness)),
        thicknessRates, rate.thickness);
}

} // namespace

// The Derivatives facet of the triangular plate family: the semi-analytic derivatives difference the facets on the
// overlaid snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse ShellTria3DrillingElement::designFieldUse(const DesignedField& field) const
{
    // ShellProp: T, 12I/T**3 and TS/T enter the section (every computation); Z1 and Z2 are the fibre distances of
    // the stress recovery only; NSM the mass only. The element's ZOFFS, THETA or MCID and corner thicknesses enter
    // every computation. A pressure reads the element's geometry and offset only, never its property (ShellTria3DrillingLoads).
    // The ShellTria3Drilling (this element with a drilling membrane) inherits this facet.
    const FieldUse use = findFieldUse(field, {{"ShellProp", "fiberDistance1", FieldUse::recoveryOnly()},
                                              {"ShellProp", "fiberDistance2", FieldUse::recoveryOnly()},
                                              {"AnisoMat2D", "stiffness", ANISO_MAT2D_TERM_USE}});
    return field.card == card() ? use : use.withoutDistributedLoads();
}

// The closed forms: dK/dT and dM/dT of the ShellProp thickness T when every corner thickness follows it
// (ShellTria3DrillingPlate::thicknessRate, massThicknessRate); dK/dG of a AnisoMat2D stiffness term of the ShellProp
// (ShellTria3DrillingPlate::sectionRate); every other field and element is differenced.
bool ShellTria3DrillingElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                const DesignedField& field, ElementMatrixBlock& block) const
{
    if (isPlyThickness(field))
    {
        const Columns columns(view);
        const Data data = resolve(view, columns, row).required();
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.toDisplacementSystems(
                         plyThicknessStiffness(plate, *rate, plate.thicknessRate(data.section.thickness))));
        return true;
    }
    if (isAnisoMat2DStiffnessTerm(field))
    {
        const Columns columns(view);
        const Data data = resolve(view, columns, row).required();
        const std::optional<ShellSectionRate> rate =
            shellSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.toDisplacementSystems(plate.sectionRate(*rate).stiffness));
        return true;
    }
    if (!isShellThickness(field))
        return false;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(plate.thicknessRate(data.section.thickness)));
    return true;
}

// The stiffness rates of several fields from one resolution and one plate (stiffnessFieldDerivative's, field by field).
// PERFORMANCE: the fields of a lamination zone (its AnisoMat2D terms and T) reach the same elements: each resolved the
// element and built its plate again for every field; the rates of each field are those of the call alone, bit for bit
// (the same plate, the same operations).
// Measured on a laminated design model with lamination zones.
void ShellTria3DrillingElement::stiffnessFieldDerivatives(const ElementBatchView& view, std::int64_t row,
                                                 std::span<const DesignedField> fields,
                                                 std::span<ElementMatrixBlock> blocks,
                                                 std::span<std::uint8_t> isClosed) const
{
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const bool followsThickness = cornersFollowThickness(
        data.section, columns.thicknessFlag[static_cast<std::size_t>(row)], columns.cornerThickness[row]);
    std::optional<ShellTria3DrillingPlate> plate;
    const auto plateOf = [&]() -> const ShellTria3DrillingPlate&
    {
        if (!plate)
            plate.emplace(data.geometry, data.section, data.thickness);
        return *plate;
    };
    std::optional<base::MatN<18>> thicknessRate; // of a laminate's T, shared by its plies' fields
    const auto thicknessRateOf = [&]() -> const base::MatN<18>&
    {
        if (!thicknessRate)
            thicknessRate = plateOf().thicknessRate(data.section.thickness);
        return *thicknessRate;
    };
    for (std::size_t k = 0; k < fields.size(); ++k)
    {
        isClosed[k] = 0;
        if (isAnisoMat2DStiffnessTerm(fields[k]))
        {
            const std::optional<ShellSectionRate> rate =
                shellSectionRate(data.section, fields[k], data.materialAngle, takesCoupling());
            if (!rate)
                continue;
            appendMatrix(blocks[k], row, data.frames.dofs(),
                         data.frames.toDisplacementSystems(plateOf().sectionRate(*rate).stiffness));
            isClosed[k] = 1;
        }
        else if (isShellThickness(fields[k]) && followsThickness)
        {
            appendMatrix(blocks[k], row, data.frames.dofs(),
                         data.frames.toDisplacementSystems(plateOf().thicknessRate(data.section.thickness)));
            isClosed[k] = 1;
        }
        else if (isPlyThickness(fields[k]))
        {
            const std::optional<LaminateSectionRate> rate =
                laminateSectionRate(data.section, fields[k], data.materialAngle, takesCoupling());
            if (!rate)
                continue;
            appendMatrix(blocks[k], row, data.frames.dofs(),
                         data.frames.toDisplacementSystems(plyThicknessStiffness(plateOf(), *rate, thicknessRateOf())));
            isClosed[k] = 1;
        }
    }
}

bool ShellTria3DrillingElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                           const MassOptions& options, ElementMatrixBlock& block) const
{
    if (isPlyThickness(field))
    {
        // dM = M's rate in ρ at fixed T (massDensityRate) + dT · dM/dT at fixed ρ (massThicknessRate).
        const Columns columns(view);
        const Data data = resolve(view, columns, row).required();
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        base::MatN<18> mass = plate.massDensityRate(options.formulation, added, rate->density);
        mass += rate->thickness * plate.massThicknessRate(options.formulation, added, data.section.thickness);
        appendMatrix(block, row, data.frames.dofs(),
                     data.frames.massToDisplacementSystems(mass, view.isMassOnOffsets()));
        return true;
    }
    if (!isShellThickness(field))
        return false;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const double added =
        options.addedNonStructuralMass.empty() ? 0.0 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    appendMatrix(
        block, row, data.frames.dofs(),
        data.frames.massToDisplacementSystems(
            plate.massThicknessRate(options.formulation, added, data.section.thickness), view.isMassOnOffsets()));
    return true;
}

// dK_d/dv at a fixed preload, as ShellQuad4DrillingElement's: K_d(dN) of the rate of the centre's membrane resultants at fixed
// displacements (centreStateSectionRates, centreStateRates); none under a temperature change.
bool ShellTria3DrillingElement::differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                            const DesignedField& field,
                                                            const NodalDisplacements& preload,
                                                            ElementMatrixBlock& block) const
{
    const bool isThickness = isShellThickness(field);
    const bool isPly = isPlyThickness(field);
    if ((!isThickness && !isPly && !isAnisoMat2DStiffnessTerm(field)) || preload.temperatures != nullptr)
        return false;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 18> q = data.frames.localDisplacements(preload);
    const ShellPlaceState state = plate.centreState(q);
    std::array<double, 3> forceRate{};
    if (isPly)
    {
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        forceRate = plyThicknessState(plate, state, q, *rate, plate.centreStateRates(state, q, data.section.thickness))
                        .membraneForce;
    }
    else if (isThickness)
    {
        if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                    columns.cornerThickness[row]))
            return false;
        forceRate = plate.centreStateRates(state, q, data.section.thickness).membraneForce;
    }
    else
    {
        const std::optional<ShellSectionRate> rate =
            shellSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        forceRate = plate.centreStateSectionRates(state, q, *rate, plate.sectionRate(*rate)).membraneForce;
    }
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(plate.differentialStiffness(forceRate)));
    return true;
}

bool ShellTria3DrillingElement::differentialStiffnessFieldMixedDerivative(
    const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
    const NodalDisplacements& preload, ElementMatrixBlock& block) const
{
    if (!isShellThickness(first) || !isShellThickness(second) || first.id != second.id ||
        preload.temperatures != nullptr)
        return ElementFamily::differentialStiffnessFieldMixedDerivative(view, row, first, second, preload, block);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const auto q = data.frames.localDisplacements(preload);
    const auto state = plate.centreState(q);
    const auto secondRate = plate.centreStateSecondRates(state, q, data.section.thickness);
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(plate.differentialStiffness(secondRate.membraneForce)));
    return true;
}

// The rows' rates with respect to T at fixed displacements (ShellTria3DrillingPlate::centreStateRates, appendShellRowRates), or
// to a AnisoMat2D stiffness term without a temperature change (centreStateSectionRates).
bool ShellTria3DrillingElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementOutput output, const NodalDisplacements& displacements,
                                          ElementResultBlock& rates) const
{
    if (isPlyThickness(field))
    {
        // The forces only, as ShellQuad4DrillingElement's.
        if (displacements.temperatures != nullptr || output != ElementOutput::Forces)
            return false;
        requireOutput(output);
        const Columns columns(view);
        const Data data = resolve(view, columns, row).required();
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 18> q = data.frames.localDisplacements(displacements);
        const ShellPlaceState state = plate.centreState(q);
        rates.reset(outputColumns(output));
        appendShellRowRates(
            rates, row, view.elementId(row), state,
            plyThicknessState(plate, state, q, *rate, plate.centreStateRates(state, q, data.section.thickness)),
            data.section, output, compositeOutputOf(view.snapshot()));
        return true;
    }
    if (isAnisoMat2DStiffnessTerm(field))
    {
        if (displacements.temperatures != nullptr)
            return false;
        requireOutput(output);
        const Columns columns(view);
        const Data data = resolve(view, columns, row).required();
        const std::optional<ShellSectionRate> rate =
            shellSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 18> q = data.frames.localDisplacements(displacements);
        const ShellPlaceState state = plate.centreState(q);
        rates.reset(outputColumns(output));
        appendShellRowRates(rates, row, view.elementId(row), state,
                            plate.centreStateSectionRates(state, q, *rate, plate.sectionRate(*rate)), data.section,
                            output, compositeOutputOf(view.snapshot()));
        return true;
    }
    if (!isShellThickness(field))
        return false;
    requireOutput(output);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 18> q = data.frames.localDisplacements(displacements);
    const ShellTemperatureChange temperature =
        displacements.temperatures == nullptr
            ? ShellTemperatureChange{}
            : resolveShellTemperature(view, row, *displacements.temperatures, data.frames.gridRows, data.section)
                  .required();
    const double thickness = data.section.thickness;
    const ShellPlaceState state = plate.centreState(q, temperature);
    rates.reset(outputColumns(output));
    appendShellRowRates(rates, row, view.elementId(row), state, plate.centreStateRates(state, q, thickness),
                        data.section, output, compositeOutputOf(view.snapshot()));
    return true;
}

// The rows' rates of several fields from one resolution, one plate and one centre state (recoveryFieldRates', field
// by field), as ShellQuad4DrillingElement's.
// PERFORMANCE: as stiffnessFieldDerivatives, for the explicit parts of the items of a lamination zone's elements.
// Measured on a laminated design model with lamination zones.
void ShellTria3DrillingElement::recoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                                  std::span<const DesignedField> fields, ElementOutput output,
                                                  const NodalDisplacements& displacements,
                                                  std::span<ElementResultBlock> rates,
                                                  std::span<std::uint8_t> isClosed) const
{
    const bool hasSectionTerm = std::any_of(fields.begin(), fields.end(), isAnisoMat2DStiffnessTerm);
    if (!hasSectionTerm || displacements.temperatures != nullptr)
    {
        ElementFamily::recoveryFieldRatesOfFields(view, row, fields, output, displacements, rates, isClosed);
        return;
    }
    requireOutput(output);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 18> q = data.frames.localDisplacements(displacements);
    const ShellPlaceState state = plate.centreState(q);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::size_t k = 0; k < fields.size(); ++k)
    {
        isClosed[k] = 0;
        if (!isAnisoMat2DStiffnessTerm(fields[k]))
        {
            isClosed[k] = recoveryFieldRates(view, row, fields[k], output, displacements, rates[k]) ? 1 : 0;
            continue;
        }
        const std::optional<ShellSectionRate> rate =
            shellSectionRate(data.section, fields[k], data.materialAngle, takesCoupling());
        if (!rate)
            continue;
        rates[k].reset(outputColumns(output));
        appendShellRowRates(rates[k], row, view.elementId(row), state,
                            plate.centreStateSectionRates(state, q, *rate, plate.sectionRate(*rate)), data.section,
                            output, composite);
        isClosed[k] = 1;
    }
}

// The ply rows' rates along a ply thickness at fixed displacements, as ShellQuad4DrillingElement's.
bool ShellTria3DrillingElement::layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row,
                                               const DesignedField& field, LayerOutput output,
                                               const NodalDisplacements& displacements, ElementResultBlock& rates) const
{
    if (!isPlyThickness(field) || displacements.temperatures != nullptr)
        return false;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const std::optional<LaminateSectionRate> rate =
        laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
    if (!rate)
        return false;
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 18> q = data.frames.localDisplacements(displacements);
    const ShellPlaceState state = plate.centreState(q);
    rates.reset(layerOutputColumns(output));
    appendLaminateRowRates(
        rates, row, view.elementId(row), state,
        plyThicknessState(plate, state, q, *rate, plate.centreStateRates(state, q, data.section.thickness)),
        data.section, rate->laminate, output, compositeOutputOf(view.snapshot()));
    return true;
}

// The ply rows' rates of several ply thicknesses from one resolution, one plate and one centre state, as
// ShellQuad4DrillingElement's.
// PERFORMANCE: as ShellQuad4DrillingElement::layerRecoveryFieldRatesOfFields.
void ShellTria3DrillingElement::layerRecoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                                       std::span<const DesignedField> fields, LayerOutput output,
                                                       const NodalDisplacements& displacements,
                                                       std::span<ElementResultBlock> rates,
                                                       std::span<std::uint8_t> isClosed) const
{
    std::fill(isClosed.begin(), isClosed.end(), std::uint8_t{0});
    if (displacements.temperatures != nullptr || std::none_of(fields.begin(), fields.end(), isPlyThickness))
        return;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 18> q = data.frames.localDisplacements(displacements);
    const ShellPlaceState state = plate.centreState(q);
    const ShellPlaceState thicknessRates = plate.centreStateRates(state, q, data.section.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::size_t k = 0; k < fields.size(); ++k)
    {
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, fields[k], data.materialAngle, takesCoupling());
        if (!rate)
            continue;
        rates[k].reset(layerOutputColumns(output));
        appendLaminateRowRates(rates[k], row, view.elementId(row), state,
                               plyThicknessState(plate, state, q, *rate, thicknessRates), data.section, rate->laminate,
                               output, composite);
        isClosed[k] = 1;
    }
}

// The second derivatives with respect to T, as ShellQuad4DrillingElement's: ShellTria3DrillingPlate::thicknessSecondRate,
// massThicknessSecondRate, the slopes of the rows' rates at the unit local displacements, and centreStateSecondRates.
bool ShellTria3DrillingElement::stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                      const DesignedField& field, ElementMatrixBlock& block) const
{
    if (!isShellThickness(field))
        return false;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(plate.thicknessSecondRate(data.section.thickness)));
    return true;
}

bool ShellTria3DrillingElement::massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                 const DesignedField& field, const MassOptions& options,
                                                 ElementMatrixBlock& block) const
{
    if (!isShellThickness(field))
        return false;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const double added =
        options.addedNonStructuralMass.empty() ? 0.0 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    appendMatrix(
        block, row, data.frames.dofs(),
        data.frames.massToDisplacementSystems(
            plate.massThicknessSecondRate(options.formulation, added, data.section.thickness), view.isMassOnOffsets()));
    return true;
}

bool ShellTria3DrillingElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementOutput output, RecoverySlopes& rates) const
{
    if (!isShellThickness(field))
        return false;
    requireOutput(output);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    const double thickness = data.section.thickness;
    shellRecoverySlopes<3>(
        data.frames, shellOutputColumnForms(output), outputColumns(output),
        [&](const std::array<double, 18>& q, ElementResultBlock& block)
        {
            const ShellPlaceState state = plate.centreState(q);
            appendShellRowRates(block, row, elementId, state, plate.centreStateRates(state, q, thickness), data.section,
                                output, composite);
        },
        rates);
    return true;
}

bool ShellTria3DrillingElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
                                                const DesignedField& field, ElementOutput output,
                                                const NodalDisplacements& displacements,
                                                ElementResultBlock& rates) const
{
    if (!isShellThickness(field) || displacements.temperatures != nullptr)
        return false;
    requireOutput(output);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const ShellTria3DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 18> q = data.frames.localDisplacements(displacements);
    const double thickness = data.section.thickness;
    const ShellPlaceState state = plate.centreState(q);
    rates.reset(outputColumns(output));
    appendShellRowSecondRates(rates, row, view.elementId(row), state, plate.centreStateRates(state, q, thickness),
                              plate.centreStateSecondRates(state, q, thickness), data.section, output,
                              compositeOutputOf(view.snapshot()));
    return true;
}

} // namespace oos::elements
