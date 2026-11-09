#include "oos/elements/shells/ShellRecoverySlopes.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingPlate.hpp"

namespace oos::elements
{

namespace
{

// What a stiffness term of a AnisoMat2D changes in a shell: the stiffness, the thermal loads (N_T = t G1 α ΔT) and the
// recovery; not the mass, which reads ρ and the thickness only (ShellQuad4DrillingPlate::massOf).
constexpr FieldUse ANISO_MAT2D_TERM_USE{true, false, true, true, false};

// dK24 along a ply thickness: the rate of the laminate's section at fixed T and T's,
// dK = sectionRate(atFixedThickness) + dT · thicknessRate(T) (a laminate's corners are its thickness), the plate's
// thicknessRate(T) given (one per element for all its plies).
[[nodiscard]] base::MatN<24> plyThicknessStiffness(const ShellQuad4DrillingPlate& plate, const LaminateSectionRate& rate,
                                                   const base::MatN<24>& thicknessRate)
{
    base::MatN<24> k = plate.sectionRate(rate.atFixedThickness).stiffness;
    k += rate.thickness * thicknessRate;
    return k;
}

// The rates of the place states along a ply thickness at fixed q: the section's at fixed T and T's, the plate's
// placeStateRates(T) given.
[[nodiscard]] std::array<ShellPlaceState, 5> plyThicknessStates(const ShellQuad4DrillingPlate& plate,
                                                                const std::array<ShellPlaceState, 5>& states,
                                                                const std::array<double, 24>& q,
                                                                const LaminateSectionRate& rate,
                                                                const std::array<ShellPlaceState, 5>& thicknessRates)
{
    const std::array<ShellPlaceState, 5> sectionRates =
        plate.placeStateSectionRates(states, q, rate.atFixedThickness, plate.sectionRate(rate.atFixedThickness, false));
    std::array<ShellPlaceState, 5> rates;
    for (std::size_t place = 0; place < states.size(); ++place)
        rates[place] = combinedStateRates(sectionRates[place], thicknessRates[place], rate.thickness);
    return rates;
}

} // namespace

// The Derivatives facet of the plate family: the semi-analytic derivatives difference the facets on the overlaid
// snapshots (design/sensitivity/ElementDerivatives); this says what each designed field can change.
FieldUse ShellQuad4DrillingElement::designFieldUse(const DesignedField& field) const
{
    // ShellProp: T, 12I/T**3 and TS/T enter the section (every computation); Z1 and Z2 are the fibre distances of
    // the stress recovery only; NSM the mass only. The element's ZOFFS, THETA or MCID and corner thicknesses enter
    // every computation. A pressure reads the element's geometry and offset only, never its property (ShellQuad4DrillingLoads).
    // The ShellQuad4Drilling (this element with a drilling membrane) inherits this facet.
    const FieldUse use = findFieldUse(field, {{"ShellProp", "fiberDistance1", FieldUse::recoveryOnly()},
                                              {"ShellProp", "fiberDistance2", FieldUse::recoveryOnly()},
                                              {"AnisoMat2D", "stiffness", ANISO_MAT2D_TERM_USE}});
    return field.card == card() ? use : use.withoutDistributedLoads();
}

// The closed forms: dK/dT and dM/dT of the ShellProp thickness T when every corner thickness follows it
// (ShellQuad4DrillingPlate::thicknessRate, massThicknessRate); dK/dG of a AnisoMat2D stiffness term of the ShellProp
// (ShellQuad4DrillingPlate::sectionRate, where shellSectionRate gives the section's rate); every other field and element is
// differenced.
bool ShellQuad4DrillingElement::stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
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
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
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
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(plate.thicknessRate(data.section.thickness)));
    return true;
}

// The stiffness rates of several fields from one resolution and one plate (stiffnessFieldDerivative's, field by field).
// PERFORMANCE: the fields of a lamination zone (its AnisoMat2D terms and T) reach the same elements: each resolved the
// element and built its plate again for every field; the rates of each field are those of the call alone, bit for bit
// (the same plate, the same operations).
// Measured on a laminated design model with lamination zones.
void ShellQuad4DrillingElement::stiffnessFieldDerivatives(const ElementBatchView& view, std::int64_t row,
                                                 std::span<const DesignedField> fields,
                                                 std::span<ElementMatrixBlock> blocks,
                                                 std::span<std::uint8_t> isClosed) const
{
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const bool followsThickness = cornersFollowThickness(
        data.section, columns.thicknessFlag[static_cast<std::size_t>(row)], columns.cornerThickness[row]);
    std::optional<ShellQuad4DrillingPlate> plate;
    const auto plateOf = [&]() -> const ShellQuad4DrillingPlate&
    {
        if (!plate)
            plate.emplace(data.geometry, data.section, data.thickness);
        return *plate;
    };
    std::optional<base::MatN<24>> thicknessRate; // of a laminate's T, shared by its plies' fields
    const auto thicknessRateOf = [&]() -> const base::MatN<24>&
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

bool ShellQuad4DrillingElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                           const MassOptions& options, ElementMatrixBlock& block) const
{
    if (isPlyThickness(field))
    {
        // The mass is linear in ρ at fixed T (ρ t, ρ t³/12; the NSM moves with no ply), so dM = M(dρ at fixed T,
        // no NSM) + dT · dM/dT at fixed ρ.
        const Columns columns(view);
        const Data data = resolve(view, columns, row).required();
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        ShellSection densityRate = data.section;
        densityRate.density = rate->density;
        densityRate.nonStructuralMass = 0.0;
        base::MatN<24> mass =
            ShellQuad4DrillingPlate::massOf(data.geometry, densityRate, data.thickness, options.formulation, 0.0);
        mass += rate->thickness * ShellQuad4DrillingPlate::massThicknessRateOf(data.geometry, data.section, data.thickness,
                                                                      options.formulation, data.section.thickness);
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
    // The NSM does not follow T (ShellQuad4DrillingPlate::massThicknessRateOf leaves it out); no plate is built.
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.massToDisplacementSystems(
                     ShellQuad4DrillingPlate::massThicknessRateOf(data.geometry, data.section, data.thickness,
                                                         options.formulation, data.section.thickness),
                     view.isMassOnOffsets()));
    return true;
}

// dK_d/dv at a fixed preload: K_d is linear in the centre's membrane resultants N of the preload and reads the geometry
// otherwise (ShellQuad4DrillingPlate::differentialStiffness), so dK_d = K_d(dN), dN their rate at fixed displacements: along a
// AnisoMat2D term (placeStateSectionRates) or T (placeStateRates, under its condition); none under a temperature change
// of the preload (N_T moves with both).
bool ShellQuad4DrillingElement::differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 24> q = data.frames.localDisplacements(preload);
    const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds);
    std::array<double, 3> forceRate{};
    if (isPly)
    {
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        forceRate =
            plyThicknessStates(plate, states, q, *rate, plate.placeStateRates(states, q, data.section.thickness))[0]
                .membraneForce;
    }
    else if (isThickness)
    {
        if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                    columns.cornerThickness[row]))
            return false;
        forceRate = plate.placeStateRates(states, q, data.section.thickness)[0].membraneForce;
    }
    else
    {
        const std::optional<ShellSectionRate> rate =
            shellSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        forceRate = plate.placeStateSectionRates(states, q, *rate, plate.sectionRate(*rate, false))[0].membraneForce;
    }
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(plate.differentialStiffness(forceRate)));
    return true;
}

// The rows' rates with respect to T at fixed displacements (ShellQuad4DrillingPlate::placeStateRates, appendShellRowRates), or
// to a AnisoMat2D stiffness term without a temperature change (placeStateSectionRates).
bool ShellQuad4DrillingElement::recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementOutput output, const NodalDisplacements& displacements,
                                          ElementResultBlock& rates) const
{
    if (isPlyThickness(field))
    {
        // The forces (the stresses and strains of a laminate are its plies', layerRecoveryFieldRates; its
        // homogeneous rows under NOCOMPS read fibres the plies move, not given).
        if (displacements.temperatures != nullptr || output != ElementOutput::Forces)
            return false;
        requireOutput(output);
        const Columns columns(view);
        const Data data = resolve(view, columns, row).required();
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, field, data.materialAngle, takesCoupling());
        if (!rate)
            return false;
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 24> q = data.frames.localDisplacements(displacements);
        const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds);
        const std::array<ShellPlaceState, 5> stateRates =
            plyThicknessStates(plate, states, q, *rate, plate.placeStateRates(states, q, data.section.thickness));
        const CompositeOutput composite = compositeOutputOf(view.snapshot());
        rates.reset(outputColumns(output));
        for (std::size_t place = 0; place < states.size(); ++place)
            appendShellRowRates(rates, row, view.elementId(row), states[place], stateRates[place], data.section, output,
                                composite);
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
        const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
        const std::array<double, 24> q = data.frames.localDisplacements(displacements);
        const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds);
        const std::array<ShellPlaceState, 5> stateRates =
            plate.placeStateSectionRates(states, q, *rate, plate.sectionRate(*rate, false));
        const CompositeOutput composite = compositeOutputOf(view.snapshot());
        rates.reset(outputColumns(output));
        for (std::size_t place = 0; place < states.size(); ++place)
            appendShellRowRates(rates, row, view.elementId(row), states[place], stateRates[place], data.section, output,
                                composite);
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 24> q = data.frames.localDisplacements(displacements);
    const ShellTemperatureChange temperature =
        displacements.temperatures == nullptr
            ? ShellTemperatureChange{}
            : resolveShellTemperature(view, row, *displacements.temperatures, data.frames.gridRows, data.section)
                  .required();
    const double thickness = data.section.thickness;
    const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds, temperature);
    const std::array<ShellPlaceState, 5> stateRates = plate.placeStateRates(states, q, thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    rates.reset(outputColumns(output));
    for (std::size_t place = 0; place < states.size(); ++place)
        appendShellRowRates(rates, row, view.elementId(row), states[place], stateRates[place], data.section, output,
                            composite);
    return true;
}

// The rows' rates of several fields from one resolution, one plate and one set of place states (recoveryFieldRates',
// field by field: a AnisoMat2D term or T; another field gets none here and is asked alone).
// PERFORMANCE: as stiffnessFieldDerivatives, for the explicit parts of the items of a lamination zone's elements.
// Measured on a laminated design model with lamination zones.
void ShellQuad4DrillingElement::recoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                                  std::span<const DesignedField> fields, ElementOutput output,
                                                  const NodalDisplacements& displacements,
                                                  std::span<ElementResultBlock> rates,
                                                  std::span<std::uint8_t> isClosed) const
{
    const bool hasSectionTerm =
        std::any_of(fields.begin(), fields.end(), isAnisoMat2DStiffnessTerm) ||
        (output == ElementOutput::Forces && std::any_of(fields.begin(), fields.end(), isPlyThickness));
    if (!hasSectionTerm || displacements.temperatures != nullptr)
    {
        ElementFamily::recoveryFieldRatesOfFields(view, row, fields, output, displacements, rates, isClosed);
        return;
    }
    requireOutput(output);
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 24> q = data.frames.localDisplacements(displacements);
    const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    std::optional<std::array<ShellPlaceState, 5>> thicknessRates; // of a laminate's T, shared by its plies' fields
    for (std::size_t k = 0; k < fields.size(); ++k)
    {
        isClosed[k] = 0;
        if (isPlyThickness(fields[k]) && output == ElementOutput::Forces)
        {
            const std::optional<LaminateSectionRate> rate =
                laminateSectionRate(data.section, fields[k], data.materialAngle, takesCoupling());
            if (!rate)
                continue;
            if (!thicknessRates)
                thicknessRates = plate.placeStateRates(states, q, data.section.thickness);
            const std::array<ShellPlaceState, 5> stateRates =
                plyThicknessStates(plate, states, q, *rate, *thicknessRates);
            rates[k].reset(outputColumns(output));
            for (std::size_t place = 0; place < states.size(); ++place)
                appendShellRowRates(rates[k], row, view.elementId(row), states[place], stateRates[place], data.section,
                                    output, composite);
            isClosed[k] = 1;
            continue;
        }
        if (!isAnisoMat2DStiffnessTerm(fields[k]))
        {
            isClosed[k] = recoveryFieldRates(view, row, fields[k], output, displacements, rates[k]) ? 1 : 0;
            continue;
        }
        const std::optional<ShellSectionRate> rate =
            shellSectionRate(data.section, fields[k], data.materialAngle, takesCoupling());
        if (!rate)
            continue;
        const std::array<ShellPlaceState, 5> stateRates =
            plate.placeStateSectionRates(states, q, *rate, plate.sectionRate(*rate, false));
        rates[k].reset(outputColumns(output));
        for (std::size_t place = 0; place < states.size(); ++place)
            appendShellRowRates(rates[k], row, view.elementId(row), states[place], stateRates[place], data.section,
                                output, composite);
        isClosed[k] = 1;
    }
}

// The ply rows' rates along a ply thickness at fixed displacements: the centre's state rates (plyThicknessStates)
// through the plies (appendLaminateRowRates); none under a temperature change.
bool ShellQuad4DrillingElement::layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row,
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 24> q = data.frames.localDisplacements(displacements);
    const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds);
    const std::array<ShellPlaceState, 5> stateRates =
        plyThicknessStates(plate, states, q, *rate, plate.placeStateRates(states, q, data.section.thickness));
    rates.reset(layerOutputColumns(output));
    appendLaminateRowRates(rates, row, view.elementId(row), states[0], stateRates[0], data.section, rate->laminate,
                           output, compositeOutputOf(view.snapshot()));
    return true;
}

// The ply rows' rates of several ply thicknesses from one resolution, one plate, one set of place states and their T
// rates (layerRecoveryFieldRates', field by field).
// PERFORMANCE: the 16 ply fields of a CompProp zone reach the same elements: each resolved the element (its laminate),
// built its plate and recovered its states again per field; the rates of each field are those of the call alone, bit
// for bit (the same plate, the same operations).
// Measured on a CompProp plate.
void ShellQuad4DrillingElement::layerRecoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 24> q = data.frames.localDisplacements(displacements);
    const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds);
    const std::array<ShellPlaceState, 5> thicknessRates = plate.placeStateRates(states, q, data.section.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::size_t k = 0; k < fields.size(); ++k)
    {
        const std::optional<LaminateSectionRate> rate =
            laminateSectionRate(data.section, fields[k], data.materialAngle, takesCoupling());
        if (!rate)
            continue;
        const std::array<ShellPlaceState, 5> stateRates = plyThicknessStates(plate, states, q, *rate, thicknessRates);
        rates[k].reset(layerOutputColumns(output));
        appendLaminateRowRates(rates[k], row, view.elementId(row), states[0], stateRates[0], data.section,
                               rate->laminate, output, composite);
        isClosed[k] = 1;
    }
}

// The second derivatives with respect to T: d²K/dT² and d²M/dT² of the plate (ShellQuad4DrillingPlate::thicknessSecondRate,
// massThicknessSecondRateOf), the slopes of the rows' rates (the rates at the unit local displacements,
// shellRecoverySlopes: exact, the rates of the Linear columns are linear in q) and the rows' second rates
// (placeStateSecondRates, appendShellRowSecondRates), under the condition of the first ones; none under a temperature
// change (the thermal resultants' dependence on T is not in the closed forms).
bool ShellQuad4DrillingElement::stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                      const DesignedField& field, ElementMatrixBlock& block) const
{
    if (!isShellThickness(field))
        return false;
    const Columns columns(view);
    const Data data = resolve(view, columns, row).required();
    if (!cornersFollowThickness(data.section, columns.thicknessFlag[static_cast<std::size_t>(row)],
                                columns.cornerThickness[row]))
        return false;
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.toDisplacementSystems(plate.thicknessSecondRate(data.section.thickness)));
    return true;
}

bool ShellQuad4DrillingElement::massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
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
    appendMatrix(block, row, data.frames.dofs(),
                 data.frames.massToDisplacementSystems(
                     ShellQuad4DrillingPlate::massThicknessSecondRateOf(data.geometry, data.section, data.thickness,
                                                               options.formulation, data.section.thickness),
                     view.isMassOnOffsets()));
    return true;
}

bool ShellQuad4DrillingElement::recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    const std::int32_t elementId = view.elementId(row);
    const double thickness = data.section.thickness;
    const ShellQuad4DrillingPlate::PlaceOperators operators = plate.placeOperators();
    const ShellQuad4DrillingPlate::PlaceThermal atRest = plate.placeThermal(operators, {});
    shellRecoverySlopes<4>(
        data.frames, shellOutputColumnForms(output), outputColumns(output),
        [&](const std::array<double, 24>& q, ElementResultBlock& block)
        {
            const std::array<ShellPlaceState, 5> states = plate.placeStates(operators, atRest, q, data.gridIds);
            const std::array<ShellPlaceState, 5> stateRates = plate.placeStateRates(states, q, thickness);
            for (std::size_t place = 0; place < states.size(); ++place)
                appendShellRowRates(block, row, elementId, states[place], stateRates[place], data.section, output,
                                    composite);
        },
        rates);
    return true;
}

bool ShellQuad4DrillingElement::recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row,
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
    const ShellQuad4DrillingPlate plate(data.geometry, data.section, data.thickness);
    const std::array<double, 24> q = data.frames.localDisplacements(displacements);
    const double thickness = data.section.thickness;
    const std::array<ShellPlaceState, 5> states = plate.placeStates(q, data.gridIds);
    const std::array<ShellPlaceState, 5> stateRates = plate.placeStateRates(states, q, thickness);
    const std::array<ShellPlaceState, 5> stateSecondRates = plate.placeStateSecondRates(states, q, thickness);
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    rates.reset(outputColumns(output));
    for (std::size_t place = 0; place < states.size(); ++place)
        appendShellRowSecondRates(rates, row, view.elementId(row), states[place], stateRates[place],
                                  stateSecondRates[place], data.section, output, composite);
    return true;
}

} // namespace oos::elements
