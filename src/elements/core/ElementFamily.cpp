#include "oos/elements/core/ElementFamily.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace oos::elements
{

namespace
{

// The element types with offsets (MDLPRM OFFDEF: the shells' ZOFFS, the bars' and beams' WA, WB).
constexpr std::array<std::string_view, 6> OFFSET_SHELLS{"ShellQuad4",        "ShellQuad4Drilling", "ShellTria3",
                                                        "ShellTria3Drilling", "ShellQuad8",        "ShellTria6"};
constexpr std::array<std::string_view, 2> OFFSET_BEAMS{"Bar2", "Beam2"};

// The offset elements of a batch: their count, the first one's row and the field of its offset.
struct OffsetUse
{
    std::int64_t count = 0;
    std::int64_t firstRow = -1;
    std::string field;

    void add(std::int64_t row, const char* name)
    {
        if (count++ == 0)
        {
            firstRow = row;
            field = name;
        }
    }
};

bool isOffset(std::span<const double> offset)
{
    return std::any_of(offset.begin(), offset.end(), [](double value) { return value != 0.0; });
}

OffsetUse offsetUseOf(const ElementBatchView& view, std::string_view card)
{
    OffsetUse use;
    if (std::find(OFFSET_SHELLS.begin(), OFFSET_SHELLS.end(), card) != OFFSET_SHELLS.end())
    {
        const std::span<const double> offset = view.columns().values<double>("offset");
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
            if (offset[static_cast<std::size_t>(row)] != 0.0)
                use.add(row, "offset");
    }
    else if (std::find(OFFSET_BEAMS.begin(), OFFSET_BEAMS.end(), card) != OFFSET_BEAMS.end())
    {
        const base::FixedListView<double> offsetA = view.columns().fixedLists<double>("offsetA");
        const base::FixedListView<double> offsetB = view.columns().fixedLists<double>("offsetB");
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
            if (isOffset(offsetA[row]) || isOffset(offsetB[row]))
                use.add(row, isOffset(offsetA[row]) ? "offsetA" : "offsetB");
    }
    return use;
}

} // namespace

void ElementFamily::checkRows(const ElementBatchView&, ElementRange, base::DiagnosticSink&) const
{
    throw std::logic_error(base::concatenate({"ElementFamily::checkRows: the family of ", std::string(card()),
                                              " checks its batch whole (isCheckedByRow false); call check() (a bug of "
                                              "O_o_Solver: report it)"}));
}

void ElementFamily::computeStiffness(const ElementBatchView& /*view*/, ElementRange /*range*/,
                                     ElementMatrixBlock& /*block*/) const
{
    refuseFacet("stiffness");
}

void ElementFamily::computeMass(const ElementBatchView& /*view*/, ElementRange /*range*/,
                                const MassOptions& /*options*/, ElementMatrixBlock& /*block*/) const
{
    refuseFacet("mass");
}

std::vector<std::string> ElementFamily::outputColumns(ElementOutput /*output*/) const
{
    return {};
}

void ElementFamily::recover(const ElementBatchView& /*view*/, ElementRange /*range*/, ElementOutput /*output*/,
                            const NodalDisplacements& /*displacements*/, ElementResultBlock& /*block*/) const
{
    refuseFacet("recovery");
}

namespace
{

// The element types without thermal expansion in a linear analysis: scalar springs and bushes (their properties
// SpringProp, BushProp, BushProp1D, GapProp have no expansion coefficient), fasteners (PFAST, WeldProp: springs between
// pierced elements), masses (no stiffness), and the shear panel (CSHEAR: a pure shear state, which a temperature change
// does not strain; a design choice of this project).
constexpr std::array<std::string_view, 18> THERMALLY_INERT_CARDS{
    "SpringDofProperty", "SpringDofInline",  "SpringScalarProperty", "SpringScalarInline",    "Bushing",    "Bushing1D",
    "Gap2",     "Fastener", "Weld",    "PointMassMatrix",     "PointMassInertia",  "MassDofProperty",
    "MassDofInline",   "MassScalarProperty",    "MassScalarInline",   "ShearPanel4", "DamperDofProperty", "DamperDofInline"};

} // namespace

bool ElementFamily::isThermallyInert() const noexcept
{
    return !facets().has(Facet::Stiffness) ||
           std::find(THERMALLY_INERT_CARDS.begin(), THERMALLY_INERT_CARDS.end(), card()) != THERMALLY_INERT_CARDS.end();
}

void ElementFamily::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                      base::DiagnosticSink& sink) const
{
    // Without a temperature load (a DEFORM alone) a family takes no thermal strain: nothing to refuse (the
    // elements a DEFORM names are checked by assembly::LoadVectorBuilder::checkThermalLoads).
    if (isThermallyInert() || view.rowCount() == 0 || !temperatures.hasTemperatureLoad)
        return;
    base::Diagnostic diagnostic;
    diagnostic.code = "unsupported-load";
    diagnostic.card = std::string(card());
    diagnostic.id = view.elementId(0);
    diagnostic.field = "TemperatureSelection(LOAD)";
    diagnostic.message = base::concatenate(
        {"expected elements whose thermal loads this version computes, found ", std::to_string(view.rowCount()), " ",
         card(), " under a TemperatureSelection(LOAD) of the subcase; remove the thermal load or these elements"});
    sink.report(std::move(diagnostic));
}

void ElementFamily::computeThermalLoads(const ElementBatchView& /*view*/, ElementRange /*range*/,
                                        const ElementTemperatures& temperatures, ElementVectorBlock& /*block*/) const
{
    if (!isThermallyInert() && temperatures.hasTemperatureLoad)
        refuseFacet("thermal loads");
}

namespace
{

// The element types with stiffness but without differential stiffness in a linear buckling analysis (a design choice
// of this project): the differential stiffness is formed for the structural elements whose internal forces turn with
// the structure — rods, bars, beams, shells, solids; the scalar springs act along fixed DOFs, and the bushes, gaps and
// fasteners are local springs whose force is not rotated in SOL 105.
constexpr std::array<std::string_view, 9> DIFFERENTIALLY_INERT_CARDS{"SpringDofProperty", "SpringDofInline",  "SpringScalarProperty",
                                                                     "SpringScalarInline", "Bushing",     "Bushing1D",
                                                                     "Gap2",     "Fastener", "Weld"};

} // namespace

bool ElementFamily::isDifferentiallyInert() const noexcept
{
    return !facets().has(Facet::Stiffness) ||
           std::find(DIFFERENTIALLY_INERT_CARDS.begin(), DIFFERENTIALLY_INERT_CARDS.end(), card()) !=
               DIFFERENTIALLY_INERT_CARDS.end();
}

void ElementFamily::checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    // Under ModelParams OFFDEF ELMOFF, the default, the offsets take no part in the differential stiffness, so an
    // offset element is refused in a linear buckling analysis (SOL 105).
    if (view.rowCount() > 0 && !view.hasOffsetDifferentialStiffness())
        if (const OffsetUse use = offsetUseOf(view, card()); use.count > 0)
        {
            base::Diagnostic diagnostic;
            diagnostic.code = "unsupported-field";
            diagnostic.card = std::string(card());
            diagnostic.id = view.elementId(use.firstRow);
            diagnostic.field = use.field;
            diagnostic.message =
                base::concatenate({"expected no offset on an element of a buckling analysis under ModelParams "
                                   "OffsetDefinition ElemOffsetsOnly (the "
                                   "default: "
                                   "no differential stiffness for offsets), found ",
                                   std::to_string(use.count), " ", card(), " with an offset (", card(), ".", use.field,
                                   " of ", card(), " ", std::to_string(view.elementId(use.firstRow)),
                                   " first); add ModelParams OffsetDefinition ElemOffsetZ to the deck (the offsets in "
                                   "the mass, the loads and the "
                                   "differential stiffness), or remove the offsets"});
            sink.report(std::move(diagnostic));
        }
    if (isDifferentiallyInert() || facets().has(Facet::Differential) || view.rowCount() == 0)
        return;
    base::Diagnostic diagnostic;
    diagnostic.code = "unsupported-card";
    diagnostic.card = std::string(card());
    diagnostic.id = view.elementId(0);
    diagnostic.message = base::concatenate(
        {"expected elements whose differential stiffness this version computes (SOL 105), found ",
         std::to_string(view.rowCount()), " ", card(),
         "; their stiffness under the preload is unknown: remove them or run another solution sequence"});
    sink.report(std::move(diagnostic));
}

void ElementFamily::computeDifferentialStiffness(const ElementBatchView& /*view*/, ElementRange /*range*/,
                                                 const NodalDisplacements& /*preload*/,
                                                 ElementMatrixBlock& /*block*/) const
{
    if (!isDifferentiallyInert())
        refuseFacet("differential stiffness");
}

std::vector<std::string> ElementFamily::layerOutputColumns(LayerOutput /*output*/) const
{
    return {};
}

void ElementFamily::recoverLayers(const ElementBatchView& /*view*/, ElementRange /*range*/, LayerOutput /*output*/,
                                  const NodalDisplacements& /*displacements*/, ElementResultBlock& /*block*/) const
{
    refuseFacet("layer recovery");
}

std::string ElementFamily::layerResultElementType() const
{
    return resultElementType(false).append("Composite");
}

std::string ElementFamily::resultElementType(bool /*underLineLoads*/) const
{
    // Model records and result tables use the exact same element type.
    return std::string(card());
}

void ElementFamily::checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                       base::DiagnosticSink& sink) const
{
    for (const ElementPressure& pressure : pressures)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "unsupported-load";
        diagnostic.card = std::string(pressure.card);
        diagnostic.id = pressure.setId;
        diagnostic.field = "elemId";
        diagnostic.message = base::concatenate({"expected a pressure on an element that takes one, found one on ",
                                                card(), " ", std::to_string(view.elementId(pressure.elementRow)),
                                                ", which takes no pressure; remove the load"});
        sink.report(std::move(diagnostic));
    }
}

void ElementFamily::computePressureLoads(const ElementBatchView& /*view*/,
                                         std::span<const ElementPressure> /*pressures*/,
                                         ElementVectorBlock& /*block*/) const
{
    refuseFacet("pressure loads");
}

void ElementFamily::checkLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                   base::DiagnosticSink& sink) const
{
    for (const ElementLineLoad& load : loads)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "unsupported-load";
        diagnostic.card = std::string(load.card);
        diagnostic.id = load.setId;
        diagnostic.field = "elemId";
        diagnostic.message = base::concatenate({"expected a line load on a bar or beam, found one on ", card(), " ",
                                                std::to_string(view.elementId(load.elementRow)),
                                                ", which takes no line load; remove the load"});
        sink.report(std::move(diagnostic));
    }
}

void ElementFamily::computeLineLoads(const ElementBatchView& /*view*/, std::span<const ElementLineLoad> /*loads*/,
                                     ElementVectorBlock& /*block*/) const
{
    refuseFacet("line loads");
}

void ElementFamily::recoverUnderLineLoads(const ElementBatchView& /*view*/, std::span<const ElementLineLoad> /*loads*/,
                                          ElementOutput /*output*/, const NodalDisplacements& /*displacements*/,
                                          ElementResultBlock& /*block*/) const
{
    refuseFacet("recovery under line loads");
}

void ElementFamily::requireOutput(ElementOutput output) const
{
    if (outputColumns(output).empty())
        throw std::invalid_argument(base::concatenate(
            {"the element family ", card(), " gives no output of kind ", std::to_string(static_cast<int>(output)),
             "; ask outputColumns() before calling recover() (an empty list means the kind is not given)"}));
}

void ElementFamily::refuseFacet(std::string_view facet) const
{
    throw std::invalid_argument(base::concatenate(
        {"the element family ", card(), " has no ", facet,
         " facet; ask facets() before calling a facet (the assembler skips the families without it)"}));
}

void ElementFamily::recoverOutputs(const ElementBatchView& view, ElementRange range,
                                   std::span<const RecoveryOutput> outputs, const NodalDisplacements& displacements,
                                   std::span<ElementResultBlock> blocks) const
{
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        if (outputs[k].isLayer)
            recoverLayers(view, range, outputs[k].layerOutput, displacements, blocks[k]);
        else
            recover(view, range, outputs[k].output, displacements, blocks[k]);
    }
}

FieldUse ElementFamily::designFieldUse(const DesignedField& /*field*/) const
{
    return FieldUse::all();
}

std::vector<ColumnForm> ElementFamily::outputColumnForms(ElementOutput /*output*/) const
{
    return {};
}

std::vector<ColumnForm> ElementFamily::layerOutputColumnForms(LayerOutput /*output*/) const
{
    return {};
}

bool ElementFamily::recoverySlopes(const ElementBatchView& /*view*/, std::int64_t /*row*/, ElementOutput /*output*/,
                                   RecoverySlopes& /*slopes*/) const
{
    return false;
}

bool ElementFamily::recoverAtUnitDisplacements(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                               ElementOutput /*output*/, std::vector<ElementDof>& /*dofs*/,
                                               std::vector<ElementResultBlock>& /*blocks*/) const
{
    return false;
}

bool ElementFamily::layerRecoverySlopes(const ElementBatchView& /*view*/, std::int64_t /*row*/, LayerOutput /*output*/,
                                        RecoverySlopes& /*slopes*/) const
{
    return false;
}

std::size_t ElementFamily::vectorisedRecoveryMinCases(ElementOutput /*output*/) const
{
    return 0;
}

std::size_t ElementFamily::vectorisedLayerRecoveryMinCases(LayerOutput /*output*/) const
{
    return 0;
}

bool ElementFamily::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                            std::span<const RecoveryOutput> outputs,
                                            std::span<RecoverySlopes> slopes) const
{
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        const bool isGiven = outputs[k].isLayer ? layerRecoverySlopes(view, row, outputs[k].layerOutput, slopes[k])
                                                : recoverySlopes(view, row, outputs[k].output, slopes[k]);
        if (!isGiven)
            return false;
    }
    return true;
}

void ElementFamily::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput /*output*/,
                                          std::span<ElementResultBlock> /*blocks*/) const
{
    refuseFacet("vectorised recovery");
}

ElementOutput ElementFamily::vectorisedSource(ElementOutput output) const
{
    return output;
}

void ElementFamily::completeRecoveredRowsFromSource(const ElementBatchView& /*view*/, ElementOutput /*output*/,
                                                    std::span<const ElementResultBlock> /*sources*/,
                                                    std::span<ElementResultBlock> /*blocks*/) const
{
    refuseFacet("vectorised recovery from a source output");
}

LayerOutput ElementFamily::vectorisedLayerSource(LayerOutput output) const
{
    return output;
}

void ElementFamily::completeRecoveredLayerRows(const ElementBatchView& /*view*/, LayerOutput /*output*/,
                                               std::span<const ElementResultBlock> /*sources*/,
                                               std::span<ElementResultBlock> /*blocks*/) const
{
    refuseFacet("vectorised layer recovery");
}

bool ElementFamily::stiffnessFieldDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                             const DesignedField& /*field*/, ElementMatrixBlock& /*block*/) const
{
    return false;
}

void ElementFamily::stiffnessFieldDerivatives(const ElementBatchView& view, std::int64_t row,
                                              std::span<const DesignedField> fields,
                                              std::span<ElementMatrixBlock> blocks,
                                              std::span<std::uint8_t> isClosed) const
{
    for (std::size_t k = 0; k < fields.size(); ++k)
        isClosed[k] = stiffnessFieldDerivative(view, row, fields[k], blocks[k]) ? 1 : 0;
}

bool ElementFamily::massFieldDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                        const DesignedField& /*field*/, const MassOptions& /*options*/,
                                        ElementMatrixBlock& /*block*/) const
{
    return false;
}

bool ElementFamily::differentialStiffnessFieldDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                         const DesignedField& /*field*/,
                                                         const NodalDisplacements& /*preload*/,
                                                         ElementMatrixBlock& /*block*/) const
{
    return false;
}

bool ElementFamily::recoveryFieldRates(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                       const DesignedField& /*field*/, ElementOutput /*output*/,
                                       const NodalDisplacements& /*displacements*/, ElementResultBlock& /*rates*/) const
{
    return false;
}

void ElementFamily::recoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                               std::span<const DesignedField> fields, ElementOutput output,
                                               const NodalDisplacements& displacements,
                                               std::span<ElementResultBlock> rates,
                                               std::span<std::uint8_t> isClosed) const
{
    for (std::size_t k = 0; k < fields.size(); ++k)
        isClosed[k] = recoveryFieldRates(view, row, fields[k], output, displacements, rates[k]) ? 1 : 0;
}

bool ElementFamily::layerRecoveryFieldRates(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                            const DesignedField& /*field*/, LayerOutput /*output*/,
                                            const NodalDisplacements& /*displacements*/,
                                            ElementResultBlock& /*rates*/) const
{
    return false;
}

void ElementFamily::layerRecoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                                    std::span<const DesignedField> fields, LayerOutput output,
                                                    const NodalDisplacements& displacements,
                                                    std::span<ElementResultBlock> rates,
                                                    std::span<std::uint8_t> isClosed) const
{
    for (std::size_t k = 0; k < fields.size(); ++k)
        isClosed[k] = layerRecoveryFieldRates(view, row, fields[k], output, displacements, rates[k]) ? 1 : 0;
}

bool ElementFamily::stiffnessFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                   const DesignedField& /*field*/, ElementMatrixBlock& /*block*/) const
{
    return false;
}

bool ElementFamily::massFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                              const DesignedField& /*field*/, const MassOptions& /*options*/,
                                              ElementMatrixBlock& /*block*/) const
{
    return false;
}

bool ElementFamily::recoverySlopeRates(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                       const DesignedField& /*field*/, ElementOutput /*output*/,
                                       RecoverySlopes& /*rates*/) const
{
    return false;
}

bool ElementFamily::recoveryFieldSecondRates(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                             const DesignedField& /*field*/, ElementOutput /*output*/,
                                             const NodalDisplacements& /*displacements*/,
                                             ElementResultBlock& /*rates*/) const
{
    return false;
}

} // namespace oos::elements
