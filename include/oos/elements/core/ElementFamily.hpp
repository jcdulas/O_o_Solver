#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/elements/core/DesignFieldUse.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"
#include "oos/elements/core/RecoverySlopes.hpp"
#include "oos/input/snapshot/SupportCatalog.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements
{

// The facets an element type may offer, each in its own source file of the element's folder.
enum class Facet : std::uint8_t
{
    Stiffness,
    Mass,
    Differential,
    Loads,
    Recovery,
    Derivatives
};

// A set of facets.
class FacetSet
{
public:
    constexpr FacetSet() = default;
    constexpr FacetSet(std::initializer_list<Facet> facets)
    {
        for (const Facet facet : facets)
            m_bits = static_cast<std::uint8_t>(m_bits | bitOf(facet));
    }
    [[nodiscard]] constexpr bool has(Facet facet) const noexcept { return (m_bits & bitOf(facet)) != 0; }

private:
    [[nodiscard]] static constexpr std::uint8_t bitOf(Facet facet) noexcept
    {
        return static_cast<std::uint8_t>(1U << static_cast<unsigned>(facet));
    }

    std::uint8_t m_bits = 0;
};

// The rows [first, last) of a batch that one facet call processes.
struct ElementRange
{
    std::int64_t first = 0;
    std::int64_t last = 0;
};

// The mass formulation of PARAM COUPMASS (COUPMASS > 0 asks for the coupled mass of the elements that have one;
// the default, -1, the lumped mass).
enum class MassFormulation : std::uint8_t
{
    Lumped,
    Coupled
};

// What the mass facet needs besides the batch. PARAM WTMASS is not applied by the elements: the element masses are in
// the model's mass units, and MassAssembler (assembly) multiplies the assembled mass matrix by WTMASS.
struct MassOptions
{
    MassFormulation formulation = MassFormulation::Lumped;
    // The non-structural mass that NSM, NonStructMass1, NonStructMassLumped, NonStructMassLumped1 cards add to each
    // element (nonStructMassTable of O_o_Reader, distributed to the elements by the assembler), indexed by batch row:
    // per unit length for a line element. Empty when none; the NSM field of the element's own property is read by the
    // element and is not included here.
    std::span<const double> addedNonStructuralMass;
};

// The output kinds a recovery facet gives, each as a result table of the O_o_Reader data contract (forces, stresses,
// strains). The strains of a shell come in two forms (Case Control STRAIN): at the fibres Z1 and Z2
// (FIBER, first column fiber_distance) or as the membrane strains and the curvatures (STRCUR, first column
// fiber_curvature); an element with one form gives it as Strains.
enum class ElementOutput : std::uint8_t
{
    Forces,
    Stresses,
    Strains,
    StrainCurvatures
};

// The output kinds of the layers of a laminated element (the composite tables, one row per ply): the ply stresses and
// strains in the ply frame, and the failure indices of the plies and of their bonds. PlyStates is no table: the
// stresses, mechanical strains and transverse shears of the plies whose failure indices are written, one row per such
// ply, every column affine in the displacements; the vectorised recovery makes the failure indices from them
// (ElementFamily::vectorisedLayerSource).
enum class LayerOutput : std::uint8_t
{
    Stresses,
    Strains,
    FailureIndices,
    PlyStates
};

// One output of a recovery: an element output (recover()) or a layer output (recoverLayers()).
struct RecoveryOutput
{
    bool isLayer = false;
    ElementOutput output = ElementOutput::Stresses;
    LayerOutput layerOutput = LayerOutput::Stresses;

    [[nodiscard]] static constexpr RecoveryOutput of(ElementOutput element) noexcept
    {
        return {false, element, LayerOutput::Stresses};
    }
    [[nodiscard]] static constexpr RecoveryOutput ofLayers(LayerOutput layer) noexcept
    {
        return {true, ElementOutput::Stresses, layer};
    }
    [[nodiscard]] constexpr bool operator==(const RecoveryOutput& other) const noexcept
    {
        return isLayer == other.isLayer && (isLayer ? layerOutput == other.layerOutput : output == other.output);
    }
};

// One distributed pressure of a load set on an element (pressureTable of O_o_Reader: ShellPressureLoad,
// FacePressureLoad per element), with the batch row of that element: the load vector builder finds the rows from the
// element IDs.
struct ElementPressure
{
    std::int64_t elementRow = 0;
    std::int32_t setId = 0;                  // the load set, for the diagnostics
    std::string_view card;                   // "FacePressureLoad"
    std::array<double, 4> cornerPressures{}; // P1 to P4, the LOAD factors applied (p1-p4 of pressureTable)
    bool hasDirection = false;               // CID or N1-N3 given: a traction along a vector, not a normal pressure
    std::int32_t directionFrameId = 0;       // CID of the direction (0 the basic system)
    std::array<double, 3> direction{};       // N1-N3 in the system CID (not normalised)
    // The corner grid point IDs of the loaded face of a solid (face_nodes of pressureTable: the corners O_o_Reader
    // finds from G1 and G3 or G4 of a FacePressureLoad), P1-P4 at these corners in this order; empty for a shell, whose
    // face is the element. Borrowed from the pressure table, which outlives the call.
    std::span<const std::int32_t> faceNodes{};
};

// One distributed or concentrated load of a load set on a line element (lineLoadsTable of O_o_Reader:
// BeamDistributedLoad on a Bar2 or Beam2), with the batch row of that element. The load and scale types are the
// texts of the card (BeamDistributedLoad kind: "ForceX" ... "MomentZAlongLength", scale: "ActualLength",
// "FractionalLength", "ProjectedLength", "FractionalProjectedLength"), which the element checks; X1, X2 are as on the
// card (X2 = X1 for a concentrated load) and P1, P2 have the LOAD factors applied (p1, p2 times scale of the table).
struct ElementLineLoad
{
    std::int64_t elementRow = 0;
    std::int32_t setId = 0;     // the load set, for the diagnostics
    std::string_view card;      // "BeamDistributedLoad"
    std::string_view loadType;  // "ForceX", "ForceYAlongLength", "MomentZ", ...
    std::string_view scaleType; // "ActualLength", "FractionalLength", "ProjectedLength", "FractionalProjectedLength"
    double x1 = 0.0;
    double p1 = 0.0;
    double x2 = 0.0;
    double p2 = 0.0;
};

// The displacements of the whole model a recovery facet reads from: six values per grid point row (translations then
// rotations, in the displacement system CD of the grid point), one per scalar point row; and the temperatures of the
// subcase when it has a thermal load (the recovery subtracts the thermal strains: σ = E (ε − α ΔT)).
//
// With `slots` (the streamed path: the displacements of some points only, a region's), `grid` holds six
// values per grid slot and `scalar` one per scalar slot, a point row's slot given by `slots` (−1: a point the caller
// did not give, which no element recovered reads).
struct DisplacementSlots
{
    std::span<const std::int32_t> gridSlots;   // per grid point row of the model
    std::span<const std::int32_t> scalarSlots; // per scalar point row of the model
};
struct NodalDisplacements
{
    std::span<const double> grid;                      // 6 · NodeTable::rowCount() values (6 per grid slot)
    std::span<const double> scalar;                    // one per scalar point row (per scalar slot)
    const ElementTemperatures* temperatures = nullptr; // null: no thermal load; not owned
    const DisplacementSlots* slots = nullptr;          // null: the rows of the model; not owned

    [[nodiscard]] double valueAt(const ElementDof& dof) const
    {
        if (dof.kind == input::PointKind::Scalar)
            return scalar[slotOf(dof)];
        return grid[slotOf(dof) * 6 + static_cast<std::size_t>(dof.component) - 1];
    }
    // The grid or scalar slot of a DOF's point (its row without slots).
    [[nodiscard]] std::size_t slotOf(const ElementDof& dof) const
    {
        if (slots == nullptr)
            return static_cast<std::size_t>(dof.row);
        const std::span<const std::int32_t> of =
            dof.kind == input::PointKind::Scalar ? slots->scalarSlots : slots->gridSlots;
        return static_cast<std::size_t>(of[static_cast<std::size_t>(dof.row)]);
    }
};

// The plug-in interface of an element type (the only virtual interface of the element library, called once per range
// of a batch, never per element). One family handles every element of one card type; its facets are hand-written in
// the files of the element's folder and loop over the rows of the range with the batch's columns read once, in place.
//
// Matrices are given in the displacement systems of the element's grid points. Masses are given without WTMASS
// (MassOptions). A facet the family does not offer throws std::invalid_argument.
//
// The facets assume a batch that check() accepted: check() reports every problem of every element as a Diagnostic
// (a missing grid point, property or material, a degenerate geometry, a field value not supported), and a facet that
// meets an element check() would refuse throws std::invalid_argument naming it (the caller skipped the check).
//
// Thread-safety: a family is immutable; its facets may run concurrently on distinct ranges and blocks.
class ElementFamily
{
public:
    ElementFamily() = default;
    ElementFamily(const ElementFamily&) = delete;
    ElementFamily& operator=(const ElementFamily&) = delete;
    ElementFamily(ElementFamily&&) = delete;
    ElementFamily& operator=(ElementFamily&&) = delete;
    virtual ~ElementFamily() = default;

    // The element card ("Rod2").
    [[nodiscard]] virtual std::string_view card() const noexcept = 0;
    // The name of the formulation this family computes the card with (upper case): a card may have several families,
    // one per formulation, and the model chooses one per card (ElementRegistry::forModel). Every family written so far
    // is the ClassicFormulation one; a new formulation of a card overrides this with its own name.
    [[nodiscard]] virtual std::string_view formulation() const noexcept { return CLASSIC_FORMULATION; }
    [[nodiscard]] virtual FacetSet facets() const noexcept = 0;
    // The property cards the element reads ("RodProp"), and the material types; for the support catalog.
    [[nodiscard]] virtual std::vector<std::string> propertyCards() const { return {}; }
    // The field values accepted only at their default, for the support catalog (SupportCatalog::check refuses the
    // others before any analysis).
    [[nodiscard]] virtual std::vector<input::FieldRule> fieldRules() const { return {}; }

    // Reports a diagnostic for every element of the batch that a facet could not process.
    virtual void check(const ElementBatchView& view, base::DiagnosticSink& sink) const = 0;
    // Whether the check is row by row (the diagnostics of a row read that row alone), so that checkRows() checks a
    // range of rows and the ranges of a batch may be checked on several threads (ElementSet::check).
    [[nodiscard]] virtual bool isCheckedByRow() const noexcept { return false; }
    // check() on the rows of `range` only, for a family that isCheckedByRow(). Throws std::logic_error otherwise.
    virtual void checkRows(const ElementBatchView& view, ElementRange range, base::DiagnosticSink& sink) const;

    // Appends the stiffness matrix of each element of the range to `block`.
    virtual void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const;
    // Appends the mass matrix of each element of the range to `block` (an element without mass appends nothing).
    virtual void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                             ElementMatrixBlock& block) const;

    // The columns of an output kind, in the order recover() fills them; empty for a kind the element does not give.
    [[nodiscard]] virtual std::vector<std::string> outputColumns(ElementOutput output) const;
    // Resets `block` to the columns of `output` and appends the rows of each element of the range. Throws
    // std::invalid_argument for a kind outputColumns() does not give.
    virtual void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                         const NodalDisplacements& displacements, ElementResultBlock& block) const;

    // Reports a diagnostic for every pressure the loads facet cannot process (by default every one: the element takes
    // no pressure). The pressures name rows of this batch.
    virtual void checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                    base::DiagnosticSink& sink) const;
    // Appends the equivalent nodal loads of each pressure to `block`, in the displacement systems of the grid points.
    // Precondition: checkPressureLoads() accepted the pressures (std::invalid_argument otherwise).
    virtual void computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                      ElementVectorBlock& block) const;

    // Reports a diagnostic for every line load the loads facet cannot process (by default every one: the element takes
    // no line load). The loads name rows of this batch.
    virtual void checkLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                base::DiagnosticSink& sink) const;
    // Appends the equivalent nodal loads of each line load to `block`, in the displacement systems of the grid points.
    // Precondition: checkLineLoads() accepted the loads (std::invalid_argument otherwise).
    virtual void computeLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                  ElementVectorBlock& block) const;

    // Reports a diagnostic for every element of the batch whose thermal load the loads facet cannot compute: a missing
    // temperature, a thermal option not supported. By default, an element with stiffness and thermal expansion is
    // refused (its family does not compute thermal loads yet), and the elements without thermal expansion (springs,
    // bushes, gaps, fasteners, masses, shear panels: isThermallyInert) take none.
    virtual void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                   base::DiagnosticSink& sink) const;
    // Appends the equivalent nodal loads of the temperatures of each element of the range to `block`, in the
    // displacement systems of the grid points (an element without thermal load appends nothing). Precondition:
    // checkThermalLoads() accepted the batch (std::invalid_argument otherwise).
    virtual void computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                     const ElementTemperatures& temperatures, ElementVectorBlock& block) const;
    // Whether the element type takes no thermal load in a linear analysis (no thermal expansion in its formulation).
    [[nodiscard]] virtual bool isThermallyInert() const noexcept;

    // Reports a diagnostic for every element of the batch whose differential stiffness (SOL 105) the facet cannot
    // compute: an option its formulation does not cover. By default an element type with stiffness whose family does
    // not offer Facet::Differential is refused (one diagnostic for the batch), and the types without differential
    // stiffness in a linear buckling analysis (isDifferentiallyInert) take none.
    virtual void checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const;
    // Appends the differential stiffness K_d of each element of the range to `block`, in the displacement systems of
    // its grid points, from the element's internal forces under `preload`: the displacements (and the temperatures,
    // when the preload has a thermal load) of the static preload subcase (SOL 105: (K + λ K_d) φ = 0, K_d of the
    // preload at scale 1, λ the load factor). The element recovers its own internal forces from the displacements, so
    // that the SOL 105 driver calls this once per batch after the static solve. An element without differential
    // stiffness appends nothing. Precondition: checkDifferentialStiffness() accepted the batch (std::invalid_argument
    // otherwise).
    virtual void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                              const NodalDisplacements& preload, ElementMatrixBlock& block) const;
    // Whether the element type has no differential stiffness in a linear buckling analysis: no stiffness (masses), or
    // a scalar or local spring whose force does not turn with the structure (CELASi, Bushing, Bushing1D, Gap2, and
    // the fasteners Fastener, Weld, springs between the elements they pierce).
    [[nodiscard]] virtual bool isDifferentiallyInert() const noexcept;

    // The columns of an output kind for the elements recovered under line loads (recoverUnderLineLoads), which may be
    // written in another layout (a Bar2 under BeamDistributedLoad: the layout of its stations); by default those
    // of outputColumns().
    [[nodiscard]] virtual std::vector<std::string> outputColumnsUnderLineLoads(ElementOutput output) const
    {
        return outputColumns(output);
    }
    // Resets `block` to outputColumnsUnderLineLoads(output) and appends the rows of each element that `loads` names,
    // once per element, its output including the effect of its loads (the loads between the element's ends and its
    // stations). Precondition: the loads of one element are contiguous in `loads`, and checkLineLoads() accepted them.
    // The elements of the batch without line loads are recovered by recover(). By default the facet is refused.
    virtual void recoverUnderLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                       ElementOutput output, const NodalDisplacements& displacements,
                                       ElementResultBlock& block) const;

    // The element type of the result tables of the data contract that hold the rows of recover() (underLineLoads
    // false) or of recoverUnderLineLoads() (true), as O_o_Reader's MemoryResults names its containers ("ShellQuad4",
    // "Bar2LoadStations"): by default the exact card type for both.
    [[nodiscard]] virtual std::string resultElementType(bool underLineLoads) const;
    // The columns of a layer output kind, in the order recoverLayers() fills them; empty (the default) for an element
    // without layers.
    [[nodiscard]] virtual std::vector<std::string> layerOutputColumns(LayerOutput output) const;
    // Resets `block` to layerOutputColumns(output) and appends the rows of the layered elements of the range (an
    // element without layers appends nothing). Throws std::invalid_argument for a kind the element does not give.
    virtual void recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                               const NodalDisplacements& displacements, ElementResultBlock& block) const;
    // The element type of the tables of recoverLayers() ("ShellQuad4Composite"); by default the exact card type
    // followed by "_composite".
    [[nodiscard]] virtual std::string layerResultElementType() const;
    // The rows of several outputs of the elements of `range` under `displacements`: blocks[k] reset and filled as
    // recover() (recoverLayers() for a layer output) fills it for outputs[k], with the same bits, each element resolved
    // and formulated once for all of them (as for the rows at rest of a block of cases). By
    // default recover() or recoverLayers() for each output. Throws std::invalid_argument for an output the element
    // does not give, as recover() does.
    virtual void recoverOutputs(const ElementBatchView& view, ElementRange range,
                                std::span<const RecoveryOutput> outputs, const NodalDisplacements& displacements,
                                std::span<ElementResultBlock> blocks) const;
    // The rows recover() gives the element of batch row `row` for `output` at a unit displacement of each DOF of its
    // stiffness, without temperatures (its recovery matrix column by column, as output::SectionLoads makes its rows):
    // `dofs` gets the DOFs computeStiffness gives the element, in their order, and blocks[d] (one block per DOF, each
    // reset) the rows of recover() under displacements all zero but that of dofs[d], one, with the same bits; the
    // element resolved and formulated once for all of them. False when the family does not give them (the caller then
    // recovers at each unit displacement); by default false.
    virtual bool recoverAtUnitDisplacements(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                            std::vector<ElementDof>& dofs,
                                            std::vector<ElementResultBlock>& blocks) const;

    // The Derivatives facet (semi-analytic sensitivities): what a designed real field of the element's cards changes in
    // its computations (stiffness, mass, loads, recovery), so that design/sensitivity/ElementDerivatives, which
    // differences the facets above on the snapshots with the field moved by ±h, computes only what can change. The
    // derivatives themselves need nothing more from a family: its facets read the overlaid columns as any other. A
    // family without Facet::Derivatives is refused by the sensitivity analysis (naming the card) when a designed field
    // reaches one of its elements; by default every computation changes (FieldUse::all()).
    [[nodiscard]] virtual FieldUse designFieldUse(const DesignedField& field) const;

    // The analytic derivatives: what a family gives in closed form, so that the sensitivities difference only the rest.
    // Each is optional; the defaults give nothing.
    //
    // The form of each column of outputColumns(output) (layerOutputColumnForms: of layerOutputColumns(output)) in the
    // displacements (ColumnForm); empty when the family declares none, and the derivatives of its items with respect
    // to the displacements are then differenced.
    [[nodiscard]] virtual std::vector<ColumnForm> outputColumnForms(ElementOutput output) const;
    [[nodiscard]] virtual std::vector<ColumnForm> layerOutputColumnForms(LayerOutput output) const;
    // The slopes of the rows recover() (layerRecoverySlopes: recoverLayers()) gives the element of batch row `row` for
    // `output` (RecoverySlopes, its DOFs those of computeStiffness), its formulation built once; false when the family
    // does not give them, and the caller takes them from recover() at unit displacements (exact as well for the
    // Linear columns, at the cost of a recovery per DOF).
    virtual bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                RecoverySlopes& slopes) const;
    virtual bool layerRecoverySlopes(const ElementBatchView& view, std::int64_t row, LayerOutput output,
                                     RecoverySlopes& slopes) const;
    // The vectorised recovery: the rows of `output` of a block of cases made from the recovery slopes of each element
    // (output::ElementResultBuilder::buildCases: the Linear columns as rest + S_e·u_e, the rows at rest those of
    // recover() at zero displacements) and completed by the family.
    //
    // The fewest cases of a block from which the family's rows of `output` pay for its slopes (about their cost in
    // recoveries, measured); 0 (the default) when the family does not complete its rows of `output`, which are then
    // recovered case by case. vectorisedLayerRecoveryMinCases: the same for the layer tables.
    [[nodiscard]] virtual std::size_t vectorisedRecoveryMinCases(ElementOutput output) const;
    [[nodiscard]] virtual std::size_t vectorisedLayerRecoveryMinCases(LayerOutput output) const;
    // The slopes of several outputs of the element of batch row `row` (slopes[k] those recoverySlopes, or
    // layerRecoverySlopes for a layer output, gives outputs[k]), its formulation built and evaluated once for all of
    // them; false when one is not given. By default recoverySlopes or layerRecoverySlopes for each output.
    virtual bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                         std::span<const RecoveryOutput> outputs,
                                         std::span<RecoverySlopes> slopes) const;
    // Sets the columns of `output` that are not Linear (outputColumnForms) in the rows of each block from their Linear
    // columns, with the functions recover() computes them with, so that a row whose Linear columns have the bits of
    // recover()'s gets its bits. The blocks hold the same rows (the same elements and places, in the order of
    // recover()) for different cases; the family reads what it needs of an element once for all of them. Throws
    // std::invalid_argument when vectorisedRecoveryMinCases(output) is 0.
    virtual void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                                       std::span<ElementResultBlock> blocks) const;
    // The element output whose rows the vectorised recovery makes from slopes for the table of `output`: `output`
    // itself (the default; its rows completed in place by completeRecoveredRows), or another whose columns are all
    // Linear and whose rows make those of `output` one for one (the Weld's stresses from its forces).
    [[nodiscard]] virtual ElementOutput vectorisedSource(ElementOutput output) const;
    // The rows of `output` of each block from those of its source (vectorisedSource(output) another output, as
    // completeRecoveredLayerRows with sources): blocks[c] (reset to outputColumns(output)) gets one row per row of
    // sources[c] (the rows of the source for the same case), at its place, with the functions recover() computes them
    // with. Throws std::invalid_argument when the family makes no table of `output` from a source.
    virtual void completeRecoveredRowsFromSource(const ElementBatchView& view, ElementOutput output,
                                                 std::span<const ElementResultBlock> sources,
                                                 std::span<ElementResultBlock> blocks) const;
    // The layer output whose rows the vectorised recovery makes from slopes for the layer table `output`: `output`
    // itself (the default; its rows completed in place by completeRecoveredLayerRows), or another whose columns are all
    // Linear and whose rows make those of `output` one for one (the failure indices from the ply states).
    [[nodiscard]] virtual LayerOutput vectorisedLayerSource(LayerOutput output) const;
    // The layer rows of `output` of each block, as completeRecoveredRows for the element rows: when
    // vectorisedLayerSource(output) is `output`, `sources` is empty and the columns that are not Linear are set in
    // place; otherwise blocks[c] (reset to layerOutputColumns(output)) gets one row per row of sources[c] (the rows of
    // the source for the same case), at its place, with the functions recoverLayers() computes them with. Throws
    // std::invalid_argument when vectorisedLayerRecoveryMinCases(output) is 0.
    virtual void completeRecoveredLayerRows(const ElementBatchView& view, LayerOutput output,
                                            std::span<const ElementResultBlock> sources,
                                            std::span<ElementResultBlock> blocks) const;
    // dK/dv and dM/dv of the element of batch row `row` with respect to the designed field `field` of its cards at its
    // value in `view` (the design point), appended to `block` as computeStiffness and computeMass append them; false
    // when the family has no closed form for that field and that element, and the caller differences the facet
    // (design/sensitivity/ElementDerivatives). `field` names a cell the element reads (the field of its own card, of
    // its property or of a material of its property).
    virtual bool stiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementMatrixBlock& block) const;
    // stiffnessFieldDerivative of the element of batch row `row` for several fields at once: blocks[k] gets what it
    // appends for fields[k], isClosed[k] its result (1 or 0). A family overrides it to resolve and formulate the
    // element once for all the fields (the 19 fields of a lamination zone reach the same elements), with
    // the bits of the calls one by one; by default stiffnessFieldDerivative for each.
    virtual void stiffnessFieldDerivatives(const ElementBatchView& view, std::int64_t row,
                                           std::span<const DesignedField> fields, std::span<ElementMatrixBlock> blocks,
                                           std::span<std::uint8_t> isClosed) const;
    virtual bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                     const MassOptions& options, ElementMatrixBlock& block) const;
    // dK_d/dv of the element of batch row `row` at fixed preload displacements and temperatures (`preload`, as
    // computeDifferentialStiffness takes it), appended as computeDifferentialStiffness appends it; false when the
    // family has no closed form for that field, element and preload (the buckling load factors' explicit
    // parts).
    virtual bool differentialStiffnessFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                      const DesignedField& field, const NodalDisplacements& preload,
                                                      ElementMatrixBlock& block) const;
    // Mixed field curvature of K_d at a fixed mechanical preload. Its displacement dependence remains linear;
    // the buckling Hessian adds both preload tangents separately. False means the formula is unavailable.
    virtual bool differentialStiffnessFieldMixedDerivative(
        const ElementBatchView& view, std::int64_t row, const DesignedField& first, const DesignedField& second,
        const NodalDisplacements& preload, ElementMatrixBlock& block) const;
    // The derivatives of the rows recover() gives the element of batch row `row` under `displacements` with respect to
    // the designed field `field` at fixed displacements (the explicit parts of its items), in closed form: `rates` in
    // the layout of recover()'s block (the same rows, places and columns), each column's ∂value/∂v, NaN where the
    // family gives none (a principal value, an angle); false when it has no closed form for that field and element,
    // and the caller differences the recovery on the perturbed snapshots.
    virtual bool recoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                    ElementOutput output, const NodalDisplacements& displacements,
                                    ElementResultBlock& rates) const;
    // recoveryFieldRates of the element of batch row `row` for several fields at once: rates[k] for fields[k],
    // isClosed[k] its result (1 or 0); a family overrides it to resolve and formulate the element and recover its
    // state once for all the fields, with the bits of the calls one by one; by default
    // recoveryFieldRates for each.
    virtual void recoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                            std::span<const DesignedField> fields, ElementOutput output,
                                            const NodalDisplacements& displacements,
                                            std::span<ElementResultBlock> rates,
                                            std::span<std::uint8_t> isClosed) const;
    // recoveryFieldRates for the layer rows of recoverLayers (`rates` in their layout, NaN where the family gives none:
    // an angle, a principal value, the element's largest failure index): the plies of a laminated shell along a ply
    // thickness; false without a closed form, and the caller differences the layer recovery.
    virtual bool layerRecoveryFieldRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                         LayerOutput output, const NodalDisplacements& displacements,
                                         ElementResultBlock& rates) const;
    // layerRecoveryFieldRates for several fields at once (rates[k], isClosed[k] for fields[k]), as
    // recoveryFieldRatesOfFields: a family resolves, formulates and recovers the element once for all of them, with the
    // bits of the calls one by one; by default layerRecoveryFieldRates for each.
    virtual void layerRecoveryFieldRatesOfFields(const ElementBatchView& view, std::int64_t row,
                                                 std::span<const DesignedField> fields, LayerOutput output,
                                                 const NodalDisplacements& displacements,
                                                 std::span<ElementResultBlock> rates,
                                                 std::span<std::uint8_t> isClosed) const;

    // The second derivatives with respect to one designed field: what a family gives in closed form for the element
    // Hessian blocks of the Hessian-vector products of a weighted sum of responses; each optional, the defaults give
    // nothing (the products then difference their linearisation).
    //
    // d²K/dv² and d²M/dv² of the element of batch row `row` at the field's value in `view`, appended as
    // computeStiffness and computeMass append them; true with nothing appended for a zero derivative; false when the
    // family has no closed form for that field and element.
    virtual bool stiffnessFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                const DesignedField& field, ElementMatrixBlock& block) const;
    virtual bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                           const MassOptions& options, ElementMatrixBlock& block) const;
    // Mixed derivatives in two fields at fixed geometry. The defaults use the same-field methods when the
    // fields match, or return a known zero when FieldUse says either field cannot change the matrix.
    virtual bool stiffnessFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                               const DesignedField& first, const DesignedField& second,
                                               ElementMatrixBlock& block) const;
    virtual bool massFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                          const DesignedField& first, const DesignedField& second,
                                          const MassOptions& options, ElementMatrixBlock& block) const;
    // The slopes of the rows of recoveryFieldRates (their derivatives at fixed displacements) with respect to the
    // element's DOFs, ∂S/∂v: the mixed second derivatives of its Linear columns, in the layout of recoverySlopes (zero
    // in the other columns); false when the family gives no closed form (the caller then takes them from
    // recoveryFieldRates at unit displacements, exact as well: the rates of a Linear column are linear in u).
    virtual bool recoverySlopeRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                    ElementOutput output, RecoverySlopes& rates) const;
    // ∂²value/∂v² of the rows recover() gives under `displacements` at fixed displacements, in the layout of
    // recoveryFieldRates, NaN where the family gives none (a column that is not Linear); false without a closed form.
    virtual bool recoveryFieldSecondRates(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                          ElementOutput output, const NodalDisplacements& displacements,
                                          ElementResultBlock& rates) const;
    // Mixed recovered rates at fixed displacement, in recover()'s row layout. False when no closed form is given.
    virtual bool recoveryFieldMixedRates(const ElementBatchView& view, std::int64_t row,
                                         const DesignedField& first, const DesignedField& second,
                                         ElementOutput output, const NodalDisplacements& displacements,
                                         ElementResultBlock& rates) const;

protected:
    // Throws std::invalid_argument: the facet is not offered by this element type.
    [[noreturn]] void refuseFacet(std::string_view facet) const;
    // Throws std::invalid_argument when outputColumns(output) is empty: the element does not give that output.
    void requireOutput(ElementOutput output) const;
};

// The whole batch as one range.
[[nodiscard]] inline ElementRange wholeBatch(const ElementBatchView& view) noexcept
{
    return {0, view.rowCount()};
}

} // namespace oos::elements
