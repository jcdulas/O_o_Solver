#pragma once

// The internal computation of the static sensitivities (StaticSensitivity), shared with the optimisation aids that
// extend it (design/aids: SensitivityMap, SecondDerivatives, DisplacementDerivatives): the linearised items, the
// subcases with their displacements and factorised systems, the pseudo-loads. Not part of the public headers: it is
// included by the sources of the design module only.

#include "EigenItems.hpp"
#include "ModalItems.hpp"
#include "TrimItems.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/design/aids/LinearityIndicator.hpp"
#include "oos/design/aids/SelfAdjointResponses.hpp"
#include "oos/design/external/ExternalResponse.hpp"
#include "oos/design/responses/ResultLookup.hpp"
#include "oos/design/responses/weight_volume/MassProperties.hpp"
#include "oos/design/sensitivity/AdjointSensitivity.hpp"
#include "oos/design/sensitivity/AffectedElements.hpp"
#include "oos/design/sensitivity/ChainedAdjoint.hpp"
#include "oos/design/sensitivity/DirectSensitivity.hpp"
#include "oos/design/sensitivity/ElementDerivatives.hpp"
#include "oos/design/sensitivity/ElementItemFunction.hpp"
#include "oos/design/sensitivity/FieldPerturbation.hpp"
#include "oos/design/sensitivity/LazyFieldPerturbation.hpp"
#include "oos/design/sensitivity/MethodChoice.hpp"
#include "oos/design/sensitivity/PseudoLoads.hpp"
#include "oos/design/sensitivity/SensitivitySystem.hpp"
#include "oos/design/sensitivity/ShapeElements.hpp"
#include "oos/design/sensitivity/StaticSensitivity.hpp"
#include "oos/design/sensitivity/ValueCheck.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"
#include "oos/output/sections/CutLoads.hpp"
#include "oos/output/sections/SectionLoads.hpp"
#include "oos/output/sections/ZoneContour.hpp"
#include "oos/reduction/external/ExternalLoadSource.hpp"
#include "oos/reduction/external/ExternalUnitAssembly.hpp"
#include "oos/reduction/external/ReplacedElements.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace oos::design::detail
{

// The derivatives of an element item with respect to the displacements of its element, DOF by DOF, by central
// differences extrapolated (Richardson: (4·D(δ/2) − D(δ))/3), δ relative to the largest |u| of the element's DOFs of
// the same kind (translations, rotations, scalar points). Most items are linear in u (affine under a thermal strain),
// for which any step is exact: a large step, LINEAR_STEP_RELATIVE, keeps their rounding ε·|r|/δ small (2e-13 relative),
// so that the derivative of an item whose DOFs nearly cancel (a moment of a higher-order shell) keeps its digits. The
// non-linear items (von Mises, principal stresses, margins, strength ratios) show it by D(δ) ≠ D(δ/2) beyond
// LINEARITY_TOLERANCE; for them the differences are taken again at NONLINEAR_STEP_RELATIVE, small enough that a stress
// state near zero at a point of the element stays in the smooth region of the function (truncation (δ/|u|)⁴ times its
// curvature) and large enough that the rounding ε/δ stays at 2e-11.
inline constexpr double LINEAR_STEP_RELATIVE = 1.0e-3;
inline constexpr double NONLINEAR_STEP_RELATIVE = 1.0e-5;
// The relative difference of D(δ) and D(δ/2) above which an item is taken as non-linear in a DOF: far above the
// rounding of a linear item at the large step, far below the curvature of a von Mises stress at it.
inline constexpr double LINEARITY_TOLERANCE = 1.0e-8;
// The step when the element does not move at all (no load reaches it): any step is exact for the affine items, and the
// non-linear ones have no derivative at a zero state.
inline constexpr double DISPLACEMENT_STEP_AT_REST = 1.0;
// The element jobs of one piece of the parallel linearisation (some 2 ms each by differences on the reference PC, 20 to
// 100 us from the slopes of a family: a piece of 32 stays under 0.1 s, and copies the displacements of its subcase once
// when a job moves them).
inline constexpr std::size_t ELEMENT_JOBS_PER_PIECE = 32;
// The field loop on the pool: a wave holds consecutive fields up to these many elements or fields
// (their perturbations alive together: each costs the copy of its relation's cells), its (field, element) tasks
// computed a chunk at a time (each task's terms held until added: a few hundred bytes per element and subcase), the
// tasks of a chunk in pieces of FIELD_TASKS_PER_PIECE (some 15 us each in closed form, 60 us by differences).
inline constexpr std::size_t FIELD_WAVE_ELEMENTS = 8192;
inline constexpr std::size_t FIELD_WAVE_FIELDS = 256;
inline constexpr std::size_t FIELD_WAVE_CHUNK_TASKS = 16384;
inline constexpr std::size_t FIELD_TASKS_PER_PIECE = 16;
// How far a recomputed item may differ from the analysis' value: ValueCheck.
// The same check for a contour force, against the largest value of the response: a contour force is a sum
// K_e·u_e whose terms cancel at the contour (a drilling moment, a free edge), so its rounding is that of the element
// forces, not of its own value; 1e-9 of the largest force still tells another function (a wrong row) from it.
inline constexpr double CONTOUR_VALUE_CHECK_TOLERANCE = 1.0e-9;
// Item 3 of an ESE output is the percentage of the total strain energy.
inline constexpr std::int32_t STRAIN_ENERGY_PERCENT_ITEM = 3;
inline constexpr double PERCENT = 100.0;

[[nodiscard]] inline double largestMagnitude(const std::vector<double>& values)
{
    double largest = 0.0;
    for (const double value : values)
        largest = std::max(largest, std::abs(value));
    return largest;
}

// `terms`: Σ_d |∂r/∂u_d·u_d| when the weights are known (ValueCheck: the rounding of the sum), 0 otherwise.
inline void checkValue(const ResponseValues& values, std::size_t item, double recomputed, double terms = 0.0)
{
    ValueCheck::require(values, item, recomputed, terms);
}

// Σ_d |w_d·u_d|: the magnitude of the terms of an affine item with weights w at the displacements u.
[[nodiscard]] inline double termMagnitude(const SparseVector& weights, std::span<const double> displacements,
                                          double scale = 1.0)
{
    double sum = 0.0;
    for (const auto& [index, weight] : weights)
        sum += std::abs(weight * displacements[static_cast<std::size_t>(index)]);
    return std::abs(scale) * sum;
}

// The rate of a plane von Mises item (ItemForm) from the rates of its components: with
// Q = a² + b² − a·b + 3·(k·c)² and vm = s·√Q, dvm = s·dQ / (2·√Q), dQ = (2a − b)·da + (2b − a)·db + 6·k²·c·dc. At
// vm = 0 the item has no derivative (a cone): 0, as the central differences of the item give it (vm(u ± h) equal).
[[nodiscard]] inline double planeVonMisesRate(const ItemForm& terms, const std::array<double, 3>& rates)
{
    const auto [a, b, c] = terms.components;
    const double k = terms.shearFactor;
    const double q = a * a + b * b - a * b + 3.0 * k * k * c * c;
    if (!(q > 0.0))
        return 0.0;
    const double dq = (2.0 * a - b) * rates[0] + (2.0 * b - a) * rates[1] + 6.0 * k * k * c * rates[2];
    return terms.scale * dq / (2.0 * std::sqrt(q));
}

// The state of one subcase that the sensitivities read.
struct SubcaseState
{
    solutions::StaticSubcase subcase;
    std::size_t group = 0;
    base::BudgetedBuffer<double> displacements;
    // Inertia relief (PARAM INREL = −1): the relief of its constraint set (null: none) and its relief
    // acceleration field D·ü_r (one value per g-set DOF).
    const solutions::KeptRelief* relief = nullptr;
    base::BudgetedBuffer<double> reliefAccelerations;
    std::unique_ptr<assembly::TemperatureField> temperatures;
    std::set<std::int32_t> distributedElements; // under a pressure or a line load of the subcase
    // The pressures and line loads of the subcase on each element, as the analysis applied them (a field
    // differences the loads of the elements it reaches, not the whole load vector).
    std::map<ElementRef, std::pair<std::vector<elements::ElementPressure>, std::vector<elements::ElementLineLoad>>>
        elementLoads;
    bool hasInertialLoads = false;
    bool hasDistributedLoads =
        false; // a PointsPressureLoad, BeamDistributedLoad, ShellPressureLoad or FacePressureLoad in its load set
    bool hasConcentratedLoads = false;                         // a FORCE, MOMENT or their variants, an ScalarPointLoad
    bool hasEnforcedDisplacements = false;                     // SPC D or EnforcedMotionConstr not zero
    std::optional<base::BudgetedBuffer<double>> elementForces; // Kgg·u, for ESE item 3 and the self-adjoint items
    double totalEnergy = 0.0;
    // A SAERO subcase: its displacements are the trimmed state, its items are differentiated on the trim's
    // linearisation (TrimItems), and it has no constraint set of its own.
    bool isAeroelastic = false;
    // A SUBCOM: its terms (their states, in its order, and coefficients a_k); its
    // displacements u_c = Σ a_k·u_k, its own temperatures; no load of its own: its pseudo-loads are its stiffness term
    // −∂K/∂v·u_c and Σ a_k times the load parts of its terms (PseudoLoads::addCombination), solved on its terms'
    // constraint set, so that du_c/dx = Σ a_k·du_k/dx.
    std::vector<std::pair<std::size_t, double>> terms;
    // A term of a SUBCOM that no item reads: linearised for its load parts, never solved.
    bool isTermOnly = false;

    [[nodiscard]] bool isCombination() const noexcept { return subcase.isCombination; }

    [[nodiscard]] elements::NodalDisplacements nodal(const assembly::DofMap& dofMap) const
    {
        const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(dofMap.gridCount());
        const std::span<const double> all = displacements.span();
        return {all.first(gridValues), all.subspan(gridValues), temperatures ? &temperatures->temperatures() : nullptr};
    }
};

// The element items of one element in one subcase.
struct ElementJob
{
    ElementRef element;
    std::size_t subcase = 0;
    std::unique_ptr<ElementItemFunction> function;
    std::vector<ItemKey> keys;
    std::vector<double> baseValues;
};

struct SpcItem
{
    ItemKey key;
    std::size_t subcase = 0;
    SparseVector w;
};

struct MassItem
{
    ItemKey key;
    const DesignResp1Response* response = nullptr;
    double currentMass = 0.0; // FRMASS: the mass of the selection at the design point
};

// A compliance or mean displacement item (design/aids/SelfAdjointResponses): r = scale·Pᵀu, its weights scale·K·u
// (= scale·P on the l set), its adjoint vector scale·u. For a mean displacement scale = 1/W, and `forceWeights` holds
// ∂W/∂P_g = F_g/|F_g| on the translations of the loaded grid points (the quotient rule of its explicit part).
struct SelfAdjointItem
{
    ItemKey key;
    std::size_t subcase = 0;
    double scale = 1.0;
    double value = 0.0;
    SparseVector forceWeights;
};

// A deflection over thickness item (design/aids/LinearityIndicator): its grid point and the translations it was
// linearised at, for the explicit part (the normal and the thickness move with the designed quantities).
struct PointItem
{
    ItemKey key;
    std::int32_t grid = 0;
    std::array<double, 3> translations{};
};

// The sections or cuts of a SECTLOAD or CUTLOAD response on the element set of the design point: their
// displacement rows are the weights ∂r/∂u of their items (exact: the loads are affine in the displacements), and
// they are evaluated again on the perturbed element sets for the explicit parts.
struct SectionRows
{
    std::unique_ptr<output::SectionLoads> sections; // SECTLOAD
    std::unique_ptr<output::CutLoads> cuts;         // CUTLOAD
    std::map<std::int32_t, std::size_t> indexOf;    // section or cut ID → its index in sections() or cuts()
};

// A SECTLOAD or CUTLOAD item: its response, section or cut, component (from 0) and subcase (an index into the
// subcases of the computation).
struct SectionItem
{
    ItemKey key;
    std::int32_t response = 0;
    std::size_t index = 0;
    std::size_t component = 0;
    std::size_t subcase = 0;
};

// A term of a zone item: weights over the rows of one quantity of one zone (contour displacements, contour
// forces, interior loads), the rows of output/sections/ZoneContour.
struct ZoneTerm
{
    std::size_t zone = 0;
    ExternalArgumentKind quantity = ExternalArgumentKind::ContourDisplacement;
    std::vector<std::pair<std::size_t, double>> weights;
};

// An item read from the zone data of one subcase: a CONTDISP or CONTFORC item (one term of one weight), or an external
// item (the partials of the plugin with respect to the zone data of its case). `rigid`: w of its MPC force part,
// r = wᵀ·(Kgg·u − P) (its weights Kgg·w, its explicit part −wᵀ·q_pseudo, as an SPC force); `loadWeights`: its
// weights over the applied loads (explicit part ∂P/∂v at fixed u).
struct ZoneItem
{
    ItemKey key;
    std::size_t subcase = 0;
    std::vector<ZoneTerm> terms;
    SparseVector rigid;
    SparseVector loadWeights;
};

// The contour forces of a zone that is a condensed unit the analysis did not recover: they are the unit's
// boundary forces K*·u_a − f*·α, so their weights over u are the rows of K* on the unit's boundary DOFs turned to the
// contour DOFs (the rows of the zone's members would read the unit's interior, which the analysis did not compute);
// the unit is driven by no variable (a driven unit is recovered), so f*·α has no derivative.
struct UnitZoneForceRows
{
    std::vector<base::DofIndex> dofs; // the unit's boundary DOFs (g set): the columns; bounded by its boundary
    std::vector<double> rows;         // contour DOFs × dofs, row-major
};

// x·y of two compressed sparse vectors.
[[nodiscard]] inline double dotSparseVectors(const SparseVector& x, const SparseVector& y) noexcept
{
    double sum = 0.0;
    std::size_t b = 0;
    for (const auto& [dof, value] : x)
    {
        while (b < y.size() && y[b].first < dof)
            ++b;
        if (b < y.size() && y[b].first == dof)
            sum += value * y[b].second;
    }
    return sum;
}

// Reads the g-set displacements of a subcase from the analysis' displacement table.
inline void readDisplacements(const solutions::StaticResults& results, const assembly::DofMap& dofMap,
                              std::int32_t subcase, std::span<double> displacements)
{
    // The g-set vectors the analysis kept (StaticOptions::keptDisplacements: the condensed path, where the
    // interior points of the units it did not recover are not computed, zero here, never read: the units a
    // sensitivity reads are recovered).
    for (const solutions::KeptDisplacements& kept : results.keptDisplacements)
        if (kept.subcase == subcase && kept.values.size() == displacements.size())
        {
            std::copy(kept.values.data(), kept.values.data() + kept.values.size(), displacements.begin());
            return;
        }
    for (const output::NodalResultTable& table : results.arrays.nodalTables)
    {
        if (table.kind != "displacements" || table.subcase != subcase)
            continue;
        std::size_t found = 0;
        for (std::size_t row = 0; row < table.rowCount(); ++row)
        {
            const bool isScalar = table.pointTypes[row] == static_cast<std::int8_t>(output::PointType::Scalar);
            for (int c = 0; c < (isScalar ? 1 : output::NodalResultTable::COMPONENT_COUNT); ++c)
                if (const std::optional<base::DofIndex> dof = dofMap.findDof(table.ids[row], isScalar ? 0 : c + 1))
                {
                    displacements[static_cast<std::size_t>(*dof)] =
                        table.values[row * output::NodalResultTable::COMPONENT_COUNT + static_cast<std::size_t>(c)];
                    ++found;
                }
        }
        // The interior points of the external units are left out of every output: their DOFs are held (zero).
        std::size_t interior = 0;
        for (const solutions::ExternalUnitRecord& record : results.externalUnits)
            interior += record.placement->interiorSize();
        if (found + interior == displacements.size())
            return;
    }
    throw base::OptimisationError(base::concatenate(
        {"the sensitivities of subcase ", std::to_string(subcase),
         " need the displacements of every point, which the analysis did not give; evaluate the design point with "
         "the displacements requested (DesignProblem::evaluateStatics with needsDisplacements)"}));
}

// The elements the external units of an analysis replace, on the families of `registry` (null without one): the
// sensitivities read the model as the analysis did.
inline std::unique_ptr<reduction::ReplacedElements> replacedElementsOf(const solutions::StaticResults* results,
                                                                       const elements::ElementRegistry& registry)
{
    if (results == nullptr || results->externalUnits.empty())
        return nullptr;
    std::vector<base::EntityId> replaced; // bounded by the zones
    for (const solutions::ExternalUnitRecord& record : results->externalUnits)
        replaced.insert(replaced.end(), record.placement->zone().replacedElements.begin(),
                        record.placement->zone().replacedElements.end());
    return std::make_unique<reduction::ReplacedElements>(registry, std::move(replaced));
}

// The designed quantities of a design point, as PseudoLoads and FieldPerturbation index them: the fields, then the
// shapes.
[[nodiscard]] inline std::vector<FieldValue> quantitiesOf(const DesignValues& values)
{
    std::vector<FieldValue> quantities = values.fields;
    quantities.insert(quantities.end(), values.shapes.begin(), values.shapes.end());
    return quantities;
}

// What a linearisation at a moved state under inertia relief reads of its design point:
// the point's element set, the elements the direction moves and the point's relief of each constraint set (by group,
// null for a set without relief). Borrowed: the point's computation outlives the moved one.
struct ReliefPoint
{
    const assembly::ElementSet* set = nullptr;
    std::vector<ElementRef> moved;
    std::vector<const solutions::KeptRelief*> reliefs;
};

// The rest of the sensitivity computation, with its state.
class SensitivityComputation
{
public:
    SensitivityComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                           base::MemoryBudget& budget)
        : m_in(inputs),
          m_options(options),
          m_budget(budget),
          m_registry(elements::ElementRegistry::forModel(*inputs.overlaid)),
          m_replaced(replacedElementsOf(inputs.results, m_registry)),
          m_index(*inputs.overlaid, budget),
          m_set(m_index, m_replaced ? m_replaced->registry() : m_registry),
          m_dofMap(inputs.overlaid->nodes(), m_index.points()),
          m_variableCount(inputs.model->independentCount()),
          m_quantities(quantitiesOf(*inputs.design))
    {
        // The eigenvalue items of the MODES and BUCK subcases (EigenItems).
        if (inputs.modes != nullptr || inputs.buckling != nullptr)
            m_eigen = std::make_unique<EigenItems>(EigenItems::Inputs{
                inputs.modes, inputs.buckling, &m_dofMap, &m_set,
                inputs.overlaid->hasTable(input::TableRole::Params) ? &inputs.overlaid->params() : nullptr,
                m_variableCount, &budget});
        if (inputs.modes != nullptr)
            m_modal = std::make_unique<ModalItems>(ModalItems::Inputs{
                inputs.overlaid, inputs.modes, &m_set, &m_dofMap, inputs.kernel, inputs.pool, &budget});
        // The items of the SAERO subcases (TrimItems), on the trim's linearisation.
        if (inputs.trim != nullptr)
            m_trim = std::make_unique<TrimItems>(TrimItems::Inputs{
                inputs.trim, &m_dofMap,
                inputs.overlaid->hasTable(input::TableRole::Params) ? &inputs.overlaid->params() : nullptr, &budget});
    }

    // What the linearisation and the solves did (the counts and times of SensitivityReport).
    [[nodiscard]] const SensitivityReport& report() const noexcept { return m_report; }

    DesignGradients run()
    {
        linearise();
        solveSubcases();
        return collect();
    }

    // The full mode: every item linearised (the options ask for every response, no mask), then G streamed.
    GradientStreamReport stream(const GradientStreamOptions& streamOptions, GradientSink& sink)
    {
        linearise();
        std::vector<ItemKey> rows; // the rows of G, response by response in the order of the evaluation
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
                rows.push_back({p, i});
        std::vector<StreamSubcase> subcases; // bounded by the subcases
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (!m_subcases[s].isAeroelastic && !m_subcases[s].isTermOnly) // complete rows (TrimItems); no item
                subcases.push_back({m_subcases[s].subcase.label, &systemOf(m_subcases[s]), s});
        GradientStreamInputs streamInputs;
        streamInputs.linearised = &m_linearised;
        streamInputs.rows = rows;
        streamInputs.subcases = subcases;
        streamInputs.loads = m_pseudoLoads.get();
        streamInputs.columnCount = m_variableCount;
        streamInputs.pool = m_in.pool;
        GradientStreamReport report = GradientStream::run(streamInputs, streamOptions, sink, m_budget);
        for (const std::unique_ptr<SensitivitySystem>& system : m_systems)
            m_report.factorisations += system->factorisationCount();
        collectCondensedStatistics(m_report);
        m_report.solves += report.solves;
        report.linearisation = std::move(m_report);
        return report;
    }

protected:
    // Steps 1 and 2 of StaticSensitivity.hpp: every wanted item with its weights and explicit part, the pseudo-loads
    // of the fields and the factorised systems of the subcases.
    void linearise()
    {
        selectResponses();
        initialiseItems();
        prepareSubcases();
        lineariseDesignResp1();
        lineariseExternals();
        const auto itemsStarted = std::chrono::steady_clock::now();
        lineariseElementJobs();
        m_report.itemSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - itemsStarted).count();
        lineariseEigenItems();
        if (m_modal)
            m_modal->prepare();
        lineariseTrimItems();
        const auto fieldsStarted = std::chrono::steady_clock::now();
        differenceFields();
        m_report.fieldSeconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - fieldsStarted).count();
        addExternalUnitLoads();
        finishEigenItems();
        if (m_modal)
        {
            m_modal->addExplicitParts(m_quantities, m_linearised);
            m_report.solves += m_modal->solveCount();
            m_report.factorisations += static_cast<std::int32_t>(m_modal->factorisationCount());
        }
        finishTrimItems();
        compressExplicitParts();
        foldSelfAdjoint();
        lineariseEquations();
    }

    [[nodiscard]] const ResponseValues& valuesAt(std::size_t p) const { return (*m_in.values)[p]; }
    // The value check of an item recomputed for its derivatives (detail::checkValue, ValueCheck), skipped at a
    // perturbed state, whose values are not the evaluation's (m_checksValues).
    void checkValue(const ResponseValues& values, std::size_t item, double recomputed, double terms = 0.0) const
    {
        if (m_checksValues)
            detail::checkValue(values, item, recomputed, terms);
    }
    // The element families of the analysis: those that leave out the elements of its external units, when it has some
    // (every element set of the sensitivities, the perturbed ones included, sees the model as the analysis did).
    [[nodiscard]] const elements::ElementRegistry& elementRegistry() const noexcept
    {
        return m_replaced ? m_replaced->registry() : m_registry;
    }
    // Whether the analysis replaced zones by external units.
    [[nodiscard]] bool hasExternalUnits() const
    {
        return m_in.results != nullptr && !m_in.results->externalUnits.empty();
    }
    [[nodiscard]] std::size_t gSize() const { return static_cast<std::size_t>(m_dofMap.gSetSize()); }

    void selectResponses()
    {
        const std::vector<ResponseValues>& values = *m_in.values;
        for (std::size_t p = 0; p < values.size(); ++p)
            m_positionOf[values[p].id] = p;
        std::set<std::int32_t> requested(m_options.responseIds.begin(), m_options.responseIds.end());
        for (const std::int32_t id : requested)
            if (m_positionOf.count(id) == 0)
                throw base::OptimisationError(base::concatenate(
                    {"gradients: expected an evaluated response, found response ", std::to_string(id),
                     ", which the evaluation does not hold (not defined, or evaluated in no subcase); ask the "
                     "gradients of the responses the evaluation holds"}));
        if (requested.empty())
            for (const ResponseValues& value : values)
                requested.insert(value.id);
        m_requested = requested;
        // The responses an equation reads, through every level (an equation asked for with a mask that keeps none of
        // its items reads nothing); every item of those is linearised for the equations.
        std::vector<std::int32_t> pending;
        for (const std::int32_t id : requested)
            if (hasWantedItem(id))
                pending.push_back(id);
        m_needed = requested;
        while (!pending.empty())
        {
            const std::int32_t id = pending.back();
            pending.pop_back();
            if (const EquationResponse* equation = m_in.responses->findEquation(id))
                for (const std::int32_t argument : equation->responseArguments())
                {
                    m_readByEquation.insert(argument);
                    if (m_needed.insert(argument).second)
                        pending.push_back(argument);
                }
        }
    }

    // Whether a requested response keeps at least one item through its mask (always without a mask).
    [[nodiscard]] bool hasWantedItem(std::int32_t id) const
    {
        const auto position = m_positionOf.find(id);
        if (position == m_positionOf.end())
            return false;
        for (std::size_t i = 0; i < valuesAt(position->second).itemCount(); ++i)
            if (isItemDifferentiated(m_options, id, i))
                return true;
        return false;
    }

    // Whether item i of the response at position p is linearised: every item of a response an equation reads, the
    // items the mask keeps of a requested one.
    [[nodiscard]] bool isWanted(std::size_t p, std::size_t i) const
    {
        const std::int32_t id = valuesAt(p).id;
        return m_readByEquation.count(id) != 0 || isItemDifferentiated(m_options, id, i);
    }

    void initialiseItems()
    {
        const std::vector<ResponseValues>& values = *m_in.values;
        m_linearised.items.resize(values.size());
        m_linearised.gradients.resize(values.size());
        for (std::size_t p = 0; p < values.size(); ++p)
        {
            if (m_needed.count(values[p].id) == 0)
                continue;
            // The self-adjoint responses read the load derivatives apart (PseudoLoads::loadPartOf).
            if (const DesignResp1Response* response = m_in.responses->findDesignResp1(values[p].id);
                response != nullptr && isSelfAdjointType(response->definition().type))
                m_keepsLoadParts = true;
            // An external response that reads the loads of a zone: its explicit part needs ∂P/∂v.
            if (const ExternalResponse* external = m_in.responses->findExternal(values[p].id); external != nullptr)
                for (const ExternalArgument& argument : external->definition().arguments)
                    m_keepsLoadParts = m_keepsLoadParts || argument.kind == ExternalArgumentKind::ZoneLoads;
            m_linearised.items[p].resize(values[p].itemCount());
            for (std::size_t i = 0; i < values[p].itemCount(); ++i)
            {
                ItemLinearisation& item = m_linearised.items[p][i];
                item.subcase = values[p].items[i].subcase;
                if (values[p].items[i].mode != 0)
                    item.subcase = 0; // the modal adjoint supplies the complete explicit row
                // A LAMA item reads the displacements of its preload subcase (EigenItems: the weights of K_d(u)).
                if (const DesignResp1Response* response = m_in.responses->findDesignResp1(values[p].id);
                    m_eigen && response != nullptr && response->definition().type == ResponseType::BucklingFactor)
                    if (const std::optional<std::int32_t> preload = m_eigen->preloadOf(item.subcase))
                        item.subcase = *preload;
                // An item that failed or was not evaluated has no gradient.
                item.isWanted = isWanted(p, i) && values[p].isOk(i);
                // The explicit parts start empty (sparse rows: an absent column is 0).
            }
        }
    }

    // The subcases the needed items are in, their displacements, temperatures, loads and constraint sets.
    void prepareSubcases()
    {
        std::set<std::int32_t> used;
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            for (const ItemLinearisation& item : m_linearised.items[p])
                if (item.subcase != 0 && item.isWanted)
                    used.insert(item.subcase);
        // The aids that need every analysed subcase whatever the items (displacement derivatives, hints).
        if (m_preparesEverySubcase)
            used.insert(m_in.subcases.begin(), m_in.subcases.end());
        base::DiagnosticSink sink;
        std::vector<base::EntityId> asked;
        for (const std::int32_t subcase : m_in.subcases)
            if (used.count(subcase) != 0)
                asked.push_back(subcase);
        if (m_trim)
            for (const std::int32_t subcase : m_trim->subcases())
                if (used.count(subcase) != 0 && std::find(asked.begin(), asked.end(), subcase) == asked.end())
                    asked.push_back(subcase);
        if (asked.empty())
        {
            m_pseudoLoads = std::make_unique<PseudoLoads>(m_quantities, m_variableCount, 0);
            return;
        }
        const input::ModelSnapshot& snapshot = *m_in.overlaid;
        const std::vector<solutions::StaticSubcase> subcases = solutions::staticSubcasesOf(snapshot, asked, sink);
        // The terms of the SUBCOMs read: their load parts make the SUBCOMs' pseudo-loads, so they are
        // linearised even when no item reads them (then never solved).
        std::set<std::int32_t> termsOnly;
        for (const solutions::StaticSubcase& subcase : subcases)
            if (subcase.isCombination && used.count(subcase.label) != 0)
            {
                refuseCombination(subcase, subcases);
                for (const base::EntityId term : subcase.combined)
                    if (used.insert(term).second)
                        termsOnly.insert(term);
            }
        const input::ConstraintTable* constraints =
            snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
        m_loads = std::make_unique<assembly::LoadVectorBuilder>(m_set, m_dofMap, m_budget);
        std::vector<std::unique_ptr<assembly::SpcSet>> spcs; // bounded by the subcases
        std::vector<std::vector<base::DofIndex>> supports;
        std::vector<std::vector<solutions::StaticSubcase>> groups;
        for (const solutions::StaticSubcase& subcase : subcases)
        {
            if (used.count(subcase.label) == 0)
                continue;
            SubcaseState state;
            state.subcase = subcase;
            state.isTermOnly = termsOnly.count(subcase.label) != 0;
            state.displacements = base::BudgetedBuffer<double>(m_budget, gSize(), "sensitivity displacements");
            std::fill(state.displacements.data(), state.displacements.data() + gSize(), 0.0);
            if (subcase.isCombination)
            {
                // A SUBCOM: no load and no constraint set of its own; its displacements, its terms and its group
                // once every subcase is read (prepareCombinations). Its own temperatures recover its items.
                state.subcase.load.reset();
                state.temperatures =
                    solutions::initialStrainsOf(snapshot, m_index.points(), subcase, m_budget, sink);
                state.group = NO_GROUP;
                m_subcaseIndex[subcase.label] = m_subcases.size();
                m_subcases.push_back(std::move(state));
                continue;
            }
            if (m_trim && m_trim->isTrimSubcase(subcase.label))
            {
                // The trimmed state: no loads, no constraint set of the static solution.
                const std::span<const double> trimmed = m_trim->displacementsOf(subcase.label);
                std::copy(trimmed.begin(), trimmed.end(), state.displacements.data());
                state.isAeroelastic = true;
                state.group = NO_GROUP;
                m_subcaseIndex[subcase.label] = m_subcases.size();
                m_subcases.push_back(std::move(state));
                continue;
            }
            readDisplacements(*m_in.results, m_dofMap, subcase.label, state.displacements.span());
            // A linearisation at a perturbed state (the Hessian-vector products): u + σ·δu.
            if (const auto shift = m_displacementShifts.find(subcase.label); shift != m_displacementShifts.end())
                for (std::size_t g = 0; g < gSize(); ++g)
                    state.displacements[g] += shift->second[g];
            state.temperatures = solutions::initialStrainsOf(snapshot, m_index.points(), subcase, m_budget, sink);
            if (subcase.load)
            {
                state.hasInertialLoads = m_loads->hasFamilyLoads(*subcase.load, assembly::LoadFamily::Inertial);
                state.hasDistributedLoads = m_loads->hasFamilyLoads(*subcase.load, assembly::LoadFamily::Distributed);
                state.hasConcentratedLoads = m_loads->hasFamilyLoads(*subcase.load, assembly::LoadFamily::Concentrated);
                const base::BudgetedBuffer<assembly::LoadVectorBuilder::CardValue> distributed =
                    m_loads->distributedValues(*subcase.load, m_budget);
                for (std::size_t k = 0; k < distributed.size(); ++k)
                    state.distributedElements.insert(distributed[k].elementId);
                if (state.hasDistributedLoads)
                {
                    base::DiagnosticSink ignored; // the analysis reported the loads it could not apply
                    const assembly::LoadVectorBuilder::ElementDistributedLoads loads =
                        m_loads->elementDistributedLoads(*subcase.load, ignored);
                    for (std::size_t b = 0; b < loads.pressures.size(); ++b)
                        for (const elements::ElementPressure& pressure : loads.pressures[b])
                            state.elementLoads[{b, pressure.elementRow}].first.push_back(pressure);
                    for (std::size_t b = 0; b < loads.lineLoads.size(); ++b)
                        for (const elements::ElementLineLoad& line : loads.lineLoads[b])
                            state.elementLoads[{b, line.elementRow}].second.push_back(line);
                }
            }
            // The constraint set, as StaticSolution groups the subcases (the same MPC set, constrained DOFs,
            // FreeBodySupport).
            auto spc = std::make_unique<assembly::SpcSet>(constraints, snapshot.findLoads(), m_dofMap, subcase.spc,
                                                          subcase.load, m_budget, sink);
            state.hasEnforcedDisplacements = !spc->enforcedDisplacements().empty();
            std::vector<base::DofIndex> support = assembly::supportDofsOf(snapshot, m_dofMap, subcase.tableId, sink);
            std::size_t group = groups.size();
            for (std::size_t g = 0; g < groups.size(); ++g)
                if (groups[g].front().mpc == subcase.mpc && groups[g].front().contact == subcase.contact &&
                    spcs[g]->hasSameConstrainedDofs(*spc) && supports[g] == support)
                {
                    group = g;
                    break;
                }
            if (group == groups.size())
            {
                groups.emplace_back();
                spcs.push_back(std::move(spc));
                supports.push_back(std::move(support));
            }
            groups[group].push_back(subcase);
            state.group = group;
            m_subcaseIndex[subcase.label] = m_subcases.size();
            m_subcases.push_back(std::move(state));
        }
        if (sink.hasErrors())
            for (const base::Diagnostic& diagnostic : sink.diagnostics())
                if (diagnostic.severity == base::Severity::Error)
                    throw base::OptimisationError(std::string("the sensitivities cannot read the subcases: ")
                                                      .append(base::formatDiagnostic(diagnostic)));
        prepareCombinations();
        for (std::vector<solutions::StaticSubcase>& group : groups)
        {
            // The condensed path: the condensed constraint set the analysis kept, the solves through the
            // boundaries (BoundaryReducedSensitivity).
            std::shared_ptr<const solutions::CondensedFactorisedSet> condensed;
            for (const std::shared_ptr<const solutions::CondensedFactorisedSet>& candidate :
                 m_in.results->condensedSystems)
                if (candidate->kernel() == m_in.kernel && candidate->dofMap().gSetSize() == m_dofMap.gSetSize() &&
                    std::all_of(group.begin(), group.end(), [&](const solutions::StaticSubcase& subcase)
                                { return candidate->solves(subcase.label); }))
                {
                    condensed = candidate;
                    break;
                }
            if (condensed)
            {
                m_systems.push_back(std::make_unique<SensitivitySystem>(std::move(condensed), m_set, std::move(group),
                                                                        m_in.kernel, m_budget, m_in.pool));
                continue;
            }
            if (m_in.results->path == solutions::AnalysisPath::Condensed)
                throw base::OptimisationError(
                    "the sensitivities of an analysis on the condensed path solve through the boundaries on the "
                    "condensed constraint sets the analysis kept, and this analysis kept none; evaluate the point "
                    "with its factors kept (keep_analysis=True, the default)");
            // The constraint set the analysis kept (StaticOptions::keepsSystems) when it solved these subcases with
            // the same kernel on this DOF map: no factorisation; otherwise the factor is made again.
            std::shared_ptr<const solutions::FactorisedSet> kept;
            for (const std::shared_ptr<const solutions::FactorisedSet>& candidate : m_in.results->systems)
                if (candidate->kernel() == m_in.kernel && candidate->dofMap().gSetSize() == m_dofMap.gSetSize() &&
                    std::all_of(group.begin(), group.end(), [&](const solutions::StaticSubcase& subcase)
                                { return candidate->solves(subcase.label); }))
                {
                    kept = candidate;
                    break;
                }
            if (!kept && hasExternalUnits())
                throw base::OptimisationError(
                    "the sensitivities of an analysis with an external unit solve on the factor the analysis kept (its "
                    "K* is in it), and this analysis kept none; evaluate the point with its factors kept "
                    "(keep_analysis=True, the default)");
            if (kept)
                m_systems.push_back(std::make_unique<SensitivitySystem>(std::move(kept), std::move(group), m_in.kernel,
                                                                        m_budget, m_in.pool));
            else
                m_systems.push_back(std::make_unique<SensitivitySystem>(m_set, m_dofMap, std::move(group), m_in.kernel,
                                                                        m_budget, m_in.pool));
        }
        // Inertia relief: the relieved subcases, their acceleration fields D·ü_r; the pseudo-loads keep
        // the load parts the relief projects. At a moved state of the Hessian-vector products the relief
        // of the moved point (movedReliefOf, movedAccelerationsOf).
        using Accelerations = std::array<double, assembly::MAX_SUPPORT_DOFS>;
        std::vector<std::optional<Accelerations>> accelerationsOf(m_subcases.size());
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const SubcaseState& state = m_subcases[s];
            if (state.isAeroelastic || state.isCombination() || m_systems[state.group]->relief() == nullptr)
                continue;
            accelerationsOf[s] = m_reliefPoint ? movedAccelerationsOf(s)
                                               : m_systems[state.group]->reliefAccelerations(state.subcase.label);
        }
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const SubcaseState& state = m_subcases[s];
            if (!state.isCombination() || state.group == NO_GROUP || m_systems[state.group]->relief() == nullptr)
                continue;
            // A SUBCOM of relieved subcases: the combination of their accelerations (its relieving loads are its
            // terms', combined with their load parts).
            Accelerations combined{};
            for (const auto& [term, coefficient] : state.terms)
                if (accelerationsOf[term])
                    for (std::size_t k = 0; k < combined.size(); ++k)
                        combined[k] += coefficient * (*accelerationsOf[term])[k];
            accelerationsOf[s] = combined;
        }
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            if (!accelerationsOf[s])
                continue;
            SubcaseState& state = m_subcases[s];
            const Accelerations& accelerations = *accelerationsOf[s];
            state.relief = m_reliefPoint ? &movedReliefOf(state.group) : m_systems[state.group]->relief();
            state.reliefAccelerations = base::BudgetedBuffer<double>(m_budget, gSize(), "sensitivity relief field");
            std::fill(state.reliefAccelerations.data(), state.reliefAccelerations.data() + gSize(), 0.0);
            const linalg::DenseMatrix& modes = state.relief->supportModes;
            for (std::size_t k = 0; k < modes.columns(); ++k)
                for (std::size_t dof = 0; dof < gSize(); ++dof)
                    state.reliefAccelerations[dof] += modes(dof, k) * accelerations[k];
            m_keepsLoadParts = true;
        }
        // A SUBCOM takes its terms' load parts (PseudoLoads::addCombination).
        if (std::any_of(m_subcases.begin(), m_subcases.end(),
                        [](const SubcaseState& state) { return state.isCombination(); }))
            m_keepsLoadParts = true;
        m_pseudoLoads =
            std::make_unique<PseudoLoads>(m_quantities, m_variableCount, m_subcases.size(), m_keepsLoadParts);
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (m_subcases[s].relief != nullptr)
                m_pseudoLoads->setRelief(s, m_subcases[s].relief);
    }

    // The change of an element's mass along a direction of the Hessian-vector products: its mass on the
    // moved point's set less its mass on the point's (ReliefPoint::set), on the element's g-set DOFs.
    struct ElementChange
    {
        std::vector<base::DofIndex> dofs;
        std::vector<double> values;
    };
    [[nodiscard]] std::optional<ElementChange> massChangeOf(const ElementRef& element,
                                                            const elements::MassOptions& options) const
    {
        const assembly::AssemblyBatch& moved = m_set.batches()[element.batch];
        const assembly::AssemblyBatch& point = m_reliefPoint->set->batches()[element.batch];
        if (!moved.family->facets().has(elements::Facet::Mass))
            return std::nullopt;
        elements::ElementMatrixBlock after;
        elements::ElementMatrixBlock before;
        moved.family->computeMass(moved.view, {element.row, element.row + 1}, options, after);
        point.family->computeMass(point.view, {element.row, element.row + 1}, options, before);
        if (after.empty() && before.empty())
            return std::nullopt;
        if (after.size() != 1 || before.size() != 1 || after.dofs(0).size() != before.dofs(0).size() ||
            !std::equal(after.dofs(0).begin(), after.dofs(0).end(), before.dofs(0).begin()))
            throw base::OptimisationError(base::concatenate(
                {"hessian_vector under inertia relief: the mass matrix of ", moved.view.card(), " ",
                 std::to_string(moved.view.elementId(element.row)),
                 " changes its layout along the direction; give a direction that keeps the element's form"}));
        ElementChange change;
        for (const elements::ElementDof& dof : after.dofs(0))
            change.dofs.push_back(m_dofMap.dofOf(dof));
        change.values.resize(after.matrix(0).size());
        for (std::size_t k = 0; k < change.values.size(); ++k)
            change.values[k] = after.matrix(0)[k] - before.matrix(0)[k];
        return change;
    }

    // The relief of constraint set `group` at the moved point: D that of the point (rigid
    // support motions: the analysis refused supports that are not statically determinate, so D does not depend on the
    // design; on the condensed path the geometric rigid-body modes Φ·C), M'·D = M·D + WTMASS·Σ_e ΔM_e·D over the
    // moved elements, and (Dᵀ·M'·D)⁻¹ = (Dᵀ·M·D + ΔR)⁻¹, ΔR = WTMASS·Σ_e Dᵀ·ΔM_e·D; made once per group.
    // PERFORMANCE (accuracy): the changes ΔR and Dᵀ·ΔP are summed over the moved elements alone and added to the
    // point's Dᵀ·M·D, formed the same way at both moved states, so that the round-off of the sums over the model (some
    // 1e-9 of ü on the free barrels in millimetres) is common to the two sides of the central difference and cancels.
    // Measured: a free barrel of mixed fidelity (the product of a CompProp ply, 7.6e-4).
    const solutions::KeptRelief& movedReliefOf(std::size_t group)
    {
        if (m_movedReliefs.size() < m_systems.size())
        {
            m_movedReliefs.resize(m_systems.size());
            m_reducedMassChanges.resize(m_systems.size());
        }
        std::unique_ptr<solutions::KeptRelief>& made = m_movedReliefs[group];
        if (made)
            return *made;
        const solutions::KeptRelief& point = *m_reliefPoint->reliefs.at(group);
        const std::size_t g = gSize();
        const std::size_t r = point.supportCount();
        made = std::make_unique<solutions::KeptRelief>();
        made->rigidResidual = point.rigidResidual;
        made->supportModes = linalg::DenseMatrix(m_budget, g, r, "moved relief D");
        made->massModes = linalg::DenseMatrix(m_budget, g, r, "moved relief M D");
        for (std::size_t k = 0; k < r; ++k)
            for (std::size_t dof = 0; dof < g; ++dof)
            {
                made->supportModes(dof, k) = point.supportModes(dof, k);
                made->massModes(dof, k) = point.massModes(dof, k);
            }
        const assembly::MassAssemblyOptions options = reliefMassOptions(group);
        elements::MassOptions massOptions;
        massOptions.formulation = options.formulation;
        std::vector<double>& change = m_reducedMassChanges[group]; // ΔR, r × r
        change.assign(r * r, 0.0);
        for (const ElementRef& element : m_reliefPoint->moved)
        {
            const std::optional<ElementChange> mass = massChangeOf(element, massOptions);
            if (!mass)
                continue;
            const std::size_t n = mass->dofs.size();
            std::vector<double> rows(n * r, 0.0); // WTMASS·ΔM_e·D on the element's DOFs, bounded by them
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t k = 0; k < r; ++k)
                {
                    double sum = 0.0;
                    for (std::size_t b = 0; b < n; ++b)
                        sum += mass->values[a * n + b] * point.supportModes(static_cast<std::size_t>(mass->dofs[b]), k);
                    rows[a * r + k] = options.weightToMass * sum;
                    made->massModes(static_cast<std::size_t>(mass->dofs[a]), k) += rows[a * r + k];
                }
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t k = 0; k < r; ++k)
                {
                    double sum = 0.0;
                    for (std::size_t a = 0; a < n; ++a)
                        sum += point.supportModes(static_cast<std::size_t>(mass->dofs[a]), i) * rows[a * r + k];
                    change[i * r + k] += sum;
                }
        }
        std::vector<double> reduced = pointReducedMassOf(group);
        for (std::size_t k = 0; k < r * r; ++k)
            reduced[k] += change[k];
        made->reducedMassInverse = inverseOf(reduced, r);
        return *made;
    }

    // Dᵀ·M·D of the point's relief of `group`, made once (the same at both moved states).
    [[nodiscard]] const std::vector<double>& pointReducedMassOf(std::size_t group)
    {
        if (m_pointReducedMasses.size() < m_systems.size())
            m_pointReducedMasses.resize(m_systems.size());
        std::vector<double>& reduced = m_pointReducedMasses[group];
        if (reduced.empty())
            reduced = reducedMassOf(*m_reliefPoint->reliefs.at(group));
        return reduced;
    }

    // ü' = (Dᵀ·M'·D)⁻¹·Dᵀ·P' of relieved subcase s at the moved point: Dᵀ·P = (Dᵀ·M·D)·ü of the point's
    // relief (its accelerations: Dᵀ·(P − M·D·ü) = 0), plus Dᵀ·ΔP of the moved elements (their inertial loads
    // WTMASS·ΔM_e·a, their pressures and line loads on both sets). The thermal loads are self-equilibrated (Dᵀ·P_T = 0
    // on rigid D) and add nothing.
    [[nodiscard]] std::optional<std::array<double, assembly::MAX_SUPPORT_DOFS>> movedAccelerationsOf(std::size_t s)
    {
        const SubcaseState& state = m_subcases[s];
        const std::optional<std::array<double, assembly::MAX_SUPPORT_DOFS>> atPoint =
            m_systems[state.group]->reliefAccelerations(state.subcase.label);
        const solutions::KeptRelief* pointRelief = m_reliefPoint->reliefs.at(state.group);
        if (!atPoint || pointRelief == nullptr)
            return atPoint;
        const solutions::KeptRelief& point = *pointRelief;
        const solutions::KeptRelief& moved = movedReliefOf(state.group);
        const std::vector<double>& change = m_reducedMassChanges[state.group];
        const std::size_t r = point.supportCount();
        // ü' = ü + (Dᵀ·M'·D)⁻¹·(Dᵀ·ΔP − ΔR·ü): Dᵀ·P = Dᵀ·M·D·ü at the point, so that only the changes are summed.
        std::array<double, assembly::MAX_SUPPORT_DOFS> generalised{};
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = 0; k < r; ++k)
                generalised[i] -= change[i * r + k] * (*atPoint)[k];
        const auto project = [&](const std::vector<base::DofIndex>& dofs, const std::vector<double>& values)
        {
            for (std::size_t a = 0; a < dofs.size(); ++a)
                for (std::size_t k = 0; k < r; ++k)
                    generalised[k] += point.supportModes(static_cast<std::size_t>(dofs[a]), k) * values[a];
        };
        if (state.hasInertialLoads && state.subcase.load)
        {
            const InertialRate& rate = inertialRateOf(s);
            elements::MassOptions massOptions;
            massOptions.formulation = rate.options.formulation;
            for (const ElementRef& element : m_reliefPoint->moved)
            {
                const std::optional<ElementChange> change = massChangeOf(element, massOptions);
                if (!change)
                    continue;
                const std::size_t n = change->dofs.size();
                std::vector<double> loads(n, 0.0); // bounded by the element's DOFs
                for (std::size_t a = 0; a < n; ++a)
                {
                    double sum = 0.0;
                    for (std::size_t b = 0; b < n; ++b)
                        sum +=
                            change->values[a * n + b] * rate.accelerations[static_cast<std::size_t>(change->dofs[b])];
                    loads[a] = rate.options.weightToMass * sum;
                }
                project(change->dofs, loads);
            }
        }
        for (const ElementRef& element : m_reliefPoint->moved)
        {
            const auto found = state.elementLoads.find(element);
            if (found == state.elementLoads.end())
                continue;
            for (const assembly::ElementSet* set :
                 std::array<const assembly::ElementSet*, 2>{&m_set, m_reliefPoint->set})
            {
                const assembly::AssemblyBatch& batch = set->batches()[element.batch];
                elements::ElementVectorBlock block;
                if (!found->second.first.empty())
                    batch.family->computePressureLoads(batch.view, found->second.first, block);
                if (!found->second.second.empty())
                    batch.family->computeLineLoads(batch.view, found->second.second, block);
                const double sign = set == &m_set ? 1.0 : -1.0;
                for (std::size_t v = 0; v < block.size(); ++v)
                {
                    std::vector<base::DofIndex> dofs; // bounded by the element's DOFs
                    std::vector<double> values;
                    for (std::size_t d = 0; d < block.dofs(v).size(); ++d)
                    {
                        dofs.push_back(m_dofMap.dofOf(block.dofs(v)[d]));
                        values.push_back(sign * block.values(v)[d]);
                    }
                    project(dofs, values);
                }
            }
        }
        std::array<double, assembly::MAX_SUPPORT_DOFS> result = *atPoint;
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = 0; k < r; ++k)
                result[i] += moved.reducedMassInverse[i * r + k] * generalised[k];
        return result;
    }

    // Dᵀ·M·D (r × r, row-major) of a relief, from its D and M·D.
    [[nodiscard]] std::vector<double> reducedMassOf(const solutions::KeptRelief& relief) const
    {
        const std::size_t r = relief.supportCount();
        std::vector<double> reduced(r * r, 0.0); // r ≤ 6
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = 0; k < r; ++k)
            {
                double sum = 0.0;
                for (std::size_t dof = 0; dof < gSize(); ++dof)
                    sum += relief.supportModes(dof, i) * relief.massModes(dof, k);
                reduced[i * r + k] = sum;
            }
        return reduced;
    }

    // The inverse of an r × r matrix (r ≤ 6, symmetric positive definite: Dᵀ·M·D) by Gauss-Jordan, as the condensed
    // relief makes it (SensitivitySystem).
    [[nodiscard]] static std::array<double, assembly::MAX_SUPPORT_DOFS * assembly::MAX_SUPPORT_DOFS> inverseOf(
        const std::vector<double>& matrix, std::size_t r)
    {
        std::array<double, assembly::MAX_SUPPORT_DOFS * 2 * assembly::MAX_SUPPORT_DOFS> augmented{};
        const std::size_t width = 2 * r;
        for (std::size_t i = 0; i < r; ++i)
        {
            for (std::size_t k = 0; k < r; ++k)
                augmented[i * width + k] = matrix[i * r + k];
            augmented[i * width + r + i] = 1.0;
        }
        for (std::size_t p = 0; p < r; ++p)
        {
            const double pivot = augmented[p * width + p];
            for (std::size_t k = 0; k < width; ++k)
                augmented[p * width + k] /= pivot;
            for (std::size_t i = 0; i < r; ++i)
            {
                if (i == p)
                    continue;
                const double factor = augmented[i * width + p];
                for (std::size_t k = 0; k < width; ++k)
                    augmented[i * width + k] -= factor * augmented[p * width + k];
            }
        }
        std::array<double, assembly::MAX_SUPPORT_DOFS * assembly::MAX_SUPPORT_DOFS> inverse{};
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = 0; k < r; ++k)
                inverse[i * r + k] = augmented[i * width + r + k];
        return inverse;
    }

    // The mass options of the relief of a constraint set: those of its first subcase (reliefRateOf's).
    [[nodiscard]] assembly::MassAssemblyOptions reliefMassOptions(std::size_t group) const
    {
        base::DiagnosticSink ignored; // the analysis reported the problems of its parameters
        const input::ParamTable* params =
            m_in.overlaid->hasTable(input::TableRole::Params) ? &m_in.overlaid->params() : nullptr;
        for (const SubcaseState& state : m_subcases)
            if (state.group == group && !state.isCombination())
                return assembly::MassAssemblyOptions::fromParams(params, state.subcase.label, ignored);
        return assembly::MassAssemblyOptions::fromParams(params, std::nullopt, ignored);
    }

    // What the sensitivities of a SUBCOM refuse, before any work: terms that are SUBCOMs or that
    // are not SUBCASEs of the analysis, an analysis with external units.
    void refuseCombination(const solutions::StaticSubcase& combination,
                           const std::vector<solutions::StaticSubcase>& subcases) const
    {
        const std::string name =
            base::concatenate({"the sensitivities of SubcaseCombination ", std::to_string(combination.label)});
        if (hasExternalUnits())
            throw base::OptimisationError(base::concatenate(
                {name,
                 ": expected an analysis without external units, found some (their loads are given on the "
                 "basis of each constraint set, not on a combination); constrain the responses of the subcases "
                 "it combines instead"}));
        for (const base::EntityId term : combination.combined)
        {
            const auto found = std::find_if(subcases.begin(), subcases.end(),
                                            [&](const solutions::StaticSubcase& s) { return s.label == term; });
            if (found == subcases.end() || found->isCombination)
                throw base::OptimisationError(
                    base::concatenate({name, ": expected SUBCASEs as its terms, found subcase ", std::to_string(term),
                                       found == subcases.end() ? std::string(", which the analysis did not solve")
                                                               : std::string(", a SubcaseCombination"),
                                       "; give the SubcaseStarts themselves in its SubcaseCombinationFactors"}));
        }
    }

    // The SUBCOMs' terms, constraint sets and displacements, once every subcase is read: u_c = Σ a_k·u_k
    // as the static solution combines them (the terms in its order, from zero: the same operations), from the
    // displacements of the analysis (a term's shift at a perturbed state does not enter: the SUBCOM's own shift,
    // u_c + σ·δu_c, is added to it).
    void prepareCombinations()
    {
        for (SubcaseState& state : m_subcases)
        {
            if (!state.isCombination())
                continue;
            const std::string name =
                base::concatenate({"the sensitivities of SubcaseCombination ", std::to_string(state.subcase.label)});
            state.terms.clear();
            base::BudgetedBuffer<double> term(m_budget, gSize(), "SUBCOM term displacements");
            for (std::size_t t = 0; t < state.subcase.combined.size(); ++t)
            {
                const std::size_t index = m_subcaseIndex.at(state.subcase.combined[t]);
                const SubcaseState& of = m_subcases[index];
                if (of.isAeroelastic)
                    throw base::OptimisationError(
                        base::concatenate({name, ": expected static SUBCASEs as its terms, found SAERO subcase ",
                                           std::to_string(of.subcase.label), "; give it static subcases to combine"}));
                if (state.terms.empty())
                    state.group = of.group;
                else if (of.group != state.group)
                    throw base::OptimisationError(base::concatenate(
                        {name,
                         ": expected its terms in one constraint set (the same SinglePointConstrSelection, "
                         "MultiPointConstrSelection, FreeBodySupport and contact), "
                         "found subcase ",
                         std::to_string(of.subcase.label), " in another one than subcase ",
                         std::to_string(m_subcases[state.terms.front().first].subcase.label),
                         " (a SubcaseCombination across constraint sets needs one adjoint per set and item: "
                         "not computed by this "
                         "version); constrain the responses of the subcases it combines instead"}));
                state.terms.emplace_back(index, state.subcase.coefficients[t]);
                state.hasEnforcedDisplacements = state.hasEnforcedDisplacements || of.hasEnforcedDisplacements;
                std::fill(term.data(), term.data() + gSize(), 0.0);
                readDisplacements(*m_in.results, m_dofMap, of.subcase.label, term.span());
                const double coefficient = state.subcase.coefficients[t];
                for (std::size_t g = 0; g < gSize(); ++g)
                    state.displacements[g] += coefficient * term[g];
            }
            if (const auto shift = m_displacementShifts.find(state.subcase.label); shift != m_displacementShifts.end())
                for (std::size_t g = 0; g < gSize(); ++g)
                    state.displacements[g] += shift->second[g];
        }
    }

    [[nodiscard]] SubcaseState& stateOf(std::int32_t subcase) { return m_subcases[m_subcaseIndex.at(subcase)]; }
    [[nodiscard]] const SensitivitySystem& systemOf(const SubcaseState& state) const
    {
        if (state.isAeroelastic)
            throw base::OptimisationError(base::concatenate(
                {"the sensitivities of SAERO subcase ", std::to_string(state.subcase.label),
                 ": this version differentiates there the TRIM and STABDER responses and the DISP, STRESS, STRAIN, "
                 "FORCE and ESE items of the trimmed state, not this response or aid; evaluate it in a "
                 "STATICS subcase"}));
        return *m_systems[state.group];
    }
    // The group of a SAERO subcase, which has no constraint set of the static solution.
    static constexpr std::size_t NO_GROUP = static_cast<std::size_t>(-1);

    [[nodiscard]] std::optional<std::size_t> setBatchOf(const std::string& card) const
    {
        for (std::size_t b = 0; b < m_set.batches().size(); ++b)
            if (m_set.batches()[b].view.card() == card)
                return b;
        return std::nullopt;
    }

    void lineariseDesignResp1()
    {
        std::map<std::tuple<std::size_t, std::int64_t, std::size_t>, std::size_t> jobOf;
        std::vector<std::vector<ElementItemSpec>> specs;
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
        {
            const ResponseValues& values = valuesAt(p);
            if (m_linearised.items[p].empty())
                continue;
            const DesignResp1Response* response = m_in.responses->findDesignResp1(values.id);
            if (response == nullptr)
                continue;
            const ResponseType type = response->definition().type;
            for (std::size_t i = 0; i < values.itemCount(); ++i)
            {
                const ItemKey key{p, i};
                const ResponseItem& label = values.items[i];
                ItemLinearisation& item = m_linearised.at(key);
                if (!item.isWanted)
                    continue;
                if (response->definition().mode)
                {
                    if (!m_modal)
                        throw base::OptimisationError("modal field sensitivities require a vibration analysis");
                    m_modal->add(key, *response, label, values.values[i]);
                    continue;
                }
                if (m_trim && m_trim->isTrimSubcase(label.subcase) && !isAeroelasticType(type) &&
                    type != ResponseType::Displacement && !isElementType(type))
                    throw base::OptimisationError(base::concatenate(
                        {"the sensitivity of ", responseTypeName(type), " response ", std::to_string(values.id),
                         " in SAERO subcase ", std::to_string(label.subcase),
                         ": this version differentiates there the TRIM and STABDER responses and the DISP, STRESS, "
                         "STRAIN, FORCE and ESE items of the trimmed state; evaluate it in a STATICS "
                         "subcase"}));
                switch (type)
                {
                case ResponseType::Weight:
                case ResponseType::Volume:
                case ResponseType::FractionalMass:
                    m_massItems.push_back({key, response, 0.0});
                    break;
                case ResponseType::Displacement:
                    if (const std::optional<base::DofIndex> dof = m_dofMap.findDof(label.entity, label.component))
                        item.weights = {{*dof, 1.0}};
                    break;
                case ResponseType::SpcForce:
                    lineariseSpcForce(key, label);
                    break;
                case ResponseType::Compliance:
                case ResponseType::MeanDisplacement:
                    lineariseSelfAdjoint(key, label, type);
                    break;
                case ResponseType::DeflectionToThickness:
                case ResponseType::Rotation:
                    linearisePoint(key, label, type);
                    break;
                case ResponseType::SectionLoad:
                case ResponseType::CutLoad:
                    lineariseSection(key, label, *response);
                    break;
                case ResponseType::ContourDisplacement:
                case ResponseType::ContourForce:
                    lineariseContour(key, label, type);
                    break;
                case ResponseType::Eigenvalue:
                case ResponseType::Frequency:
                case ResponseType::BucklingFactor:
                    if (!m_eigen)
                        throw base::OptimisationError(base::concatenate(
                            {"the sensitivity of ", responseTypeName(type), " response ", std::to_string(values.id),
                             ": the analysis has no eigenvalue solve; analyse the MODES or BUCK "
                             "subcases of the response"}));
                    m_eigen->add(key, type, label, values.values[i]);
                    break;
                case ResponseType::TrimVariable:
                case ResponseType::StabilityDerivative:
                    if (!m_trim)
                        throw base::OptimisationError(base::concatenate(
                            {"the sensitivity of ", responseTypeName(type), " response ", std::to_string(values.id),
                             ": the analysis has no trim; analyse the SAERO subcases of the "
                             "response"}));
                    m_trim->add(key, type, label);
                    break;
                default:
                {
                    const auto& targets = response->elementTargets();
                    const auto target =
                        std::find_if(targets.begin(), targets.end(), [&](const DesignResp1Response::ElementTarget& t)
                                     { return t.elementId == label.entity; });
                    if (target == targets.end())
                        break;
                    const auto location =
                        std::find_if(target->items.begin(), target->items.end(),
                                     [&](const auto& entry) { return entry.first == label.component; });
                    const std::optional<std::size_t> batch = setBatchOf(target->card);
                    if (location == target->items.end() || !batch)
                        break;
                    const std::size_t subcase = m_subcaseIndex.at(label.subcase);
                    const auto jobKey = std::make_tuple(*batch, target->row, subcase);
                    auto [found, isNew] = jobOf.try_emplace(jobKey, m_jobs.size());
                    if (isNew)
                    {
                        ElementJob job;
                        job.element = {*batch, target->row};
                        job.subcase = subcase;
                        m_jobs.push_back(std::move(job));
                        specs.emplace_back();
                    }
                    specs[found->second].push_back({type, label.component, location->second, label.layer});
                    m_jobs[found->second].keys.push_back(key);
                    break;
                }
                }
            }
        }
        for (std::size_t j = 0; j < m_jobs.size(); ++j)
        {
            const assembly::AssemblyBatch& batch = m_set.batches()[m_jobs[j].element.batch];
            m_jobs[j].function =
                std::make_unique<ElementItemFunction>(*batch.family, m_jobs[j].element.row, std::move(specs[j]));
        }
        // The jobs of each element, in job order (differenceQuantity: a quantity visits the jobs of the elements it
        // reaches, not every job: 100 000 variables times 100 000 items).
        m_jobsByElement.clear();
        for (std::size_t j = 0; j < m_jobs.size(); ++j)
            m_jobsByElement.emplace_back(m_jobs[j].element, j);
        std::stable_sort(m_jobsByElement.begin(), m_jobsByElement.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        // The jobs with an ESE percentage: every quantity changes their total energy.
        m_percentJobs.clear();
        for (std::size_t j = 0; j < m_jobs.size(); ++j)
            for (std::size_t k = 0; k < m_jobs[j].keys.size(); ++k)
                if (isPercent(m_jobs[j], k))
                {
                    m_percentJobs.push_back(j);
                    break;
                }
    }

    // An SPC force r = wᵀ·(Kgg·u − P), w = e_d + Σ_i Gm(i, d)·e_m(i) (RigidElimination::computeMpcForces: the force of
    // constraint at d is the element force minus the load minus the MPC force there, q_n = −Gmᵀ·q_m); c = Kgg·w.
    void lineariseSpcForce(const ItemKey& key, const ResponseItem& label)
    {
        const std::optional<base::DofIndex> dof = m_dofMap.findDof(label.entity, label.component);
        SubcaseState& state = stateOf(label.subcase);
        const SensitivitySystem& system = systemOf(state);
        if (!dof || !system.isConstrained(*dof))
            return; // no force of constraint there: the value is 0 whatever the design
        SparseVector w{{*dof, 1.0}};
        const assembly::RigidElimination& gm = system.gm();
        for (std::size_t r = 0; r < gm.rowCount(); ++r)
            for (const assembly::ConstraintTerm& term : gm.row(r))
                if (term.dof == *dof)
                    w.emplace_back(gm.mDofs()[r], term.coefficient);
        compressSparse(w);
        base::BudgetedBuffer<double> dense(m_budget, gSize(), "SPC force weights");
        std::fill(dense.data(), dense.data() + gSize(), 0.0);
        for (const auto& [index, value] : w)
            dense[static_cast<std::size_t>(index)] = value;
        base::BudgetedBuffer<double> product(m_budget, gSize(), "SPC force weights K·w");
        system.kgg().multiply(dense.span(), product.span());
        ItemLinearisation& item = m_linearised.at(key);
        item.weights.clear();
        for (std::size_t g = 0; g < gSize(); ++g)
            if (product[g] != 0.0)
                item.weights.emplace_back(static_cast<base::DofIndex>(g), product[g]);
        m_spcItems.push_back({key, m_subcaseIndex.at(label.subcase), std::move(w)});
    }

    // A compliance c = Pᵀu or a mean displacement m = c/W (SelfAdjointResponses): the weights ∂r/∂u = scale·P, taken as
    // scale·Kgg·u, which has the same l rows (K_ll·u_l = (Tᵀ·P)_l without enforced displacements) and gives the same
    // right-hand side to the adjoint solve; the value is checked against the analysis' (uᵀ·Kgg·u against twice the
    // external work).
    void lineariseSelfAdjoint(const ItemKey& key, const ResponseItem& label, ResponseType type)
    {
        SubcaseState& state = stateOf(label.subcase);
        const ResponseValues& values = valuesAt(key.response);
        if (state.hasEnforcedDisplacements)
            throw base::OptimisationError(base::concatenate(
                {"the sensitivity of response ", std::to_string(values.id), " (", values.label, ") in subcase ",
                 std::to_string(label.subcase),
                 ": the subcase enforces displacements, so the displacement is not the adjoint vector of its "
                 "compliance; constrain displacements instead"}));
        prepareTotalEnergy(state);
        SelfAdjointItem item{key, m_subcaseIndex.at(label.subcase), 1.0, values.values[key.item], {}};
        if (type == ResponseType::MeanDisplacement)
        {
            const std::vector<GridForce> forces =
                SelfAdjointResponses::appliedForces(m_in.results->arrays, label.subcase);
            const double magnitude = SelfAdjointResponses::forceMagnitude(forces);
            if (!(magnitude > 0.0))
                throw base::OptimisationError(
                    base::concatenate({"the sensitivity of response ", std::to_string(values.id), " (", values.label,
                                       "): the subcase has no applied force; use the compliance in that subcase"}));
            item.scale = 1.0 / magnitude;
            for (const GridForce& entry : forces)
            {
                const double length = std::sqrt(entry.force[0] * entry.force[0] + entry.force[1] * entry.force[1] +
                                                entry.force[2] * entry.force[2]);
                for (int c = 0; c < 3; ++c)
                    if (const std::optional<base::DofIndex> dof = m_dofMap.findDof(entry.grid, c + 1))
                        item.forceWeights.emplace_back(*dof, entry.force[static_cast<std::size_t>(c)] / length);
            }
            compressSparse(item.forceWeights);
        }
        checkValue(values, key.item, item.scale * 2.0 * state.totalEnergy);
        ItemLinearisation& linear = m_linearised.at(key);
        linear.weights.clear();
        for (std::size_t g = 0; g < gSize(); ++g)
            if ((*state.elementForces)[g] != 0.0)
                linear.weights.emplace_back(static_cast<base::DofIndex>(g), item.scale * (*state.elementForces)[g]);
        m_selfAdjointItems.push_back(std::move(item));
    }

    // A deflection over thickness |n·u|/t̄ or a rotation |θ| of a grid point (LinearityIndicator): the weights on its
    // three translations (rotations) in CD; the deflection keeps its point for the explicit part.
    void linearisePoint(const ItemKey& key, const ResponseItem& label, ResponseType type)
    {
        const SubcaseState& state = stateOf(label.subcase);
        const bool isDeflection = type == ResponseType::DeflectionToThickness;
        const int first = isDeflection ? 1 : 4;
        std::array<base::DofIndex, 3> dofs{};
        std::array<double, 3> components{};
        for (int c = 0; c < 3; ++c)
        {
            const std::optional<base::DofIndex> dof = m_dofMap.findDof(label.entity, first + c);
            if (!dof)
                throw base::OptimisationError(
                    base::concatenate({"the sensitivity of grid point ", std::to_string(label.entity),
                                       ": expected a grid point with six components; name grid points, not scalar "
                                       "points"}));
            dofs[static_cast<std::size_t>(c)] = *dof;
            components[static_cast<std::size_t>(c)] = state.displacements[static_cast<std::size_t>(*dof)];
        }
        std::array<double, 3> weights{};
        double value = 0.0;
        if (isDeflection)
        {
            const ShellPointGeometry* point = linearity().findShellPoint(label.entity);
            if (point == nullptr)
                throw base::OptimisationError(base::concatenate(
                    {"the sensitivity of the deflection over thickness of grid point ", std::to_string(label.entity),
                     ": it has no shell element of positive area and thickness around it; leave it out of nodes= or "
                     "give its shells a thickness"}));
            weights = LinearityIndicator::deflectionWeights(*point, components);
            value = LinearityIndicator::deflectionRatio(*point, components);
            m_pointItems.push_back({key, label.entity, components});
        }
        else
        {
            weights = LinearityIndicator::rotationWeights(components);
            value = LinearityIndicator::rotationMagnitude(components);
        }
        checkValue(valuesAt(key.response), key.item, value);
        ItemLinearisation& linear = m_linearised.at(key);
        linear.weights.clear();
        for (std::size_t c = 0; c < 3; ++c)
            if (weights[c] != 0.0)
                linear.weights.emplace_back(dofs[c], weights[c]);
        compressSparse(linear.weights);
    }

    // The sections or cuts of a response on the element set of the design point, made at the first item.
    [[nodiscard]] SectionRows& sectionRowsOf(const DesignResp1Response& response)
    {
        const DesignResp1Definition& definition = response.definition();
        auto [found, isNew] = m_sectionRows.try_emplace(definition.id);
        SectionRows& rows = found->second;
        if (!isNew)
            return rows;
        base::DiagnosticSink sink;
        if (definition.type == ResponseType::SectionLoad)
        {
            rows.sections =
                std::make_unique<output::SectionLoads>(definition.sections, *m_in.overlaid, m_set, m_dofMap, sink);
            for (std::size_t s = 0; s < rows.sections->sections().size(); ++s)
                rows.indexOf[definition.sections[rows.sections->sections()[s].definition].id] = s;
        }
        else
        {
            rows.cuts = std::make_unique<output::CutLoads>(definition.cuts, *m_in.overlaid, m_set, m_dofMap, sink);
            for (std::size_t c = 0; c < rows.cuts->cuts().size(); ++c)
                rows.indexOf[definition.cuts[rows.cuts->cuts()[c].definition].id] = c;
        }
        if (sink.hasErrors())
            throw base::OptimisationError(
                base::concatenate({"the sensitivity of response ", std::to_string(definition.id), " (",
                                   definition.label, "): its definitions do not hold at this design point: ",
                                   base::formatDiagnostic(sink.diagnostics().front())}));
        return rows;
    }

    // The loads of section or cut `index` (every component) at the displacements and temperatures of a subcase, the
    // elements computed by `set` (the design point's, or a perturbed one).
    [[nodiscard]] std::vector<double> sectionLoadsOf(const SectionRows& rows, std::size_t index,
                                                     const assembly::ElementSet& set, const SubcaseState& state) const
    {
        const elements::ElementTemperatures* temperatures =
            state.temperatures ? &state.temperatures->temperatures() : nullptr;
        if (rows.sections)
        {
            std::vector<double> values(rows.sections->componentCount(index));
            rows.sections->evaluateSection(index, set, state.displacements.span(), temperatures, values);
            return values;
        }
        const std::array<double, 6> loads =
            rows.cuts->evaluateCut(index, set, state.displacements.span(), temperatures);
        return {loads.begin(), loads.end()};
    }

    // A section or cut load item (output/sections): its weights the displacement row of its component (the loads are
    // affine in u: the row is ∂r/∂u exactly), its value checked against the analysis'.
    void lineariseSection(const ItemKey& key, const ResponseItem& label, const DesignResp1Response& response)
    {
        SectionRows& rows = sectionRowsOf(response);
        const auto found = rows.indexOf.find(label.entity);
        if (found == rows.indexOf.end())
            throw base::OptimisationError(
                base::concatenate({"the sensitivity of response ", std::to_string(response.definition().id),
                                   ": it has no ", rows.sections ? "section " : "cut ", std::to_string(label.entity),
                                   "; declare it in sections= or cuts= of the response"}));
        const std::size_t index = found->second;
        const auto component = static_cast<std::size_t>(label.component - 1);
        const SubcaseState& state = stateOf(label.subcase);
        const output::SparseRows& matrix = rows.sections ? rows.sections->displacementRows(m_budget, m_in.pool)
                                                         : rows.cuts->displacementRows(m_budget, m_in.pool);
        const std::size_t row =
            (rows.sections ? rows.sections->sections()[index].firstRow : rows.cuts->cuts()[index].firstRow) + component;
        ItemLinearisation& linear = m_linearised.at(key);
        linear.weights.clear();
        const std::span<const base::DofIndex> dofs = matrix.dofsOf(row);
        const std::span<const double> coefficients = matrix.coefficientsOf(row);
        for (std::size_t k = 0; k < dofs.size(); ++k)
            linear.weights.emplace_back(dofs[k], coefficients[k]);
        // The rounding of the section's recovery: the largest products of its rows (ValueCheck).
        // PERFORMANCE: the loads of a section in a subcase and the products of its rows, made once for its components
        // (each item recovered every member and summed every row of its section again: the section items of the wing
        // campaign's fused rows took 7.0 s of a linearisation of 9.7 s at 1/128 with 20 cases); the same values.
        // Measured: the wing model at 1/128 of its full size.
        // At a perturbed state no value is checked (checkValue): nothing recovered.
        const auto valueKey = std::make_tuple(response.definition().id, index, label.subcase);
        auto cached = m_sectionValues.find(valueKey);
        if (m_checksValues && cached == m_sectionValues.end())
        {
            SectionValues values;
            values.loads = sectionLoadsOf(rows, index, m_set, state);
            const std::size_t first = row - component;
            for (std::size_t c = 0; c < values.loads.size(); ++c)
            {
                const std::span<const base::DofIndex> rowDofs = matrix.dofsOf(first + c);
                const std::span<const double> rowCoefficients = matrix.coefficientsOf(first + c);
                double sum = 0.0;
                for (std::size_t k = 0; k < rowDofs.size(); ++k)
                    sum +=
                        std::abs(rowCoefficients[k] * state.displacements.span()[static_cast<std::size_t>(rowDofs[k])]);
                values.products = std::max(values.products, sum);
            }
            cached = m_sectionValues.emplace(valueKey, std::move(values)).first;
        }
        if (m_checksValues)
            checkValue(valuesAt(key.response), key.item, cached->second.loads[component], cached->second.products);
        m_sectionItems.push_back({key, response.definition().id, index, component, m_subcaseIndex.at(label.subcase)});
    }

    // The explicit parts of the section and cut items whose elements the quantity reaches: each section or cut
    // evaluated on the four perturbed element sets at fixed u (FieldPerturbation::derivative), once for all its
    // components. A shape moves the frames, the areas (the default weights) and the moment arms with the elements.
    // PERFORMANCE: the groups of section and cut items (a section or cut in a subcase) and the groups each element is
    // a member of, made once; a field visits the groups of the elements it reaches, in the groups' order, where every
    // field looked at every group's members (fields × sections: the linearisation of the sizing loop's section loads
    // grew as N^1.6 over the barrel's ladder, 15 s of the 1/16 rung). The same groups in the same order: the same bits.
    // Measured: the second order of the reserve factors on the barrel models of increasing size.
    using SectionGroup = std::tuple<std::int32_t, std::size_t, std::size_t>; // (response, section or cut, subcase)
    void indexSectionGroups()
    {
        if (m_hasSectionGroups)
            return;
        m_hasSectionGroups = true;
        for (const SectionItem& item : m_sectionItems)
            m_sectionGroups[{item.response, item.index, item.subcase}].push_back(&item);
        for (const auto& [group, items] : m_sectionGroups)
        {
            const auto& [response, index, subcase] = group;
            const SectionRows& rows = m_sectionRows.at(response);
            if (rows.sections)
                for (const output::SectionLoads::Member& member : rows.sections->sections()[index].members)
                    m_sectionGroupsOf[ElementRef{member.batch, member.row}].push_back(group);
            else
                for (const output::CutLoads::Element& element : rows.cuts->cuts()[index].elements)
                    m_sectionGroupsOf[ElementRef{element.batch, element.row}].push_back(group);
        }
    }

    // `designed` (null: a shape, or a field to difference): the field whose members' force rates the families may give
    // in closed form (SectionLoads::sectionFieldRates), a section's rates then their sum, and a cut's
    // the grid point forces of its elements' stiffness rates (CutLoads::cutFieldRates); a section or cut whose
    // reached elements do not all give them is differenced on the perturbed sides.
    void differenceSectionItems(const FieldValue& quantity, const LazyFieldPerturbation& perturbation,
                                const std::set<ElementRef>& affected, const elements::DesignedField* designed)
    {
        if (m_sectionItems.empty())
            return;
        indexSectionGroups();
        std::set<SectionGroup> reached;
        for (const ElementRef& element : affected)
            if (const auto found = m_sectionGroupsOf.find(element); found != m_sectionGroupsOf.end())
                reached.insert(found->second.begin(), found->second.end());
        const auto isReached = [&affected](std::size_t batch, std::int64_t row)
        { return affected.count(ElementRef{batch, row}) != 0; };
        for (const SectionGroup& group : reached)
        {
            const std::vector<const SectionItem*>& items = m_sectionGroups.at(group);
            const auto& [response, index, subcase] = group;
            const SectionRows& rows = m_sectionRows.at(response);
            if (designed != nullptr && rows.sections)
            {
                const SubcaseState& state = m_subcases[subcase];
                const elements::ElementTemperatures* temperatures =
                    state.temperatures ? &state.temperatures->temperatures() : nullptr;
                std::vector<double> rates(rows.sections->componentCount(index));
                if (rows.sections->sectionFieldRates(index, m_set, state.displacements.span(), temperatures, *designed,
                                                     isReached, rates))
                {
                    for (const SectionItem* item : items)
                        addExplicit(item->key, quantity, rates[item->component]);
                    m_report.analyticExplicitParts += static_cast<std::int64_t>(items.size());
                    continue;
                }
            }
            if (designed != nullptr && rows.cuts)
            {
                const SubcaseState& state = m_subcases[subcase];
                const elements::ElementTemperatures* temperatures =
                    state.temperatures ? &state.temperatures->temperatures() : nullptr;
                std::array<double, 6> rates{};
                if (rows.cuts->cutFieldRates(index, m_set, state.displacements.span(), temperatures, *designed,
                                             isReached, rates))
                {
                    for (const SectionItem* item : items)
                        addExplicit(item->key, quantity, rates[item->component]);
                    m_report.analyticExplicitParts += static_cast<std::int64_t>(items.size());
                    continue;
                }
            }
            std::array<std::vector<double>, FIELD_SIDE_COUNT> sides;
            for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
                sides[side] = sectionLoadsOf(rows, index, perturbation.get().set(static_cast<FieldSide>(side)),
                                             m_subcases[subcase]);
            for (const SectionItem* item : items)
            {
                const std::size_t c = item->component;
                addExplicit(item->key, quantity,
                            perturbation.get().derivative(sides[0][c], sides[1][c], sides[2][c], sides[3][c]));
            }
        }
    }

    // The zones of the problem on the design point's element set, made at the first zone item: their rows
    // are the weights of the contour and external items.
    [[nodiscard]] output::ZoneContour& zoneContour()
    {
        if (!m_zoneContour)
        {
            const ResponseParts& parts = m_in.responses->parts();
            base::DiagnosticSink sink;
            m_zoneContour = std::make_unique<output::ZoneContour>(parts.zones, parts.zoneSubcases, *m_in.overlaid,
                                                                  m_set, m_dofMap, m_budget, sink);
            if (sink.hasErrors())
                throw base::OptimisationError(
                    base::concatenate({"the sensitivities of the zones: they do not hold at this design point: ",
                                       base::formatDiagnostic(sink.diagnostics().front())}));
        }
        return *m_zoneContour;
    }

    [[nodiscard]] std::size_t zoneIndexOf(std::int32_t zone)
    {
        const std::optional<std::size_t> found = zoneContour().findZone(zone);
        if (!found)
            throw base::OptimisationError(
                base::concatenate({"the sensitivities of zone ", std::to_string(zone),
                                   ": the problem has no such zone; declare it with DesignProblem.add_zone"}));
        return *found;
    }

    // The row of a contour DOF (grid, component) of zone z.
    [[nodiscard]] std::size_t contourRowOf(std::size_t z, std::int32_t grid, std::int32_t component)
    {
        const std::vector<output::ZoneContour::ZoneDof>& dofs = zoneContour().zones()[z].contourDofs;
        for (std::size_t r = 0; r < dofs.size(); ++r)
            if (dofs[r].grid == grid && dofs[r].component == component)
                return r;
        throw base::OptimisationError(
            base::concatenate({"the sensitivities of the zones: grid point ", std::to_string(grid), " component ",
                               std::to_string(component),
                               " is not a contour DOF of its zone; name a grid point and component of the zone's "
                               "contour"}));
    }

    // The contour forces of zone z (contour DOFs, the element part with the thermal loads) at the displacements and
    // temperatures of a subcase, the elements computed by `set` (the design point's, or a perturbed one).
    [[nodiscard]] std::vector<double> zoneForcesOf(std::size_t z, const assembly::ElementSet& set,
                                                   const SubcaseState& state)
    {
        const elements::ElementTemperatures* temperatures =
            state.temperatures ? &state.temperatures->temperatures() : nullptr;
        return zoneContour().toContourDofs(
            z, zoneContour().elementForces(z, set, state.displacements.span(), temperatures));
    }

    // w = Mᵀ·c for the MPC forces q = M·(Kgg·u − P) (RigidElimination::computeMpcForces: q_m = r_m on the m set,
    // q_n = −Σ_i Gm(i, n)·r_m(i)), so that cᵀ·q = wᵀ·(Kgg·u − P): w[m(i)] = c[m(i)] − Σ_n Gm(i, n)·c[n], over the rows
    // whose dependent DOF or terms c reaches (an index of the rows by term DOF, made once per constraint set).
    [[nodiscard]] SparseVector mpcWeights(const SparseVector& c, const SubcaseState& state)
    {
        const assembly::RigidElimination& gm = systemOf(state).gm();
        auto [found, isNew] = m_gmTermIndex.try_emplace(state.group);
        std::unordered_map<base::DofIndex, std::vector<std::pair<std::size_t, double>>>& index = found->second;
        if (isNew)
            for (std::size_t i = 0; i < gm.rowCount(); ++i)
                for (const assembly::ConstraintTerm& term : gm.row(i))
                    index[term.dof].emplace_back(i, term.coefficient);
        SparseVector w;
        for (const auto& [dof, value] : c)
        {
            if (const std::optional<std::size_t> row = gm.findRow(dof))
                w.emplace_back(dof, value);
            if (const auto terms = index.find(dof); terms != index.end())
                for (const auto& [row, coefficient] : terms->second)
                    w.emplace_back(gm.mDofs()[row], -coefficient * value);
        }
        compressSparse(w);
        return w;
    }

    // The force rows of zone z when it is a condensed unit the analysis did not recover (UnitZoneForceRows), made once;
    // null for any other zone.
    [[nodiscard]] const UnitZoneForceRows* unitZoneForceRows(std::size_t z, const SubcaseState& state)
    {
        if (const auto found = m_unitZoneRows.find(z); found != m_unitZoneRows.end())
            return found->second ? &*found->second : nullptr;
        std::optional<UnitZoneForceRows>& entry = m_unitZoneRows[z];
        const BoundaryReducedSensitivity* condensed = systemOf(state).condensed();
        output::ZoneContour& contour = zoneContour();
        const std::int64_t unitId = contour.definitions()[contour.zones()[z].definition].unit;
        if (condensed == nullptr || unitId == 0)
            return nullptr;
        const solutions::KeptCondensedUnit* unit = nullptr;
        for (const solutions::KeptCondensedUnit& candidate : condensed->set().units())
            if (candidate.unitId == unitId && !candidate.isRecovered)
                unit = &candidate;
        if (unit == nullptr)
            return nullptr;
        const linalg::DenseMatrix& stiffness = unit->condensation->reducedUnit().stiffness();
        const reduction::UnitSystem& system = *unit->system;
        const std::size_t boundary = stiffness.rows();
        std::vector<base::DofIndex> dofOfRow(boundary, -1); // bounded by the boundary
        for (std::size_t i = 0; i < boundary; ++i)
            dofOfRow[i] = unit->modelDofOfLocal[static_cast<std::size_t>(
                system.localDofOfEquation(system.interiorDofCount() + static_cast<std::int32_t>(i)))];
        const std::vector<output::ZoneContour::ZonePoint>& points = contour.zones()[z].contour;
        const std::size_t contourDofs = contour.zones()[z].contourDofs.size();
        UnitZoneForceRows built;
        std::vector<std::size_t> columns; // the boundary rows of K* that are DOFs of the model
        for (std::size_t b = 0; b < boundary; ++b)
            if (dofOfRow[b] >= 0)
            {
                built.dofs.push_back(dofOfRow[b]);
                columns.push_back(b);
            }
        built.rows.assign(contourDofs * built.dofs.size(), 0.0);
        constexpr std::size_t COMPONENTS = 6;
        std::vector<double> perPoint(COMPONENTS * points.size());
        for (std::size_t c = 0; c < columns.size(); ++c)
        {
            // Column b of K*: the forces at the boundary DOFs of a unit motion of DOF b, per contour point in CD.
            std::fill(perPoint.begin(), perPoint.end(), 0.0);
            for (std::size_t i = 0; i < boundary; ++i)
            {
                if (dofOfRow[i] < 0)
                    continue;
                const linalg::NodeDof nodeDof = m_dofMap.nodeDofOf(dofOfRow[i]);
                for (std::size_t p = 0; p < points.size(); ++p)
                    if (points[p].gridRow == nodeDof.node)
                    {
                        perPoint[COMPONENTS * p + static_cast<std::size_t>(nodeDof.component)] +=
                            stiffness(i, columns[c]);
                        break;
                    }
            }
            const std::vector<double> column = contour.toContourDofs(z, perPoint);
            for (std::size_t r = 0; r < contourDofs; ++r)
                built.rows[r * built.dofs.size() + c] = column[r];
        }
        entry = std::move(built);
        return &*entry;
    }

    // The weights and the parts of a zone item from its terms (ZoneItem), registered for its explicit parts.
    void lineariseZoneItem(ZoneItem item)
    {
        output::ZoneContour& contour = zoneContour();
        SparseVector weights;
        SparseVector constraintWeights; // over the MPC forces
        for (const ZoneTerm& term : item.terms)
        {
            const output::ZoneContour::ZoneRows& rows = contour.rows(term.zone, m_budget);
            if (term.quantity == ExternalArgumentKind::ContourForce)
                if (const UnitZoneForceRows* unitRows = unitZoneForceRows(term.zone, m_subcases[item.subcase]))
                {
                    const std::size_t columns = unitRows->dofs.size();
                    for (const auto& [r, weight] : term.weights)
                        for (std::size_t c = 0; c < columns; ++c)
                            if (const double value = unitRows->rows[r * columns + c]; value != 0.0)
                                weights.emplace_back(unitRows->dofs[c], weight * value);
                    continue;
                }
            const auto add = [&](const output::SparseRows& matrix, SparseVector& target, double sign)
            {
                for (const auto& [r, weight] : term.weights)
                {
                    const std::span<const base::DofIndex> dofs = matrix.dofsOf(r);
                    const std::span<const double> coefficients = matrix.coefficientsOf(r);
                    for (std::size_t k = 0; k < dofs.size(); ++k)
                        target.emplace_back(dofs[k], sign * weight * coefficients[k]);
                }
            };
            switch (term.quantity)
            {
            case ExternalArgumentKind::ContourDisplacement:
                add(rows.displacements, weights, 1.0);
                break;
            case ExternalArgumentKind::ContourForce:
                add(rows.forces, weights, 1.0);
                add(rows.forceConstraints, constraintWeights, 1.0);
                break;
            case ExternalArgumentKind::ZoneLoads:
                add(rows.loads, item.loadWeights, 1.0);
                break;
            case ExternalArgumentKind::DesignVariable:
                break;
            }
        }
        compressSparse(weights);
        compressSparse(constraintWeights);
        compressSparse(item.loadWeights);
        if (!constraintWeights.empty())
        {
            // The MPC force part: r = cᵀ·q = wᵀ·(Kgg·u − P), weights Kgg·w (as an SPC force).
            const SubcaseState& state = m_subcases[item.subcase];
            item.rigid = mpcWeights(constraintWeights, state);
            base::BudgetedBuffer<double> dense(m_budget, gSize(), "zone item MPC force weights");
            std::fill(dense.data(), dense.data() + gSize(), 0.0);
            for (const auto& [index, value] : item.rigid)
                dense[static_cast<std::size_t>(index)] = value;
            base::BudgetedBuffer<double> product(m_budget, gSize(), "zone item MPC force weights K·w");
            systemOf(state).kgg().multiply(dense.span(), product.span());
            for (std::size_t g = 0; g < gSize(); ++g)
                if (product[g] != 0.0)
                    weights.emplace_back(static_cast<base::DofIndex>(g), product[g]);
            compressSparse(weights);
        }
        m_linearised.at(item.key).weights = std::move(weights);
        m_zoneItems.push_back(std::move(item));
    }

    // A CONTDISP or CONTFORC item: the row of its contour DOF (exact: the contour data are affine in u), its value
    // checked against the analysis' when it has no MPC force part.
    void lineariseContour(const ItemKey& key, const ResponseItem& label, ResponseType type)
    {
        const std::size_t z = zoneIndexOf(label.entity);
        const std::size_t r = contourRowOf(z, label.layer, label.component);
        const bool isForce = type == ResponseType::ContourForce;
        ZoneItem item;
        item.key = key;
        item.subcase = m_subcaseIndex.at(label.subcase);
        item.terms.push_back(
            {z, isForce ? ExternalArgumentKind::ContourForce : ExternalArgumentKind::ContourDisplacement, {{r, 1.0}}});
        const SubcaseState& state = m_subcases[item.subcase];
        if (!isForce)
        {
            auto [cached, isNew] = m_zoneDisplacements.try_emplace({z, item.subcase});
            if (isNew)
            {
                cached->second.resize(zoneContour().rows(z, m_budget).displacements.rowCount());
                zoneContour().builtRows(z).displacements.apply(state.displacements.span(), cached->second);
            }
            checkValue(valuesAt(key.response), key.item, cached->second[r]);
        }
        else if (!zoneContour().zones()[z].contour.empty() && unitZoneForceRows(z, state) == nullptr &&
                 std::none_of(zoneContour().zones()[z].contour.begin(), zoneContour().zones()[z].contour.end(),
                              [](const output::ZoneContour::ZonePoint& point) { return point.takesConstraintForces; }))
        {
            const ResponseValues& values = valuesAt(key.response);
            auto [cached, isNew] = m_zoneForces.try_emplace({z, item.subcase});
            if (isNew)
                cached->second = zoneForcesOf(z, m_set, state);
            const double recomputed = cached->second[r];
            if (std::abs(recomputed - values.values[key.item]) >
                CONTOUR_VALUE_CHECK_TOLERANCE * largestMagnitude(values.values))
                checkValue(values, key.item, recomputed); // reports it
        }
        lineariseZoneItem(std::move(item));
    }

    // The external responses: their items evaluated again with the partials of the plugin (or of the
    // supplied values) with respect to the zone data of their case and their variables; the zone data chained with the
    // rows of the zone (ZoneItem), the variables taken in the explicit part. An external item is a plain item: one
    // adjoint solve serves it whatever the size of its contour.
    void lineariseExternals()
    {
        const std::vector<ExternalResponse>& externals = m_in.responses->externals();
        if (externals.empty())
            return;
        KnownResponses known;
        for (const ResponseValues& values : *m_in.values)
            known[values.id] = &values;
        ResponseContext context;
        context.snapshot = m_in.overlaid.get();
        context.snapshotOwner = m_in.overlaid;
        context.model = m_in.model;
        context.design = m_in.design;
        context.massFormulation = m_in.massFormulation;
        context.budget = &m_budget;
        context.zones = m_in.results->zones;
        for (const ExternalResponse& response : externals)
        {
            const std::int32_t id = response.definition().id;
            const auto position = m_positionOf.find(id);
            if (m_needed.count(id) == 0 || position == m_positionOf.end())
                continue;
            const std::size_t p = position->second;
            if (m_linearised.items[p].empty())
                continue;
            bool isAnyWanted = false;
            for (const ItemLinearisation& item : m_linearised.items[p])
                isAnyWanted = isAnyWanted || item.isWanted;
            if (!isAnyWanted)
                continue;
            if (m_in.externals == nullptr)
                throw base::OptimisationError(
                    base::concatenate({"the sensitivity of external response ", std::to_string(id),
                                       ": expected its evaluator (the plugin or the supplied values), found none; "
                                       "register the plugin or supply the values before the gradients"}));
            const ResponseValues& values = valuesAt(p);
            const ExternalValues evaluated = response.evaluate(m_in.responseSubcases->at(id), known, context,
                                                               *m_in.externals, m_in.blockItems, true);
            if (evaluated.values.itemCount() != values.itemCount())
                throw base::OptimisationError(base::concatenate(
                    {"the sensitivity of external response ", std::to_string(id),
                     ": its items evaluated again with their partials are not those of the evaluation (a selection "
                     "that changed between the two calls?); keep select= the same for the evaluation and its "
                     "gradients"}));
            for (std::size_t i = 0; i < values.itemCount(); ++i)
            {
                const ItemKey key{p, i};
                ItemLinearisation& linear = m_linearised.at(key);
                if (!linear.isWanted || !evaluated.values.isOk(i))
                {
                    linear.isWanted = false;
                    continue;
                }
                checkValue(values, i, evaluated.values.values[i]);
                const ResponseItem& label = values.items[i];
                const std::size_t z = zoneIndexOf(label.entity);
                ZoneItem item;
                item.key = key;
                item.subcase = m_subcaseIndex.at(label.subcase);
                std::map<ExternalArgumentKind, std::size_t> termOf;
                for (const ExternalTerm& term : evaluated.partials[i])
                {
                    const ExternalArgument& argument = response.definition().arguments[term.argument];
                    if (argument.kind == ExternalArgumentKind::DesignVariable)
                    {
                        const std::size_t variable = response.variableOf(term.argument, label.entity);
                        for (const auto& [column, derivative] : m_in.design->variables.derivatives[variable])
                            linear.explicitPart.emplace_back(column, term.weight * derivative);
                        continue;
                    }
                    auto [found, isNew] = termOf.try_emplace(argument.kind, item.terms.size());
                    if (isNew)
                        item.terms.push_back({z, argument.kind, {}});
                    item.terms[found->second].weights.emplace_back(term.component, term.weight);
                }
                lineariseZoneItem(std::move(item));
            }
        }
    }

    // The explicit parts of the zone items: the contour forces of each zone a quantity reaches, evaluated
    // again on the four perturbed element sets at fixed u (FieldPerturbation's differences); the MPC force parts
    // −wᵀ·q (q the pseudo-load); the loads parts ∂P/∂v (the applied load part of the pseudo-loads).
    void differenceZoneItems(std::size_t q, const FieldValue& quantity, const LazyFieldPerturbation& perturbation,
                             const std::set<ElementRef>& affected)
    {
        if (m_zoneItems.empty())
            return;
        std::map<std::pair<std::size_t, std::size_t>, std::vector<double>> derivatives; // (zone, subcase) → dF/dv
        for (const ZoneItem& item : m_zoneItems)
        {
            double derivative = 0.0;
            for (const ZoneTerm& term : item.terms)
            {
                if (term.quantity != ExternalArgumentKind::ContourForce)
                    continue;
                const output::ZoneContour::Zone& zone = zoneContour().zones()[term.zone];
                const bool isReached =
                    std::any_of(zone.members.begin(), zone.members.end(), [&](const output::ZoneContour::Member& member)
                                { return affected.count(ElementRef{member.batch, member.row}) != 0; });
                if (!isReached)
                    continue;
                auto [found, isNew] = derivatives.try_emplace({term.zone, item.subcase});
                if (isNew)
                {
                    std::array<std::vector<double>, FIELD_SIDE_COUNT> sides;
                    for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
                        sides[side] = zoneForcesOf(term.zone, perturbation.get().set(static_cast<FieldSide>(side)),
                                                   m_subcases[item.subcase]);
                    found->second.resize(sides[0].size());
                    for (std::size_t r = 0; r < sides[0].size(); ++r)
                        found->second[r] =
                            perturbation.get().derivative(sides[0][r], sides[1][r], sides[2][r], sides[3][r]);
                }
                for (const auto& [r, weight] : term.weights)
                    derivative += weight * found->second[r];
            }
            if (!item.rigid.empty())
                derivative -= dotSparseVectors(item.rigid, m_pseudoLoads->of(q, item.subcase));
            if (!item.loadWeights.empty())
                derivative += dotSparseVectors(item.loadWeights, m_pseudoLoads->appliedLoadPartOf(q, item.subcase));
            addExplicit(item.key, quantity, derivative);
        }
    }

    // The linearity indicator of the design point's snapshot, built on first use.
    [[nodiscard]] const LinearityIndicator& linearity()
    {
        if (!m_linearity)
            m_linearity = std::make_unique<LinearityIndicator>(*m_in.overlaid, m_in.overlaid, m_budget);
        return *m_linearity;
    }

    void prepareTotalEnergy(SubcaseState& state)
    {
        if (state.elementForces)
            return;
        // The total strain energy reads every displacement: on the condensed path, every unit recovered.
        if (const BoundaryReducedSensitivity* condensed = systemOf(state).condensed();
            condensed != nullptr && !condensed->isEveryDofKnown())
            throw base::OptimisationError(base::concatenate(
                {"the sensitivities of subcase ", std::to_string(state.subcase.label),
                 ": a compliance, a mean displacement or an ESE percentage reads the strain energy of the whole model, "
                 "and the analysis on the condensed path did not recover every unit; recover every unit (a plan with "
                 "recover=\"all\") or use the full path"}));
        state.elementForces.emplace(m_budget, gSize(), "sensitivity element forces Kgg u");
        systemOf(state).kgg().multiply(state.displacements.span(), state.elementForces->span());
        double sum = 0.0;
        for (std::size_t g = 0; g < gSize(); ++g)
            sum += state.displacements[g] * (*state.elementForces)[g];
        state.totalEnergy = 0.5 * sum;
    }

    [[nodiscard]] bool isPercent(const ElementJob& job, std::size_t k) const
    {
        const ElementItemSpec& spec = job.function->items()[k];
        return spec.type == ResponseType::StrainEnergy && spec.code == STRAIN_ENERGY_PERCENT_ITEM;
    }

    // The values of a job's items, the percentages of ESE item 3 formed from the energies.
    [[nodiscard]] std::vector<double> jobValues(const ElementJob& job, const elements::ElementBatchView& view,
                                                const std::shared_ptr<const input::ModelSnapshot>& snapshot,
                                                const elements::NodalDisplacements& displacements)
    {
        return job.function->values(view, snapshot, displacements, m_budget);
    }

    // The element jobs of each subcase, in parallel (the derivatives of the element items with respect to
    // the displacements were the largest cost of the products on the larger models, some 2 ms per element and
    // subcase on one thread). A job writes only its own items and reads the displacements of its subcase perturbed
    // in a copy of its piece's own, so the jobs have the bits of a serial run whatever the threads.
    void lineariseElementJobs()
    {
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            std::vector<std::size_t> jobs; // bounded by the jobs
            bool hasPercent = false;
            for (std::size_t j = 0; j < m_jobs.size(); ++j)
                if (m_jobs[j].subcase == s)
                {
                    jobs.push_back(j);
                    for (std::size_t k = 0; k < m_jobs[j].keys.size(); ++k)
                        hasPercent = hasPercent || isPercent(m_jobs[j], k);
                }
            if (jobs.empty())
                continue;
            SubcaseState& state = m_subcases[s];
            if (hasPercent)
                prepareTotalEnergy(state);
            std::atomic<std::int64_t> differences{0};
            std::atomic<std::int64_t> familySlopes{0};
            std::atomic<std::int64_t> unitRecoveries{0};
            base::parallelFor(m_in.pool, jobs.size(), ELEMENT_JOBS_PER_PIECE,
                              [&](std::size_t begin, std::size_t end)
                              {
                                  // PERFORMANCE: the piece's copy of the displacements is made at its first job that
                                  // moves them (the differences, the recoveries at unit displacements); a job a family
                                  // gives the slopes of reads the subcase's own, and the copy (a g-set vector: 3 MB at
                                  // 400 000 DOFs) cost more than its jobs.
                                  // Measured: the element derivatives of a design point at 400 000 DOFs.
                                  std::optional<base::BudgetedBuffer<double>> perturbed;
                                  const auto perturbedOf = [&]
                                  {
                                      if (!perturbed)
                                      {
                                          perturbed.emplace(m_budget, gSize(), "sensitivity perturbed displacements");
                                          std::copy(state.displacements.data(), state.displacements.data() + gSize(),
                                                    perturbed->data());
                                      }
                                      return perturbed->span();
                                  };
                                  JobCounts counts;
                                  for (std::size_t k = begin; k < end; ++k)
                                      lineariseElementJob(m_jobs[jobs[k]], state, perturbedOf, counts);
                                  differences += counts.differences;
                                  familySlopes += counts.familySlopes;
                                  unitRecoveries += counts.unitRecoveries;
                              });
            m_report.displacementDifferences += differences.load();
            m_report.recoverySlopes += familySlopes.load();
            m_report.unitRecoveries += unitRecoveries.load();
        }
    }

    // What the element jobs of one piece did (lineariseElementJob).
    struct JobCounts
    {
        std::int64_t differences = 0;    // the recoveries differenced
        std::int64_t familySlopes = 0;   // the outputs whose slopes the families gave
        std::int64_t unitRecoveries = 0; // the recoveries at unit displacements
    };

    // The values and the weights ∂r/∂u_e of one element job: from the slopes of its recoveries
    // (ElementItemFunction::slopes) when every item has them and the analytic derivatives are asked for, else by
    // differences. `perturbedOf()` gives a copy of the subcase's displacements the job may move (restored on return).
    template <typename PerturbedOf>
    void lineariseElementJob(ElementJob& job, const SubcaseState& state, const PerturbedOf& perturbedOf,
                             JobCounts& counts)
    {
        const elements::ElementBatchView& view = m_set.batches()[job.element.batch].view;
        const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap.gridCount());
        std::vector<SparseVector> weights(job.keys.size());
        // PERFORMANCE: the slopes of a recovery are its matrix, made once with the element's formulation (a ShellQuad4's
        // plate built once for its 24 DOFs), where the differences recovered the element two to eight times per DOF;
        // exact, so the items linear in u have their weights to round-off and a von Mises its chain rule.
        // Measured: the element derivatives of a design point, slopes against differences.
        if (m_options.elementDerivatives == ElementDerivativeMethod::Analytic)
            if (std::optional<ItemSlopes> slopes = job.function->slopes(
                    view, m_in.overlaid, state.nodal(m_dofMap),
                    [&]
                    {
                        const std::span<double> copy = perturbedOf();
                        return ScratchDisplacements{copy.first(gridValues), copy.subspan(gridValues)};
                    },
                    m_budget))
            {
                counts.familySlopes += slopes->familySlopes;
                counts.unitRecoveries += slopes->unitRecoveries;
                job.baseValues = std::move(slopes->values);
                const std::size_t n = slopes->dofs.size();
                std::vector<base::DofIndex> indices(n); // bounded by the element's DOFs
                for (std::size_t d = 0; d < n; ++d)
                    indices[d] = m_dofMap.dofOf(slopes->dofs[d]);
                for (std::size_t k = 0; k < job.keys.size(); ++k)
                    for (std::size_t d = 0; d < n; ++d)
                        if (const double weight = slopes->weights[k * n + d]; weight != 0.0)
                            weights[k].emplace_back(indices[d], weight);
                finishElementJob(job, state, view, state.nodal(m_dofMap), weights);
                return;
            }
        const std::span<double> perturbed = perturbedOf();
        const elements::NodalDisplacements displacements{
            perturbed.first(gridValues), perturbed.subspan(gridValues),
            state.temperatures ? &state.temperatures->temperatures() : nullptr};
        std::vector<ItemForm> forms; // bounded by the job's items
        job.baseValues = job.function->values(view, m_in.overlaid, displacements, m_budget, &forms);
        // ∂r/∂u_e by central differences on each DOF of the element, the others fixed.
        const std::vector<elements::ElementDof> dofs = job.function->dofs(view);
        std::array<double, 3> largest{}; // translations, rotations, scalar points
        const auto kindOf = [](const elements::ElementDof& dof)
        { return dof.kind == input::PointKind::Scalar ? 2 : (dof.component <= 3 ? 0 : 1); };
        for (const elements::ElementDof& dof : dofs)
            largest[static_cast<std::size_t>(kindOf(dof))] =
                std::max(largest[static_cast<std::size_t>(kindOf(dof))],
                         std::abs(perturbed[static_cast<std::size_t>(m_dofMap.dofOf(dof))]));
        // PERFORMANCE: a job whose items are affine in u (the components of the shells' and rods' stresses and strains)
        // or shells' von Mises (an optimiser's stress constraints) differences the values and the in-plane components
        // once per DOF: exact for an affine function whatever the step, and a von Mises item chained through its
        // function (planeVonMisesRate), instead of the coarse, fine and, for an item not linear in u, small
        // differences of every item: 2 recoveries per DOF instead of 4 or 8 (ItemForm).
        // Measured: on one thread, the element items of the 444-shell caisson 1.69 s
        // to 0.42 s, of the 2 162-bar truss 0.59 s to 0.30 s.
        const bool isFormedJob = !forms.empty() && std::all_of(forms.begin(), forms.end(), [](const ItemForm& form)
                                                               { return form.kind != ItemForm::Kind::Other; });
        if (isFormedJob)
        {
            std::vector<ItemForm> plusForms;  // bounded by the job's items
            std::vector<ItemForm> minusForms; // bounded by the job's items
            for (const elements::ElementDof& dof : dofs)
            {
                const auto g = static_cast<std::size_t>(m_dofMap.dofOf(dof));
                const double scale = largest[static_cast<std::size_t>(kindOf(dof))];
                const double original = perturbed[g];
                const double step = scale > 0.0 ? LINEAR_STEP_RELATIVE * scale : DISPLACEMENT_STEP_AT_REST;
                perturbed[g] = original + step;
                const double up = perturbed[g];
                const std::vector<double> plus =
                    job.function->values(view, m_in.overlaid, displacements, m_budget, &plusForms);
                perturbed[g] = original - step;
                const double down = perturbed[g];
                const std::vector<double> minus =
                    job.function->values(view, m_in.overlaid, displacements, m_budget, &minusForms);
                perturbed[g] = original;
                counts.differences += 2;
                for (std::size_t k = 0; k < job.keys.size(); ++k)
                {
                    double derivative = (plus[k] - minus[k]) / (up - down);
                    if (forms[k].kind == ItemForm::Kind::PlaneVonMises)
                    {
                        std::array<double, 3> rates{};
                        for (std::size_t c = 0; c < rates.size(); ++c)
                            rates[c] = (plusForms[k].components[c] - minusForms[k].components[c]) / (up - down);
                        derivative = planeVonMisesRate(forms[k], rates);
                    }
                    if (derivative != 0.0)
                        weights[k].emplace_back(static_cast<base::DofIndex>(g), derivative);
                }
            }
        }
        // The other jobs: the coarse and fine differences of each item, and two smaller ones for an item not linear in
        // u.
        const std::span<const elements::ElementDof> differenced =
            isFormedJob ? std::span<const elements::ElementDof>{} : std::span<const elements::ElementDof>(dofs);
        for (const elements::ElementDof& dof : differenced)
        {
            const auto g = static_cast<std::size_t>(m_dofMap.dofOf(dof));
            const double scale = largest[static_cast<std::size_t>(kindOf(dof))];
            const double original = perturbed[g];
            // The central difference at a step: the values at u_k ± step over the distance actually taken.
            const auto difference = [&](double step)
            {
                perturbed[g] = original + step;
                const double up = perturbed[g];
                const std::vector<double> plus = jobValues(job, view, m_in.overlaid, displacements);
                perturbed[g] = original - step;
                const double down = perturbed[g];
                std::vector<double> minus = jobValues(job, view, m_in.overlaid, displacements);
                for (std::size_t k = 0; k < minus.size(); ++k)
                    minus[k] = (plus[k] - minus[k]) / (up - down);
                counts.differences += 2;
                return minus;
            };
            const double large = scale > 0.0 ? LINEAR_STEP_RELATIVE * scale : DISPLACEMENT_STEP_AT_REST;
            const std::vector<double> coarse = difference(large);
            const std::vector<double> fine = difference(large / 2.0);
            std::optional<std::pair<std::vector<double>, std::vector<double>>> small;
            for (std::size_t k = 0; k < job.keys.size(); ++k)
            {
                double derivative = (4.0 * fine[k] - coarse[k]) / 3.0;
                const bool isLinear = std::abs(coarse[k] - fine[k]) <= LINEARITY_TOLERANCE * std::abs(fine[k]);
                if (!isLinear && scale > 0.0)
                {
                    if (!small)
                        small.emplace(difference(NONLINEAR_STEP_RELATIVE * scale),
                                      difference(NONLINEAR_STEP_RELATIVE * scale / 2.0));
                    derivative = (4.0 * small->second[k] - small->first[k]) / 3.0;
                }
                if (derivative != 0.0)
                    weights[k].emplace_back(static_cast<base::DofIndex>(g), derivative);
            }
            perturbed[g] = original;
        }
        finishElementJob(job, state, view, displacements, weights);
    }

    // The weights of a job's items (job.baseValues their values): each value checked against the analysis' with the
    // rounding of its terms, the percentages of ESE item 3 formed, the weights stored.
    void finishElementJob(ElementJob& job, const SubcaseState& state, const elements::ElementBatchView& view,
                          const elements::NodalDisplacements& displacements, std::vector<SparseVector>& weights)
    {
        std::optional<double> forceProducts; // of the element, once for its force items
        for (std::size_t k = 0; k < job.keys.size(); ++k)
        {
            compressSparse(weights[k]);
            // The value checked against the analysis' with the rounding of its terms (the weights just differenced):
            // an item that cancels to rounding where the element moves far as a rigid body (ValueCheck).
            const bool isPercentItem = isPercent(job, k);
            const double percentScale =
                isPercentItem ? (state.totalEnergy != 0.0 ? PERCENT / state.totalEnergy : 0.0) : 1.0;
            double products = termMagnitude(weights[k], state.displacements.span(), percentScale);
            if (job.function->items()[k].type == ResponseType::Force)
            {
                if (!forceProducts)
                    forceProducts = job.function->forceProducts(view, displacements);
                products = std::max(products, *forceProducts);
            }
            checkValue(valuesAt(job.keys[k].response), job.keys[k].item, percentScale * job.baseValues[k], products);
            if (isPercentItem && state.totalEnergy != 0.0)
            {
                // ESE item 3, p = 100·E/T, T = ½·uᵀ·K·u: ∂p/∂u = (100/T)·∂E/∂u − (100·E/T²)·K·u.
                const double energy = job.baseValues[k];
                const double total = state.totalEnergy;
                SparseVector scaled;
                for (const auto& [index, value] : weights[k])
                    scaled.emplace_back(index, PERCENT / total * value);
                SparseVector forces;
                for (std::size_t g = 0; g < gSize(); ++g)
                    if ((*state.elementForces)[g] != 0.0)
                        forces.emplace_back(static_cast<base::DofIndex>(g), (*state.elementForces)[g]);
                weights[k] = addScaledSparse(scaled, -PERCENT * energy / (total * total), forces);
            }
            m_linearised.at(job.keys[k]).weights = std::move(weights[k]);
        }
    }

    // The pseudo-loads of the external units, b_j = ∂f*/∂x_j − ∂K*/∂x_j·u_a, on the contour DOFs of each unit, for
    // each variable its plugin gives derivatives for: ∂K*/∂x_j as the plugin gives it (symmetrised,
    // ExternalUnitAssembly), ∂f*/∂x_j from its load source on the zone loads of the load basis of each constraint set
    // (the basis the analysis gave it), combined with the coefficients α of each subcase; and the inertial loads of
    // ∂M*/∂x_j in a subcase with inertial loads (the loads of the subcase's GravityLoad, ACCEL, RotationalForceLoad
    // with ∂M*/∂x_j as the mass: they are linear in it). A variable may drive no field of the model (a fillet radius of
    // the local model): its pseudo-load is then the unit's alone. The SPC force items get the explicit part −wᵀ·b_j (r
    // = wᵀ·(K·u − P), K holding K*, P holding f* and −M*·a).
    void addExternalUnitLoads()
    {
        if (!hasExternalUnits())
            return;
        bool hasLoads = false;
        for (const solutions::ExternalUnitRecord& record : m_in.results->externalUnits)
        {
            const reduction::ExternalUnitAssembly& placement = *record.placement;
            const reduction::ExternalUnit& unit = placement.unit();
            const std::span<const reduction::ExternalUnitDerivative> derivatives = unit.derivatives();
            if (derivatives.empty())
                continue;
            if (!m_selfAdjointItems.empty())
                throw base::OptimisationError(base::concatenate(
                    {"the gradients of a compliance or mean displacement response with the external unit of ",
                     placement.zone().displayName(),
                     " are not given in this version (its condensed loads move with the variables); remove the "
                     "response or the unit's derivatives"}));
            for (const reduction::ExternalUnitDerivative& derivative : derivatives)
                if (derivative.variable >= m_variableCount)
                    throw base::OptimisationError(base::concatenate(
                        {"the external unit of ", placement.zone().displayName(),
                         " gives derivatives with respect to "
                         "variable column ",
                         std::to_string(derivative.variable), ", and the problem has ", std::to_string(m_variableCount),
                         " independent variables; make the unit's derivatives name the columns of "
                         "the problem's independent variables"}));
            const std::size_t n = placement.contourSize();
            const std::span<const std::size_t> unitRow = placement.unitRowOfContour();
            // ∂f*/∂x_j of the basis vectors of every constraint set, in the contour order.
            std::vector<std::vector<linalg::DenseMatrix>> rates; // [basis][derivative]: contour × k
            for (const solutions::ExternalUnitBasis& basis : record.bases)
            {
                const std::size_t k = basis.interiorLoads.columns();
                std::vector<linalg::DenseMatrix> ofUnit; // bounded by the derivatives
                ofUnit.reserve(derivatives.size());
                std::vector<linalg::DenseMatrixView> views;
                for (std::size_t d = 0; d < derivatives.size(); ++d)
                {
                    ofUnit.emplace_back(m_budget, unit.size(), k, "external unit df*/dx");
                    views.push_back(ofUnit.back().view());
                }
                bool isLoaded = false;
                for (std::size_t v = 0; v < k && !isLoaded; ++v)
                    for (std::size_t i = 0; i < basis.interiorLoads.rows() && !isLoaded; ++i)
                        isLoaded = basis.interiorLoads(i, v) != 0.0;
                if (isLoaded && unit.loads() != nullptr)
                    unit.loads()->condensedLoadDerivatives(unit, basis.interiorLoads.view(), views);
                std::vector<linalg::DenseMatrix> ofContour; // bounded by the derivatives
                for (std::size_t d = 0; d < derivatives.size(); ++d)
                {
                    ofContour.emplace_back(m_budget, n, k, "external unit df*/dx on the contour");
                    for (std::size_t v = 0; v < k; ++v)
                        for (std::size_t a = 0; a < n; ++a)
                            ofContour.back()(a, v) = ofUnit[d](unitRow[a], v);
                }
                rates.push_back(std::move(ofContour));
            }
            std::vector<double> contourDisplacements(n);
            std::vector<double> stiffnessRate(n);
            std::vector<double> load(n);
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                const std::int32_t label = m_subcases[s].subcase.label;
                std::optional<std::pair<std::size_t, std::size_t>> column; // (basis, case)
                for (std::size_t b = 0; b < record.bases.size() && !column; ++b)
                    for (std::size_t j = 0; j < record.bases[b].cases.size(); ++j)
                        if (record.bases[b].cases[j] == label)
                        {
                            column = std::pair{b, j};
                            break;
                        }
                placement.gatherContour(m_subcases[s].displacements.span(), contourDisplacements);
                for (std::size_t d = 0; d < derivatives.size(); ++d)
                {
                    reduction::ExternalUnitAssembly::multiply(placement.stiffnessDerivative(d), contourDisplacements,
                                                              stiffnessRate);
                    for (std::size_t a = 0; a < n; ++a)
                    {
                        double rate = 0.0;
                        if (column)
                        {
                            const solutions::ExternalUnitBasis& basis = record.bases[column->first];
                            const linalg::DenseMatrix& basisRate = rates[column->first][d];
                            for (std::size_t v = 0; v < basisRate.columns(); ++v)
                                rate += basisRate(a, v) * basis.coefficients(v, column->second);
                        }
                        load[a] = rate - stiffnessRate[a];
                    }
                    if (const linalg::DenseMatrix* massRate = placement.massDerivative(d);
                        massRate != nullptr && m_subcases[s].hasInertialLoads && m_subcases[s].subcase.load)
                    {
                        // The inertial loads are linear in the mass: those of ∂M*/∂x_j are ∂P/∂x_j.
                        base::DiagnosticSink ignored; // the analysis reported the problems of its loads
                        linalg::BlockSparseMatrix rate(systemOf(m_subcases[s]).pattern(), m_budget,
                                                       "external unit dM*/dx");
                        placement.addMatrixTo(*massRate, rate, m_dofMap);
                        base::BudgetedBuffer<double> inertial(m_budget, gSize(), "external unit inertial load rate");
                        std::fill(inertial.data(), inertial.data() + gSize(), 0.0);
                        m_loads->addFamilyLoads(*m_subcases[s].subcase.load, assembly::LoadFamily::Inertial, &rate,
                                                inertial.span(), m_budget, ignored);
                        for (std::size_t a = 0; a < n; ++a)
                            load[a] += inertial[static_cast<std::size_t>(placement.zone().contourDofs[a])];
                    }
                    m_pseudoLoads->addVariableLoad(derivatives[d].variable, s, placement.zone().contourDofs, load);
                    hasLoads = true;
                }
            }
        }
        m_pseudoLoads->finishVariableLoads();
        if (!hasLoads)
            return;
        for (const SpcItem& item : m_spcItems)
            for (std::size_t j = 0; j < m_variableCount; ++j)
            {
                const SparseVector& load = m_pseudoLoads->variableLoadOf(j, item.subcase);
                double product = 0.0;
                std::size_t a = 0;
                for (const auto& [dof, weight] : item.w)
                {
                    while (a < load.size() && load[a].first < dof)
                        ++a;
                    if (a < load.size() && load[a].first == dof)
                        product += weight * load[a].second;
                }
                if (product != 0.0)
                    m_linearised.at(item.key).explicitPart.emplace_back(j, -product);
            }
    }

    void differenceFields()
    {
        const std::vector<FieldValue>& fields = m_in.design->fields;
        const std::vector<FieldValue>& shapes = m_in.design->shapes;
        if (fields.empty() && shapes.empty())
            return;
        for (MassItem& item : m_massItems)
            if (item.response->definition().type == ResponseType::FractionalMass)
                item.currentMass = MassProperties::rigidBodyMass(*m_in.overlaid, {0.0, 0.0, 0.0}, m_in.massFormulation,
                                                                 item.response->massFilter(), m_budget)(0, 0);
        if (!fields.empty())
        {
            // The values of the point: a field of a designed laminate or section moves its card's derived cells with
            // the others at these values.
            std::vector<double> pointFields;
            pointFields.reserve(fields.size());
            for (const FieldValue& value : fields)
                pointFields.push_back(value.value);
            const AffectedElements affected(*m_in.model, m_set);
            // The designed sections linearised at the point, one call of the library for all.
            std::optional<SectionLinearisation> sections;
            if (m_in.targets->derivedProperties().hasSections())
                sections = m_in.targets->linearSections(pointFields, m_in.sectionReference, wantedGroups());
            const FieldBatches batches = fieldBatchesOf(affected);
            std::shared_ptr<const LazyFieldPerturbation> batchSides; // the sides of the batch being differenced
            const SectionLinearisation* linearisedSections = sections ? &*sections : nullptr;
            // PERFORMANCE: the element terms of the fields (their stiffness, load and mass rates) computed a wave of
            // fields at a time on the pool and added in field and element order (FieldWave): each element's terms
            // are those of the serial loop, so every pseudo-load, energy rate and explicit part keeps its chain; the
            // serial loop for the eigenvalue and trim items (they take each element's matrices as it is met).
            // Measured: the threaded evaluation of a design point, of a Jacobian and of a ladder of model sizes.
            // On one thread too: the wave groups an element's fields into one call of its family (one plate
            // for the 19 fields of a lamination zone), which the field-by-field loop does not; the same terms.
            // Measured: a twin plate at 576 zones on one thread, the fields' linearisation 4.53 → 1.26 s.
            const bool isWaved = !m_eigen && !m_trim;
            FieldWave wave(*this);
            for (std::size_t r = 0; r < fields.size(); ++r)
            {
                const FieldValue& field = fields[r];
                const std::vector<ElementRef>& elements = affected.of(r);
                if (field.gradient.empty() || elements.empty())
                    continue;
                const DesignRelation& relation = m_in.model->relations()[r];
                const auto [batch, member] = batches.placeOf[r];
                if (!isFieldWanted(field))
                {
                    // A field no wanted variable drives: not differenced. Its batch's sides are still
                    // made lazily at its first member and dropped after its last, so that the batches of the other
                    // members are those of the whole linearisation (the same sides, the same bits).
                    if (batch != NO_BATCH && member == 0)
                    {
                        const std::vector<BatchedField>& fields = batches.batches[batch];
                        batchSides = std::make_shared<const LazyFieldPerturbation>(
                            [this, &fields]
                            { return FieldPerturbation(m_in.overlaid, m_index, fields, elementRegistry()); });
                    }
                    if (batch != NO_BATCH && member + 1 == batches.batches[batch].size())
                        batchSides.reset();
                    continue;
                }
                std::unique_ptr<LazyFieldPerturbation> own;
                if (batch == NO_BATCH)
                {
                    // The sides made from the design point's snapshot with this relation's cells alone, sharing its
                    // index (FieldPerturbation's third constructor: the same snapshots, O(the relation), not O(model)),
                    // at their first use: a field whose computations are all in closed form never makes them (step
                    // 10.53). The captures live until the wave is flushed, below.
                    const FieldTarget& target = (*m_in.targets)[r];
                    const double value = field.value;
                    own = std::make_unique<LazyFieldPerturbation>(
                        [this, &target, value, r, &pointFields, linearisedSections]
                        {
                            const double step = fieldStep(value, fieldStepScale(*m_in.overlaid, target));
                            return FieldPerturbation(m_in.overlaid, m_index, target, value, r, step, elementRegistry(),
                                                     m_in.targets, pointFields, linearisedSections);
                        });
                }
                else
                {
                    // A batch's sides (fieldBatchesOf), made at the first use of one of its members' (none when
                    // every member is in closed form) and held by the members until they are differenced.
                    if (member == 0)
                    {
                        const std::vector<BatchedField>& fields = batches.batches[batch];
                        batchSides = std::make_shared<const LazyFieldPerturbation>(
                            [this, &fields]
                            { return FieldPerturbation(m_in.overlaid, m_index, fields, elementRegistry()); });
                    }
                    own = std::make_unique<LazyFieldPerturbation>([sides = batchSides, member]
                                                                  { return sides->get().member(member); });
                    if (member + 1 == batches.batches[batch].size())
                        batchSides.reset();
                }
                const elements::DesignedField designed{relation.targetCard, relation.field, relation.targetId,
                                                       relation.fieldItem};
                // The closed forms of the families (the AnisoMat2D terms included) take a field that
                // is one cell, the field itself (a scalar column, or one item of a fixed-list column): not the derived
                // cells of a section, nor a field written in two tables or scaled. A laminate's ply thickness is its
                // card's cell, the derived stack the families' rates follow (LaminateRate).
                const std::vector<TargetCell>& cells = (*m_in.targets)[r].cells;
                const std::optional<std::size_t> derivedGroup = m_in.targets->derivedProperties().groupOf(r);
                const bool isPlainCell = cells.size() == 1 && cells.front().item == relation.fieldItem &&
                                         cells.front().scale == 1.0 &&
                                         !m_in.targets->hasDerivedMaterial(r) &&
                                         (!derivedGroup || m_in.targets->derivedProperties().isLaminate(*derivedGroup));
                if (isWaved)
                {
                    wave.add(r, elements, std::move(own), designed, isPlainCell, relation.label());
                    if (wave.isFull())
                        wave.flush();
                    continue;
                }
                const LazyFieldPerturbation& perturbation = *own;
                differenceQuantity(r, field, perturbation, elements, relation.label(),
                                   isPlainCell ? &designed : nullptr, [&](const elements::ElementFamily& family)
                                   { return family.designFieldUse(designed); });
            }
            wave.flush();
        }
        if (!shapes.empty())
            differenceShapes();
    }

    // Whether a designed field or shape is differenced: every one, or with
    // GradientOptions::isVariableWanted those whose gradient reaches a wanted variable.
    [[nodiscard]] bool isFieldWanted(const FieldValue& field) const
    {
        const std::vector<std::uint8_t>& wanted = m_options.isVariableWanted;
        if (wanted.empty())
            return true;
        return std::any_of(field.gradient.begin(), field.gradient.end(),
                           [&](const auto& term) { return term.first < wanted.size() && wanted[term.first] != 0; });
    }

    // The derived groups (laminates and sections, DerivedProperties::groupOf) of the fields differenced: empty, every
    // group, without GradientOptions::isVariableWanted (the sections of the members a bay's directions
    // move, not the some 0.75 s of every section of a stiffened-panel model).
    [[nodiscard]] std::vector<std::uint8_t> wantedGroups() const
    {
        if (m_options.isVariableWanted.empty())
            return {};
        const DerivedProperties& derived = m_in.targets->derivedProperties();
        std::vector<std::uint8_t> wanted(derived.groupCount(), 0); // bounded by the groups
        const std::vector<FieldValue>& fields = m_in.design->fields;
        for (std::size_t r = 0; r < fields.size(); ++r)
            if (const std::optional<std::size_t> group = derived.groupOf(r); group && isFieldWanted(fields[r]))
                wanted[*group] = 1;
        return wanted;
    }

    static constexpr std::size_t NO_BATCH = static_cast<std::size_t>(-1);
    // The fields differenced on shared sides (FieldPerturbation's fourth constructor): per batch its fields in relation
    // order, and per relation its batch and its place there (NO_BATCH: its own sides).
    struct FieldBatches
    {
        std::vector<std::vector<BatchedField>> batches;
        std::vector<std::pair<std::size_t, std::size_t>> placeOf;
    };

    // The batches of the fields (for performance): consecutive relations whose cells and reached elements
    // are disjoint from those of the batch, placed greedily in relation order, so that each batch's members come in
    // the order the relations are differenced (its sides made at its first member, dropped after its last: the
    // batches are not interleaved). Only where every computation on a perturbation reads the elements the field
    // reaches and nothing else: no shape variable, no designed laminate or section (their derived cells), and none
    // of the items whose differences read the model around the elements (the deflection over thickness, the section,
    // cut and zone items, the ESE percentages, the eigenvalue and trim items, the external units). Elsewhere every
    // field has its own sides, as before.
    [[nodiscard]] FieldBatches fieldBatchesOf(const AffectedElements& affected) const
    {
        const std::vector<FieldValue>& fields = m_in.design->fields;
        FieldBatches result;
        result.placeOf.assign(fields.size(), {NO_BATCH, 0});
        const bool canBatch = m_in.design->shapes.empty() && m_in.targets->movedRows().empty() &&
                              m_pointItems.empty() && m_sectionItems.empty() && m_zoneItems.empty() &&
                              m_percentJobs.empty() && !m_eigen && !m_trim && !hasExternalUnits();
        if (!canBatch || std::getenv("SV_NO_BATCH") != nullptr)
            return result;
        std::set<ElementRef> elements;                                                    // of the open batch
        std::set<std::tuple<std::string, std::string, std::int64_t, std::int32_t>> cells; // of the open batch
        std::vector<BatchedField> open;
        std::vector<std::size_t> openRelations;
        const auto close = [&]
        {
            if (open.size() > 1)
            {
                for (std::size_t k = 0; k < openRelations.size(); ++k)
                    result.placeOf[openRelations[k]] = {result.batches.size(), k};
                result.batches.push_back(std::move(open));
            }
            open.clear();
            openRelations.clear();
            elements.clear();
            cells.clear();
        };
        for (std::size_t r = 0; r < fields.size(); ++r)
        {
            const std::vector<ElementRef>& reached = affected.of(r);
            if (fields[r].gradient.empty() || reached.empty())
                continue;
            if (m_in.targets->derivedProperties().groupOf(r).has_value() || m_in.targets->hasDerivedMaterial(r))
            {
                close();
                continue;
            }
            const FieldTarget& target = (*m_in.targets)[r];
            bool isDisjoint = std::none_of(reached.begin(), reached.end(),
                                           [&](const ElementRef& element) { return elements.count(element) != 0; });
            for (const TargetCell& cell : target.cells)
                isDisjoint = isDisjoint && cells.count({cell.table, cell.column, cell.row, cell.item}) == 0;
            if (!isDisjoint)
                close();
            elements.insert(reached.begin(), reached.end());
            for (const TargetCell& cell : target.cells)
                cells.insert({cell.table, cell.column, cell.row, cell.item});
            open.push_back(
                {&target, fields[r].value, r, fieldStep(fields[r].value, fieldStepScale(*m_in.overlaid, target))});
            openRelations.push_back(r);
        }
        close();
        return result;
    }

    // The shape variables: each basis moved along its vector (FieldPerturbation), on the elements its moves
    // reach (ShapeElements); every computation of those elements may change (FieldUse::all()), and the distributed
    // loads of every subcase that has some (a PointsPressureLoad on moved grid points has no element) are differenced
    // on the whole load vector.
    void differenceShapes()
    {
        const std::vector<FieldValue>& shapes = m_in.design->shapes;
        const ShapeElements reached(*m_in.model, *m_in.targets, m_set);
        const std::size_t fieldCount = m_in.design->fields.size();
        for (std::size_t b = 0; b < shapes.size(); ++b)
        {
            const FieldValue& shape = shapes[b];
            if (shape.gradient.empty() || !isFieldWanted(shape) || reached.step(b) == 0.0)
                continue;
            const ShapeBasis& basis = m_in.model->shapes().bases()[b];
            const std::string label =
                base::concatenate({"the DesignNodeMove of DesignVar ", std::to_string(basis.variableId)});
            const std::vector<DeformedRigidElement> deformed = reached.deformedRigidElements(b);
            if (!deformed.empty())
                throw base::OptimisationError(base::concatenate(
                    {"the sensitivities of ", label, ": they move grid point ", std::to_string(deformed.front().gridId),
                     " of ", deformed.front().type, " ", std::to_string(deformed.front().elementId),
                     " relative to its other grid points (", std::to_string(deformed.size()),
                     " rigid element(s) so deformed), which changes its constraint equations; their derivative is "
                     "not formed in this version. Move every grid point of the rigid element together "
                     "(the same DesignNodeMove vector), or leave them out of the shape"}));
            const LazyFieldPerturbation perturbation(FieldPerturbation(m_in.base, *m_in.model, *m_in.targets,
                                                                       *m_in.design, fieldCount + b, reached.step(b),
                                                                       elementRegistry(), m_budget));
            ++m_report.shapePerturbations;
            differenceQuantity(fieldCount + b, shape, perturbation, reached.of(b), label, nullptr,
                               [](const elements::ElementFamily&) { return elements::FieldUse::all(); });
        }
    }

    // What the element derivatives of an element did (the report's counts).
    struct DerivativeCounts
    {
        std::int64_t analytic = 0;    // the families' closed forms
        std::int64_t differenced = 0; // the stiffness rates differenced
    };

    // One addition to the pseudo-loads of a designed quantity, as the serial loop of differenceQuantity made it: a
    // stiffness rate −∂K_e/∂v·u_s with its energy (PseudoLoads::addStiffnessTerms), a load rate (addLoad: a thermal
    // load) or an applied load rate (addAppliedLoad: pressures and line loads, inertial and relieving loads).
    struct LoadTerm
    {
        enum class Kind : std::uint8_t
        {
            Stiffness,
            Load,
            AppliedLoad
        };
        Kind kind = Kind::Load;
        std::size_t subcase = 0;
        std::vector<base::DofIndex> dofs; // bounded by the element's DOFs
        std::vector<double> values;
        double energy = 0.0; // a stiffness rate's u_sᵀ·∂K_e/∂v·u_s
    };

    // The terms of one element for one designed quantity, in the order the serial loop adds them (elementTermsOf).
    struct ElementTerms
    {
        std::vector<LoadTerm> terms; // bounded by five per subcase
        DerivativeCounts counts;
        ElementMatrixDerivative stiffness; // ∂K_e/∂v, kept for the eigenvalue and trim items
        // The closed-form rates of the element's jobs (FieldWave::computeGroup): (job, its fieldRates), the jobs whose
        // family gives none left out (differenceQuantity then asks or differences them).
        std::vector<std::pair<std::size_t, std::vector<double>>> jobRates;
    };

    class FieldWave;
    // The element terms of a quantity computed in a FieldWave: the wave and the quantity's first task in it.
    struct WaveSlot
    {
        FieldWave* terms = nullptr;
        std::size_t firstTask = 0;
    };

    // The pseudo-loads and explicit parts of one designed quantity q (a field, or a shape at fieldCount + basis) from
    // its perturbation, on the elements it reaches; `useOf` says what it changes in an element of a family; `designed`
    // the field of a relation whose element matrices a family may give in closed form (null: a shape, or a field to
    // difference).
    // `wave`: the field's element terms computed on the pool (FieldWave), taken in element order; null: computed
    // here, element by element.
    template <typename UseOf>
    void differenceQuantity(std::size_t q, const FieldValue& quantity, const LazyFieldPerturbation& perturbation,
                            const std::vector<ElementRef>& elements, const std::string& label,
                            const elements::DesignedField* designed, const UseOf& useOf, const WaveSlot* wave = nullptr)
    {
        const bool isShape = q >= m_in.design->fields.size();
        if (m_options.elementDerivatives != ElementDerivativeMethod::Analytic)
            designed = nullptr;
        const input::ParamTable* params =
            m_in.overlaid->hasTable(input::TableRole::Params) ? &m_in.overlaid->params() : nullptr;
        std::set<std::pair<std::string, std::int32_t>> affectedIds;
        std::set<ElementRef> affectedSet(elements.begin(), elements.end());
        std::vector<std::uint8_t> needsDistributed(m_subcases.size(), 0);
        // A shape moves the trim's splines, its rigid-body modes and its support modes with the grid points, which the
        // derivatives of TrimItems (∂K and ∂M at fixed geometry) do not hold.
        // Under inertia relief the support motions D are those of the grid points as they are.
        if (isShape && std::any_of(m_subcases.begin(), m_subcases.end(),
                                   [](const SubcaseState& state) { return state.relief != nullptr; }))
            throw base::OptimisationError(base::concatenate(
                {"the sensitivities under inertia relief along ", label,
                 ": expected a property, material or connection variable, found a shape, which moves the rigid-body "
                 "motions of the supports this version keeps fixed; evaluate the shape design without gradients, or "
                 "without PARAM InertiaRelief"}));
        if (isShape && m_trim && !m_trim->isEmpty())
            throw base::OptimisationError(base::concatenate(
                {"the sensitivities of the SAERO subcases along ", label,
                 ": expected a property, material or connection variable, found a shape, whose derivative of the trim "
                 "(the splines, rigid-body modes and support modes moved with the grid points) this version does not "
                 "give; evaluate the shape design without gradients, or remove the responses of the SAERO subcases"}));
        if (wave == nullptr)
            prepareLoadRates(elements, useOf, isShape);
        // The jobs' rates the wave made with the element terms (copied: a later chunk may replace them).
        std::map<std::size_t, std::vector<double>> jobRatesOf;
        for (std::size_t k = 0; k < elements.size(); ++k)
        {
            const ElementRef& element = elements[k];
            const assembly::AssemblyBatch& batch = m_set.batches()[element.batch];
            const elements::ElementFamily& family = *batch.family;
            if (!m_pointItems.empty()) // differencePointItems reads them
                affectedIds.emplace(batch.view.card(), batch.view.elementId(element.row));
            requireDerivatives(family, batch.view, element.row, label);
            if (wave != nullptr)
            {
                const ElementTerms& taken = wave->terms->take(wave->firstTask + k);
                applyElementTerms(q, taken);
                for (const auto& [job, rates] : taken.jobRates)
                    jobRatesOf.emplace(job, rates);
                continue;
            }
            ElementTerms terms = elementTermsOf(element, perturbation, designed, useOf, isShape, m_eigen || m_trim);
            applyElementTerms(q, terms);
            if (m_eigen || m_trim)
            {
                const elements::FieldUse use = useOf(family);
                const ElementDifferences differences(family, perturbation, element.batch);
                if (m_eigen)
                    m_eigen->addElement(
                        q, family, use, terms.stiffness,
                        [&](const elements::MassOptions& options)
                        {
                            const ElementMatrices matrices{family, batch.view, element.row, designed, differences};
                            DerivativeCounts ignored;
                            ElementMatrixDerivative mass = massDerivativeOf(matrices, options, ignored);
                            m_report.analyticDerivatives += ignored.analytic;
                            return mass;
                        },
                        [&](const elements::NodalDisplacements& preload)
                        {
                            return differentialDerivativeOf(family, batch.view, element.row, designed, differences,
                                                            preload);
                        });
                if (m_trim)
                    m_trim->addElement(q, family, batch.view.elementId(element.row), use, terms.stiffness,
                        [&](const elements::MassOptions& options)
                        {
                            const ElementMatrices matrices{family, batch.view, element.row, designed, differences};
                            DerivativeCounts counts;
                            auto mass = massDerivativeOf(matrices, options, counts);
                            m_report.analyticDerivatives += counts.analytic;
                            m_report.elementDerivatives += counts.differenced;
                            return mass;
                        });
                if (m_modal && !m_modal->isEmpty())
                    m_modal->addElement(q, element, use, terms.stiffness,
                        [&](const elements::MassOptions& options)
                        {
                            const ElementMatrices matrices{family, batch.view, element.row, designed, differences};
                            DerivativeCounts counts;
                            auto mass = massDerivativeOf(matrices, options, counts);
                            m_report.analyticDerivatives += counts.analytic;
                            m_report.elementDerivatives += counts.differenced;
                            return mass;
                        }, designed, perturbation);
            }
        }
        if (m_trim)
            m_trim->finishQuantity(q, m_linearised);
        // A shape that turns a concentrated load (a PointForceLoad1 or PointForceLoad2 whose grid points it moves, a
        // FORCE in a system it moves: DerivedGeometry) changes the concentrated loads of the subcases that have some.
        const std::size_t fieldCount = m_in.design->fields.size();
        const bool turnsLoads = isShape && m_in.targets->derived().changesConcentratedLoads(q - fieldCount);
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const solutions::StaticSubcase& subcase = m_subcases[s].subcase;
            if (isShape && m_subcases[s].hasDistributedLoads)
                needsDistributed[s] = 1;
            if (needsDistributed[s] != 0)
                addLoadDifference(q, s, perturbation.get(), assembly::LoadFamily::Distributed, *subcase.load, params);
            if (isShape && m_subcases[s].hasInertialLoads)
                addLoadDifference(q, s, perturbation.get(), assembly::LoadFamily::Inertial, *subcase.load, params);
            if (turnsLoads && m_subcases[s].hasConcentratedLoads)
                addLoadDifference(q, s, perturbation.get(), assembly::LoadFamily::Concentrated, *subcase.load, params);
        }
        m_pseudoLoads->finish(q);
        // The SUBCOMs' pseudo-loads: their stiffness terms (above) and their terms' load parts.
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (m_subcases[s].isCombination())
                m_pseudoLoads->addCombination(q, s, m_subcases[s].terms);
        m_pseudoLoads->projectRelief(q);

        // The explicit parts of the items on the elements the quantity reaches: their jobs, in job order (each item is
        // one job's: the order of the terms of an item, the quantities in order, is that of a pass over every job),
        // and the ESE percentages of the others.
        std::vector<std::size_t> reachedJobs; // bounded by the jobs
        for (const ElementRef& element : affectedSet)
        {
            auto found = std::lower_bound(m_jobsByElement.begin(), m_jobsByElement.end(), element,
                                          [](const auto& entry, const ElementRef& key) { return entry.first < key; });
            for (; found != m_jobsByElement.end() && found->first == element; ++found)
                reachedJobs.push_back(found->second);
        }
        std::sort(reachedJobs.begin(), reachedJobs.end());
        std::vector<std::size_t> visitedJobs = reachedJobs; // bounded by the jobs
        visitedJobs.insert(visitedJobs.end(), m_percentJobs.begin(), m_percentJobs.end());
        std::sort(visitedJobs.begin(), visitedJobs.end());
        visitedJobs.erase(std::unique(visitedJobs.begin(), visitedJobs.end()), visitedJobs.end());
        for (const std::size_t j : visitedJobs)
        {
            ElementJob& job = m_jobs[j];
            SubcaseState& state = m_subcases[job.subcase];
            if (!std::binary_search(reachedJobs.begin(), reachedJobs.end(), j))
            {
                // The quantity leaves the element unchanged, but ESE item 3 is a share of the total energy, which it
                // changes: ∂p/∂v = −(100·E/T²)·½·uᵀ·(∂K/∂v)·u.
                for (std::size_t k = 0; k < job.keys.size(); ++k)
                    if (isPercent(job, k) && state.totalEnergy != 0.0)
                        addExplicit(job.keys[k], quantity,
                                    -PERCENT * job.baseValues[k] / (state.totalEnergy * state.totalEnergy) * 0.5 *
                                        m_pseudoLoads->energyRate(q, job.subcase));
                continue;
            }
            const elements::NodalDisplacements displacements = state.nodal(m_dofMap);
            // PERFORMANCE: the explicit parts in closed form where the family gives them for the field,
            // instead of four recoveries of the job on the perturbed sides.
            // Measured: the element derivatives of a design point, closed forms against differences.
            std::optional<std::vector<double>> rates;
            if (const auto made = jobRatesOf.find(j); made != jobRatesOf.end())
                rates = made->second;
            else if (designed != nullptr)
                rates = job.function->fieldRates(m_set.batches()[job.element.batch].view, displacements, *designed);
            std::array<std::vector<double>, FIELD_SIDE_COUNT> sides;
            if (rates)
                ++m_report.analyticExplicitParts;
            else
                for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
                {
                    const auto which = static_cast<FieldSide>(side);
                    sides[side] = jobValues(job, perturbation.get().set(which).batches()[job.element.batch].view,
                                            perturbation.get().snapshot(which), displacements);
                }
            for (std::size_t k = 0; k < job.keys.size(); ++k)
            {
                double derivative =
                    rates ? (*rates)[k]
                          : perturbation.get().derivative(sides[0][k], sides[1][k], sides[2][k], sides[3][k]);
                if (isPercent(job, k))
                {
                    // ∂p/∂v = (100/T)·∂E/∂v − (100·E/T²)·∂T/∂v, ∂T/∂v = ½·uᵀ·(∂K/∂v)·u.
                    const double total = state.totalEnergy;
                    const double energy = job.baseValues[k];
                    const double totalRate = 0.5 * m_pseudoLoads->energyRate(q, job.subcase);
                    derivative = total != 0.0
                                     ? PERCENT * derivative / total - PERCENT * energy / (total * total) * totalRate
                                     : 0.0;
                }
                addExplicit(job.keys[k], quantity, derivative);
            }
        }
        for (const SpcItem& item : m_spcItems)
        {
            // r = wᵀ·(K·u − P): ∂r/∂v = wᵀ·(∂K/∂v·u − ∂P/∂v) = −wᵀ·q.
            const SparseVector& load = m_pseudoLoads->of(q, item.subcase);
            double product = 0.0;
            std::size_t a = 0;
            for (const auto& [dof, weight] : item.w)
            {
                while (a < load.size() && load[a].first < dof)
                    ++a;
                if (a < load.size() && load[a].first == dof)
                    product += weight * load[a].second;
            }
            addExplicit(item.key, quantity, -product);
        }
        // A field that changes no mass of the elements it reaches (a AnisoMat2D stiffness term of a shell) leaves
        // WEIGHT, VOLUME and FRMASS where they are: no differences of them.
        const bool changesMass =
            isShape || std::any_of(elements.begin(), elements.end(), [&](const ElementRef& element)
                                   { return useOf(*m_set.batches()[element.batch].family).changesMass; });
        if (changesMass)
            differenceMassItems(quantity, perturbation, affectedSet, isShape);
        differenceSelfAdjointItems(q, quantity);
        differencePointItems(quantity, perturbation, affectedIds);
        differenceSectionItems(quantity, perturbation, affectedSet, designed);
        differenceZoneItems(q, quantity, perturbation, affectedSet);
        // The fields differenced somewhere (their perturbation made: LazyFieldPerturbation); the shapes are counted
        // apart.
        if (!isShape && perturbation.isBuilt())
            ++m_report.fieldPerturbations;
    }

    // The fields of a wave of the field loop (differenceFields): consecutive fields with their
    // perturbations and their (field, element) tasks, the element terms of the tasks computed on the pool a chunk at a
    // time and taken in task order by differenceQuantity (a WaveSlot), so that the pseudo-loads receive each element's
    // terms in the order of the serial loop: the same bits whatever the threads. Not for a shape, nor with eigenvalue
    // or trim items (they read each element's matrices as it is met).
    // Thread-safety: used by the computation's thread; the tasks of a chunk run on the pool, each writing its own
    // terms.
    class FieldWave
    {
        struct Field
        {
            std::size_t relation = 0;
            const std::vector<ElementRef>* elements = nullptr;
            std::unique_ptr<LazyFieldPerturbation> perturbation;
            elements::DesignedField designed;
            bool isPlainCell = false;
            std::string label;
            std::size_t firstTask = 0;
        };
        // What the field changes in an element of a family (the useOf of the serial loop).
        struct UseOfField
        {
            const Field* field;
            [[nodiscard]] elements::FieldUse operator()(const elements::ElementFamily& family) const
            {
                return family.designFieldUse(field->designed);
            }
        };

    public:
        explicit FieldWave(SensitivityComputation& owner) : m_owner(&owner) {}

        // Adds field `relation` with its perturbation (its elements: AffectedElements, which must outlive the wave),
        // after the checks of the serial loop (an element type without derivatives) and the load rates its elements
        // read (prepareLoadRates).
        void add(std::size_t relation, const std::vector<ElementRef>& elements,
                 std::unique_ptr<LazyFieldPerturbation> perturbation, const elements::DesignedField& designed,
                 bool isPlainCell, std::string label)
        {
            Field& field = m_fields.emplace_back();
            field.relation = relation;
            field.elements = &elements;
            field.perturbation = std::move(perturbation);
            field.designed = designed;
            field.isPlainCell = isPlainCell;
            field.label = std::move(label);
            field.firstTask = m_taskCount;
            for (const ElementRef& element : elements)
            {
                const assembly::AssemblyBatch& batch = m_owner->m_set.batches()[element.batch];
                requireDerivatives(*batch.family, batch.view, element.row, field.label);
            }
            m_owner->prepareLoadRates(elements, UseOfField{&field}, false);
            m_taskCount += elements.size();
        }

        [[nodiscard]] bool isFull() const noexcept
        {
            return m_taskCount >= FIELD_WAVE_ELEMENTS || m_fields.size() >= FIELD_WAVE_FIELDS;
        }

        // The pseudo-loads and explicit parts of the wave's fields, in order (differenceQuantity); then empty.
        void flush()
        {
            for (Field& field : m_fields)
            {
                const WaveSlot slot{this, field.firstTask};
                m_owner->differenceQuantity(field.relation, m_owner->m_in.design->fields[field.relation],
                                            *field.perturbation, *field.elements, field.label,
                                            field.isPlainCell ? &field.designed : nullptr, UseOfField{&field}, &slot);
            }
            m_fields.clear();
            m_taskCount = 0;
            m_chunk.clear();
            m_chunkBegin = 0;
            m_chunkEnd = 0;
        }

        // The terms of task `task`, the tasks taken in increasing order: its chunk computed when first reached.
        [[nodiscard]] const ElementTerms& take(std::size_t task)
        {
            if (task < m_chunkBegin || task >= m_chunkEnd)
                computeChunk(task);
            return m_chunk[task - m_chunkBegin];
        }

    private:
        void computeChunk(std::size_t first)
        {
            m_chunkBegin = first;
            m_chunkEnd = std::min(m_taskCount, first + FIELD_WAVE_CHUNK_TASKS);
            m_chunk.clear();
            m_chunk.resize(m_chunkEnd - m_chunkBegin);
            const bool isAnalytic = m_owner->m_options.elementDerivatives == ElementDerivativeMethod::Analytic;
            // PERFORMANCE: the chunk's tasks grouped by element (in task order within a group), the closed-form
            // stiffness rates of a group's fields given by one call of the family (stiffnessFieldDerivatives: the
            // element resolved and formulated once for the 19 fields of a lamination zone); each task's terms are
            // then made as alone (elementTermsOf with that rate), so their bits.
            // Measured: laminated models of increasing size.
            std::vector<std::pair<ElementRef, std::size_t>> byElement; // bounded by the chunk's tasks
            byElement.reserve(m_chunk.size());
            for (std::size_t k = 0; k < m_chunk.size(); ++k)
            {
                const std::size_t task = m_chunkBegin + k;
                const Field& field = m_fields[fieldOf(task)];
                byElement.emplace_back((*field.elements)[task - field.firstTask], k);
            }
            std::sort(byElement.begin(), byElement.end());
            std::vector<std::size_t> groupStarts; // bounded by the chunk's tasks
            for (std::size_t g = 0; g < byElement.size(); ++g)
                if (g == 0 || byElement[g].first != byElement[g - 1].first)
                    groupStarts.push_back(g);
            groupStarts.push_back(byElement.size());
            const std::size_t groupCount = groupStarts.size() - 1;
            const std::size_t groupsPerPiece =
                std::max<std::size_t>(1, FIELD_TASKS_PER_PIECE * groupCount / std::max<std::size_t>(1, m_chunk.size()));
            base::parallelFor(m_owner->m_in.pool, groupCount, groupsPerPiece,
                              [&](std::size_t begin, std::size_t end)
                              {
                                  for (std::size_t group = begin; group < end; ++group)
                                      computeGroup(std::span(byElement).subspan(
                                                       groupStarts[group], groupStarts[group + 1] - groupStarts[group]),
                                                   isAnalytic);
                              });
        }

        // The terms of the tasks of one element (computeChunk), its closed-form stiffness rates in their fields made
        // by one call of its family when there are several.
        void computeGroup(std::span<const std::pair<ElementRef, std::size_t>> tasks, bool isAnalytic)
        {
            const ElementRef& element = tasks.front().first;
            const assembly::AssemblyBatch& batch = m_owner->m_set.batches()[element.batch];
            const elements::ElementFamily& family = *batch.family;
            std::vector<elements::DesignedField> designedFields; // bounded by the wave's fields
            std::vector<std::size_t> placeOf(tasks.size(), NO_PLACE);
            if (isAnalytic && tasks.size() > 1 && family.facets().has(elements::Facet::Stiffness))
                for (std::size_t i = 0; i < tasks.size(); ++i)
                {
                    const Field& field = m_fields[fieldOf(m_chunkBegin + tasks[i].second)];
                    if (field.isPlainCell && family.designFieldUse(field.designed).changesStiffness)
                    {
                        placeOf[i] = designedFields.size();
                        designedFields.push_back(field.designed);
                    }
                }
            std::vector<elements::ElementMatrixBlock> rates(designedFields.size());
            std::vector<std::uint8_t> isClosed(designedFields.size(), 0);
            if (designedFields.size() > 1)
                family.stiffnessFieldDerivatives(batch.view, element.row, designedFields, rates, isClosed);
            // The closed-form rates of the element's jobs in the tasks' fields (differenceQuantity's explicit parts),
            // made here, on the pool, by one call per job for all of them.
            std::vector<elements::DesignedField> jobFields; // bounded by the wave's fields
            std::vector<std::size_t> jobPlaceOf(tasks.size(), NO_PLACE);
            if (isAnalytic)
                for (std::size_t i = 0; i < tasks.size(); ++i)
                {
                    const Field& field = m_fields[fieldOf(m_chunkBegin + tasks[i].second)];
                    if (field.isPlainCell)
                    {
                        jobPlaceOf[i] = jobFields.size();
                        jobFields.push_back(field.designed);
                    }
                }
            std::vector<std::pair<std::size_t, std::vector<std::optional<std::vector<double>>>>> jobRates;
            if (!jobFields.empty())
            {
                const auto& byElement = m_owner->m_jobsByElement;
                auto found =
                    std::lower_bound(byElement.begin(), byElement.end(), element,
                                     [](const auto& entry, const ElementRef& key) { return entry.first < key; });
                for (; found != byElement.end() && found->first == element; ++found)
                {
                    const ElementJob& job = m_owner->m_jobs[found->second];
                    jobRates.emplace_back(
                        found->second,
                        job.function->fieldRatesOfFields(
                            batch.view, m_owner->m_subcases[job.subcase].nodal(m_owner->m_dofMap), jobFields));
                }
            }
            for (std::size_t i = 0; i < tasks.size(); ++i)
            {
                const std::size_t k = tasks[i].second;
                const Field& field = m_fields[fieldOf(m_chunkBegin + k)];
                const elements::DesignedField* designed = field.isPlainCell && isAnalytic ? &field.designed : nullptr;
                const std::size_t place = placeOf[i];
                const elements::ElementMatrixBlock* closed =
                    designedFields.size() > 1 && place != NO_PLACE && isClosed[place] != 0 ? &rates[place] : nullptr;
                m_chunk[k] = m_owner->elementTermsOf(element, *field.perturbation, designed, UseOfField{&field}, false,
                                                     false, closed);
                if (jobPlaceOf[i] != NO_PLACE)
                    for (const auto& [job, ofFields] : jobRates)
                        if (ofFields[jobPlaceOf[i]])
                            m_chunk[k].jobRates.emplace_back(job, *ofFields[jobPlaceOf[i]]);
            }
        }

        static constexpr std::size_t NO_PLACE = static_cast<std::size_t>(-1);

        // The field of a task: the last whose first task is at most it.
        [[nodiscard]] std::size_t fieldOf(std::size_t task) const noexcept
        {
            const auto after = std::upper_bound(m_fields.begin(), m_fields.end(), task,
                                                [](std::size_t t, const Field& field) { return t < field.firstTask; });
            return static_cast<std::size_t>(after - m_fields.begin()) - 1;
        }

        SensitivityComputation* m_owner;
        std::vector<Field> m_fields; // bounded by FIELD_WAVE_FIELDS
        std::size_t m_taskCount = 0;
        std::vector<ElementTerms> m_chunk; // bounded by FIELD_WAVE_CHUNK_TASKS
        std::size_t m_chunkBegin = 0;
        std::size_t m_chunkEnd = 0;
    };

    // The check of the serial loop: an element a designed quantity reaches has design derivatives.
    static void requireDerivatives(const elements::ElementFamily& family, const elements::ElementBatchView& view,
                                   std::int64_t row, const std::string& label)
    {
        if (!family.facets().has(elements::Facet::Derivatives))
            throw base::OptimisationError(base::concatenate(
                {"the sensitivities of ", label, " reach ", view.card(), " ", std::to_string(view.elementId(row)),
                 ", whose element type has no design derivatives in this version; remove the element from "
                 "the designed property, material or shape"}));
    }

    // The inertial and relieving load rates the elements of a field read (inertialRateOf, reliefRateOf: made at
    // their first use), made before its element terms, which may be computed on the pool (elementTermsOf).
    template <typename UseOf>
    void prepareLoadRates(const std::vector<ElementRef>& elements, const UseOf& useOf, bool isShape)
    {
        if (isShape)
            return;
        const bool changesMass =
            std::any_of(elements.begin(), elements.end(),
                        [&](const ElementRef& element)
                        {
                            const elements::ElementFamily& family = *m_set.batches()[element.batch].family;
                            return useOf(family).changesMass && family.facets().has(elements::Facet::Mass);
                        });
        if (!changesMass)
            return;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            if (m_subcases[s].hasInertialLoads && m_subcases[s].subcase.load)
                static_cast<void>(inertialRateOf(s));
            if (m_subcases[s].relief != nullptr)
                static_cast<void>(reliefRateOf(s));
        }
    }

    // The terms one element adds to the pseudo-loads of a designed quantity (the order of the serial loop: its
    // stiffness rate in every subcase, then its thermal load rates, its pressure and line load rates, and subcase by
    // subcase its inertial and relieving load rates). Reads the computation only (the load rates made beforehand by
    // prepareLoadRates), so it may run on any thread. `keepsStiffness`: keep ∂K_e/∂v in the terms (the eigenvalue and
    // trim items).
    template <typename UseOf>
    [[nodiscard]] ElementTerms elementTermsOf(const ElementRef& element, const LazyFieldPerturbation& perturbation,
                                              const elements::DesignedField* designed, const UseOf& useOf, bool isShape,
                                              bool keepsStiffness,
                                              const elements::ElementMatrixBlock* closedStiffness = nullptr) const
    {
        ElementTerms out;
        const assembly::AssemblyBatch& batch = m_set.batches()[element.batch];
        const elements::ElementFamily& family = *batch.family;
        const elements::FieldUse use = useOf(family);
        const ElementDifferences derivatives(family, perturbation, element.batch);
        const ElementMatrices matrices{family, batch.view, element.row, designed, derivatives};
        if (use.changesStiffness && family.facets().has(elements::Facet::Stiffness))
        {
            // `closedStiffness`: the family's closed form, made with those of other fields (FieldWave::computeGroup).
            ElementMatrixDerivative stiffness = closedStiffness != nullptr
                                                    ? derivativeIn(*closedStiffness)
                                                    : stiffnessDerivativeOf(matrices, out.counts);
            if (closedStiffness != nullptr)
                ++out.counts.analytic;
            if (!stiffness.isEmpty())
            {
                std::vector<base::DofIndex> dofs; // bounded by the element's DOFs
                for (const elements::ElementDof& dof : stiffness.dofs)
                    dofs.push_back(m_dofMap.dofOf(dof));
                for (std::size_t s = 0; s < m_subcases.size(); ++s)
                {
                    PseudoLoads::StiffnessTerms terms =
                        PseudoLoads::stiffnessTerms(dofs, stiffness.values, m_subcases[s].displacements.span());
                    out.terms.push_back({LoadTerm::Kind::Stiffness, s, dofs, std::move(terms.loads), terms.energy});
                }
            }
            if (keepsStiffness)
                out.stiffness = std::move(stiffness);
        }
        if (use.changesLoads && !family.isThermallyInert())
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                // A SUBCOM's temperatures recover its items; its loads are its terms' (addCombination).
                if (!m_subcases[s].temperatures || m_subcases[s].isCombination())
                    continue;
                ElementVectorDerivative loads =
                    derivatives().thermalLoads(element.row, m_subcases[s].temperatures->temperatures());
                std::vector<base::DofIndex> dofs; // bounded by the element's DOFs
                for (const elements::ElementDof& dof : loads.dofs)
                    dofs.push_back(m_dofMap.dofOf(dof));
                out.terms.push_back({LoadTerm::Kind::Load, s, std::move(dofs), std::move(loads.values), 0.0});
            }
        // The pressures and line loads on the element (the element's own); a shape moves the PLOADs on
        // grid points too, so its distributed loads are differenced on the whole load vectors (differenceQuantity).
        // A field that does not move them (a section or material field of a family that says so: FieldUse) gives
        // none.
        if (use.changesDistributedLoads && !isShape)
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                const auto found = m_subcases[s].elementLoads.find(element);
                if (found == m_subcases[s].elementLoads.end())
                    continue;
                ElementVectorDerivative loads =
                    derivatives().distributedLoads(element.row, found->second.first, found->second.second);
                std::vector<base::DofIndex> dofs; // bounded by the element's DOFs
                for (const elements::ElementDof& dof : loads.dofs)
                    dofs.push_back(m_dofMap.dofOf(dof));
                out.terms.push_back({LoadTerm::Kind::AppliedLoad, s, std::move(dofs), std::move(loads.values), 0.0});
            }
        // The inertial loads of a field that changes the mass, element by element (addInertialLoadRate); a shape
        // moves the acceleration field too (an RotationalForceLoad radius), so its inertial loads are differenced on
        // the whole load vectors.
        if (!isShape && use.changesMass && family.facets().has(elements::Facet::Mass))
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                if (m_subcases[s].hasInertialLoads && m_subcases[s].subcase.load)
                    addInertialLoadRate(s, matrices, out);
                // The relieving loads −M·D·ü_r of a field that changes the mass; a SUBCOM's are its
                // terms'.
                if (m_subcases[s].relief != nullptr && !m_subcases[s].isCombination())
                    addReliefLoadRate(s, matrices, out);
            }
        return out;
    }

    // Adds an element's terms to the pseudo-loads of quantity q, in their order, and its counts to the report.
    void applyElementTerms(std::size_t q, const ElementTerms& terms)
    {
        m_report.analyticDerivatives += terms.counts.analytic;
        m_report.elementDerivatives += terms.counts.differenced;
        for (const LoadTerm& term : terms.terms)
            switch (term.kind)
            {
            case LoadTerm::Kind::Stiffness:
                m_pseudoLoads->addStiffnessTerms(q, term.subcase, term.dofs, term.values, term.energy);
                break;
            case LoadTerm::Kind::Load:
                m_pseudoLoads->addLoad(q, term.subcase, term.dofs, term.values);
                break;
            case LoadTerm::Kind::AppliedLoad:
                m_pseudoLoads->addAppliedLoad(q, term.subcase, term.dofs, term.values);
                break;
            }
    }

    // The differences of one element's facets on a quantity's perturbation (ElementDerivatives), the perturbation
    // made at the first one asked for (LazyFieldPerturbation). Thread-safety: used by one task; the perturbation it
    // borrows may be shared by the tasks of a wave.
    class ElementDifferences
    {
    public:
        ElementDifferences(const elements::ElementFamily& family, const LazyFieldPerturbation& perturbation,
                           std::size_t batch) noexcept
            : m_family(&family), m_perturbation(&perturbation), m_batch(batch)
        {
        }
        [[nodiscard]] const ElementDerivatives& operator()() const
        {
            if (!m_derivatives)
                m_derivatives.emplace(*m_family, m_perturbation->get(), m_batch);
            return *m_derivatives;
        }

    private:
        const elements::ElementFamily* m_family;
        const LazyFieldPerturbation* m_perturbation;
        std::size_t m_batch;
        mutable std::optional<ElementDerivatives> m_derivatives;
    };

    // An element reached by a designed quantity and the ways to differentiate its matrices: in closed form when its
    // family gives them for the designed field, else by the differences of ElementDerivatives.
    struct ElementMatrices
    {
        const elements::ElementFamily& family;
        const elements::ElementBatchView& view;
        std::int64_t row;
        const elements::DesignedField* designed; // null: differences only
        const ElementDifferences& derivatives;
    };

    // The one matrix of an element in a block (empty when the facet appended none).
    [[nodiscard]] static ElementMatrixDerivative derivativeIn(const elements::ElementMatrixBlock& block)
    {
        ElementMatrixDerivative derivative;
        if (block.empty())
            return derivative;
        derivative.dofs.assign(block.dofs(0).begin(), block.dofs(0).end());
        derivative.values.assign(block.matrix(0).begin(), block.matrix(0).end());
        return derivative;
    }

    // dK_e/dv: the family's closed form, else the differences (counted in `counts`).
    [[nodiscard]] static ElementMatrixDerivative stiffnessDerivativeOf(const ElementMatrices& matrices,
                                                                       DerivativeCounts& counts)
    {
        if (matrices.designed != nullptr)
        {
            elements::ElementMatrixBlock block;
            if (matrices.family.stiffnessFieldDerivative(matrices.view, matrices.row, *matrices.designed, block))
            {
                ++counts.analytic;
                return derivativeIn(block);
            }
        }
        ++counts.differenced;
        return matrices.derivatives().stiffness(matrices.row);
    }

    // dM_e/dv with `options`: the family's closed form (counted in `counts`), else the differences.
    [[nodiscard]] static ElementMatrixDerivative massDerivativeOf(const ElementMatrices& matrices,
                                                                  const elements::MassOptions& options,
                                                                  DerivativeCounts& counts)
    {
        if (matrices.designed != nullptr)
        {
            elements::ElementMatrixBlock block;
            if (matrices.family.massFieldDerivative(matrices.view, matrices.row, *matrices.designed, options, block))
            {
                ++counts.analytic;
                return derivativeIn(block);
            }
        }
        return matrices.derivatives().mass(matrices.row, options);
    }

    // dK_d/dv at fixed preload displacements and temperatures (the eigenvalue items of SOL 105): the family's closed
    // form, else the differences.
    [[nodiscard]] static ElementMatrixDerivative differentialDerivativeOf(const elements::ElementFamily& family,
                                                                          const elements::ElementBatchView& view,
                                                                          std::int64_t row,
                                                                          const elements::DesignedField* designed,
                                                                          const ElementDifferences& differences,
                                                                          const elements::NodalDisplacements& preload)
    {
        if (designed != nullptr)
        {
            elements::ElementMatrixBlock block;
            if (family.differentialStiffnessFieldDerivative(view, row, *designed, preload, block))
                return derivativeIn(block);
        }
        return differences().differentialStiffness(row, preload);
    }

    // The eigenvalue items (EigenItems): their roots grouped (repeated roots), the preloads of the LAMA
    // items, and the weights of the LAMA items of single roots (their preload part, solved by the orders of the static
    // items).
    void lineariseEigenItems()
    {
        if (!m_eigen || m_eigen->isEmpty())
            return;
        m_eigen->formClusters();
        for (const SubcaseState& state : m_subcases)
            m_eigen->setPreload(state.subcase.label, state.nodal(m_dofMap));
        for (auto& [key, weights] : m_eigen->bucklingWeights())
            m_linearised.at(key).weights = std::move(weights);
        m_report.eigenvalueItems = static_cast<std::int64_t>(m_eigen->itemCount());
        m_report.repeatedRoots = static_cast<std::int64_t>(m_eigen->repeatedClusterCount());
    }

    // The preload parts of the repeated buckling roots (the direct method on the preload's system) and the explicit
    // parts of every eigenvalue item, once the pseudo-loads are complete.
    void finishEigenItems()
    {
        if (!m_eigen || m_eigen->isEmpty())
            return;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
            if (!m_subcases[s].isAeroelastic)
                m_report.solves +=
                    m_eigen->addPreloadTerms(m_subcases[s].subcase.label, systemOf(m_subcases[s]), *m_pseudoLoads, s);
        m_eigen->addExplicitParts(m_quantities, m_linearised);
    }

    // The items of the SAERO subcases (TrimItems): the static items of those subcases with their weights,
    // then the directions of the trim.
    void lineariseTrimItems()
    {
        if (!m_trim)
            return;
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
            for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
            {
                const ItemLinearisation& item = m_linearised.items[p][i];
                if (item.isWanted && !item.isEquation && item.readsDisplacements() &&
                    m_trim->isTrimSubcase(item.subcase))
                    m_trim->addStaticItem({p, i}, item.subcase);
            }
        m_trim->prepare();
    }

    // The derivatives of the items of the SAERO subcases into their explicit parts, once every quantity is
    // differenced.
    void finishTrimItems()
    {
        if (!m_trim || m_trim->isEmpty())
            return;
        m_trim->addExplicitParts(m_quantities, m_linearised);
        m_report.solves += m_trim->solveCount();
    }

    // The explicit parts of the compliance and mean displacement items (SelfAdjointResponses): at fixed u,
    // ∂c/∂v = uᵀ·∂P/∂v, and ∂m/∂v = (uᵀ·∂P/∂v)/W − m·(∂W/∂v)/W with ∂W/∂v = Σ_g (F_g·∂F_g/∂v)/|F_g| (the applied loads
    // only: the thermal loads are not in W).
    void differenceSelfAdjointItems(std::size_t q, const FieldValue& quantity)
    {
        for (const SelfAdjointItem& item : m_selfAdjointItems)
        {
            const SubcaseState& state = m_subcases[item.subcase];
            double derivative =
                item.scale * dotSparse(m_pseudoLoads->loadPartOf(q, item.subcase), state.displacements.span());
            if (!item.forceWeights.empty())
                derivative -= item.value * item.scale *
                              dotSparseVectors(item.forceWeights, m_pseudoLoads->appliedLoadPartOf(q, item.subcase));
            addExplicit(item.key, quantity, derivative);
        }
    }

    // The explicit parts of the deflection over thickness items whose shell elements the quantity reaches: the normal
    // and the mean thickness at the grid point, on the four sides of the perturbation, at fixed u.
    void differencePointItems(const FieldValue& quantity, const LazyFieldPerturbation& perturbation,
                              const std::set<std::pair<std::string, std::int32_t>>& affectedIds)
    {
        std::array<std::unique_ptr<LinearityIndicator>, FIELD_SIDE_COUNT> sides; // built when an item needs them
        for (const PointItem& item : m_pointItems)
        {
            const ShellPointGeometry* point = linearity().findShellPoint(item.grid);
            const bool isReached = std::any_of(point->elements.begin(), point->elements.end(),
                                               [&](std::size_t e)
                                               {
                                                   const ShellElementGeometry& element = linearity().shellElements()[e];
                                                   return affectedIds.count({element.card, element.elementId}) != 0;
                                               });
            if (!isReached)
                continue;
            std::array<double, FIELD_SIDE_COUNT> values{};
            for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
            {
                if (!sides[side])
                {
                    const std::shared_ptr<const input::ModelSnapshot>& snapshot =
                        perturbation.get().snapshot(static_cast<FieldSide>(side));
                    sides[side] = std::make_unique<LinearityIndicator>(*snapshot, snapshot, m_budget);
                }
                const ShellPointGeometry* moved = sides[side]->findShellPoint(item.grid);
                if (moved == nullptr)
                    throw base::OptimisationError(base::concatenate(
                        {"the sensitivity of the deflection over thickness of grid point ", std::to_string(item.grid),
                         ": a perturbation of the design leaves it without a shell of positive thickness; leave it "
                         "out of nodes= or keep the thickness of its shells away from zero (bounds of the "
                         "variables)"}));
                values[side] = LinearityIndicator::deflectionRatio(*moved, item.translations);
            }
            addExplicit(item.key, quantity, perturbation.get().derivative(values[0], values[1], values[2], values[3]));
        }
    }

    // The self-adjoint items (GradientOptions::usesSelfAdjoint): their adjoint vector is scale·u, so λᵀ·b_j is formed
    // with the pseudo-loads at once (PseudoLoads::dotAll) and added to the explicit part, and the weights are dropped:
    // no solve is counted for them (MethodChoice, GradientStream), and an equation that reads one chains its whole
    // gradient through the explicit part.
    void foldSelfAdjoint()
    {
        if (!m_options.usesSelfAdjoint || m_selfAdjointItems.empty())
            return;
        std::vector<double> products(m_variableCount); // bounded by the design variables
        for (const SelfAdjointItem& item : m_selfAdjointItems)
        {
            ItemLinearisation& linear = m_linearised.at(item.key);
            if (!linear.isWanted || !linear.readsDisplacements())
                continue;
            const SubcaseState& state = m_subcases[item.subcase];
            if (m_variableCount > 0)
                m_pseudoLoads->dotAll(item.subcase, state.displacements.span(), products);
            for (std::size_t j = 0; j < m_variableCount; ++j)
                if (products[j] != 0.0)
                    linear.explicitPart.emplace_back(j, item.scale * products[j]);
            compressRow(linear.explicitPart);
            linear.weights.clear();
            ++m_report.selfAdjointItems;
        }
    }

    // The explicit parts of WEIGHT, VOLUME and FRMASS for one quantity: the mass property of the elements it reaches on
    // each side (the others do not change; MassProperties is linear in the fields, exact on the extrapolation). A shape
    // that moves the reference point of WEIGHT (PARAM GRDPNT) changes the moments of every element about it: then
    // every element is taken, about the reference of each side.
    void differenceMassItems(const FieldValue& quantity, const LazyFieldPerturbation& perturbation,
                             const std::set<ElementRef>& affectedSet, bool isShape)
    {
        if (m_massItems.empty())
            return;
        // The elements the quantity reaches, in increasing (batch, row): MassProperties computes their masses alone
        // (rigidBodyMassOf, volumeOf), the same terms in the same order as a filter that takes them.
        std::vector<MassProperties::SetElement> affected; // bounded by the elements the quantity reaches
        affected.reserve(affectedSet.size());
        for (const ElementRef& element : affectedSet)
            affected.push_back({element.batch, element.row});
        const base::Vec3 reference = DesignResp1Response::weightReference(*m_in.overlaid);
        std::array<base::Vec3, FIELD_SIDE_COUNT> references{reference, reference, reference, reference};
        bool movesReference = false;
        if (isShape)
            for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
            {
                references[side] =
                    DesignResp1Response::weightReference(*perturbation.get().snapshot(static_cast<FieldSide>(side)));
                movesReference = movesReference || !(references[side] == reference);
            }
        const auto isAny = [](std::string_view, std::int32_t, std::int32_t) { return true; };
        for (const MassItem& item : m_massItems)
        {
            const ResponseType type = item.response->definition().type;
            const double value = valuesAt(item.key.response).values[item.key.item];
            if (type == ResponseType::FractionalMass && (value == 0.0 || item.currentMass == 0.0))
                continue;
            // FRMASS: the elements of its selection among those the quantity reaches.
            std::vector<MassProperties::SetElement> selected; // bounded by the elements the quantity reaches
            if (type == ResponseType::FractionalMass)
            {
                const ElementFilter selection = item.response->massFilter();
                for (const MassProperties::SetElement& element : affected)
                {
                    const elements::ElementBatchView& view = m_set.batches()[element.batch].view;
                    const input::ElementBatch& batch = view.batch();
                    const std::int32_t propId =
                        batch.hasProperty() ? batch.propertyIds()[static_cast<std::size_t>(element.row)] : 0;
                    if (!selection || selection(view.card(), view.elementId(element.row), propId))
                        selected.push_back(element);
                }
            }
            // The mass property of the elements the quantity reaches, on one side.
            const auto propertyOn = [&](const assembly::ElementSet& set,
                                        const std::shared_ptr<const input::ModelSnapshot>& owner, base::Vec3 point)
            {
                if (type == ResponseType::Weight)
                {
                    const auto [row, column] = item.response->weightEntry();
                    const base::MatN<6> mass =
                        movesReference
                            ? MassProperties::rigidBodyMass(*owner, point, m_in.massFormulation, isAny, m_budget)
                            : MassProperties::rigidBodyMassOf(set, affected, m_registry, point, m_in.massFormulation);
                    return mass(static_cast<std::size_t>(row - 1), static_cast<std::size_t>(column - 1));
                }
                if (type == ResponseType::Volume)
                    return MassProperties::volumeOf(set, owner, affected, m_registry);
                // FRMASS = m/m0, m0 = m / value at the design point.
                return MassProperties::rigidBodyMassOf(set, selected, m_registry, {0.0, 0.0, 0.0},
                                                       m_in.massFormulation)(0, 0) /
                       (item.currentMass / value);
            };
            std::array<double, FIELD_SIDE_COUNT> sides{};
            for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
            {
                const auto which = static_cast<FieldSide>(side);
                sides[side] =
                    propertyOn(perturbation.get().set(which), perturbation.get().snapshot(which), references[side]);
            }
            addExplicit(item.key, quantity, perturbation.get().derivative(sides[0], sides[1], sides[2], sides[3]));
        }
    }

    // ∂P/∂v of the inertial loads of subcase `subcase` on one element whose mass a designed field changes: the inertial
    // loads are P = Mgg·a (InertialLoads), linear in the mass, and a field moves no grid point, so the acceleration
    // field a is that of the design point and ∂P/∂v = WTMASS·(∂M_e/∂v)·a_e on the element's DOFs (∂M_e/∂v the
    // extrapolated differences of ElementDerivatives::mass, the formulation of the subcase's PARAM COUPMASS), an
    // applied load of the pseudo-load. A selected non-structural mass share is constant with respect to a material or
    // section field; a shape can change its geometry, so shape directions difference the complete selected mass.
    // PERFORMANCE: the derivative was formed on the whole load vectors of the four perturbed snapshots, each with the
    // mass matrix of every element assembled on the pattern of the model: four mass assemblies of the model per
    // variable and subcase; here the mass of the elements the field reaches and one acceleration field per subcase.
    // Measured: 200 thickness variables on 250 000 DOFs under GravityLoad, the inertial pseudo-loads 114.6 s → 3.0 s.
    // The bits of these pseudo-loads change (the rounding of the other elements' masses no
    // longer enters the differences): that Jacobian moved by 3e-14 of its largest entry; the gradients under
    // GravityLoad and RotationalForceLoad, lumped and coupled, checked against the finite differences of the analysis.
    // (The rate of subcase `subcase` made beforehand: prepareLoadRates.)
    void addInertialLoadRate(std::size_t subcase, const ElementMatrices& matrices, ElementTerms& out) const
    {
        const InertialRate& rate = *m_inertialRates[subcase];
        elements::MassOptions options;
        options.formulation = rate.options.formulation;
        const ElementMatrixDerivative mass = massDerivativeOf(matrices, options, out.counts);
        if (mass.isEmpty())
            return;
        const std::size_t n = mass.dofs.size();
        std::vector<base::DofIndex> dofs(n); // bounded by the element's DOFs
        std::vector<double> values(n);
        for (std::size_t i = 0; i < n; ++i)
            dofs[i] = m_dofMap.dofOf(mass.dofs[i]);
        for (std::size_t r = 0; r < n; ++r)
        {
            double sum = 0.0;
            for (std::size_t c = 0; c < n; ++c)
                sum += mass.values[r * n + c] * rate.accelerations[static_cast<std::size_t>(dofs[c])];
            values[r] = rate.options.weightToMass * sum;
        }
        out.terms.push_back({LoadTerm::Kind::AppliedLoad, subcase, std::move(dofs), std::move(values), 0.0});
    }

    // The mass options and the acceleration field of the inertial loads of a subcase, made at its first use.
    struct InertialRate
    {
        assembly::MassAssemblyOptions options;
        base::BudgetedBuffer<double> accelerations; // a, one value per g-set DOF
    };

    const InertialRate& inertialRateOf(std::size_t subcase)
    {
        if (m_inertialRates.empty())
            m_inertialRates.resize(m_subcases.size());
        std::unique_ptr<InertialRate>& rate = m_inertialRates[subcase];
        if (!rate)
        {
            base::DiagnosticSink ignored; // the analysis reported the problems of its loads and parameters
            const input::ParamTable* params =
                m_in.overlaid->hasTable(input::TableRole::Params) ? &m_in.overlaid->params() : nullptr;
            auto made = std::make_unique<InertialRate>();
            made->options =
                assembly::MassAssemblyOptions::fromParams(params, m_subcases[subcase].subcase.label, ignored);
            made->accelerations = base::BudgetedBuffer<double>(m_budget, gSize(), "sensitivity inertial accelerations");
            std::fill(made->accelerations.data(), made->accelerations.data() + gSize(), 0.0);
            m_loads->addInertialAccelerations(*m_subcases[subcase].subcase.load, made->accelerations.span(), m_budget,
                                              ignored);
            rate = std::move(made);
        }
        return *rate;
    }

    // −∂M_e/∂v·D·ü_r of one element in a relieved subcase: the derivative of the relieving loads at fixed
    // accelerations, an applied load of the field (the relief projects it with the others: PseudoLoads::projectRelief).
    // (The rate of subcase `subcase` made beforehand: prepareLoadRates.)
    void addReliefLoadRate(std::size_t subcase, const ElementMatrices& matrices, ElementTerms& out) const
    {
        const InertialRate& rate = *m_reliefRates[subcase];
        elements::MassOptions options;
        options.formulation = rate.options.formulation;
        const ElementMatrixDerivative mass = massDerivativeOf(matrices, options, out.counts);
        if (mass.isEmpty())
            return;
        const std::size_t n = mass.dofs.size();
        std::vector<base::DofIndex> dofs(n); // bounded by the element's DOFs
        std::vector<double> values(n);
        const std::span<const double> field = m_subcases[subcase].reliefAccelerations.span();
        for (std::size_t i = 0; i < n; ++i)
            dofs[i] = m_dofMap.dofOf(mass.dofs[i]);
        for (std::size_t r = 0; r < n; ++r)
        {
            double sum = 0.0;
            for (std::size_t c = 0; c < n; ++c)
                sum += mass.values[r * n + c] * field[static_cast<std::size_t>(dofs[c])];
            values[r] = -rate.options.weightToMass * sum;
        }
        out.terms.push_back({LoadTerm::Kind::AppliedLoad, subcase, std::move(dofs), std::move(values), 0.0});
    }

    // The mass options of the relieving loads of a subcase (those of its analysis' Mgg), made at its first use.
    const InertialRate& reliefRateOf(std::size_t subcase)
    {
        if (m_reliefRates.empty())
            m_reliefRates.resize(m_subcases.size());
        std::unique_ptr<InertialRate>& rate = m_reliefRates[subcase];
        if (!rate)
        {
            base::DiagnosticSink ignored; // the analysis reported the problems of its parameters
            const input::ParamTable* params =
                m_in.overlaid->hasTable(input::TableRole::Params) ? &m_in.overlaid->params() : nullptr;
            rate = std::make_unique<InertialRate>();
            rate->options =
                assembly::MassAssemblyOptions::fromParams(params, m_subcases[subcase].subcase.label, ignored);
        }
        return *rate;
    }

    // ∂P/∂v of one family of loads of a subcase, from the whole load vectors on the perturbed snapshots (the
    // distributed loads of an element whose field changes them; the inertial loads M·a of a mass that changes: they are
    // linear in the mass matrix, assembled on each side).
    void addLoadDifference(std::size_t field, std::size_t subcase, const FieldPerturbation& perturbation,
                           assembly::LoadFamily family, base::EntityId loadSet, const input::ParamTable* params)
    {
        base::DiagnosticSink sink;
        std::vector<base::BudgetedBuffer<double>> sides; // the load vectors of the four sides
        for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
        {
            const assembly::ElementSet& set = perturbation.set(static_cast<FieldSide>(side));
            // The loads of a side read its own tables and positions: the directions and frames a shape
            // derives again (loadsTable, inertialLoadsTable), the moved positions of the radius of an
            // RotationalForceLoad or the location of an ACCEL. The side's DOF map numbers the DOFs as the design
            // point's (the same points).
            const assembly::DofMap sideMap(set.index().snapshot().nodes(), set.index().points());
            base::BudgetedBuffer<double> loads(m_budget, gSize(), "sensitivity load difference");
            std::fill(loads.data(), loads.data() + gSize(), 0.0);
            const assembly::LoadVectorBuilder builder(set, sideMap, m_budget);
            if (family == assembly::LoadFamily::Inertial)
            {
                const assembly::MassAssemblyOptions options =
                    assembly::MassAssemblyOptions::fromParams(params, m_subcases[subcase].subcase.label, sink);
                linalg::BlockSparseMatrix mass(systemOf(m_subcases[subcase]).pattern(), m_budget, "sensitivity mass");
                const assembly::NonStructuralMassShares shares(
                    set, m_subcases[subcase].subcase.nonStructuralMass, m_budget, sink);
                assembly::MassAssembler(set, sideMap, shares, options).assemble(mass, m_in.pool);
                builder.addFamilyLoads(loadSet, family, &mass, loads.span(), m_budget, sink);
            }
            else
                builder.addFamilyLoads(loadSet, family, nullptr, loads.span(), m_budget, sink);
            sides.push_back(std::move(loads));
        }
        base::BudgetedBuffer<double> derivative(m_budget, gSize(), "sensitivity load derivative");
        perturbation.derivatives(sides[0].span(), sides[1].span(), sides[2].span(), sides[3].span(), derivative.span());
        m_pseudoLoads->addDenseLoad(field, subcase, derivative.span());
    }

    // Appends the field's terms to the item's explicit part; compressExplicitParts() sums them per column, in the order
    // they were added (the fields in order), as the dense accumulation did.
    void addExplicit(const ItemKey& key, const FieldValue& field, double derivative)
    {
        if (derivative == 0.0)
            return;
        VariableRow& explicitPart = m_linearised.at(key).explicitPart;
        for (const auto& [column, gradient] : field.gradient)
            explicitPart.emplace_back(column, derivative * gradient);
    }

    void compressExplicitParts()
    {
        for (std::vector<ItemLinearisation>& items : m_linearised.items)
            for (ItemLinearisation& item : items)
                compressRow(item.explicitPart);
    }

    void lineariseEquations()
    {
        KnownResponses known;
        for (const ResponseValues& values : *m_in.values)
            known[values.id] = &values;
        ResultLookup lookup(m_in.results->arrays);
        ResponseContext context;
        context.snapshot = m_in.overlaid.get();
        context.snapshotOwner = m_in.overlaid;
        context.results = &lookup;
        context.model = m_in.model;
        context.design = m_in.design;
        context.massFormulation = m_in.massFormulation;
        context.budget = &m_budget;
        for (const EquationResponse& equation : m_in.responses->equations())
        {
            const std::int32_t id = equation.definition().id;
            const auto position = m_positionOf.find(id);
            if (m_needed.count(id) == 0 || position == m_positionOf.end())
                continue;
            const std::size_t p = position->second;
            const ResponseValues& values = valuesAt(p);
            const EquationPartials partials = equation.evaluateWithPartials(m_in.responseSubcases->at(id), known,
                                                                            context, *m_in.callables, m_in.blockItems);
            if (partials.values.itemCount() != values.itemCount())
                throw base::OptimisationError(base::concatenate(
                    {"the sensitivity of ", equation.definition().card, " ", std::to_string(id),
                     ": its items evaluated again with their partials are not those of the evaluation; keep the "
                     "selection of its arguments the same for the evaluation and its gradients"}));
            for (std::size_t i = 0; i < values.itemCount(); ++i)
            {
                const ItemKey key{p, i};
                ItemLinearisation& item = m_linearised.at(key);
                item.isEquation = true;
                if (!item.isWanted)
                    continue;
                checkValue(values, i, partials.values.values[i]);
                for (const auto& [source, weight] : partials.items[i])
                {
                    switch (source.kind)
                    {
                    case ArgumentSource::Kind::Variable:
                        for (const auto& [column, derivative] : m_in.design->variables.derivatives[source.index])
                            item.ownExplicit.emplace_back(column, weight * derivative);
                        break;
                    case ArgumentSource::Kind::Field:
                        for (const auto& [column, derivative] : m_in.design->fields[source.index].gradient)
                            item.ownExplicit.emplace_back(column, weight * derivative);
                        break;
                    case ArgumentSource::Kind::Response:
                        item.arguments.push_back({ItemKey{m_positionOf.at(source.responseId), source.index}, weight});
                        break;
                    case ArgumentSource::Kind::Coordinate:
                        // A DNODE coordinate c = c0 + Σ_b s_b·T_b: ∂h/∂x = ∂h/∂c · Σ_b T_b · ds_b/dx.
                        for (const auto& [basis, rate] : equation.coordinateShapes(source.index))
                            for (const auto& [column, derivative] : m_in.design->shapes[basis].gradient)
                                item.ownExplicit.emplace_back(column, weight * rate * derivative);
                        break;
                    case ArgumentSource::Kind::Constant:
                        break;
                    }
                }
                compressRow(item.ownExplicit);
                ChainedAdjoint::chain(key, m_linearised);
            }
        }
    }

    void solveSubcases()
    {
        // Every gradient starts at the explicit part (the global items keep it).
        for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
        {
            m_linearised.gradients[p].resize(m_linearised.items[p].size());
            for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
            {
                std::vector<double>& gradient = m_linearised.gradients[p][i];
                gradient.assign(m_variableCount, 0.0);
                addRowTo(m_linearised.items[p][i].explicitPart, gradient);
            }
        }
        // The equation responses in their dependency order, for the argument-first combination.
        std::vector<std::size_t> equationOrder;
        for (const EquationResponse& equation : m_in.responses->equations())
            if (const auto found = m_positionOf.find(equation.definition().id);
                found != m_positionOf.end() && !m_linearised.items[found->second].empty())
                equationOrder.push_back(found->second);
        // The adjoint and chained subcases of one full-path factor, gathered and solved together:
        // consecutive subcases on the same system; what is gathered is solved before any
        // other work on the gradients (a subcase of another method or factor, or a combination of arguments), so that
        // every gradient is read when it was before. Each item's gradient receives its one product, the bits of a
        // subcase solved alone.
        // PERFORMANCE: the chained adjoint took 1.8 times the vjp on a ladder of model sizes (a solve per subcase and
        // block of its items, the right-hand sides and products O(g set) per item on the calling thread); now the vjp's
        // machinery (AdjointColumns).
        // Measured: the chained adjoint on the same ladder.
        const SensitivitySystem* gatheredSystem = nullptr;
        std::vector<AdjointSensitivity::SubcaseItems> gathered; // bounded by the subcases
        std::vector<std::size_t> gatheredChoices;               // their entries of m_report.subcases
        const auto solveGathered = [&]
        {
            if (gathered.empty())
                return;
            const std::vector<std::int64_t> solves = AdjointSensitivity::solveTogether(
                *gatheredSystem, *m_pseudoLoads, gathered, m_linearised, m_budget, m_in.pool);
            for (std::size_t k = 0; k < gathered.size(); ++k)
            {
                m_report.subcases[gatheredChoices[k]].solves = solves[k];
                m_report.solves += solves[k];
            }
            gathered.clear();
            gatheredChoices.clear();
            gatheredSystem = nullptr;
        };
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            if (m_subcases[s].isAeroelastic)
                continue; // the rows of its items are complete (TrimItems)
            if (m_subcases[s].isTermOnly)
                continue; // a SUBCOM's term no item reads
            const std::int32_t label = m_subcases[s].subcase.label;
            std::vector<ItemKey> plainNeeded;
            std::vector<ItemKey> plainRequested;
            for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
                {
                    const ItemLinearisation& item = m_linearised.items[p][i];
                    if (item.isEquation || item.subcase != label || !item.isWanted || !item.readsDisplacements())
                        continue;
                    plainNeeded.push_back({p, i});
                    if (m_requested.count(valuesAt(p).id) != 0)
                        plainRequested.push_back({p, i});
                }
            std::vector<ItemKey> equationNeeded;
            std::vector<ItemKey> equationRequested;
            for (const std::size_t p : equationOrder)
                for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
                {
                    if (m_linearised.items[p][i].subcase != label || !m_linearised.items[p][i].isWanted)
                        continue;
                    equationNeeded.push_back({p, i});
                    if (m_requested.count(valuesAt(p).id) != 0 && m_linearised.items[p][i].readsDisplacements())
                        equationRequested.push_back({p, i});
                }
            std::int64_t variables = 0;
            for (std::size_t j = 0; j < m_variableCount; ++j)
                variables += m_pseudoLoads->hasLoad(j, s) ? 1 : 0;
            SubcaseSensitivity choice = MethodChoice::choose(
                label, variables, static_cast<std::int64_t>(plainNeeded.size()),
                static_cast<std::int64_t>(plainRequested.size() + equationRequested.size()), m_options.method);
            const SensitivitySystem& system = systemOf(m_subcases[s]);
            const SubcaseSolve context{&system, m_pseudoLoads.get(), s, &m_budget};
            const bool isGathered = !system.isCondensed() && (choice.method == SensitivityMethod::Adjoint ||
                                                              choice.method == SensitivityMethod::Chained);
            if (!isGathered || &system != gatheredSystem)
                solveGathered();
            if (isGathered)
            {
                // The adjoint method solves the plain items its equations need; the chained method the plain items
                // asked, then the equation items asked with their chained weights.
                std::vector<ItemKey> items = choice.method == SensitivityMethod::Adjoint ? plainNeeded : plainRequested;
                if (choice.method == SensitivityMethod::Chained)
                    items.insert(items.end(), equationRequested.begin(), equationRequested.end());
                gatheredSystem = &system;
                gathered.push_back({s, std::move(items)});
                gatheredChoices.push_back(m_report.subcases.size());
                m_report.subcases.push_back(choice);
                if (choice.method == SensitivityMethod::Adjoint && !equationNeeded.empty())
                {
                    solveGathered();
                    AdjointSensitivity::combineArguments(equationNeeded, m_linearised);
                }
                continue;
            }
            switch (choice.method)
            {
            case SensitivityMethod::Adjoint:
                choice.solves = AdjointSensitivity::solve(context, plainNeeded, m_linearised);
                AdjointSensitivity::combineArguments(equationNeeded, m_linearised);
                break;
            case SensitivityMethod::Chained:
                choice.solves = AdjointSensitivity::solve(context, plainRequested, m_linearised) +
                                ChainedAdjoint::solve(context, equationRequested, m_linearised);
                break;
            default:
            {
                std::vector<ItemKey> all = plainNeeded;
                all.insert(all.end(), equationNeeded.begin(), equationNeeded.end());
                choice.solves = DirectSensitivity::solve(context, all, m_linearised);
                break;
            }
            }
            m_report.solves += choice.solves;
            m_report.subcases.push_back(choice);
        }
        solveGathered();
        m_report.factorisations = m_modal ? static_cast<std::int32_t>(m_modal->factorisationCount()) : 0;
        for (const std::unique_ptr<SensitivitySystem>& system : m_systems)
            m_report.factorisations += system->factorisationCount();
        collectCondensedStatistics(m_report);
    }

    // The statistics of the solves through the boundaries, per condensed constraint set.
    void collectCondensedStatistics(SensitivityReport& report) const
    {
        report.condensed.clear();
        for (const std::unique_ptr<SensitivitySystem>& system : m_systems)
        {
            if (system->condensed() == nullptr)
                continue;
            BoundaryReducedStatistics statistics = system->condensed()->statistics();
            for (const solutions::StaticSubcase& subcase : system->subcases())
                statistics.subcases.push_back(subcase.label);
            report.condensed.push_back(std::move(statistics));
        }
    }

    DesignGradients collect()
    {
        DesignGradients gradients;
        gradients.variableCount = m_variableCount;
        for (std::size_t p = 0; p < m_in.values->size(); ++p)
        {
            const ResponseValues& values = valuesAt(p);
            if (m_requested.count(values.id) == 0)
                continue;
            ResponseGradient gradient;
            gradient.id = values.id;
            gradient.itemCount = values.itemCount();
            for (std::size_t i = 0; i < m_linearised.gradients[p].size(); ++i)
            {
                if (m_linearised.items[p][i].isWanted)
                {
                    const std::vector<double>& row = m_linearised.gradients[p][i];
                    gradient.values.insert(gradient.values.end(), row.begin(), row.end());
                }
                else
                    gradient.values.insert(gradient.values.end(), m_variableCount,
                                           std::numeric_limits<double>::quiet_NaN());
            }
            gradients.responses.push_back(std::move(gradient));
        }
        // The Rayleigh quotient parts of the EIGN and FREQ items differentiated.
        if (m_eigen && !m_eigen->isEmpty())
        {
            std::map<ItemKey, std::pair<std::int32_t, std::size_t>> wanted;
            for (std::size_t p = 0; p < m_linearised.items.size(); ++p)
                for (std::size_t i = 0; i < m_linearised.items[p].size(); ++i)
                    if (m_linearised.items[p][i].isWanted && m_requested.count(valuesAt(p).id) != 0)
                        wanted[{p, i}] = {valuesAt(p).id, i};
            gradients.rayleigh = m_eigen->rayleighParts(m_quantities, wanted);
        }
        gradients.report = std::move(m_report);
        return gradients;
    }

    const StaticSensitivityInputs& m_in;
    const GradientOptions& m_options;
    base::MemoryBudget& m_budget;
    elements::ElementRegistry m_registry;
    // The elements the analysis' external units replace (null without one), before the set built on their families.
    std::unique_ptr<reduction::ReplacedElements> m_replaced;
    elements::SnapshotIndex m_index;
    assembly::ElementSet m_set;
    assembly::DofMap m_dofMap;
    std::size_t m_variableCount;
    // The designed quantities as the pseudo-loads index them: the fields (one per relation), then the shapes (one per
    // shape basis).
    std::vector<FieldValue> m_quantities;
    std::map<std::int32_t, std::size_t> m_positionOf;
    std::set<std::int32_t> m_requested;
    std::set<std::int32_t> m_needed;
    std::set<std::int32_t> m_readByEquation; // the responses a needed equation reads (every item linearised)
    LinearisedItems m_linearised;
    std::vector<SubcaseState> m_subcases;
    std::map<std::int32_t, std::size_t> m_subcaseIndex;
    std::vector<std::unique_ptr<SensitivitySystem>> m_systems;
    std::vector<std::unique_ptr<InertialRate>> m_reliefRates; // per subcase: the mass options of its relief
    std::unique_ptr<assembly::LoadVectorBuilder> m_loads;
    std::vector<std::unique_ptr<InertialRate>> m_inertialRates; // per subcase, made at first use
    std::unique_ptr<PseudoLoads> m_pseudoLoads;
    std::vector<ElementJob> m_jobs;
    std::vector<std::pair<ElementRef, std::size_t>> m_jobsByElement; // (element, job index), sorted by element
    std::vector<std::size_t> m_percentJobs;                          // the jobs with an ESE percentage, in order
    std::vector<SpcItem> m_spcItems;
    std::vector<MassItem> m_massItems;
    std::vector<SelfAdjointItem> m_selfAdjointItems;   // compliance, mean displacement
    std::vector<PointItem> m_pointItems;               // deflection over thickness
    std::map<std::int32_t, SectionRows> m_sectionRows; // by response ID
    std::vector<SectionItem> m_sectionItems;           // section and cut loads
    // The items of each section group and the groups of each element (indexSectionGroups, made at the first field).
    bool m_hasSectionGroups = false;
    std::map<std::tuple<std::int32_t, std::size_t, std::size_t>, std::vector<const SectionItem*>> m_sectionGroups;
    // The loads of each section or cut in a subcase and the largest products of its rows (lineariseSection), by
    // (response, index, subcase label): made once for its components.
    struct SectionValues
    {
        std::vector<double> loads; // bounded by a section's components
        double products = 0.0;
    };
    std::map<std::tuple<std::int32_t, std::size_t, std::int32_t>, SectionValues> m_sectionValues;
    std::map<ElementRef, std::vector<std::tuple<std::int32_t, std::size_t, std::size_t>>> m_sectionGroupsOf;
    std::unique_ptr<output::ZoneContour> m_zoneContour; // the zones on the design point's set
    std::vector<ZoneItem> m_zoneItems;                  // contour and external items
    // The contour forces of each zone and subcase at the design point, for the value checks of the contour items.
    std::map<std::pair<std::size_t, std::size_t>, std::vector<double>> m_zoneForces;
    std::map<std::pair<std::size_t, std::size_t>, std::vector<double>> m_zoneDisplacements;
    // The zones that are condensed units not recovered, by zone index: their force rows from K*.
    std::map<std::size_t, std::optional<UnitZoneForceRows>> m_unitZoneRows;
    // Per constraint set (SubcaseState::group): the rows of Gm by term DOF (mpcWeights).
    std::map<std::size_t, std::unordered_map<base::DofIndex, std::vector<std::pair<std::size_t, double>>>>
        m_gmTermIndex;
    std::unique_ptr<LinearityIndicator> m_linearity; // of the design point's snapshot, built on first use
    std::unique_ptr<EigenItems> m_eigen;             // the EIGN, FREQ, LAMA items, with eigenvalue solves
    std::unique_ptr<ModalItems> m_modal;             // displacement and recovered fields of vibration modes
    std::unique_ptr<TrimItems> m_trim;               // the items of the SAERO subcases
    bool m_keepsLoadParts = false;                   // a self-adjoint response is linearised
    bool m_preparesEverySubcase = false;             // set by the aids that read every analysed subcase
    // A linearisation at a perturbed state: the values recomputed are not the evaluation's (no value
    // check), and the displacements of a subcase are shifted by the g-set vector given for its label.
    bool m_checksValues = true;
    std::map<std::int32_t, std::vector<double>> m_displacementShifts;
    // A linearisation at a moved state under inertia relief: the relief of the moved
    // point is made from it (movedReliefOf, movedAccelerationsOf).
    std::optional<ReliefPoint> m_reliefPoint;
    std::vector<std::unique_ptr<solutions::KeptRelief>> m_movedReliefs; // by group, made at first use
    std::vector<std::vector<double>> m_reducedMassChanges;              // ΔR by group (movedReliefOf)
    std::vector<std::vector<double>> m_pointReducedMasses;              // the point's Dᵀ·M·D by group
    SensitivityReport m_report;
};

} // namespace oos::design::detail
