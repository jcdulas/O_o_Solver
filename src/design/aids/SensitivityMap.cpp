#include "oos/design/aids/SensitivityMap.hpp"

#include "../sensitivity/SensitivityComputation.hpp"
#include "oos/design/aids/ColumnPerturbation.hpp"
#include "oos/design/responses/ElementCatalog.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

constexpr std::array<FieldSide, FIELD_SIDE_COUNT> SIDES{FieldSide::Plus, FieldSide::Minus, FieldSide::HalfPlus,
                                                        FieldSide::HalfMinus};

[[nodiscard]] bool isMaterialIdColumn(const std::string& name)
{
    // matId, matId1 ... and the materials a shell names by their role (membraneMatId, bendingMatId, ...); not
    // matFrameId, a coordinate system.
    return name == "matId" || name.ends_with("MatId") ||
           (name.size() == 6 && name.starts_with("matId") && name[5] >= '1' && name[5] <= '9');
}

// The material IDs a row of a table names in its int32 columns "matId", "membraneMatId" ... (blank ones are null or 0).
void collectMaterialIds(const base::ContractReader& reader, std::int64_t row, std::set<std::int32_t>& matIds)
{
    for (const std::string& column : reader.table().columnNames())
    {
        if (!isMaterialIdColumn(column) || reader.columnType(column) != "int32")
            continue;
        const std::span<const std::int32_t> values = reader.values<std::int32_t>(column, base::NullPolicy::Allow);
        if (reader.validity(column).isValid(row) && values[static_cast<std::size_t>(row)] > 0)
            matIds.insert(values[static_cast<std::size_t>(row)]);
    }
}

// One element the map reaches: its place in the element set, the ID of the row of the column it reads, its labels.
struct MapElement
{
    ElementRef ref;
    std::int32_t key = 0;
    std::string card;
    std::int32_t elementId = 0;
    std::int32_t propId = 0;
    std::string propertyCard;
    // ∂K_e/∂v over the element's g-set DOFs (row-major), and per subcase the pseudo-load q_e,s = ∂P_e/∂v − ∂K_e/∂v·u_e
    // with the energy rate u_eᵀ·∂K_e/∂v·u_e and the load part ∂P_e/∂v alone (the thermal loads of the element).
    std::vector<base::DofIndex> dofs;
    std::vector<elements::ElementDof> elementDofs; // the same DOFs as the element gives them
    std::vector<double> stiffness;
    std::vector<SparseVector> loads;
    std::vector<SparseVector> loadParts;
    std::vector<double> energyRates;
};

// The map computation: the linearisation of StaticSensitivity for the response's items, then the element by element
// derivatives on a ColumnPerturbation, the adjoint vectors, and the products.
class MapComputation : public detail::SensitivityComputation
{
public:
    MapComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                   const SensitivityMapRequest& request, base::MemoryBudget& budget)
        : detail::SensitivityComputation(inputs, options, budget), m_request(request)
    {
    }

    SensitivityMapResult run()
    {
        if (hasExternalUnits())
            throw base::OptimisationError(
                "sensitivity map: the analysis has an external unit, whose zone has no elements of the model to map "
                "(its stiffness is the plugin's); make the map without the external unit");
        // Steps 1 of StaticSensitivity.hpp: the items with their weights and the explicit parts that do not depend
        // on a field; no designed quantity is differenced (the map's field is the model's, not a relation).
        selectResponses();
        initialiseItems();
        prepareSubcases();
        lineariseDesignResp1();
        lineariseElementJobs();
        lineariseEigenItems();
        compressExplicitParts();
        lineariseEquations();
        const auto position = m_positionOf.find(m_request.responseId);
        if (position == m_positionOf.end() || valuesAt(position->second).itemCount() == 0)
            throw base::OptimisationError(base::concatenate(
                {"the sensitivity map of response ", std::to_string(m_request.responseId),
                 ": expected a response evaluated at the design point with at least one item, found none; map a "
                 "response the analyses of the problem evaluate (its subcases and its analysis kind)"}));
        m_column =
            std::make_unique<ColumnPerturbation>(m_in.overlaid, m_request.card, m_request.field, m_registry, m_budget);
        collectElements();
        checkLoads();
        differenceElements();
        explicitParts();
        return assemble(position->second);
    }

private:
    [[nodiscard]] std::string fieldDescription() const
    {
        return base::concatenate({m_request.card, " '", m_request.field, "'"});
    }

    // The elements that read a row of the column, and the ID of that row.
    void collectElements()
    {
        const input::ModelSnapshot& snapshot = *m_in.overlaid;
        const ElementCatalog catalog(snapshot);
        const TargetKind kind = m_column->kind();
        for (std::size_t b = 0; b < m_set.batches().size(); ++b)
        {
            const assembly::AssemblyBatch& batch = m_set.batches()[b];
            for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            {
                const std::int32_t elementId = batch.view.elementId(row);
                const ElementEntry* entry = catalog.find(elementId);
                MapElement element;
                element.ref = {b, row};
                element.card = batch.view.card();
                element.elementId = elementId;
                if (entry != nullptr)
                {
                    element.propId = entry->propId;
                    element.propertyCard = entry->propertyCard;
                }
                std::optional<std::int32_t> key;
                if (kind == TargetKind::Element && element.card == m_request.card)
                    key = elementId;
                else if (kind == TargetKind::Property && element.propertyCard == m_request.card)
                    key = element.propId;
                else if (kind == TargetKind::Material)
                    key = materialOf(snapshot, batch, row, element);
                if (!key || !m_column->isMoved(*key))
                    continue;
                if (!batch.family->facets().has(elements::Facet::Derivatives))
                    throw base::OptimisationError(base::concatenate(
                        {"the sensitivity map of ", fieldDescription(), " reaches ", element.card, " ",
                         std::to_string(elementId),
                         ", whose element type has no design derivatives in this version; map a field of the element "
                         "types with the facet 'design'"}));
                element.key = *key;
                m_elements.push_back(std::move(element));
            }
        }
        for (std::size_t e = 0; e < m_elements.size(); ++e)
            m_elementOf[m_elements[e].ref] = e;
    }

    // The material ID of the map's card that an element reads (its property's MIDs, its own, the plies of its
    // laminate), or nothing; two of them are refused.
    std::optional<std::int32_t> materialOf(const input::ModelSnapshot& snapshot, const assembly::AssemblyBatch& batch,
                                           std::int64_t row, const MapElement& element) const
    {
        std::set<std::int32_t> matIds;
        collectMaterialIds(batch.view.columns(), row, matIds);
        if (!element.propertyCard.empty())
            if (const input::CardTable* properties = snapshot.findPropertyTable(element.propertyCard))
            {
                const std::span<const std::int32_t> ids = properties->ids();
                const auto found = std::find(ids.begin(), ids.end(), element.propId);
                if (found != ids.end())
                    collectMaterialIds(properties->reader(), found - ids.begin(), matIds);
            }
        if ((element.propertyCard == "CompProp" || element.propertyCard == "CompPropGlobal") &&
            snapshot.hasTable(input::TableRole::LaminatePlies))
        {
            const base::ContractReader& plies = snapshot.laminatePlies();
            const std::span<const std::int32_t> propIds = plies.values<std::int32_t>("propId");
            for (std::int64_t ply = 0; ply < plies.rowCount(); ++ply)
                if (propIds[static_cast<std::size_t>(ply)] == element.propId)
                    collectMaterialIds(plies, ply, matIds);
        }
        std::vector<std::int32_t> moved;
        for (const std::int32_t matId : matIds)
            if (m_column->isMoved(matId))
                moved.push_back(matId);
        if (moved.empty())
            return std::nullopt;
        if (moved.size() > 1)
            throw base::OptimisationError(base::concatenate(
                {"the sensitivity map of ", fieldDescription(), ": ", element.card, " ",
                 std::to_string(element.elementId), " reads ", std::to_string(moved.size()),
                 " materials of that card (", std::to_string(moved[0]), " and ", std::to_string(moved[1]),
                 "), so it has no single value of its own; map the field of each material by a design relation "
                 "instead"}));
        return moved.front();
    }

    // A field that moves the inertial or the distributed loads of a subcase cannot be mapped element by element here
    // (the inertial load of each element would be needed apart): refused, after a check that they do change.
    void checkLoads()
    {
        const elements::DesignedField designed{m_request.card, m_request.field};
        bool changesMass = false;
        std::set<std::int32_t> loadedElements;
        for (const MapElement& element : m_elements)
        {
            const elements::FieldUse use = m_set.batches()[element.ref.batch].family->designFieldUse(designed);
            changesMass = changesMass || use.changesMass;
            if (use.changesLoads || use.changesDistributedLoads)
                loadedElements.insert(element.elementId);
        }
        for (detail::SubcaseState& state : m_subcases)
        {
            if (!state.subcase.load)
                continue;
            const bool isUnderPressure =
                std::any_of(state.distributedElements.begin(), state.distributedElements.end(),
                            [&](std::int32_t elementId) { return loadedElements.count(elementId) != 0; });
            const bool movesInertial = state.hasInertialLoads && changesMass;
            const bool movesDistributed =
                state.hasDistributedLoads && isUnderPressure && distributedLoadsChange(*state.subcase.load);
            if (movesInertial || movesDistributed)
                throw base::OptimisationError(base::concatenate(
                    {"the sensitivity map of ", fieldDescription(), " in subcase ", std::to_string(state.subcase.label),
                     ": the field changes the ", movesInertial ? "inertial" : "distributed",
                     " loads of the subcase, whose share of each element this version does not form; take the "
                     "gradient of a design relation instead, or a subcase without those loads"}));
        }
    }

    [[nodiscard]] bool distributedLoadsChange(base::EntityId loadSet)
    {
        std::array<std::vector<double>, 2> vectors;
        for (std::size_t side = 0; side < 2; ++side)
        {
            base::DiagnosticSink sink;
            base::BudgetedBuffer<double> loads(m_budget, gSize(), "sensitivity map distributed loads");
            std::fill(loads.data(), loads.data() + gSize(), 0.0);
            const assembly::LoadVectorBuilder builder(m_column->set(SIDES[side]), m_dofMap, m_budget);
            builder.addFamilyLoads(loadSet, assembly::LoadFamily::Distributed, nullptr, loads.span(), m_budget, sink);
            vectors[side].assign(loads.data(), loads.data() + gSize());
        }
        return vectors[0] != vectors[1];
    }

    // ∂K_e/∂v and, per subcase, q_e,s for every element reached.
    void differenceElements()
    {
        const elements::DesignedField designed{m_request.card, m_request.field};
        for (MapElement& element : m_elements)
        {
            const assembly::AssemblyBatch& batch = m_set.batches()[element.ref.batch];
            const elements::ElementFamily& family = *batch.family;
            const elements::FieldUse use = family.designFieldUse(designed);
            const std::int64_t row = element.ref.row;
            const auto viewOf = [&](std::size_t side) -> const elements::ElementBatchView&
            { return m_column->set(SIDES[side]).batches()[element.ref.batch].view; };
            element.loads.assign(m_subcases.size(), {});
            element.loadParts.assign(m_subcases.size(), {});
            element.energyRates.assign(m_subcases.size(), 0.0);
            ++m_elementDerivatives;
            // The family's closed form of ∂K_e/∂v where it has one, as the gradients take it, so that the
            // map sums to them.
            elements::ElementMatrixBlock closedForm;
            const bool isClosedForm = use.changesStiffness && family.facets().has(elements::Facet::Stiffness) &&
                                      m_options.elementDerivatives == ElementDerivativeMethod::Analytic &&
                                      family.stiffnessFieldDerivative(batch.view, row, designed, closedForm) &&
                                      !closedForm.empty();
            if (isClosedForm)
            {
                for (const elements::ElementDof& dof : closedForm.dofs(0))
                {
                    element.dofs.push_back(m_dofMap.dofOf(dof));
                    element.elementDofs.push_back(dof);
                }
                element.stiffness.assign(closedForm.matrix(0).begin(), closedForm.matrix(0).end());
            }
            else if (use.changesStiffness && family.facets().has(elements::Facet::Stiffness))
            {
                std::array<elements::ElementMatrixBlock, FIELD_SIDE_COUNT> blocks;
                for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
                    family.computeStiffness(viewOf(side), {row, row + 1}, blocks[side]);
                if (!blocks[0].empty())
                {
                    for (const elements::ElementMatrixBlock& block : blocks)
                        if (block.size() != 1 || !std::equal(block.dofs(0).begin(), block.dofs(0).end(),
                                                             blocks[0].dofs(0).begin(), blocks[0].dofs(0).end()))
                            throw std::logic_error(base::concatenate(
                                {"the sensitivity map: the stiffness of ", element.card, " ",
                                 std::to_string(element.elementId), " changes its layout with the field"}));
                    for (const elements::ElementDof& dof : blocks[0].dofs(0))
                    {
                        element.dofs.push_back(m_dofMap.dofOf(dof));
                        element.elementDofs.push_back(dof);
                    }
                    element.stiffness.resize(blocks[0].matrix(0).size());
                    m_column->derivatives(element.key, blocks[0].matrix(0), blocks[1].matrix(0), blocks[2].matrix(0),
                                          blocks[3].matrix(0), element.stiffness);
                }
            }
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                const detail::SubcaseState& state = m_subcases[s];
                // On the condensed path the map reads the displacements of the element: its unit must have
                // been recovered by the analysis.
                if (const BoundaryReducedSensitivity* condensed = systemOf(state).condensed(); condensed != nullptr)
                    for (const base::DofIndex dof : element.dofs)
                        if (!condensed->isKnownDof(dof))
                            throw base::OptimisationError(base::concatenate(
                                {"the sensitivity map reads ", element.card, " ", std::to_string(element.elementId),
                                 ", an element of a condensed unit the analysis did not recover (the map needs its "
                                 "displacements); recover its unit (DesignProblem recover=) or use the full path"}));
                SparseVector& load = element.loads[s];
                const std::size_t n = element.dofs.size();
                for (std::size_t r = 0; r < n; ++r)
                {
                    double force = 0.0;
                    for (std::size_t c = 0; c < n; ++c)
                        force += element.stiffness[r * n + c] *
                                 state.displacements[static_cast<std::size_t>(element.dofs[c])];
                    load.emplace_back(element.dofs[r], -force);
                    element.energyRates[s] += state.displacements[static_cast<std::size_t>(element.dofs[r])] * force;
                }
                if (use.changesLoads && state.temperatures && !family.isThermallyInert() && !state.isCombination())
                    element.loadParts[s] = thermalLoadDerivative(family, element, viewOf, *state.temperatures);
                for (const auto& entry : element.loadParts[s])
                    load.push_back(entry);
                compressSparse(load);
            }
            // A SUBCOM: its stiffness term (above) and its terms' load parts.
            for (std::size_t s = 0; s < m_subcases.size(); ++s)
            {
                if (!m_subcases[s].isCombination())
                    continue;
                SparseVector parts;
                for (const auto& [term, coefficient] : m_subcases[s].terms)
                    for (const auto& [dof, value] : element.loadParts[term])
                        parts.emplace_back(dof, coefficient * value);
                if (parts.empty())
                    continue;
                compressSparse(parts);
                element.loadParts[s] = parts;
                for (const auto& entry : parts)
                    element.loads[s].push_back(entry);
                compressSparse(element.loads[s]);
            }
        }
    }

    template <typename ViewOf>
    [[nodiscard]] SparseVector thermalLoadDerivative(const elements::ElementFamily& family, const MapElement& element,
                                                     const ViewOf& viewOf,
                                                     const assembly::TemperatureField& temperatures) const
    {
        // The loads of each side summed DOF by DOF, in the order of the plus side (ElementDerivatives::thermalLoads).
        std::vector<base::DofIndex> dofs;
        std::array<std::vector<double>, FIELD_SIDE_COUNT> sums;
        for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
        {
            elements::ElementVectorBlock block;
            family.computeThermalLoads(viewOf(side), {element.ref.row, element.ref.row + 1},
                                       temperatures.temperatures(), block);
            for (std::size_t i = 0; i < block.size(); ++i)
                for (std::size_t d = 0; d < block.dofs(i).size(); ++d)
                {
                    const base::DofIndex dof = m_dofMap.dofOf(block.dofs(i)[d]);
                    auto found = std::find(dofs.begin(), dofs.end(), dof);
                    if (found == dofs.end())
                    {
                        dofs.push_back(dof);
                        for (std::vector<double>& sum : sums)
                            sum.push_back(0.0);
                        found = dofs.end() - 1;
                    }
                    sums[side][static_cast<std::size_t>(found - dofs.begin())] += block.values(i)[d];
                }
        }
        std::vector<double> derivative(dofs.size());
        m_column->derivatives(element.key, sums[0], sums[1], sums[2], sums[3], derivative);
        SparseVector result;
        for (std::size_t k = 0; k < dofs.size(); ++k)
            if (derivative[k] != 0.0)
                result.emplace_back(dofs[k], derivative[k]);
        compressSparse(result);
        return result;
    }

    // The explicit row of an item: one value per element of the map, its memory taken from the budget when the row
    // is made (items × elements in all: it grows with the model).
    [[nodiscard]] std::vector<double>& explicitOf(const ItemKey& key)
    {
        const auto [found, isNew] = m_explicit.try_emplace(key);
        if (isNew)
        {
            m_explicitReservations.push_back(
                m_budget.reserve(m_elements.size() * sizeof(double), "sensitivity map explicit parts"));
            found->second.assign(m_elements.size(), 0.0);
        }
        return found->second;
    }

    // The explicit part of every linearised item with respect to the value of each element reached (MapComputation
    // class comment), the equations last (in their dependency order).
    void explicitParts()
    {
        if (!m_sectionItems.empty())
            throw base::OptimisationError(
                "the sensitivity map of a section or cut load: its explicit part element by element is not "
                "formed in this version; map the element forces it combines instead (kind=\"force\")");
        elementItemParts();
        for (const detail::SpcItem& item : m_spcItems)
        {
            std::vector<double>& row = explicitOf(item.key);
            for (std::size_t e = 0; e < m_elements.size(); ++e)
                row[e] -= detail::dotSparseVectors(item.w, m_elements[e].loads[item.subcase]);
        }
        massItemParts();
        eigenItemParts();
        for (const detail::SelfAdjointItem& item : m_selfAdjointItems)
        {
            // ∂(scale·Pᵀu)/∂v_e = scale·u_eᵀ·∂P_e/∂v_e (the applied loads do not move: checkLoads).
            std::vector<double>& row = explicitOf(item.key);
            const detail::SubcaseState& state = m_subcases[item.subcase];
            for (std::size_t e = 0; e < m_elements.size(); ++e)
                row[e] += item.scale * dotSparse(m_elements[e].loadParts[item.subcase], state.displacements.span());
        }
        pointItemParts();
        for (const EquationResponse& equation : m_in.responses->equations())
        {
            const auto position = m_positionOf.find(equation.definition().id);
            if (position == m_positionOf.end())
                continue;
            for (std::size_t i = 0; i < m_linearised.items[position->second].size(); ++i)
            {
                const ItemKey key{position->second, i};
                const ItemLinearisation& item = m_linearised.at(key);
                if (!item.isWanted || !item.isEquation)
                    continue;
                // The rows of m_explicit keep their place when others are added (a std::map).
                std::vector<double>& row = explicitOf(key);
                std::fill(row.begin(), row.end(), 0.0);
                for (const auto& [argument, weight] : item.arguments)
                {
                    const std::vector<double>& source = explicitOf(argument);
                    for (std::size_t e = 0; e < m_elements.size(); ++e)
                        row[e] += weight * source[e];
                }
            }
        }
    }

    void elementItemParts()
    {
        for (detail::ElementJob& job : m_jobs)
        {
            const detail::SubcaseState& state = m_subcases[job.subcase];
            const auto found = m_elementOf.find(job.element);
            if (found != m_elementOf.end())
            {
                const MapElement& element = m_elements[found->second];
                const elements::NodalDisplacements displacements = state.nodal(m_dofMap);
                std::array<std::vector<double>, FIELD_SIDE_COUNT> sides;
                for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
                    sides[side] = jobValues(job, m_column->set(SIDES[side]).batches()[job.element.batch].view,
                                            m_column->snapshot(SIDES[side]), displacements);
                for (std::size_t k = 0; k < job.keys.size(); ++k)
                {
                    double derivative =
                        m_column->derivative(element.key, sides[0][k], sides[1][k], sides[2][k], sides[3][k]);
                    if (isPercent(job, k))
                        derivative = state.totalEnergy != 0.0 ? detail::PERCENT * derivative / state.totalEnergy : 0.0;
                    explicitOf(job.keys[k])[found->second] += derivative;
                }
            }
            // ESE item 3 is a share of the total energy T, which every element changes: −(100·E/T²)·½·u_eᵀ·∂K_e·u_e.
            for (std::size_t k = 0; k < job.keys.size(); ++k)
                if (isPercent(job, k) && state.totalEnergy != 0.0)
                {
                    std::vector<double>& row = explicitOf(job.keys[k]);
                    const double factor =
                        -detail::PERCENT * job.baseValues[k] / (state.totalEnergy * state.totalEnergy) * 0.5;
                    for (std::size_t e = 0; e < m_elements.size(); ++e)
                        row[e] += factor * m_elements[e].energyRates[job.subcase];
                }
        }
    }

    // EIGN, FREQ, LAMA (detail::EigenItems): per element φᵀ·(∂K_e − λ·∂M_e)·φ or λ·φᵀ·(∂K_e + λ·∂K_d,e)·φ
    // with the element's own value moved; the preload part of a LAMA comes from the adjoint of its weights (assemble).
    void eigenItemParts()
    {
        if (!m_eigen || m_eigen->isEmpty())
            return;
        const elements::DesignedField designed{m_request.card, m_request.field};
        for (std::size_t e = 0; e < m_elements.size(); ++e)
        {
            const MapElement& element = m_elements[e];
            const assembly::AssemblyBatch& batch = m_set.batches()[element.ref.batch];
            const elements::ElementFamily& family = *batch.family;
            const std::int64_t row = element.ref.row;
            const auto viewOf = [&](std::size_t side) -> const elements::ElementBatchView&
            { return m_column->set(SIDES[side]).batches()[element.ref.batch].view; };
            // A matrix facet on the four sides of the column, differenced for this element's value.
            const auto differenced = [&](const auto& compute)
            {
                std::array<elements::ElementMatrixBlock, FIELD_SIDE_COUNT> blocks;
                for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
                    compute(viewOf(side), blocks[side]);
                ElementMatrixDerivative derivative;
                if (blocks[0].empty())
                    return derivative;
                derivative.dofs.assign(blocks[0].dofs(0).begin(), blocks[0].dofs(0).end());
                derivative.values.resize(blocks[0].matrix(0).size());
                m_column->derivatives(element.key, blocks[0].matrix(0), blocks[1].matrix(0), blocks[2].matrix(0),
                                      blocks[3].matrix(0), derivative.values);
                return derivative;
            };
            ElementMatrixDerivative stiffness;
            stiffness.dofs = element.elementDofs;
            stiffness.values = element.stiffness;
            const auto massOf = [&](const elements::MassOptions& options)
            {
                return differenced([&](const elements::ElementBatchView& view, elements::ElementMatrixBlock& block)
                                   { family.computeMass(view, {row, row + 1}, options, block); });
            };
            const auto differentialOf = [&](const elements::NodalDisplacements& preload)
            {
                return differenced([&](const elements::ElementBatchView& view, elements::ElementMatrixBlock& block)
                                   { family.computeDifferentialStiffness(view, {row, row + 1}, preload, block); });
            };
            for (const auto& [key, derivative] : m_eigen->elementDerivatives(family, family.designFieldUse(designed),
                                                                             stiffness, massOf, differentialOf))
                explicitOf(key)[e] += derivative;
        }
    }

    // WEIGHT, VOLUME, FRMASS: the share of each element on the four sides (the mass is linear in the fields it
    // depends on: the extrapolation is exact).
    void massItemParts()
    {
        if (m_massItems.empty())
            return;
        for (const detail::MassItem& item : m_massItems)
        {
            const ResponseType type = item.response->definition().type;
            std::array<std::map<std::int32_t, double>, FIELD_SIDE_COUNT> sides;
            double denominator = 1.0;
            if (type == ResponseType::FractionalMass)
            {
                // FRMASS = m/m0, m0 = m / value at the design point.
                const double value = valuesAt(item.key.response).values[item.key.item];
                const double mass = MassProperties::rigidBodyMass(*m_in.overlaid, {0.0, 0.0, 0.0}, m_in.massFormulation,
                                                                  item.response->massFilter(), m_budget)(0, 0);
                denominator = value != 0.0 ? mass / value : 0.0;
            }
            for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
            {
                const std::shared_ptr<const input::ModelSnapshot>& snapshot = m_column->snapshot(SIDES[side]);
                if (type == ResponseType::Weight)
                {
                    const auto [rowEntry, columnEntry] = item.response->weightEntry();
                    sides[side] = MassProperties::elementRigidBodyMass(
                        *snapshot, DesignResp1Response::weightReference(*snapshot), m_in.massFormulation, {},
                        static_cast<std::size_t>(rowEntry - 1), static_cast<std::size_t>(columnEntry - 1), m_budget);
                }
                else if (type == ResponseType::Volume)
                    sides[side] = MassProperties::elementVolumes(*snapshot, snapshot, {}, m_budget);
                else
                    sides[side] = MassProperties::elementRigidBodyMass(*snapshot, {0.0, 0.0, 0.0}, m_in.massFormulation,
                                                                       item.response->massFilter(), 0, 0, m_budget);
            }
            std::vector<double>& row = explicitOf(item.key);
            const auto valueOf = [](const std::map<std::int32_t, double>& values, std::int32_t elementId)
            {
                const auto found = values.find(elementId);
                return found == values.end() ? 0.0 : found->second;
            };
            for (std::size_t e = 0; e < m_elements.size(); ++e)
            {
                const std::int32_t elementId = m_elements[e].elementId;
                const double derivative =
                    m_column->derivative(m_elements[e].key, valueOf(sides[0], elementId), valueOf(sides[1], elementId),
                                         valueOf(sides[2], elementId), valueOf(sides[3], elementId));
                row[e] += denominator != 0.0 ? derivative / denominator : 0.0;
            }
        }
    }

    // The deflection over thickness r = |n·u|/t̄, t̄ = (1/n_g)·Σ_k t_k over the shells around the grid point:
    // ∂r/∂v_e = −(r/t̄)·(1/n_g)·∂t_e/∂v_e for each shell e around it (the normals do not move with a field).
    void pointItemParts()
    {
        if (m_pointItems.empty())
            return;
        std::array<std::unique_ptr<LinearityIndicator>, FIELD_SIDE_COUNT> sides;
        for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
        {
            const std::shared_ptr<const input::ModelSnapshot>& snapshot = m_column->snapshot(SIDES[side]);
            sides[side] = std::make_unique<LinearityIndicator>(*snapshot, snapshot, m_budget);
        }
        std::array<std::unordered_map<std::int32_t, double>, FIELD_SIDE_COUNT> thickness;
        for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
            for (const ShellElementGeometry& element : sides[side]->shellElements())
                thickness[side][element.elementId] = element.thickness;
        std::unordered_map<std::int32_t, std::size_t> elementOfId;
        for (std::size_t e = 0; e < m_elements.size(); ++e)
            elementOfId[m_elements[e].elementId] = e;
        for (const detail::PointItem& item : m_pointItems)
        {
            const ShellPointGeometry* point = linearity().findShellPoint(item.grid);
            const double ratio = LinearityIndicator::deflectionRatio(*point, item.translations);
            const double share = ratio / point->thickness / static_cast<double>(point->elements.size());
            std::vector<double>& row = explicitOf(item.key);
            for (const std::size_t index : point->elements)
            {
                const std::int32_t elementId = linearity().shellElements()[index].elementId;
                const auto found = elementOfId.find(elementId);
                if (found == elementOfId.end())
                    continue;
                const MapElement& element = m_elements[found->second];
                const double rate =
                    m_column->derivative(element.key, thickness[0].at(elementId), thickness[1].at(elementId),
                                         thickness[2].at(elementId), thickness[3].at(elementId));
                row[found->second] -= share * rate;
            }
        }
    }

    [[nodiscard]] const detail::SelfAdjointItem* selfAdjointOf(const ItemKey& key) const
    {
        for (const detail::SelfAdjointItem& item : m_selfAdjointItems)
            if (item.key == key)
                return &item;
        return nullptr;
    }

    // The map of the response at `position`: λ_i of each item (solved per subcase by blocks, or scale·u), then
    // dr_i/dv_e = ∂r_i/∂v_e + λ_iᵀ·q_e; then the groups.
    SensitivityMapResult assemble(std::size_t position)
    {
        const ResponseValues& values = valuesAt(position);
        const std::size_t itemCount = values.itemCount();
        const std::size_t elementCount = m_elements.size();
        base::BudgetedBuffer<double> map(m_budget, itemCount * elementCount, "sensitivity map (items × elements)");
        for (std::size_t i = 0; i < itemCount; ++i)
        {
            const std::vector<double>& row = explicitOf({position, i});
            std::copy(row.begin(), row.end(), map.data() + i * elementCount);
        }
        std::int64_t solves = 0;
        std::int64_t selfAdjoint = 0;
        const auto addProducts = [&](std::size_t item, std::size_t subcase, std::span<const double> adjointG)
        {
            for (std::size_t e = 0; e < elementCount; ++e)
                map[item * elementCount + e] += dotSparse(m_elements[e].loads[subcase], adjointG);
        };
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            std::vector<std::size_t> solved; // the items of the subcase whose adjoint is solved for
            for (std::size_t i = 0; i < itemCount; ++i)
            {
                const ItemLinearisation& item = m_linearised.at({position, i});
                if (!item.readsDisplacements() || m_subcaseIndex.at(item.subcase) != s)
                    continue;
                const detail::SelfAdjointItem* self = selfAdjointOf({position, i});
                if (self != nullptr && m_request.usesSelfAdjoint)
                {
                    // λ = scale·u: no solve (SelfAdjointResponses).
                    base::BudgetedBuffer<double> adjoint(m_budget, gSize(), "sensitivity map adjoint vector",
                                                         base::FOR_OVERWRITE);
                    for (std::size_t dof = 0; dof < gSize(); ++dof)
                        adjoint[dof] = self->scale * m_subcases[s].displacements[dof];
                    addProducts(i, s, adjoint.span());
                    ++selfAdjoint;
                    continue;
                }
                solved.push_back(i);
            }
            if (solved.empty())
                continue;
            const SensitivitySystem& system = systemOf(m_subcases[s]);
            const auto equations = static_cast<std::size_t>(system.equationCount());
            base::BudgetedBuffer<double> vectorG(m_budget, gSize(), "sensitivity map adjoint vector");
            base::BudgetedBuffer<double> column(m_budget, equations, "sensitivity map equations");
            // The condensed path recovers the adjoint vectors in the units of the map's elements only.
            base::BudgetedBuffer<std::uint8_t> reads;
            if (system.isCondensed())
            {
                reads = base::BudgetedBuffer<std::uint8_t>(m_budget, gSize(), "sensitivity map reads");
                std::fill(reads.data(), reads.data() + gSize(), std::uint8_t{0});
                for (const MapElement& element : m_elements)
                    for (const base::DofIndex dof : element.dofs)
                        reads[static_cast<std::size_t>(dof)] = 1;
            }
            for (std::size_t first = 0; first < solved.size(); first += SENSITIVITY_BLOCK_COLUMNS)
            {
                const std::size_t count = std::min(SENSITIVITY_BLOCK_COLUMNS, solved.size() - first);
                linalg::DenseMatrix rightHandSides(m_budget, equations, count, "sensitivity map right-hand sides");
                for (std::size_t b = 0; b < count; ++b)
                {
                    std::fill(vectorG.data(), vectorG.data() + gSize(), 0.0);
                    for (const auto& [dof, weight] : m_linearised.at({position, solved[first + b]}).weights)
                        vectorG[static_cast<std::size_t>(dof)] = weight;
                    system.toEquations(vectorG.span(), column.span());
                    for (std::size_t e = 0; e < equations; ++e)
                        rightHandSides(e, b) = column[e];
                }
                if (equations > 0)
                    system.solve(rightHandSides.view(), reads.span());
                solves += static_cast<std::int64_t>(count);
                for (std::size_t b = 0; b < count; ++b)
                {
                    for (std::size_t e = 0; e < equations; ++e)
                        column[e] = rightHandSides(e, b);
                    system.fromEquations(column.span(), vectorG.span());
                    addProducts(solved[first + b], s, vectorG.span());
                }
            }
        }
        SensitivityMapResult result = group(map.span(), itemCount);
        result.responseId = values.id;
        result.items = values.items;
        result.solves = solves;
        result.selfAdjointItems = selfAdjoint;
        result.elementDerivatives = m_elementDerivatives;
        for (const std::unique_ptr<SensitivitySystem>& system : m_systems)
            result.factorisations += system->factorisationCount();
        return result;
    }

    // The columns of the map by group, the elements of a group summed in their order in the element set.
    SensitivityMapResult group(std::span<const double> map, std::size_t itemCount) const
    {
        const std::size_t elementCount = m_elements.size();
        std::vector<std::pair<std::string, std::int64_t>> labels(elementCount);
        if (m_request.grouping == MapGrouping::Element)
            for (std::size_t e = 0; e < elementCount; ++e)
                labels[e] = {m_elements[e].card, m_elements[e].elementId};
        else if (m_request.grouping == MapGrouping::Property)
            for (std::size_t e = 0; e < elementCount; ++e)
                labels[e] = {m_elements[e].propertyCard, m_elements[e].propId};
        else
        {
            const input::ModelSnapshot& snapshot = *m_in.overlaid;
            if (!snapshot.hasTable(input::TableRole::Components) ||
                !snapshot.hasTable(input::TableRole::ComponentElements))
                throw base::OptimisationError(base::concatenate(
                    {"the sensitivity map of ", fieldDescription(),
                     " by unit: the model was given without its unit tables (connectedComponents); take the "
                     "snapshot with units=True, or map by element or property"}));
            const input::UnitTable& units = snapshot.units();
            std::unordered_map<std::int32_t, std::int64_t> unitOf;
            for (std::size_t k = 0; k < units.elementIds().size(); ++k)
                unitOf[units.elementIds()[k]] = units.elementComponents()[k];
            for (std::size_t e = 0; e < elementCount; ++e)
            {
                const auto found = unitOf.find(m_elements[e].elementId);
                labels[e] = {"", found == unitOf.end() ? 0 : found->second};
            }
        }
        SensitivityMapResult result;
        std::map<std::pair<std::string, std::int64_t>, std::size_t> columnOf;
        std::vector<std::size_t> columnOfElement(elementCount);
        for (std::size_t e = 0; e < elementCount; ++e)
        {
            const auto [found, isNew] = columnOf.try_emplace(labels[e], result.groupIds.size());
            if (isNew)
            {
                result.groupCards.push_back(labels[e].first);
                result.groupIds.push_back(labels[e].second);
            }
            columnOfElement[e] = found->second;
        }
        const std::size_t groupCount = result.groupIds.size();
        result.values.assign(itemCount * groupCount, 0.0);
        for (std::size_t i = 0; i < itemCount; ++i)
            for (std::size_t e = 0; e < elementCount; ++e)
                result.values[i * groupCount + columnOfElement[e]] += map[i * elementCount + e];
        return result;
    }

    const SensitivityMapRequest& m_request;
    std::unique_ptr<ColumnPerturbation> m_column;
    std::vector<MapElement> m_elements;
    std::map<ElementRef, std::size_t> m_elementOf;
    std::map<ItemKey, std::vector<double>> m_explicit;
    std::vector<base::MemoryReservation> m_explicitReservations; // the memory of the rows of m_explicit
    std::int64_t m_elementDerivatives = 0;
};

} // namespace

std::string_view mapGroupingName(MapGrouping grouping) noexcept
{
    switch (grouping)
    {
    case MapGrouping::Element:
        return "element";
    case MapGrouping::Property:
        return "property";
    case MapGrouping::Unit:
        return "unit";
    }
    return "";
}

std::optional<MapGrouping> findMapGrouping(std::string_view name) noexcept
{
    for (const MapGrouping grouping : {MapGrouping::Element, MapGrouping::Property, MapGrouping::Unit})
        if (mapGroupingName(grouping) == name)
            return grouping;
    return std::nullopt;
}

SensitivityMapResult SensitivityMap::compute(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                             const SensitivityMapRequest& request, const CallableRegistry& callables,
                                             base::MemoryBudget& budget,
                                             const solutions::StaticOptions& analysisOptions, std::size_t blockItems)
{
    const ResponseSubcases subcases = problem.responseSubcases(evaluation);
    const StaticSensitivityInputs inputs =
        problem.sensitivityInputs(evaluation, subcases, callables, analysisOptions, blockItems);
    GradientOptions options;
    options.responseIds = {request.responseId};
    options.usesSelfAdjoint = false; // the map takes the self-adjoint vectors itself (assemble)
    MapComputation computation(inputs, options, request, budget);
    return computation.run();
}

} // namespace oos::design
