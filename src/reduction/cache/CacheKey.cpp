#include "oos/reduction/cache/CacheKey.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/cache/CacheHasher.hpp"

#include <algorithm>
#include <bit>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::reduction
{

namespace
{

void mixLabel(CacheHasher& hasher, const UnitDofLabel& label)
{
    hasher.mix(static_cast<std::uint64_t>(label.kind) | (static_cast<std::uint64_t>(label.side) << 8U) |
               (static_cast<std::uint64_t>(label.component) << 16U));
    hasher.mix(static_cast<std::int32_t>(label.id));
}

void mixMatrix(CacheHasher& hasher, linalg::ConstDenseMatrixView matrix)
{
    hasher.mix(static_cast<std::uint64_t>(matrix.rows));
    hasher.mix(static_cast<std::uint64_t>(matrix.columns));
    for (std::size_t j = 0; j < matrix.columns; ++j)
        for (std::size_t i = 0; i < matrix.rows; ++i)
            hasher.mix(matrix(i, j));
}

constexpr int GRID_DOF_COUNT = 6;

// The g-set DOF of a component (from 0) of a model node (UnitSystem::modelDofOfNode).
base::DofIndex modelDofOfNode(const assembly::DofMap& dofMap, std::int32_t node, int component) noexcept
{
    return node < dofMap.gridCount() ? dofMap.gridDof(node, component + 1)
                                     : dofMap.scalarDof(node - dofMap.gridCount());
}

std::string parameterText(const std::vector<CacheParameter>& parameters, std::string_view name)
{
    for (const CacheParameter& parameter : parameters)
        if (parameter.name == name)
            return parameter.value.empty() ? std::string("default") : parameter.value;
    return "default";
}

} // namespace

std::uint64_t CacheKey::hash() const noexcept
{
    CacheHasher hasher;
    hasher.mix(std::string_view(solverVersion));
    hasher.mix(std::string_view(kernel));
    hasher.mix(static_cast<std::uint64_t>(elementFormulation));
    hasher.mix(unitIdentity);
    hasher.mix(revision);
    hasher.mix(designStamp);
    hasher.mix(constraintStamp);
    hasher.mix(boundaryStamp);
    hasher.mix(spiderStamp);
    hasher.mix(static_cast<std::uint64_t>(reduction));
    hasher.mix(modeCount);
    hasher.mix(modeCutoff);
    hasher.mix(static_cast<std::uint64_t>(parameters.size()));
    for (const CacheParameter& parameter : parameters)
    {
        hasher.mix(std::string_view(parameter.name));
        hasher.mix(std::string_view(parameter.value));
    }
    hasher.mix(patternStamp);
    hasher.mix(inputStamp);
    return hasher.value();
}

std::string CacheKey::differenceFrom(const CacheKey& stored) const
{
    if (solverVersion != stored.solverVersion)
        return std::string("written by O_o_Solver ")
            .append(stored.solverVersion)
            .append(", this is O_o_Solver ")
            .append(solverVersion)
            .append(" (an entry of another version is never used)");
    if (kernel != stored.kernel)
        return std::string("condensed with the ")
            .append(stored.kernel)
            .append(" dense kernel, this run uses the ")
            .append(kernel)
            .append(" kernel (another kernel gives other bits)");
    if (elementFormulation != stored.elementFormulation)
        return std::string("condensed with element formulation ")
            .append(std::to_string(stored.elementFormulation))
            .append(", this O_o_Solver computes formulation ")
            .append(std::to_string(elementFormulation))
            .append(" (an element's stiffness, mass or loads changed since: every unit is condensed again)");
    if (parameters != stored.parameters)
    {
        // Name the first parameter whose value differs (both lists hold MATRIX_PARAMETERS and
        // OFFSET_DEFINITION_PARAMETER, by name; an entry of an older O_o_Solver lacks the latter).
        std::vector<std::string_view> names(MATRIX_PARAMETERS.begin(), MATRIX_PARAMETERS.end());
        names.push_back(OFFSET_DEFINITION_PARAMETER);
        for (const std::string_view name : names)
        {
            const std::string before = parameterText(stored.parameters, name);
            const std::string now = parameterText(parameters, name);
            if (before != now)
                return std::string("parameter ")
                    .append(name)
                    .append(" changed (")
                    .append(before)
                    .append(" -> ")
                    .append(now)
                    .append("): every unit is condensed again");
        }
        return "the parameters that change the matrices differ";
    }
    if (unitIdentity != stored.unitIdentity)
        return "another unit (other elements)";
    if (revision != stored.revision)
        return "the unit's revision stamp changed (an edit of its cards or of what they depend on, of an absorbed "
               "cluster or spider)";
    if (designStamp != stored.designStamp)
        return "the unit's design stamp changed (a design point changed a field or a grid point of the unit)";
    if (constraintStamp != stored.constraintStamp)
        return "the DOFs held in every subcase changed in the unit";
    if (boundaryStamp != stored.boundaryStamp)
        return "the unit's boundary changed";
    if (spiderStamp != stored.spiderStamp)
        return "the spiders absorbed into the unit changed (a decision or a weighting matrix)";
    if (reduction != stored.reduction || modeCount != stored.modeCount || modeCutoff != stored.modeCutoff)
        return "the reduction changed (static or Craig-Bampton, the modes asked for)";
    if (patternStamp != stored.patternStamp)
        return "the unit's load patterns or its mass changed (other loaded DOFs, other values of a basis of load sets "
               "or "
               "cases, or another mass for its inertial patterns)";
    if (inputStamp != stored.inputStamp)
        return "the unit's inputs changed while its stamps did not (its points' positions or displacement systems, "
               "the order of its elements, the relations inside it, the non-structural mass selected or the options "
               "of its assembly)";
    return {};
}

bool CacheKey::matches(const CacheKey& stored) const
{
    return solverVersion == stored.solverVersion && kernel == stored.kernel &&
           elementFormulation == stored.elementFormulation && unitIdentity == stored.unitIdentity &&
           revision == stored.revision && designStamp == stored.designStamp &&
           constraintStamp == stored.constraintStamp && boundaryStamp == stored.boundaryStamp &&
           spiderStamp == stored.spiderStamp && reduction == stored.reduction && modeCount == stored.modeCount &&
           modeCutoff == stored.modeCutoff && parameters == stored.parameters && patternStamp == stored.patternStamp &&
           inputStamp == stored.inputStamp;
}

std::uint64_t unitIdentityOf(std::span<const base::EntityId> sortedElementIds) noexcept
{
    CacheHasher hasher;
    hasher.mix(static_cast<std::uint64_t>(sortedElementIds.size()));
    for (const base::EntityId elementId : sortedElementIds)
        hasher.mix(static_cast<std::int32_t>(elementId));
    return hasher.value();
}

std::uint64_t unitInputStamp(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget)
{
    if (model.elements == nullptr || model.dofMap == nullptr || model.elementIndex == nullptr ||
        model.unitRelations == nullptr)
        throw std::invalid_argument(base::concatenate(
            {"unitInputStamp (unit ", std::to_string(description.unitId),
             "): the model needs its element set, DOF map, element index and the Gm of the units; give all four"}));
    const assembly::DofMap& dofMap = *model.dofMap;
    CacheHasher hasher;
    // The points, in the order of the unit's numbering: their nodes in the model (which number the local DOFs and order
    // the equations), IDs, positions and displacement systems (a shape overlay, a coordinate system), and per DOF
    // whether it is held in every subcase and the relation of the units' Gm that makes it dependent, with its terms.
    const base::BudgetedBuffer<std::int32_t> nodes = unitPointNodes(dofMap, description, budget);
    hasher.mix(static_cast<std::uint64_t>(nodes.size()));
    const assembly::RigidElimination& gm = *model.unitRelations;
    for (const std::int32_t node : nodes.span())
    {
        const bool isGrid = node < dofMap.gridCount();
        hasher.mix(node);
        hasher.mix(
            static_cast<std::int32_t>(isGrid ? dofMap.gridId(node) : dofMap.scalarId(node - dofMap.gridCount())));
        if (isGrid)
        {
            const base::Vec3 position = dofMap.positionInBasic(node);
            hasher.mix(position.x());
            hasher.mix(position.y());
            hasher.mix(position.z());
            for (const double value : dofMap.cdToBasic(node).values)
                hasher.mix(value);
        }
        for (int c = 0; c < (isGrid ? GRID_DOF_COUNT : 1); ++c)
        {
            const base::DofIndex dof = modelDofOfNode(dofMap, node, c);
            hasher.mix(static_cast<std::uint64_t>(!model.isConstrainedEverywhere.empty() &&
                                                  model.isConstrainedEverywhere[static_cast<std::size_t>(dof)] != 0));
            const std::optional<std::size_t> row = gm.findRow(dof);
            if (!row)
            {
                hasher.mix(std::uint64_t{0});
                continue;
            }
            hasher.mix(static_cast<std::uint64_t>(gm.row(*row).size()) + 1);
            hasher.mix(static_cast<std::uint64_t>(gm.sourceOf(*row)));
            hasher.mix(static_cast<std::int32_t>(gm.sourceIdOf(*row)));
            for (const assembly::ConstraintTerm& term : gm.row(*row))
            {
                const linalg::NodeDof nodeDof = dofMap.nodeDofOf(term.dof);
                hasher.mix(nodeDof.node);
                hasher.mix(nodeDof.component);
                hasher.mix(term.coefficient);
            }
        }
    }
    // The boundary points (the rows of K*).
    hasher.mix(static_cast<std::uint64_t>(description.boundaryNodeIds.size()));
    for (const base::EntityId nodeId : description.boundaryNodeIds)
        hasher.mix(static_cast<std::int32_t>(nodeId));
    // The elements, in the order of the model's sums (the batches and their rows: UnitSystem's assembly order), with
    // their cards and non-structural mass (the NSM set a subcase selects).
    hasher.mix(static_cast<std::uint64_t>(description.elementIds.size()));
    const std::span<const assembly::AssemblyBatch> batches = model.elements->batches();
    for (const base::EntityId elementId : description.elementIds)
    {
        hasher.mix(static_cast<std::int32_t>(elementId));
        const std::optional<ElementPlace> place = model.elementIndex->find(elementId);
        if (!place)
        {
            hasher.mix(std::uint64_t{0});
            continue;
        }
        hasher.mix(static_cast<std::uint64_t>(place->batchIndex) + 1);
        hasher.mix(place->row);
        hasher.mix(std::string_view(batches[place->batchIndex].view.card()));
        // The formulation its card is computed with (ElementRegistry::forModel): another one gives another matrix.
        hasher.mix(batches[place->batchIndex].family->formulation());
        const std::span<const double> shares = model.nonStructuralMass != nullptr
                                                   ? model.nonStructuralMass->sharesOf(place->batchIndex)
                                                   : std::span<const double>();
        hasher.mix(shares.empty() ? 0.0 : shares[static_cast<std::size_t>(place->row)]);
    }
    // The options of its assembly, its mass and its singularity processor (the parameters are in the key too, by name).
    hasher.mix(static_cast<std::uint64_t>(model.massOptions.formulation));
    hasher.mix(model.massOptions.weightToMass);
    hasher.mix(static_cast<std::uint64_t>(model.assembly.elementsPerPiece));
    hasher.mix(static_cast<std::uint64_t>(model.assembly.piecesPerWave));
    hasher.mix(static_cast<std::uint64_t>(model.singularity.isAutoSpcOn));
    hasher.mix(model.singularity.pivotZeroThreshold);
    hasher.mix(static_cast<std::uint64_t>(model.keepsMass));
    return hasher.value();
}

CacheKey makeUnitKey(const UnitKeySources& sources, base::MemoryBudget& budget)
{
    if (sources.plan == nullptr || sources.description == nullptr || sources.model == nullptr ||
        sources.model->dofMap == nullptr)
        throw std::invalid_argument(
            "makeUnitKey: the key of a unit needs its plan, its description and the model it is cut from (UnitModel)");
    const UnitPlan& plan = *sources.plan;
    const UnitDescription& description = *sources.description;
    const UnitModel& model = *sources.model;
    const assembly::DofMap& dofMap = *model.dofMap;
    CacheKey key;
    key.solverVersion = std::string(sources.solverVersion);
    key.kernel = std::string(sources.kernel);
    key.elementFormulation = sources.elementFormulation;
    key.unitId = plan.unitId;
    key.unitIdentity = unitIdentityOf(plan.elementIds);
    key.revision = plan.revision;
    key.designStamp = sources.designStamp;
    key.reduction = sources.reduction;
    if (sources.reduction == ReductionKind::CraigBampton)
    {
        key.modeCount = sources.modeCount;
        key.modeCutoff = sources.modeCount == CacheKey::MODE_CUTOFF_REQUEST
                             ? std::bit_cast<std::uint64_t>(sources.modeCutoff)
                             : std::uint64_t{0};
    }
    key.parameters.assign(sources.parameters.begin(), sources.parameters.end());

    // The DOFs held in every subcase, by label (before the singularity processor: its AUTOSPC follows from the rest).
    {
        CacheHasher hasher;
        const base::BudgetedBuffer<std::int32_t> nodes = unitPointNodes(dofMap, description, budget);
        if (!model.isConstrainedEverywhere.empty())
            for (const std::int32_t node : nodes.span())
            {
                const bool isGrid = node < dofMap.gridCount();
                for (int c = 0; c < (isGrid ? GRID_DOF_COUNT : 1); ++c)
                {
                    if (model.isConstrainedEverywhere[static_cast<std::size_t>(modelDofOfNode(dofMap, node, c))] == 0)
                        continue;
                    UnitDofLabel label;
                    label.kind = isGrid ? UnitPointKind::Grid : UnitPointKind::Scalar;
                    label.id = isGrid ? dofMap.gridId(node) : dofMap.scalarId(node - dofMap.gridCount());
                    label.component = isGrid ? static_cast<std::uint8_t>(c + 1) : std::uint8_t{0};
                    mixLabel(hasher, label);
                }
            }
        key.constraintStamp = hasher.value();
    }
    // The boundary points, and the generated points of the fastener sides (on the boundary without being listed).
    {
        CacheHasher hasher;
        hasher.mix(static_cast<std::uint64_t>(description.boundaryNodeIds.size()));
        for (const base::EntityId nodeId : description.boundaryNodeIds)
            hasher.mix(static_cast<std::int32_t>(nodeId));
        for (const UnitSpider& spider : description.spiders)
            if (spider.relation.referenceDofs.empty())
                mixLabel(hasher, {UnitPointKind::FastenerPoint, spider.key.elementId, spider.key.side, 1});
        key.boundaryStamp = hasher.value();
    }
    // The absorbed spiders: their keys and weighting matrices (the relations eliminated by change of basis).
    {
        CacheHasher hasher;
        hasher.mix(static_cast<std::uint64_t>(description.spiders.size()));
        for (const UnitSpider& spider : description.spiders)
        {
            hasher.mix(static_cast<std::int32_t>(spider.key.elementId));
            hasher.mix(static_cast<std::uint64_t>(spider.key.side));
            const SpiderWeightMatrix& weights = spider.relation.weights;
            hasher.mix(static_cast<std::uint64_t>(weights.referenceDofCount));
            hasher.mix(static_cast<std::uint64_t>(weights.legDofCount));
            hasher.mix(std::span<const double>(weights.values));
            hasher.mix(weights.characteristicLength);
            for (const base::DofIndex dof : spider.relation.referenceDofs)
                hasher.mix(static_cast<std::int64_t>(dof));
            for (const base::DofIndex dof : spider.relation.legDofs)
                hasher.mix(static_cast<std::int64_t>(dof));
            hasher.mix(spider.generatedPointPosition.x());
            hasher.mix(spider.generatedPointPosition.y());
            hasher.mix(spider.generatedPointPosition.z());
        }
        key.spiderStamp = hasher.value();
    }
    // The patterns: the basis' stamp and their values on the unit's DOFs.
    {
        CacheHasher hasher;
        hasher.mix(sources.patterns.stamp);
        mixMatrix(hasher, sources.patterns.loads);
        mixMatrix(hasher, sources.patterns.enforced);
        hasher.mix(static_cast<std::uint64_t>(sources.hasInertialPatterns));
        key.patternStamp = hasher.value();
    }
    key.inputStamp = unitInputStamp(model, description, budget);
    return key;
}

std::uint64_t equationStampOf(const UnitSystem& system, ReductionKind reduction)
{
    // The matrix condensed, in the order of its equations, and its trailing (boundary) count.
    CacheHasher hasher;
    const linalg::SymmetricCsc& matrix = system.matrix();
    hasher.mix(static_cast<std::int32_t>(matrix.dimension()));
    hasher.mix(static_cast<std::int32_t>(system.boundaryDofCount()));
    for (const std::int64_t offset : matrix.columnOffsets())
        hasher.mix(offset);
    for (const std::int32_t row : matrix.rowIndices())
        hasher.mix(row);
    for (const double value : matrix.values())
        hasher.mix(value);
    // Craig-Bampton: the mass its modes and reduced mass are made from.
    if (reduction == ReductionKind::CraigBampton && system.hasMass())
        for (const double value : system.mass().values())
            hasher.mix(value);
    return hasher.value();
}

} // namespace oos::reduction
