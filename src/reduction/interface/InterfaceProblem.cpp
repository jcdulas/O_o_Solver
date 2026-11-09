#include "oos/reduction/interface/InterfaceProblem.hpp"

#include "oos/assembly/singularities/SingularBlock.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/sparse/ContributionBlock.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"
#include "oos/reduction/condense/SymmetricBlockAccess.hpp"
#include "oos/reduction/plan/InterfaceEstimate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

constexpr int GRID_DOF_COUNT = 6;
constexpr int TRANSLATION_COUNT = 3;
constexpr std::size_t RIGID_BODY_MODE_COUNT = 6;
constexpr std::int32_t NOT_A_POINT = -1;
constexpr std::int32_t GENERATED = -1;
// The model node of a point of modal coordinates (a Craig-Bampton unit's q set): no point of the model.
constexpr std::int32_t MODAL = -2;

std::string spiderName(const SpiderKey& key)
{
    if (key.side == FastenerSide::None)
        return base::concatenate({"spider ", std::to_string(key.elementId)});
    return base::concatenate(
        {"fastener ", std::to_string(key.elementId), key.side == FastenerSide::A ? " side A" : " side B"});
}

// An array that grows by appends, its capacity reserved in the memory budget before it is allocated (as UnitSystem's):
// the node groups of the interface elements, whose sizes the element facets give.
template <typename T>
class GrowingArray
{
public:
    static constexpr std::size_t INITIAL_CAPACITY = 1024;

    GrowingArray(base::MemoryBudget& budget, std::string label) : m_budget(&budget), m_label(std::move(label)) {}

    void append(std::span<const T> values)
    {
        const std::size_t needed = m_values.size() + values.size();
        if (needed > m_values.capacity())
        {
            const std::size_t capacity = std::max({needed, 2 * m_values.capacity(), INITIAL_CAPACITY});
            base::MemoryReservation larger = m_budget->reserve(capacity * sizeof(T), m_label);
            m_values.reserve(capacity);
            m_reservation = std::move(larger);
        }
        m_values.insert(m_values.end(), values.begin(), values.end());
    }
    void push(const T& value) { append(std::span<const T>(&value, 1)); }
    [[nodiscard]] std::size_t size() const noexcept { return m_values.size(); }
    [[nodiscard]] std::span<const T> span() const noexcept { return m_values; }

private:
    base::MemoryBudget* m_budget;
    std::string m_label;
    base::MemoryReservation m_reservation;
    std::vector<T> m_values;
};

struct GroupLists
{
    GrowingArray<std::int64_t> offsets;
    GrowingArray<std::int32_t> nodes;

    explicit GroupLists(base::MemoryBudget& budget)
        : offsets(budget, "interface node groups"), nodes(budget, "interface node groups")
    {
        offsets.push(0);
    }
    void add(std::span<const std::int32_t> group)
    {
        nodes.append(group);
        offsets.push(static_cast<std::int64_t>(nodes.size()));
    }
    [[nodiscard]] linalg::NodeGroupList view() const { return {offsets.span(), nodes.span()}; }
};

void sortUnique(std::vector<std::int32_t>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

// The structure of a unit's K* by its points: the rows of K* of each of its points, and whether
// the block of two points holds a value other than zero. A unit whose interior has several connected parts (the
// dissection's units merge sibling subtrees) has a K* that is a union of cliques: its other blocks are exactly zero
// (no term of K_bb and no part's update matrix writes there: the trailing front starts from zeros), so the interface
// problem leaves them out of its pattern, as the plan's estimate does (InterfaceEstimate: a clique per connected part).
// A dense K* (every block written) keeps the clique of all its points, the pattern before the step.
struct KstarStructure
{
    std::vector<std::int32_t> points;     // the interface points of the unit, increasing (its boundary)
    std::vector<std::int32_t> pointOfRow; // per row of K*: its index in `points`
    std::vector<std::uint8_t> isCoupled;  // points × points, symmetric, the diagonal set
    bool isDense = true;

    [[nodiscard]] bool coupled(std::size_t a, std::size_t b) const noexcept
    {
        return isCoupled[a * points.size() + b] != 0;
    }
};

// rowPoints: the interface point of each row of K*.
KstarStructure kstarStructureOf(const linalg::DenseMatrix& stiffness, std::span<const std::int32_t> rowPoints)
{
    KstarStructure structure;
    structure.points.assign(rowPoints.begin(), rowPoints.end());
    sortUnique(structure.points);
    const std::size_t count = structure.points.size();
    structure.pointOfRow.resize(rowPoints.size());
    for (std::size_t i = 0; i < rowPoints.size(); ++i)
        structure.pointOfRow[i] =
            static_cast<std::int32_t>(std::lower_bound(structure.points.begin(), structure.points.end(), rowPoints[i]) -
                                      structure.points.begin());
    structure.isCoupled.assign(count * count, 0);
    for (std::size_t a = 0; a < count; ++a)
        structure.isCoupled[a * count + a] = 1;
    for (std::size_t j = 0; j < rowPoints.size(); ++j)
    {
        const auto b = static_cast<std::size_t>(structure.pointOfRow[j]);
        for (std::size_t i = 0; i < rowPoints.size(); ++i)
            if (stiffness(i, j) != 0.0)
            {
                const auto a = static_cast<std::size_t>(structure.pointOfRow[i]);
                structure.isCoupled[a * count + b] = 1;
                structure.isCoupled[b * count + a] = 1;
            }
    }
    structure.isDense = std::all_of(structure.isCoupled.begin(), structure.isCoupled.end(),
                                    [](std::uint8_t flag) { return flag != 0; });
    return structure;
}

void computeStiffnessOrMass(const assembly::AssemblyBatch& batch, elements::ElementRange range,
                            elements::ElementMatrixBlock& block)
{
    if (batch.family->facets().has(elements::Facet::Stiffness))
        batch.family->computeStiffness(batch.view, range, block);
    else if (batch.family->facets().has(elements::Facet::Mass))
        batch.family->computeMass(batch.view, range, elements::MassOptions{}, block);
}

// A pair (dependent point, point of one of its terms) of the relations.
struct PointPair
{
    std::int32_t point = 0;
    std::int32_t other = 0;

    friend constexpr bool operator==(const PointPair&, const PointPair&) = default;
};

} // namespace

// An absorbed spider on the local DOFs: W, its right inverse W⁺ (the change of basis of the unit with q = 0) and the
// local DOFs of its reference.
struct InterfaceProblem::LocalSpider
{
    SpiderKey key;
    std::size_t referenceCount = 0;       // m
    std::size_t legCount = 0;             // n
    std::vector<double> weights;          // W, m × n row-major (bounded by the spider)
    std::vector<double> rightInverse;     // W⁺, n × m row-major: u_legs = W⁺·u_ref
    std::vector<std::int64_t> references; // the local DOFs of W's rows
};

InterfaceProblem::~InterfaceProblem() = default;

InterfaceProblem::InterfaceProblem(const InterfaceInputs& inputs, const linalg::DenseKernels& kernels,
                                   base::MemoryBudget& budget, base::ThreadPool* pool,
                                   const base::CancellationToken* cancellation)
    : m_dofMap(inputs.dofMap), m_kernels(&kernels)
{
    if (inputs.elements == nullptr || inputs.dofMap == nullptr || inputs.elementIndex == nullptr ||
        inputs.relations == nullptr || inputs.modelSets == nullptr)
        throw std::invalid_argument(
            "InterfaceProblem: the inputs need the element set, the DOF map, the element index, the relations of "
            "the interface and the DOF sets of the full path; give all five");
    if (inputs.isInteriorNode.size() != static_cast<std::size_t>(inputs.dofMap->nodeCount()))
        throw std::invalid_argument(base::concatenate(
            {"InterfaceProblem: ", std::to_string(inputs.isInteriorNode.size()), " interior flags for ",
             std::to_string(inputs.dofMap->nodeCount()), " points of the model; give one flag per point"}));
    numberPoints(inputs, budget);
    mapUnits(inputs, budget);
    readRelations(inputs, budget);
    buildGraph(inputs, budget, pool);
    assemble(inputs, budget, pool, cancellation);
    checkSingularities(inputs, budget, pool);
    computeMassProperties(inputs, budget, pool);
    assembleSecond(inputs, budget, pool);
    order(inputs, kernels, budget, pool, cancellation);
}

// ---------------------------------------------------------------------------------------------------------------------
// Numbering

std::int32_t InterfaceProblem::pointOfLocalDof(std::int64_t localDof) const noexcept
{
    const std::int64_t* first = m_firstDof.data();
    return static_cast<std::int32_t>(std::upper_bound(first, first + m_firstDof.size(), localDof) - first) - 1;
}

linalg::NodeDof InterfaceProblem::nodeDofOfLocal(std::int64_t localDof) const noexcept
{
    const std::int32_t point = pointOfLocalDof(localDof);
    return {point, static_cast<std::int32_t>(localDof - m_firstDof[static_cast<std::size_t>(point)])};
}

std::optional<std::int64_t> InterfaceProblem::localDofOf(base::DofIndex dof) const noexcept
{
    const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(dof);
    const std::int32_t point = m_pointOfNode[static_cast<std::size_t>(nodeDof.node)];
    if (point == NOT_A_POINT)
        return std::nullopt;
    return m_firstDof[static_cast<std::size_t>(point)] + nodeDof.component;
}

std::optional<base::DofIndex> InterfaceProblem::modelDofOf(std::int64_t localDof) const noexcept
{
    const linalg::NodeDof nodeDof = nodeDofOfLocal(localDof);
    const std::int32_t node = m_points[static_cast<std::size_t>(nodeDof.node)].modelNode;
    if (node < 0)
        return std::nullopt; // a generated fastener point, or modal coordinates
    return node < m_dofMap->gridCount() ? m_dofMap->gridDof(node, nodeDof.component + 1)
                                        : m_dofMap->scalarDof(node - m_dofMap->gridCount());
}

base::GridComponent InterfaceProblem::nameOf(std::int64_t localDof) const
{
    const linalg::NodeDof nodeDof = nodeDofOfLocal(localDof);
    const InterfacePoint& point = m_points[static_cast<std::size_t>(nodeDof.node)];
    // Modal coordinates are named by their unit's ID and their mode (from 1).
    if (point.modelNode == MODAL)
        for (std::size_t u = 0; u + 1 < m_modalDofOffsets.size(); ++u)
            if (localDof >= m_modalDofOffsets[u] && localDof < m_modalDofOffsets[u + 1])
                return {point.label.id, static_cast<int>(localDof - m_modalDofOffsets[u] + 1)};
    // A generated fastener point is named by its fastener's ID (it has no point of the model).
    return {point.label.id, point.label.kind == UnitPointKind::Scalar ? 0 : nodeDof.component + 1};
}

void InterfaceProblem::numberPoints(const InterfaceInputs& inputs, base::MemoryBudget& budget)
{
    const assembly::DofMap& dofMap = *inputs.dofMap;
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    std::size_t modelPoints = 0;
    for (std::size_t node = 0; node < nodeCount; ++node)
        modelPoints += inputs.isInteriorNode[node] == 0 ? 1 : 0;
    std::size_t generated = 0;
    for (const InterfaceSpider& spider : inputs.spiders)
        generated += spider.referenceDofs.empty() ? 1 : 0;
    // The modal coordinates of the Craig-Bampton units: points of up to six coordinates, unit by unit.
    const auto modeCountOf = [&](std::size_t unit)
    {
        const ModalReduction* modal = inputs.units[unit].modal;
        return modal != nullptr ? modal->modeCount() : std::size_t{0};
    };
    const auto modalPointsOf = [&](std::size_t unit)
    { return (modeCountOf(unit) + GRID_DOF_COUNT - 1) / GRID_DOF_COUNT; };
    std::size_t modalPoints = 0;
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
        modalPoints += modalPointsOf(u);
    const std::size_t pointCount = modelPoints + generated + modalPoints;
    m_points = base::BudgetedBuffer<InterfacePoint>(budget, pointCount, "interface points");
    m_pointOfNode = base::BudgetedBuffer<std::int32_t>(budget, nodeCount, "interface point of node");
    m_firstDof = base::BudgetedBuffer<std::int64_t>(budget, pointCount + 1, "interface first DOFs");
    std::size_t next = 0;
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        if (inputs.isInteriorNode[node] != 0)
        {
            m_pointOfNode[node] = NOT_A_POINT;
            continue;
        }
        const auto modelNode = static_cast<std::int32_t>(node);
        const bool isGrid = modelNode < dofMap.gridCount();
        InterfacePoint& point = m_points[next];
        point.modelNode = modelNode;
        point.dofCount = isGrid ? GRID_DOF_COUNT : 1;
        point.label = isGrid ? UnitDofLabel{UnitPointKind::Grid, dofMap.gridId(modelNode), FastenerSide::None, 1}
                             : UnitDofLabel{UnitPointKind::Scalar, dofMap.scalarId(modelNode - dofMap.gridCount()),
                                            FastenerSide::None, 0};
        point.position = isGrid ? dofMap.positionInBasic(modelNode) : base::Vec3{};
        m_pointOfNode[node] = static_cast<std::int32_t>(next);
        ++next;
    }
    for (const InterfaceSpider& spider : inputs.spiders)
    {
        if (!spider.referenceDofs.empty())
            continue;
        InterfacePoint& point = m_points[next++];
        point.modelNode = GENERATED;
        point.dofCount = GRID_DOF_COUNT;
        point.label = UnitDofLabel{UnitPointKind::FastenerPoint, spider.key.elementId, spider.key.side, 1};
        point.position = spider.generatedPointPosition;
    }
    base::BudgetedBuffer<std::size_t> firstModalPoint(budget, inputs.units.size() + 1, "interface modal points");
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
    {
        firstModalPoint[u] = next;
        std::size_t remaining = modeCountOf(u);
        for (std::size_t k = 0; k < modalPointsOf(u); ++k)
        {
            InterfacePoint& point = m_points[next++];
            point.modelNode = MODAL;
            point.dofCount = static_cast<std::uint8_t>(std::min<std::size_t>(remaining, GRID_DOF_COUNT));
            remaining -= point.dofCount;
            point.label = UnitDofLabel{
                UnitPointKind::ModalCoordinate,
                static_cast<base::EntityId>(inputs.units[u].reduced != nullptr ? inputs.units[u].reduced->unitId() : 0),
                FastenerSide::None, 1};
        }
    }
    firstModalPoint[inputs.units.size()] = next;
    for (std::size_t p = 0; p < pointCount; ++p)
        m_firstDof[p + 1] = m_firstDof[p] + m_points[p].dofCount;
    m_modalDofOffsets = base::BudgetedBuffer<std::int64_t>(budget, inputs.units.size() + 1, "interface modal DOFs");
    for (std::size_t u = 0; u <= inputs.units.size(); ++u)
        m_modalDofOffsets[u] = m_firstDof[firstModalPoint[u]];

    // The spiders: W, W⁺ (the expansion of each reference DOF alone, q = 0) and their references, the legs sorted.
    std::size_t legCount = 0;
    for (const InterfaceSpider& spider : inputs.spiders)
        legCount += spider.legDofs.size();
    m_legs = base::BudgetedBuffer<LegEntry>(budget, legCount, "interface spider legs");
    std::size_t nextLeg = 0;
    std::size_t nextGenerated = modelPoints;
    for (std::size_t s = 0; s < inputs.spiders.size(); ++s)
    {
        const InterfaceSpider& spider = inputs.spiders[s];
        const SpiderWeightMatrix& weights = spider.weights;
        if (weights.legDofCount != spider.legDofs.size() ||
            (!spider.referenceDofs.empty() && weights.referenceDofCount != spider.referenceDofs.size()) ||
            (spider.referenceDofs.empty() && weights.referenceDofCount != static_cast<std::size_t>(GRID_DOF_COUNT)))
            throw std::invalid_argument(base::concatenate(
                {"InterfaceProblem: ", spiderName(spider.key), " has a W of ",
                 std::to_string(weights.referenceDofCount), " x ", std::to_string(weights.legDofCount), " for ",
                 std::to_string(spider.referenceDofs.size()), " reference and ", std::to_string(spider.legDofs.size()),
                 " leg DOFs; give one DOF per row and column (six rows for a generated point)"}));
        const SpiderChangeOfBasis changeOfBasis(weights, budget);
        if (changeOfBasis.isRankDeficient())
            throw std::invalid_argument(base::concatenate(
                {"InterfaceProblem: ", spiderName(spider.key),
                 " has a rank-deficient W; such a spider is kept on its legs (AbsorptionSafeguard; a bug of "
                 "O_o_Solver: report it)"}));
        auto local = std::make_unique<LocalSpider>();
        local->key = spider.key;
        local->referenceCount = weights.referenceDofCount;
        local->legCount = weights.legDofCount;
        local->weights = weights.values;
        local->rightInverse.assign(local->legCount * local->referenceCount, 0.0);
        std::vector<double> reference(local->referenceCount, 0.0);
        const std::vector<double> deformation(changeOfBasis.deformationDofCount(), 0.0);
        std::vector<double> legs(local->legCount, 0.0);
        for (std::size_t r = 0; r < local->referenceCount; ++r)
        {
            std::fill(reference.begin(), reference.end(), 0.0);
            reference[r] = 1.0;
            changeOfBasis.expand(reference, deformation, legs);
            for (std::size_t l = 0; l < local->legCount; ++l)
                local->rightInverse[l * local->referenceCount + r] = legs[l];
        }
        if (spider.referenceDofs.empty())
        {
            for (int c = 0; c < GRID_DOF_COUNT; ++c)
                local->references.push_back(m_firstDof[nextGenerated] + c);
            ++nextGenerated;
        }
        else
        {
            for (const base::DofIndex dof : spider.referenceDofs)
            {
                const std::optional<std::int64_t> reference = localDofOf(dof);
                if (!reference)
                    throw std::invalid_argument(base::concatenate(
                        {"InterfaceProblem: the reference point ", std::to_string(dofMap.pointIdOf(dof)), " of ",
                         spiderName(spider.key),
                         " is an interior point of a unit; an absorbed spider's reference is on its unit's boundary "
                         "(a bug of O_o_Solver: report it)"}));
                local->references.push_back(*reference);
            }
        }
        for (std::size_t l = 0; l < spider.legDofs.size(); ++l)
            m_legs[nextLeg++] = LegEntry{spider.legDofs[l], static_cast<std::int32_t>(s), static_cast<std::int32_t>(l)};
        m_spiders.push_back(std::move(local));
    }
    std::sort(m_legs.data(), m_legs.data() + m_legs.size(),
              [](const LegEntry& a, const LegEntry& b) { return a.dof < b.dof; });
    for (std::size_t k = 1; k < m_legs.size(); ++k)
        if (m_legs[k].dof == m_legs[k - 1].dof)
            throw std::invalid_argument(base::concatenate(
                {"InterfaceProblem: a DOF of point ", std::to_string(dofMap.pointIdOf(m_legs[k].dof)),
                 " is a leg of two absorbed spiders; such spiders are kept on their legs (SpiderAbsorption; a bug "
                 "of O_o_Solver: report it)"}));

    // The interface elements with a matrix, in the model's order of batches and rows.
    std::size_t places = 0;
    for (const base::EntityId elementId : inputs.elementIds)
        places += inputs.elementIndex->find(elementId) ? 1 : 0;
    m_elementPlaces = base::BudgetedBuffer<ElementPlace>(budget, places, "interface elements");
    std::size_t nextPlace = 0;
    for (const base::EntityId elementId : inputs.elementIds)
        if (const std::optional<ElementPlace> place = inputs.elementIndex->find(elementId))
            m_elementPlaces[nextPlace++] = *place;
    std::sort(m_elementPlaces.data(), m_elementPlaces.data() + places, [](const ElementPlace& a, const ElementPlace& b)
              { return a.batchIndex != b.batchIndex ? a.batchIndex < b.batchIndex : a.row < b.row; });
}

void InterfaceProblem::mapUnits(const InterfaceInputs& inputs, base::MemoryBudget& budget)
{
    std::size_t total = 0;
    for (const InterfaceUnit& unit : inputs.units)
    {
        if (unit.reduced == nullptr || unit.system == nullptr)
            throw std::invalid_argument(
                "InterfaceProblem: a condensed unit without its reduced unit or its system; give both");
        total += unit.reduced->boundaryDofCount();
    }
    m_unitDofOffsets = base::BudgetedBuffer<std::int64_t>(budget, inputs.units.size() + 1, "interface unit DOFs");
    m_unitDofs = base::BudgetedBuffer<std::int64_t>(budget, total, "interface unit DOFs");
    std::size_t next = 0;
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
    {
        const ReducedUnit& reduced = *inputs.units[u].reduced;
        m_unitDofOffsets[u] = static_cast<std::int64_t>(next);
        for (const UnitDofLabel& label : reduced.labels())
        {
            std::optional<std::int64_t> local;
            if (label.kind == UnitPointKind::FastenerPoint)
            {
                for (std::size_t p = 0; p < m_points.size() && !local; ++p)
                {
                    const UnitDofLabel& name = m_points[p].label;
                    if (m_points[p].modelNode == GENERATED && name.id == label.id && name.side == label.side)
                        local = m_firstDof[p] + label.component - 1;
                }
            }
            else if (const std::optional<base::DofIndex> dof =
                         inputs.dofMap->findDof(label.id, label.kind == UnitPointKind::Scalar ? 0 : label.component))
            {
                local = localDofOf(*dof);
            }
            if (!local)
                throw std::invalid_argument(base::concatenate(
                    {"InterfaceProblem: unit ", std::to_string(reduced.unitId()), " has a boundary DOF (point ",
                     std::to_string(label.id), ", component ", std::to_string(label.component),
                     ") that is not a DOF of the interface problem; the units and the interface must come from one "
                     "plan"}));
            m_unitDofs[next++] = *local;
        }
    }
    m_unitDofOffsets[inputs.units.size()] = static_cast<std::int64_t>(next);

    // The external units: their contour DOFs are DOFs of interface points.
    std::size_t externalTotal = 0;
    for (const InterfaceExternalBlock& block : inputs.externalBlocks)
    {
        if (block.stiffness == nullptr || block.stiffness->rows() != block.dofs.size() ||
            block.stiffness->columns() != block.dofs.size())
            throw std::invalid_argument(base::concatenate(
                {"InterfaceProblem: the external unit of zone ", std::to_string(block.zoneId),
                 " gives no stiffness or one that is not square on its ", std::to_string(block.dofs.size()),
                 " contour DOFs; make the unit give a square stiffness on its contour"}));
        externalTotal += block.dofs.size();
    }
    m_externalDofOffsets =
        base::BudgetedBuffer<std::int64_t>(budget, inputs.externalBlocks.size() + 1, "interface external DOFs");
    m_externalDofs = base::BudgetedBuffer<std::int64_t>(budget, externalTotal, "interface external DOFs");
    std::size_t nextExternal = 0;
    for (std::size_t b = 0; b < inputs.externalBlocks.size(); ++b)
    {
        const InterfaceExternalBlock& block = inputs.externalBlocks[b];
        m_externalDofOffsets[b] = static_cast<std::int64_t>(nextExternal);
        for (const base::DofIndex dof : block.dofs)
        {
            const std::optional<std::int64_t> local = localDofOf(dof);
            if (!local)
                throw std::invalid_argument(base::concatenate(
                    {"InterfaceProblem: the external unit of zone ", std::to_string(block.zoneId),
                     " has a contour DOF of point ", std::to_string(inputs.dofMap->pointIdOf(dof)),
                     ", an interior point of a condensed unit; the units holding the zone and its contour are kept "
                     "full (a bug of O_o_Solver: report it)"}));
            m_externalDofs[nextExternal++] = *local;
        }
    }
    m_externalDofOffsets[inputs.externalBlocks.size()] = static_cast<std::int64_t>(nextExternal);
    // A unit's modal coordinates stand at the centre of its boundary points (the positions order the separators of the
    // nested dissection; the coordinates have no place of their own).
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
    {
        if (modalDofCount(u) == 0)
            continue;
        if (inputs.units[u].modal->boundaryDofCount() != inputs.units[u].reduced->boundaryDofCount())
            throw std::invalid_argument(base::concatenate(
                {"InterfaceProblem: unit ", std::to_string(inputs.units[u].reduced->unitId()), " has a q set of ",
                 std::to_string(inputs.units[u].modal->boundaryDofCount()), " boundary DOFs for a K* of ",
                 std::to_string(inputs.units[u].reduced->boundaryDofCount()),
                 "; the modal part must be the unit's own (CraigBampton::reduceModes)"}));
        base::Vec3 centre{};
        std::vector<std::int32_t> boundaryPoints; // bounded by the unit's boundary
        for (const std::int64_t local : unitLocalDofs(u))
            boundaryPoints.push_back(pointOfLocalDof(local));
        sortUnique(boundaryPoints);
        for (const std::int32_t point : boundaryPoints)
            centre = centre + m_points[static_cast<std::size_t>(point)].position;
        if (!boundaryPoints.empty())
            centre = centre * (1.0 / static_cast<double>(boundaryPoints.size()));
        for (std::int32_t point = pointOfLocalDof(firstModalDof(u));
             point <= pointOfLocalDof(firstModalDof(u) + modalDofCount(u) - 1); ++point)
            m_points[static_cast<std::size_t>(point)].position = centre;
    }
}

void InterfaceProblem::readRelations(const InterfaceInputs& inputs, base::MemoryBudget& budget)
{
    const assembly::RigidElimination& gm = *inputs.relations;
    const assembly::DofMap& dofMap = *inputs.dofMap;
    std::size_t termCount = 0;
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
        termCount += gm.row(i).size();
    m_relations = LocalRelations(gm.rowCount(), termCount, budget, "interface Gm rows");
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    m_role = base::BudgetedBuffer<std::uint8_t>(budget, dofCount, "interface DOF roles");
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
    {
        const base::DofIndex dependent = gm.mDofs()[i];
        const std::optional<std::int64_t> local = localDofOf(dependent);
        if (!local)
            throw std::invalid_argument(base::concatenate(
                {"InterfaceProblem: ", std::string(assembly::cardNameOf(gm.sourceOf(i))), " ",
                 std::to_string(gm.sourceIdOf(i)), " makes a DOF of point ",
                 std::to_string(dofMap.pointIdOf(dependent)),
                 " dependent, an interior point of a unit; a relation inside a unit is eliminated in it (leave it "
                 "out of the interface's relations)"}));
        m_relations.appendRow(*local);
        m_role[static_cast<std::size_t>(*local)] = DEPENDENT;
        for (const assembly::ConstraintTerm& term : gm.row(i))
        {
            const std::optional<std::int64_t> independent = localDofOf(term.dof);
            if (!independent)
                throw std::invalid_argument(base::concatenate(
                    {"InterfaceProblem: ", std::string(assembly::cardNameOf(gm.sourceOf(i))), " ",
                     std::to_string(gm.sourceIdOf(i)), " makes point ", std::to_string(dofMap.pointIdOf(dependent)),
                     " depend on point ", std::to_string(dofMap.pointIdOf(term.dof)),
                     ", an interior point of a unit; the plan puts the points of the interface's relations on the "
                     "boundaries (BoundaryPromotion; a bug of O_o_Solver: report it)"}));
            m_relations.appendTerm(LocalTerm{*independent, term.coefficient});
        }
    }
    // The held DOFs of the full path's sets (s and r): the subcases' constraints and the supports; or, with conditioned
    // supports, those in place of the r set.
    const bool isConditioned = !inputs.conditionedSupports.empty();
    for (std::size_t local = 0; local < dofCount; ++local)
    {
        const std::optional<base::DofIndex> dof = modelDofOf(static_cast<std::int64_t>(local));
        if (!dof || m_role[local] == DEPENDENT)
            continue;
        const assembly::ExclusiveSet set = inputs.modelSets->exclusiveSetOf(*dof);
        if (set == assembly::ExclusiveSet::Sb || set == assembly::ExclusiveSet::Sg ||
            (set == assembly::ExclusiveSet::R && !isConditioned))
            m_role[local] = HELD;
    }
    for (const base::DofIndex dof : inputs.conditionedSupports)
    {
        const std::optional<std::int64_t> local = localDofOf(dof);
        if (!local || m_role[static_cast<std::size_t>(*local)] == DEPENDENT)
            throw std::invalid_argument(base::concatenate(
                {"InterfaceProblem: the conditioned support DOF of point ", std::to_string(dofMap.pointIdOf(dof)),
                 " is not an independent DOF of the interface; choose the supports among its free DOFs (a bug of "
                 "O_o_Solver: report it)"}));
        m_role[static_cast<std::size_t>(*local)] = HELD;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Matrices

template <typename Facet>
void InterfaceProblem::addElementContributions(const InterfaceInputs& inputs, std::size_t begin, std::size_t end,
                                               Facet&& facet, bool isRaw,
                                               linalg::ContributionBlock& contributions) const
{
    const assembly::DofMap& dofMap = *inputs.dofMap;
    elements::ElementMatrixBlock block;
    // Per element (bounded by an element): the column of each element DOF in the carried matrix P (|D| × |D'|) and the
    // carried DOFs D'.
    std::vector<std::int64_t> carriedDofs;
    std::vector<double> carry;
    std::vector<double> product;
    std::vector<double> carried;
    std::vector<linalg::NodeDof> nodeDofs;
    const auto columnOf = [&](std::int64_t localDof)
    {
        const auto found = std::find(carriedDofs.begin(), carriedDofs.end(), localDof);
        if (found != carriedDofs.end())
            return static_cast<std::size_t>(found - carriedDofs.begin());
        carriedDofs.push_back(localDof);
        return carriedDofs.size() - 1;
    };
    std::size_t k = begin;
    while (k < end)
    {
        std::size_t last = k + 1;
        while (last < end && m_elementPlaces[last].batchIndex == m_elementPlaces[k].batchIndex &&
               m_elementPlaces[last].row == m_elementPlaces[last - 1].row + 1)
            ++last;
        const assembly::AssemblyBatch& batch = inputs.elements->batches()[m_elementPlaces[k].batchIndex];
        block.clear();
        facet(static_cast<std::size_t>(m_elementPlaces[k].batchIndex), batch,
              elements::ElementRange{m_elementPlaces[k].row, m_elementPlaces[last - 1].row + 1}, block);
        for (std::size_t e = 0; e < block.size(); ++e)
        {
            const std::span<const elements::ElementDof> dofs = block.dofs(e);
            const std::span<const double> matrix = block.matrix(e);
            const std::size_t n = dofs.size();
            // The carried columns: an interface DOF is itself; a leg DOF of an absorbed spider its W⁺ row onto the
            // reference DOFs; a DOF of an interior point with no term is dropped (the rotations of the grid points
            // of a fastener's pierced element, whose interpolation reads the translations only).
            carriedDofs.clear();
            std::vector<std::int32_t> spiderOfRow(n, -1); // bounded by an element
            std::vector<std::int32_t> slotOfRow(n, -1);
            for (std::size_t i = 0; i < n; ++i)
            {
                const base::DofIndex dof = dofMap.dofOf(dofs[i]);
                if (const std::optional<std::int64_t> local = localDofOf(dof))
                {
                    columnOf(*local);
                    continue;
                }
                const LegEntry* leg =
                    std::lower_bound(m_legs.data(), m_legs.data() + m_legs.size(), dof,
                                     [](const LegEntry& entry, base::DofIndex d) { return entry.dof < d; });
                const bool isLeg = leg != m_legs.data() + m_legs.size() && leg->dof == dof;
                bool isZero = true;
                for (std::size_t j = 0; j < n && isZero; ++j)
                    isZero = matrix[i * n + j] == 0.0;
                if (!isLeg && !isZero)
                    throw std::invalid_argument(base::concatenate(
                        {"InterfaceProblem: element ", std::to_string(batch.view.elementId(block.elementRow(e))), " (",
                         batch.view.card(), ") acts on point ", std::to_string(dofMap.pointIdOf(dof)),
                         ", an interior point of a unit that is no leg of an absorbed spider; an element of the "
                         "interface acts on interface points (a bug of O_o_Solver: report it)"}));
                if (!isLeg || isRaw)
                    continue;
                spiderOfRow[i] = leg->spider;
                slotOfRow[i] = leg->slot;
                for (const std::int64_t reference : m_spiders[static_cast<std::size_t>(leg->spider)]->references)
                    columnOf(reference);
            }
            const std::size_t carriedCount = carriedDofs.size();
            if (carriedCount == 0)
                continue;
            carry.assign(n * carriedCount, 0.0);
            for (std::size_t i = 0; i < n; ++i)
            {
                const base::DofIndex dof = dofMap.dofOf(dofs[i]);
                if (const std::optional<std::int64_t> local = localDofOf(dof))
                {
                    carry[i * carriedCount + columnOf(*local)] = 1.0;
                    continue;
                }
                if (spiderOfRow[i] < 0)
                    continue;
                const LocalSpider& spider = *m_spiders[static_cast<std::size_t>(spiderOfRow[i])];
                for (std::size_t r = 0; r < spider.referenceCount; ++r)
                    carry[i * carriedCount + columnOf(spider.references[r])] =
                        spider.rightInverse[static_cast<std::size_t>(slotOfRow[i]) * spider.referenceCount + r];
            }
            // The rows of the element on a spider's legs lie in the row space of W (InterfaceSpider): their part
            // outside it, (I − (W⁺·W)ᵀ)·K_L, is round-off.
            if (!isRaw)
            {
                double largest = 0.0;
                for (const double value : matrix)
                    largest = std::max(largest, std::abs(value));
                // PERFORMANCE: the spiders whose legs the element's DOFs reach, in increasing order (the spiders it
                // does not reach have no row of it), where every spider of the interface was visited for every
                // element: O(elements × spiders), a vector allocated per pair. The same checks in the same order.
                // Measured: 3 000 plates of 3.1 M DOFs, the interface's graph and
                // assembly 9.1 → 0.1 s.
                std::vector<std::int32_t> touchedSpiders; // bounded by an element
                for (const std::int32_t s : spiderOfRow)
                    if (s >= 0)
                        touchedSpiders.push_back(s);
                std::sort(touchedSpiders.begin(), touchedSpiders.end());
                touchedSpiders.erase(std::unique(touchedSpiders.begin(), touchedSpiders.end()), touchedSpiders.end());
                for (const std::int32_t touched : touchedSpiders)
                {
                    const auto s = static_cast<std::size_t>(touched);
                    const LocalSpider& spider = *m_spiders[s];
                    std::vector<double> legRows(spider.legCount * n, 0.0); // Y: W's columns × element DOFs
                    for (std::size_t i = 0; i < n; ++i)
                        if (spiderOfRow[i] == touched)
                            for (std::size_t j = 0; j < n; ++j)
                                legRows[static_cast<std::size_t>(slotOfRow[i]) * n + j] = matrix[i * n + j];
                    // Z = W⁺ᵀ·Y (m × n), then Y − Wᵀ·Z.
                    std::vector<double> z(spider.referenceCount * n, 0.0);
                    for (std::size_t r = 0; r < spider.referenceCount; ++r)
                        for (std::size_t l = 0; l < spider.legCount; ++l)
                        {
                            const double factor = spider.rightInverse[l * spider.referenceCount + r];
                            if (factor == 0.0)
                                continue;
                            for (std::size_t j = 0; j < n; ++j)
                                z[r * n + j] += factor * legRows[l * n + j];
                        }
                    double outside = 0.0;
                    for (std::size_t l = 0; l < spider.legCount; ++l)
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            double projected = 0.0;
                            for (std::size_t r = 0; r < spider.referenceCount; ++r)
                                projected += spider.weights[r * spider.legCount + l] * z[r * n + j];
                            outside = std::max(outside, std::abs(legRows[l * n + j] - projected));
                        }
                    if (outside > FASTENER_FACTOR_TOLERANCE * largest)
                        throw std::logic_error(base::concatenate(
                            {"InterfaceProblem: element ", std::to_string(batch.view.elementId(block.elementRow(e))),
                             " (", batch.view.card(), ") acts on the legs of ", spiderName(spider.key),
                             " outside the relation of its spider (", std::to_string(outside / std::max(largest, 1.0)),
                             " of its largest term): its stiffness would be lost by the change of basis; keep the "
                             "spider on its legs (plan with absorb_spiders=False) and report it"}));
                }
            }
            // K' = Pᵀ·K·P.
            product.assign(n * carriedCount, 0.0);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                {
                    const double kij = matrix[i * n + j];
                    if (kij == 0.0)
                        continue;
                    for (std::size_t c = 0; c < carriedCount; ++c)
                        product[i * carriedCount + c] += kij * carry[j * carriedCount + c];
                }
            carried.assign(carriedCount * carriedCount, 0.0);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t a = 0; a < carriedCount; ++a)
                {
                    const double pia = carry[i * carriedCount + a];
                    if (pia == 0.0)
                        continue;
                    for (std::size_t b = 0; b < carriedCount; ++b)
                        carried[a * carriedCount + b] += pia * product[i * carriedCount + b];
                }
            // Exactly symmetric (the mean of the two triangles, in a fixed order).
            for (std::size_t a = 0; a < carriedCount; ++a)
                for (std::size_t b = a + 1; b < carriedCount; ++b)
                {
                    const double mean = 0.5 * (carried[a * carriedCount + b] + carried[b * carriedCount + a]);
                    carried[a * carriedCount + b] = mean;
                    carried[b * carriedCount + a] = mean;
                }
            nodeDofs.clear();
            for (const std::int64_t local : carriedDofs)
                nodeDofs.push_back(nodeDofOfLocal(local));
            contributions.append(nodeDofs, carried);
        }
        k = last;
    }
}

void InterfaceProblem::buildGraph(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool)
{
    // 1. The points each element couples once carried (the stiffness, else the mass).
    GroupLists elementGroups(budget);
    {
        linalg::ContributionBlock contributions;
        std::vector<std::int32_t> group; // bounded by an element
        constexpr std::size_t PIECE = 256;
        for (std::size_t begin = 0; begin < m_elementPlaces.size(); begin += PIECE)
        {
            contributions.clear();
            addElementContributions(
                inputs, begin, std::min(begin + PIECE, m_elementPlaces.size()),
                [](std::size_t, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                   elements::ElementMatrixBlock& block) { computeStiffnessOrMass(batch, range, block); },
                false, contributions);
            for (std::size_t c = 0; c < contributions.size(); ++c)
            {
                group.clear();
                for (const linalg::NodeDof& dof : contributions.dofs(c))
                    group.push_back(dof.node);
                sortUnique(group);
                if (group.size() > 1)
                    elementGroups.add(group);
            }
        }
    }
    // 2. Each condensed unit's K*: one clique on its boundary points, and its modal coordinates (M_CB couples them with
    // the boundary); the same for each external unit's K*. A boundary point whose every DOF is held in every subcase
    // (the leg of a relation of the interface on a clamped edge: BoundaryPromotion) brings no DOF to K*, but its raw
    // blocks with the other boundary points enter the singularity check (checkSingularities): it is in the clique too.
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
    {
        // Step 10.25: a unit of a static analysis whose K* is not dense brings the pairs of points of its
        // blocks written (KstarStructure) and of its raw boundary blocks; a dense one, or one with modal coordinates,
        // one clique.
        if (modalDofCount(u) == 0 && inputs.second == InterfaceSecondMatrix::None)
        {
            std::vector<std::int32_t> rowPoints; // bounded by the unit's boundary
            for (const std::int64_t local : unitLocalDofs(u))
                rowPoints.push_back(pointOfLocalDof(local));
            const KstarStructure structure = kstarStructureOf(inputs.units[u].reduced->stiffness(), rowPoints);
            if (!structure.isDense)
            {
                std::array<std::int32_t, 2> pair{};
                const std::size_t count = structure.points.size();
                for (std::size_t a = 0; a < count; ++a)
                    for (std::size_t b = a + 1; b < count; ++b)
                        if (structure.coupled(a, b))
                        {
                            pair = {structure.points[a], structure.points[b]};
                            elementGroups.add(pair);
                        }
                for (const RawBoundaryBlock& block : inputs.units[u].system->rawBoundaryBlocks())
                {
                    const std::int32_t row = m_pointOfNode[static_cast<std::size_t>(block.rowNode)];
                    const std::int32_t column = m_pointOfNode[static_cast<std::size_t>(block.columnNode)];
                    if (row != NOT_A_POINT && column != NOT_A_POINT && row != column)
                    {
                        pair = {std::min(row, column), std::max(row, column)};
                        elementGroups.add(pair);
                    }
                }
                continue;
            }
        }
        std::vector<std::int32_t> clique; // bounded by the unit's boundary (the K* front) and its modal points
        for (const std::int64_t local : unitLocalDofs(u))
            clique.push_back(pointOfLocalDof(local));
        for (const RawBoundaryBlock& block : inputs.units[u].system->rawBoundaryBlocks())
            for (const std::int32_t node : {block.rowNode, block.columnNode})
                if (const std::int32_t point = m_pointOfNode[static_cast<std::size_t>(node)]; point != NOT_A_POINT)
                    clique.push_back(point);
        for (std::int64_t q = 0; q < modalDofCount(u); q += GRID_DOF_COUNT)
            clique.push_back(pointOfLocalDof(firstModalDof(u) + q));
        sortUnique(clique);
        if (clique.size() > 1)
            elementGroups.add(clique);
    }
    for (std::size_t b = 0; b < externalBlockCount(); ++b)
    {
        std::vector<std::int32_t> clique; // bounded by the zone's contour
        for (const std::int64_t local : externalLocalDofs(b))
            clique.push_back(pointOfLocalDof(local));
        sortUnique(clique);
        if (clique.size() > 1)
            elementGroups.add(clique);
    }
    // 3. Each group extended with the points of the relations' rows of its dependent points (MsetCouplings).
    base::BudgetedBuffer<PointPair> pairs(budget, m_relations.termCount(), "interface Gm couplings");
    std::size_t nextPair = 0;
    for (std::size_t r = 0; r < m_relations.rowCount(); ++r)
    {
        const std::int32_t dependentPoint = pointOfLocalDof(m_relations.dependent(r));
        for (const LocalTerm& term : m_relations.terms(r))
            pairs[nextPair++] = PointPair{dependentPoint, pointOfLocalDof(term.dof)};
    }
    std::sort(pairs.data(), pairs.data() + pairs.size(), [](const PointPair& a, const PointPair& b)
              { return a.point != b.point ? a.point < b.point : a.other < b.other; });
    const auto pairCount =
        static_cast<std::size_t>(std::unique(pairs.data(), pairs.data() + pairs.size()) - pairs.data());
    const auto termPointsOf = [&](std::int32_t point)
    {
        const PointPair* first = pairs.data();
        const PointPair* last = first + pairCount;
        const PointPair* begin =
            std::lower_bound(first, last, point, [](const PointPair& pair, std::int32_t p) { return pair.point < p; });
        const PointPair* end = begin;
        while (end != last && end->point == point)
            ++end;
        return std::span<const PointPair>(begin, end);
    };
    GroupLists groups(budget);
    std::vector<std::int32_t> group; // bounded by a group and the points of its dependents' rows
    const auto addExtended = [&](std::span<const std::int32_t> points)
    {
        group.assign(points.begin(), points.end());
        const std::size_t own = group.size();
        for (std::size_t k = 0; k < own; ++k)
            for (const PointPair& pair : termPointsOf(group[k]))
                group.push_back(pair.other);
        sortUnique(group);
        groups.add(group);
    };
    const linalg::NodeGroupList elementList = elementGroups.view();
    for (std::size_t g = 0; g < elementList.groupCount(); ++g)
        addExtended(elementList.group(g));
    for (std::size_t k = 0; k < pairCount; ++k)
        if (k == 0 || pairs[k].point != pairs[k - 1].point)
            addExtended(std::span<const std::int32_t>(&pairs[k].point, 1));
    const std::vector<linalg::NodeGroupList> lists{groups.view()};
    m_graph = linalg::NodeGraph(static_cast<std::int32_t>(m_points.size()), lists, budget, pool);

    // A scalar point takes the position of a grid point it is coupled to (FsetOrdering does the same).
    for (std::size_t p = 0; p < m_points.size(); ++p)
    {
        if (m_points[p].dofCount != 1)
            continue;
        for (const std::int32_t neighbour : m_graph.neighbours(static_cast<std::int32_t>(p)))
            if (m_points[static_cast<std::size_t>(neighbour)].dofCount == GRID_DOF_COUNT)
            {
                m_points[p].position = m_points[static_cast<std::size_t>(neighbour)].position;
                break;
            }
    }
}

void InterfaceProblem::assemble(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool,
                                const base::CancellationToken* cancellation)
{
    base::BudgetedBuffer<std::uint8_t> dofCounts(budget, m_points.size(), "interface point DOF counts");
    for (std::size_t p = 0; p < m_points.size(); ++p)
        dofCounts[p] = m_points[p].dofCount;
    m_pattern = std::make_unique<linalg::SparsePattern>(m_graph, dofCounts.span(), budget);
    m_stiffness = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "interface stiffness");
    // 1. The interface elements, in the model's order (the order of every sum of the full path's assembly).
    linalg::SparseAssembler(inputs.assembly)
        .assemble(
            m_elementPlaces.size(),
            [&](std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions)
            {
                addElementContributions(
                    inputs, begin, end,
                    [](std::size_t, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                       elements::ElementMatrixBlock& block)
                    {
                        if (batch.family->facets().has(elements::Facet::Stiffness))
                            batch.family->computeStiffness(batch.view, range, block);
                    },
                    false, contributions);
            },
            *m_stiffness, pool, cancellation);
    // 2. The K* of the units, in their order, onto their labels.
    const SymmetricBlockAccess access(*m_pattern, m_firstDof.span(), m_stiffness->values());
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
    {
        const linalg::DenseMatrix& stiffness = inputs.units[u].reduced->stiffness();
        const std::span<const std::int64_t> locals = unitLocalDofs(u);
        // Step 10.25: the blocks of K* that are exactly zero are not in the pattern of a unit whose K* is not
        // dense (buildGraph): left out, every value written added as before.
        std::vector<std::int32_t> rowPoints; // bounded by the unit's boundary
        for (const std::int64_t local : locals)
            rowPoints.push_back(pointOfLocalDof(local));
        const KstarStructure structure = modalDofCount(u) == 0 && inputs.second == InterfaceSecondMatrix::None
                                             ? kstarStructureOf(stiffness, rowPoints)
                                             : KstarStructure{};
        for (std::size_t j = 0; j < locals.size(); ++j)
            for (std::size_t i = 0; i < locals.size(); ++i)
                if (structure.isDense || structure.coupled(static_cast<std::size_t>(structure.pointOfRow[i]),
                                                           static_cast<std::size_t>(structure.pointOfRow[j])))
                    access.add(locals[i], locals[j], stiffness(i, j));
        // 3. Its modal stiffness K_qq on its modal coordinates (K_aq is zero: ModalReduction).
        const auto modes = static_cast<std::size_t>(modalDofCount(u));
        if (modes == 0)
            continue;
        const linalg::DenseMatrix& modal = inputs.units[u].modal->modalStiffness();
        const std::int64_t first = firstModalDof(u);
        for (std::size_t j = 0; j < modes; ++j)
            for (std::size_t i = 0; i < modes; ++i)
                access.add(first + static_cast<std::int64_t>(i), first + static_cast<std::int64_t>(j), modal(i, j));
    }
    // 4. The K* of the external units, in their order, onto their contour DOFs.
    for (std::size_t b = 0; b < externalBlockCount(); ++b)
    {
        const linalg::DenseMatrix& stiffness = *inputs.externalBlocks[b].stiffness;
        const std::span<const std::int64_t> locals = externalLocalDofs(b);
        for (std::size_t j = 0; j < locals.size(); ++j)
            for (std::size_t i = 0; i < locals.size(); ++i)
                access.add(locals[i], locals[j], stiffness(i, j));
    }
}

void InterfaceProblem::checkSingularities(const InterfaceInputs& inputs, base::MemoryBudget& budget,
                                          base::ThreadPool* pool)
{
    // The full path's Knn at the interface points: the interface elements on their own DOFs and each unit's raw
    // boundary blocks, the interface's relations eliminated.
    linalg::BlockSparseMatrix raw(*m_pattern, budget, "interface singularity stiffness");
    linalg::SparseAssembler(inputs.assembly)
        .assemble(
            m_elementPlaces.size(),
            [&](std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions)
            {
                addElementContributions(
                    inputs, begin, end,
                    [](std::size_t, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                       elements::ElementMatrixBlock& block)
                    {
                        if (batch.family->facets().has(elements::Facet::Stiffness))
                            batch.family->computeStiffness(batch.view, range, block);
                    },
                    true, contributions);
            },
            raw, pool);
    const SymmetricBlockAccess access(*m_pattern, m_firstDof.span(), raw.values());
    double largest = 0.0;
    for (const InterfaceUnit& unit : inputs.units)
    {
        largest = std::max(largest, unit.system->largestFreeDiagonal());
        const std::span<const double> values = unit.system->rawBoundaryValues();
        for (const RawBoundaryBlock& block : unit.system->rawBoundaryBlocks())
        {
            const std::int32_t rowPoint = m_pointOfNode[static_cast<std::size_t>(block.rowNode)];
            const std::int32_t columnPoint = m_pointOfNode[static_cast<std::size_t>(block.columnNode)];
            if (rowPoint == NOT_A_POINT || columnPoint == NOT_A_POINT)
                throw std::invalid_argument(base::concatenate(
                    {"InterfaceProblem: unit ", std::to_string(unit.reduced->unitId()),
                     " gives a boundary block of an interior point; the units and the interface must come from one "
                     "plan"}));
            const int rows = m_points[static_cast<std::size_t>(rowPoint)].dofCount;
            const int columns = m_points[static_cast<std::size_t>(columnPoint)].dofCount;
            for (int j = 0; j < columns; ++j)
                for (int i = 0; i < rows; ++i)
                    access.add(m_firstDof[static_cast<std::size_t>(rowPoint)] + i,
                               m_firstDof[static_cast<std::size_t>(columnPoint)] + j,
                               values[static_cast<std::size_t>(block.valueOffset + i + rows * j)]);
        }
    }
    // The external units' K*, as the full path's Kgg holds it.
    for (std::size_t b = 0; b < externalBlockCount(); ++b)
    {
        const linalg::DenseMatrix& stiffness = *inputs.externalBlocks[b].stiffness;
        const std::span<const std::int64_t> locals = externalLocalDofs(b);
        for (std::size_t j = 0; j < locals.size(); ++j)
            for (std::size_t i = 0; i < locals.size(); ++i)
                access.add(locals[i], locals[j], stiffness(i, j));
    }
    m_relations.eliminate(*m_pattern, m_firstDof.span(), m_graph, raw.values(), budget);

    // The grid point singularity processor on the diagonal blocks (GridPointSingularities' rules), on the full path's
    // f set.
    const assembly::DofSets& sets = *inputs.modelSets;
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    base::BudgetedBuffer<double> ratios(budget, dofCount, "interface singularity ratios");
    base::BudgetedBuffer<std::uint8_t> isAligned(budget, dofCount, "interface singularity alignment");
    std::fill(ratios.data(), ratios.data() + dofCount, assembly::NOT_SINGULAR_RATIO);
    const auto isFree = [&](std::int64_t local)
    {
        const std::optional<base::DofIndex> dof = modelDofOf(local);
        return dof && sets.contains(assembly::DofSet::F, *dof);
    };
    constexpr std::size_t BLOCK_SIZE = 3;
    for (std::size_t p = 0; p < m_points.size(); ++p)
    {
        if (m_points[p].modelNode < 0)
            continue; // a generated fastener point, modal coordinates: no point of the model
        const int count = m_points[p].dofCount;
        const std::int64_t first = m_firstDof[p];
        const std::span<const double> diagonal = raw.blockValues(m_pattern->columnBegin(static_cast<std::int32_t>(p)));
        for (int c = 0; c < count; ++c)
            if (isFree(first + c))
                largest = std::max(largest, std::abs(diagonal[static_cast<std::size_t>(c + count * c)]));
        if (count != GRID_DOF_COUNT)
            continue;
        for (std::size_t half = 0; half < 2; ++half)
        {
            std::array<double, BLOCK_SIZE * BLOCK_SIZE> block{};
            std::array<bool, BLOCK_SIZE> isFreeComponent{};
            for (std::size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                isFreeComponent[i] = isFree(first + static_cast<std::int64_t>(BLOCK_SIZE * half + i));
                for (std::size_t j = 0; j < BLOCK_SIZE; ++j)
                    block[i * BLOCK_SIZE + j] =
                        diagonal[(BLOCK_SIZE * half + i) + GRID_DOF_COUNT * (BLOCK_SIZE * half + j)];
            }
            const assembly::SingularBlock analysis =
                assembly::SingularBlock::analyse(block, isFreeComponent, inputs.singularity.pivotZeroThreshold);
            for (std::size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                const auto local = static_cast<std::size_t>(first) + BLOCK_SIZE * half + i;
                ratios[local] = analysis.ratios[i];
                isAligned[local] = analysis.isAligned ? 1 : 0;
            }
        }
    }
    for (std::size_t p = 0; p < m_points.size(); ++p)
    {
        if (m_points[p].modelNode < 0 || m_points[p].dofCount != 1 || !isFree(m_firstDof[p]))
            continue;
        const double diagonal = raw.blockValues(m_pattern->columnBegin(static_cast<std::int32_t>(p)))[0];
        if (diagonal <= inputs.singularity.pivotZeroThreshold * largest)
            ratios[static_cast<std::size_t>(m_firstDof[p])] = largest > 0.0 ? std::max(diagonal / largest, 0.0) : 0.0;
    }
    std::size_t count = 0;
    for (std::size_t local = 0; local < dofCount; ++local)
        count += ratios[local] != assembly::NOT_SINGULAR_RATIO ? 1 : 0;
    m_singularities = base::BudgetedBuffer<assembly::SingularityRow>(budget, count, "interface singularity table");
    std::size_t next = 0;
    for (std::size_t local = 0; local < dofCount; ++local)
    {
        if (ratios[local] == assembly::NOT_SINGULAR_RATIO)
            continue;
        const base::DofIndex dof = *modelDofOf(static_cast<std::int64_t>(local));
        const assembly::DofLabel label = m_dofMap->labelOf(dof);
        assembly::SingularityRow row;
        row.dof = dof;
        row.pointId = m_dofMap->pointIdOf(dof);
        row.isScalarPoint = label.kind == input::PointKind::Scalar;
        row.component = label.component;
        row.stiffnessRatio = ratios[local];
        row.oldExclusive = sets.exclusiveSetOf(dof);
        row.isConstrained = inputs.singularity.isAutoSpcOn;
        row.isAligned = row.isScalarPoint || isAligned[local] != 0;
        m_singularities[next++] = row;
        if (inputs.singularity.isAutoSpcOn)
            m_role[local] = HELD;
    }
    std::sort(m_singularities.data(), m_singularities.data() + count,
              [](const assembly::SingularityRow& a, const assembly::SingularityRow& b)
              { return a.pointId != b.pointId ? a.pointId < b.pointId : a.component < b.component; });
}

void InterfaceProblem::computeMassProperties(const InterfaceInputs& inputs, base::MemoryBudget& budget,
                                             base::ThreadPool* pool)
{
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    m_rigidBodyModes = linalg::DenseMatrix(budget, dofCount, RIGID_BODY_MODE_COUNT, "interface rigid-body modes");
    for (std::size_t p = 0; p < m_points.size(); ++p)
    {
        const std::int32_t node = m_points[p].modelNode;
        const bool isGenerated = node == GENERATED;
        // A scalar point does not move with the body; nor do modal coordinates (a rigid motion of a unit is its
        // constraint modes alone, q = 0).
        if (m_points[p].dofCount != GRID_DOF_COUNT || node == MODAL)
            continue;
        // u = t + θ × x in the point's displacement system (u_cd = Rᵀ·u_basic); a generated point's in the basic one.
        const base::Mat3 toCd = isGenerated ? base::Mat3::identity() : m_dofMap->cdToBasic(node).transposed();
        const base::Vec3 x = m_points[p].position;
        const auto first = static_cast<std::size_t>(m_firstDof[p]);
        for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
        {
            base::Vec3 translation{};
            base::Vec3 rotation{};
            if (mode < static_cast<std::size_t>(TRANSLATION_COUNT))
                translation[mode] = 1.0;
            else
            {
                rotation[mode - TRANSLATION_COUNT] = 1.0;
                translation = base::cross(rotation, x);
            }
            const base::Vec3 u = toCd * translation;
            const base::Vec3 theta = toCd * rotation;
            for (std::size_t k = 0; k < static_cast<std::size_t>(TRANSLATION_COUNT); ++k)
            {
                m_rigidBodyModes(first + k, mode) = u[k];
                m_rigidBodyModes(first + TRANSLATION_COUNT + k, mode) = theta[k];
            }
        }
    }
    const bool isEigenMass = inputs.second == InterfaceSecondMatrix::Mass;
    if (!inputs.computesMass && !isEigenMass)
        return;
    auto massBlocks = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "interface mass");
    linalg::BlockSparseMatrix& mass = *massBlocks;
    const assembly::MassAssemblyOptions massOptions = inputs.massOptions;
    const assembly::NonStructuralMassShares* shares = inputs.nonStructuralMass;
    linalg::SparseAssembler(inputs.assembly)
        .assemble(
            m_elementPlaces.size(),
            [&](std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions)
            {
                addElementContributions(
                    inputs, begin, end,
                    [&](std::size_t batchIndex, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                        elements::ElementMatrixBlock& block)
                    {
                        if (!batch.family->facets().has(elements::Facet::Mass))
                            return;
                        elements::MassOptions options;
                        options.formulation = massOptions.formulation;
                        if (shares != nullptr)
                            options.addedNonStructuralMass = shares->sharesOf(batchIndex);
                        batch.family->computeMass(batch.view, range, options, block);
                    },
                    false, contributions);
            },
            mass, pool);
    // PARAM WTMASS multiplies the structural mass once assembled (as assembly::MassAssembler).
    if (massOptions.weightToMass != 1.0)
        for (double& value : mass.values())
            value *= massOptions.weightToMass;
    // The mass of a vibration analysis is the second matrix of its pencil (the units' parts added by assembleSecond).
    if (isEigenMass)
        m_second = std::move(massBlocks);
    if (!inputs.computesMass)
        return;
    m_inertialLoads = linalg::DenseMatrix(budget, dofCount, RIGID_BODY_MODE_COUNT, "interface inertial loads");
    base::BudgetedBuffer<double> column(budget, dofCount, "interface rigid-body mode");
    base::BudgetedBuffer<double> product(budget, dofCount, "interface mass times mode");
    for (std::size_t j = 0; j < RIGID_BODY_MODE_COUNT; ++j)
    {
        for (std::size_t dof = 0; dof < dofCount; ++dof)
            column[dof] = m_rigidBodyModes(dof, j);
        mass.multiply(column.span(), product.span());
        for (std::size_t dof = 0; dof < dofCount; ++dof)
            m_inertialLoads(dof, j) = -product[dof];
        for (std::size_t i = j; i < RIGID_BODY_MODE_COUNT; ++i)
        {
            double sum = 0.0;
            for (std::size_t dof = 0; dof < dofCount; ++dof)
                sum += m_rigidBodyModes(dof, i) * product[dof];
            m_rigidBodyMass(i, j) = sum;
            m_rigidBodyMass(j, i) = sum;
        }
    }
}

void InterfaceProblem::assembleSecond(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool)
{
    if (inputs.second == InterfaceSecondMatrix::None)
        return;
    if (inputs.second == InterfaceSecondMatrix::DifferentialStiffness)
    {
        if (inputs.preload == nullptr)
            throw std::invalid_argument(
                "InterfaceProblem: the differential stiffness of a buckling analysis needs the preload displacements");
        // The interface elements' K_d from the preload, in the model's order, carried to the references of the
        // absorbed spiders as the stiffness (the fasteners there are differentially inert: nothing is carried).
        m_second = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "interface differential stiffness");
        const elements::NodalDisplacements& preload = *inputs.preload;
        linalg::SparseAssembler(inputs.assembly)
            .assemble(
                m_elementPlaces.size(),
                [&](std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions)
                {
                    addElementContributions(
                        inputs, begin, end,
                        [&preload](std::size_t, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                                   elements::ElementMatrixBlock& block)
                        {
                            if (batch.family->facets().has(elements::Facet::Stiffness))
                                batch.family->computeDifferentialStiffness(batch.view, range, preload, block);
                        },
                        false, contributions);
                },
                *m_second, pool);
    }
    // The units' parts on their boundary DOFs then their modal coordinates, in the order of the units.
    const SymmetricBlockAccess access(*m_pattern, m_firstDof.span(), m_second->values());
    std::vector<std::int64_t> locals; // bounded by a unit's boundary and modes
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
    {
        const linalg::DenseMatrix* part = inputs.units[u].second;
        if (part == nullptr)
            continue;
        const std::span<const std::int64_t> boundary = unitLocalDofs(u);
        locals.assign(boundary.begin(), boundary.end());
        for (std::int64_t q = 0; q < modalDofCount(u); ++q)
            locals.push_back(firstModalDof(u) + q);
        if (part->rows() != locals.size() || part->columns() != locals.size())
            throw std::invalid_argument(
                base::concatenate({"InterfaceProblem: unit ", std::to_string(inputs.units[u].reduced->unitId()),
                                   " gives a ", std::to_string(part->rows()), " x ", std::to_string(part->columns()),
                                   " part of the second matrix for ", std::to_string(locals.size()),
                                   " boundary DOFs and modal coordinates; give it on both, in that order"}));
        for (std::size_t j = 0; j < locals.size(); ++j)
            for (std::size_t i = 0; i < locals.size(); ++i)
                access.add(locals[i], locals[j], (*part)(i, j));
    }
}

void InterfaceProblem::order(const InterfaceInputs& inputs, const linalg::DenseKernels& kernels,
                             base::MemoryBudget& budget, base::ThreadPool* pool,
                             const base::CancellationToken* cancellation)
{
    // The relations on the stiffness (step 3), in place, and on the second matrix of an eigenvalue analysis (the m set
    // of every matrix is eliminated by the same Gm).
    m_relations.eliminate(*m_pattern, m_firstDof.span(), m_graph, m_stiffness->values(), budget);
    if (m_second)
        m_relations.eliminate(*m_pattern, m_firstDof.span(), m_graph, m_second->values(), budget);

    const auto dofCount = static_cast<std::size_t>(localDofCount());
    base::BudgetedBuffer<std::uint8_t> eliminated(budget, dofCount, "interface eliminated DOFs");
    for (std::size_t local = 0; local < dofCount; ++local)
        eliminated[local] = m_role[local] == FREE ? 0 : 1;
    std::size_t heldCount = 0;
    for (std::size_t local = 0; local < dofCount; ++local)
        heldCount += m_role[local] == HELD ? 1 : 0;
    m_heldDofs = base::BudgetedBuffer<std::int64_t>(budget, heldCount, "interface held DOFs");
    std::size_t nextHeld = 0;
    for (std::size_t local = 0; local < dofCount; ++local)
        if (m_role[local] == HELD)
            m_heldDofs[nextHeld++] = static_cast<std::int64_t>(local);

    // The points with an equation, their weights and positions, and their vertex of the quotient graph: the boundary
    // of each condensed unit is one vertex (its first unit's), every other point one of its own.
    const std::size_t pointCount = m_points.size();
    base::BudgetedBuffer<std::int32_t> vertexOfPoint(budget, pointCount, "interface quotient vertices");
    std::fill(vertexOfPoint.data(), vertexOfPoint.data() + pointCount, -1);
    for (std::size_t u = 0; u < inputs.units.size(); ++u)
    {
        for (const std::int64_t local : unitLocalDofs(u))
        {
            std::int32_t& vertex = vertexOfPoint[static_cast<std::size_t>(pointOfLocalDof(local))];
            if (vertex < 0)
                vertex = static_cast<std::int32_t>(u);
        }
        // A unit's modal coordinates belong to its vertex (they couple with its boundary only).
        for (std::int64_t q = 0; q < modalDofCount(u); ++q)
            vertexOfPoint[static_cast<std::size_t>(pointOfLocalDof(firstModalDof(u) + q))] =
                static_cast<std::int32_t>(u);
    }
    base::BudgetedBuffer<std::int32_t> keptPoints(budget, pointCount, "interface points with equations");
    base::BudgetedBuffer<std::int32_t> weights(budget, pointCount, "interface point weights");
    base::BudgetedBuffer<base::Vec3> positions(budget, pointCount, "interface point positions");
    base::BudgetedBuffer<std::int32_t> keptVertices(budget, pointCount, "interface kept vertices");
    std::size_t kept = 0;
    auto vertexCount = static_cast<std::int32_t>(inputs.units.size());
    for (std::size_t p = 0; p < pointCount; ++p)
    {
        std::int32_t free = 0;
        for (auto local = m_firstDof[p]; local < m_firstDof[p + 1]; ++local)
            free += m_role[static_cast<std::size_t>(local)] == FREE ? 1 : 0;
        if (free == 0)
            continue;
        keptPoints[kept] = static_cast<std::int32_t>(p);
        weights[kept] = free;
        positions[kept] = m_points[p].position;
        keptVertices[kept] = vertexOfPoint[p] >= 0 ? vertexOfPoint[p] : vertexCount++;
        ++kept;
    }
    keepHeldRows(budget);
    base::BudgetedBuffer<std::uint8_t> dofCounts(budget, pointCount, "interface point DOF counts");
    for (std::size_t p = 0; p < pointCount; ++p)
        dofCounts[p] = m_points[p].dofCount;
    if (kept == 0)
    {
        m_selection = linalg::DofSelection(dofCounts.span(), eliminated.span(), {}, budget);
        m_stiffness.reset();
        m_second.reset();
        m_pattern.reset();
        m_isFactorised = true; // nothing to factorise: every solve is empty
        return;
    }
    // The vertices numbered 0, 1, ... in their order of appearance (the units with no kept point leave a gap).
    {
        std::vector<std::int32_t> renumber(static_cast<std::size_t>(vertexCount), -1); // bounded by the vertices
        std::int32_t next = 0;
        for (std::size_t k = 0; k < kept; ++k)
        {
            std::int32_t& vertex = renumber[static_cast<std::size_t>(keptVertices[k])];
            if (vertex < 0)
                vertex = next++;
            keptVertices[k] = vertex;
        }
        vertexCount = next;
    }
    const std::span<const std::int32_t> nodes = keptPoints.span().first(kept);
    const std::span<const std::int32_t> nodeWeights = weights.span().first(kept);
    const linalg::NodeGraph subgraph = m_graph.inducedSubgraph(nodes, budget);
    const linalg::NodePermutation ordering =
        InterfaceEstimate::orderOnUnits(subgraph, keptVertices.span().first(kept), vertexCount, nodeWeights,
                                        positions.span().first(kept), budget, pool);
    m_symbolic =
        linalg::SymbolicFactor::analyse(subgraph, nodeWeights, ordering, 0, linalg::AmalgamationThresholds{}, budget);
    base::BudgetedBuffer<std::int32_t> nodeOrder(budget, kept, "interface point order");
    const std::span<const std::int32_t> symbolicOrder = m_symbolic.nodeOrder().nodes();
    for (std::size_t k = 0; k < kept; ++k)
        nodeOrder[k] = nodes[static_cast<std::size_t>(symbolicOrder[k])];
    m_selection = linalg::DofSelection(dofCounts.span(), eliminated.span(), nodeOrder.span(), budget);
    m_matrix = linalg::SymmetricCsc::fromBlocks(*m_stiffness, m_selection, budget, pool);
    m_stiffness.reset();
    if (m_second)
    {
        m_secondMatrix = linalg::SymmetricCsc::fromBlocks(*m_second, m_selection, budget, pool);
        m_second.reset();
    }
    m_pattern.reset();
    // An eigenvalue analysis factorises K − σ·B at its shifts (linalg::SparsePencil, BucklingPencil), not K.
    if (!inputs.factorises)
        return;
    m_isFactorised = true;
    // Without the factor: each solve factorises the stiffness streamed (solveStreamed).
    if (!inputs.keepsFactor)
    {
        m_isStreamed = true;
        m_regionValues = inputs.regionValues;
        m_pivotPolicy = inputs.pivotPolicy;
        m_budget = &budget;
        return;
    }

    linalg::FactorOptions options;
    options.pivoting = linalg::PivotingMode::PositiveDefinite;
    options.pivotPolicy = inputs.pivotPolicy;
    options.equationNamer = [this](std::int32_t equation) { return nameOf(m_selection.dofOfEquation(equation)); };
    options.cancellation = cancellation;
    // The factorisation's memory is estimated before it is made (the memory budget): its working storage
    // grows with the threads that run subtrees at once, so when the threaded estimate does not fit the budget left the
    // factorisation runs on one thread, with the same bits (the factor does not depend on the threads); when one thread
    // does not fit either, SupernodalFactor refuses it before any numeric work.
    const linalg::DenseKernelOptions& threaded = kernels.options();
    const std::size_t threads = threaded.pool != nullptr ? threaded.pool->threadCount() : 1;
    m_factorisationThreadCount = threads;
    if (threads > 1 &&
        linalg::SupernodalFactor::estimateMemoryBytes(m_symbolic, threads, options.pivoting) > budget.available())
    {
        linalg::DenseKernelOptions serialOptions = threaded;
        serialOptions.pool = nullptr;
        const linalg::DenseKernels serial(kernels.kind(), serialOptions);
        m_factorisationThreadCount = 1;
        m_factor = linalg::SupernodalFactor::factorise(m_symbolic, m_matrix, serial, options, budget);
        return;
    }
    m_factor = linalg::SupernodalFactor::factorise(m_symbolic, m_matrix, kernels, options, budget);
}

void InterfaceProblem::keepHeldRows(base::MemoryBudget& budget)
{
    const SymmetricBlockAccess access(*m_pattern, m_firstDof.span(), m_stiffness->values());
    // The DOFs of a point and of its neighbours that are not dependent (the dependent ones' terms are stale).
    std::vector<std::int64_t> coupled; // bounded by a point's degree
    const auto gather = [&](std::int64_t dof)
    {
        coupled.clear();
        const std::int32_t point = pointOfLocalDof(dof);
        const auto addPoint = [&](std::int32_t p)
        {
            for (auto local = m_firstDof[static_cast<std::size_t>(p)];
                 local < m_firstDof[static_cast<std::size_t>(p) + 1]; ++local)
                if (m_role[static_cast<std::size_t>(local)] != DEPENDENT && access.read(dof, local) != 0.0)
                    coupled.push_back(local);
        };
        bool isOwnAdded = false;
        for (const std::int32_t neighbour : m_graph.neighbours(point))
        {
            if (!isOwnAdded && neighbour > point)
            {
                addPoint(point);
                isOwnAdded = true;
            }
            addPoint(neighbour);
        }
        if (!isOwnAdded)
            addPoint(point);
    };
    std::size_t termCount = 0;
    for (const std::int64_t held : m_heldDofs.span())
    {
        gather(held);
        termCount += coupled.size();
    }
    m_heldOffsets = base::BudgetedBuffer<std::int64_t>(budget, m_heldDofs.size() + 1, "interface held rows");
    m_heldColumns = base::BudgetedBuffer<std::int64_t>(budget, termCount, "interface held rows");
    m_heldValues = base::BudgetedBuffer<double>(budget, termCount, "interface held rows");
    std::size_t next = 0;
    for (std::size_t k = 0; k < m_heldDofs.size(); ++k)
    {
        m_heldOffsets[k] = static_cast<std::int64_t>(next);
        gather(m_heldDofs[k]);
        for (const std::int64_t local : coupled)
        {
            m_heldColumns[next] = local;
            m_heldValues[next] = access.read(m_heldDofs[k], local);
            ++next;
        }
    }
    m_heldOffsets[m_heldDofs.size()] = static_cast<std::int64_t>(next);
}

// ---------------------------------------------------------------------------------------------------------------------
// Loads and solutions

void InterfaceProblem::reduceLoads(linalg::DenseMatrixView loadsLocal) const
{
    if (loadsLocal.rows != static_cast<std::size_t>(localDofCount()))
        throw std::invalid_argument(
            base::concatenate({"InterfaceProblem::reduceLoads: ", std::to_string(loadsLocal.rows), " rows for ",
                               std::to_string(localDofCount()), " local DOFs; give one row per local DOF"}));
    for (std::size_t j = 0; j < loadsLocal.columns; ++j)
        m_relations.reduce(std::span<double>(&loadsLocal(0, j), loadsLocal.rows));
}

void InterfaceProblem::equationLoads(linalg::ConstDenseMatrixView reducedLocal,
                                     linalg::ConstDenseMatrixView enforcedLocal,
                                     linalg::DenseMatrixView equationValues) const
{
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    const auto equations = static_cast<std::size_t>(equationCount());
    const bool hasEnforced = enforcedLocal.columns > 0;
    if (reducedLocal.rows != dofCount || equationValues.rows != equations ||
        equationValues.columns != reducedLocal.columns ||
        (hasEnforced && (enforcedLocal.rows != dofCount || enforcedLocal.columns != reducedLocal.columns)))
        throw std::invalid_argument(base::concatenate(
            {"InterfaceProblem::equationLoads: ", std::to_string(reducedLocal.rows), " x ",
             std::to_string(reducedLocal.columns), " loads into ", std::to_string(equationValues.rows), " x ",
             std::to_string(equationValues.columns), " equation values; expected ", std::to_string(dofCount),
             " local rows and ", std::to_string(equations), " equation rows, as many columns"}));
    for (std::size_t j = 0; j < reducedLocal.columns; ++j)
    {
        for (std::size_t e = 0; e < equations; ++e)
            equationValues(e, j) =
                reducedLocal(static_cast<std::size_t>(m_selection.dofOfEquation(static_cast<std::int32_t>(e))), j);
        if (!hasEnforced)
            continue;
        // −K_ls·u_s, from the rows of the held DOFs (Knn is symmetric).
        for (std::size_t k = 0; k < m_heldDofs.size(); ++k)
        {
            const double value = enforcedLocal(static_cast<std::size_t>(m_heldDofs[k]), j);
            if (value == 0.0)
                continue;
            for (auto t = m_heldOffsets[k]; t < m_heldOffsets[k + 1]; ++t)
            {
                const std::int32_t equation = m_selection.equationOfDof(m_heldColumns[static_cast<std::size_t>(t)]);
                if (equation != linalg::DofSelection::ELIMINATED)
                    equationValues(static_cast<std::size_t>(equation), j) -=
                        m_heldValues[static_cast<std::size_t>(t)] * value;
            }
        }
    }
}

void InterfaceProblem::solve(linalg::DenseMatrixView rightHandSides, const base::CancellationToken* cancellation) const
{
    solve(rightHandSides, *m_kernels, cancellation);
}

void InterfaceProblem::solve(linalg::DenseMatrixView rightHandSides, const linalg::DenseKernels& kernels,
                             const base::CancellationToken* cancellation) const
{
    if (equationCount() == 0)
        return;
    if (!m_isFactorised)
        throw std::logic_error(
            "InterfaceProblem::solve: the stiffness of an eigenvalue analysis is not factorised "
            "(InterfaceInputs::factorises); solve with the pencil of its shifts");
    if (m_isStreamed)
    {
        solveStreamed(rightHandSides, kernels, cancellation);
        return;
    }
    linalg::FactorSolve(m_factor, kernels).solve(rightHandSides, cancellation);
}

void InterfaceProblem::solveStreamed(linalg::DenseMatrixView rightHandSides, const linalg::DenseKernels& kernels,
                                     const base::CancellationToken* cancellation) const
{
    const std::lock_guard lock(m_streamMutex);
    linalg::FactorOptions options;
    options.pivoting = linalg::PivotingMode::PositiveDefinite;
    options.pivotPolicy = m_pivotPolicy;
    options.equationNamer = [this](std::int32_t equation) { return nameOf(m_selection.dofOfEquation(equation)); };
    options.cancellation = cancellation;
    // As the factorisation with the factor (order()): on one thread when the threaded working storage does not fit the
    // budget left, with the same bits.
    const linalg::DenseKernelOptions& threaded = kernels.options();
    const std::size_t threads = threaded.pool != nullptr ? threaded.pool->threadCount() : 1;
    linalg::DenseKernelOptions serialOptions = threaded;
    serialOptions.pool = nullptr;
    const linalg::DenseKernels serial(kernels.kind(), serialOptions);
    const bool isSerial =
        threads > 1 &&
        linalg::StreamedSolve::estimate(m_symbolic, threads).bytes(rightHandSides.columns, m_regionValues) >
            m_budget->available();
    const linalg::StreamedSolveResult result = linalg::StreamedSolve::solve(
        m_symbolic, m_matrix, rightHandSides, isSerial ? serial : kernels, options, *m_budget, m_regionValues);
    m_streamedThreadCount = isSerial ? 1 : threads;
    if (m_solveFigures.solveCount == 0)
        m_streamedDiagnostics = result.diagnostics;
    ++m_solveFigures.solveCount;
    m_solveFigures.columnCount += static_cast<std::int64_t>(rightHandSides.columns);
    m_solveFigures.regionRunCount += result.regions.regionRunCount;
    m_solveFigures.recomputedFlops += result.regions.recomputedFlops;
    m_solveFigures.keptPanelValues = std::max(m_solveFigures.keptPanelValues, result.regions.keptValues);
}

InterfaceSolveFigures InterfaceProblem::solveFigures() const
{
    const std::lock_guard lock(m_streamMutex);
    return m_solveFigures;
}

void InterfaceProblem::expand(linalg::ConstDenseMatrixView equationValues, linalg::ConstDenseMatrixView enforcedLocal,
                              linalg::DenseMatrixView displacementsLocal) const
{
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    const auto equations = static_cast<std::size_t>(equationCount());
    const bool hasEnforced = enforcedLocal.columns > 0;
    if (equationValues.rows != equations || displacementsLocal.rows != dofCount ||
        displacementsLocal.columns != equationValues.columns ||
        (hasEnforced && (enforcedLocal.rows != dofCount || enforcedLocal.columns != equationValues.columns)))
        throw std::invalid_argument(base::concatenate(
            {"InterfaceProblem::expand: ", std::to_string(equationValues.rows), " x ",
             std::to_string(equationValues.columns), " equation values into ", std::to_string(displacementsLocal.rows),
             " x ", std::to_string(displacementsLocal.columns), " local displacements; expected ",
             std::to_string(equations), " and ", std::to_string(dofCount), " rows, as many columns"}));
    for (std::size_t j = 0; j < equationValues.columns; ++j)
    {
        for (std::size_t local = 0; local < dofCount; ++local)
            displacementsLocal(local, j) = 0.0;
        for (std::size_t e = 0; e < equations; ++e)
            displacementsLocal(static_cast<std::size_t>(m_selection.dofOfEquation(static_cast<std::int32_t>(e))), j) =
                equationValues(e, j);
        if (hasEnforced)
            for (const std::int64_t held : m_heldDofs.span())
                displacementsLocal(static_cast<std::size_t>(held), j) =
                    enforcedLocal(static_cast<std::size_t>(held), j);
        m_relations.expand(std::span<double>(&displacementsLocal(0, j), dofCount));
    }
}

void InterfaceProblem::residuals(linalg::ConstDenseMatrixView equationValues,
                                 linalg::ConstDenseMatrixView rightHandSides, linalg::DenseMatrixView residuals) const
{
    const auto equations = static_cast<std::size_t>(equationCount());
    if (equationValues.rows != equations || rightHandSides.rows != equations || residuals.rows != equations ||
        rightHandSides.columns != equationValues.columns || residuals.columns != equationValues.columns)
        throw std::invalid_argument(
            base::concatenate({"InterfaceProblem::residuals: ", std::to_string(equationValues.rows), ", ",
                               std::to_string(rightHandSides.rows), " and ", std::to_string(residuals.rows),
                               " rows for ", std::to_string(equations), " equations; give the same columns"}));
    std::vector<double> product(equations); // bounded by the interface: one column
    for (std::size_t j = 0; j < equationValues.columns; ++j)
    {
        if (equations == 0)
            continue;
        m_matrix.multiply(std::span<const double>(equationValues.pointer(0, j), equations), product);
        for (std::size_t e = 0; e < equations; ++e)
            residuals(e, j) = rightHandSides(e, j) - product[e];
    }
}

void InterfaceProblem::heldReactions(linalg::ConstDenseMatrixView displacementsLocal,
                                     linalg::ConstDenseMatrixView reducedLocal, linalg::DenseMatrixView reactions) const
{
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    if (displacementsLocal.rows != dofCount || reducedLocal.rows != dofCount || reactions.rows != m_heldDofs.size() ||
        reducedLocal.columns != displacementsLocal.columns || reactions.columns != displacementsLocal.columns)
        throw std::invalid_argument(
            base::concatenate({"InterfaceProblem::heldReactions: ", std::to_string(displacementsLocal.rows), " and ",
                               std::to_string(reducedLocal.rows), " local rows into ", std::to_string(reactions.rows),
                               " reactions; expected ", std::to_string(dofCount), " local rows and ",
                               std::to_string(m_heldDofs.size()), " held DOFs, as many columns"}));
    for (std::size_t j = 0; j < displacementsLocal.columns; ++j)
        for (std::size_t k = 0; k < m_heldDofs.size(); ++k)
        {
            double value = -reducedLocal(static_cast<std::size_t>(m_heldDofs[k]), j);
            for (auto t = m_heldOffsets[k]; t < m_heldOffsets[k + 1]; ++t)
                value += m_heldValues[static_cast<std::size_t>(t)] *
                         displacementsLocal(static_cast<std::size_t>(m_heldColumns[static_cast<std::size_t>(t)]), j);
            reactions(k, j) = value;
        }
}

} // namespace oos::reduction
