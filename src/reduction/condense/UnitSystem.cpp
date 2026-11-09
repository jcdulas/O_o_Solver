#include "oos/reduction/condense/UnitSystem.hpp"

#include "oos/assembly/singularities/SingularBlock.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/sparse/ContributionBlock.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"
#include "oos/reduction/condense/ReducedUnit.hpp"
#include "oos/reduction/condense/SymmetricBlockAccess.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace oos::reduction
{

namespace
{

constexpr int GRID_DOF_COUNT = 6;
constexpr int TRANSLATION_COUNT = 3;
constexpr std::int32_t GENERATED_POINT = -1;
constexpr std::uint8_t BOUNDARY = 1;
constexpr std::uint8_t ELIMINATED = 1;
constexpr std::int32_t NO_SPIDER = -1;
constexpr std::size_t RIGID_BODY_MODE_COUNT = 6;
// The elements whose matrices are computed in one block (as the model's assembly, RANGE_ELEMENTS): the block is a work
// buffer of std::vector, bounded by this count times the largest element matrix (a ShellQuad4's 24 × 24: 9 MB for 2048),
// never by the unit (a run of 800 000 consecutive ShellQuad4 would hold 3.7 GB outside the budget). The sums
// are made element by element in the same order whatever the runs: the same bits.
constexpr std::size_t ELEMENTS_PER_RUN = 2048;
// buildGraph keeps the element stiffness matrices for assemble() while they take at most this fraction of the memory
// left (1/8), as the full path's ElementMatrixStore keeps a share of the budget: beyond it, assemble() computes them
// again.
constexpr std::uint64_t KEPT_STIFFNESS_MEMORY_DIVISOR = 8;

// An array that grows by appends, its capacity reserved in the memory budget before it is allocated (as the assembly
// module's internal BudgetedGrowingArray): the node groups of the unit's elements, whose sizes the
// element facets give.
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

// A NodeGroupList built by appends.
struct GroupLists
{
    GrowingArray<std::int64_t> offsets;
    GrowingArray<std::int32_t> nodes;

    explicit GroupLists(base::MemoryBudget& budget)
        : offsets(budget, "unit node groups"), nodes(budget, "unit node groups")
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

std::string unitText(std::int64_t unitId)
{
    return base::concatenate({"unit ", std::to_string(unitId)});
}

std::string spiderText(const SpiderKey& key)
{
    if (key.side == FastenerSide::None)
        return base::concatenate({"spider ", std::to_string(key.elementId)});
    return base::concatenate(
        {"fastener ", std::to_string(key.elementId), key.side == FastenerSide::A ? " side A" : " side B"});
}

void sortUnique(std::vector<std::int32_t>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

} // namespace

UnitSystem::~UnitSystem() = default;

// The stiffness matrices of a unit's elements on its local points, in the order of the element list: computed once by
// buildGraph() (for the couplings) and assembled by assemble(), their bytes reserved in the budget; not
// kept (isKept false) when they would take more than KEPT_STIFFNESS_MEMORY_DIVISOR-th of the memory left.
struct UnitSystem::KeptStiffness
{
    bool isKept = false;
    linalg::ContributionBlock block;
    std::vector<std::size_t> firstOfElement; // element k's matrices: [firstOfElement[k], firstOfElement[k + 1])
    base::MemoryReservation reservation;
};

UnitSystem::UnitSystem(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                       base::ThreadPool* pool)
    : m_unitId(description.unitId), m_dofMap(model.dofMap)
{
    if (model.elements == nullptr || model.dofMap == nullptr || model.elementIndex == nullptr ||
        model.unitRelations == nullptr)
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): the model needs its element set, DOF map, element index and the Gm of the units; give all four"}));
    if (!model.isConstrainedEverywhere.empty() &&
        model.isConstrainedEverywhere.size() != static_cast<std::size_t>(model.dofMap->gSetSize()))
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId), "): ", std::to_string(model.isConstrainedEverywhere.size()),
             " constraint flags for a g set of ", std::to_string(model.dofMap->gSetSize()),
             " DOFs; give one flag per g-set DOF"}));
    buildStiffness(model, description, budget, pool, true);
    order(budget, pool);
}

void UnitSystem::buildStiffness(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                                base::ThreadPool* pool, bool assemblesMass)
{
    numberPoints(model, description, budget);
    readSpiders(description, budget);
    readRelations(model, budget);
    assignRoles(model, budget);
    {
        KeptStiffness kept;
        buildGraph(model, description, budget, pool, kept);
        assemble(model, description, budget, pool, kept, assemblesMass);
    }
    eliminateDependents(budget);
    keepRawBoundaryStiffness(budget);
    constrainSingularities(model, budget);
    transformSpiders(*m_pattern, m_stiffness->values());
    if (m_massBlocks)
        transformSpiders(*m_pattern, m_massBlocks->values());
}

UnitSystem::UnitSystem(const UnitModel& model, const UnitDescription& description, const WithoutMass& withoutMass,
                       base::MemoryBudget& budget, base::ThreadPool* pool)
    : m_unitId(description.unitId), m_dofMap(model.dofMap)
{
    if (model.elements == nullptr || model.dofMap == nullptr || model.elementIndex == nullptr ||
        model.unitRelations == nullptr)
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): the model needs its element set, DOF map, element index and the Gm of the units; give all four"}));
    if (model.keepsMass)
        throw std::invalid_argument(base::concatenate({"UnitSystem (", unitText(m_unitId),
                                                       "): the equations of a Craig-Bampton unit (keepsMass) are "
                                                       "never released; build them with their mass"}));
    // PERFORMANCE: no element mass computed nor assembled: the recovery of a unit whose equations were released reads
    // its stiffness and relations only.
    // Measured: 3 000 plates of 3.1 M DOFs in the mode low_memory, the cases (the
    // 3 000 builds again, refactorisations and recoveries) 4.4 → 3.7 s.
    buildStiffness(model, description, budget, pool, false);
    order(budget, pool);
    m_rigidBodyMass = withoutMass.rigidBodyMass;
}

UnitSystem::UnitSystem(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                       base::ThreadPool* pool, StiffnessOnly)
    : m_unitId(description.unitId), m_dofMap(model.dofMap), m_hasEquations(false)
{
    if (model.elements == nullptr || model.dofMap == nullptr || model.elementIndex == nullptr ||
        model.unitRelations == nullptr)
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): the model needs its element set, DOF map, element index and the Gm of the units; give all four"}));
    if (model.keepsMass)
        throw std::invalid_argument(base::concatenate({"UnitSystem (", unitText(m_unitId),
                                                       "): the boundary data of a unit are made without its mass; "
                                                       "build a Craig-Bampton unit (keepsMass) with its "
                                                       "equations"}));
    buildStiffness(model, description, budget, pool, false);
    findHeldCouplings(budget);
}

UnitBoundaryData UnitSystem::boundaryDataOf(const UnitModel& model, const UnitDescription& description,
                                            const ReducedUnit& reduced, base::MemoryBudget& budget,
                                            base::ThreadPool* pool)
{
    const UnitSystem unit(model, description, budget, pool, StiffnessOnly{});
    UnitBoundaryData data;
    data.singularities = base::BudgetedBuffer<assembly::SingularityRow>(budget, unit.m_singularities.size(),
                                                                        "kept unit singularity table");
    std::copy(unit.m_singularities.data(), unit.m_singularities.data() + unit.m_singularities.size(),
              data.singularities.data());
    data.largestFreeDiagonal = unit.m_largestFreeDiagonal;
    data.rawBlocks =
        base::BudgetedBuffer<RawBoundaryBlock>(budget, unit.m_rawBlocks.size(), "kept unit raw boundary blocks");
    std::copy(unit.m_rawBlocks.data(), unit.m_rawBlocks.data() + unit.m_rawBlocks.size(), data.rawBlocks.data());
    data.rawValues = base::BudgetedBuffer<double>(budget, unit.m_rawValues.size(), "kept unit raw boundary values");
    std::copy(unit.m_rawValues.data(), unit.m_rawValues.data() + unit.m_rawValues.size(), data.rawValues.data());
    data.boundaryElements =
        base::BudgetedBuffer<base::EntityId>(budget, unit.m_boundaryElements.size(), "kept unit boundary elements");
    std::copy(unit.m_boundaryElements.data(), unit.m_boundaryElements.data() + unit.m_boundaryElements.size(),
              data.boundaryElements.data());
    data.isHeldCoupled = base::BudgetedBuffer<std::uint8_t>(budget, unit.m_isHeldCoupled.size(), "kept unit held DOFs");
    std::copy(unit.m_isHeldCoupled.data(), unit.m_isHeldCoupled.data() + unit.m_isHeldCoupled.size(),
              data.isHeldCoupled.data());
    // The boundary equations in the order of K*'s rows (the entry's labels), their local DOFs by label; the interior
    // equations those of the role Interior (the full build's DofSelection keeps Interior and Boundary DOFs).
    const std::span<const UnitDofLabel> labels = reduced.labels();
    std::vector<std::pair<UnitDofLabel, std::int64_t>> boundaryDofs; // bounded by the unit's boundary
    std::int32_t interior = 0;
    for (std::int64_t local = 0; local < unit.localDofCount(); ++local)
    {
        const UnitDofRole role = unit.m_roles[static_cast<std::size_t>(local)];
        interior += role == UnitDofRole::Interior ? 1 : 0;
        if (role == UnitDofRole::Boundary)
            boundaryDofs.emplace_back(unit.labelOf(local), local);
    }
    const auto byLabel =
        [](const std::pair<UnitDofLabel, std::int64_t>& a, const std::pair<UnitDofLabel, std::int64_t>& b)
    {
        const UnitDofLabel& x = a.first;
        const UnitDofLabel& y = b.first;
        return std::tie(x.kind, x.id, x.side, x.component) < std::tie(y.kind, y.id, y.side, y.component);
    };
    std::sort(boundaryDofs.begin(), boundaryDofs.end(), byLabel);
    if (boundaryDofs.size() != labels.size())
        throw base::InternalError(base::concatenate(
            {"UnitSystem (", unitText(description.unitId), "): the cached reduced unit has ",
             std::to_string(labels.size()), " boundary DOFs, the unit ", std::to_string(boundaryDofs.size()),
             "; the cache key missed an input of its equations (a bug of O_o_Solver: report it)"}));
    data.boundaryLabels = base::BudgetedBuffer<UnitDofLabel>(budget, labels.size(), "kept unit boundary labels");
    data.boundaryLocalDofs = base::BudgetedBuffer<std::int64_t>(budget, labels.size(), "kept unit boundary DOFs");
    for (std::size_t k = 0; k < labels.size(); ++k)
    {
        const auto found = std::lower_bound(boundaryDofs.begin(), boundaryDofs.end(),
                                            std::pair<UnitDofLabel, std::int64_t>{labels[k], 0}, byLabel);
        if (found == boundaryDofs.end() || !(found->first == labels[k]))
            throw base::InternalError(base::concatenate(
                {"UnitSystem (", unitText(description.unitId),
                 "): a boundary DOF of the cached reduced unit is not one of the unit's; the cache key missed an input "
                 "of its equations (a bug of O_o_Solver: report it)"}));
        data.boundaryLabels[k] = labels[k];
        data.boundaryLocalDofs[k] = found->second;
    }
    data.interiorDofCount = interior;
    data.rigidBodyMass = reduced.rigidBodyMass();
    return data;
}

void UnitSystem::findHeldCouplings(base::MemoryBudget& budget)
{
    const SymmetricBlockAccess access(*m_pattern, m_firstDof.span(), m_stiffness->values());
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    const auto hasEquation = [&](std::int64_t dof)
    {
        const UnitDofRole role = m_roles[static_cast<std::size_t>(dof)];
        return role == UnitDofRole::Interior || role == UnitDofRole::Boundary;
    };
    std::size_t constrained = 0;
    for (std::size_t dof = 0; dof < dofCount; ++dof)
        constrained += m_roles[dof] == UnitDofRole::Constrained ? 1 : 0;
    m_constrainedDofs = base::BudgetedBuffer<std::int64_t>(budget, constrained, "unit held DOFs");
    m_isHeldCoupled = base::BudgetedBuffer<std::uint8_t>(budget, constrained, "unit held DOF couplings");
    std::size_t next = 0;
    for (std::size_t dof = 0; dof < dofCount; ++dof)
    {
        if (m_roles[dof] != UnitDofRole::Constrained)
            continue;
        bool isCoupled = false;
        forEachCoupledDof(pointOfLocalDof(static_cast<std::int64_t>(dof)),
                          [&](std::int64_t other)
                          {
                              isCoupled = isCoupled ||
                                          ((hasEquation(other) ||
                                            m_roles[static_cast<std::size_t>(other)] == UnitDofRole::Constrained) &&
                                           access.read(other, static_cast<std::int64_t>(dof)) != 0.0);
                          });
        m_isHeldCoupled[next] = isCoupled ? 1 : 0;
        m_constrainedDofs[next++] = static_cast<std::int64_t>(dof);
    }
}

UnitSystem::UnitSystem(const UnitModel& model, const UnitDescription& description, const UnitBoundaryData& data,
                       base::MemoryBudget& budget)
    : m_unitId(description.unitId), m_dofMap(model.dofMap), m_hasEquations(false)
{
    if (model.dofMap == nullptr || model.unitRelations == nullptr)
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): a unit without its equations needs the DOF map and the Gm of the units of its model; give both"}));
    if (!model.isConstrainedEverywhere.empty() &&
        model.isConstrainedEverywhere.size() != static_cast<std::size_t>(model.dofMap->gSetSize()))
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId), "): ", std::to_string(model.isConstrainedEverywhere.size()),
             " constraint flags for a g set of ", std::to_string(model.dofMap->gSetSize()),
             " DOFs; give one flag per g-set DOF"}));
    // Steps 1 and 3's roles as the full build makes them (no element matrix), then 3b's from the build of the same key.
    numberPoints(model, description, budget);
    readSpiders(description, budget);
    readRelations(model, budget);
    assignRoles(model, budget);
    const auto mismatch = [&](const char* what)
    {
        return base::InternalError(base::concatenate(
            {"UnitSystem (", unitText(m_unitId), "): the boundary data kept for its cache key do not fit the unit (",
             what,
             "): the key missed an input of its equations (a bug of O_o_Solver: report it; clearing the cache and the "
             "kept structures goes round it)"}));
    };
    m_singularities =
        base::BudgetedBuffer<assembly::SingularityRow>(budget, data.singularities.size(), "unit singularity table");
    for (std::size_t k = 0; k < data.singularities.size(); ++k)
    {
        const assembly::SingularityRow& row = data.singularities[k];
        m_singularities[k] = row;
        if (!row.isConstrained)
            continue;
        const std::optional<std::int64_t> local = localDofOf(row.dof);
        if (!local || m_roles[static_cast<std::size_t>(*local)] != UnitDofRole::Interior)
            throw mismatch("a DOF its AutoSingularityConstr holds is not one of its interior DOFs");
        m_roles[static_cast<std::size_t>(*local)] = UnitDofRole::Constrained;
    }
    m_largestFreeDiagonal = data.largestFreeDiagonal;
    m_rawBlocks = base::BudgetedBuffer<RawBoundaryBlock>(budget, data.rawBlocks.size(), "unit raw boundary blocks");
    std::copy(data.rawBlocks.data(), data.rawBlocks.data() + data.rawBlocks.size(), m_rawBlocks.data());
    m_rawValues = base::BudgetedBuffer<double>(budget, data.rawValues.size(), "unit raw boundary values");
    std::copy(data.rawValues.data(), data.rawValues.data() + data.rawValues.size(), m_rawValues.data());
    m_boundaryElements =
        base::BudgetedBuffer<base::EntityId>(budget, data.boundaryElements.size(), "unit boundary elements");
    std::copy(data.boundaryElements.data(), data.boundaryElements.data() + data.boundaryElements.size(),
              m_boundaryElements.data());
    // The DOFs held in every subcase, increasing (as keepEnforcedCouplings lists them), and whether each is coupled.
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    std::size_t constrained = 0;
    for (std::size_t dof = 0; dof < dofCount; ++dof)
        constrained += m_roles[dof] == UnitDofRole::Constrained ? 1 : 0;
    if (constrained != data.isHeldCoupled.size())
        throw mismatch("its DOFs held in every subcase");
    m_constrainedDofs = base::BudgetedBuffer<std::int64_t>(budget, constrained, "unit held DOFs");
    m_isHeldCoupled = base::BudgetedBuffer<std::uint8_t>(budget, constrained, "unit held DOF couplings");
    for (std::size_t dof = 0, next = 0; dof < dofCount; ++dof)
        if (m_roles[dof] == UnitDofRole::Constrained)
        {
            m_isHeldCoupled[next] = data.isHeldCoupled[next];
            m_constrainedDofs[next++] = static_cast<std::int64_t>(dof);
        }
    // The boundary equations: their labels (the rows of K*) and local DOFs.
    if (data.boundaryLabels.size() != data.boundaryLocalDofs.size())
        throw mismatch("its boundary equations");
    m_boundaryLabels = base::BudgetedBuffer<UnitDofLabel>(budget, data.boundaryLabels.size(), "unit boundary labels");
    m_boundaryLocalDofs =
        base::BudgetedBuffer<std::int64_t>(budget, data.boundaryLocalDofs.size(), "unit boundary DOFs");
    for (std::size_t k = 0; k < data.boundaryLabels.size(); ++k)
    {
        const std::int64_t local = data.boundaryLocalDofs[k];
        if (local < 0 || local >= localDofCount() ||
            m_roles[static_cast<std::size_t>(local)] != UnitDofRole::Boundary ||
            !(labelOf(local) == data.boundaryLabels[k]))
            throw mismatch("its boundary equations");
        m_boundaryLabels[k] = data.boundaryLabels[k];
        m_boundaryLocalDofs[k] = local;
    }
    m_interiorEquationCount = data.interiorDofCount;
    m_rigidBodyMass = data.rigidBodyMass;
}

void UnitSystem::requireEquations(const char* what) const
{
    if (!m_hasEquations)
        throw base::InternalError(base::concatenate(
            {"UnitSystem (", unitText(m_unitId), "): ", what,
             " needs the unit's equations, which a unit taken from the cache without them does not have (a bug of "
             "O_o_Solver: report it)"}));
}

std::int64_t UnitSystem::localDofOfEquation(std::int32_t equation) const
{
    if (m_hasEquations)
        return m_selection.dofOfEquation(equation);
    if (equation < m_interiorEquationCount || equation >= equationCount())
        requireEquations("the numbering of its interior equations");
    return m_boundaryLocalDofs[static_cast<std::size_t>(equation - m_interiorEquationCount)];
}

std::int32_t UnitSystem::equationOfLocalDof(std::int64_t localDof) const
{
    requireEquations("the equation of a local DOF");
    return m_selection.equationOfDof(localDof);
}

const linalg::SymbolicFactor& UnitSystem::symbolic() const
{
    requireEquations("its symbolic factor");
    return m_symbolic;
}

const linalg::SymmetricCsc& UnitSystem::matrix() const
{
    requireEquations("its matrix");
    return m_matrix;
}

const linalg::DenseMatrix& UnitSystem::inertialLoads() const
{
    requireEquations("its inertial loads (its mass)");
    if (m_inertialLoads.rows() != static_cast<std::size_t>(localDofCount()))
        throw base::InternalError(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): its inertial loads need its mass, which its equations built again for its recovery leave out (a bug "
             "of O_o_Solver: report it)"}));
    return m_inertialLoads;
}

UnitBoundaryData UnitSystem::boundaryData(base::MemoryBudget& budget) const
{
    requireEquations("its boundary data");
    UnitBoundaryData data;
    data.singularities =
        base::BudgetedBuffer<assembly::SingularityRow>(budget, m_singularities.size(), "kept unit singularity table");
    std::copy(m_singularities.data(), m_singularities.data() + m_singularities.size(), data.singularities.data());
    data.largestFreeDiagonal = m_largestFreeDiagonal;
    data.rawBlocks =
        base::BudgetedBuffer<RawBoundaryBlock>(budget, m_rawBlocks.size(), "kept unit raw boundary blocks");
    std::copy(m_rawBlocks.data(), m_rawBlocks.data() + m_rawBlocks.size(), data.rawBlocks.data());
    data.rawValues = base::BudgetedBuffer<double>(budget, m_rawValues.size(), "kept unit raw boundary values");
    std::copy(m_rawValues.data(), m_rawValues.data() + m_rawValues.size(), data.rawValues.data());
    data.boundaryElements =
        base::BudgetedBuffer<base::EntityId>(budget, m_boundaryElements.size(), "kept unit boundary elements");
    std::copy(m_boundaryElements.data(), m_boundaryElements.data() + m_boundaryElements.size(),
              data.boundaryElements.data());
    data.isHeldCoupled = base::BudgetedBuffer<std::uint8_t>(budget, m_isHeldCoupled.size(), "kept unit held DOFs");
    std::copy(m_isHeldCoupled.data(), m_isHeldCoupled.data() + m_isHeldCoupled.size(), data.isHeldCoupled.data());
    data.boundaryLabels =
        base::BudgetedBuffer<UnitDofLabel>(budget, m_boundaryLabels.size(), "kept unit boundary labels");
    std::copy(m_boundaryLabels.data(), m_boundaryLabels.data() + m_boundaryLabels.size(), data.boundaryLabels.data());
    data.boundaryLocalDofs =
        base::BudgetedBuffer<std::int64_t>(budget, m_boundaryLabels.size(), "kept unit boundary DOFs");
    for (std::size_t k = 0; k < m_boundaryLabels.size(); ++k)
        data.boundaryLocalDofs[k] = m_selection.dofOfEquation(m_interiorEquationCount + static_cast<std::int32_t>(k));
    data.interiorDofCount = m_interiorEquationCount;
    data.rigidBodyMass = m_rigidBodyMass;
    return data;
}

bool UnitBoundaryData::isEqual(const UnitBoundaryData& other) const noexcept
{
    const auto sameBits = [](double a, double b)
    { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); };
    const auto sameRow = [&](const assembly::SingularityRow& a, const assembly::SingularityRow& b)
    {
        return a.dof == b.dof && a.pointId == b.pointId && a.isScalarPoint == b.isScalarPoint &&
               a.component == b.component && sameBits(a.stiffnessRatio, b.stiffnessRatio) &&
               a.oldExclusive == b.oldExclusive && a.isConstrained == b.isConstrained && a.isAligned == b.isAligned;
    };
    const auto sameBlock = [](const RawBoundaryBlock& a, const RawBoundaryBlock& b)
    { return a.rowNode == b.rowNode && a.columnNode == b.columnNode && a.valueOffset == b.valueOffset; };
    const auto equalSpans = [](auto a, auto b, auto&& same)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t k = 0; k < a.size(); ++k)
            if (!same(a[k], b[k]))
                return false;
        return true;
    };
    const auto equal = [](const auto& a, const auto& b) { return a == b; };
    bool isSame = equalSpans(singularities.span(), other.singularities.span(), sameRow) &&
                  sameBits(largestFreeDiagonal, other.largestFreeDiagonal) &&
                  equalSpans(rawBlocks.span(), other.rawBlocks.span(), sameBlock) &&
                  equalSpans(rawValues.span(), other.rawValues.span(), sameBits) &&
                  equalSpans(boundaryElements.span(), other.boundaryElements.span(), equal) &&
                  equalSpans(isHeldCoupled.span(), other.isHeldCoupled.span(), equal) &&
                  equalSpans(boundaryLabels.span(), other.boundaryLabels.span(), equal) &&
                  equalSpans(boundaryLocalDofs.span(), other.boundaryLocalDofs.span(), equal) &&
                  interiorDofCount == other.interiorDofCount;
    for (std::size_t k = 0; k < rigidBodyMass.values.size() && isSame; ++k)
        isSame = sameBits(rigidBodyMass.values[k], other.rigidBodyMass.values[k]);
    return isSame;
}

// ---------------------------------------------------------------------------------------------------------------------
// Local numbering

std::optional<std::int32_t> UnitSystem::localPointOf(std::int32_t modelNode) const noexcept
{
    const std::int32_t* first = m_sortedModelNodes.data();
    const std::int32_t* last = first + m_sortedModelNodes.size();
    const std::int32_t* found = std::lower_bound(first, last, modelNode);
    if (found == last || *found != modelNode)
        return std::nullopt;
    return static_cast<std::int32_t>(found - first);
}

std::optional<std::int64_t> UnitSystem::localDofOfNode(std::int32_t modelNode, int component) const noexcept
{
    const std::optional<std::int32_t> point = localPointOf(modelNode);
    if (!point)
        return std::nullopt;
    return m_firstDof[static_cast<std::size_t>(*point)] + component;
}

std::optional<std::int64_t> UnitSystem::localDofOf(base::DofIndex dof) const noexcept
{
    const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(dof);
    return localDofOfNode(nodeDof.node, nodeDof.component);
}

std::int32_t UnitSystem::pointOfLocalDof(std::int64_t localDof) const noexcept
{
    const std::int64_t* first = m_firstDof.data();
    return static_cast<std::int32_t>(std::upper_bound(first, first + m_firstDof.size(), localDof) - first) - 1;
}

base::DofIndex UnitSystem::modelDofOfNode(std::int32_t modelNode, int component) const noexcept
{
    return modelNode < m_dofMap->gridCount() ? m_dofMap->gridDof(modelNode, component + 1)
                                             : m_dofMap->scalarDof(modelNode - m_dofMap->gridCount());
}

std::optional<base::DofIndex> UnitSystem::modelDofOf(std::int64_t localDof) const noexcept
{
    const std::int32_t point = pointOfLocalDof(localDof);
    const std::int32_t node = m_modelNodes[static_cast<std::size_t>(point)];
    if (node == GENERATED_POINT)
        return std::nullopt;
    return modelDofOfNode(node, static_cast<int>(localDof - m_firstDof[static_cast<std::size_t>(point)]));
}

base::EntityId UnitSystem::pointIdOf(std::int32_t localPoint) const noexcept
{
    const std::int32_t node = m_modelNodes[static_cast<std::size_t>(localPoint)];
    if (node == GENERATED_POINT)
        return m_fastenerPoints[static_cast<std::size_t>(localPoint) - m_sortedModelNodes.size()].id;
    return node < m_dofMap->gridCount() ? m_dofMap->gridId(node) : m_dofMap->scalarId(node - m_dofMap->gridCount());
}

UnitDofLabel UnitSystem::labelOf(std::int64_t localDof) const noexcept
{
    const std::int32_t point = pointOfLocalDof(localDof);
    const auto component = static_cast<int>(localDof - m_firstDof[static_cast<std::size_t>(point)]);
    const std::int32_t node = m_modelNodes[static_cast<std::size_t>(point)];
    if (node == GENERATED_POINT)
    {
        UnitDofLabel label = m_fastenerPoints[static_cast<std::size_t>(point) - m_sortedModelNodes.size()];
        label.component = static_cast<std::uint8_t>(component + 1);
        return label;
    }
    if (node < m_dofMap->gridCount())
        return {UnitPointKind::Grid, m_dofMap->gridId(node), FastenerSide::None,
                static_cast<std::uint8_t>(component + 1)};
    return {UnitPointKind::Scalar, m_dofMap->scalarId(node - m_dofMap->gridCount()), FastenerSide::None, 0};
}

void UnitSystem::forEachCoupledDof(std::int32_t point, const std::function<void(std::int64_t)>& visit) const
{
    const std::span<const std::int32_t> neighbours = m_graph.neighbours(point);
    bool isOwnVisited = false;
    const auto visitPoint = [&](std::int32_t p)
    {
        for (std::int64_t dof = m_firstDof[static_cast<std::size_t>(p)];
             dof < m_firstDof[static_cast<std::size_t>(p) + 1]; ++dof)
            visit(dof);
    };
    for (const std::int32_t neighbour : neighbours)
    {
        if (!isOwnVisited && neighbour > point)
        {
            visitPoint(point);
            isOwnVisited = true;
        }
        visitPoint(neighbour);
    }
    if (!isOwnVisited)
        visitPoint(point);
}

base::BudgetedBuffer<std::int32_t> unitPointNodes(const assembly::DofMap& dofMap, const UnitDescription& description,
                                                  base::MemoryBudget& budget)
{
    base::BudgetedBuffer<std::int32_t> nodes(budget, description.nodeIds.size(), "unit points");
    for (std::size_t k = 0; k < description.nodeIds.size(); ++k)
    {
        const std::optional<input::PointRef> point = dofMap.points().find(description.nodeIds[k]);
        if (!point)
            throw std::invalid_argument(base::concatenate(
                {"UnitSystem (", unitText(description.unitId), "): point ", std::to_string(description.nodeIds[k]),
                 " is not a grid or scalar point of the model; the unit and the model must come "
                 "from one snapshot"}));
        nodes[k] = point->kind == input::PointKind::Grid ? point->row : dofMap.gridCount() + point->row;
    }
    std::sort(nodes.data(), nodes.data() + nodes.size());
    if (std::adjacent_find(nodes.data(), nodes.data() + nodes.size()) != nodes.data() + nodes.size())
        throw std::invalid_argument(
            base::concatenate({"UnitSystem (", unitText(description.unitId),
                               "): a point is listed twice; list each point of the unit once"}));
    return nodes;
}

base::BudgetedBuffer<base::DofIndex> unitLocalModelDofs(const assembly::DofMap& dofMap,
                                                        const UnitDescription& description, base::MemoryBudget& budget)
{
    const base::BudgetedBuffer<std::int32_t> nodes = unitPointNodes(dofMap, description, budget);
    std::size_t count = 0;
    for (const std::int32_t node : nodes.span())
        count += node < dofMap.gridCount() ? GRID_DOF_COUNT : 1;
    for (const UnitSpider& spider : description.spiders)
        count += spider.relation.referenceDofs.empty() ? GRID_DOF_COUNT : 0;
    base::BudgetedBuffer<base::DofIndex> dofs(budget, count, "unit local DOFs");
    std::size_t next = 0;
    for (const std::int32_t node : nodes.span())
    {
        if (node >= dofMap.gridCount())
        {
            dofs[next++] = dofMap.scalarDof(node - dofMap.gridCount());
            continue;
        }
        for (int c = 0; c < GRID_DOF_COUNT; ++c)
            dofs[next++] = dofMap.gridDof(node, c + 1);
    }
    for (; next < count; ++next)
        dofs[next] = -1;
    return dofs;
}

void UnitSystem::numberPoints(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget)
{
    const assembly::DofMap& dofMap = *model.dofMap;
    const auto nodeOf = [&](const input::PointRef& point)
    { return point.kind == input::PointKind::Grid ? point.row : dofMap.gridCount() + point.row; };
    // The unit's points in the order of their nodes in the model, then one point per generated fastener point.
    base::BudgetedBuffer<std::int32_t> nodes = unitPointNodes(dofMap, description, budget);
    std::size_t generated = 0;
    for (const UnitSpider& spider : description.spiders)
        generated += spider.relation.referenceDofs.empty() ? 1 : 0;
    const std::size_t pointTotal = nodes.size() + generated;
    m_modelNodes = base::BudgetedBuffer<std::int32_t>(budget, pointTotal, "unit points");
    m_pointDofCounts = base::BudgetedBuffer<std::uint8_t>(budget, pointTotal, "unit point DOF counts");
    m_positions = base::BudgetedBuffer<base::Vec3>(budget, pointTotal, "unit point positions");
    m_firstDof = base::BudgetedBuffer<std::int64_t>(budget, pointTotal + 1, "unit first DOFs");
    m_sortedModelNodes = std::move(nodes);
    for (std::size_t p = 0; p < m_sortedModelNodes.size(); ++p)
    {
        const std::int32_t node = m_sortedModelNodes[p];
        m_modelNodes[p] = node;
        const bool isGrid = node < dofMap.gridCount();
        m_pointDofCounts[p] = isGrid ? GRID_DOF_COUNT : 1;
        m_positions[p] = isGrid ? dofMap.positionInBasic(node) : base::Vec3{};
    }
    std::size_t next = m_sortedModelNodes.size();
    for (const UnitSpider& spider : description.spiders)
    {
        if (!spider.relation.referenceDofs.empty())
            continue;
        m_modelNodes[next] = GENERATED_POINT;
        m_pointDofCounts[next] = GRID_DOF_COUNT;
        m_positions[next] = spider.generatedPointPosition;
        m_fastenerPoints.push_back({UnitPointKind::FastenerPoint, spider.key.elementId, spider.key.side, 1});
        ++next;
    }
    for (std::size_t p = 0; p < pointTotal; ++p)
        m_firstDof[p + 1] = m_firstDof[p] + m_pointDofCounts[p];

    m_isBoundaryPoint = base::BudgetedBuffer<std::uint8_t>(budget, pointTotal, "unit boundary points");
    for (const base::EntityId nodeId : description.boundaryNodeIds)
    {
        const std::optional<input::PointRef> point = dofMap.points().find(nodeId);
        const std::optional<std::int32_t> local = point ? localPointOf(nodeOf(*point)) : std::nullopt;
        if (!local)
            throw std::invalid_argument(
                base::concatenate({"UnitSystem (", unitText(m_unitId), "): boundary point ", std::to_string(nodeId),
                                   " is not a point of the unit; the boundary must be among "
                                   "the unit's points"}));
        m_isBoundaryPoint[static_cast<std::size_t>(*local)] = BOUNDARY;
    }
    for (std::size_t p = m_sortedModelNodes.size(); p < pointTotal; ++p)
        m_isBoundaryPoint[p] = BOUNDARY;
}

void UnitSystem::readSpiders(const UnitDescription& description, base::MemoryBudget& budget)
{
    std::size_t generated = m_sortedModelNodes.size();
    for (const UnitSpider& spider : description.spiders)
    {
        const SpiderWeightMatrix& weights = spider.relation.weights;
        const std::string name = spiderText(spider.key);
        LocalSpider local{spider.key, SpiderChangeOfBasis(weights, budget), {}, {}, {}};
        if (local.changeOfBasis.isRankDeficient())
            throw std::invalid_argument(base::concatenate(
                {"UnitSystem (", unitText(m_unitId), "): ", name,
                 " has a rank-deficient weighting matrix and cannot be absorbed by change of basis; keep it on its "
                 "legs (AbsorptionSafeguard decides this in the plan)"}));
        if (spider.relation.legDofs.size() != weights.legDofCount)
            throw std::invalid_argument(
                base::concatenate({"UnitSystem (", unitText(m_unitId), "): ", name, " has ",
                                   std::to_string(spider.relation.legDofs.size()), " leg DOFs for a W of ",
                                   std::to_string(weights.legDofCount), " columns; give one DOF per column"}));
        for (const base::DofIndex dof : spider.relation.legDofs)
        {
            const std::optional<std::int64_t> leg = localDofOf(dof);
            if (!leg)
                throw std::invalid_argument(
                    base::concatenate({"UnitSystem (", unitText(m_unitId), "): a leg of ", name, " (point ",
                                       std::to_string(m_dofMap->pointIdOf(dof)),
                                       ") is not a point of the unit; an absorbed spider has every leg in its unit"}));
            local.legs.push_back(*leg);
            local.points.push_back(pointOfLocalDof(*leg));
        }
        std::int32_t referencePoint = 0;
        if (spider.relation.referenceDofs.empty())
        {
            if (weights.referenceDofCount != static_cast<std::size_t>(GRID_DOF_COUNT))
                throw std::invalid_argument(
                    base::concatenate({"UnitSystem (", unitText(m_unitId), "): ", name,
                                       " has a generated point but a W of ", std::to_string(weights.referenceDofCount),
                                       " rows; a fastener side gives the six motions of its point (a bug of "
                                       "O_o_Solver: report it)"}));
            for (int c = 0; c < GRID_DOF_COUNT; ++c)
                local.references.push_back(m_firstDof[generated] + c);
            referencePoint = static_cast<std::int32_t>(generated);
            ++generated;
        }
        else
        {
            if (spider.relation.referenceDofs.size() != weights.referenceDofCount)
                throw std::invalid_argument(base::concatenate(
                    {"UnitSystem (", unitText(m_unitId), "): ", name, " has ",
                     std::to_string(spider.relation.referenceDofs.size()), " reference DOFs for a W of ",
                     std::to_string(weights.referenceDofCount), " rows; give one DOF per row"}));
            for (const base::DofIndex dof : spider.relation.referenceDofs)
            {
                const std::optional<std::int64_t> reference = localDofOf(dof);
                if (!reference)
                    throw std::invalid_argument(base::concatenate(
                        {"UnitSystem (", unitText(m_unitId), "): the reference point ",
                         std::to_string(m_dofMap->pointIdOf(dof)), " of ", name,
                         " is not a point of the unit; the plan puts an absorbed spider's reference in its unit (a "
                         "bug of O_o_Solver: report it)"}));
                local.references.push_back(*reference);
            }
            referencePoint = pointOfLocalDof(local.references.front());
        }
        if (m_isBoundaryPoint[static_cast<std::size_t>(referencePoint)] != BOUNDARY)
            throw std::invalid_argument(
                base::concatenate({"UnitSystem (", unitText(m_unitId), "): the reference point ",
                                   std::to_string(pointIdOf(referencePoint)), " of ", name,
                                   " is not on the unit's boundary; the plan puts an absorbed spider's reference "
                                   "there (a bug of O_o_Solver: report it)"}));
        for (const std::int32_t point : local.points)
            if (m_isBoundaryPoint[static_cast<std::size_t>(point)] == BOUNDARY)
                throw std::invalid_argument(base::concatenate(
                    {"UnitSystem (", unitText(m_unitId), "): leg point ", std::to_string(pointIdOf(point)), " of ",
                     name,
                     " is on the unit's boundary; a spider whose leg must stay on the boundary is kept on its legs "
                     "(AbsorptionEligibility)"}));
        local.points.push_back(referencePoint);
        sortUnique(local.points);
        m_spiders.push_back(std::move(local));
    }
    // A DOF in two spiders would be transformed twice (SpiderAbsorption keeps such spiders on their legs).
    std::vector<std::int64_t> used; // bounded by the spiders' DOFs
    for (const LocalSpider& spider : m_spiders)
    {
        used.insert(used.end(), spider.legs.begin(), spider.legs.end());
        used.insert(used.end(), spider.references.begin(), spider.references.end());
    }
    std::sort(used.begin(), used.end());
    if (std::adjacent_find(used.begin(), used.end()) != used.end())
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): two absorbed spiders share a DOF (a leg, or a reference that is another's leg); such spiders are "
             "kept on their legs (SpiderAbsorption; a bug of O_o_Solver: report it)"}));
}

void UnitSystem::readRelations(const UnitModel& model, base::MemoryBudget& budget)
{
    const assembly::RigidElimination& gm = *model.unitRelations;
    // The rows of the unit's DOFs, in increasing local DOF (the points in model order, the components in order).
    std::size_t rowCount = 0;
    std::size_t termCount = 0;
    for (std::size_t p = 0; p < m_sortedModelNodes.size(); ++p)
        for (int c = 0; c < m_pointDofCounts[p]; ++c)
            if (const std::optional<std::size_t> row = gm.findRow(modelDofOfNode(m_sortedModelNodes[p], c)))
            {
                ++rowCount;
                termCount += gm.row(*row).size();
            }
    m_relations = LocalRelations(rowCount, termCount, budget, "unit Gm rows");
    m_isSpiderReferenceRow = base::BudgetedBuffer<std::uint8_t>(budget, rowCount, "unit Gm rows");
    std::vector<std::int64_t> references; // bounded by the spiders' reference DOFs
    for (const LocalSpider& spider : m_spiders)
        references.insert(references.end(), spider.references.begin(), spider.references.end());
    std::sort(references.begin(), references.end());
    std::size_t nextRow = 0;
    for (std::size_t p = 0; p < m_sortedModelNodes.size(); ++p)
        for (int c = 0; c < m_pointDofCounts[p]; ++c)
        {
            const base::DofIndex dof = modelDofOfNode(m_sortedModelNodes[p], c);
            const std::optional<std::size_t> row = gm.findRow(dof);
            if (!row)
                continue;
            const std::int64_t dependent = m_firstDof[p] + c;
            m_relations.appendRow(dependent);
            m_isSpiderReferenceRow[nextRow] =
                std::binary_search(references.begin(), references.end(), dependent) ? 1 : 0;
            for (const assembly::ConstraintTerm& term : gm.row(*row))
            {
                const std::optional<std::int64_t> local = localDofOf(term.dof);
                if (!local)
                    throw std::invalid_argument(base::concatenate(
                        {"UnitSystem (", unitText(m_unitId), "): ",
                         std::string(assembly::cardNameOf(gm.sourceOf(*row))), " ", std::to_string(gm.sourceIdOf(*row)),
                         " makes a DOF of point ", std::to_string(m_dofMap->pointIdOf(dof)), " depend on point ",
                         std::to_string(m_dofMap->pointIdOf(term.dof)),
                         ", which is not the unit's; a relation reaching another unit is eliminated at the interface "
                         "(leave it out of the Gm of the units)"}));
                m_relations.appendTerm(LocalTerm{*local, term.coefficient});
            }
            ++nextRow;
        }
}

void UnitSystem::assignRoles(const UnitModel& model, base::MemoryBudget& budget)
{
    m_roles = base::BudgetedBuffer<UnitDofRole>(budget, static_cast<std::size_t>(localDofCount()), "unit DOF roles");
    for (std::size_t p = 0; p < m_pointDofCounts.size(); ++p)
        for (int c = 0; c < m_pointDofCounts[p]; ++c)
        {
            UnitDofRole role = m_isBoundaryPoint[p] == BOUNDARY ? UnitDofRole::Boundary : UnitDofRole::Interior;
            const std::int32_t node = m_modelNodes[p];
            if (node != GENERATED_POINT && !model.isConstrainedEverywhere.empty() &&
                model.isConstrainedEverywhere[static_cast<std::size_t>(modelDofOfNode(node, c))] != 0)
                role = UnitDofRole::Constrained;
            m_roles[static_cast<std::size_t>(m_firstDof[p] + c)] = role;
        }
    // A dependent DOF is eliminated (as DofSets keeps a DOF in m when an SPC names it too: the full path reports it).
    for (std::size_t r = 0; r < m_relations.rowCount(); ++r)
        if (m_isSpiderReferenceRow[r] == 0)
            m_roles[static_cast<std::size_t>(m_relations.dependent(r))] = UnitDofRole::Dependent;
    for (const LocalSpider& spider : m_spiders)
    {
        const std::size_t m = spider.references.size();
        for (const std::int64_t reference : spider.references)
        {
            if (m_roles[static_cast<std::size_t>(reference)] == UnitDofRole::Constrained)
                throw std::invalid_argument(base::concatenate({"UnitSystem (", unitText(m_unitId),
                                                               "): a reference DOF of ", spiderText(spider.key),
                                                               " is held in every subcase; such a spider is kept on "
                                                               "its legs (a bug of O_o_Solver: report it)"}));
            m_roles[static_cast<std::size_t>(reference)] = UnitDofRole::Boundary;
        }
        for (std::size_t k = 0; k < spider.legs.size(); ++k)
        {
            UnitDofRole& role = m_roles[static_cast<std::size_t>(spider.legs[k])];
            if (role != UnitDofRole::Interior)
                throw std::invalid_argument(base::concatenate(
                    {"UnitSystem (", unitText(m_unitId), "): a leg DOF of ", spiderText(spider.key),
                     " is held in every subcase or dependent elsewhere; such a spider is kept on its legs "
                     "(AbsorptionEligibility; a bug of O_o_Solver: report it)"}));
            role = k < m ? UnitDofRole::Absent : UnitDofRole::Interior;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Matrices

namespace
{

// The unit's elements that have a matrix, in the model's order of batches and rows (the order of every sum of the
// full path's assembly), and their matrices computed run by run of consecutive rows.
class UnitElements
{
public:
    UnitElements(const UnitModel& model, std::span<const base::EntityId> elementIds, base::MemoryBudget& budget)
        : m_set(model.elements)
    {
        std::size_t count = 0;
        for (const base::EntityId elementId : elementIds)
            count += model.elementIndex->find(elementId) ? 1 : 0;
        m_places = base::BudgetedBuffer<ElementPlace>(budget, count, "unit elements");
        std::size_t next = 0;
        for (const base::EntityId elementId : elementIds)
            if (const std::optional<ElementPlace> place = model.elementIndex->find(elementId))
                m_places[next++] = *place;
        std::sort(m_places.data(), m_places.data() + count, [](const ElementPlace& a, const ElementPlace& b)
                  { return a.batchIndex != b.batchIndex ? a.batchIndex < b.batchIndex : a.row < b.row; });
    }

    [[nodiscard]] std::size_t size() const noexcept { return m_places.size(); }
    // The row of the element of list index k.
    [[nodiscard]] std::int64_t rowOf(std::size_t k) const noexcept { return m_places[k].row; }

    // Calls visit(batch, block, first) with the matrices of the elements [begin, end) of the list, run by run (each run
    // one batch, consecutive rows, at most ELEMENTS_PER_RUN elements: the block is a work buffer bounded by a run, not
    // by the unit; `first` the list index of its first element), facet(batchIndex, batch, range, block) computing them.
    // The list index of the element of a matrix of the block is first + (its row − the run's first row).
    template <typename Facet, typename Visit>
    void compute(std::size_t begin, std::size_t end, Facet&& facet, Visit&& visit) const
    {
        elements::ElementMatrixBlock block;
        std::size_t k = begin;
        while (k < end)
        {
            std::size_t last = k + 1;
            while (last < end && last - k < ELEMENTS_PER_RUN && m_places[last].batchIndex == m_places[k].batchIndex &&
                   m_places[last].row == m_places[last - 1].row + 1)
                ++last;
            const assembly::AssemblyBatch& batch = m_set->batches()[m_places[k].batchIndex];
            block.clear();
            facet(static_cast<std::size_t>(m_places[k].batchIndex), batch,
                  elements::ElementRange{m_places[k].row, m_places[last - 1].row + 1}, block);
            visit(batch, block, k);
            k = last;
        }
    }

private:
    const assembly::ElementSet* m_set;
    base::BudgetedBuffer<ElementPlace> m_places;
};

void computeStiffnessOrMass(std::size_t, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                            elements::ElementMatrixBlock& block)
{
    if (batch.family->facets().has(elements::Facet::Stiffness))
        batch.family->computeStiffness(batch.view, range, block);
    else if (batch.family->facets().has(elements::Facet::Mass))
        batch.family->computeMass(batch.view, range, elements::MassOptions{}, block);
}

// A pair (point, point it is coupled to) of the units' Gm rows: a dependent point and the point of one of its terms.
struct PointPair
{
    std::int32_t point = 0;
    std::int32_t other = 0;

    friend constexpr bool operator==(const PointPair&, const PointPair&) = default;
};

} // namespace

void UnitSystem::buildGraph(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                            base::ThreadPool* pool, KeptStiffness& kept)
{
    const UnitElements elements(model, description.elementIds, budget);
    const assembly::DofMap& dofMap = *model.dofMap;

    // 1. The points each element couples: the DOFs of its stiffness, else of its mass (as ElementConnectivity); the
    // elements with a point on the boundary.
    // PERFORMANCE: the stiffness matrices computed here are kept for assemble() (KeptStiffness), which computed every
    // element's stiffness a second time: a third of the element work of a unit's equations, and the equations are
    // the most of a cache hit (the key's checksum reads them). The same matrices in the same order: the same bits.
    // Measured on a condensed analysis of a large model.
    GroupLists elementGroups(budget);
    GrowingArray<base::EntityId> boundaryElements(budget, "unit boundary elements");
    std::vector<std::int32_t> group; // bounded by an element
    const std::uint64_t keptLimit = budget.available() / KEPT_STIFFNESS_MEMORY_DIVISOR;
    std::uint64_t keptBytes = 0;
    kept.isKept = true;
    kept.firstOfElement.assign(elements.size() + 1, 0);
    std::vector<linalg::NodeDof> localDofs; // bounded by an element
    elements.compute(
        0, elements.size(), computeStiffnessOrMass,
        [&](const assembly::AssemblyBatch& batch, const elements::ElementMatrixBlock& block, std::size_t first)
        {
            const bool isStiffness = batch.family->facets().has(elements::Facet::Stiffness);
            for (std::size_t e = 0; e < block.size(); ++e)
            {
                group.clear();
                localDofs.clear();
                for (const elements::ElementDof& dof : block.dofs(e))
                {
                    const linalg::NodeDof nodeDof = dofMap.nodeDofOf(dofMap.dofOf(dof));
                    const std::optional<std::int32_t> point = localPointOf(nodeDof.node);
                    if (!point)
                        throw std::invalid_argument(base::concatenate(
                            {"UnitSystem (", unitText(m_unitId), "): element ",
                             std::to_string(batch.view.elementId(block.elementRow(e))), " (", batch.view.card(),
                             ") uses point ", std::to_string(dofMap.pointIdOf(dofMap.dofOf(dof))),
                             ", which is not the unit's; a unit holds every point of its elements (a bug of "
                             "O_o_Solver: report it)"}));
                    group.push_back(*point);
                    localDofs.push_back({*point, nodeDof.component});
                }
                if (kept.isKept && isStiffness)
                {
                    keptBytes += localDofs.size() * sizeof(linalg::NodeDof) + block.matrix(e).size() * sizeof(double) +
                                 2 * sizeof(std::size_t);
                    if (keptBytes > keptLimit)
                    {
                        kept.isKept = false;
                        kept.block = linalg::ContributionBlock{};
                    }
                    else
                    {
                        kept.block.append(localDofs, block.matrix(e));
                        ++kept.firstOfElement[first +
                                              static_cast<std::size_t>(block.elementRow(e) - elements.rowOf(first)) +
                                              1];
                    }
                }
                sortUnique(group);
                if (std::any_of(group.begin(), group.end(), [&](std::int32_t point)
                                { return m_isBoundaryPoint[static_cast<std::size_t>(point)] == BOUNDARY; }))
                    boundaryElements.push(batch.view.elementId(block.elementRow(e)));
                if (group.size() > 1)
                    elementGroups.add(group);
            }
        });
    if (kept.isKept)
    {
        for (std::size_t k = 0; k < elements.size(); ++k)
            kept.firstOfElement[k + 1] += kept.firstOfElement[k];
        try
        {
            kept.reservation = budget.reserve(keptBytes, "unit element stiffness kept for the assembly");
        }
        catch (const base::MemoryBudgetError&)
        {
            // An optimisation: without room, assemble() computes the matrices again.
            kept.isKept = false;
        }
    }
    if (!kept.isKept)
    {
        kept.block = linalg::ContributionBlock{};
        kept.firstOfElement = {};
    }
    {
        base::BudgetedBuffer<base::EntityId> sorted(budget, boundaryElements.size(), "unit boundary elements");
        std::copy(boundaryElements.span().begin(), boundaryElements.span().end(), sorted.data());
        std::sort(sorted.data(), sorted.data() + sorted.size());
        const auto distinct =
            static_cast<std::size_t>(std::unique(sorted.data(), sorted.data() + sorted.size()) - sorted.data());
        m_boundaryElements = base::BudgetedBuffer<base::EntityId>(budget, distinct, "unit boundary elements");
        std::copy(sorted.data(), sorted.data() + distinct, m_boundaryElements.data());
    }

    // 2. Each group extended with the points of the Gm rows of its dependent points (MsetCouplings): Knn's terms.
    base::BudgetedBuffer<PointPair> pairs(budget, m_relations.termCount(), "unit Gm couplings");
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
    // 3. Then with every leg and the reference point of each spider any of whose points the group holds.
    base::BudgetedBuffer<std::int32_t> spiderOfPoint(budget, m_pointDofCounts.size(), "unit spider points");
    std::fill(spiderOfPoint.data(), spiderOfPoint.data() + spiderOfPoint.size(), NO_SPIDER);
    for (std::size_t s = 0; s < m_spiders.size(); ++s)
        for (const std::int32_t point : m_spiders[s].points)
            spiderOfPoint[static_cast<std::size_t>(point)] = static_cast<std::int32_t>(s);
    GroupLists groups(budget);
    const auto addExtended = [&](std::span<const std::int32_t> points)
    {
        group.assign(points.begin(), points.end());
        const std::size_t own = group.size();
        for (std::size_t k = 0; k < own; ++k)
            for (const PointPair& pair : termPointsOf(group[k]))
                group.push_back(pair.other);
        const std::size_t extended = group.size();
        for (std::size_t k = 0; k < extended; ++k)
            if (const std::int32_t spider = spiderOfPoint[static_cast<std::size_t>(group[k])]; spider != NO_SPIDER)
            {
                const std::vector<std::int32_t>& spiderPoints = m_spiders[static_cast<std::size_t>(spider)].points;
                group.insert(group.end(), spiderPoints.begin(), spiderPoints.end());
            }
        sortUnique(group);
        groups.add(group);
    };
    const linalg::NodeGroupList elementList = elementGroups.view();
    for (std::size_t g = 0; g < elementList.groupCount(); ++g)
        addExtended(elementList.group(g));
    // Each dependent point alone too (as MsetCouplings): its own diagonal block, K(d, e) of two DOFs of one point (a
    // grounded spring or a mass on it, which makes no group of two points), couples the points of their rows.
    for (std::size_t k = 0; k < pairCount; ++k)
        if (k == 0 || pairs[k].point != pairs[k - 1].point)
            addExtended(std::span<const std::int32_t>(&pairs[k].point, 1));
    for (const LocalSpider& spider : m_spiders)
        groups.add(spider.points);
    const std::vector<linalg::NodeGroupList> lists{groups.view()};
    m_graph = linalg::NodeGraph(pointCount(), lists, budget, pool);

    // A scalar point takes the position of a grid point it is coupled to (FsetOrdering does the same).
    for (std::size_t p = 0; p < m_sortedModelNodes.size(); ++p)
    {
        if (m_sortedModelNodes[p] < dofMap.gridCount())
            continue;
        for (const std::int32_t neighbour : m_graph.neighbours(static_cast<std::int32_t>(p)))
            if (m_modelNodes[static_cast<std::size_t>(neighbour)] != GENERATED_POINT &&
                m_modelNodes[static_cast<std::size_t>(neighbour)] < dofMap.gridCount())
            {
                m_positions[p] = m_positions[static_cast<std::size_t>(neighbour)];
                break;
            }
    }
}

void UnitSystem::assemble(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                          base::ThreadPool* pool, const KeptStiffness& kept, bool assemblesMass)
{
    const UnitElements elements(model, description.elementIds, budget);
    const assembly::DofMap& dofMap = *model.dofMap;
    m_pattern = std::make_unique<linalg::SparsePattern>(m_graph, m_pointDofCounts.span(), budget);
    m_stiffness = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "unit stiffness");

    // The element matrices on the local points, in the order of the list (the model's order of batches and rows).
    const auto writeOnPoints = [&](const elements::ElementMatrixBlock& block, linalg::ContributionBlock& contributions)
    {
        std::vector<linalg::NodeDof> nodeDofs; // bounded by an element
        for (std::size_t e = 0; e < block.size(); ++e)
        {
            nodeDofs.clear();
            for (const elements::ElementDof& dof : block.dofs(e))
            {
                const linalg::NodeDof nodeDof = dofMap.nodeDofOf(dofMap.dofOf(dof));
                nodeDofs.push_back({*localPointOf(nodeDof.node), nodeDof.component});
            }
            contributions.append(nodeDofs, block.matrix(e));
        }
    };
    linalg::SparseAssembler(model.assembly)
        .assemble(
            elements.size(),
            [&](std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions)
            {
                if (kept.isKept)
                {
                    for (std::size_t c = kept.firstOfElement[begin]; c < kept.firstOfElement[end]; ++c)
                        contributions.append(kept.block.dofs(c), kept.block.matrix(c));
                    return;
                }
                elements.compute(
                    begin, end,
                    [](std::size_t, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                       elements::ElementMatrixBlock& block)
                    {
                        if (batch.family->facets().has(elements::Facet::Stiffness))
                            batch.family->computeStiffness(batch.view, range, block);
                    },
                    [&](const assembly::AssemblyBatch&, const elements::ElementMatrixBlock& block, std::size_t)
                    { writeOnPoints(block, contributions); });
            },
            *m_stiffness, pool);
    if (!assemblesMass)
        return;
    auto massBlocks = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "unit mass");
    linalg::BlockSparseMatrix& mass = *massBlocks;
    const assembly::MassAssemblyOptions massOptions = model.massOptions;
    const assembly::NonStructuralMassShares* shares = model.nonStructuralMass;
    linalg::SparseAssembler(model.assembly)
        .assemble(
            elements.size(),
            [&](std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions)
            {
                elements.compute(
                    begin, end,
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
                    [&](const assembly::AssemblyBatch&, const elements::ElementMatrixBlock& block, std::size_t)
                    { writeOnPoints(block, contributions); });
            },
            mass, pool);
    // PARAM WTMASS multiplies the structural mass once assembled (as assembly::MassAssembler).
    if (massOptions.weightToMass != 1.0)
        for (double& value : mass.values())
            value *= massOptions.weightToMass;
    computeMassProperties(mass, budget);
    if (model.keepsMass)
        m_massBlocks = std::move(massBlocks);
}

linalg::DenseMatrix UnitSystem::makeRigidBodyModes(base::MemoryBudget& budget) const
{
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    linalg::DenseMatrix modes(budget, dofCount, RIGID_BODY_MODE_COUNT, "unit rigid-body modes");
    for (std::size_t p = 0; p < m_pointDofCounts.size(); ++p)
    {
        const std::int32_t node = m_modelNodes[p];
        const bool isGenerated = node == GENERATED_POINT;
        if (!isGenerated && node >= m_dofMap->gridCount())
            continue; // a scalar point does not move with the body
        // RigidBodyModes' motion about the basic origin: u = t + θ × x, written in the point's displacement system
        // (u_cd = Rᵀ·u_basic); a generated fastener point's DOFs are in the basic system.
        const base::Mat3 toCd = isGenerated ? base::Mat3::identity() : m_dofMap->cdToBasic(node).transposed();
        const base::Vec3 x = m_positions[p];
        const auto first = static_cast<std::size_t>(m_firstDof[p]);
        for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
        {
            base::Vec3 translation{};
            base::Vec3 rotation{};
            if (mode < TRANSLATION_COUNT)
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
                modes(first + k, mode) = u[k];
                modes(first + TRANSLATION_COUNT + k, mode) = theta[k];
            }
        }
    }
    return modes;
}

void UnitSystem::computeMassProperties(const linalg::BlockSparseMatrix& mass, base::MemoryBudget& budget)
{
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    // Φ_r for the mass properties only: not kept (48 bytes a DOF), made again when read.
    const linalg::DenseMatrix rigidBodyModes = makeRigidBodyModes(budget);
    // −M·Φ_r, and Φ_rᵀ·M·Φ_r summed in the order of the DOFs (made exactly symmetric from its lower triangle).
    m_inertialLoads = linalg::DenseMatrix(budget, dofCount, RIGID_BODY_MODE_COUNT, "unit inertial loads");
    base::BudgetedBuffer<double> column(budget, dofCount, "unit rigid-body mode");
    base::BudgetedBuffer<double> product(budget, dofCount, "unit mass times mode");
    for (std::size_t j = 0; j < RIGID_BODY_MODE_COUNT; ++j)
    {
        for (std::size_t dof = 0; dof < dofCount; ++dof)
            column[dof] = rigidBodyModes(dof, j);
        mass.multiply(column.span(), product.span());
        for (std::size_t dof = 0; dof < dofCount; ++dof)
            m_inertialLoads(dof, j) = -product[dof];
        for (std::size_t i = j; i < RIGID_BODY_MODE_COUNT; ++i)
        {
            double sum = 0.0;
            for (std::size_t dof = 0; dof < dofCount; ++dof)
                sum += rigidBodyModes(dof, i) * product[dof];
            m_rigidBodyMass(i, j) = sum;
            m_rigidBodyMass(j, i) = sum;
        }
    }
}

void UnitSystem::eliminateDependents(base::MemoryBudget& budget)
{
    // Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm (linear elimination; assembly::MsetElimination), in place, one
    // serial pass in a fixed order (LocalRelations::eliminate): the bits do not depend on the threads.
    m_relations.eliminate(*m_pattern, m_firstDof.span(), m_graph, m_stiffness->values(), budget);
    // The mass of Craig-Bampton the same way (the m set of every matrix is eliminated by the same Gm).
    if (m_massBlocks)
        m_relations.eliminate(*m_pattern, m_firstDof.span(), m_graph, m_massBlocks->values(), budget);
}

void UnitSystem::keepRawBoundaryStiffness(base::MemoryBudget& budget)
{
    // The DOFs whose rows the unit's relations folded onto others (dependent DOFs, the reference DOFs of the spiders):
    // their stale terms are left out (zero), as the full path's Knn has no row for them.
    base::BudgetedBuffer<std::uint8_t> isFolded(budget, static_cast<std::size_t>(localDofCount()), "unit folded DOFs");
    for (std::size_t r = 0; r < m_relations.rowCount(); ++r)
        isFolded[static_cast<std::size_t>(m_relations.dependent(r))] = 1;
    const auto isRawPoint = [&](std::int32_t point)
    {
        return m_isBoundaryPoint[static_cast<std::size_t>(point)] == BOUNDARY &&
               m_modelNodes[static_cast<std::size_t>(point)] != GENERATED_POINT;
    };
    std::size_t blockCount = 0;
    std::size_t valueCount = 0;
    for (std::int32_t column = 0; column < pointCount(); ++column)
    {
        if (!isRawPoint(column))
            continue;
        for (std::int64_t block = m_pattern->columnBegin(column); block < m_pattern->columnBegin(column + 1); ++block)
        {
            const std::int32_t row = m_pattern->blockRow(block);
            if (!isRawPoint(row))
                continue;
            ++blockCount;
            valueCount += static_cast<std::size_t>(m_pattern->nodeDofCount(row) * m_pattern->nodeDofCount(column));
        }
    }
    m_rawBlocks = base::BudgetedBuffer<RawBoundaryBlock>(budget, blockCount, "unit raw boundary blocks");
    m_rawValues = base::BudgetedBuffer<double>(budget, valueCount, "unit raw boundary values");
    std::size_t nextBlock = 0;
    std::size_t nextValue = 0;
    for (std::int32_t column = 0; column < pointCount(); ++column)
    {
        if (!isRawPoint(column))
            continue;
        for (std::int64_t block = m_pattern->columnBegin(column); block < m_pattern->columnBegin(column + 1); ++block)
        {
            const std::int32_t row = m_pattern->blockRow(block);
            if (!isRawPoint(row))
                continue;
            const int rows = m_pattern->nodeDofCount(row);
            const int columns = m_pattern->nodeDofCount(column);
            m_rawBlocks[nextBlock++] =
                RawBoundaryBlock{m_modelNodes[static_cast<std::size_t>(row)],
                                 m_modelNodes[static_cast<std::size_t>(column)], static_cast<std::int64_t>(nextValue)};
            const std::span<const double> values = m_stiffness->blockValues(block);
            for (int j = 0; j < columns; ++j)
                for (int i = 0; i < rows; ++i)
                {
                    const bool isKept =
                        isFolded[static_cast<std::size_t>(m_firstDof[static_cast<std::size_t>(row)] + i)] == 0 &&
                        isFolded[static_cast<std::size_t>(m_firstDof[static_cast<std::size_t>(column)] + j)] == 0;
                    m_rawValues[nextValue++] = isKept ? values[static_cast<std::size_t>(i + rows * j)] : 0.0;
                }
        }
    }
}

void UnitSystem::constrainSingularities(const UnitModel& model, base::MemoryBudget& budget)
{
    const assembly::SingularityOptions& options = model.singularity;
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    // The leg DOFs of the absorbed spiders (before their change of basis, the first m leg slots are Absent): free.
    base::BudgetedBuffer<std::uint8_t> isLeg(budget, dofCount, "unit spider legs");
    for (const LocalSpider& spider : m_spiders)
        for (const std::int64_t leg : spider.legs)
            isLeg[static_cast<std::size_t>(leg)] = 1;
    const auto isChecked = [&](std::int64_t dof)
    {
        const UnitDofRole role = m_roles[static_cast<std::size_t>(dof)];
        return role == UnitDofRole::Interior || role == UnitDofRole::Absent;
    };
    base::BudgetedBuffer<double> ratios(budget, dofCount, "unit singularity ratios");
    base::BudgetedBuffer<std::uint8_t> isAligned(budget, dofCount, "unit singularity alignment");
    std::fill(ratios.data(), ratios.data() + dofCount, assembly::NOT_SINGULAR_RATIO);
    double largest = 0.0;
    constexpr std::size_t BLOCK_SIZE = 3;
    for (std::int32_t point = 0; point < pointCount(); ++point)
    {
        const auto p = static_cast<std::size_t>(point);
        if (m_modelNodes[p] == GENERATED_POINT)
            continue;
        const int count = m_pointDofCounts[p];
        const std::int64_t first = m_firstDof[p];
        const std::span<const double> diagonal = m_stiffness->blockValues(m_pattern->columnBegin(point));
        for (int c = 0; c < count; ++c)
        {
            const UnitDofRole role = m_roles[static_cast<std::size_t>(first + c)];
            if (role == UnitDofRole::Interior || role == UnitDofRole::Absent || role == UnitDofRole::Boundary)
                largest = std::max(largest, std::abs(diagonal[static_cast<std::size_t>(c + count * c)]));
        }
        if (count != GRID_DOF_COUNT || m_isBoundaryPoint[p] == BOUNDARY)
            continue;
        for (std::size_t half = 0; half < 2; ++half)
        {
            std::array<double, BLOCK_SIZE * BLOCK_SIZE> block{};
            std::array<bool, BLOCK_SIZE> isFree{};
            for (std::size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                isFree[i] = isChecked(first + static_cast<std::int64_t>(BLOCK_SIZE * half + i));
                for (std::size_t j = 0; j < BLOCK_SIZE; ++j)
                    block[i * BLOCK_SIZE + j] =
                        diagonal[(BLOCK_SIZE * half + i) + GRID_DOF_COUNT * (BLOCK_SIZE * half + j)];
            }
            const assembly::SingularBlock analysis =
                assembly::SingularBlock::analyse(block, isFree, options.pivotZeroThreshold);
            for (std::size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                const auto dof = static_cast<std::size_t>(first) + BLOCK_SIZE * half + i;
                ratios[dof] = analysis.ratios[i];
                isAligned[dof] = analysis.isAligned ? 1 : 0;
            }
        }
    }
    m_largestFreeDiagonal = largest;
    // The scalar points of the interior: singular when their diagonal is at most EPZERO times the largest.
    for (std::int32_t point = 0; point < pointCount(); ++point)
    {
        const auto p = static_cast<std::size_t>(point);
        if (m_modelNodes[p] == GENERATED_POINT || m_pointDofCounts[p] != 1 || m_isBoundaryPoint[p] == BOUNDARY ||
            !isChecked(m_firstDof[p]))
            continue;
        const double diagonal = m_stiffness->blockValues(m_pattern->columnBegin(point))[0];
        if (diagonal <= options.pivotZeroThreshold * largest)
            ratios[static_cast<std::size_t>(m_firstDof[p])] = largest > 0.0 ? std::max(diagonal / largest, 0.0) : 0.0;
    }

    std::size_t count = 0;
    for (std::size_t dof = 0; dof < dofCount; ++dof)
        count += ratios[dof] != assembly::NOT_SINGULAR_RATIO ? 1 : 0;
    m_singularities = base::BudgetedBuffer<assembly::SingularityRow>(budget, count, "unit singularity table");
    std::size_t next = 0;
    for (std::size_t dof = 0; dof < dofCount; ++dof)
    {
        if (ratios[dof] == assembly::NOT_SINGULAR_RATIO)
            continue;
        const std::int32_t point = pointOfLocalDof(static_cast<std::int64_t>(dof));
        const bool isScalar = m_pointDofCounts[static_cast<std::size_t>(point)] == 1;
        if (isLeg[dof] != 0)
            throw std::invalid_argument(base::concatenate(
                {"UnitSystem (", unitText(m_unitId), "): a leg of an absorbed spider (point ",
                 std::to_string(pointIdOf(point)), ", component ",
                 std::to_string(static_cast<std::int64_t>(dof) - m_firstDof[static_cast<std::size_t>(point)] + 1),
                 ") has no stiffness in the unit; the spider must stay on its legs (plan with absorb_spiders=False)"}));
        assembly::SingularityRow row;
        row.dof = *modelDofOf(static_cast<std::int64_t>(dof));
        row.pointId = pointIdOf(point);
        row.isScalarPoint = isScalar;
        row.component =
            isScalar
                ? 0
                : static_cast<int>(static_cast<std::int64_t>(dof) - m_firstDof[static_cast<std::size_t>(point)] + 1);
        row.stiffnessRatio = ratios[dof];
        row.oldExclusive = assembly::ExclusiveSet::B;
        row.isConstrained = options.isAutoSpcOn;
        row.isAligned = isScalar || isAligned[dof] != 0;
        m_singularities[next++] = row;
        if (options.isAutoSpcOn)
            m_roles[dof] = UnitDofRole::Constrained;
    }
    std::sort(m_singularities.data(), m_singularities.data() + count,
              [](const assembly::SingularityRow& a, const assembly::SingularityRow& b)
              { return a.pointId != b.pointId ? a.pointId < b.pointId : a.component < b.component; });
}

void UnitSystem::transformSpiders(const linalg::SparsePattern& pattern, std::span<double> values) const
{
    const SymmetricBlockAccess access(pattern, m_firstDof.span(), values);
    for (const LocalSpider& spider : m_spiders)
    {
        const std::size_t n = spider.legs.size();
        const std::size_t m = spider.references.size();
        // The DOFs coupled to the legs, the legs and the references left out: the columns of the leg rows.
        std::vector<std::int64_t> legsAndReferences(spider.legs); // bounded by the spider
        legsAndReferences.insert(legsAndReferences.end(), spider.references.begin(), spider.references.end());
        std::sort(legsAndReferences.begin(), legsAndReferences.end());
        std::vector<std::int64_t> columns; // bounded by the points coupled to one spider
        std::vector<std::int32_t> legPoints;
        for (const std::int64_t leg : spider.legs)
            legPoints.push_back(pointOfLocalDof(leg));
        sortUnique(legPoints);
        for (const std::int32_t point : legPoints)
            forEachCoupledDof(point,
                              [&](std::int64_t dof)
                              {
                                  if (!std::binary_search(legsAndReferences.begin(), legsAndReferences.end(), dof))
                                      columns.push_back(dof);
                              });
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());

        // K[L, L] and K[L, C], then Tᵀ·K[L, L]·T and Tᵀ·K[L, C] (SpiderChangeOfBasis).
        std::vector<double> legBlock(n * n);
        std::vector<double> coupling(n * columns.size());
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i)
                legBlock[i + j * n] = access.read(spider.legs[i], spider.legs[j]);
        for (std::size_t j = 0; j < columns.size(); ++j)
            for (std::size_t i = 0; i < n; ++i)
                coupling[i + j * n] = access.read(spider.legs[i], columns[j]);
        spider.changeOfBasis.transformSymmetric(linalg::DenseMatrixView{legBlock.data(), n, n, n});
        if (!columns.empty())
            spider.changeOfBasis.transformRows(linalg::DenseMatrixView{coupling.data(), n, columns.size(), n});

        // The old rows of the legs and the references cleared (a reference's are stale once Gm folded it onto the
        // legs), then the new ones written: slot k < m is reference DOF k, slot k ≥ m the deformation DOF k − m in leg
        // slot k; leg slots 0 to m − 1 stay empty.
        for (const std::int64_t dof : legsAndReferences)
            forEachCoupledDof(pointOfLocalDof(dof), [&](std::int64_t other) { access.setSymmetric(dof, other, 0.0); });
        const auto slot = [&](std::size_t k) { return k < m ? spider.references[k] : spider.legs[k]; };
        for (std::size_t l = 0; l < n; ++l)
            for (std::size_t k = l; k < n; ++k)
                access.setSymmetric(slot(k), slot(l), legBlock[k + l * n]);
        for (std::size_t j = 0; j < columns.size(); ++j)
            for (std::size_t k = 0; k < n; ++k)
                access.setSymmetric(slot(k), columns[j], coupling[k + j * n]);
    }
}

void UnitSystem::order(base::MemoryBudget& budget, base::ThreadPool* pool)
{
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    base::BudgetedBuffer<std::uint8_t> eliminated(budget, dofCount, "unit eliminated DOFs");
    for (std::size_t dof = 0; dof < dofCount; ++dof)
    {
        const UnitDofRole role = m_roles[dof];
        eliminated[dof] = role == UnitDofRole::Interior || role == UnitDofRole::Boundary ? 0 : ELIMINATED;
    }
    // The points with an equation, their equation counts (the weights of the ordering) and positions; the boundary
    // points among them, last (a boundary point brings all its DOFs left to the unit).
    const std::size_t points = m_pointDofCounts.size();
    base::BudgetedBuffer<std::int32_t> keptPoints(budget, points, "unit points with equations");
    base::BudgetedBuffer<std::int32_t> weights(budget, points, "unit point weights");
    base::BudgetedBuffer<base::Vec3> positions(budget, points, "unit point positions");
    base::BudgetedBuffer<std::int32_t> trailing(budget, points, "unit boundary points");
    std::size_t kept = 0;
    std::size_t trailingCount = 0;
    for (std::size_t p = 0; p < points; ++p)
    {
        std::int32_t free = 0;
        for (auto dof = m_firstDof[p]; dof < m_firstDof[p + 1]; ++dof)
        {
            const UnitDofRole role = m_roles[static_cast<std::size_t>(dof)];
            if (role != UnitDofRole::Interior && role != UnitDofRole::Boundary)
                continue;
            if ((role == UnitDofRole::Boundary) != (m_isBoundaryPoint[p] == BOUNDARY))
                throw std::logic_error(base::concatenate(
                    {"UnitSystem (", unitText(m_unitId), "): point ",
                     std::to_string(pointIdOf(static_cast<std::int32_t>(p))),
                     " has interior and boundary equations; a point is interior or on the boundary as a whole (an "
                     "internal error: report it)"}));
            ++free;
        }
        if (free == 0)
            continue;
        keptPoints[kept] = static_cast<std::int32_t>(p);
        weights[kept] = free;
        positions[kept] = m_positions[p];
        if (m_isBoundaryPoint[p] == BOUNDARY)
            trailing[trailingCount++] = static_cast<std::int32_t>(kept);
        ++kept;
    }
    if (kept == 0)
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): every DOF of the unit is held or dependent: there is nothing to condense; keep the unit full"}));
    const std::span<const std::int32_t> nodes = keptPoints.span().first(kept);
    const std::span<const std::int32_t> nodeWeights = weights.span().first(kept);
    const linalg::NodeGraph subgraph = m_graph.inducedSubgraph(nodes, budget);
    const linalg::NodePermutation ordering = linalg::ConstrainedOrdering().order(
        subgraph, nodeWeights, positions.span().first(kept), trailing.span().first(trailingCount),
        linalg::OrderingMethod::Automatic, budget, pool);
    m_symbolic =
        linalg::SymbolicFactor::analyse(subgraph, nodeWeights, ordering, static_cast<std::int32_t>(trailingCount),
                                        linalg::AmalgamationThresholds{}, budget);
    base::BudgetedBuffer<std::int32_t> nodeOrder(budget, kept, "unit point order");
    const std::span<const std::int32_t> symbolicOrder = m_symbolic.nodeOrder().nodes();
    for (std::size_t k = 0; k < kept; ++k)
        nodeOrder[k] = nodes[static_cast<std::size_t>(symbolicOrder[k])];
    m_selection = linalg::DofSelection(m_pointDofCounts.span(), eliminated.span(), nodeOrder.span(), budget);
    m_matrix = linalg::SymmetricCsc::fromBlocks(*m_stiffness, m_selection, budget, pool);
    if (m_massBlocks)
    {
        m_mass = linalg::SymmetricCsc::fromBlocks(*m_massBlocks, m_selection, budget, pool);
        m_massBlocks.reset();
        m_hasMass = true;
    }

    const auto equations = static_cast<std::int32_t>(m_matrix.dimension());
    const std::int32_t interior = equations - m_symbolic.trailingDofCount();
    m_interiorEquationCount = interior;
    m_boundaryLabels = base::BudgetedBuffer<UnitDofLabel>(
        budget, static_cast<std::size_t>(m_symbolic.trailingDofCount()), "unit boundary labels");
    for (std::int32_t equation = interior; equation < equations; ++equation)
        m_boundaryLabels[static_cast<std::size_t>(equation - interior)] = labelOf(m_selection.dofOfEquation(equation));
    keepEnforcedCouplings(budget);
    // K is in m_matrix now (and K_fs kept): the block form is no longer needed.
    m_stiffness.reset();
    m_pattern.reset();
}

void UnitSystem::keepEnforcedCouplings(base::MemoryBudget& budget)
{
    const SymmetricBlockAccess access(*m_pattern, m_firstDof.span(), m_stiffness->values());
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    std::size_t constrained = 0;
    for (std::size_t dof = 0; dof < dofCount; ++dof)
        constrained += m_roles[dof] == UnitDofRole::Constrained ? 1 : 0;
    m_constrainedDofs = base::BudgetedBuffer<std::int64_t>(budget, constrained, "unit held DOFs");
    m_constrainedOffsets = base::BudgetedBuffer<std::int64_t>(budget, constrained + 1, "unit held DOF couplings");
    std::size_t next = 0;
    std::size_t termCount = 0;
    std::vector<std::int64_t> coupled; // bounded by a point's degree
    for (std::size_t dof = 0; dof < dofCount; ++dof)
    {
        if (m_roles[dof] != UnitDofRole::Constrained)
            continue;
        m_constrainedDofs[next++] = static_cast<std::int64_t>(dof);
        forEachCoupledDof(pointOfLocalDof(static_cast<std::int64_t>(dof)),
                          [&](std::int64_t other)
                          {
                              if (m_selection.equationOfDof(other) != linalg::DofSelection::ELIMINATED &&
                                  access.read(other, static_cast<std::int64_t>(dof)) != 0.0)
                                  ++termCount;
                          });
    }
    m_constrainedEquations = base::BudgetedBuffer<std::int32_t>(budget, termCount, "unit held DOF couplings");
    m_constrainedValues = base::BudgetedBuffer<double>(budget, termCount, "unit held DOF couplings");
    std::size_t term = 0;
    for (std::size_t k = 0; k < constrained; ++k)
    {
        const std::int64_t dof = m_constrainedDofs[k];
        m_constrainedOffsets[k] = static_cast<std::int64_t>(term);
        forEachCoupledDof(pointOfLocalDof(dof),
                          [&](std::int64_t other)
                          {
                              const std::int32_t equation = m_selection.equationOfDof(other);
                              const double value = access.read(other, dof);
                              if (equation == linalg::DofSelection::ELIMINATED || value == 0.0)
                                  return;
                              m_constrainedEquations[term] = equation;
                              m_constrainedValues[term] = value;
                              ++term;
                          });
    }
    m_constrainedOffsets[constrained] = static_cast<std::int64_t>(term);

    // K_ss among the held DOFs (their reactions under enforced displacements, heldReactions).
    base::BudgetedBuffer<std::int32_t> heldIndexOf(budget, dofCount, "unit held DOF indices");
    std::fill(heldIndexOf.data(), heldIndexOf.data() + dofCount, -1);
    for (std::size_t k = 0; k < constrained; ++k)
        heldIndexOf[static_cast<std::size_t>(m_constrainedDofs[k])] = static_cast<std::int32_t>(k);
    std::size_t heldTermCount = 0;
    for (std::size_t k = 0; k < constrained; ++k)
        forEachCoupledDof(pointOfLocalDof(m_constrainedDofs[k]),
                          [&](std::int64_t other)
                          {
                              if (heldIndexOf[static_cast<std::size_t>(other)] >= 0 &&
                                  access.read(other, m_constrainedDofs[k]) != 0.0)
                                  ++heldTermCount;
                          });
    m_heldOffsets = base::BudgetedBuffer<std::int64_t>(budget, constrained + 1, "unit held DOF couplings");
    m_heldIndices = base::BudgetedBuffer<std::int32_t>(budget, heldTermCount, "unit held DOF couplings");
    m_heldValues = base::BudgetedBuffer<double>(budget, heldTermCount, "unit held DOF couplings");
    std::size_t heldTerm = 0;
    for (std::size_t k = 0; k < constrained; ++k)
    {
        m_heldOffsets[k] = static_cast<std::int64_t>(heldTerm);
        forEachCoupledDof(pointOfLocalDof(m_constrainedDofs[k]),
                          [&](std::int64_t other)
                          {
                              const std::int32_t index = heldIndexOf[static_cast<std::size_t>(other)];
                              const double value = access.read(other, m_constrainedDofs[k]);
                              if (index < 0 || value == 0.0)
                                  return;
                              m_heldIndices[heldTerm] = index;
                              m_heldValues[heldTerm] = value;
                              ++heldTerm;
                          });
    }
    m_heldOffsets[constrained] = static_cast<std::int64_t>(heldTerm);
    m_isHeldCoupled = base::BudgetedBuffer<std::uint8_t>(budget, constrained, "unit held DOF couplings");
    for (std::size_t k = 0; k < constrained; ++k)
        m_isHeldCoupled[k] =
            m_constrainedOffsets[k + 1] > m_constrainedOffsets[k] || m_heldOffsets[k + 1] > m_heldOffsets[k] ? 1 : 0;
}

void UnitSystem::heldReactions(linalg::ConstDenseMatrixView equationValues, linalg::ConstDenseMatrixView enforcedLocal,
                               linalg::ConstDenseMatrixView loadsLocal, linalg::DenseMatrixView reactions,
                               base::MemoryBudget& budget) const
{
    requireEquations("heldReactions");
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    const auto equations = static_cast<std::size_t>(equationCount());
    const std::size_t held = m_constrainedDofs.size();
    const std::size_t columns = equationValues.columns;
    const bool hasEnforced = enforcedLocal.columns > 0;
    if (equationValues.rows != equations || loadsLocal.rows != dofCount || loadsLocal.columns != columns ||
        reactions.rows != held || reactions.columns != columns ||
        (hasEnforced && (enforcedLocal.rows != dofCount || enforcedLocal.columns != columns)))
        throw std::invalid_argument(base::concatenate({"UnitSystem::heldReactions (",
                                                       unitText(m_unitId),
                                                       "): ",
                                                       std::to_string(equationValues.rows),
                                                       " x ",
                                                       std::to_string(columns),
                                                       " equation values, ",
                                                       std::to_string(loadsLocal.rows),
                                                       " x ",
                                                       std::to_string(loadsLocal.columns),
                                                       " loads, ",
                                                       std::to_string(enforcedLocal.rows),
                                                       " x ",
                                                       std::to_string(enforcedLocal.columns),
                                                       " enforced values into ",
                                                       std::to_string(reactions.rows),
                                                       " x ",
                                                       std::to_string(reactions.columns),
                                                       " reactions; expected ",
                                                       std::to_string(equations),
                                                       " equation rows, ",
                                                       std::to_string(dofCount),
                                                       " local rows, ",
                                                       std::to_string(held),
                                                       " held DOFs, as many columns"}));
    base::BudgetedBuffer<double> work(budget, dofCount, "unit reaction loads");
    for (std::size_t j = 0; j < columns; ++j)
    {
        for (std::size_t dof = 0; dof < dofCount; ++dof)
            work[dof] = loadsLocal(dof, j);
        // P_n += Gmᵀ·P_m, as equationLoads.
        m_relations.reduce(work.span());
        for (std::size_t k = 0; k < held; ++k)
        {
            double value = -work[static_cast<std::size_t>(m_constrainedDofs[k])];
            for (auto t = m_constrainedOffsets[k]; t < m_constrainedOffsets[k + 1]; ++t)
                value +=
                    m_constrainedValues[static_cast<std::size_t>(t)] *
                    equationValues(static_cast<std::size_t>(m_constrainedEquations[static_cast<std::size_t>(t)]), j);
            if (hasEnforced)
                for (auto t = m_heldOffsets[k]; t < m_heldOffsets[k + 1]; ++t)
                    value += m_heldValues[static_cast<std::size_t>(t)] *
                             enforcedLocal(static_cast<std::size_t>(m_constrainedDofs[static_cast<std::size_t>(
                                               m_heldIndices[static_cast<std::size_t>(t)])]),
                                           j);
            reactions(k, j) = value;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Loads and displacements

void UnitSystem::gatherLocal(std::span<const double> valuesG, std::span<double> valuesLocal) const
{
    if (valuesG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()) ||
        valuesLocal.size() != static_cast<std::size_t>(localDofCount()))
        throw std::invalid_argument(
            base::concatenate({"UnitSystem::gatherLocal: ", std::to_string(valuesG.size()), " g-set values into ",
                               std::to_string(valuesLocal.size()), " local ones; expected ",
                               std::to_string(m_dofMap->gSetSize()), " and ", std::to_string(localDofCount())}));
    std::fill(valuesLocal.begin(), valuesLocal.end(), 0.0);
    for (std::size_t p = 0; p < m_sortedModelNodes.size(); ++p)
        for (int c = 0; c < m_pointDofCounts[p]; ++c)
            valuesLocal[static_cast<std::size_t>(m_firstDof[p] + c)] =
                valuesG[static_cast<std::size_t>(modelDofOfNode(m_sortedModelNodes[p], c))];
}

void UnitSystem::equationLoads(linalg::ConstDenseMatrixView loadsLocal, linalg::ConstDenseMatrixView enforcedLocal,
                               linalg::DenseMatrixView equationValues, base::MemoryBudget& budget) const
{
    requireEquations("equationLoads");
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    const auto equations = static_cast<std::size_t>(equationCount());
    const bool hasEnforced = enforcedLocal.columns > 0;
    if (loadsLocal.rows != dofCount || equationValues.rows != equations ||
        equationValues.columns != loadsLocal.columns ||
        (hasEnforced && (enforcedLocal.rows != dofCount || enforcedLocal.columns != loadsLocal.columns)))
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem::equationLoads: ", std::to_string(loadsLocal.rows), " x ", std::to_string(loadsLocal.columns),
             " local loads, ", std::to_string(enforcedLocal.rows), " x ", std::to_string(enforcedLocal.columns),
             " enforced displacements into ", std::to_string(equationValues.rows), " x ",
             std::to_string(equationValues.columns), " equation loads; expected ", std::to_string(dofCount),
             " local rows (or no enforced column), ", std::to_string(equations), " equation rows, as many columns"}));
    base::BudgetedBuffer<double> work(budget, dofCount, "unit load work");
    std::vector<double> legs; // bounded by a spider
    for (std::size_t j = 0; j < loadsLocal.columns; ++j)
    {
        for (std::size_t dof = 0; dof < dofCount; ++dof)
            work[dof] = loadsLocal(dof, j);
        // P_n += Gmᵀ·P_m, P_m = 0 (RigidElimination::reduceToN), the rows resolved onto the n set.
        m_relations.reduce(work.span());
        // Tᵀ·P_legs: the reference part to the reference DOFs, the deformation part to the last leg slots.
        for (const LocalSpider& spider : m_spiders)
        {
            const std::size_t n = spider.legs.size();
            const std::size_t m = spider.references.size();
            legs.resize(n);
            for (std::size_t k = 0; k < n; ++k)
                legs[k] = work[static_cast<std::size_t>(spider.legs[k])];
            spider.changeOfBasis.transformRows(linalg::DenseMatrixView{legs.data(), n, 1, n});
            for (std::size_t k = 0; k < n; ++k)
            {
                if (k < m)
                {
                    work[static_cast<std::size_t>(spider.references[k])] += legs[k];
                    work[static_cast<std::size_t>(spider.legs[k])] = 0.0;
                }
                else
                    work[static_cast<std::size_t>(spider.legs[k])] = legs[k];
            }
        }
        for (std::size_t equation = 0; equation < equations; ++equation)
            equationValues(equation, j) =
                work[static_cast<std::size_t>(m_selection.dofOfEquation(static_cast<std::int32_t>(equation)))];
        // −K_fs·u_s.
        if (!hasEnforced)
            continue;
        for (std::size_t k = 0; k < m_constrainedDofs.size(); ++k)
        {
            const double value = enforcedLocal(static_cast<std::size_t>(m_constrainedDofs[k]), j);
            if (value == 0.0)
                continue;
            for (auto t = m_constrainedOffsets[k]; t < m_constrainedOffsets[k + 1]; ++t)
                equationValues(static_cast<std::size_t>(m_constrainedEquations[static_cast<std::size_t>(t)]), j) -=
                    m_constrainedValues[static_cast<std::size_t>(t)] * value;
        }
    }
}

void UnitSystem::expandSolution(linalg::ConstDenseMatrixView equationValues, linalg::ConstDenseMatrixView enforcedLocal,
                                linalg::DenseMatrixView displacementsLocal) const
{
    requireEquations("expandSolution");
    const auto dofCount = static_cast<std::size_t>(localDofCount());
    const auto equations = static_cast<std::size_t>(equationCount());
    const bool hasEnforced = enforcedLocal.columns > 0;
    if (equationValues.rows != equations || displacementsLocal.rows != dofCount ||
        displacementsLocal.columns != equationValues.columns ||
        (hasEnforced && (enforcedLocal.rows != dofCount || enforcedLocal.columns != equationValues.columns)))
        throw std::invalid_argument(base::concatenate(
            {"UnitSystem::expandSolution: ", std::to_string(equationValues.rows), " x ",
             std::to_string(equationValues.columns), " equation values into ", std::to_string(displacementsLocal.rows),
             " x ", std::to_string(displacementsLocal.columns), " local displacements; expected ",
             std::to_string(equations), " and ", std::to_string(dofCount), " rows, as many columns"}));
    std::vector<double> reference; // bounded by a spider
    std::vector<double> deformation;
    std::vector<double> legs;
    for (std::size_t j = 0; j < equationValues.columns; ++j)
    {
        for (std::size_t dof = 0; dof < dofCount; ++dof)
            displacementsLocal(dof, j) = 0.0;
        for (std::size_t equation = 0; equation < equations; ++equation)
            displacementsLocal(static_cast<std::size_t>(m_selection.dofOfEquation(static_cast<std::int32_t>(equation))),
                               j) = equationValues(equation, j);
        if (hasEnforced)
            for (const std::int64_t dof : m_constrainedDofs.span())
                displacementsLocal(static_cast<std::size_t>(dof), j) = enforcedLocal(static_cast<std::size_t>(dof), j);
        // u_legs = W⁺·u_ref + N·q (the deformation DOFs in the last leg slots).
        for (const LocalSpider& spider : m_spiders)
        {
            const std::size_t n = spider.legs.size();
            const std::size_t m = spider.references.size();
            reference.resize(m);
            deformation.resize(n - m);
            legs.resize(n);
            for (std::size_t k = 0; k < m; ++k)
                reference[k] = displacementsLocal(static_cast<std::size_t>(spider.references[k]), j);
            for (std::size_t k = m; k < n; ++k)
                deformation[k - m] = displacementsLocal(static_cast<std::size_t>(spider.legs[k]), j);
            spider.changeOfBasis.expand(reference, deformation, legs);
            for (std::size_t k = 0; k < n; ++k)
                displacementsLocal(static_cast<std::size_t>(spider.legs[k]), j) = legs[k];
        }
        // u_m = Gm·u_n; a spider's reference DOFs keep their boundary values (W·u_legs = u_ref to round-off).
        m_relations.expand(std::span<double>(&displacementsLocal(0, j), dofCount), m_isSpiderReferenceRow.span());
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Craig-Bampton

const linalg::SymmetricCsc& UnitSystem::mass() const
{
    requireEquations("its mass");
    if (!m_hasMass)
        throw std::logic_error(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): its mass was not kept on its equations; build it with UnitModel::keepsMass (Craig-Bampton)"}));
    return m_mass;
}

linalg::SymbolicFactor UnitSystem::interiorSymbolic(base::MemoryBudget& budget) const
{
    requireEquations("interiorSymbolic");
    // The points with an equation in increasing order and their equation counts: the nodes of the subgraph order()
    // analysed (the same loop).
    const std::size_t points = m_pointDofCounts.size();
    base::BudgetedBuffer<std::int32_t> keptPoints(budget, points, "unit interior points");
    base::BudgetedBuffer<std::int32_t> weights(budget, points, "unit interior weights");
    std::size_t kept = 0;
    std::size_t interiorCount = 0;
    for (std::size_t p = 0; p < points; ++p)
    {
        std::int32_t free = 0;
        for (auto dof = m_firstDof[p]; dof < m_firstDof[p + 1]; ++dof)
        {
            const UnitDofRole role = m_roles[static_cast<std::size_t>(dof)];
            free += role == UnitDofRole::Interior || role == UnitDofRole::Boundary ? 1 : 0;
        }
        if (free == 0)
            continue;
        keptPoints[kept] = static_cast<std::int32_t>(p);
        weights[kept] = free;
        interiorCount += m_isBoundaryPoint[p] == BOUNDARY ? 0 : 1;
        ++kept;
    }
    // The interior points in the unit's elimination order: its leading positions (the boundary is the trailing set).
    const std::span<const std::int32_t> order = m_symbolic.nodeOrder().nodes();
    base::BudgetedBuffer<std::int32_t> interiorPoints(budget, interiorCount, "unit interior points");
    base::BudgetedBuffer<std::int32_t> interiorWeights(budget, interiorCount, "unit interior weights");
    base::BudgetedBuffer<std::int32_t> identity(budget, interiorCount, "unit interior order");
    for (std::size_t k = 0; k < interiorCount; ++k)
    {
        const auto node = static_cast<std::size_t>(order[k]);
        interiorPoints[k] = keptPoints[node];
        interiorWeights[k] = weights[node];
        identity[k] = static_cast<std::int32_t>(k);
    }
    const linalg::NodeGraph graph = m_graph.inducedSubgraph(interiorPoints.span(), budget);
    const linalg::NodePermutation ordering(std::move(identity), budget);
    linalg::SymbolicFactor symbolic = linalg::SymbolicFactor::analyse(graph, interiorWeights.span(), ordering, 0,
                                                                      linalg::AmalgamationThresholds{}, budget);
    // The interior's elimination tree is the unit's restricted to it (the fill between two interior equations comes
    // from interior equations only), so the postorder of the analysis is the given order: the equations of the pencil
    // are those of matrix().
    const std::span<const std::int32_t> analysed = symbolic.nodeOrder().nodes();
    bool isSameOrder = symbolic.dofCount() == interiorDofCount();
    for (std::size_t k = 0; k < analysed.size() && isSameOrder; ++k)
        isSameOrder = analysed[k] == static_cast<std::int32_t>(k);
    if (!isSameOrder)
        throw base::InternalError(base::concatenate(
            {"UnitSystem (", unitText(m_unitId),
             "): the symbolic analysis of the interior reordered the unit's interior equations; the fixed-interface "
             "modes need the unit's order (an internal error: report it with the model)"}));
    return symbolic;
}

linalg::SymmetricCsc UnitSystem::differentialStiffness(const UnitModel& model, const UnitDescription& description,
                                                       const elements::NodalDisplacements& preload,
                                                       base::MemoryBudget& budget, base::ThreadPool* pool) const
{
    requireEquations("differentialStiffness");
    if (model.dofMap != m_dofMap || description.unitId != m_unitId)
        throw std::invalid_argument(
            base::concatenate({"UnitSystem::differentialStiffness (", unitText(m_unitId),
                               "): the model and the description must be those the unit was built from"}));
    const UnitElements elements(model, description.elementIds, budget);
    const assembly::DofMap& dofMap = *model.dofMap;
    const linalg::SparsePattern pattern(m_graph, m_pointDofCounts.span(), budget);
    linalg::BlockSparseMatrix blocks(pattern, budget, "unit differential stiffness");
    linalg::SparseAssembler(model.assembly)
        .assemble(
            elements.size(),
            [&](std::size_t begin, std::size_t end, linalg::ContributionBlock& contributions)
            {
                std::vector<linalg::NodeDof> nodeDofs; // bounded by an element
                elements.compute(
                    begin, end,
                    [&preload](std::size_t, const assembly::AssemblyBatch& batch, elements::ElementRange range,
                               elements::ElementMatrixBlock& block)
                    {
                        if (batch.family->facets().has(elements::Facet::Stiffness))
                            batch.family->computeDifferentialStiffness(batch.view, range, preload, block);
                    },
                    [&](const assembly::AssemblyBatch&, const elements::ElementMatrixBlock& block, std::size_t)
                    {
                        for (std::size_t e = 0; e < block.size(); ++e)
                        {
                            nodeDofs.clear();
                            for (const elements::ElementDof& dof : block.dofs(e))
                            {
                                const linalg::NodeDof nodeDof = dofMap.nodeDofOf(dofMap.dofOf(dof));
                                nodeDofs.push_back({*localPointOf(nodeDof.node), nodeDof.component});
                            }
                            contributions.append(nodeDofs, block.matrix(e));
                        }
                    });
            },
            blocks, pool);
    // K_d is reduced to the n set by the same Gm, and the unit absorbs its spiders by the same change of basis.
    m_relations.eliminate(pattern, m_firstDof.span(), m_graph, blocks.values(), budget);
    transformSpiders(pattern, blocks.values());
    return linalg::SymmetricCsc::fromBlocks(blocks, m_selection, budget, pool);
}

} // namespace oos::reduction
