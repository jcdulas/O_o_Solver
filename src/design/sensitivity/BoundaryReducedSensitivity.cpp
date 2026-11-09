#include "oos/design/sensitivity/BoundaryReducedSensitivity.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

// The columns of `matrix` (rows × columns) that hold a non-zero value.
std::vector<std::size_t> nonZeroColumns(const linalg::DenseMatrix& matrix)
{
    std::vector<std::size_t> columns; // bounded by the columns of a block
    for (std::size_t j = 0; j < matrix.columns(); ++j)
        for (std::size_t i = 0; i < matrix.rows(); ++i)
            if (matrix(i, j) != 0.0)
            {
                columns.push_back(j);
                break;
            }
    return columns;
}

// The bytes a basis holds (its padded columns).
std::uint64_t basisBytes(const linalg::DenseMatrix& basis) noexcept
{
    return static_cast<std::uint64_t>(basis.leadingDimension()) * basis.columns() * sizeof(double);
}

} // namespace

BoundaryReducedSensitivity::~BoundaryReducedSensitivity() = default;

BoundaryReducedSensitivity::BoundaryReducedSensitivity(std::shared_ptr<const solutions::CondensedFactorisedSet> set,
                                                       linalg::KernelKind kernel, base::MemoryBudget& budget,
                                                       base::ThreadPool* pool,
                                                       std::optional<std::uint64_t> basisRetentionBytes)
    : m_set(std::move(set)), m_budget(&budget), m_kernels(kernel, linalg::DenseKernelOptions{pool})
{
    if (!m_set)
        throw std::invalid_argument("BoundaryReducedSensitivity: expected a kept condensed constraint set, found none");
    m_basisRetentionBytes =
        basisRetentionBytes
            ? *basisRetentionBytes
            : static_cast<std::uint64_t>(BOUNDARY_BASIS_RETENTION_SHARE * static_cast<double>(budget.available()));
    const auto gSize = static_cast<std::size_t>(m_set->dofMap().gSetSize());
    m_isKnown = base::BudgetedBuffer<std::uint8_t>(budget, gSize, "condensed sensitivity known DOFs");
    std::fill(m_isKnown.data(), m_isKnown.data() + gSize, std::uint8_t{1});
    const solutions::KeptCondensedUnits& units = m_set->units();
    m_bases.resize(units.size());
    m_basisLastBlock.assign(units.size(), 0);
    for (const solutions::KeptCondensedUnit& unit : units)
    {
        UnitSensitivityStatistics statistics;
        statistics.unitId = unit.unitId;
        statistics.boundaryDofCount = unit.system->boundaryDofCount();
        statistics.isDriven = unit.isDriven;
        m_statistics.units.push_back(statistics);
        if (unit.isRecovered)
            continue;
        // The interior displacements of a unit the analysis did not recover are not computed.
        for (std::size_t local = 0; local < unit.modelDofOfLocal.size(); ++local)
            if (unit.isInteriorLocal[local] != 0 && unit.modelDofOfLocal[local] >= 0)
            {
                m_isKnown[static_cast<std::size_t>(unit.modelDofOfLocal[local])] = 0;
                m_isEveryDofKnown = false;
            }
    }
}

void BoundaryReducedSensitivity::requireFactor(std::size_t u)
{
    const solutions::KeptCondensedUnit& unit = m_set->units()[u];
    if (unit.condensation->isInteriorFactorKept())
        return;
    // The same bits as the condensation's own factor (StaticCondensation checks its K* to the bit).
    unit.condensation->recomputeInteriorFactor(m_kernels, *m_budget, nullptr);
    ++m_statistics.units[u].refactorisations;
}

void BoundaryReducedSensitivity::solveInterface(linalg::DenseMatrixView loadsLocal,
                                                linalg::DenseMatrixView displacementsLocal)
{
    const reduction::InterfaceProblem& interface = m_set->interface();
    interface.reduceLoads(loadsLocal);
    linalg::DenseMatrix equations(*m_budget, static_cast<std::size_t>(interface.equationCount()), loadsLocal.columns,
                                  "condensed sensitivity interface right-hand sides");
    interface.equationLoads(loadsLocal, {}, equations.view());
    interface.solve(equations.view(), m_kernels, nullptr);
    interface.expand(equations.view(), {}, displacementsLocal);
}

const linalg::DenseMatrix& BoundaryReducedSensitivity::boundaryBasis(std::size_t u)
{
    m_basisLastBlock[u] = m_block;
    if (m_bases[u])
        return *m_bases[u];
    const reduction::InterfaceProblem& interface = m_set->interface();
    const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    linalg::DenseMatrix loads(*m_budget, localDofs, locals.size(), "condensed sensitivity unit basis loads");
    for (std::size_t i = 0; i < locals.size(); ++i)
        loads(static_cast<std::size_t>(locals[i]), i) = 1.0;
    linalg::DenseMatrix basis(*m_budget, localDofs, locals.size(), "condensed sensitivity unit basis");
    solveInterface(loads.view(), basis.view());
    const auto solves = static_cast<std::int64_t>(locals.size());
    m_statistics.units[u].basisSolves += solves;
    m_statistics.interfaceSolves += solves;
    m_retainedBasisBytes += basisBytes(basis);
    m_bases[u] = std::move(basis);
    return *m_bases[u];
}

void BoundaryReducedSensitivity::releaseBasis(std::size_t u)
{
    m_retainedBasisBytes -= basisBytes(*m_bases[u]);
    m_bases[u].reset();
    ++m_statistics.basisReleases;
}

void BoundaryReducedSensitivity::retainOrRelease(std::size_t u)
{
    m_statistics.retainedBasisPeakBytes = std::max(m_statistics.retainedBasisPeakBytes, m_retainedBasisBytes);
    while (m_retainedBasisBytes > m_basisRetentionBytes)
    {
        // The least recently used other basis (the lower index on a tie: a fixed order).
        std::optional<std::size_t> oldest;
        for (std::size_t v = 0; v < m_bases.size(); ++v)
            if (v != u && m_bases[v] && (!oldest || m_basisLastBlock[v] < m_basisLastBlock[*oldest]))
                oldest = v;
        if (!oldest)
            break;
        releaseBasis(*oldest);
    }
    if (m_retainedBasisBytes > m_basisRetentionBytes)
        releaseBasis(u);
}

void BoundaryReducedSensitivity::solve(linalg::DenseMatrixView columnsG, std::span<const std::uint8_t> isReadDof)
{
    const auto gSize = static_cast<std::size_t>(m_set->dofMap().gSetSize());
    if (columnsG.rows != gSize || (!isReadDof.empty() && isReadDof.size() != gSize))
        throw std::invalid_argument(
            base::concatenate({"BoundaryReducedSensitivity::solve: expected ", std::to_string(gSize),
                               " g-set rows and read flags, found ", std::to_string(columnsG.rows), " rows and ",
                               std::to_string(isReadDof.size()), " flags"}));
    const std::size_t columnCount = columnsG.columns;
    if (columnCount == 0)
        return;
    ++m_block;
    m_statistics.columns += static_cast<std::int64_t>(columnCount);
    const reduction::InterfaceProblem& interface = m_set->interface();
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    const solutions::KeptCondensedUnits& units = m_set->units();

    // 1. The units whose interior rows a column loads: those rows on their equations (their relations and spiders
    // applied: UnitSystem::equationLoads), condensed onto their boundaries, f* = f_a − K_ao·K_oo⁻¹·f_o.
    struct LoadedUnit
    {
        std::vector<std::size_t> columns; // bounded by the columns of the block
        linalg::DenseMatrix equations;    // equationCount() × columns: the unit's loads on its equations
        linalg::DenseMatrix condensed;    // n_a × columns
    };
    std::vector<LoadedUnit> loaded(units.size()); // bounded by the units
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        const solutions::KeptCondensedUnit& unit = units[u];
        const std::size_t localCount = unit.modelDofOfLocal.size();
        const auto isLoadRow = [&](std::size_t local)
        { return unit.isInteriorLocal[local] != 0 && unit.modelDofOfLocal[local] >= 0; };
        LoadedUnit& entry = loaded[u];
        for (std::size_t j = 0; j < columnCount; ++j)
            for (std::size_t local = 0; local < localCount; ++local)
                if (isLoadRow(local) && columnsG(static_cast<std::size_t>(unit.modelDofOfLocal[local]), j) != 0.0)
                {
                    entry.columns.push_back(j);
                    break;
                }
        if (entry.columns.empty())
            continue;
        linalg::DenseMatrix localLoads(*m_budget, localCount, entry.columns.size(), "condensed sensitivity unit loads");
        for (std::size_t k = 0; k < entry.columns.size(); ++k)
            for (std::size_t local = 0; local < localCount; ++local)
                if (isLoadRow(local))
                    localLoads(local, k) =
                        columnsG(static_cast<std::size_t>(unit.modelDofOfLocal[local]), entry.columns[k]);
        entry.equations = linalg::DenseMatrix(*m_budget, static_cast<std::size_t>(unit.system->equationCount()),
                                              entry.columns.size(), "condensed sensitivity unit equations");
        unit.system->equationLoads(localLoads.view(), {}, entry.equations.view(), *m_budget);
        requireFactor(u);
        entry.condensed = unit.condensation->condenseLoads(entry.equations.view(), m_kernels, *m_budget);
        m_statistics.units[u].condensedColumns += static_cast<std::int64_t>(entry.columns.size());
    }

    // 2. The interface: its own rows, then each unit's condensed loads, added to the columns' right-hand sides or, when
    // more columns load the unit (its interior or its boundary) than it has boundary DOFs, through its boundary basis.
    linalg::DenseMatrix interfaceLoads(*m_budget, localDofs, columnCount, "condensed sensitivity interface loads");
    for (std::size_t local = 0; local < localDofs; ++local)
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(local)))
            for (std::size_t j = 0; j < columnCount; ++j)
                interfaceLoads(local, j) = columnsG(static_cast<std::size_t>(*dof), j);
    // The columns of each unit: those loading its interior or its boundary (an element's pseudo-load touches the DOFs
    // of its unit only); a column loading interface rows on no condensed unit's boundary (a full unit's interior, an
    // external unit's contour) is counted on its own in the bound.
    std::vector<std::uint8_t> isUnitDof(localDofs, 0);               // bounded by the interface's DOFs
    std::vector<std::vector<std::uint8_t>> isColumnOf(units.size()); // bounded by the units × the columns of a block
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        isColumnOf[u].assign(columnCount, 0);
        for (const std::size_t j : loaded[u].columns)
            isColumnOf[u][j] = 1;
        for (const std::int64_t local : interface.unitLocalDofs(u))
        {
            isUnitDof[static_cast<std::size_t>(local)] = 1;
            for (std::size_t j = 0; j < columnCount; ++j)
                if (interfaceLoads(static_cast<std::size_t>(local), j) != 0.0)
                    isColumnOf[u][j] = 1;
        }
    }
    for (std::size_t j = 0; j < columnCount; ++j)
        for (std::size_t local = 0; local < localDofs; ++local)
            if (isUnitDof[local] == 0 && interfaceLoads(local, j) != 0.0)
            {
                ++m_statistics.interfaceSolveBound;
                break;
            }
    // A unit through its basis takes, besides its condensed loads, every column's loads on its boundary DOFs: a load on
    // its boundary DOF i is the load of its basis column i (the interface is linear), so that those columns need no
    // solve of their own. Its coefficients: n_a × the columns of the block.
    std::vector<std::optional<linalg::DenseMatrix>> coefficients(units.size()); // bounded by the units
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        const auto unitColumns =
            static_cast<std::size_t>(std::count(isColumnOf[u].begin(), isColumnOf[u].end(), std::uint8_t{1}));
        if (unitColumns == 0)
            continue;
        const LoadedUnit& entry = loaded[u];
        const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
        m_statistics.interfaceSolveBound +=
            static_cast<std::int64_t>(std::min(unitColumns, static_cast<std::size_t>(locals.size())));
        // Its loads span at most n_a vectors: past n_a columns, the n_a solves of its basis serve them all (and those
        // of the later blocks of the set while the basis is kept). The choice depends on the block's columns only, not
        // on whether a basis is kept: the same bits whatever the budget.
        if (!locals.empty() && unitColumns > locals.size())
        {
            linalg::DenseMatrix& factors = coefficients[u].emplace(*m_budget, locals.size(), columnCount,
                                                                   "condensed sensitivity unit basis coefficients");
            for (std::size_t k = 0; k < entry.columns.size(); ++k)
                for (std::size_t i = 0; i < locals.size(); ++i)
                    factors(i, entry.columns[k]) = entry.condensed(i, k);
            for (std::size_t i = 0; i < locals.size(); ++i)
                for (std::size_t j = 0; j < columnCount; ++j)
                {
                    double& load = interfaceLoads(static_cast<std::size_t>(locals[i]), j);
                    factors(i, j) += load;
                    load = 0.0;
                }
            continue;
        }
        for (std::size_t k = 0; k < entry.columns.size(); ++k)
            for (std::size_t i = 0; i < locals.size(); ++i)
                interfaceLoads(static_cast<std::size_t>(locals[i]), entry.columns[k]) += entry.condensed(i, k);
    }
    linalg::DenseMatrix displacements(*m_budget, localDofs, columnCount, "condensed sensitivity interface solution");
    const std::vector<std::size_t> solved = nonZeroColumns(interfaceLoads);
    if (!solved.empty())
    {
        linalg::DenseMatrix loads(*m_budget, localDofs, solved.size(), "condensed sensitivity interface columns");
        for (std::size_t k = 0; k < solved.size(); ++k)
            for (std::size_t local = 0; local < localDofs; ++local)
                loads(local, k) = interfaceLoads(local, solved[k]);
        linalg::DenseMatrix solution(*m_budget, localDofs, solved.size(), "condensed sensitivity interface columns");
        solveInterface(loads.view(), solution.view());
        m_statistics.interfaceSolves += static_cast<std::int64_t>(solved.size());
        for (std::size_t k = 0; k < solved.size(); ++k)
            for (std::size_t local = 0; local < localDofs; ++local)
                displacements(local, solved[k]) = solution(local, k);
    }
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        if (!coefficients[u])
            continue;
        // The interface is linear (its relations, constraints and expansion without enforced values): the response to
        // the unit's loads on its boundary is Y_u·f, summed in the order of the basis columns. One basis made at a
        // time: those not kept are released before the next is made.
        const linalg::DenseMatrix& basis = boundaryBasis(u);
        const linalg::DenseMatrix& factors = *coefficients[u];
        for (std::size_t j = 0; j < columnCount; ++j)
            for (std::size_t i = 0; i < basis.columns(); ++i)
            {
                const double factor = factors(i, j);
                if (factor == 0.0)
                    continue;
                for (std::size_t local = 0; local < localDofs; ++local)
                    displacements(local, j) += basis(local, i) * factor;
            }
        coefficients[u].reset();
        retainOrRelease(u);
    }

    // 3. The solution on the g set: the interface's points, then the interiors of the units read.
    for (std::size_t j = 0; j < columnCount; ++j)
        for (std::size_t g = 0; g < gSize; ++g)
            columnsG(g, j) = 0.0;
    for (std::size_t local = 0; local < localDofs; ++local)
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(local)))
            for (std::size_t j = 0; j < columnCount; ++j)
                columnsG(static_cast<std::size_t>(*dof), j) = displacements(local, j);
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        const solutions::KeptCondensedUnit& unit = units[u];
        const std::size_t localCount = unit.modelDofOfLocal.size();
        const auto isInteriorDof = [&](std::size_t local)
        { return unit.isInteriorLocal[local] != 0 && unit.modelDofOfLocal[local] >= 0; };
        bool isRead = isReadDof.empty();
        for (std::size_t local = 0; local < localCount && !isRead; ++local)
            isRead = isInteriorDof(local) && isReadDof[static_cast<std::size_t>(unit.modelDofOfLocal[local])] != 0;
        if (!isRead)
            continue;
        // The columns that load it or move its boundary; the others leave its interior at rest.
        const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
        const LoadedUnit& entry = loaded[u];
        std::vector<std::size_t> recovered;    // bounded by the columns of the block
        std::vector<std::size_t> loadedColumn; // per recovered column: its column in entry, or SIZE_MAX
        std::size_t next = 0;
        for (std::size_t j = 0; j < columnCount; ++j)
        {
            while (next < entry.columns.size() && entry.columns[next] < j)
                ++next;
            const bool isLoaded = next < entry.columns.size() && entry.columns[next] == j;
            bool isMoved = false;
            for (std::size_t i = 0; i < locals.size() && !isMoved; ++i)
                isMoved = displacements(static_cast<std::size_t>(locals[i]), j) != 0.0;
            if (!isLoaded && !isMoved)
                continue;
            recovered.push_back(j);
            loadedColumn.push_back(isLoaded ? next : static_cast<std::size_t>(-1));
        }
        if (recovered.empty())
            continue;
        const auto equations = static_cast<std::size_t>(unit.system->equationCount());
        linalg::DenseMatrix solution(*m_budget, equations, recovered.size(), "condensed sensitivity unit solution");
        linalg::DenseMatrix boundary(*m_budget, locals.size(), recovered.size(), "condensed sensitivity unit boundary");
        for (std::size_t k = 0; k < recovered.size(); ++k)
        {
            if (loadedColumn[k] != static_cast<std::size_t>(-1))
                for (std::size_t e = 0; e < equations; ++e)
                    solution(e, k) = entry.equations(e, loadedColumn[k]);
            for (std::size_t i = 0; i < locals.size(); ++i)
                boundary(i, k) = displacements(static_cast<std::size_t>(locals[i]), recovered[k]);
        }
        requireFactor(u);
        unit.condensation->recover(solution.view(), boundary.view(), m_kernels);
        linalg::DenseMatrix localValues(*m_budget, localCount, recovered.size(), "condensed sensitivity unit values");
        unit.system->expandSolution(solution.view(), {}, localValues.view());
        for (std::size_t k = 0; k < recovered.size(); ++k)
            for (std::size_t local = 0; local < localCount; ++local)
                if (isInteriorDof(local))
                    columnsG(static_cast<std::size_t>(unit.modelDofOfLocal[local]), recovered[k]) =
                        localValues(local, k);
        m_statistics.units[u].recoveredColumns += static_cast<std::int64_t>(recovered.size());
    }
}

} // namespace oos::design
