#include "oos/solutions/sol101/CondensedLoadBasis.hpp"

#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/reduction/condense/StaticCondensation.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/reduction/interface/InterfaceProblem.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/BasisFixedRows.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::solutions
{

namespace
{

void checkCancelled(const base::CancellationToken* cancellation)
{
    if (cancellation != nullptr)
        cancellation->throwIfCancelled();
}

// The names of a unit's equations for the diagnostics of its factorisations (as the condensed analysis names them).
linalg::EquationNamer equationNamerOf(const reduction::UnitSystem* system)
{
    return [system](std::int32_t equation)
    {
        const reduction::UnitDofLabel label = system->labelOf(system->localDofOfEquation(equation));
        return base::GridComponent{label.id, label.component};
    };
}

// Whether a unit's system numbers its local DOFs as the kept unit does (the numbering its loads are gathered in).
bool isSameNumbering(const KeptCondensedUnit& unit, const reduction::UnitSystem& system)
{
    if (static_cast<std::size_t>(system.localDofCount()) != unit.modelDofOfLocal.size())
        return false;
    for (std::size_t local = 0; local < unit.modelDofOfLocal.size(); ++local)
        if (system.modelDofOf(static_cast<std::int64_t>(local)).value_or(-1) != unit.modelDofOfLocal[local])
            return false;
    return true;
}

// One use of a kept unit's interior (under the memory budget): the condensation of the basis vectors' loads
// (Purpose::Loads) or a block's recovery of its interior (Purpose::Recovery). Its equations are built again when the
// memory mode released them (KeptCondensedUnit::rebuild; with the mass when inertia relief reads its inertial loads:
// the same stiffness to the bit either way), its numbering checked and its condensation bound to them; for a recovery
// its interior factor is made again when not kept (its K* checked to the bit), or its recovery made by regions of its
// elimination tree when its factor exceeds the plan's bound. finish() gives back what the use made: the factor of a
// unit that keeps none (KeptCondensedUnit::releasesFactor; in the mode that keeps every recovery the factor made stays
// with the set) and the equations built again (the system without them bound again). Destroyed without finish() (an
// exception), it releases the factor and leaves the equations bound until the next use.
class UnitUse
{
public:
    enum class Purpose : std::uint8_t
    {
        Loads,
        Recovery
    };

    UnitUse(const KeptCondensedUnit& unit, Purpose purpose, bool needsMass, const linalg::DenseKernels& kernels,
            base::MemoryBudget& budget, const base::CancellationToken* cancellation)
        : m_unit(&unit)
    {
        reduction::StaticCondensation& condensation = *unit.condensation;
        if (!unit.system->hasEquations())
        {
            if (!unit.rebuild)
                throw base::InternalError(base::concatenate(
                    {"the load basis: unit ", std::to_string(unit.unitId),
                     " has released its equations but kept nothing to build them again (an internal error: report "
                     "it)"}));
            const reduction::UnitModel& model = unit.rebuild->model();
            base::ThreadPool* pool = kernels.options().pool;
            m_rebuilt = needsMass ? std::make_shared<const reduction::UnitSystem>(model, unit.description, budget, pool)
                                  : std::make_shared<const reduction::UnitSystem>(
                                        model, unit.description,
                                        reduction::UnitSystem::WithoutMass{unit.system->rigidBodyMass()}, budget, pool);
            if (!isSameNumbering(unit, *m_rebuilt))
                throw base::InternalError(base::concatenate(
                    {"the load basis: unit ", std::to_string(unit.unitId),
                     ": its equations built again number its DOFs otherwise than its condensation's (an internal "
                     "error: report it)"}));
            condensation.releaseInteriorFactor();
            condensation.rebindSystem(m_rebuilt, equationNamerOf(m_rebuilt.get()));
        }
        if (purpose == Purpose::Recovery && !condensation.isInteriorFactorKept())
        {
            if (unit.regionValues > 0)
            {
                condensation.enableRecoveryWithoutFactor(unit.regionValues, budget);
                m_isByRegions = true;
            }
            else
                condensation.recomputeInteriorFactor(kernels, budget, cancellation);
        }
    }
    UnitUse(const UnitUse&) = delete;
    UnitUse& operator=(const UnitUse&) = delete;
    UnitUse(UnitUse&&) = delete;
    UnitUse& operator=(UnitUse&&) = delete;
    ~UnitUse() { release(); }

    // The unit's equations for this use.
    [[nodiscard]] const reduction::UnitSystem& system() const noexcept
    {
        return m_rebuilt ? *m_rebuilt : *m_unit->system;
    }

    // f* = f_b − K_bo·K_oo⁻¹·f_o of columns on its equations: by the kept factor's forward sweep; without a factor, by
    // the streamed factorisation when the unit keeps none (the same bits, its peak the fronts), else by the factor made
    // again, which stays with the set.
    [[nodiscard]] linalg::DenseMatrix condenseLoads(linalg::ConstDenseMatrixView loads,
                                                    const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                                    const base::CancellationToken* cancellation) const
    {
        reduction::StaticCondensation& condensation = *m_unit->condensation;
        if (!condensation.isInteriorFactorKept() && m_unit->releasesFactor)
            return condensation.condenseLoadsWithoutFactor(loads, kernels, budget, cancellation);
        if (!condensation.isInteriorFactorKept())
            condensation.recomputeInteriorFactor(kernels, budget, cancellation);
        return condensation.condenseLoads(loads, kernels, budget, cancellation);
    }

    void finish()
    {
        release();
        if (m_rebuilt)
        {
            m_unit->condensation->rebindSystem(m_unit->system, equationNamerOf(m_unit->system.get()));
            m_rebuilt.reset();
        }
    }

private:
    void release() noexcept
    {
        if (m_isByRegions)
        {
            m_unit->condensation->disableRecoveryWithoutFactor();
            m_isByRegions = false;
        }
        if (m_unit->releasesFactor)
            m_unit->condensation->releaseInteriorFactor();
    }

    const KeptCondensedUnit* m_unit;
    std::shared_ptr<const reduction::UnitSystem> m_rebuilt;
    bool m_isByRegions = false;
};

// Whether local DOF `local` of a unit is a row of its interior loads: an interior point's DOF of the model.
bool isInteriorRow(const KeptCondensedUnit& unit, std::size_t local) noexcept
{
    return unit.isInteriorLocal[local] != 0 && unit.modelDofOfLocal[local] >= 0;
}

constexpr std::size_t RIGID_MOTIONS = assembly::RigidBodyModes::MODE_COUNT;

// x = A⁻¹·b for the r × r reduced mass DᵀMD (r ≤ 6, row-major, symmetric positive definite: the analysis refused a
// singular one), by Gauss-Jordan elimination with partial pivoting.
std::array<double, assembly::MAX_SUPPORT_DOFS> solveReduced(const std::vector<double>& a, std::size_t r,
                                                            std::array<double, assembly::MAX_SUPPORT_DOFS> b)
{
    std::vector<double> m(a); // bounded by six × six
    for (std::size_t column = 0; column < r; ++column)
    {
        std::size_t pivotRow = column;
        for (std::size_t row = column + 1; row < r; ++row)
            if (std::abs(m[row * r + column]) > std::abs(m[pivotRow * r + column]))
                pivotRow = row;
        for (std::size_t k = 0; k < r; ++k)
            std::swap(m[column * r + k], m[pivotRow * r + k]);
        std::swap(b[column], b[pivotRow]);
        const double pivot = m[column * r + column];
        for (std::size_t row = 0; row < r; ++row)
        {
            if (row == column)
                continue;
            const double factor = m[row * r + column] / pivot;
            for (std::size_t k = 0; k < r; ++k)
                m[row * r + k] -= factor * m[column * r + k];
            b[row] -= factor * b[column];
        }
    }
    for (std::size_t k = 0; k < r; ++k)
        b[k] /= m[k * r + k];
    return b;
}

} // namespace

std::shared_ptr<const KeptCondensedBasis> solveCondensedBasis(std::shared_ptr<const CondensedFactorisedSet> set,
                                                              const StaticLoadCases& cases,
                                                              const linalg::DenseKernels& kernels,
                                                              base::MemoryBudget& budget, base::DiagnosticSink& sink,
                                                              const base::CancellationToken* cancellation,
                                                              KeptFixedRows* fixedRows)
{
    if (!set)
        throw std::invalid_argument(
            "solveCondensedBasis: expected a kept condensed constraint set, found none (a bug "
            "of O_o_Solver: report it)");
    const reduction::InterfaceProblem& interface = set->interface();
    const KeptCondensedUnits& units = set->units();
    const auto gSize = static_cast<std::size_t>(set->dofMap().gSetSize());
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    const std::size_t k = cases.basisSize();
    auto basis = std::make_shared<KeptCondensedBasis>();
    basis->set = set;
    basis->unitLoads.resize(units.size());
    const CondensedRelief* relief =
        set->relief() != nullptr && set->relief()->reliefLoads.columns() == RIGID_MOTIONS ? set->relief() : nullptr;
    basis->reliefAmplitudes =
        linalg::DenseMatrix(budget, RIGID_MOTIONS, relief != nullptr ? k : 0, "condensed load basis relief amplitudes");

    // 1. The vectors' rows: the interface's on its local DOFs, each unit's interior rows on its local DOFs (the columns
    // that load it, one local vector each, in increasing order).
    linalg::DenseMatrix interfaceLoads(budget, localDofs, k, "condensed load basis interface loads");
    std::vector<std::vector<base::BudgetedBuffer<double>>> unitColumns(units.size()); // bounded by units × k
    {
        base::BudgetedBuffer<double> loads(budget, gSize, "condensed load basis vector loads");
        base::BudgetedBuffer<double> enforced(budget, gSize, "condensed load basis vector enforced values");
        for (std::size_t b = 0; b < k; ++b)
        {
            checkCancelled(cancellation);
            cases.basisVector(b, loads.span(), enforced.span(), budget, sink);
            if (std::any_of(enforced.data(), enforced.data() + gSize, [](double value) { return value != 0.0; }))
                throw std::logic_error(
                    "solveCondensedBasis: a basis vector enforces a displacement, which the load "
                    "basis of the condensed path refuses beforehand (a bug of O_o_Solver: report "
                    "it)");
            // Inertia relief: ü = (DᵀMD)⁻¹·Cᵀ·Φᵀ·P of the vector's loads, a = C·ü the amplitudes of its relief loads
            // −M·Φ·a (the units' condensed inertial patterns and the interface's inertial loads at the interface, the
            // units' inertial loads in their interiors at the recovery).
            if (relief != nullptr && !relief->rigidMass.empty())
            {
                // Six supports, C exact: a = Mr⁻¹·Φᵀ·P without C, the relieved loads independent of the
                // FreeBodySupport DOFs in floating point.
                const std::array<double, RIGID_MOTIONS> amplitudes = solveReduced(
                    relief->rigidMass, RIGID_MOTIONS, assembly::rigidResultantOf(set->dofMap(), loads.span()));
                for (std::size_t m = 0; m < RIGID_MOTIONS; ++m)
                    basis->reliefAmplitudes(m, b) = amplitudes[m];
            }
            else if (relief != nullptr)
            {
                const std::array<double, RIGID_MOTIONS> resultant =
                    assembly::rigidResultantOf(set->dofMap(), loads.span());
                const std::size_t r = relief->supportCount;
                std::array<double, assembly::MAX_SUPPORT_DOFS> projected{};
                for (std::size_t j = 0; j < r; ++j)
                    for (std::size_t m = 0; m < RIGID_MOTIONS; ++m)
                        projected[j] += relief->fit[j * RIGID_MOTIONS + m] * resultant[m];
                const std::array<double, assembly::MAX_SUPPORT_DOFS> accelerations =
                    solveReduced(relief->reducedMass, r, projected);
                for (std::size_t m = 0; m < RIGID_MOTIONS; ++m)
                {
                    double amplitude = 0.0;
                    for (std::size_t j = 0; j < r; ++j)
                        amplitude += relief->fit[j * RIGID_MOTIONS + m] * accelerations[j];
                    basis->reliefAmplitudes(m, b) = amplitude;
                }
            }
            for (std::size_t local = 0; local < localDofs; ++local)
                if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(local)))
                    interfaceLoads(local, b) = loads[static_cast<std::size_t>(*dof)];
            // appliedLoad at the DOFs the basis keeps: the vector's loads, and at the interface's points its own relief
            // loads, as statics writes them (the units' at their recovery).
            if (fixedRows != nullptr)
            {
                BasisFixedRows::take(*fixedRows, b, loads.span());
                if (relief != nullptr)
                    for (std::size_t local = 0; local < localDofs; ++local)
                        if (const std::optional<base::DofIndex> dof =
                                interface.modelDofOf(static_cast<std::int64_t>(local)))
                            if (const std::optional<std::size_t> place = fixedRows->placeOf(*dof))
                            {
                                double value = 0.0;
                                for (std::size_t m = 0; m < RIGID_MOTIONS; ++m)
                                    value += interface.inertialLoads()(local, m) * basis->reliefAmplitudes(m, b);
                                fixedRows->loads(*place, b) += value;
                            }
            }
            for (std::size_t u = 0; u < units.size(); ++u)
            {
                const KeptCondensedUnit& unit = units[u];
                const std::size_t localCount = unit.modelDofOfLocal.size();
                bool isLoaded = false;
                for (std::size_t local = 0; local < localCount && !isLoaded; ++local)
                    isLoaded = isInteriorRow(unit, local) &&
                               loads[static_cast<std::size_t>(unit.modelDofOfLocal[local])] != 0.0;
                if (!isLoaded)
                    continue;
                base::BudgetedBuffer<double> column(budget, localCount, "condensed load basis unit loads");
                for (std::size_t local = 0; local < localCount; ++local)
                    if (isInteriorRow(unit, local))
                        column[local] = loads[static_cast<std::size_t>(unit.modelDofOfLocal[local])];
                basis->unitLoads[u].columns.push_back(b);
                unitColumns[u].push_back(std::move(column));
            }
        }
    }

    // 2. Each loaded unit: its loads on its equations (kept for the recovery), condensed onto its boundary and added to
    // the interface's loads there.
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        KeptCondensedBasis::UnitLoads& entry = basis->unitLoads[u];
        if (entry.columns.empty())
            continue;
        checkCancelled(cancellation);
        const KeptCondensedUnit& unit = units[u];
        const std::size_t localCount = unit.modelDofOfLocal.size();
        linalg::DenseMatrix localLoads(budget, localCount, entry.columns.size(), "condensed load basis unit loads");
        for (std::size_t c = 0; c < entry.columns.size(); ++c)
            std::copy(unitColumns[u][c].data(), unitColumns[u][c].data() + localCount,
                      localLoads.data() + c * localLoads.leadingDimension());
        unitColumns[u].clear();
        UnitUse use(unit, UnitUse::Purpose::Loads, false, kernels, budget, cancellation);
        entry.equations = linalg::DenseMatrix(budget, static_cast<std::size_t>(use.system().equationCount()),
                                              entry.columns.size(), "condensed load basis unit equation loads");
        use.system().equationLoads(localLoads.view(), {}, entry.equations.view(), budget);
        const linalg::DenseMatrix condensed = use.condenseLoads(entry.equations.view(), kernels, budget, cancellation);
        use.finish();
        const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
        for (std::size_t c = 0; c < entry.columns.size(); ++c)
            for (std::size_t i = 0; i < locals.size(); ++i)
                interfaceLoads(static_cast<std::size_t>(locals[i]), entry.columns[c]) += condensed(i, c);
    }

    // The relief loads of each vector at the interface: Σ_m a_m·(the units' condensed inertial patterns + the
    // interface's inertial loads), the statics' relief block.
    if (relief != nullptr)
        for (std::size_t b = 0; b < k; ++b)
            for (std::size_t local = 0; local < localDofs; ++local)
            {
                double value = 0.0;
                for (std::size_t m = 0; m < RIGID_MOTIONS; ++m)
                    value += relief->reliefLoads(local, m) * basis->reliefAmplitudes(m, b);
                interfaceLoads(local, b) += value;
            }

    // 3. The interface problem, once for the k vectors.
    interface.reduceLoads(interfaceLoads.view());
    linalg::DenseMatrix equations(budget, static_cast<std::size_t>(interface.equationCount()), k,
                                  "condensed load basis interface right-hand sides");
    interface.equationLoads(interfaceLoads.view(), {}, equations.view());
    interface.solve(equations.view(), kernels, cancellation);
    // Without its factor the interface's pivots come with its first solve: a factor kept with failed pivots (PARAM
    // BAILOUT -1) is reported here, as the analysis reports it with the factor.
    if (!interface.isFactorKept() && !set->subcases().empty())
        reportKeptPivots(
            interface.pivotDiagnostics(), [&interface](std::int32_t equation)
            { return interface.nameOf(interface.localDofOfEquation(equation)); }, set->subcases().front(), sink);
    basis->interfaceDisplacements =
        linalg::DenseMatrix(budget, localDofs, k, "condensed load basis interface displacements");
    interface.expand(equations.view(), {}, basis->interfaceDisplacements.view());
    // Conditioned supports: the rigid motion −Φ·C·u_r of each vector that brings its FreeBodySupport DOFs back to rest,
    // added when its displacements are filled (after the interiors are recovered from the interface's solution).
    if (relief != nullptr && relief->isConditioned)
    {
        basis->gaugeAmplitudes = linalg::DenseMatrix(budget, RIGID_MOTIONS, k, "condensed load basis gauge");
        for (std::size_t b = 0; b < k; ++b)
            for (std::size_t j = 0; j < relief->supportCount; ++j)
            {
                const double atSupport = basis->interfaceDisplacements(relief->supportLocals[j], b);
                for (std::size_t m = 0; m < RIGID_MOTIONS; ++m)
                    basis->gaugeAmplitudes(m, b) -= relief->fit[j * RIGID_MOTIONS + m] * atSupport;
            }
    }
    return basis;
}

CondensedBasisColumns::CondensedBasisColumns(const KeptCondensedBasis& basis, std::span<const std::size_t> columns,
                                             std::span<const std::uint8_t> isUnitRead,
                                             const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                             const base::CancellationToken* cancellation)
    : m_basis(&basis), m_columns(columns.begin(), columns.end())
{
    const CondensedFactorisedSet& set = *basis.set;
    const reduction::InterfaceProblem& interface = set.interface();
    const KeptCondensedUnits& units = set.units();
    const std::size_t unitCount = basis.interfaceDisplacements.columns();
    if (isUnitRead.size() != units.size())
        throw std::invalid_argument(base::concatenate({"CondensedBasisColumns: expected ", std::to_string(units.size()),
                                                       " unit flags, found ", std::to_string(isUnitRead.size())}));
    for (const std::size_t column : m_columns)
        if (column >= unitCount)
            throw std::invalid_argument(
                base::concatenate({"CondensedBasisColumns: expected columns below ", std::to_string(unitCount),
                                   ", found ", std::to_string(column)}));
    m_interiors.resize(units.size());
    m_inertial.resize(units.size());
    const std::size_t count = m_columns.size();
    if (count == 0)
        return;
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        if (isUnitRead[u] == 0)
            continue;
        checkCancelled(cancellation);
        const KeptCondensedUnit& unit = units[u];
        const KeptCondensedBasis::UnitLoads& loads = basis.unitLoads[u];
        const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
        const std::size_t boundary = locals.size();
        const bool isRelieved = basis.reliefAmplitudes.columns() > 0;
        UnitUse use(unit, UnitUse::Purpose::Recovery, isRelieved, kernels, budget, cancellation);
        const reduction::UnitSystem& system = use.system();
        const auto equationCount = static_cast<std::size_t>(system.equationCount());
        // The position of each column asked among those loading the unit (or none).
        std::vector<std::optional<std::size_t>> loadedAt(count); // bounded by the columns asked
        std::size_t loadedCount = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto found = std::lower_bound(loads.columns.begin(), loads.columns.end(), m_columns[i]);
            if (found != loads.columns.end() && *found == m_columns[i])
            {
                loadedAt[i] = static_cast<std::size_t>(found - loads.columns.begin());
                ++loadedCount;
            }
        }
        // Inertia relief: the unit's inertial loads −M·Φ on its equations, those of a column a_m times each.
        linalg::DenseMatrix inertial(budget, equationCount, isRelieved ? RIGID_MOTIONS : 0,
                                     "condensed load basis unit inertial loads");
        if (isRelieved)
            system.equationLoads(system.inertialLoads().view(), {}, inertial.view(), budget);
        if (isRelieved)
        {
            // Its −M·Φ on its local DOFs, kept for the interior part of the columns' appliedLoad.
            const linalg::DenseMatrix& local = system.inertialLoads();
            linalg::DenseMatrix& kept =
                m_inertial[u].emplace(budget, local.rows(), local.columns(), "condensed load basis unit relief loads");
            for (std::size_t m = 0; m < local.columns(); ++m)
                for (std::size_t r = 0; r < local.rows(); ++r)
                    kept(r, m) = local(r, m);
        }
        const std::size_t inertialCount = isRelieved ? RIGID_MOTIONS : 0;
        linalg::DenseMatrix solution(budget, equationCount, count, "condensed load basis unit solution");
        if (count <= boundary + loadedCount + inertialCount)
        {
            // The columns asked: u_o = K_oo⁻¹·(f_o − K_oa·u_a), one solve each.
            linalg::DenseMatrix boundaryValues(budget, boundary, count, "condensed load basis unit boundary");
            for (std::size_t i = 0; i < count; ++i)
            {
                if (loadedAt[i])
                    for (std::size_t e = 0; e < equationCount; ++e)
                        solution(e, i) = loads.equations(e, *loadedAt[i]);
                for (std::size_t m = 0; m < inertialCount; ++m)
                {
                    const double amplitude = basis.reliefAmplitudes(m, m_columns[i]);
                    for (std::size_t e = 0; e < equationCount && amplitude != 0.0; ++e)
                        solution(e, i) += amplitude * inertial(e, m);
                }
                for (std::size_t a = 0; a < boundary; ++a)
                    boundaryValues(a, i) =
                        basis.interfaceDisplacements(static_cast<std::size_t>(locals[a]), m_columns[i]);
            }
            unit.condensation->recover(solution.view(), boundaryValues.view(), kernels, cancellation);
            m_interiorSolves += static_cast<std::int64_t>(count);
        }
        else
        {
            // H = −K_oo⁻¹·K_oa (a unit motion of each boundary DOF, no load: every row of its solution, the boundary's
            // the identity), the loaded columns' own g = K_oo⁻¹·f_o (their boundary at rest), then u_c = H·u_a,c + g_c.
            linalg::DenseMatrix motions(budget, equationCount, boundary, "condensed load basis unit boundary motions");
            linalg::DenseMatrix identity(budget, boundary, boundary, "condensed load basis unit boundary identity");
            for (std::size_t a = 0; a < boundary; ++a)
                identity(a, a) = 1.0;
            unit.condensation->recover(motions.view(), identity.view(), kernels, cancellation);
            std::vector<std::size_t> loadedColumns; // bounded by the columns asked
            for (std::size_t i = 0; i < count; ++i)
                if (loadedAt[i])
                    loadedColumns.push_back(i);
            if (!loadedColumns.empty())
            {
                linalg::DenseMatrix own(budget, equationCount, loadedColumns.size(), "condensed load basis unit g");
                for (std::size_t c = 0; c < loadedColumns.size(); ++c)
                    for (std::size_t e = 0; e < equationCount; ++e)
                        own(e, c) = loads.equations(e, *loadedAt[loadedColumns[c]]);
                const linalg::DenseMatrix atRest(budget, boundary, loadedColumns.size(),
                                                 "condensed load basis unit boundary at rest");
                unit.condensation->recover(own.view(), atRest.view(), kernels, cancellation);
                for (std::size_t c = 0; c < loadedColumns.size(); ++c)
                    for (std::size_t e = 0; e < equationCount; ++e)
                        solution(e, loadedColumns[c]) = own(e, c);
            }
            // The relief: g_m = K_oo⁻¹·(−M·Φ)_m (its boundary at rest), then solution += Σ_m a_m·g_m.
            if (inertialCount > 0)
            {
                const linalg::DenseMatrix atRest(budget, boundary, inertialCount,
                                                 "condensed load basis unit boundary at rest");
                unit.condensation->recover(inertial.view(), atRest.view(), kernels, cancellation);
                for (std::size_t i = 0; i < count; ++i)
                    for (std::size_t m = 0; m < inertialCount; ++m)
                    {
                        const double amplitude = basis.reliefAmplitudes(m, m_columns[i]);
                        for (std::size_t e = 0; e < equationCount && amplitude != 0.0; ++e)
                            solution(e, i) += amplitude * inertial(e, m);
                    }
            }
            // solution ← solution − H·(−U_a): the boundary displacements of the columns, negated.
            linalg::DenseMatrix negated(budget, boundary, count, "condensed load basis unit boundary");
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t a = 0; a < boundary; ++a)
                    negated(a, i) = -basis.interfaceDisplacements(static_cast<std::size_t>(locals[a]), m_columns[i]);
            kernels.subtractProduct(solution.view(), motions.view(), negated.view());
            m_interiorSolves += static_cast<std::int64_t>(boundary + loadedColumns.size() + inertialCount);
        }
        linalg::DenseMatrix& values =
            m_interiors[u].emplace(budget, unit.modelDofOfLocal.size(), count, "condensed load basis unit interior");
        system.expandSolution(solution.view(), {}, values.view());
        use.finish();
    }
}

void CondensedBasisColumns::fill(std::size_t i, std::span<double> displacementsG) const
{
    const CondensedFactorisedSet& set = *m_basis->set;
    const reduction::InterfaceProblem& interface = set.interface();
    const KeptCondensedUnits& units = set.units();
    if (i >= m_columns.size() || displacementsG.size() != static_cast<std::size_t>(set.dofMap().gSetSize()))
        throw std::invalid_argument(base::concatenate(
            {"CondensedBasisColumns::fill: expected a column below ", std::to_string(m_columns.size()), " and ",
             std::to_string(set.dofMap().gSetSize()), " g-set values, found column ", std::to_string(i), " and ",
             std::to_string(displacementsG.size()), " values"}));
    std::fill(displacementsG.begin(), displacementsG.end(), 0.0);
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    for (std::size_t local = 0; local < localDofs; ++local)
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(local)))
            displacementsG[static_cast<std::size_t>(*dof)] = m_basis->interfaceDisplacements(local, m_columns[i]);
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        if (!m_interiors[u])
            continue;
        const KeptCondensedUnit& unit = units[u];
        const linalg::DenseMatrix& values = *m_interiors[u];
        for (std::size_t local = 0; local < unit.modelDofOfLocal.size(); ++local)
            if (isInteriorRow(unit, local))
                displacementsG[static_cast<std::size_t>(unit.modelDofOfLocal[local])] = values(local, i);
    }
    // Conditioned supports: back to the gauge of the FreeBodySupport DOFs (every DOF but the held ones), the
    // FreeBodySupport DOFs at rest.
    if (m_basis->gaugeAmplitudes.columns() > 0)
    {
        std::array<double, RIGID_MOTIONS> amplitudes{};
        for (std::size_t m = 0; m < RIGID_MOTIONS; ++m)
            amplitudes[m] = m_basis->gaugeAmplitudes(m, m_columns[i]);
        assembly::addRigidMotion(set.dofMap(), amplitudes, displacementsG, set.isConstrained());
        for (const std::size_t local : set.relief()->supportLocals)
            if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(local)))
                displacementsG[static_cast<std::size_t>(*dof)] = 0.0;
    }
}

void CondensedBasisColumns::addUnitReliefLoads(std::size_t i, std::span<double> loadsG) const
{
    const KeptCondensedUnits& units = m_basis->set->units();
    if (i >= m_columns.size() || m_basis->reliefAmplitudes.columns() == 0)
        return;
    for (std::size_t u = 0; u < units.size(); ++u)
    {
        if (!m_inertial[u])
            continue;
        const KeptCondensedUnit& unit = units[u];
        const linalg::DenseMatrix& inertial = *m_inertial[u];
        // Every local DOF of the unit, its boundary's too: a unit's relief pattern −M·Φ is on all of them (its mass at
        // its boundary points is its own), and statics adds each unit's part at a boundary point.
        for (std::size_t local = 0; local < unit.modelDofOfLocal.size(); ++local)
        {
            if (unit.modelDofOfLocal[local] < 0)
                continue;
            double value = 0.0;
            for (std::size_t m = 0; m < inertial.columns(); ++m)
                value += inertial(local, m) * m_basis->reliefAmplitudes(m, m_columns[i]);
            loadsG[static_cast<std::size_t>(unit.modelDofOfLocal[local])] += value;
        }
    }
}

} // namespace oos::solutions
