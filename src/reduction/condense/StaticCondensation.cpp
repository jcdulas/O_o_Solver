#include "oos/reduction/condense/StaticCondensation.hpp"

#include "oos/base/errors/CacheError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

constexpr std::size_t INERTIAL_PATTERN_COUNT = 6;

linalg::DenseMatrix copyOf(const linalg::DenseMatrix& matrix, base::MemoryBudget& budget, const std::string& label)
{
    linalg::DenseMatrix copy(budget, matrix.rows(), matrix.columns(), label);
    for (std::size_t j = 0; j < matrix.columns(); ++j)
        for (std::size_t i = 0; i < matrix.rows(); ++i)
            copy(i, j) = matrix(i, j);
    return copy;
}

} // namespace

StaticCondensation::StaticCondensation(std::shared_ptr<const UnitSystem> system, const UnitPatterns& patterns,
                                       const linalg::DenseKernels& kernels, const linalg::FactorOptions& options,
                                       base::MemoryBudget& budget)
    : m_system(std::move(system)), m_options(options)
{
    if (m_system == nullptr)
        throw std::invalid_argument("StaticCondensation: no unit system; build the unit's UnitSystem first");
    const UnitSystem& unit = *m_system;
    const auto localDofs = static_cast<std::size_t>(unit.localDofCount());
    if (patterns.loads.rows != localDofs && patterns.loads.columns > 0)
        throw std::invalid_argument(
            base::concatenate({"StaticCondensation (unit ", std::to_string(unit.unitId()), "): patterns of ",
                               std::to_string(patterns.loads.rows), " rows for ", std::to_string(localDofs),
                               " local DOFs; give the patterns on the unit's local DOFs (UnitSystem::gatherLocal)"}));
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const std::size_t patternCount = patterns.loads.columns;

    // The loads of the equations: the patterns, then the six inertial patterns −M·Φ_r.
    linalg::DenseMatrix loads(budget, equations, patternCount, "unit pattern loads");
    if (patternCount > 0)
        unit.equationLoads(patterns.loads, patterns.enforced, loads.view(), budget);
    linalg::DenseMatrix inertial(budget, equations, INERTIAL_PATTERN_COUNT, "unit inertial loads");
    unit.equationLoads(unit.inertialLoads().view(), {}, inertial.view(), budget);

    m_factor = linalg::PartialFactor::factorise(unit.symbolic(), unit.matrix(), kernels, m_options, budget);
    linalg::DenseMatrix condensed =
        patternCount > 0
            ? m_factor.condenseLoads(loads.view(), kernels, budget, m_options.cancellation)
            : linalg::DenseMatrix(budget, static_cast<std::size_t>(unit.boundaryDofCount()), 0, "condensed loads");
    linalg::DenseMatrix condensedInertial =
        m_factor.condenseLoads(inertial.view(), kernels, budget, m_options.cancellation);

    base::BudgetedBuffer<UnitDofLabel> labels(budget, unit.boundaryLabels().size(), "reduced unit labels");
    for (std::size_t k = 0; k < labels.size(); ++k)
        labels[k] = unit.boundaryLabels()[k];
    m_reduced = std::make_shared<const ReducedUnit>(
        unit.unitId(), std::move(labels), copyOf(m_factor.schurComplement(), budget, "reduced unit K*"),
        std::move(condensed), patterns.stamp, std::move(condensedInertial), unit.rigidBodyMass());
    m_hasFactorised = true;
    m_kernel = kernels.kind();
    m_largestPivotRatio = m_factor.interiorFactor().pivotDiagnostics().largestRatio();
}

StaticCondensation::StaticCondensation(std::shared_ptr<const UnitSystem> system, const UnitPatterns& patterns,
                                       const linalg::DenseKernels& kernels, const linalg::FactorOptions& options,
                                       base::MemoryBudget& budget, WithoutFactor)
    : m_system(std::move(system)), m_options(options), m_isWithoutFactor(true)
{
    if (m_system == nullptr)
        throw std::invalid_argument("StaticCondensation: no unit system; build the unit's UnitSystem first");
    const UnitSystem& unit = *m_system;
    const auto localDofs = static_cast<std::size_t>(unit.localDofCount());
    if (patterns.loads.rows != localDofs && patterns.loads.columns > 0)
        throw std::invalid_argument(
            base::concatenate({"StaticCondensation (unit ", std::to_string(unit.unitId()), "): patterns of ",
                               std::to_string(patterns.loads.rows), " rows for ", std::to_string(localDofs),
                               " local DOFs; give the patterns on the unit's local DOFs (UnitSystem::gatherLocal)"}));
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto interior = static_cast<std::size_t>(unit.interiorDofCount());
    const auto boundary = static_cast<std::size_t>(unit.boundaryDofCount());
    const std::size_t patternCount = patterns.loads.columns;

    // The loads of the patterns, then the six inertial patterns −M·Φ_r, on the equations, swept forward in place while
    // the interior is factorised (each column has the bits it has alone: the first constructor's condenseLoads).
    linalg::DenseMatrix loads(budget, equations, patternCount + INERTIAL_PATTERN_COUNT, "unit pattern loads");
    const linalg::DenseMatrixView all = loads.view();
    if (patternCount > 0)
        unit.equationLoads(patterns.loads, patterns.enforced, all.block(0, 0, equations, patternCount), budget);
    unit.equationLoads(unit.inertialLoads().view(), {}, all.block(0, patternCount, equations, INERTIAL_PATTERN_COUNT),
                       budget);
    linalg::PartialCondensation condensation =
        linalg::PartialFactor::condenseWithoutFactor(unit.symbolic(), unit.matrix(), all, kernels, m_options, budget);

    linalg::DenseMatrix condensed(budget, boundary, patternCount, "condensed loads");
    for (std::size_t j = 0; j < patternCount; ++j)
        std::copy_n(&loads(interior, j), boundary, &condensed(0, j));
    linalg::DenseMatrix condensedInertial(budget, boundary, INERTIAL_PATTERN_COUNT, "condensed loads");
    for (std::size_t j = 0; j < INERTIAL_PATTERN_COUNT; ++j)
        std::copy_n(&loads(interior, patternCount + j), boundary, &condensedInertial(0, j));
    // The pattern work G = yᵀ·D⁻¹·y with y = L⁻¹·f_o, the expressions of patternWork(): z = D⁻¹·y as FactorSolve's
    // diagonal sweep writes it (x / d, without −0), then the sums over the interior rows in increasing order.
    m_patternWork = linalg::DenseMatrix(budget, patternCount, patternCount, "unit pattern work");
    if (patternCount > 0)
    {
        linalg::DenseMatrix scaled(budget, interior, patternCount, "unit pattern sweep scaled");
        for (std::size_t j = 0; j < patternCount; ++j)
            for (std::size_t i = 0; i < interior; ++i)
                scaled(i, j) = loads(i, j) / condensation.pivots[i] + 0.0;
        for (std::size_t b = 0; b < patternCount; ++b)
            for (std::size_t a = b; a < patternCount; ++a)
            {
                double sum = 0.0;
                for (std::size_t i = 0; i < interior; ++i)
                    sum += loads(i, a) * scaled(i, b);
                m_patternWork(a, b) = sum;
                m_patternWork(b, a) = sum;
            }
    }

    base::BudgetedBuffer<UnitDofLabel> labels(budget, unit.boundaryLabels().size(), "reduced unit labels");
    for (std::size_t k = 0; k < labels.size(); ++k)
        labels[k] = unit.boundaryLabels()[k];
    m_reduced = std::make_shared<const ReducedUnit>(unit.unitId(), std::move(labels),
                                                    std::move(condensation.schurComplement), std::move(condensed),
                                                    patterns.stamp, std::move(condensedInertial), unit.rigidBodyMass());
    m_hasFactorised = true;
    m_kernel = kernels.kind();
    m_largestPivotRatio = condensation.largestPivotRatio;
}

StaticCondensation::StaticCondensation(std::shared_ptr<const UnitSystem> system, const UnitPatterns& patterns,
                                       KeptInteriorFactor kept, const linalg::DenseKernels& kernels,
                                       const linalg::FactorOptions& options, base::MemoryBudget& budget)
    : m_system(std::move(system)), m_options(options)
{
    if (m_system == nullptr)
        throw std::invalid_argument("StaticCondensation: no unit system; build the unit's UnitSystem first");
    const UnitSystem& unit = *m_system;
    const auto localDofs = static_cast<std::size_t>(unit.localDofCount());
    if (patterns.loads.rows != localDofs && patterns.loads.columns > 0)
        throw std::invalid_argument(
            base::concatenate({"StaticCondensation (unit ", std::to_string(unit.unitId()), "): patterns of ",
                               std::to_string(patterns.loads.rows), " rows for ", std::to_string(localDofs),
                               " local DOFs; give the patterns on the unit's local DOFs (UnitSystem::gatherLocal)"}));
    takeKeptFactor(std::move(kept), kernels, budget);
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const std::size_t patternCount = patterns.loads.columns;

    // As the first constructor, the factorisation replaced by the kept factor (the same bits: it is the factorisation
    // of these equations): the patterns and the inertial patterns −M·Φ_r of the unit's mass now, condensed by its
    // forward sweep.
    linalg::DenseMatrix loads(budget, equations, patternCount, "unit pattern loads");
    if (patternCount > 0)
        unit.equationLoads(patterns.loads, patterns.enforced, loads.view(), budget);
    linalg::DenseMatrix inertial(budget, equations, INERTIAL_PATTERN_COUNT, "unit inertial loads");
    unit.equationLoads(unit.inertialLoads().view(), {}, inertial.view(), budget);
    linalg::DenseMatrix condensed =
        patternCount > 0
            ? m_factor.condenseLoads(loads.view(), kernels, budget, m_options.cancellation)
            : linalg::DenseMatrix(budget, static_cast<std::size_t>(unit.boundaryDofCount()), 0, "condensed loads");
    linalg::DenseMatrix condensedInertial =
        m_factor.condenseLoads(inertial.view(), kernels, budget, m_options.cancellation);

    base::BudgetedBuffer<UnitDofLabel> labels(budget, unit.boundaryLabels().size(), "reduced unit labels");
    for (std::size_t k = 0; k < labels.size(); ++k)
        labels[k] = unit.boundaryLabels()[k];
    m_reduced = std::make_shared<const ReducedUnit>(
        unit.unitId(), std::move(labels), copyOf(m_factor.schurComplement(), budget, "reduced unit K*"),
        std::move(condensed), patterns.stamp, std::move(condensedInertial), unit.rigidBodyMass());
    m_hasFactorised = true;
    m_kernel = kernels.kind();
    m_largestPivotRatio = m_factor.interiorFactor().pivotDiagnostics().largestRatio();
}

StaticCondensation::StaticCondensation(std::shared_ptr<const UnitSystem> system,
                                       std::shared_ptr<const ReducedUnit> reduced, double largestPivotRatio,
                                       const linalg::FactorOptions& options)
    : m_system(std::move(system)),
      m_options(options),
      m_reduced(std::move(reduced)),
      m_isFromCache(true),
      m_largestPivotRatio(largestPivotRatio)
{
    if (m_system == nullptr || m_reduced == nullptr)
        throw std::invalid_argument(
            "StaticCondensation: a condensation from the cache needs the unit's system and its cached reduced unit");
    const std::span<const UnitDofLabel> labels = m_system->boundaryLabels();
    const std::span<const UnitDofLabel> cached = m_reduced->labels();
    if (!std::equal(labels.begin(), labels.end(), cached.begin(), cached.end()))
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(m_system->unitId()), "): the cached reduced unit has ",
             std::to_string(cached.size()), " boundary DOFs other than the unit's ", std::to_string(labels.size()),
             "; the cache key must name the unit's boundary"}));
}

std::uint64_t StaticCondensation::interiorFactorBytes() const noexcept
{
    if (!m_factor.isInteriorFactorKept())
        return 0;
    const linalg::SupernodalFactor& interior = m_factor.interiorFactor();
    const auto trailing = static_cast<std::uint64_t>(m_factor.trailingDofCount());
    std::uint64_t rows = 0;
    for (std::int32_t s = 0; s < interior.supernodeCount(); ++s)
        rows += interior.panel(s).rows.size();
    return static_cast<std::uint64_t>(interior.allocatedPanelValueCount()) * sizeof(double) +
           rows * sizeof(std::int32_t) + trailing * trailing * sizeof(double);
}

void StaticCondensation::rebindSystem(std::shared_ptr<const UnitSystem> system, linalg::EquationNamer equationNamer)
{
    if (system == nullptr)
        throw std::invalid_argument("StaticCondensation::rebindSystem: no unit system; give another build of the unit");
    if (m_factor.isInteriorFactorKept())
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(m_system->unitId()),
             "): its interior factor is kept while its equations are replaced; release the factor first"}));
    const std::span<const UnitDofLabel> labels = system->boundaryLabels();
    const std::span<const UnitDofLabel> kept = m_reduced->labels();
    if (system->unitId() != m_system->unitId() || !std::equal(labels.begin(), labels.end(), kept.begin(), kept.end()) ||
        system->interiorDofCount() != m_system->interiorDofCount())
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(m_system->unitId()), "): the system of unit ",
             std::to_string(system->unitId()), " with ", std::to_string(labels.size()), " boundary and ",
             std::to_string(system->interiorDofCount()), " interior equations replaces one of ",
             std::to_string(kept.size()), " and ", std::to_string(m_system->interiorDofCount()),
             "; give a build of the same unit"}));
    m_system = std::move(system);
    m_options.equationNamer = std::move(equationNamer);
}

void StaticCondensation::takeKeptFactor(KeptInteriorFactor kept, const linalg::DenseKernels& kernels,
                                        base::MemoryBudget& budget)
{
    const UnitSystem& unit = *m_system;
    if (!kept.factor.isFactorisationOf(unit.symbolic(), unit.matrix(), kernels.kind(), m_options.pivotPolicy))
        throw base::InternalError(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(unit.unitId()),
             "): the interior factor kept by an earlier analysis is not the factorisation of the unit's equations "
             "with these kernels and pivot policy (KeptInteriorFactors compares them: an internal error, report it)"}));
    // The loan first: when its bytes do not fit, nothing has changed and the kept factor is freed with `kept`.
    base::MemoryReservation loan =
        budget.reserve(kept.bytes, base::concatenate({"kept interior factor of unit ", std::to_string(unit.unitId())}));
    // The owner of the factor's budget is moved before the factor (released after it).
    m_factorMemoryOwner = std::move(kept.factorMemoryOwner);
    m_factor = std::move(kept.factor);
    m_factorLoan = std::move(loan);
    m_isInteriorFactorReused = true;
}

bool StaticCondensation::isSchurComplementTheReducedOne() const noexcept
{
    const linalg::DenseMatrix& computed = m_factor.schurComplement();
    const linalg::DenseMatrix& kept = m_reduced->stiffness();
    bool isSame = computed.rows() == kept.rows() && computed.columns() == kept.columns();
    for (std::size_t j = 0; j < computed.columns() && isSame; ++j)
        for (std::size_t i = 0; i < computed.rows() && isSame; ++i)
            isSame = std::bit_cast<std::uint64_t>(computed(i, j)) == std::bit_cast<std::uint64_t>(kept(i, j));
    return isSame;
}

void StaticCondensation::adoptInteriorFactor(KeptInteriorFactor kept, const linalg::DenseKernels& kernels,
                                             base::MemoryBudget& budget)
{
    if (m_factor.isInteriorFactorKept())
        throw std::logic_error(base::concatenate({"StaticCondensation (unit ", std::to_string(m_system->unitId()),
                                                  "): its interior factor is kept; a kept factor replaces a factor "
                                                  "released or never made"}));
    if (m_hasFactorised && kernels.kind() != m_kernel)
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(m_system->unitId()), "): its factor was made with the ",
             linalg::kernelName(m_kernel), " kernels, the kept factor is taken with the ",
             linalg::kernelName(kernels.kind()), " kernels, which give other bits; take it with the same family"}));
    takeKeptFactor(std::move(kept), kernels, budget);
    const bool isFirst = !m_hasFactorised;
    m_hasFactorised = true;
    m_kernel = kernels.kind();
    // Its K* must be the reduced unit's, to the bit, as a recomputation's (recompute).
    if (isSchurComplementTheReducedOne())
        return;
    releaseInteriorFactor();
    if (m_isFromCache && isFirst)
        throw base::CacheError(base::concatenate(
            {"the cached condensation of unit ", std::to_string(m_system->unitId()),
             " differs from the one its equations give now (its K* is not the same to the bit): the cache entry does "
             "not describe the unit as it is; clear the cache (O_o_Solver.cache.clear()) and run again, and report "
             "the case (a cache key that misses a change)"}));
    throw base::InternalError(base::concatenate(
        {"the interior factor of unit ", std::to_string(m_system->unitId()),
         " kept by an earlier analysis gives a K* other than its condensation's: the factorisation is not "
         "deterministic (an internal error: report it with the model)"}));
}

std::optional<KeptInteriorFactor> StaticCondensation::giveInteriorFactor(
    std::uint64_t unitIdentity, const std::shared_ptr<base::MemoryBudget>& analysisBudget)
{
    if (!m_factor.isInteriorFactorKept() || !m_system->hasEquations())
        return std::nullopt;
    std::shared_ptr<base::MemoryBudget> owner = m_factorMemoryOwner ? m_factorMemoryOwner : analysisBudget;
    if (owner == nullptr)
        return std::nullopt;
    const std::uint64_t factorBytes = interiorFactorBytes();
    KeptInteriorFactor kept = keepInteriorFactor(unitIdentity, m_system->unitId(), *m_system, std::move(m_factor),
                                                 factorBytes, std::move(owner));
    m_factor = linalg::PartialFactor{};
    m_factorLoan.release();
    m_factorMemoryOwner.reset();
    m_isInteriorFactorReused = false;
    return kept;
}

void StaticCondensation::recomputeInteriorFactor(const linalg::DenseKernels& kernels, base::MemoryBudget& budget)
{
    recompute(kernels, budget, m_options);
}

void StaticCondensation::recomputeInteriorFactor(const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                                 const base::CancellationToken* cancellation)
{
    linalg::FactorOptions options = m_options;
    options.cancellation = cancellation;
    recompute(kernels, budget, options);
}

void StaticCondensation::recompute(const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                   const linalg::FactorOptions& options)
{
    if (m_hasFactorised && kernels.kind() != m_kernel)
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(m_system->unitId()), "): its factor was made with the ",
             linalg::kernelName(m_kernel), " kernels, the recomputation was given the ",
             linalg::kernelName(kernels.kind()), " kernels, which give other bits; recompute with the same family"}));
    // A new partial factor: its K* must be the reduced unit's, to the bit (the factorisation is deterministic for a
    // kernel family, whatever its number of threads). From the cache, another K* means an entry whose key missed a
    // change; after a release, a broken determinism.
    m_factor = linalg::PartialFactor::factorise(m_system->symbolic(), m_system->matrix(), kernels, options, budget);
    // A kept factor this one replaces is freed now: its loan and its budget's owner go with it.
    m_factorLoan.release();
    m_factorMemoryOwner.reset();
    m_isInteriorFactorReused = false;
    const bool isFirst = !m_hasFactorised;
    m_hasFactorised = true;
    m_kernel = kernels.kind();
    if (isSchurComplementTheReducedOne())
        return;
    if (m_isFromCache && isFirst)
        throw base::CacheError(base::concatenate(
            {"the cached condensation of unit ", std::to_string(m_system->unitId()),
             " differs from the one its equations give now (its K* is not the same to the bit): the cache entry does "
             "not describe the unit as it is; clear the cache (O_o_Solver.cache.clear()) and run again, and report "
             "the case (a cache key that misses a change)"}));
    throw base::InternalError(base::concatenate(
        {"the interior factor of unit ", std::to_string(m_system->unitId()),
         " computed again for its recovery gives a K* other than its condensation's: the factorisation is not "
         "deterministic (an internal error: report it with the model)"}));
}

void StaticCondensation::enableRecoveryWithoutFactor(std::int64_t regionValues, base::MemoryBudget& budget)
{
    if (m_factor.isInteriorFactorKept())
        throw std::logic_error(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(m_system->unitId()),
             "): its interior factor is kept; a recovery without it is for a unit whose factor was released"}));
    m_regionValues = std::max<std::int64_t>(regionValues, 1);
    m_regionBudget = &budget;
}

void StaticCondensation::disableRecoveryWithoutFactor() noexcept
{
    m_regionBudget = nullptr;
    m_regionValues = 0;
}

void StaticCondensation::recover(linalg::DenseMatrixView loadsToSolution, linalg::ConstDenseMatrixView boundaryValues,
                                 const linalg::DenseKernels& kernels, const base::CancellationToken* cancellation) const
{
    if (m_factor.isInteriorFactorKept() || m_regionBudget == nullptr)
    {
        m_factor.recover(loadsToSolution, boundaryValues, kernels, cancellation);
        return;
    }
    if (m_hasFactorised && kernels.kind() != m_kernel)
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(m_system->unitId()), "): its condensation was made with the ",
             linalg::kernelName(m_kernel), " kernels, its recovery was given the ", linalg::kernelName(kernels.kind()),
             " kernels, which give other bits; recover with the same family"}));
    linalg::FactorOptions options = m_options;
    options.cancellation = cancellation;
    const linalg::DenseMatrix computed =
        linalg::PartialFactor::recoverWithoutFactor(m_system->symbolic(), m_system->matrix(), loadsToSolution,
                                                    boundaryValues, kernels, options, *m_regionBudget, m_regionValues);
    ++m_regionRecoveryCount;
    // Its K*, made by the factorisation of its forward sweep: the reduced unit's to the bit (the recomputation's
    // check).
    const linalg::DenseMatrix& kept = m_reduced->stiffness();
    bool isSame = computed.rows() == kept.rows() && computed.columns() == kept.columns();
    for (std::size_t j = 0; j < computed.columns() && isSame; ++j)
        for (std::size_t i = 0; i < computed.rows() && isSame; ++i)
            isSame = std::bit_cast<std::uint64_t>(computed(i, j)) == std::bit_cast<std::uint64_t>(kept(i, j));
    if (isSame)
        return;
    if (m_isFromCache)
        throw base::CacheError(base::concatenate(
            {"the cached condensation of unit ", std::to_string(m_system->unitId()),
             " differs from the one its equations give now (its K* is not the same to the bit): the cache entry does "
             "not describe the unit as it is; clear the cache (O_o_Solver.cache.clear()) and run again, and report "
             "the case (a cache key that misses a change)"}));
    throw base::InternalError(base::concatenate(
        {"the interior of unit ", std::to_string(m_system->unitId()),
         " factorised again by regions for its recovery gives a K* other than its condensation's: the factorisation "
         "is not deterministic (an internal error: report it with the model)"}));
}

linalg::DenseMatrix StaticCondensation::condenseLoadsWithoutFactor(linalg::ConstDenseMatrixView loads,
                                                                   const linalg::DenseKernels& kernels,
                                                                   base::MemoryBudget& budget,
                                                                   const base::CancellationToken* cancellation) const
{
    const UnitSystem& unit = *m_system;
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto interior = static_cast<std::size_t>(unit.interiorDofCount());
    const auto boundary = static_cast<std::size_t>(unit.boundaryDofCount());
    if (loads.rows != equations)
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(unit.unitId()), "): loads of ", std::to_string(loads.rows),
             " rows for ", std::to_string(equations), " equations; give them on its equations (equationLoads)"}));
    if (m_hasFactorised && kernels.kind() != m_kernel)
        throw std::invalid_argument(base::concatenate(
            {"StaticCondensation (unit ", std::to_string(unit.unitId()), "): its condensation was made with the ",
             linalg::kernelName(m_kernel), " kernels, the condensation of its loads was given the ",
             linalg::kernelName(kernels.kind()), " kernels, which give other bits; condense with the same family"}));
    linalg::DenseMatrix swept(budget, equations, loads.columns, "unit loads swept without the factor");
    for (std::size_t j = 0; j < loads.columns; ++j)
        for (std::size_t i = 0; i < equations; ++i)
            swept(i, j) = loads(i, j);
    linalg::FactorOptions options = m_options;
    options.cancellation = cancellation;
    const linalg::PartialCondensation condensation = linalg::PartialFactor::condenseWithoutFactor(
        unit.symbolic(), unit.matrix(), swept.view(), kernels, options, budget);
    // Its K*: the reduced unit's to the bit (the factorisation is deterministic).
    const linalg::DenseMatrix& computed = condensation.schurComplement;
    const linalg::DenseMatrix& kept = m_reduced->stiffness();
    bool isSame = computed.rows() == kept.rows() && computed.columns() == kept.columns();
    for (std::size_t j = 0; j < computed.columns() && isSame; ++j)
        for (std::size_t i = 0; i < computed.rows() && isSame; ++i)
            isSame = std::bit_cast<std::uint64_t>(computed(i, j)) == std::bit_cast<std::uint64_t>(kept(i, j));
    if (!isSame && m_isFromCache)
        throw base::CacheError(base::concatenate(
            {"the cached condensation of unit ", std::to_string(unit.unitId()),
             " differs from the one its equations give now (its K* is not the same to the bit): the cache entry does "
             "not describe the unit as it is; clear the cache (O_o_Solver.cache.clear()) and run again, and report "
             "the case (a cache key that misses a change)"}));
    if (!isSame)
        throw base::InternalError(base::concatenate(
            {"the interior of unit ", std::to_string(unit.unitId()),
             " factorised again without its factor for the condensation of loads gives a K* other than its "
             "condensation's: the factorisation is not deterministic (an internal error: report it with the model)"}));
    linalg::DenseMatrix condensed(budget, boundary, loads.columns, "condensed loads");
    for (std::size_t j = 0; j < loads.columns; ++j)
        std::copy_n(&swept(interior, j), boundary, &condensed(0, j));
    return condensed;
}

linalg::DenseMatrix StaticCondensation::patternWork(const UnitPatterns& patterns, const linalg::DenseKernels& kernels,
                                                    base::MemoryBudget& budget) const
{
    if (m_isWithoutFactor)
    {
        // Made at the condensation, from the swept patterns and the pivots (the factor is gone).
        linalg::DenseMatrix copy(budget, m_patternWork.rows(), m_patternWork.columns(), "unit pattern work");
        for (std::size_t j = 0; j < copy.columns(); ++j)
            for (std::size_t i = 0; i < copy.rows(); ++i)
                copy(i, j) = m_patternWork(i, j);
        return copy;
    }
    const UnitSystem& unit = *m_system;
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto interior = static_cast<std::size_t>(unit.interiorDofCount());
    const std::size_t count = patterns.loads.columns;
    linalg::DenseMatrix work(budget, count, count, "unit pattern work");
    if (count == 0)
        return work;
    // y = L⁻¹·f on the interior rows (the forward sweep of the interior panels), then z = D⁻¹·y.
    linalg::DenseMatrix swept(budget, equations, count, "unit pattern sweep");
    unit.equationLoads(patterns.loads, patterns.enforced, swept.view(), budget);
    const linalg::FactorSolve solve(m_factor.interiorFactor(), kernels);
    solve.eliminateForward(swept.view());
    linalg::DenseMatrix scaled(budget, equations, count, "unit pattern sweep scaled");
    for (std::size_t j = 0; j < count; ++j)
        for (std::size_t i = 0; i < equations; ++i)
            scaled(i, j) = swept(i, j);
    solve.solveDiagonal(scaled.view());
    for (std::size_t b = 0; b < count; ++b)
        for (std::size_t a = b; a < count; ++a)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < interior; ++i)
                sum += swept(i, a) * scaled(i, b);
            work(a, b) = sum;
            work(b, a) = sum;
        }
    return work;
}

} // namespace oos::reduction
