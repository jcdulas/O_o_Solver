#include "oos/linalg/supernodal/PartialFactor.hpp"

#include "FactorSchedule.hpp"
#include "ForwardStream.hpp"
#include "RegionSweep.hpp"
#include "SolvePlan.hpp"
#include "SolveSweeps.hpp"
#include "StreamedFactorisation.hpp"
#include "WorkspacePool.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace oos::linalg
{
namespace
{

void requirePositiveDefinite(const FactorOptions& options)
{
    if (options.pivoting != PivotingMode::PositiveDefinite)
        throw std::invalid_argument(
            "PartialFactor: the interior of a unit is factorised in positive definite mode (a delayed pivot would "
            "move an interior equation into the boundary); pass PivotingMode::PositiveDefinite");
}

void requireRowCount(const char* operation, std::size_t rows, std::size_t expected, const char* what)
{
    if (rows != expected)
        throw std::invalid_argument(
            base::concatenate({"PartialFactor::", operation, ": ", what, " have ", std::to_string(rows),
                               " rows, expected ", std::to_string(expected)}));
}

// FNV-1a (Fowler, Noll, Vo) over the bits of the offsets, the row indices and the values: a recomputation from another
// matrix is refused instead of silently giving another factor. Not cryptographic; one pass, cheap beside a
// factorisation.
std::uint64_t checksumOf(const SymmetricCsc& matrix) noexcept
{
    constexpr std::uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;
    std::uint64_t hash = FNV_OFFSET_BASIS;
    const auto mix = [&hash](std::uint64_t word)
    {
        hash ^= word;
        hash *= FNV_PRIME;
    };
    mix(static_cast<std::uint64_t>(matrix.dimension()));
    for (const std::int64_t offset : matrix.columnOffsets())
        mix(static_cast<std::uint64_t>(offset));
    for (const std::int32_t row : matrix.rowIndices())
        mix(static_cast<std::uint32_t>(row));
    for (const double value : matrix.values())
        mix(std::bit_cast<std::uint64_t>(value));
    return hash;
}

} // namespace

PartialFactor PartialFactor::factorise(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                       const DenseKernels& kernels, const FactorOptions& options,
                                       base::MemoryBudget& budget)
{
    requirePositiveDefinite(options);
    PartialFactor result;
    result.m_dimension = symbolic.dofCount();
    result.m_trailingDofCount = symbolic.trailingDofCount();
    const auto trailing = static_cast<std::size_t>(result.m_trailingDofCount);
    // K* is reserved first: a unit whose Schur complement does not fit is refused before any work.
    result.m_schurComplement = DenseMatrix(budget, trailing, trailing, "Schur complement K*");
    result.m_interior = SupernodalFactor::factoriseSupernodes(symbolic, matrix, kernels, options, true,
                                                              &result.m_schurComplement, budget);
    result.m_interiorSupernodeCount = result.m_interior.supernodeCount();
    result.m_isInteriorFactorKept = true;
    result.m_kernel = kernels.kind();
    result.m_pivotPolicy = options.pivotPolicy;
    result.m_matrixChecksum = checksumOf(matrix);
    return result;
}

bool PartialFactor::isFactorisationOf(const SymbolicFactor& symbolic, const SymmetricCsc& matrix, KernelKind kernel,
                                      const PivotPolicy& policy) const noexcept
{
    return m_isInteriorFactorKept && m_dimension == symbolic.dofCount() && m_dimension == matrix.dimension() &&
           m_trailingDofCount == symbolic.trailingDofCount() && m_kernel == kernel &&
           m_pivotPolicy.maxRatio == policy.maxRatio && m_pivotPolicy.bailout == policy.bailout &&
           m_matrixChecksum == checksumOf(matrix);
}

void PartialFactor::requireKept(const char* operation) const
{
    if (!m_isInteriorFactorKept)
        throw std::logic_error(base::concatenate(
            {"PartialFactor::", operation,
             ": the interior factor was released; call recomputeInteriorFactor() first (the same bits as the first "
             "factorisation)"}));
}

const SupernodalFactor& PartialFactor::interiorFactor() const
{
    requireKept("interiorFactor");
    return m_interior;
}

DenseMatrix PartialFactor::condenseLoads(ConstDenseMatrixView loads, const DenseKernels& kernels,
                                         base::MemoryBudget& budget, const base::CancellationToken* cancellation) const
{
    requireKept("condenseLoads");
    requireRowCount("condenseLoads", loads.rows, static_cast<std::size_t>(m_dimension), "the loads");
    const std::size_t columns = loads.columns;
    DenseMatrix work(budget, loads.rows, columns, "condensed loads: work");
    for (std::size_t j = 0; j < columns; ++j)
        std::copy_n(loads.pointer(0, j), loads.rows, &work(0, j));
    FactorSolve(m_interior, kernels).eliminateForward(work.view(), cancellation);
    const auto trailing = static_cast<std::size_t>(m_trailingDofCount);
    const auto interior = static_cast<std::size_t>(interiorDofCount());
    DenseMatrix condensed(budget, trailing, columns, "condensed loads");
    for (std::size_t j = 0; j < columns; ++j)
        std::copy_n(&work(interior, j), trailing, &condensed(0, j));
    return condensed;
}

void PartialFactor::recover(DenseMatrixView loadsToSolution, ConstDenseMatrixView trailingValues,
                            const DenseKernels& kernels, const base::CancellationToken* cancellation) const
{
    requireKept("recover");
    requireRowCount("recover", loadsToSolution.rows, static_cast<std::size_t>(m_dimension), "the loads");
    requireRowCount("recover", trailingValues.rows, static_cast<std::size_t>(m_trailingDofCount),
                    "the trailing values");
    if (trailingValues.columns != loadsToSolution.columns)
        throw std::invalid_argument(base::concatenate(
            {"PartialFactor::recover: ", std::to_string(trailingValues.columns), " columns of trailing values for ",
             std::to_string(loadsToSolution.columns), " load columns; give one per load column"}));
    const FactorSolve solve(m_interior, kernels);
    // The forward sweep leaves f* in the trailing rows; they are replaced by u_b before the diagonal and backward
    // sweeps, which then give u_i = L⁻ᵀ·(D⁻¹·L⁻¹·f_i − L_biᵀ·u_b) = K_ii⁻¹·(f_i − K_ib·u_b).
    solve.eliminateForward(loadsToSolution, cancellation);
    const auto interior = static_cast<std::size_t>(interiorDofCount());
    for (std::size_t j = 0; j < loadsToSolution.columns; ++j)
        std::copy_n(trailingValues.pointer(0, j), trailingValues.rows, &loadsToSolution(interior, j));
    solve.solveDiagonal(loadsToSolution, cancellation);
    solve.substituteBackward(loadsToSolution, cancellation);
}

void PartialFactor::releaseInteriorFactor() noexcept
{
    m_interior = SupernodalFactor();
    m_isInteriorFactorKept = false;
}

void PartialFactor::recomputeInteriorFactor(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                            const DenseKernels& kernels, const FactorOptions& options,
                                            base::MemoryBudget& budget)
{
    requirePositiveDefinite(options);
    if (symbolic.dofCount() != m_dimension || symbolic.trailingDofCount() != m_trailingDofCount)
        throw std::invalid_argument(base::concatenate(
            {"PartialFactor::recomputeInteriorFactor: the symbolic factor has ", std::to_string(symbolic.dofCount()),
             " equations and ", std::to_string(symbolic.trailingDofCount()), " trailing ones, the partial factor ",
             std::to_string(m_dimension), " and ", std::to_string(m_trailingDofCount),
             "; pass the symbolic factor and the matrix of the first factorisation"}));
    if (kernels.kind() != m_kernel)
        throw std::invalid_argument(
            base::concatenate({"PartialFactor::recomputeInteriorFactor: the kernel family is ",
                               kernelName(kernels.kind()), ", the first factorisation used ", kernelName(m_kernel),
                               "; recompute with the same family (another one gives other bits)"}));
    if (options.pivotPolicy.maxRatio != m_pivotPolicy.maxRatio || options.pivotPolicy.bailout != m_pivotPolicy.bailout)
        throw std::invalid_argument(base::concatenate(
            {"PartialFactor::recomputeInteriorFactor: the pivot policy (ratio limit ",
             std::to_string(options.pivotPolicy.maxRatio), ", bail-out ", std::to_string(options.pivotPolicy.bailout),
             ") differs from the first factorisation's (", std::to_string(m_pivotPolicy.maxRatio), ", ",
             std::to_string(m_pivotPolicy.bailout), "); pass the options of the first factorisation"}));
    if (checksumOf(matrix) != m_matrixChecksum)
        throw std::invalid_argument(
            "PartialFactor::recomputeInteriorFactor: the matrix differs from the one first factorised (its checksum "
            "of structure and values changed); pass the same matrix, or factorise the unit again");
    // K* is not assembled again: it is kept.
    SupernodalFactor interior =
        SupernodalFactor::factoriseSupernodes(symbolic, matrix, kernels, options, true, nullptr, budget);
    if (interior.supernodeCount() != m_interiorSupernodeCount)
        throw std::invalid_argument(
            "PartialFactor::recomputeInteriorFactor: the symbolic factor has another supernode structure than the "
            "first factorisation; pass the same symbolic factor");
    m_interior = std::move(interior);
    m_isInteriorFactorKept = true;
}

PartialCondensation PartialFactor::condenseWithoutFactor(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                         DenseMatrixView loads, const DenseKernels& kernels,
                                                         const FactorOptions& options, base::MemoryBudget& budget)
{
    requirePositiveDefinite(options);
    requireRowCount("condenseWithoutFactor", loads.rows, static_cast<std::size_t>(symbolic.dofCount()), "the loads");
    PartialCondensation result;
    const auto trailing = static_cast<std::size_t>(symbolic.trailingDofCount());
    // K* is reserved first: a unit whose Schur complement does not fit is refused before any work (factorise).
    result.schurComplement = DenseMatrix(budget, trailing, trailing, "Schur complement K*");
    StreamedFactorisation streamed(symbolic, matrix, kernels, options, true, budget);
    {
        ForwardStream forward(streamed, loads, kernels, budget, options.cancellation);
        streamed.run(&forward, {}, trailing > 0 ? &result.schurComplement : nullptr);
    }
    const std::span<const double> pivots = streamed.pivots();
    result.pivots = base::BudgetedBuffer<double>(budget, pivots.size(), "interior pivots", base::FOR_OVERWRITE);
    std::copy(pivots.begin(), pivots.end(), result.pivots.data());
    result.largestPivotRatio = streamed.diagnostics().largestRatio();
    result.workspace = streamed.workspaceUsage();
    return result;
}

std::uint64_t WithoutFactorEstimate::bytes(std::size_t columnCount, std::int64_t keptPanelValues) const noexcept
{
    // The widest vector length of the kernels' rows: a tile's columns are padded to it.
    constexpr std::size_t WIDEST_ROW_LANES = 8;
    const std::uint64_t padded = (columnCount + WIDEST_ROW_LANES - 1) / WIDEST_ROW_LANES * WIDEST_ROW_LANES;
    return fixedBytes +
           static_cast<std::uint64_t>(runValues + std::max<std::int64_t>(keptPanelValues, 0)) * sizeof(double) +
           static_cast<std::uint64_t>(accumulatorRowCount) * padded * sizeof(double);
}

WithoutFactorEstimate PartialFactor::estimateWithoutFactor(const SymbolicFactor& symbolic, std::size_t threadCount)
{
    constexpr std::uint64_t VALUE_BYTES = sizeof(double);
    // Per supernode of the skeleton: two offsets, first column, eliminated count, row count, the panel pointer, the
    // status, zero pivot, unit end and subtree first, a panel's WorkBuffer in the table, a pending slot: rounded up.
    constexpr std::uint64_t PER_SUPERNODE_BYTES = 192;
    base::MemoryBudget scratch(base::MemoryBudget::UNLIMITED);
    const std::size_t threads = std::max<std::size_t>(threadCount, 1);
    const FactorSchedule schedule(symbolic, true, threads, PivotingMode::PositiveDefinite, scratch,
                                  FactorSchedule::TopOrder::Serial);
    const std::int32_t trailing = schedule.excludedTrailing();
    const std::int32_t count = trailing >= 0 ? symbolic.supernodeCount() - 1 : symbolic.supernodeCount();
    std::vector<std::uint8_t> isInSubtree(static_cast<std::size_t>(std::max(count, 0)), 0); // one per supernode
    WithoutFactorEstimate estimate;
    std::int64_t largestSubtreeValues = 0; // the panels of the largest subtree of the cut: a unit holds no more
    for (const FactorSubtree& subtree : schedule.subtrees())
    {
        std::int64_t values = 0;
        for (std::int32_t s = subtree.first; s <= subtree.last; ++s)
        {
            isInSubtree[static_cast<std::size_t>(s)] = 1;
            const std::int64_t c = symbolic.columnCount(s);
            values += WorkspacePool::classSize(c * (c + symbolic.rowCount(s)));
        }
        largestSubtreeValues = std::max(largestSubtreeValues, values);
        estimate.accumulatorRowCount += symbolic.rowCount(subtree.last);
    }
    if (schedule.subtrees().size() < 2)
        estimate.accumulatorRowCount = 0; // the sweeps in supernode order: no accumulator
    std::int64_t rows = 0;
    std::int64_t largestSubtreePanel = 0;
    std::int64_t largestTopPanel = 0;
    std::int64_t allValues = 0;
    for (std::int32_t s = 0; s < count; ++s)
    {
        const std::int64_t c = symbolic.columnCount(s);
        const std::int64_t r = symbolic.rowCount(s);
        rows += c + r;
        const std::int64_t values = WorkspacePool::classSize(c * (c + r));
        allValues += values;
        estimate.largestPanelValues = std::max(estimate.largestPanelValues, values);
        if (isInSubtree[static_cast<std::size_t>(s)] != 0)
            largestSubtreePanel = std::max(largestSubtreePanel, values);
        else
            largestTopPanel = std::max(largestTopPanel, values);
    }
    // A unit of the solve plan inside a subtree: one supernode, or supernodes whose rows number at most UNIT_ROWS
    // (their panels then at most UNIT_ROWS² values, each rounded up to its block's size class: a sixteenth more at
    // most), and no more than its subtree's; a cut of fewer than two subtrees sweeps its units in order (one at a
    // time). Never more than every panel.
    constexpr std::int64_t UNIT_ROWS = SolvePlan::UNIT_ROWS;
    constexpr std::int64_t UNIT_VALUES =
        UNIT_ROWS * UNIT_ROWS + UNIT_ROWS * UNIT_ROWS / WorkspacePool::CLASSES_PER_OCTAVE;
    const bool isByTree = schedule.subtrees().size() >= 2;
    const std::int64_t subtreeUnit =
        std::min(std::max(largestSubtreePanel, UNIT_VALUES), isByTree ? largestSubtreeValues : allValues);
    const std::int64_t resident =
        std::min(std::max(static_cast<std::int64_t>(isByTree ? threads : 1) * subtreeUnit, largestTopPanel), allValues);
    estimate.runValues = schedule.workspaceValues() + resident;
    const auto dimension = static_cast<std::uint64_t>(symbolic.dofCount());
    const auto boundary = static_cast<std::uint64_t>(symbolic.trailingDofCount());
    estimate.fixedBytes = boundary * boundary * VALUE_BYTES + static_cast<std::uint64_t>(rows) * sizeof(std::int32_t) +
                          static_cast<std::uint64_t>(std::max(count, 0)) * PER_SUPERNODE_BYTES +
                          SolvePlan::estimateMemoryBytes(count, rows, symbolic.dofCount()) +
                          (dimension - boundary) * VALUE_BYTES;
    return estimate;
}

DenseMatrix PartialFactor::recoverWithoutFactor(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                DenseMatrixView loadsToSolution, ConstDenseMatrixView trailingValues,
                                                const DenseKernels& kernels, const FactorOptions& options,
                                                base::MemoryBudget& budget, std::int64_t regionValues)
{
    requirePositiveDefinite(options);
    const auto dimension = static_cast<std::size_t>(symbolic.dofCount());
    const auto trailing = static_cast<std::size_t>(symbolic.trailingDofCount());
    requireRowCount("recoverWithoutFactor", loadsToSolution.rows, dimension, "the loads");
    requireRowCount("recoverWithoutFactor", trailingValues.rows, trailing, "the trailing values");
    if (trailingValues.columns != loadsToSolution.columns)
        throw std::invalid_argument(
            base::concatenate({"PartialFactor::recoverWithoutFactor: ", std::to_string(trailingValues.columns),
                               " columns of trailing values for ", std::to_string(loadsToSolution.columns),
                               " load columns; give one per load column"}));
    DenseMatrix kstar(budget, trailing, trailing, "Schur complement K* (recovery without the factor)");
    StreamedFactorisation streamed(symbolic, matrix, kernels, options, true, budget);
    static_cast<void>(regions::sweep(streamed, loadsToSolution, trailingValues, kernels, options, budget, regionValues,
                                     trailing > 0 ? &kstar : nullptr));
    return kstar;
}

} // namespace oos::linalg
