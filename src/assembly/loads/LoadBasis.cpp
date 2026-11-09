#include "oos/assembly/loads/LoadBasis.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/FullRankCertificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

// The columns of one QR update shared by a task of the pool: enough work (a few hundred thousand multiply-adds for a
// tall matrix) to pay for the hand-out, few enough tasks for the six threads of a test.
constexpr std::size_t COLUMNS_PER_TASK = 64;
// Below this many columns × rows an update runs on the calling thread.
constexpr std::size_t PARALLEL_UPDATE_TERMS = 1U << 16;
// The cases one task of the reconstruction check takes: a few hundred thousand multiply-adds each for a basis of a few
// hundred vectors, and pieces enough to share 6 000 cases between the threads. It decides the pieces, never a result.
constexpr std::size_t CHECKED_CASES_PER_TASK = 16;
// A diagonal of R at most this fraction of the largest one is a dependent basis column (the least-squares solve
// skips it): the basis columns are chosen independent, so this only guards against a joined case that is a
// combination of the others to round-off.
constexpr double DEPENDENT_COLUMN_RATIO = 1.0e-14;
// The fraction of its last exact value (squared) below which a downdated remaining norm is computed again.
constexpr double NORM_RECOMPUTE_FRACTION = 1.0e-2;
// The density of the dense copy for the rank test compared with the budget left: the copy and the Householder vectors
// are one matrix; half of the budget left keeps room for the factorisation that follows.
constexpr double RANK_TEST_BUDGET_FRACTION = 0.5;

// A candidate basis: its vectors, the coefficients of every case, and the residuals of the check.
struct Candidate
{
    LoadBasisKind kind = LoadBasisKind::Empty;
    std::vector<BasisVector> vectors; // bounded by the cases of the family
    linalg::DenseMatrix coefficients; // vectors × cases
    std::int64_t joined = 0;
    double largestResidual = 0.0;
    base::BudgetedBuffer<double> residuals; // relative residual of each case
};

// The compact values of a basis vector.
struct ColumnRef
{
    std::span<const std::int64_t> rows;
    std::span<const double> values;
    std::span<const double> precisions;
    std::int64_t unitRow = -1; // a unit vector on this row (LoadedRow) when ≥ 0
};

ColumnRef columnOf(const BasisVector& vector, const CompactLoadMatrix& cases, const CompactLoadMatrix* loadSets)
{
    switch (vector.source)
    {
    case BasisVector::Source::LoadSet:
        return {loadSets->rowsOf(vector.index), loadSets->valuesOf(vector.index), loadSets->precisionsOf(vector.index),
                -1};
    case BasisVector::Source::LoadedRow:
        return {{}, {}, {}, vector.index};
    case BasisVector::Source::Case:
        break;
    }
    return {cases.rowsOf(vector.index), cases.valuesOf(vector.index), cases.precisionsOf(vector.index), -1};
}

// The reconstruction check of the cases [begin, end) (checkCases), with its own scratch.
void checkCaseRange(const CompactLoadMatrix& cases, std::span<const ColumnRef> columns,
                    const linalg::DenseMatrix& coefficients, bool usesWrittenPrecision, std::size_t begin,
                    std::size_t end, std::span<double> residuals, std::span<std::uint8_t> fails,
                    base::MemoryBudget& budget)
{
    const auto rowCount = static_cast<std::size_t>(cases.rowCount());
    base::BudgetedBuffer<double> work(budget, rowCount, "load basis residual");
    base::BudgetedBuffer<double> precision(budget, rowCount, "load basis residual precision");
    base::BudgetedBuffer<std::uint8_t> isTouched(budget, rowCount, "load basis residual rows");
    std::vector<std::int64_t> touched; // bounded by the nonzeros of one case and its basis vectors
    const auto touch = [&](std::int64_t row)
    {
        if (isTouched[static_cast<std::size_t>(row)] == 0)
        {
            isTouched[static_cast<std::size_t>(row)] = 1;
            touched.push_back(row);
        }
    };
    for (auto c = static_cast<std::int64_t>(begin); c < static_cast<std::int64_t>(end); ++c)
    {
        const std::span<const std::int64_t> rows = cases.rowsOf(c);
        const std::span<const double> values = cases.valuesOf(c);
        const std::span<const double> precisions = cases.precisionsOf(c);
        double largest = 0.0;
        for (std::size_t k = 0; k < rows.size(); ++k)
        {
            touch(rows[k]);
            work[static_cast<std::size_t>(rows[k])] = values[k];
            precision[static_cast<std::size_t>(rows[k])] = usesWrittenPrecision ? precisions[k] : 0.0;
            largest = std::max(largest, std::abs(values[k]));
        }
        for (std::size_t i = 0; i < columns.size(); ++i)
        {
            const double alpha = coefficients(i, static_cast<std::size_t>(c));
            if (alpha == 0.0)
                continue;
            if (columns[i].unitRow >= 0)
            {
                touch(columns[i].unitRow);
                work[static_cast<std::size_t>(columns[i].unitRow)] -= alpha;
                continue;
            }
            for (std::size_t k = 0; k < columns[i].rows.size(); ++k)
            {
                const auto row = static_cast<std::size_t>(columns[i].rows[k]);
                touch(columns[i].rows[k]);
                work[row] -= alpha * columns[i].values[k];
                if (usesWrittenPrecision)
                    precision[row] += std::abs(alpha) * columns[i].precisions[k];
            }
        }
        double residualSquared = 0.0;
        double normalisedSquared = 0.0; // Σ (r_i / bound_i)²
        const double slack = LOAD_BASIS_RESIDUAL_LIMIT * largest;
        for (const std::int64_t row : touched)
        {
            const double r = work[static_cast<std::size_t>(row)];
            residualSquared += r * r;
            if (usesWrittenPrecision)
            {
                const double bound = precision[static_cast<std::size_t>(row)] + slack;
                normalisedSquared += bound > 0.0 ? (r / bound) * (r / bound) : (r != 0.0 ? 1.0e300 : 0.0);
            }
            work[static_cast<std::size_t>(row)] = 0.0;
            precision[static_cast<std::size_t>(row)] = 0.0;
            isTouched[static_cast<std::size_t>(row)] = 0;
        }
        const std::size_t valueCount = touched.size();
        touched.clear();
        const double norm = cases.normOf(c);
        const double relative = norm > 0.0 ? std::sqrt(residualSquared) / norm : 0.0;
        residuals[static_cast<std::size_t>(c)] = relative;
        // Deck tolerance: the root mean square of the residuals over their bounds at most 1. The true combination of
        // the true shapes meets every bound, so its normalised residual has a norm of at most √m; the least-squares
        // combination has a smaller residual (in the norm of nearly uniform bounds), which may still pass a bound at
        // a few values: the mean over the values of the case is what "within its written precision" checks.
        const bool passes = usesWrittenPrecision
                                ? normalisedSquared <= static_cast<double>(std::max<std::size_t>(valueCount, 1))
                                : relative <= LOAD_BASIS_RESIDUAL_LIMIT;
        fails[static_cast<std::size_t>(c)] = passes ? 0 : 1;
    }
}

// The reconstruction check of every case: r = f_case − Σ α_i f_i in the compact space. Fills the
// relative residual ‖r‖/‖f_case‖ of each case and returns the cases that fail: with the round-off tolerance a relative
// residual above LOAD_BASIS_RESIDUAL_LIMIT; with the deck tolerance a case whose residuals exceed, in root mean square,
// the written precision of the reconstruction, δ_case + Σ |α_i| δ_i per value (the case and the basis vectors are
// written values, each rounded within its precision: the true combination of the true shapes meets this bound at every
// value), plus LOAD_BASIS_RESIDUAL_LIMIT times the largest value of the case (the values written as exact zeros and the
// round-off of the combination).
std::vector<std::int64_t> checkCases(const CompactLoadMatrix& cases, const CompactLoadMatrix* loadSets,
                                     std::span<const BasisVector> vectors, const linalg::DenseMatrix& coefficients,
                                     bool usesWrittenPrecision, std::span<double> residuals, base::ThreadPool* pool,
                                     base::MemoryBudget& budget)
{
    std::vector<ColumnRef> columns; // bounded by the basis vectors
    columns.reserve(vectors.size());
    for (const BasisVector& vector : vectors)
        columns.push_back(columnOf(vector, cases, loadSets));
    const auto caseCount = static_cast<std::size_t>(cases.columnCount());
    base::BudgetedBuffer<std::uint8_t> fails(budget, caseCount, "load basis failing cases");
    // PERFORMANCE: the cases are checked in parallel, by pieces of CHECKED_CASES_PER_TASK, each piece with its own
    // scratch; every case is computed alone as before (the same bits), the failing ones gathered in case order. The
    // written-precision tolerance checks every case once per step of its bisection (fewestWrittenPrecisionPivots).
    base::parallelFor(pool, caseCount, CHECKED_CASES_PER_TASK,
                      [&](std::size_t begin, std::size_t end)
                      {
                          checkCaseRange(cases, columns, coefficients, usesWrittenPrecision, begin, end, residuals,
                                         fails.span(), budget);
                      });
    std::vector<std::int64_t> failing; // bounded by the cases
    for (std::size_t c = 0; c < caseCount; ++c)
        if (fails[c] != 0)
            failing.push_back(static_cast<std::int64_t>(c));
    return failing;
}

// Householder reflector of x (Golub and Van Loan, Algorithm 5.1.1): v with v[0] = 1 and β such that
// (I − β v vᵀ) x = ∓‖x‖ e₁. Overwrites x[1:] with v[1:] and x[0] with the new diagonal; returns β.
double makeReflector(double* x, std::size_t length)
{
    double tail = 0.0;
    for (std::size_t i = 1; i < length; ++i)
        tail += x[i] * x[i];
    if (tail == 0.0)
        return 0.0;
    const double alpha = x[0];
    const double norm = std::sqrt(alpha * alpha + tail);
    const double diagonal = alpha <= 0.0 ? norm : -norm;
    const double v0 = alpha - diagonal;
    for (std::size_t i = 1; i < length; ++i)
        x[i] /= v0;
    x[0] = diagonal;
    return -v0 / diagonal; // β = 2 v0² / (v0² + tail) written as (diagonal − alpha) / diagonal
}

// y ← (I − β v vᵀ) y for the column y of length `length`, v[0] = 1 implied and v[1:] in `v`.
void applyReflector(const double* v, double beta, double* y, std::size_t length)
{
    double dot = y[0];
    for (std::size_t i = 1; i < length; ++i)
        dot += v[i] * y[i];
    const double scale = beta * dot;
    y[0] -= scale;
    for (std::size_t i = 1; i < length; ++i)
        y[i] -= scale * v[i];
}

// Runs `update(column)` for the columns [first, last), shared between the threads of the pool by blocks of columns
// when the work is large enough: each column is updated by one thread, so the result does not depend on them.
template <typename Update>
void forColumns(std::size_t first, std::size_t last, std::size_t rows, base::ThreadPool* pool, Update&& update)
{
    const std::size_t count = last > first ? last - first : 0;
    if (pool == nullptr || count * rows < PARALLEL_UPDATE_TERMS || count <= COLUMNS_PER_TASK)
    {
        for (std::size_t j = first; j < last; ++j)
            update(j);
        return;
    }
    const std::size_t tasks = (count + COLUMNS_PER_TASK - 1) / COLUMNS_PER_TASK;
    pool->runBatch(tasks,
                   [&](std::size_t task)
                   {
                       const std::size_t begin = first + task * COLUMNS_PER_TASK;
                       const std::size_t end = std::min(last, begin + COLUMNS_PER_TASK);
                       for (std::size_t j = begin; j < end; ++j)
                           update(j);
                   });
}

// The least-squares coefficients of every case on the basis vectors (Householder QR of the dense basis columns,
// α = R⁻¹·(Qᵀ f)[0:k], Golub and Van Loan §5.3.3); a dependent basis column gets zero coefficients.
linalg::DenseMatrix leastSquares(const CompactLoadMatrix& cases, const CompactLoadMatrix* loadSets,
                                 std::span<const BasisVector> vectors, base::ThreadPool* pool,
                                 base::MemoryBudget& budget)
{
    const auto rows = static_cast<std::size_t>(cases.rowCount());
    const std::size_t k = vectors.size();
    const auto caseCount = static_cast<std::size_t>(cases.columnCount());
    linalg::DenseMatrix coefficients(budget, k, caseCount, "load basis coefficients");
    if (k == 0)
        return coefficients;
    linalg::DenseMatrix basis(budget, rows, k, "load basis columns");
    for (std::size_t i = 0; i < k; ++i)
    {
        const ColumnRef column = columnOf(vectors[i], cases, loadSets);
        if (column.unitRow >= 0)
            basis(static_cast<std::size_t>(column.unitRow), i) = 1.0;
        for (std::size_t e = 0; e < column.rows.size(); ++e)
            basis(static_cast<std::size_t>(column.rows[e]), i) = column.values[e];
    }
    base::BudgetedBuffer<double> betas(budget, k, "load basis reflectors");
    const std::size_t steps = std::min(rows, k);
    for (std::size_t p = 0; p < steps; ++p)
    {
        double* x = &basis(p, p);
        betas[p] = makeReflector(x, rows - p);
        forColumns(p + 1, k, rows - p, pool,
                   [&](std::size_t j) { applyReflector(x, betas[p], &basis(p, j), rows - p); });
    }
    double largestDiagonal = 0.0;
    for (std::size_t p = 0; p < steps; ++p)
        largestDiagonal = std::max(largestDiagonal, std::abs(basis(p, p)));
    base::BudgetedBuffer<std::uint8_t> isIndependent(budget, steps, "load basis independent columns");
    for (std::size_t p = 0; p < steps; ++p)
        isIndependent[p] = std::abs(basis(p, p)) > DEPENDENT_COLUMN_RATIO * largestDiagonal ? 1 : 0;

    forColumns(0, caseCount, rows * k, pool,
               [&](std::size_t c)
               {
                   std::vector<double> y(rows, 0.0); // one case at a time: bounded by the compact rows
                   const std::span<const std::int64_t> caseRows = cases.rowsOf(static_cast<std::int64_t>(c));
                   const std::span<const double> caseValues = cases.valuesOf(static_cast<std::int64_t>(c));
                   if (caseRows.empty())
                       return;
                   for (std::size_t e = 0; e < caseRows.size(); ++e)
                       y[static_cast<std::size_t>(caseRows[e])] = caseValues[e];
                   for (std::size_t p = 0; p < steps; ++p)
                       applyReflector(&basis(p, p), betas[p], y.data() + p, rows - p);
                   // Back substitution R·α = y[0:k], the dependent columns left at zero.
                   for (std::size_t q = steps; q-- > 0;)
                   {
                       if (isIndependent[q] == 0)
                           continue;
                       double sum = y[q];
                       for (std::size_t j = q + 1; j < steps; ++j)
                           sum -= basis(q, j) * coefficients(j, c);
                       coefficients(q, c) = sum / basis(q, q);
                   }
               });
    return coefficients;
}

// The coefficients and the check of a basis given by its first vectors, the failing cases joining it until every case
// passes.
Candidate completeByCheck(LoadBasisKind kind, std::vector<BasisVector> vectors, const LoadBasis::Input& input,
                          bool usesWrittenPrecision, std::optional<linalg::DenseMatrix> givenCoefficients,
                          base::MemoryBudget& budget)
{
    const CompactLoadMatrix& cases = *input.cases;
    Candidate candidate;
    candidate.kind = kind;
    candidate.residuals =
        base::BudgetedBuffer<double>(budget, static_cast<std::size_t>(cases.columnCount()), "load basis residuals");
    linalg::DenseMatrix coefficients = givenCoefficients
                                           ? std::move(*givenCoefficients)
                                           : leastSquares(cases, input.loadSets, vectors, input.pool, budget);
    const std::size_t initial = vectors.size();
    for (;;)
    {
        const std::vector<std::int64_t> failing =
            checkCases(cases, input.loadSets, vectors, coefficients, usesWrittenPrecision, candidate.residuals.span(),
                       input.pool, budget);
        if (failing.empty())
            break;
        for (const std::int64_t c : failing)
            vectors.push_back({BasisVector::Source::Case, c});
        // The load sets' coefficients are exact factors: the joined cases take the identity, the others keep theirs.
        if (kind == LoadBasisKind::LoadSets)
        {
            linalg::DenseMatrix extended(budget, vectors.size(), static_cast<std::size_t>(cases.columnCount()),
                                         "load basis coefficients");
            for (std::size_t c = 0; c < extended.columns(); ++c)
                for (std::size_t i = 0; i < coefficients.rows(); ++i)
                    extended(i, c) = coefficients(i, c);
            for (std::size_t f = 0; f < failing.size(); ++f)
            {
                const auto c = static_cast<std::size_t>(failing[f]);
                for (std::size_t i = 0; i < coefficients.rows(); ++i)
                    extended(i, c) = 0.0;
                extended(coefficients.rows() + f, c) = 1.0;
            }
            coefficients = std::move(extended);
            continue;
        }
        coefficients = leastSquares(cases, input.loadSets, vectors, input.pool, budget);
    }
    candidate.joined = static_cast<std::int64_t>(vectors.size() - initial);
    for (std::size_t c = 0; c < candidate.residuals.size(); ++c)
        candidate.largestResidual = std::max(candidate.largestResidual, candidate.residuals[c]);
    candidate.vectors = std::move(vectors);
    candidate.coefficients = std::move(coefficients);
    return candidate;
}

// Whether the full-rank certificate (RANK_CERTIFICATE_MARGIN) proves that the pivoted QR of the cases `columns` takes
// them all, the scaled matrix having a smallest singular value above the bound; false when it cannot (fewer rows than
// cases, a case without reference, a dependent case met by the unpivoted QR).
// A full-rank load matrix makes the pivoted QR visit every column and repeatedly read the trailing matrix. The
// certificate uses the dense kernels' matrix products to avoid that memory-bound pass when its bound proves full rank.
bool provesNoRankGain(const CompactLoadMatrix& cases, std::span<const std::int64_t> columns, bool usesWrittenPrecision,
                      const LoadBasis::Input& input, base::MemoryBudget& budget)
{
    const auto rows = static_cast<std::size_t>(cases.rowCount());
    const std::size_t n = columns.size();
    if (n < RANK_CERTIFICATE_MIN_CASES || rows < n)
        return false;
    linalg::DenseMatrix scaled(budget, rows, n, "load basis rank certificate");
    double frobeniusSquared = 0.0;
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::span<const std::int64_t> caseRows = cases.rowsOf(columns[j]);
        const std::span<const double> values = cases.valuesOf(columns[j]);
        const std::span<const double> precisions = cases.precisionsOf(columns[j]);
        double norm = 0.0;
        double precisionNorm = 0.0;
        for (std::size_t e = 0; e < caseRows.size(); ++e)
        {
            norm += values[e] * values[e];
            precisionNorm += precisions[e] * precisions[e];
        }
        const double reference = usesWrittenPrecision ? std::sqrt(precisionNorm) : std::sqrt(norm);
        if (!(reference > 0.0))
            return false;
        for (std::size_t e = 0; e < caseRows.size(); ++e)
            scaled(static_cast<std::size_t>(caseRows[e]), j) = values[e] / reference;
        frobeniusSquared += norm / (reference * reference);
    }
    const double threshold = usesWrittenPrecision ? 1.0 : LOAD_RANK_TOLERANCE;
    const auto columnCount = static_cast<double>(n);
    const double bound =
        RANK_CERTIFICATE_MARGIN * threshold * std::sqrt(columnCount) +
        RANK_CERTIFICATE_ROUNDING * columnCount * std::numeric_limits<double>::epsilon() * std::sqrt(frobeniusSquared);
    linalg::DenseKernelOptions options;
    options.pool = input.pool;
    const linalg::DenseKernels kernels(input.kernel, options);
    return linalg::FullRankCertificate::prove(scaled.view(), bound, kernels, budget).isProved;
}

// The pivots of the column-pivoted Householder QR of the dense compact matrix (the nonzero cases), scaled pivoting:
// the next pivot is the column with the largest remaining norm relative to its reference (its norm, or the norm of its
// precisions with the deck tolerance); the elimination stops when that ratio is at most `threshold`, or returns
// nothing once `limit` pivots are taken (the rank could not be the smallest basis).
std::optional<std::vector<std::int64_t>> rankPivots(const CompactLoadMatrix& cases,
                                                    std::span<const std::int64_t> columns, bool usesWrittenPrecision,
                                                    std::size_t limit, base::ThreadPool* pool,
                                                    base::MemoryBudget& budget)
{
    const auto rows = static_cast<std::size_t>(cases.rowCount());
    const std::size_t n = columns.size();
    linalg::DenseMatrix a(budget, rows, n, "load basis rank test");
    base::BudgetedBuffer<double> reference(budget, n, "load basis rank references");
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::span<const std::int64_t> caseRows = cases.rowsOf(columns[j]);
        const std::span<const double> values = cases.valuesOf(columns[j]);
        const std::span<const double> precisions = cases.precisionsOf(columns[j]);
        double norm = 0.0;
        double precisionNorm = 0.0;
        for (std::size_t e = 0; e < caseRows.size(); ++e)
        {
            a(static_cast<std::size_t>(caseRows[e]), j) = values[e];
            norm += values[e] * values[e];
            precisionNorm += precisions[e] * precisions[e];
        }
        reference[j] = usesWrittenPrecision ? std::sqrt(precisionNorm) : std::sqrt(norm);
    }
    const double threshold = usesWrittenPrecision ? 1.0 : LOAD_RANK_TOLERANCE;
    base::BudgetedBuffer<std::size_t> order(budget, n, "load basis rank order"); // the case column at position p
    std::iota(order.data(), order.data() + n, std::size_t{0});
    base::BudgetedBuffer<double> remaining(budget, n, "load basis remaining norms");
    std::vector<std::int64_t> pivots; // bounded by the rank
    const std::size_t steps = std::min(rows, n);
    // The remaining norms: computed exactly once, then downdated after each step (‖x‖² minus the square of the entry
    // the reflector fixed) and computed again when the downdate has cancelled most of the norm, where it loses its
    // digits (LAPACK xLAQP2 recomputes below a fraction of the last exact value; here a tenth of the norm, so that a
    // norm near LOAD_RANK_TOLERANCE, twelve orders below its start, has always been computed again on its way down).
    // The last exactly computed remaining norm of each position.
    base::BudgetedBuffer<double> exact(budget, n, "load basis exact norms");
    const auto recompute = [&](std::size_t j, std::size_t from)
    {
        double sum = 0.0;
        for (std::size_t i = from; i < rows; ++i)
            sum += a(i, j) * a(i, j);
        remaining[j] = std::sqrt(sum);
        exact[j] = remaining[j];
    };
    forColumns(0, n, rows, pool, [&](std::size_t j) { recompute(j, 0); });
    for (std::size_t p = 0; p < steps; ++p)
    {
        std::size_t best = n;
        double bestRatio = threshold;
        for (std::size_t j = p; j < n; ++j)
        {
            const double ratio = reference[order[j]] > 0.0 ? remaining[j] / reference[order[j]] : 0.0;
            if (ratio > bestRatio)
            {
                bestRatio = ratio;
                best = j;
            }
        }
        if (best == n)
            break;
        if (pivots.size() >= limit)
            return std::nullopt;
        if (best != p)
        {
            for (std::size_t i = 0; i < rows; ++i)
                std::swap(a(i, p), a(i, best));
            std::swap(order[p], order[best]);
            std::swap(remaining[p], remaining[best]);
            std::swap(exact[p], exact[best]);
        }
        pivots.push_back(columns[order[p]]);
        double* x = &a(p, p);
        const double beta = makeReflector(x, rows - p);
        forColumns(p + 1, n, rows - p, pool,
                   [&](std::size_t j)
                   {
                       applyReflector(x, beta, &a(p, j), rows - p);
                       const double fixed = a(p, j);
                       const double downdated = remaining[j] * remaining[j] - fixed * fixed;
                       if (downdated <= NORM_RECOMPUTE_FRACTION * exact[j] * exact[j])
                           recompute(j, p + 1);
                       else
                           remaining[j] = std::sqrt(downdated);
                   });
    }
    return pivots;
}

// With the deck tolerance, the fewest leading pivots that reconstruct every case within its written precision. The
// QR stops when the remaining norms fall below the precision of each column, but a remaining norm also holds the
// rounding of the pivots it was projected on (Σ α_i e_i), so it stops a few pivots late; the reconstruction check,
// which accounts for that rounding (checkCases), decides: a bisection on the number of leading pivots (more pivots
// never reconstruct worse in practice; the last count is checked again by completeByCheck, which joins any failing
// case).
std::size_t fewestWrittenPrecisionPivots(const std::vector<std::int64_t>& pivots, const LoadBasis::Input& input,
                                         base::MemoryBudget& budget)
{
    const CompactLoadMatrix& cases = *input.cases;
    base::BudgetedBuffer<double> residuals(budget, static_cast<std::size_t>(cases.columnCount()),
                                           "load basis residuals");
    const auto passes = [&](std::size_t count)
    {
        std::vector<BasisVector> vectors;
        for (std::size_t i = 0; i < count; ++i)
            vectors.push_back({BasisVector::Source::Case, pivots[i]});
        const linalg::DenseMatrix coefficients = leastSquares(cases, input.loadSets, vectors, input.pool, budget);
        return checkCases(cases, input.loadSets, vectors, coefficients, true, residuals.span(), input.pool, budget)
            .empty();
    };
    std::size_t low = 1;
    std::size_t high = pivots.size();
    while (low < high)
    {
        const std::size_t middle = low + (high - low) / 2;
        if (passes(middle))
            high = middle;
        else
            low = middle + 1;
    }
    return high;
}

LoadBasisReport reportOf(const LoadBasis::Input& input, const Candidate& chosen, std::int64_t caseCount)
{
    LoadBasisReport report;
    report.family = input.family;
    report.kind = chosen.kind;
    report.tolerance = input.tolerance;
    report.size = static_cast<std::int64_t>(chosen.vectors.size());
    report.caseCount = caseCount;
    report.joinedCaseCount = chosen.joined;
    report.largestResidual = chosen.largestResidual;
    return report;
}

std::size_t bytesOfDense(std::size_t rows, std::size_t columns)
{
    return linalg::DenseMatrix::alignedLeadingDimension(rows) * columns * sizeof(double);
}

} // namespace

const char* nameOf(LoadBasisKind kind) noexcept
{
    switch (kind)
    {
    case LoadBasisKind::Empty:
        return "empty";
    case LoadBasisKind::LoadSets:
        return "load sets";
    case LoadBasisKind::LoadedDofs:
        return "loaded DOFs";
    case LoadBasisKind::Rank:
        return "rank";
    case LoadBasisKind::Cases:
        return "cases";
    }
    return "";
}

LoadBasis LoadBasis::chooseFromSets(const Input& input, base::MemoryBudget& budget)
{
    if (input.loadSets == nullptr || input.setCoefficients == nullptr)
        throw std::invalid_argument(
            "LoadBasis::choose: cases given as combinations of the load sets need the load sets and their factors");
    const CompactLoadMatrix& sets = *input.loadSets;
    const linalg::DenseMatrix& factors = *input.setCoefficients;
    const std::size_t caseCount = input.caseCount;
    if (sets.filledColumnCount() != sets.columnCount() ||
        factors.rows() != static_cast<std::size_t>(sets.columnCount()) || factors.columns() != caseCount)
        throw std::invalid_argument(
            base::concatenate({"LoadBasis::choose: expected factors of (", std::to_string(sets.columnCount()),
                               " load sets × ", std::to_string(caseCount), " cases), found ",
                               std::to_string(factors.rows()), " × ", std::to_string(factors.columns())}));
    // The sets some case takes, with a value on the rows; the cases taking one of them.
    std::vector<std::int64_t> used; // bounded by the load sets
    base::BudgetedBuffer<std::uint8_t> isLoaded(budget, caseCount, "load basis loaded cases");
    for (std::int64_t s = 0; s < sets.columnCount(); ++s)
    {
        if (!(sets.normOf(s) > 0.0))
            continue;
        bool isTaken = false;
        for (std::size_t c = 0; c < caseCount; ++c)
            if (factors(static_cast<std::size_t>(s), c) != 0.0)
            {
                isTaken = true;
                isLoaded[c] = 1;
            }
        if (isTaken)
            used.push_back(s);
    }
    const auto loadedCount =
        static_cast<std::int64_t>(std::count(isLoaded.data(), isLoaded.data() + caseCount, std::uint8_t{1}));
    LoadBasis basis;
    basis.m_report.family = input.family;
    basis.m_report.tolerance = input.tolerance;
    if (used.empty())
    {
        basis.m_vectors = base::BudgetedBuffer<BasisVector>(budget, 0, "load basis vectors");
        basis.m_coefficients = linalg::DenseMatrix(budget, 0, caseCount, "load basis coefficients");
        basis.m_report.explanation = "no case of the family has a load";
        return basis;
    }
    // The loaded DOFs, when the rows are DOFs: the smaller basis is taken (the sets on a tie).
    std::vector<std::int64_t> loadedRows; // bounded by the rows
    if (input.offersLoadedDofs)
    {
        base::BudgetedBuffer<std::uint8_t> isRowLoaded(budget, static_cast<std::size_t>(sets.rowCount()),
                                                       "load basis loaded rows");
        for (const std::int64_t s : used)
            for (const std::int64_t row : sets.rowsOf(s))
                isRowLoaded[static_cast<std::size_t>(row)] = 1;
        for (std::size_t row = 0; row < isRowLoaded.size(); ++row)
            if (isRowLoaded[row] != 0)
                loadedRows.push_back(static_cast<std::int64_t>(row));
    }
    const bool takesRows = input.offersLoadedDofs && loadedRows.size() < used.size();
    const std::size_t vectorCount = takesRows ? loadedRows.size() : used.size();
    basis.m_vectors = base::BudgetedBuffer<BasisVector>(budget, vectorCount, "load basis vectors");
    basis.m_coefficients = linalg::DenseMatrix(budget, vectorCount, caseCount, "load basis coefficients");
    if (takesRows)
    {
        // A case's value on a row: Σ over the sets in their order of factor × value.
        base::BudgetedBuffer<std::int64_t> position(budget, static_cast<std::size_t>(sets.rowCount()),
                                                    "load basis positions of rows");
        for (std::size_t i = 0; i < loadedRows.size(); ++i)
        {
            position[static_cast<std::size_t>(loadedRows[i])] = static_cast<std::int64_t>(i);
            basis.m_vectors[i] = {BasisVector::Source::LoadedRow, loadedRows[i]};
        }
        for (std::size_t c = 0; c < caseCount; ++c)
            for (const std::int64_t s : used)
            {
                const double factor = factors(static_cast<std::size_t>(s), c);
                if (factor == 0.0)
                    continue;
                const std::span<const std::int64_t> rows = sets.rowsOf(s);
                const std::span<const double> values = sets.valuesOf(s);
                for (std::size_t e = 0; e < rows.size(); ++e)
                    basis.m_coefficients(static_cast<std::size_t>(position[static_cast<std::size_t>(rows[e])]), c) +=
                        factor * values[e];
            }
    }
    else
        for (std::size_t i = 0; i < used.size(); ++i)
        {
            basis.m_vectors[i] = {BasisVector::Source::LoadSet, used[i]};
            for (std::size_t c = 0; c < caseCount; ++c)
                basis.m_coefficients(i, c) = factors(static_cast<std::size_t>(used[i]), c);
        }
    basis.m_report.kind = takesRows ? LoadBasisKind::LoadedDofs : LoadBasisKind::LoadSets;
    basis.m_report.size = static_cast<std::int64_t>(vectorCount);
    basis.m_report.caseCount = loadedCount;
    basis.m_report.loadSetCount = static_cast<std::int64_t>(used.size());
    basis.m_report.loadedDofCount = input.offersLoadedDofs ? static_cast<std::int64_t>(loadedRows.size()) : -1;
    std::string explanation =
        base::concatenate({nameOf(basis.m_report.kind), " basis of ", std::to_string(vectorCount), " vectors for ",
                           std::to_string(loadedCount), " cases; load sets as written: ", std::to_string(used.size())});
    if (input.offersLoadedDofs)
        explanation.append("; loaded DOFs: ").append(std::to_string(loadedRows.size()));
    explanation.append(
        "; the cases combine the load sets exactly (LoadCombination cards bringing whole sets): no rank test or "
        "reconstruction "
        "check");
    basis.m_report.explanation = std::move(explanation);
    return basis;
}

LoadBasis LoadBasis::choose(const Input& input, base::MemoryBudget& budget)
{
    if (input.isCombinationOfSets)
        return chooseFromSets(input, budget);
    if (input.cases == nullptr)
        throw std::invalid_argument("LoadBasis::choose: expected the matrix of the cases, found none");
    const CompactLoadMatrix& cases = *input.cases;
    const auto caseCount = static_cast<std::size_t>(cases.columnCount());
    if (cases.filledColumnCount() != cases.columnCount())
        throw std::invalid_argument("LoadBasis::choose: the matrix of the cases is not filled");
    if ((input.loadSets == nullptr) != (input.setCoefficients == nullptr))
        throw std::invalid_argument("LoadBasis::choose: expected the load sets with their coefficients, or neither");
    if (input.loadSets != nullptr &&
        (input.loadSets->rowCount() != cases.rowCount() ||
         input.setCoefficients->rows() != static_cast<std::size_t>(input.loadSets->columnCount()) ||
         input.setCoefficients->columns() != caseCount))
        throw std::invalid_argument(base::concatenate(
            {"LoadBasis::choose: expected load sets of ", std::to_string(cases.rowCount()),
             " rows and coefficients of (load sets × ", std::to_string(caseCount), " cases), found ",
             std::to_string(input.loadSets->rowCount()), " rows and ", std::to_string(input.setCoefficients->rows()),
             " × ", std::to_string(input.setCoefficients->columns())}));

    std::size_t loadedCount = 0;
    for (std::int64_t c = 0; c < cases.columnCount(); ++c)
        loadedCount += cases.normOf(c) > 0.0 ? 1 : 0;
    base::BudgetedBuffer<std::int64_t> loadedCases(budget, loadedCount, "load basis loaded cases");
    for (std::int64_t c = 0, next = 0; c < cases.columnCount(); ++c)
        if (cases.normOf(c) > 0.0)
            loadedCases[static_cast<std::size_t>(next++)] = c;
    const std::span<const std::int64_t> loaded = loadedCases.span(); // the cases with a load
    LoadBasis basis;
    if (loaded.empty())
    {
        basis.m_vectors = base::BudgetedBuffer<BasisVector>(budget, 0, "load basis vectors");
        basis.m_coefficients = linalg::DenseMatrix(budget, 0, caseCount, "load basis coefficients");
        basis.m_report.family = input.family;
        basis.m_report.tolerance = input.tolerance;
        basis.m_report.explanation = "no case of the family has a load";
        return basis;
    }

    // The deck tolerance needs the precision of every value; without it the rank falls back to the round-off.
    bool usesWrittenPrecision = input.tolerance == LoadBasisTolerance::WrittenPrecision;
    bool isWrittenPrecisionMissing = false;
    if (usesWrittenPrecision)
        for (const std::int64_t c : loaded)
            for (const double precision : cases.precisionsOf(c))
                isWrittenPrecisionMissing = isWrittenPrecisionMissing || !std::isfinite(precision);
    if (isWrittenPrecisionMissing)
        usesWrittenPrecision = false;

    // The exact constructions: the cases and the loaded DOFs need no check (their residual is zero by construction),
    // so only their vectors are listed here, their coefficients made once one of them is chosen (a basis of the cases
    // has an identity of cases × cases coefficients: 6 000 cases are 288 MB not worth making for a basis not taken).
    std::vector<Candidate> candidates; // at most four
    const auto exactCandidate = [&](LoadBasisKind kind, std::vector<BasisVector> vectors)
    {
        Candidate candidate;
        candidate.kind = kind;
        candidate.vectors = std::move(vectors);
        candidate.residuals = base::BudgetedBuffer<double>(budget, caseCount, "load basis residuals");
        return candidate;
    };
    {
        std::vector<BasisVector> vectors;
        for (const std::int64_t c : loaded)
            vectors.push_back({BasisVector::Source::Case, c});
        candidates.push_back(exactCandidate(LoadBasisKind::Cases, std::move(vectors)));
    }
    std::int64_t loadedDofCount = -1;
    if (input.offersLoadedDofs)
    {
        base::BudgetedBuffer<std::uint8_t> isLoaded(budget, static_cast<std::size_t>(cases.rowCount()),
                                                    "load basis loaded rows");
        for (const std::int64_t c : loaded)
            for (const std::int64_t row : cases.rowsOf(c))
                isLoaded[static_cast<std::size_t>(row)] = 1;
        std::vector<BasisVector> vectors;
        for (std::size_t row = 0; row < isLoaded.size(); ++row)
            if (isLoaded[row] != 0)
                vectors.push_back({BasisVector::Source::LoadedRow, static_cast<std::int64_t>(row)});
        loadedDofCount = static_cast<std::int64_t>(vectors.size());
        candidates.push_back(exactCandidate(LoadBasisKind::LoadedDofs, std::move(vectors)));
    }
    std::int64_t loadSetCount = -1;
    if (input.loadSets != nullptr)
    {
        std::vector<BasisVector> vectors;
        for (std::int64_t s = 0; s < input.loadSets->columnCount(); ++s)
            vectors.push_back({BasisVector::Source::LoadSet, s});
        linalg::DenseMatrix factors(budget, vectors.size(), caseCount, "load basis coefficients");
        for (std::size_t c = 0; c < caseCount; ++c)
            for (std::size_t s = 0; s < vectors.size(); ++s)
                factors(s, c) = (*input.setCoefficients)(s, c);
        candidates.push_back(
            completeByCheck(LoadBasisKind::LoadSets, std::move(vectors), input, false, std::move(factors), budget));
        loadSetCount = static_cast<std::int64_t>(candidates.back().vectors.size());
    }

    // The rank, when it fits the budget and could be smaller than every candidate so far.
    std::size_t smallest = std::numeric_limits<std::size_t>::max();
    for (const Candidate& candidate : candidates)
        smallest = std::min(smallest, candidate.vectors.size());
    std::string rankNote;
    std::int64_t rank = -1;
    bool isRankFull = false; // the rank equals the cases (or reaches the basis taken): no gain
    const std::size_t denseBytes = bytesOfDense(static_cast<std::size_t>(cases.rowCount()), loaded.size());
    if (smallest <= 1)
        rankNote = "the rank test was not needed (a basis of one vector)";
    else if (static_cast<double>(denseBytes) > RANK_TEST_BUDGET_FRACTION * static_cast<double>(budget.available()))
        rankNote = base::concatenate({"the rank test was skipped: its dense copy needs ", std::to_string(denseBytes),
                                      " bytes, more than half of the memory budget left"});
    else if (smallest == loaded.size() && provesNoRankGain(cases, loaded, usesWrittenPrecision, input, budget))
    {
        rankNote = base::concatenate({"the rank is ", std::to_string(loaded.size()),
                                      ", no fewer vectors than the basis taken (proved without pivoting)"});
        isRankFull = true;
    }
    else
    {
        const std::optional<std::vector<std::int64_t>> pivots =
            rankPivots(cases, loaded, usesWrittenPrecision, smallest - 1, input.pool, budget);
        if (!pivots)
        {
            rankNote = base::concatenate(
                {"the rank is at least ", std::to_string(smallest - 1), ", no fewer vectors than the basis taken"});
            isRankFull = true;
        }
        else
        {
            std::size_t taken = pivots->size();
            if (usesWrittenPrecision)
                taken = fewestWrittenPrecisionPivots(*pivots, input, budget);
            std::vector<BasisVector> vectors;
            for (std::size_t i = 0; i < taken; ++i)
                vectors.push_back({BasisVector::Source::Case, (*pivots)[i]});
            candidates.push_back(completeByCheck(LoadBasisKind::Rank, std::move(vectors), input, usesWrittenPrecision,
                                                 std::nullopt, budget));
            rank = static_cast<std::int64_t>(candidates.back().vectors.size());
            rankNote = base::concatenate(
                {"rank ", std::to_string(taken), " at ", usesWrittenPrecision ? "the deck precision" : "round-off",
                 " (", std::to_string(candidates.back().joined), " cases joined by the reconstruction check)"});
        }
    }

    // The smallest; on a tie the exact constructions first, in the order LoadSets, LoadedDofs, Cases, then Rank.
    const auto preference = [](LoadBasisKind kind)
    {
        switch (kind)
        {
        case LoadBasisKind::LoadSets:
            return 0;
        case LoadBasisKind::LoadedDofs:
            return 1;
        case LoadBasisKind::Cases:
            return 2;
        default:
            return 3;
        }
    };
    std::size_t chosen = 0;
    for (std::size_t i = 1; i < candidates.size(); ++i)
    {
        const std::size_t size = candidates[i].vectors.size();
        const std::size_t best = candidates[chosen].vectors.size();
        if (size < best || (size == best && preference(candidates[i].kind) < preference(candidates[chosen].kind)))
            chosen = i;
    }
    Candidate& winner = candidates[chosen];
    if (winner.kind == LoadBasisKind::Cases)
    {
        winner.coefficients = linalg::DenseMatrix(budget, winner.vectors.size(), caseCount, "load basis coefficients");
        for (std::size_t i = 0; i < winner.vectors.size(); ++i)
            winner.coefficients(i, static_cast<std::size_t>(winner.vectors[i].index)) = 1.0;
    }
    else if (winner.kind == LoadBasisKind::LoadedDofs)
    {
        // The coefficients of a case on the unit vectors are its values.
        winner.coefficients = linalg::DenseMatrix(budget, winner.vectors.size(), caseCount, "load basis coefficients");
        base::BudgetedBuffer<std::int64_t> position(budget, static_cast<std::size_t>(cases.rowCount()),
                                                    "load basis positions of rows");
        std::fill(position.data(), position.data() + position.size(), -1);
        for (std::size_t i = 0; i < winner.vectors.size(); ++i)
            position[static_cast<std::size_t>(winner.vectors[i].index)] = static_cast<std::int64_t>(i);
        for (const std::int64_t c : loaded)
        {
            const std::span<const std::int64_t> rows = cases.rowsOf(c);
            const std::span<const double> caseValues = cases.valuesOf(c);
            for (std::size_t e = 0; e < rows.size(); ++e)
                winner.coefficients(static_cast<std::size_t>(position[static_cast<std::size_t>(rows[e])]),
                                    static_cast<std::size_t>(c)) += caseValues[e];
        }
    }
    basis.m_report = reportOf(input, winner, static_cast<std::int64_t>(loaded.size()));
    basis.m_report.loadSetCount = loadSetCount;
    basis.m_report.loadedDofCount = loadedDofCount;
    basis.m_report.rank = rank;
    if (isWrittenPrecisionMissing)
        basis.m_report.tolerance = LoadBasisTolerance::Roundoff;
    std::string explanation =
        base::concatenate({nameOf(winner.kind), " basis of ", std::to_string(winner.vectors.size()), " vectors for ",
                           std::to_string(loaded.size()), " cases"});
    if (loadSetCount >= 0)
        explanation.append("; load sets as written: ").append(std::to_string(loadSetCount));
    if (loadedDofCount >= 0)
        explanation.append("; loaded DOFs: ").append(std::to_string(loadedDofCount));
    explanation.append("; ").append(rankNote);
    isRankFull = isRankFull || rank >= static_cast<std::int64_t>(loaded.size());
    if (isRankFull && input.tolerance == LoadBasisTolerance::Roundoff && input.family != LoadFamily::Thermal &&
        input.family != LoadFamily::Enforced)
        explanation.append(
            "; no rank gain at round-off: the written load cases are independent; "
            "loadBasisTolerance=\"writtenPrecision\" uses the recorded numeric precision, or express repeated shapes "
            "as their own load sets combined by LoadCombination");
    if (isWrittenPrecisionMissing)
        explanation.append(
            "; written precision was requested, but some load cards lack significantDigits metadata; "
            "the rank used round-off instead: take a new snapshot from O_o_Reader to include numeric precision");
    basis.m_report.explanation = std::move(explanation);
    if (input.tolerance == LoadBasisTolerance::WrittenPrecision && !isWrittenPrecisionMissing)
    {
        // The worst cases first, the lower case on a tie (a total order: the same cases whatever the sort).
        base::BudgetedBuffer<std::int64_t> order(budget, loaded.size(), "load basis worst cases");
        std::copy(loaded.begin(), loaded.end(), order.data());
        const std::size_t shown = std::min<std::size_t>(order.size(), WRITTEN_PRECISION_CHECK_CASES);
        std::partial_sort(order.data(), order.data() + shown, order.data() + order.size(),
                          [&](std::int64_t left, std::int64_t right)
                          {
                              const double a = winner.residuals[static_cast<std::size_t>(left)];
                              const double b = winner.residuals[static_cast<std::size_t>(right)];
                              return a > b || (a == b && left < right);
                          });
        for (std::size_t i = 0; i < shown; ++i)
        {
            if (winner.residuals[static_cast<std::size_t>(order[i])] <= 0.0)
                break;
            basis.m_report.worstCases.push_back(order[i]);
            basis.m_report.worstResiduals.push_back(winner.residuals[static_cast<std::size_t>(order[i])]);
        }
    }
    basis.m_vectors = base::BudgetedBuffer<BasisVector>(budget, winner.vectors.size(), "load basis vectors");
    std::copy(winner.vectors.begin(), winner.vectors.end(), basis.m_vectors.data());
    basis.m_coefficients = std::move(winner.coefficients);
    return basis;
}

} // namespace oos::assembly
