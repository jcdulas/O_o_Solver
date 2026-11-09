#include "LanczosRun.hpp"

#include "oos/base/threading/ParallelFor.hpp"
#include "oos/linalg/dense/SmallEigenSolver.hpp"
#include "oos/linalg/kernels/MatrixProduct.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace oos::linalg
{
namespace
{

constexpr double EPSILON = std::numeric_limits<double>::epsilon();

// The rows of one piece of the products V·S (restart, Ritz vectors): 256 rows of a basis of a few hundred columns stay
// in the L2 cache with their piece of the result. The pieces depend on the dimension only (determinism).
constexpr std::size_t ROWS_PER_PIECE = 256;

// The passes of block Gram-Schmidt against the basis, and of modified Gram-Schmidt inside a block ("twice is enough").
constexpr int ORTHOGONALISATION_PASSES = 2;

// splitmix64 (Steele, Lea & Flood, OOPSLA 2014): a fixed sequence for a fixed seed, on every machine.
std::uint64_t nextRandom(std::uint64_t& state) noexcept
{
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

DenseMatrixView columnOf(DenseMatrixView matrix, std::size_t column) noexcept
{
    return matrix.block(0, column, matrix.rows, 1);
}

// out(:, 0:k) ← a(:, 0:p)·s (s p × k, column-major with leading dimension lds), by pieces of rows: each piece is
// computed into a buffer, then copied, so `out` may be the first columns of `a` (the in-place product of a restart).
void multiplyByRows(const DenseKernels& kernels, ConstDenseMatrixView a, const double* s, std::size_t lds,
                    std::size_t k, DenseMatrixView out, const base::CancellationToken* cancellation)
{
    const std::size_t p = a.columns;
    if (k == 0 || a.rows == 0)
        return;
    base::parallelFor(
        kernels.options().pool, a.rows, ROWS_PER_PIECE,
        [&](std::size_t first, std::size_t end)
        {
            const std::size_t rows = end - first;
            std::vector<double> piece(rows * k, 0.0); // bounded by ROWS_PER_PIECE × the basis
            if (p > 0)
                multiplyMatrices(kernels.kind(), {a.pointer(first, 0), a.leadingDimension, s,
                                                  std::max<std::size_t>(lds, 1), piece.data(), rows, rows, k, p});
            for (std::size_t j = 0; j < k; ++j)
                std::copy_n(piece.data() + j * rows, rows, &out(first, j));
        },
        cancellation);
}

} // namespace

LanczosWorkspace::LanczosWorkspace(base::MemoryBudget& budget, std::size_t dimension, std::size_t basisColumns,
                                   std::size_t blockSize)
    : basis(budget, dimension, basisColumns, "Lanczos: basis"),
      next(budget, dimension, blockSize, "Lanczos: blocks"),
      product(budget, dimension, blockSize, "Lanczos: blocks"),
      massScratch(budget, dimension, blockSize, "Lanczos: blocks"),
      massOfNext(budget, dimension, blockSize, "Lanczos: blocks")
{
}

std::uint64_t LanczosWorkspace::estimateBytes(std::size_t dimension, std::size_t basisColumns, std::size_t blockSize)
{
    const auto rows = static_cast<std::uint64_t>(DenseMatrix::alignedLeadingDimension(dimension));
    return rows * (static_cast<std::uint64_t>(basisColumns) + 4 * static_cast<std::uint64_t>(blockSize)) *
           sizeof(double);
}

LanczosRun::LanczosRun(ShiftedPencil& pencil, const DenseKernels& kernels, LanczosWorkspace& workspace,
                       base::MemoryBudget& budget, const LanczosRunSettings& settings,
                       const base::OperationControl& control)
    : m_pencil(pencil),
      m_kernels(kernels),
      m_workspace(workspace),
      m_budget(budget),
      m_settings(settings),
      m_control(control),
      m_orthogonalizer(pencil, kernels, budget, workspace.massScratch.view(), settings.innerProductPieceRows),
      m_dimension(static_cast<std::size_t>(pencil.dimension())),
      m_capacity(settings.basisColumns)
{
}

ConstDenseMatrixView LanczosRun::basisView(std::size_t columns) const
{
    return {m_workspace.basis.data(), m_dimension, columns, m_workspace.basis.leadingDimension()};
}

void LanczosRun::fillRandom(DenseMatrixView block)
{
    // Uniform in [−1, 1): the 53 high bits of a draw.
    constexpr double TWO_TO_MINUS_52 = 1.0 / 4503599627370496.0;
    for (std::size_t j = 0; j < block.columns; ++j)
    {
        for (std::size_t i = 0; i < block.rows; ++i)
            block(i, j) = static_cast<double>(nextRandom(m_randomState) >> 11) * TWO_TO_MINUS_52 - 1.0;
    }
}

void LanczosRun::applyOperator(DenseMatrixView block)
{
    // Op·X = (K − σM)⁻¹·(M·X).
    const DenseMatrixView mass = m_workspace.massScratch.view().block(0, 0, block.rows, block.columns);
    m_pencil.multiplyMass(block, mass);
    for (std::size_t j = 0; j < block.columns; ++j)
        std::copy_n(&mass(0, j), block.rows, &block(0, j));
    m_pencil.solve(block);
}

std::size_t LanczosRun::factoriseBlock(DenseMatrixView block, std::vector<double>* triangle,
                                       const EigenvectorStore& store)
{
    const std::size_t width = block.columns;
    const DenseMatrixView massOfQ = m_workspace.massOfNext.view();
    const double rootDimension = std::sqrt(static_cast<double>(std::max<std::size_t>(m_dimension, 1)));
    if (triangle != nullptr)
        triangle->assign(width * width, 0.0);
    std::size_t filled = 0;
    // The columns before c are M-orthonormal (or empty) with M·q in massOfQ: q_iᵀ·M·w = (M·q_i)ᵀ·w.
    // PERFORMANCE: column c against the columns before it by classical Gram-Schmidt twice, the
    // update of one pass and the coefficients of the next in one sweep of the rows (each row of w updated before it
    // enters the sums): three sweeps for any c, on the pool by the orthogonalizer's pieces. Before: modified
    // Gram-Schmidt twice, 2 + 2c sweeps of the rows on one thread (54 sweeps for a block of 7, the update of column i
    // fused with the product by M·q_{i+1}).
    // Measured: eigenvalue solve of 99 846 DOFs and 100 modes.
    const auto subtractEarlierColumns = [&](std::size_t c, bool isRecorded)
    {
        if (c == 0)
            return;
        double* w = &block(0, c);
        const ConstDenseMatrixView earlier = ConstDenseMatrixView(block).block(0, 0, m_dimension, c);
        const ConstDenseMatrixView earlierMass = ConstDenseMatrixView(massOfQ).block(0, 0, m_dimension, c);
        std::vector<double> coefficients(c); // bounded by the block
        std::vector<double> next(c);         // bounded by the block
        m_orthogonalizer.coefficients(earlierMass, w, coefficients);
        for (int pass = 0; pass < ORTHOGONALISATION_PASSES; ++pass)
        {
            if (isRecorded && triangle != nullptr)
            {
                for (std::size_t i = 0; i < c; ++i)
                    (*triangle)[i + c * width] += coefficients[i];
            }
            const bool isLast = pass + 1 == ORTHOGONALISATION_PASSES;
            m_orthogonalizer.subtractColumns(w, earlier, coefficients, isLast ? ConstDenseMatrixView() : earlierMass,
                                             isLast ? std::span<double>() : std::span<double>(next));
            std::swap(coefficients, next);
        }
    };
    const auto normalise = [&](std::size_t c, double threshold)
    {
        m_orthogonalizer.multiplyColumn(&block(0, c), &massOfQ(0, c));
        const double norm = std::sqrt(std::max(0.0, m_orthogonalizer.dot(&block(0, c), &massOfQ(0, c), m_dimension)));
        if (!(norm > threshold))
            return 0.0;
        m_orthogonalizer.scale(&block(0, c), &massOfQ(0, c), 1.0 / norm, m_dimension);
        return norm;
    };

    for (std::size_t c = 0; c < width; ++c)
    {
        subtractEarlierColumns(c, true);
        const double norm = normalise(c, DEFLATION_FACTOR * EPSILON * rootDimension * m_referenceNorms[c]);
        if (norm > 0.0)
        {
            if (triangle != nullptr)
                (*triangle)[c + c * width] = norm;
            ++filled;
            continue;
        }
        // A deflation: the column is (to rounding) in the span of the basis; B keeps a zero there and the column is
        // replaced by a new direction, Op applied to a random vector.
        bool isReplaced = false;
        for (std::size_t attempt = 0; attempt < REPLACEMENT_ATTEMPTS && !isReplaced; ++attempt)
        {
            const DenseMatrixView column = columnOf(block, c);
            fillRandom(column);
            applyOperator(column);
            double referenceNorm = 0.0;
            for (int pass = 0; pass < ORTHOGONALISATION_PASSES; ++pass)
                m_orthogonalizer.projectOut(column, store.locked(), basisView(m_basisInUse), nullptr,
                                            pass == 0 ? std::span<double>(&referenceNorm, 1) : std::span<double>());
            subtractEarlierColumns(c, false);
            isReplaced = normalise(c, DEFLATION_FACTOR * EPSILON * rootDimension * referenceNorm) > 0.0;
        }
        if (!isReplaced)
        {
            // The space orthogonal to the locked set and the basis is exhausted: an empty column.
            for (std::size_t row = 0; row < m_dimension; ++row)
            {
                block(row, c) = 0.0;
                massOfQ(row, c) = 0.0;
            }
        }
        else
        {
            ++filled;
        }
    }
    return filled;
}

void LanczosRun::startBlock(const EigenvectorStore& store)
{
    const DenseMatrixView next = m_workspace.next.view();
    fillRandom(next);
    // Op applied once puts the block in the range of Op, which excludes the null space of M (Ericsson & Ruhe 1980).
    applyOperator(next);
    m_referenceNorms.assign(next.columns, 0.0);
    for (int pass = 0; pass < ORTHOGONALISATION_PASSES; ++pass)
        m_orthogonalizer.projectOut(next, store.locked(), basisView(0), nullptr,
                                    pass == 0 ? std::span<double>(m_referenceNorms) : std::span<double>());
    m_basisInUse = 0;
    m_nextColumnCount = factoriseBlock(next, nullptr, store);
}

LanczosRun::RitzPairs LanczosRun::computeRitzPairs() const
{
    const std::size_t p = m_columns;
    const std::size_t b = m_settings.blockSize;
    const std::size_t ld = m_settings.basisColumns;
    const SymmetricEigenDecomposition decomposition =
        SmallEigenSolver::decompose({m_projected.data(), p, p, std::max<std::size_t>(ld, 1)}, m_budget);
    std::vector<std::size_t> order(p);
    std::iota(order.begin(), order.end(), std::size_t{0});
    // Nearest the shift first: decreasing |θ|, ties by index (the order of SmallEigenSolver, itself fixed).
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y)
                     { return std::abs(decomposition.eigenvalues[x]) > std::abs(decomposition.eigenvalues[y]); });
    RitzPairs ritz;
    ritz.values.resize(p);
    ritz.vectors.resize(p * p);
    ritz.residuals.resize(p);
    ritz.isConverged.assign(p, 0);
    const double largest = p > 0 ? std::abs(decomposition.eigenvalues[order[0]]) : 0.0;
    for (std::size_t k = 0; k < p; ++k)
    {
        const std::size_t from = order[k];
        ritz.values[k] = decomposition.eigenvalues[from];
        for (std::size_t i = 0; i < p; ++i)
            ritz.vectors[i + k * p] = decomposition.eigenvectors(i, from);
        // ‖G·s‖₂, G = m_coupling (b × p).
        double squares = 0.0;
        for (std::size_t r = 0; r < b; ++r)
        {
            double value = 0.0;
            for (std::size_t i = 0; i < p; ++i)
                value += m_coupling[r + i * b] * ritz.vectors[i + k * p];
            squares += value * value;
        }
        ritz.residuals[k] = std::sqrt(squares);
        const double theta = std::abs(ritz.values[k]);
        const bool isAccurate = theta > 0.0 && theta >= ACCURATE_RITZ_RATIO * largest;
        ritz.accurateCount += isAccurate ? 1 : 0;
        ritz.isConverged[k] = isAccurate && ritz.residuals[k] <= m_settings.convergenceTolerance * theta ? 1 : 0;
    }
    return ritz;
}

bool LanczosRun::hasConvergedTarget(const RitzPairs& ritz) const
{
    // The target, or every accurate Ritz value when fewer are (the others belong to later shifts); a basis smaller
    // than the target has not seen enough of the spectrum yet.
    if (ritz.values.size() < m_settings.target)
        return false;
    const std::size_t wanted = std::min(m_settings.target, ritz.accurateCount);
    if (wanted == 0)
        return false;
    for (std::size_t k = 0; k < wanted; ++k)
    {
        if (ritz.isConverged[k] == 0)
            return false;
    }
    return true;
}

void LanczosRun::restart(const RitzPairs& ritz)
{
    const std::size_t p = m_columns;
    const std::size_t b = m_settings.blockSize;
    const std::size_t ld = m_settings.basisColumns;
    // Keep about half the basis (Wu & Simon 2000: the nearest Ritz vectors carry the convergence), at least the target
    // and a block, leaving room for two blocks of new directions.
    const std::size_t room = m_capacity - 2 * b;
    const std::size_t kept = std::min({room, p - b, std::max(m_capacity / 2, std::min(m_settings.target + b, room))});
    multiplyByRows(m_kernels, basisView(p), ritz.vectors.data(), p, kept, m_workspace.basis.view(),
                   m_control.cancellation);
    // G·S_k: the coupling of the next block with the kept Ritz vectors.
    std::vector<double> coupling(b * kept, 0.0);
    for (std::size_t k = 0; k < kept; ++k)
    {
        for (std::size_t r = 0; r < b; ++r)
        {
            double value = 0.0;
            for (std::size_t i = 0; i < p; ++i)
                value += m_coupling[r + i * b] * ritz.vectors[i + k * p];
            coupling[r + k * b] = value;
        }
    }
    std::fill(m_coupling.begin(), m_coupling.end(), 0.0);
    std::copy(coupling.begin(), coupling.end(), m_coupling.begin());
    std::fill(m_projected.begin(), m_projected.end(), 0.0);
    for (std::size_t k = 0; k < kept; ++k)
        m_projected[k + k * ld] = ritz.values[k];
    m_columns = kept;
}

void LanczosRun::extractConverged(const RitzPairs& ritz, double shift, EigenvectorStore& store,
                                  LanczosRunResult& result)
{
    const std::size_t p = m_columns;
    const std::size_t b = m_settings.blockSize;
    std::vector<std::size_t> converged;
    const double largest = ritz.values.empty() ? 0.0 : std::abs(ritz.values[0]);
    for (std::size_t k = 0; k < ritz.values.size(); ++k)
    {
        if (ritz.isConverged[k] != 0)
            converged.push_back(k);
        else if (std::abs(ritz.values[k]) > NEGLIGIBLE_RITZ_VALUE * largest)
            result.ritzEstimates.push_back(shift + 1.0 / ritz.values[k]);
    }
    std::sort(result.ritzEstimates.begin(), result.ritzEstimates.end());
    const std::size_t count = converged.size();
    result.convergedCount = count;
    if (count == 0)
        return;

    // S_c and Y = G·S_c·diag(1/θ), the purification term.
    std::vector<double> selected(p * count);
    std::vector<double> negatedCorrection(b * count, 0.0);
    std::vector<double> values(count);
    std::vector<double> bounds(count);
    for (std::size_t j = 0; j < count; ++j)
    {
        const std::size_t k = converged[j];
        std::copy_n(ritz.vectors.data() + k * p, p, selected.data() + j * p);
        const double theta = ritz.values[k];
        for (std::size_t r = 0; r < b; ++r)
        {
            double value = 0.0;
            for (std::size_t i = 0; i < p; ++i)
                value += m_coupling[r + i * b] * ritz.vectors[i + k * p];
            negatedCorrection[r + j * b] = -value / theta;
        }
        // λ = σ + 1/θ (the inverse of the spectral transformation).
        values[j] = shift + 1.0 / theta;
        bounds[j] = ritz.residuals[k] / std::abs(theta);
    }
    const DenseMatrixView out = store.prepareColumns(count);
    multiplyByRows(m_kernels, basisView(p), selected.data(), p, count, out, m_control.cancellation);
    // φ = V·s + V_next·(G·s)/θ.
    m_kernels.subtractProduct(out, m_workspace.next.view(),
                              ConstDenseMatrixView(negatedCorrection.data(), b, count, b));
    // Normalise in the inner product of the pencil (M, or K for a buckling pencil), by blocks of the scratch width.
    const DenseMatrixView mass = m_workspace.massScratch.view();
    for (std::size_t first = 0; first < count; first += b)
    {
        const std::size_t width = std::min(b, count - first);
        const DenseMatrixView columns = out.block(0, first, m_dimension, width);
        m_pencil.multiplyInnerProduct(columns, mass.block(0, 0, m_dimension, width));
        for (std::size_t j = 0; j < width; ++j)
        {
            const double norm =
                std::sqrt(std::max(0.0, m_orthogonalizer.dot(&columns(0, j), &mass(0, j), m_dimension)));
            const double inverse = norm > 0.0 ? 1.0 / norm : 0.0;
            m_orthogonalizer.scale(&columns(0, j), nullptr, inverse, m_dimension);
        }
    }
    store.commit(values, bounds);
}

LanczosRunResult LanczosRun::run(double shift, EigenvectorStore& store)
{
    LanczosRunResult result;
    const std::size_t b = m_settings.blockSize;
    const std::size_t ld = m_settings.basisColumns;
    m_randomState = m_settings.seed;
    m_projected.assign(ld * ld, 0.0);
    m_coupling.assign(b * ld, 0.0);
    m_columns = 0;
    m_operatorNorm = 0.0;
    startBlock(store);
    if (m_nextColumnCount == 0)
    {
        result.isSpaceExhausted = true;
        return result;
    }

    RitzPairs ritz;
    std::vector<double> coefficients;
    std::vector<double> secondCoefficients;
    std::vector<double> triangle;
    std::vector<double> norms(b, 0.0);
    while (true)
    {
        m_control.throwIfCancelled();
        if (m_columns + b > m_capacity)
        {
            // A restart keeps at least a block and adds two: a basis of fewer than three blocks (a pencil of one or
            // two equations) has already spanned the whole space.
            if (result.restarts >= m_settings.maximumRestarts || m_capacity < 3 * b)
                break;
            restart(ritz);
            ++result.restarts;
        }
        const std::size_t p = m_columns;
        // Append the next block: T gains the coupling G (below) and Gᵀ (beside).
        for (std::size_t j = 0; j < p; ++j)
        {
            for (std::size_t r = 0; r < b; ++r)
            {
                m_projected[(p + r) + j * ld] = m_coupling[r + j * b];
                m_projected[j + (p + r) * ld] = m_coupling[r + j * b];
            }
        }
        DenseMatrix& basis = m_workspace.basis;
        for (std::size_t r = 0; r < b; ++r)
            std::copy_n(&m_workspace.next(0, r), m_dimension, &basis(0, p + r));
        m_basisInUse = p + b;

        // W = Op·V_j, then M-orthogonalised twice against the locked set and the whole basis.
        const DenseMatrixView w = m_workspace.product.view();
        for (std::size_t r = 0; r < b; ++r)
            std::copy_n(&basis(0, p + r), m_dimension, &w(0, r));
        applyOperator(w);
        m_orthogonalizer.projectOut(w, store.locked(), basisView(p + b), &coefficients, norms);
        m_orthogonalizer.projectOut(w, store.locked(), basisView(p + b), &secondCoefficients, {});
        for (const double norm : norms)
            m_operatorNorm = std::max(m_operatorNorm, norm);
        // A_j = V_jᵀ·M·W from both passes, made exactly symmetric (T is symmetric in exact arithmetic).
        const std::size_t rows = p + b;
        for (std::size_t c = 0; c < b; ++c)
        {
            for (std::size_t r = 0; r <= c; ++r)
            {
                const double upper = coefficients[(p + r) + c * rows] + secondCoefficients[(p + r) + c * rows];
                const double lower = coefficients[(p + c) + r * rows] + secondCoefficients[(p + c) + r * rows];
                const double value = 0.5 * (upper + lower);
                m_projected[(p + r) + (p + c) * ld] = value;
                m_projected[(p + c) + (p + r) * ld] = value;
            }
        }
        // W = V_next·B.
        m_referenceNorms.assign(b, m_operatorNorm);
        m_nextColumnCount = factoriseBlock(w, &triangle, store);
        std::swap(m_workspace.next, m_workspace.product);
        std::fill(m_coupling.begin(), m_coupling.end(), 0.0);
        for (std::size_t c = 0; c < b; ++c)
        {
            for (std::size_t r = 0; r < b; ++r)
                m_coupling[r + (p + c) * b] = triangle[r + c * b];
        }
        m_columns = p + b;
        ++result.blockSteps;

        ritz = computeRitzPairs();
        if (hasConvergedTarget(ritz) || m_nextColumnCount == 0)
            break;
    }
    extractConverged(ritz, shift, store, result);
    return result;
}

} // namespace oos::linalg
