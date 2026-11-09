#include "oos/solutions/common/MechanismShapes.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/SmallEigenSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace oos::solutions
{

namespace
{

// A fixed pseudo-random value in [−1, 1) for entry (row, column) of the start block: SplitMix64 (Steele, Lea, Flood,
// "Fast splittable pseudorandom number generators", OOPSLA 2014) of the entry's index, so that the start block, and
// therefore the shapes, do not depend on anything but the matrix.
double startValue(std::size_t row, std::size_t column, std::size_t columns)
{
    std::uint64_t z = static_cast<std::uint64_t>(row) * columns + column + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    z ^= z >> 31U;
    constexpr double TWO_TO_MINUS_53 = 1.0 / 9007199254740992.0;
    return 2.0 * static_cast<double>(z >> 11U) * TWO_TO_MINUS_53 - 1.0;
}

double columnDot(const linalg::DenseMatrix& x, std::size_t a, std::size_t b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < x.rows(); ++i)
        sum += x(i, a) * x(i, b);
    return sum;
}

// Modified Gram-Schmidt, twice ("twice is enough": Giraud, Langou, Rozložník, Comput. Math. Appl. 50, 2005), column by
// column in increasing order; returns the number of independent columns, moved to the front.
std::size_t orthonormalise(linalg::DenseMatrix& x, std::size_t columns)
{
    std::size_t kept = 0;
    for (std::size_t j = 0; j < columns; ++j)
    {
        const double before = std::sqrt(columnDot(x, j, j));
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t p = 0; p < kept; ++p)
            {
                const double projection = columnDot(x, p, j);
                for (std::size_t i = 0; i < x.rows(); ++i)
                    x(i, j) -= projection * x(i, p);
            }
        const double after = std::sqrt(columnDot(x, j, j));
        if (!(after > MechanismShapes::DEPENDENT_COLUMN_RATIO * before) || !std::isfinite(after))
            continue;
        for (std::size_t i = 0; i < x.rows(); ++i)
            x(i, kept) = x(i, j) / after;
        ++kept;
    }
    return kept;
}

} // namespace

linalg::BlockSparseMatrix MechanismShapes::shiftedStiffness(const linalg::BlockSparseMatrix& knn,
                                                            base::MemoryBudget& budget)
{
    const linalg::SparsePattern& pattern = knn.pattern();
    linalg::BlockSparseMatrix shifted(pattern, budget, "mechanism shapes K + shift diag(K)");
    std::copy(knn.values().begin(), knn.values().end(), shifted.values().begin());
    for (std::int32_t node = 0; node < pattern.nodeCount(); ++node)
    {
        // The diagonal block comes first in its block column (SparsePattern layout), column-major.
        const std::span<double> block = shifted.blockValues(pattern.columnBegin(node));
        const auto dofs = static_cast<std::size_t>(pattern.nodeDofCount(node));
        for (std::size_t c = 0; c < dofs; ++c)
            block[c * dofs + c] *= 1.0 + SHIFT;
    }
    return shifted;
}

std::vector<base::MechanismShape> MechanismShapes::compute(const FsetSolver& solver,
                                                           const linalg::BlockSparseMatrix& knn,
                                                           const assembly::RigidElimination& gm,
                                                           const assembly::DofMap& dofMap, std::size_t failedPivots,
                                                           base::MemoryBudget& budget,
                                                           const base::CancellationToken* cancellation)
{
    const auto equations = static_cast<std::size_t>(solver.equationCount());
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    std::size_t count = std::min({failedPivots + GUARD_COLUMNS, MAX_SHAPES, equations});
    if (gSize > 0)
        count = std::min(count, MAX_LISTED_ENTRIES / gSize);
    if (count == 0)
        return {};

    linalg::DenseMatrix x(budget, equations, count, "mechanism shapes");
    for (std::size_t j = 0; j < count; ++j)
        for (std::size_t e = 0; e < equations; ++e)
            x(e, j) = startValue(e, j, count);
    std::size_t kept = count;
    for (std::size_t iteration = 0; iteration < INVERSE_ITERATIONS && kept > 0; ++iteration)
    {
        solver.solve(linalg::DenseMatrixView{x.data(), equations, kept, x.leadingDimension()}, cancellation);
        kept = orthonormalise(x, kept);
    }
    if (kept == 0)
        return {};

    // The Rayleigh-Ritz projection: H = Xᵀ K X and G = Xᵀ D X, K X by the serial product of Knn on g-set vectors whose
    // DOFs outside the l set are zero.
    base::BudgetedBuffer<double> motion(budget, gSize, "mechanism shape in the g set");
    base::BudgetedBuffer<double> product(budget, gSize, "mechanism shape K x");
    linalg::DenseMatrix kx(budget, equations, kept, "mechanism shapes K X");
    base::BudgetedBuffer<double> diagonal(budget, equations, "mechanism shapes diag(K)");
    for (std::size_t j = 0; j < kept; ++j)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        std::fill(motion.data(), motion.data() + gSize, 0.0);
        for (std::size_t e = 0; e < equations; ++e)
            motion[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))] = x(e, j);
        knn.multiply(motion.span(), product.span());
        for (std::size_t e = 0; e < equations; ++e)
            kx(e, j) = product[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))];
    }
    {
        // diag(K_ll): K e_dof read through the pattern's diagonal blocks.
        const linalg::SparsePattern& pattern = knn.pattern();
        for (std::size_t e = 0; e < equations; ++e)
        {
            const linalg::NodeDof nodeDof = dofMap.nodeDofOf(solver.dofOfEquation(static_cast<std::int32_t>(e)));
            const std::span<const double> block = knn.blockValues(pattern.columnBegin(nodeDof.node));
            const auto dofs = static_cast<std::size_t>(pattern.nodeDofCount(nodeDof.node));
            const auto c = static_cast<std::size_t>(nodeDof.component);
            diagonal[e] = block[c * dofs + c];
        }
    }
    linalg::DenseMatrix h(budget, kept, kept, "mechanism shapes H");
    linalg::DenseMatrix g(budget, kept, kept, "mechanism shapes G");
    for (std::size_t a = 0; a < kept; ++a)
        for (std::size_t b = 0; b <= a; ++b)
        {
            double energy = 0.0;
            double weight = 0.0;
            for (std::size_t e = 0; e < equations; ++e)
            {
                energy += x(e, a) * kx(e, b);
                weight += x(e, a) * diagonal[e] * x(e, b);
            }
            h(a, b) = h(b, a) = energy;
            g(a, b) = g(b, a) = weight;
        }
    // G = L Lᵀ (Cholesky; G is positive definite: X has independent columns and D > 0 on the l set), then the
    // standard problem C = L⁻¹ H L⁻ᵀ, C z = θ z, y = L⁻ᵀ z.
    for (std::size_t k = 0; k < kept; ++k)
    {
        for (std::size_t p = 0; p < k; ++p)
            g(k, k) -= g(k, p) * g(k, p);
        if (!(g(k, k) > 0.0))
            return {};
        g(k, k) = std::sqrt(g(k, k));
        for (std::size_t i = k + 1; i < kept; ++i)
        {
            for (std::size_t p = 0; p < k; ++p)
                g(i, k) -= g(i, p) * g(k, p);
            g(i, k) /= g(k, k);
        }
    }
    // C = L⁻¹ H L⁻ᵀ: forward substitution on the columns of H, then on the rows.
    for (std::size_t column = 0; column < kept; ++column)
        for (std::size_t i = 0; i < kept; ++i)
        {
            for (std::size_t p = 0; p < i; ++p)
                h(i, column) -= g(i, p) * h(p, column);
            h(i, column) /= g(i, i);
        }
    for (std::size_t row = 0; row < kept; ++row)
        for (std::size_t i = 0; i < kept; ++i)
        {
            for (std::size_t p = 0; p < i; ++p)
                h(row, i) -= g(i, p) * h(row, p);
            h(row, i) /= g(i, i);
        }
    for (std::size_t a = 0; a < kept; ++a)
        for (std::size_t b = 0; b < a; ++b)
            h(a, b) = h(b, a) = 0.5 * (h(a, b) + h(b, a));
    const linalg::SymmetricEigenDecomposition ritz = linalg::SmallEigenSolver::decompose(h.view(), budget);
    // y = L⁻ᵀ z (backward substitution), the Ritz vectors x = X y.
    std::vector<base::MechanismShape> shapes;
    std::vector<double> y(kept); // bounded by the block
    for (std::size_t r = 0; r < kept && ritz.eigenvalues[r] <= ZERO_ENERGY_RATIO; ++r)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        for (std::size_t i = kept; i-- > 0;)
        {
            double value = ritz.eigenvectors(i, r);
            for (std::size_t p = i + 1; p < kept; ++p)
                value -= g(p, i) * y[p];
            y[i] = value / g(i, i);
        }
        std::fill(motion.data(), motion.data() + gSize, 0.0);
        for (std::size_t e = 0; e < equations; ++e)
        {
            double value = 0.0;
            for (std::size_t j = 0; j < kept; ++j)
                value += x(e, j) * y[j];
            motion[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))] = value;
        }
        gm.expandToG(motion.span());
        // The largest amplitude (the first of equal ones) sets the scale and the sign.
        std::size_t largest = 0;
        for (std::size_t dof = 1; dof < gSize; ++dof)
            if (std::abs(motion[dof]) > std::abs(motion[largest]))
                largest = dof;
        const double scale = motion[largest];
        if (scale == 0.0)
            continue;
        base::MechanismShape shape;
        for (std::size_t dof = 0; dof < gSize; ++dof)
        {
            const double amplitude = motion[dof] / scale;
            if (std::abs(amplitude) <= AMPLITUDE_FLOOR)
                continue;
            const auto index = static_cast<base::DofIndex>(dof);
            const assembly::DofLabel label = dofMap.labelOf(index);
            shape.dofs.push_back({dofMap.pointIdOf(index),
                                  label.kind == input::PointKind::Scalar ? 0 : static_cast<int>(label.component)});
            shape.amplitudes.push_back(amplitude);
        }
        shapes.push_back(std::move(shape));
    }
    return shapes;
}

} // namespace oos::solutions
