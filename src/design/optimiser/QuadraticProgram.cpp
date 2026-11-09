#include "oos/design/optimiser/QuadraticProgram.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

namespace oos::design
{

namespace
{

// A row is satisfied when its residual is above −FEASIBILITY_TOLERANCE times its scale (1, |d_i| and |c_i|·|x|):
// round-off of the products, far below the CTMIN of the optimisers.
constexpr double FEASIBILITY_TOLERANCE = 1.0e-10;
// A new normal whose part outside the span of the active ones holds less than this fraction of its G⁻¹-norm is
// dependent on them (Goldfarb-Idnani: z = 0, a step in the dual space only).
constexpr double DEPENDENCE_TOLERANCE = 1.0e-12;
// The pivot of the Cholesky factor of G below this fraction of the largest diagonal means G is not positive definite.
constexpr double DEFINITENESS_TOLERANCE = 1.0e-14;
// The method adds or drops one row per step and never cycles in exact arithmetic; this bound guards round-off.
constexpr std::size_t STEP_LIMIT_PER_ROW = 20;

// A square row-major matrix of n × n in budgeted storage.
class Square
{
public:
    Square(base::MemoryBudget& budget, std::size_t n, const char* label) : m_n(n), m_values(budget, n * n, label) {}
    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept { return m_values[i * m_n + j]; }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept { return m_values[i * m_n + j]; }

private:
    std::size_t m_n;
    base::BudgetedBuffer<double> m_values;
};

class DualActiveSet
{
public:
    DualActiveSet(std::span<const double> g, std::span<const double> a, std::span<const double> c,
                  std::span<const double> d, base::MemoryBudget& budget)
        : m_n(a.size()),
          m_m(d.size()),
          m_g(g),
          m_a(a),
          m_c(c),
          m_d(d),
          m_budget(budget),
          m_j(budget, m_n, "quadratic program J"),
          m_r(budget, m_n, "quadratic program R")
    {
    }

    QuadraticProgramSolution run()
    {
        factorise();
        QuadraticProgramSolution solution;
        // The unconstrained minimum x = −G⁻¹·a = −J·Jᵀ·a.
        std::vector<double> x(m_n, 0.0);
        std::vector<double> jta = multiplyJTransposed(m_a);
        for (std::size_t i = 0; i < m_n; ++i)
            for (std::size_t k = 0; k < m_n; ++k)
                x[i] -= m_j(i, k) * jta[k];
        std::vector<bool> isActive(m_m, false);
        const std::size_t stepLimit = STEP_LIMIT_PER_ROW * (m_m + m_n) + m_n;
        std::size_t steps = 0;
        while (true)
        {
            // The most violated row.
            std::optional<std::size_t> violated;
            double worst = 0.0;
            for (std::size_t i = 0; i < m_m; ++i)
            {
                if (isActive[i])
                    continue;
                const double residual = rowResidual(i, x);
                if (residual < -FEASIBILITY_TOLERANCE * rowScale(i, x) && residual < worst)
                {
                    worst = residual;
                    violated = i;
                }
            }
            if (!violated)
                break;
            const std::size_t p = *violated;
            const std::span<const double> normal = m_c.subspan(p * m_n, m_n);
            double added = 0.0; // the multiplier of row p
            while (true)
            {
                if (++steps > stepLimit)
                    throw base::OptimisationError(base::concatenate(
                        {"quadratic program: no solution after ", std::to_string(stepLimit),
                         " active-set changes (round-off on nearly dependent rows); scale the problem"}));
                std::vector<double> dvec = multiplyJTransposed(normal);
                // The primal direction z = J₂·d₂ (outside the active normals) and the dual one r = R⁻¹·d₁.
                std::vector<double> z(m_n, 0.0);
                double outside = 0.0;
                double total = 0.0;
                for (std::size_t k = 0; k < m_n; ++k)
                {
                    total += dvec[k] * dvec[k];
                    if (k < m_active.size())
                        continue;
                    outside += dvec[k] * dvec[k];
                    for (std::size_t i = 0; i < m_n; ++i)
                        z[i] += m_j(i, k) * dvec[k];
                }
                const std::vector<double> r = solveR(dvec);
                // The partial step: the largest t keeping every active multiplier ≥ 0.
                double partial = std::numeric_limits<double>::infinity();
                std::optional<std::size_t> dropped;
                for (std::size_t k = 0; k < m_active.size(); ++k)
                    if (r[k] > 0.0 && m_multipliers[k] / r[k] < partial)
                    {
                        partial = m_multipliers[k] / r[k];
                        dropped = k;
                    }
                // The full step: row p satisfied with equality (zᵀ·n_p = |d₂|²).
                const bool hasPrimalStep = outside > DEPENDENCE_TOLERANCE * total;
                const double full =
                    hasPrimalStep ? -rowResidual(p, x) / outside : std::numeric_limits<double>::infinity();
                const double t = std::min(partial, full);
                if (!std::isfinite(t))
                {
                    solution.status = QuadraticProgramStatus::Infeasible;
                    return finish(solution, x, steps);
                }
                for (std::size_t k = 0; k < m_active.size(); ++k)
                    m_multipliers[k] -= t * r[k];
                added += t;
                if (hasPrimalStep)
                    for (std::size_t i = 0; i < m_n; ++i)
                        x[i] += t * z[i];
                if (hasPrimalStep && full <= partial)
                {
                    addRow(dvec, p, added);
                    isActive[p] = true;
                    break;
                }
                isActive[m_active[*dropped]] = false;
                dropRow(*dropped);
            }
        }
        solution.status = QuadraticProgramStatus::Optimal;
        return finish(solution, x, steps);
    }

private:
    // G = L·Lᵀ (Cholesky), then J = L⁻ᵀ, upper triangular.
    void factorise()
    {
        Square l(m_budget, m_n, "quadratic program L");
        double largest = 0.0;
        for (std::size_t i = 0; i < m_n; ++i)
            largest = std::max(largest, std::abs(m_g[i * m_n + i]));
        for (std::size_t j = 0; j < m_n; ++j)
        {
            double diagonal = m_g[j * m_n + j];
            for (std::size_t k = 0; k < j; ++k)
                diagonal -= l(j, k) * l(j, k);
            if (!(diagonal > DEFINITENESS_TOLERANCE * largest))
                throw base::OptimisationError(
                    base::concatenate({"quadratic program: expected a positive definite matrix G, found the pivot ",
                                       std::to_string(diagonal), " at row ", std::to_string(j),
                                       "; the quasi-Newton matrix of SQP lost its definiteness (report the problem)"}));
            const double pivot = std::sqrt(diagonal);
            l(j, j) = pivot;
            for (std::size_t i = j + 1; i < m_n; ++i)
            {
                double value = m_g[i * m_n + j];
                for (std::size_t k = 0; k < j; ++k)
                    value -= l(i, k) * l(j, k);
                l(i, j) = value / pivot;
            }
        }
        // L⁻¹ by forward substitution, column by column; J = (L⁻¹)ᵀ.
        for (std::size_t column = 0; column < m_n; ++column)
            for (std::size_t i = column; i < m_n; ++i)
            {
                double value = i == column ? 1.0 : 0.0;
                for (std::size_t k = column; k < i; ++k)
                    value -= l(i, k) * m_j(column, k);
                m_j(column, i) = value / l(i, i);
            }
    }

    [[nodiscard]] std::vector<double> multiplyJTransposed(std::span<const double> v) const
    {
        std::vector<double> result(m_n, 0.0);
        for (std::size_t k = 0; k < m_n; ++k)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < m_n; ++i)
                sum += m_j(i, k) * v[i];
            result[k] = sum;
        }
        return result;
    }

    [[nodiscard]] double rowResidual(std::size_t row, const std::vector<double>& x) const
    {
        double value = -m_d[row];
        for (std::size_t i = 0; i < m_n; ++i)
            value += m_c[row * m_n + i] * x[i];
        return value;
    }

    [[nodiscard]] double rowScale(std::size_t row, const std::vector<double>& x) const
    {
        double normal = 0.0;
        double size = 0.0;
        for (std::size_t i = 0; i < m_n; ++i)
        {
            normal = std::max(normal, std::abs(m_c[row * m_n + i]));
            size = std::max(size, std::abs(x[i]));
        }
        return std::max({1.0, std::abs(m_d[row]), normal * size});
    }

    // r = R⁻¹·d₁ over the active rows (back substitution).
    [[nodiscard]] std::vector<double> solveR(const std::vector<double>& dvec) const
    {
        const std::size_t q = m_active.size();
        std::vector<double> r(q, 0.0);
        for (std::size_t k = q; k-- > 0;)
        {
            double value = dvec[k];
            for (std::size_t j = k + 1; j < q; ++j)
                value -= m_r(k, j) * r[j];
            r[k] = value / m_r(k, k);
        }
        return r;
    }

    // Adds row p: Givens rotations of the columns q … n−1 of J that bring d(q+1 … n−1) to zero, then d(0 … q) is the
    // new column of R.
    void addRow(std::vector<double>& dvec, std::size_t p, double multiplier)
    {
        const std::size_t q = m_active.size();
        for (std::size_t j = m_n - 1; j > q; --j)
        {
            const double h = std::hypot(dvec[j - 1], dvec[j]);
            if (h == 0.0)
                continue;
            const double cosine = dvec[j - 1] / h;
            const double sine = dvec[j] / h;
            dvec[j - 1] = h;
            dvec[j] = 0.0;
            for (std::size_t i = 0; i < m_n; ++i)
            {
                const double left = m_j(i, j - 1);
                const double right = m_j(i, j);
                m_j(i, j - 1) = cosine * left + sine * right;
                m_j(i, j) = -sine * left + cosine * right;
            }
        }
        for (std::size_t k = 0; k <= q; ++k)
            m_r(k, q) = dvec[k];
        m_active.push_back(p);
        m_multipliers.push_back(multiplier);
    }

    // Drops the active row of position `position`: its column of R removed, the Hessenberg R made triangular again by
    // Givens rotations of its rows, the same rotations applied to the columns of J.
    void dropRow(std::size_t position)
    {
        const std::size_t q = m_active.size();
        for (std::size_t column = position; column + 1 < q; ++column)
            for (std::size_t k = 0; k < q; ++k)
                m_r(k, column) = m_r(k, column + 1);
        for (std::size_t k = 0; k < q; ++k)
            m_r(k, q - 1) = 0.0;
        m_active.erase(m_active.begin() + static_cast<std::ptrdiff_t>(position));
        m_multipliers.erase(m_multipliers.begin() + static_cast<std::ptrdiff_t>(position));
        for (std::size_t k = position; k + 1 < q; ++k)
        {
            const double h = std::hypot(m_r(k, k), m_r(k + 1, k));
            if (h == 0.0)
                continue;
            const double cosine = m_r(k, k) / h;
            const double sine = m_r(k + 1, k) / h;
            for (std::size_t column = k; column + 1 < q; ++column)
            {
                const double upper = m_r(k, column);
                const double lower = m_r(k + 1, column);
                m_r(k, column) = cosine * upper + sine * lower;
                m_r(k + 1, column) = -sine * upper + cosine * lower;
            }
            for (std::size_t i = 0; i < m_n; ++i)
            {
                const double left = m_j(i, k);
                const double right = m_j(i, k + 1);
                m_j(i, k) = cosine * left + sine * right;
                m_j(i, k + 1) = -sine * left + cosine * right;
            }
        }
    }

    QuadraticProgramSolution& finish(QuadraticProgramSolution& solution, std::vector<double>& x, std::size_t steps)
    {
        solution.iterations = steps;
        solution.multipliers.assign(m_m, 0.0);
        for (std::size_t k = 0; k < m_active.size(); ++k)
            solution.multipliers[m_active[k]] = std::max(m_multipliers[k], 0.0);
        solution.active = m_active;
        double objective = 0.0;
        for (std::size_t i = 0; i < m_n; ++i)
        {
            double gx = 0.0;
            for (std::size_t k = 0; k < m_n; ++k)
                gx += m_g[i * m_n + k] * x[k];
            objective += x[i] * (0.5 * gx + m_a[i]);
        }
        solution.objective = objective;
        solution.x = std::move(x);
        return solution;
    }

    std::size_t m_n;
    std::size_t m_m;
    std::span<const double> m_g;
    std::span<const double> m_a;
    std::span<const double> m_c;
    std::span<const double> m_d;
    base::MemoryBudget& m_budget;
    Square m_j;
    Square m_r;
    std::vector<std::size_t> m_active;
    std::vector<double> m_multipliers;
};

} // namespace

QuadraticProgramSolution QuadraticProgram::solve(std::span<const double> g, std::span<const double> a,
                                                 std::span<const double> c, std::span<const double> d,
                                                 base::MemoryBudget& budget)
{
    const std::size_t n = a.size();
    if (n == 0 || g.size() != n * n || c.size() != d.size() * n)
        throw base::OptimisationError(base::concatenate(
            {"quadratic program: expected G of ", std::to_string(n), " × ", std::to_string(n), " values and C of ",
             std::to_string(d.size()), " × ", std::to_string(n), " values (n > 0), found ", std::to_string(g.size()),
             " and ", std::to_string(c.size()), "; give G and C row by row for at least one variable"}));
    DualActiveSet method(g, a, c, d, budget);
    QuadraticProgramSolution solution = method.run();
    return solution;
}

} // namespace oos::design
