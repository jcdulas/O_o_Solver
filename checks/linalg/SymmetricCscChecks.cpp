// The checks of the products of a symmetric sparse matrix (linalg::SymmetricCsc): the product of a block of vectors,
// made by groups of columns in one pass over the terms, and the product by the matrix without its zero terms, have the
// bits of the product of one vector.

#include "CheckRunner.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using oos::linalg::DenseMatrix;
using oos::linalg::SymmetricCsc;

namespace
{

constexpr std::uint64_t BUDGET_BYTES = std::uint64_t{1} << 28;
constexpr std::uint32_t SEED = 7;
constexpr std::int32_t DIMENSION = 157;
// A column keeps its diagonal and each row below it with this chance; a kept term is zero with the second chance
// (the zero terms a structural pattern stores).
constexpr double TERM_CHANCE = 0.08;
constexpr double ZERO_CHANCE = 0.6;

double uniform(std::mt19937& random)
{
    return static_cast<double>(random()) / 4294967296.0;
}

SymmetricCsc randomMatrix(oos::base::MemoryBudget& budget)
{
    std::mt19937 random(SEED);
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> rows;
    std::vector<double> values;
    for (std::int32_t j = 0; j < DIMENSION; ++j)
    {
        for (std::int32_t i = j; i < DIMENSION; ++i)
        {
            if (i != j && uniform(random) >= TERM_CHANCE)
                continue;
            rows.push_back(i);
            values.push_back(uniform(random) < ZERO_CHANCE ? 0.0 : 2.0 * uniform(random) - 1.0);
        }
        offsets.push_back(static_cast<std::int64_t>(rows.size()));
    }
    return SymmetricCsc::fromCompressedColumns(DIMENSION, offsets, rows, values, budget);
}

DenseMatrix randomBlock(std::size_t columns, oos::base::MemoryBudget& budget)
{
    std::mt19937 random(SEED + 1);
    DenseMatrix block(budget, static_cast<std::size_t>(DIMENSION), columns, "check: block");
    for (std::size_t c = 0; c < columns; ++c)
        for (std::size_t r = 0; r < block.rows(); ++r)
            block(r, c) = 2.0 * uniform(random) - 1.0;
    return block;
}

// Whether column `column` of the block product y has the bits of the product of that column of x alone by
// `reference`.
bool hasBitsOfColumnProduct(const SymmetricCsc& reference, const DenseMatrix& x, const DenseMatrix& y,
                            std::size_t column)
{
    const auto dimension = static_cast<std::size_t>(DIMENSION);
    std::vector<double> alone(dimension);
    reference.multiply({x.view().pointer(0, column), dimension}, alone);
    return std::memcmp(alone.data(), y.view().pointer(0, column), dimension * sizeof(double)) == 0;
}

} // namespace

CHECK_CASE("symmetric CSC: a block product has the bits of the products of its columns")
{
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const SymmetricCsc matrix = randomMatrix(budget);
    // Fewer columns than a group, a whole group, and more than two groups.
    for (const std::size_t columns : {std::size_t{1}, std::size_t{3}, SymmetricCsc::MULTIPLY_GROUP_COLUMNS,
                                      2 * SymmetricCsc::MULTIPLY_GROUP_COLUMNS + 3})
    {
        const DenseMatrix x = randomBlock(columns, budget);
        DenseMatrix y(budget, x.rows(), columns, "check: product");
        matrix.multiply(x.view(), y.view());
        for (std::size_t c = 0; c < columns; ++c)
            runner.expect(hasBitsOfColumnProduct(matrix, x, y, c),
                          "column " + std::to_string(c) + " of " + std::to_string(columns));
    }
}

CHECK_CASE("symmetric CSC: the matrix without its zero terms gives the bits of the same products")
{
    oos::base::MemoryBudget budget(BUDGET_BYTES);
    const SymmetricCsc matrix = randomMatrix(budget);
    const SymmetricCsc compact = SymmetricCsc::withoutZeroTerms(matrix, budget);
    runner.expect(matrix.zeroTermCount() > 0, "the matrix of the check stores zero terms");
    runner.expect(compact.zeroTermCount() == 0, "no zero term is kept");
    runner.expect(compact.nonZeroCount() == matrix.nonZeroCount() - matrix.zeroTermCount(),
                  "every other term is kept");
    const std::size_t columns = SymmetricCsc::MULTIPLY_GROUP_COLUMNS + 1;
    const DenseMatrix x = randomBlock(columns, budget);
    DenseMatrix y(budget, x.rows(), columns, "check: product");
    compact.multiply(x.view(), y.view());
    for (std::size_t c = 0; c < columns; ++c)
        runner.expect(hasBitsOfColumnProduct(matrix, x, y, c), "column " + std::to_string(c));
}
