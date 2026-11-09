#include "oos/design/sensitivity/AdjointSensitivity.hpp"

#include "oos/base/threading/ParallelFor.hpp"
#include "oos/design/sensitivity/AdjointColumns.hpp"
#include "oos/design/sensitivity/SolutionReads.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <vector>

namespace oos::design
{

namespace
{

// The pieces per thread of the right-hand sides of solveTogether(), and the fewest columns of a piece: a piece
// allocates a g-set vector of its own.
constexpr std::size_t PIECES_PER_THREAD = 2;
constexpr std::size_t COLUMNS_PER_PIECE = 4;

} // namespace

std::int64_t AdjointSensitivity::solve(const SubcaseSolve& context, std::span<const ItemKey> items,
                                       LinearisedItems& linearised)
{
    const SensitivitySystem& system = *context.system;
    const PseudoLoads& loads = *context.loads;
    std::vector<ItemKey> solved; // bounded by the items of the subcase
    for (const ItemKey& key : items)
        if (linearised.at(key).readsDisplacements())
            solved.push_back(key);
    if (solved.empty())
        return 0;
    const auto gSize = static_cast<std::size_t>(system.dofMap().gSetSize());
    const auto equations = static_cast<std::size_t>(system.equationCount());
    base::BudgetedBuffer<double> vectorG(*context.budget, gSize, "adjoint sensitivity weights");
    base::BudgetedBuffer<double> column(*context.budget, equations, "adjoint sensitivity equations");
    std::vector<double> products(loads.variableCount()); // bounded by the design variables
    // The condensed path recovers the adjoint vectors in the driven units only.
    const base::BudgetedBuffer<std::uint8_t> reads =
        SolutionReads::ofPseudoLoads(system, loads, context.subcase, *context.budget);
    std::int64_t solves = 0;
    for (std::size_t first = 0; first < solved.size(); first += SENSITIVITY_BLOCK_COLUMNS)
    {
        const std::size_t count = std::min(SENSITIVITY_BLOCK_COLUMNS, solved.size() - first);
        linalg::DenseMatrix rightHandSides(*context.budget, equations, count, "adjoint sensitivity right-hand sides");
        for (std::size_t b = 0; b < count; ++b)
        {
            std::fill(vectorG.data(), vectorG.data() + gSize, 0.0);
            for (const auto& [dof, weight] : linearised.at(solved[first + b]).weights)
                vectorG[static_cast<std::size_t>(dof)] = weight;
            system.toEquations(vectorG.span(), column.span());
            for (std::size_t e = 0; e < equations; ++e)
                rightHandSides(e, b) = column[e];
        }
        if (equations > 0)
            system.solve(rightHandSides.view(), reads.span());
        solves += static_cast<std::int64_t>(count);
        for (std::size_t b = 0; b < count; ++b)
        {
            for (std::size_t e = 0; e < equations; ++e)
                column[e] = rightHandSides(e, b);
            system.fromEquations(column.span(), vectorG.span());
            loads.dotAll(context.subcase, vectorG.span(), products);
            std::vector<double>& gradient = linearised.gradientOf(solved[first + b]);
            for (std::size_t j = 0; j < products.size(); ++j)
                gradient[j] += products[j];
        }
    }
    return solves;
}

std::vector<std::int64_t> AdjointSensitivity::solveTogether(const SensitivitySystem& system, const PseudoLoads& loads,
                                                            std::span<const SubcaseItems> subcases,
                                                            LinearisedItems& linearised, base::MemoryBudget& budget,
                                                            base::ThreadPool* pool)
{
    std::vector<std::int64_t> solves(subcases.size(), 0); // bounded by the subcases
    std::vector<std::size_t> indices;                     // bounded by the subcases
    for (const SubcaseItems& part : subcases)
        indices.push_back(part.subcase);
    AdjointColumns columns(system, loads, indices, false, budget, pool);
    if (columns.readsEvery())
    {
        // Inertia relief: the products read the whole adjoint vector, subcase by subcase as before.
        for (std::size_t k = 0; k < subcases.size(); ++k)
            solves[k] = solve({&system, &loads, subcases[k].subcase, &budget}, subcases[k].items, linearised);
        return solves;
    }
    // The columns: per subcase its items that read the displacements, in order.
    struct Column
    {
        std::size_t part = 0;
        ItemKey key;
    };
    std::vector<Column> all; // bounded by the items
    for (std::size_t k = 0; k < subcases.size(); ++k)
        for (const ItemKey& key : subcases[k].items)
            if (linearised.at(key).readsDisplacements())
            {
                all.push_back({k, key});
                ++solves[k];
            }
    if (all.empty())
        return solves;
    const auto g = static_cast<std::size_t>(system.dofMap().gSetSize());
    const auto equations = static_cast<std::size_t>(system.equationCount());
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    linalg::DenseMatrix rightHandSides(budget, equations, std::min(SENSITIVITY_BLOCK_COLUMNS, all.size()),
                                       "adjoint sensitivity right-hand sides");
    const std::span<const base::DofIndex> readDofs = columns.readDofs();
    for (std::size_t first = 0; first < all.size(); first += SENSITIVITY_BLOCK_COLUMNS)
    {
        const std::size_t count = std::min(SENSITIVITY_BLOCK_COLUMNS, all.size() - first);
        const linalg::DenseMatrixView view = rightHandSides.view().block(0, 0, equations, count);
        // The columns placed in the solve by the first equation of their items (AdjointColumns::solveSlots).
        std::vector<std::int32_t> firstEquations(count); // bounded by a block
        for (std::size_t c = 0; c < count; ++c)
            firstEquations[c] = columns.firstEquation(linearised.at(all[first + c].key).weights);
        const std::vector<std::size_t> slot = AdjointColumns::solveSlots(firstEquations);
        const auto columnOf = [&](std::size_t c)
        { return std::span<double>(view.data + slot[c] * view.leadingDimension, equations); };
        // The right-hand sides on the pool, each from its item's entries (its own g-set vector per piece): the values
        // solve() makes with toEquations().
        const std::size_t piece = std::max(COLUMNS_PER_PIECE, base::pieceCount(count, PIECES_PER_THREAD * threads));
        base::parallelFor(pool, count, piece,
                          [&](std::size_t begin, std::size_t end)
                          {
                              std::vector<double> scratch(g, 0.0); // one per piece
                              std::vector<base::DofIndex> touched; // the entries of an item
                              for (std::size_t c = begin; c < end; ++c)
                              {
                                  for (const auto& [dof, weight] : linearised.at(all[first + c].key).weights)
                                  {
                                      scratch[static_cast<std::size_t>(dof)] = weight;
                                      touched.push_back(dof);
                                  }
                                  columns.reduceToEquations(scratch, touched, columnOf(c));
                              }
                          });
        if (equations > 0)
            system.solve(view);
        std::vector<std::size_t> subcaseOf(count); // bounded by a block
        std::vector<double*> destinations(count);  // the same
        for (std::size_t c = 0; c < count; ++c)
        {
            subcaseOf[c] = subcases[all[first + c].part].subcase;
            destinations[c] = linearised.gradientOf(all[first + c].key).data();
        }
        columns.addProducts(subcaseOf, destinations,
                            [&](std::size_t firstRead, std::size_t endRead, base::DofIndex firstDof,
                                std::span<const std::size_t> pieceColumns, double* block, std::size_t stride)
                            {
                                for (std::size_t k = 0; k < pieceColumns.size(); ++k)
                                {
                                    const std::span<const double> solved = columnOf(pieceColumns[k]);
                                    for (std::size_t r = firstRead; r < endRead; ++r)
                                        block[static_cast<std::size_t>(readDofs[r] - firstDof) * stride + k] =
                                            columns.valueAtRead(solved, r);
                                }
                            });
    }
    return solves;
}

void AdjointSensitivity::combineArguments(std::span<const ItemKey> equationItems, LinearisedItems& linearised)
{
    for (const ItemKey& key : equationItems)
    {
        const ItemLinearisation& item = linearised.at(key);
        std::vector<double> gradient(linearised.gradientOf(key).size(), 0.0);
        addRowTo(item.ownExplicit, gradient);
        for (const auto& [argument, weight] : item.arguments)
        {
            const std::vector<double>& argumentGradient = linearised.gradientOf(argument);
            for (std::size_t j = 0; j < gradient.size(); ++j)
                gradient[j] += weight * argumentGradient[j];
        }
        linearised.gradientOf(key) = std::move(gradient);
    }
}

} // namespace oos::design
