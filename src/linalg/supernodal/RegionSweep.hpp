#pragma once

// Internal to linalg/supernodal: the backward sweep by regions of a factorisation that keeps no factor, shared
// by the recovery of a unit (PartialFactor::recoverWithoutFactor) and the complete solve (StreamedSolve), and the
// plan of its regions from the symbolic structure alone (the estimates).

#include "StreamedFactorisation.hpp"

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace oos::linalg::regions
{

// The first supernode of the largest suffix of [0, count) whose panels fit regionValues (count when none does): the top
// of the tree kept from the forward run (a suffix of a postorder holds the ancestors of its members).
template <typename ValuesOf>
[[nodiscard]] std::int32_t topFirst(std::int32_t count, std::int64_t regionValues, ValuesOf valuesOf)
{
    std::int32_t first = count;
    std::int64_t values = 0;
    while (first > 0 && values + valuesOf(first - 1) <= regionValues)
    {
        --first;
        values += valuesOf(first);
    }
    return first;
}

// Visits the regions below the top [top, count) in the order of the sweep: visit(kept, root), the supernodes
// [kept, root] kept from a factorisation of the subtree of `root` (its supernodes [subtreeFirst(root), root]); the
// largest suffix of them that fits regionValues, at least the root. The rest of the subtree is a forest whose ancestors
// are done, taken the same way after the other roots of its forest (a stack bounded by the depth of the tree).
template <typename ValuesOf, typename SubtreeFirst, typename Visit>
void forEach(std::int32_t top, std::int64_t regionValues, ValuesOf valuesOf, SubtreeFirst subtreeFirst, Visit visit)
{
    std::vector<std::pair<std::int32_t, std::int32_t>> forests; // bounded by the depth of the tree
    if (top > 0)
        forests.emplace_back(0, top - 1);
    while (!forests.empty())
    {
        const auto [low, high] = forests.back();
        forests.pop_back();
        for (std::int32_t root = high; root >= low;)
        {
            const std::int32_t first = subtreeFirst(root);
            std::int32_t kept = root;
            std::int64_t values = valuesOf(root);
            while (kept > first && values + valuesOf(kept - 1) <= regionValues)
            {
                --kept;
                values += valuesOf(kept);
            }
            visit(kept, root, values);
            if (kept > first)
                forests.emplace_back(first, kept - 1);
            root = first - 1;
        }
    }
}

// The forward sweep of `loadsToSolution` while `streamed` factorises (the columns with a nonzero row of the eliminated
// part only: the others are left as they are), the top kept; the trailing rows set to trailingValues (none without a
// trailing set); the diagonal from the pivots; the backward sweep of the top, then of each region factorised again.
// K* assembled into schurComplement when given (a partial factorisation). The bits of FactorSolve's forward, diagonal
// and backward sweeps over the eliminated part. Returns the regions swept (the factorisation's diagnostics are
// streamed's). Throws as StreamedFactorisation::run.
RegionSweepCost sweep(StreamedFactorisation& streamed, DenseMatrixView loadsToSolution,
                      ConstDenseMatrixView trailingValues, const DenseKernels& kernels, const FactorOptions& options,
                      base::MemoryBudget& budget, std::int64_t regionValues, DenseMatrix* schurComplement);

} // namespace oos::linalg::regions
