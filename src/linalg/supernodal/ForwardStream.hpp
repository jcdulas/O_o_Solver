#pragma once

#include "SolveSweeps.hpp"
#include "StreamedFactorisation.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::linalg
{

// FactorSolve's forward sweep run on a streamed factorisation (internal to linalg/supernodal): each unit of
// the solve plan swept as soon as the factorisation has made its panels, so that no factor is held. The steps are those
// of FactorSolve::eliminateForward, in the same order for every value (the subtrees of the cut each into its
// accumulator, the accumulators added in subtree order, then the top supernodes in increasing order; or every unit in
// order for a plan not by tree), on the same tiles' active supernodes: the right-hand sides get the bits
// eliminateForward gives with the factor of the same symbolic factor, matrix and kernel family.
//
// The right-hand sides are swept in tiles of FactorSolve::RHS_TILE_COLUMNS columns, every tile while the unit's panels
// are resident; each tile has its accumulators (the update rows of the subtrees' roots, budgeted) until afterSubtrees.
//
// Thread-safety: consumeUnit runs on the subtrees' tasks at once (each unit writes its own rows and its subtree's
// accumulator). Ownership: references the factorisation, the right-hand sides and the kernels, which outlive it.
class ForwardStream final : public StreamedFactorisation::UnitConsumer
{
public:
    // rightHandSides: the factor's dimension() rows. The trailing rows are written without −0 at once.
    ForwardStream(const StreamedFactorisation& factorisation, DenseMatrixView rightHandSides,
                  const DenseKernels& kernels, base::MemoryBudget& budget, const base::CancellationToken* cancellation);

    void consumeUnit(std::size_t unit) override;
    void afterSubtrees() override;

    // The bytes of the tiles' accumulators for `columns` right-hand sides (for the estimates).
    [[nodiscard]] static std::uint64_t accumulatorBytes(const SolvePlan& plan, std::size_t columns,
                                                        std::size_t rowLanes) noexcept;

private:
    struct TileState
    {
        sweeps::Tile tile;
        std::vector<std::uint8_t> isActive; // per supernode
        base::BudgetedBuffer<double> accumulators;
    };

    [[nodiscard]] sweeps::SweepContext contextOf() const noexcept;

    const StreamedFactorisation& m_factorisation;
    const DenseKernels& m_kernels;
    const base::CancellationToken* m_cancellation;
    std::vector<TileState> m_tiles; // bounded by the columns over a tile's width
    sweeps::BlockPool m_blocks;
};

} // namespace oos::linalg
