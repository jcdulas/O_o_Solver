#include "HessianBlocks.hpp"

#include <utility>

namespace oos::design::detail
{

namespace
{

std::uint64_t bytesOf(const std::vector<HessianBlocks::Block>& blocks)
{
    std::uint64_t bytes = blocks.capacity() * sizeof(HessianBlocks::Block);
    for (const HessianBlocks::Block& block : blocks)
        bytes += block.dofs.capacity() * sizeof(base::DofIndex) +
                 (block.displacement.capacity() + block.mixed.capacity()) * sizeof(double);
    return bytes;
}

} // namespace

HessianBlocks::HessianBlocks(std::vector<Block> blocks, base::MemoryBudget& budget)
    : m_blocks(std::move(blocks)), m_reservation(budget.reserve(bytesOf(m_blocks), "Hessian blocks"))
{
}

void HessianBlocks::apply(std::span<const std::vector<double>> moves, std::span<const double> fieldMoves,
                          std::vector<std::vector<double>>& loads, std::vector<double>& terms) const
{
    for (const Block& block : m_blocks)
    {
        const std::vector<double>& move = moves[block.subcase];
        std::vector<double>& load = loads[block.subcase];
        const std::size_t n = block.dofs.size();
        const double fieldMove = block.field == NO_FIELD ? 0.0 : fieldMoves[block.field];
        double term = block.field == NO_FIELD ? 0.0 : block.fieldTerm * fieldMove;
        for (std::size_t i = 0; i < n; ++i)
        {
            double sum = 0.0;
            if (!block.displacement.empty() && !move.empty())
                for (std::size_t j = 0; j < n; ++j)
                    sum += block.displacement[i * n + j] * move[static_cast<std::size_t>(block.dofs[j])];
            if (!block.mixed.empty())
            {
                sum += block.mixed[i] * fieldMove;
                if (!move.empty())
                    term += block.mixed[i] * move[static_cast<std::size_t>(block.dofs[i])];
            }
            load[static_cast<std::size_t>(block.dofs[i])] += sum;
        }
        if (block.field != NO_FIELD)
            terms[block.field] += term;
    }
}

} // namespace oos::design::detail
