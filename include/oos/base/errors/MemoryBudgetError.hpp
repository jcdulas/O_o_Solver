#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace oos::base
{

// A part of the memory budget held under a label ("factor of unit 12"), as a refusal reports it.
struct MemoryConsumer
{
    std::string label;
    std::uint64_t bytes = 0;
};

// What a refused reservation asked for and what the budget had left.
struct MemoryShortfall
{
    std::uint64_t requestedBytes = 0;
    std::uint64_t availableBytes = 0;
    std::uint64_t limitBytes = 0;
    std::vector<MemoryConsumer> largestConsumers; // largest first
};

// An allocation would exceed the memory budget (an analysis that cannot fit is refused before it starts).
// Raised before anything is allocated; carries the estimate and the largest consumers, which the message also names.
class MemoryBudgetError : public SolverError
{
public:
    MemoryBudgetError(const std::string& message, MemoryShortfall shortfall)
        : SolverError(message), m_shortfall(std::move(shortfall))
    {
    }

    [[nodiscard]] const MemoryShortfall& shortfall() const noexcept { return m_shortfall; }

private:
    MemoryShortfall m_shortfall;
};

} // namespace oos::base
