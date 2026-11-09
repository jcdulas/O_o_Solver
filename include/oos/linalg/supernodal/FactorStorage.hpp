#pragma once

#include "oos/base/memory/AlignedBuffer.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>

namespace oos::linalg
{

// The memory of a factor's panels, given back by a factor no longer needed (SupernodalFactor::releaseStorage) for a
// later factorisation of the same structure (FactorOptions::storage).
// A design loop factorises one symbolic factor at every point: a fresh panel buffer comes from the system as pages
// that each first write faults in (4 kB at a time: a million faults at 1 M DOFs), while this memory is mapped already.
// The values are whatever the last factor left: a factorisation writes every value it reads.
//
// Memory: outside any budget while it is held here (its holder reports it: StructureCache::bytes()); a factorisation
// that takes it reserves its bytes in its own budget, as a new allocation would be.
//
// Thread-safety: as a value. Ownership: owns its buffers; movable, not copyable.
class FactorStorage
{
public:
    FactorStorage() = default;

    [[nodiscard]] bool empty() const noexcept { return m_values.empty() && m_rows.empty(); }
    [[nodiscard]] std::uint64_t bytes() const noexcept
    {
        return static_cast<std::uint64_t>(m_values.size()) * sizeof(double) +
               static_cast<std::uint64_t>(m_rows.size()) * sizeof(std::int32_t);
    }
    // Whether a complete factorisation of `symbolic` takes it: its panels' values and rows have these sizes (the
    // symbolic shape, SymbolicFactor::panelValueCount and the rows of every supernode). O(supernodes).
    [[nodiscard]] bool fits(const SymbolicFactor& symbolic) const noexcept;

private:
    friend class SupernodalFactor;
    friend class MultifrontalDriver;

    base::AlignedBuffer<double> m_values;     // the panels' values
    base::AlignedBuffer<std::int32_t> m_rows; // the panels' row indices
};

} // namespace oos::linalg
