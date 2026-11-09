#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::linalg
{

class SupernodalFactor;

// The plan of the solves with a supernodal factor (internal to linalg/supernodal, FactorSolve): the
// supernodes grouped into units, each with the rows its supernodes touch, so that a sweep gathers the right-hand sides
// of a unit's rows once into a block stored by rows, runs the unit's supernodes on that block, and writes the rows
// back once, instead of gathering and scattering the rows of every supernode (SolvePlan's "panels grouped for the
// solve").
//
// A unit is a range of consecutive supernodes [first, last] (in the postorder of the factor) whose rows, the union of
// their panels' rows, number at most UNIT_ROWS, or one supernode with more rows than that or a large panel
// (SINGLETON_PANEL_VALUES). Its rows are numbered in
// its block: first the rows its supernodes eliminate, supernode after supernode in pivot order, then the other rows
// (of the ancestors, or the trailing rows of a partial factor) in increasing equation order. Since a sweep runs the
// supernodes of a unit in their order, and no other unit touches the rows of a unit while it runs, each value of a
// row receives its terms in the order of the sweep in supernode order: the units change no bit.
//
// With the independent subtrees of the factorisation and no delayed pivot (FactorSolve's sweeps by subtrees), the
// units are cut inside each subtree, the supernodes above them (the top) being units of their own; the rows of a
// subtree's unit beyond the subtree are rows of its accumulator (the update rows of the subtree's root, in their
// order). Otherwise (one subtree, or delayed pivots) the units cover the supernodes in order, with no accumulator.
//
// Thread-safety: immutable once built. Memory: budgeted (estimateMemoryBytes bounds it before the factorisation).
class SolvePlan
{
public:
    // The rows of a unit's block: 1 024 rows of a tile of 128 right-hand sides take 1 MB, which stays in the L2 of a
    // core of the reference PC (1.25 MB) while the unit's supernodes are swept; a unit of that size holds a subtree of
    // some 700 equations of a shell model with its boundary rows.
    // PERFORMANCE: one gather and one scatter per unit instead of one per supernode (the rows of a shell's supernodes
    // were gathered about five times each: 5.0 panel rows per equation on the 1 M DOF shell).
    // Measured: units of 512 to 3 072 rows within 5 % of each other as long as the
    // block stays near 1 MB; blocks of 4 MB lose 10 to 15 %.
    static constexpr std::int32_t UNIT_ROWS = 1024;
    // A supernode with this many panel values or more is a unit alone, so that FactorSolve can share its rows between
    // threads (a panel of 256 × 256: some 16 M multiply-adds per tile of 256 columns, against the few microseconds a
    // join of the threads costs).
    static constexpr std::int64_t SINGLETON_PANEL_VALUES = std::int64_t{1} << 16;

    struct Unit
    {
        std::int32_t firstSupernode = 0;
        std::int32_t lastSupernode = 0;
        std::int64_t firstRow = 0;           // in rows() and accumulatorRows()
        std::int32_t rowCount = 0;           // the rows of its block
        std::int32_t eliminatedRowCount = 0; // the first rows of its block, eliminated by its supernodes
        std::int32_t subtree = NO_SUBTREE;   // the subtree of the factorisation that holds it, or NO_SUBTREE
    };

    struct Subtree
    {
        std::int32_t firstUnit = 0;
        std::int32_t endUnit = 0;
        std::int64_t firstOutsideRow = 0; // in outsideRows()
        std::int32_t outsideRowCount = 0;
    };

    static constexpr std::int32_t NO_SUBTREE = -1;
    static constexpr std::int32_t NO_PARENT = -1;
    // An entry of accumulatorRows() for a row read and written in the right-hand sides themselves.
    static constexpr std::int32_t IN_RIGHT_HAND_SIDES = -1;

    SolvePlan() = default;
    // The plan of a factor whose panels are final. parents[s]: the supernode whose front received the update matrix of
    // s, or NO_PARENT.
    SolvePlan(const SupernodalFactor& factor, std::span<const std::int32_t> parents, base::MemoryBudget& budget);

    // An upper bound of the bytes the plan of a factor of `supernodeCount` supernodes, `panelRowCount` panel rows in
    // all and `dimension` equations reserves, its temporaries included.
    [[nodiscard]] static std::uint64_t estimateMemoryBytes(std::int64_t supernodeCount, std::int64_t panelRowCount,
                                                           std::int64_t dimension) noexcept;

    // True when the sweeps run by the subtrees of the factorisation (the class comment).
    [[nodiscard]] bool isByTree() const noexcept { return m_isByTree; }
    [[nodiscard]] std::span<const Unit> units() const noexcept { return m_units; }
    [[nodiscard]] std::span<const Subtree> subtrees() const noexcept { return m_subtrees; }
    // The equations of a unit's rows, in the order of its block.
    [[nodiscard]] std::span<const std::int32_t> rows(const Unit& unit) const noexcept
    {
        return m_unitRows.span().subspan(static_cast<std::size_t>(unit.firstRow),
                                         static_cast<std::size_t>(unit.rowCount));
    }
    // Per row of a unit: its row in the accumulator of the unit's subtree, or IN_RIGHT_HAND_SIDES.
    [[nodiscard]] std::span<const std::int32_t> accumulatorRows(const Unit& unit) const noexcept
    {
        return m_accumulatorRows.span().subspan(static_cast<std::size_t>(unit.firstRow),
                                                static_cast<std::size_t>(unit.rowCount));
    }
    // The equations of a subtree's accumulator rows (the update rows of its root), increasing.
    [[nodiscard]] std::span<const std::int32_t> outsideRows(const Subtree& subtree) const noexcept
    {
        return m_outsideRows.span().subspan(static_cast<std::size_t>(subtree.firstOutsideRow),
                                            static_cast<std::size_t>(subtree.outsideRowCount));
    }
    // The rows of a supernode's panel in its unit's block, one per panel row.
    [[nodiscard]] std::span<const std::int32_t> localRows(std::int32_t supernode) const noexcept
    {
        const auto at = static_cast<std::size_t>(supernode);
        return m_localRows.span().subspan(static_cast<std::size_t>(m_localRowOffset[at]),
                                          static_cast<std::size_t>(m_localRowOffset[at + 1] - m_localRowOffset[at]));
    }
    [[nodiscard]] std::int32_t parent(std::int32_t supernode) const noexcept
    {
        return m_parent[static_cast<std::size_t>(supernode)];
    }
    // The most rows of a unit's block, and the most accumulator rows of a subtree (the scratch of a sweep).
    [[nodiscard]] std::int32_t largestUnitRowCount() const noexcept { return m_largestUnitRowCount; }
    [[nodiscard]] std::int64_t outsideRowCount() const noexcept
    {
        return static_cast<std::int64_t>(m_outsideRows.size());
    }
    // Whether the plan was made for a factor of the shape of `factor` with these parents: the same supernodes, panel
    // row counts, parents and subtrees (a plan kept from an earlier factorisation of the same symbolic factor, step
    // 10.13; the key of the cache that keeps it guarantees it, this checks it).
    [[nodiscard]] bool isFor(const SupernodalFactor& factor, std::span<const std::int32_t> parents) const noexcept;

private:
    bool m_isByTree = false;
    // The vectors of units and subtrees are reserved here (declared first: reserved before, given back after).
    base::MemoryReservation m_unitReservation;
    std::vector<Unit> m_units;       // at most one per supernode
    std::vector<Subtree> m_subtrees; // one per subtree of the factorisation
    base::BudgetedBuffer<std::int32_t> m_unitRows;
    base::BudgetedBuffer<std::int32_t> m_accumulatorRows;
    base::BudgetedBuffer<std::int32_t> m_outsideRows;
    base::BudgetedBuffer<std::int64_t> m_localRowOffset; // supernodes + 1
    base::BudgetedBuffer<std::int32_t> m_localRows;
    base::BudgetedBuffer<std::int32_t> m_parent;
    std::int32_t m_largestUnitRowCount = 0;
};

} // namespace oos::linalg
