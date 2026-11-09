#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace oos::assembly
{

// Half a unit of the last written digit of a load value. O_o_Reader records the significant digits of each number
// in the significantDigits column of the load card tables (ModelSnapshot::loadCardTables).
//
// The precision is looked up by load set and card type: the cards of one type of one set may have been written in
// different precisions, and the finest of them is taken, so that no value is assumed rounded more than it may have
// been. A LoadCombination set takes the precision of the sets it combines; the rounding of its factors is not counted,
// which also assumes
// a finer precision. Assuming a finer precision than the true one only makes the cases less often reconstructed within
// it: they join the basis, and no result is wrong (LoadBasis checks every case).
//
// Thread-safety: immutable. Ownership: borrows the snapshot (the card names of its load card tables).
class FieldPrecision
{
public:
    explicit FieldPrecision(const input::ModelSnapshot& snapshot);

    // Half a unit of the last digit of `value` as the cards `card` of load set `setId` wrote it (the finest of their
    // precisions; 0 for a zero value), or nothing when the snapshot does not give the written digits of these cards.
    [[nodiscard]] std::optional<double> halfUnitOf(base::EntityId setId, std::string_view card, double value) const;

    // Half a unit of the last significant digit (0 for a zero value: "0." is exact).
    [[nodiscard]] static double halfUnit(double value, std::int32_t significantDigits);

private:
    // The precision of the cards of one type of one load set.
    struct Entry
    {
        base::EntityId setId = 0;
        std::string_view card;
        std::int32_t significantDigits = 0;
        bool isKnown = true; // false when the written digits of one of the cards are not known
    };

    // Combines the precisions of `from` and `into` (same set and card).
    static void merge(Entry& into, const Entry& from) noexcept;
    // The order of m_entries: by set, then card.
    [[nodiscard]] static bool isBefore(const Entry& a, const Entry& b) noexcept;

    std::vector<Entry> m_entries; // sorted by (setId, card), one per key; bounded by the load sets × card types
};

} // namespace oos::assembly
