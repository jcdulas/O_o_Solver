#include "oos/assembly/loads/FieldPrecision.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <span>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

// The card that combines load sets (LOAD), and its column of the sets it combines (O_o_Reader: Li).
constexpr std::string_view COMBINATION_CARD = "LoadCombination";
constexpr const char* COMBINED_SETS_COLUMN = "termLoad";

// A LOAD set and one of the sets it combines.
struct Combination
{
    base::EntityId setId = 0;
    base::EntityId set = 0;
};

} // namespace

FieldPrecision::FieldPrecision(const input::ModelSnapshot& snapshot)
{
    std::vector<Combination> combinations;
    for (const input::CardTable& table : snapshot.loadCardTables())
    {
        const base::ContractReader& reader = table.reader();
        const std::span<const std::int32_t> setIds = table.ids();
        if (table.card() == COMBINATION_CARD)
        {
            const base::ListView<std::int32_t> combined = reader.lists<std::int32_t>(COMBINED_SETS_COLUMN);
            for (std::int64_t row = 0; row < table.rowCount(); ++row)
                for (const std::int32_t set : combined[row])
                    combinations.push_back({setIds[static_cast<std::size_t>(row)], set});
            continue;
        }
        const bool hasPrecision = reader.hasColumn("significantDigits");
        const auto digits =
            hasPrecision ? std::optional(reader.lists<std::uint8_t>("significantDigits", base::NullPolicy::Allow))
                         : std::nullopt;
        const base::ValidityView hasDigits = hasPrecision ? reader.validity("significantDigits") : base::ValidityView();
        // One entry per set of this card type (a card table may hold many cards per set, in any order).
        std::map<base::EntityId, Entry> bySet;
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            const base::EntityId setId = setIds[static_cast<std::size_t>(row)];
            Entry rowPrecision;
            rowPrecision.setId = setId;
            rowPrecision.card = table.card();
            if (digits && hasDigits.isValid(row))
            {
                // The most significant digits of the real fields (an integer field counts none).
                const std::span<const std::uint8_t> fieldDigits = (*digits)[row];
                const auto most = std::max_element(fieldDigits.begin(), fieldDigits.end());
                rowPrecision.significantDigits = most == fieldDigits.end() ? 0 : static_cast<std::int32_t>(*most);
                rowPrecision.isKnown = rowPrecision.significantDigits > 0;
            }
            else
                rowPrecision.isKnown = false;
            const auto [found, isNew] = bySet.try_emplace(setId, rowPrecision);
            if (!isNew)
                merge(found->second, rowPrecision);
        }
        for (const auto& [setId, entry] : bySet)
            m_entries.push_back(entry);
    }
    std::sort(m_entries.begin(), m_entries.end(), isBefore);
    // A LOAD set takes an entry per card type of each set it combines.
    std::map<std::pair<base::EntityId, std::string_view>, Entry> combined;
    for (const Combination& combination : combinations)
    {
        auto member = std::lower_bound(m_entries.begin(), m_entries.end(), combination.set,
                                       [](const Entry& entry, base::EntityId id) { return entry.setId < id; });
        for (; member != m_entries.end() && member->setId == combination.set; ++member)
        {
            Entry entry = *member;
            entry.setId = combination.setId;
            const auto [found, isNew] = combined.try_emplace({entry.setId, entry.card}, entry);
            if (!isNew)
                merge(found->second, entry);
        }
    }
    for (const auto& [key, entry] : combined)
    {
        const auto at = std::lower_bound(m_entries.begin(), m_entries.end(), entry, isBefore);
        if (at != m_entries.end() && at->setId == entry.setId && at->card == entry.card)
            merge(*at, entry); // a LOAD set that also holds cards of its own (unusual in a deck, but accepted)
        else
            m_entries.insert(at, entry);
    }
}

bool FieldPrecision::isBefore(const Entry& a, const Entry& b) noexcept
{
    return a.setId < b.setId || (a.setId == b.setId && a.card < b.card);
}

void FieldPrecision::merge(Entry& into, const Entry& from) noexcept
{
    into.significantDigits = std::max(into.significantDigits, from.significantDigits);
    into.isKnown = into.isKnown && from.isKnown;
}

std::optional<double> FieldPrecision::halfUnitOf(base::EntityId setId, std::string_view card, double value) const
{
    Entry key;
    key.setId = setId;
    key.card = card;
    const auto found = std::lower_bound(m_entries.begin(), m_entries.end(), key, isBefore);
    if (found == m_entries.end() || found->setId != setId || found->card != card || !found->isKnown)
        return std::nullopt;
    return halfUnit(value, found->significantDigits);
}

double FieldPrecision::halfUnit(double value, std::int32_t significantDigits)
{
    const double magnitude = std::abs(value);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude))
        return 0.0;
    const auto exponent = static_cast<std::int32_t>(std::floor(std::log10(magnitude)));
    return 0.5 * std::pow(10.0, exponent - significantDigits + 1);
}

} // namespace oos::assembly
