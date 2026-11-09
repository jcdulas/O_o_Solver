#include "oos/solutions/sol101/StaticLoadCases.hpp"

#include "oos/assembly/loads/CompactLoadMatrix.hpp"
#include "oos/assembly/loads/FieldPrecision.hpp"
#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace oos::solutions
{

namespace
{

using assembly::LoadFamily;

// Adds (row, value, precision) to a column being gathered: a repeated row sums its values and precisions (the
// precisions of rounded terms add up). The arrays are budgeted once: one position per compact row, and the entries of
// the largest column (a column has at most as many distinct rows as values), both counted by a first pass.
class ColumnBuilder
{
public:
    ColumnBuilder(std::int64_t rowCount, std::size_t largestColumn, base::MemoryBudget& budget)
        : m_position(budget, static_cast<std::size_t>(rowCount), "load column positions"),
          m_rows(budget, largestColumn, "load column rows"),
          m_values(budget, largestColumn, "load column values"),
          m_precisions(budget, largestColumn, "load column precisions")
    {
        std::fill(m_position.data(), m_position.data() + m_position.size(), -1);
    }

    void add(std::int64_t row, double value, double precision)
    {
        std::int64_t& position = m_position[static_cast<std::size_t>(row)];
        if (position < 0)
        {
            position = static_cast<std::int64_t>(m_size);
            m_rows[m_size] = row;
            m_values[m_size] = value;
            m_precisions[m_size] = precision;
            ++m_size;
            return;
        }
        m_values[static_cast<std::size_t>(position)] += value;
        m_precisions[static_cast<std::size_t>(position)] += precision;
    }

    // Appends the column gathered to `matrix`, its exact zeros left out (a value cancelled by two cards is no load);
    // the builder is reset.
    void appendTo(assembly::CompactLoadMatrix& matrix)
    {
        std::size_t kept = 0;
        for (std::size_t k = 0; k < m_size; ++k)
        {
            m_position[static_cast<std::size_t>(m_rows[k])] = -1;
            if (m_values[k] == 0.0)
                continue;
            m_rows[kept] = m_rows[k];
            m_values[kept] = m_values[k];
            m_precisions[kept] = m_precisions[k];
            ++kept;
        }
        matrix.appendColumn({m_rows.data(), kept}, {m_values.data(), kept}, {m_precisions.data(), kept});
        m_size = 0;
    }

private:
    base::BudgetedBuffer<std::int64_t> m_position; // per compact row: its place in the column, −1 when absent
    base::BudgetedBuffer<std::int64_t> m_rows;
    base::BudgetedBuffer<double> m_values;
    base::BudgetedBuffer<double> m_precisions;
    std::size_t m_size = 0;
};

// The rows of `count` items: the items equal under `isLess` share one row, and the rows are numbered in the order of
// the first item of each (so the numbering does not depend on how `isLess` orders unequal items, which may compare
// addresses). A sort of the item indices by (item, index), budgeted, in place of a map from item to row.
struct ItemRows
{
    base::BudgetedBuffer<std::int64_t> rowOfItem;
    std::int64_t rowCount = 0;
};

template <typename IsLess>
ItemRows rowsByFirstAppearance(std::size_t count, IsLess&& isLess, base::MemoryBudget& budget)
{
    ItemRows result{base::BudgetedBuffer<std::int64_t>(budget, count, "load rows of items"), 0};
    base::BudgetedBuffer<std::int64_t> order(budget, count, "load item order");
    std::iota(order.data(), order.data() + count, std::int64_t{0});
    std::sort(order.data(), order.data() + count,
              [&](std::int64_t left, std::int64_t right)
              {
                  if (isLess(left, right))
                      return true;
                  return !isLess(right, left) && left < right;
              });
    // groupOf[k]: the group of the item at sorted position k; the first item of a group is its smallest index.
    base::BudgetedBuffer<std::int64_t> groupOf(budget, count, "load item groups");
    base::BudgetedBuffer<std::int64_t> groupOfFirstItem(budget, count, "load item first of group");
    std::fill(groupOfFirstItem.data(), groupOfFirstItem.data() + count, -1);
    std::int64_t groups = 0;
    for (std::size_t k = 0; k < count; ++k)
    {
        if (k == 0 || isLess(order[k - 1], order[k]))
        {
            groupOfFirstItem[static_cast<std::size_t>(order[k])] = groups;
            ++groups;
        }
        groupOf[k] = groups - 1;
    }
    base::BudgetedBuffer<std::int64_t> rowOfGroup(budget, static_cast<std::size_t>(groups), "load rows of groups");
    for (std::size_t item = 0; item < count; ++item)
        if (groupOfFirstItem[item] >= 0)
            rowOfGroup[static_cast<std::size_t>(groupOfFirstItem[item])] = result.rowCount++;
    for (std::size_t k = 0; k < count; ++k)
        result.rowOfItem[static_cast<std::size_t>(order[k])] = rowOfGroup[static_cast<std::size_t>(groupOf[k])];
    return result;
}

// The identity of a distributed card value: its element and corner (slot), and its variant (card, CID, direction,
// BeamDistributedLoad type, scale and positions), on which the equivalent loads are linear in the value. The texts are
// views into the dictionaries of the load tables: one text, one address, so they compare by their address and length
// without a copy (the rows are numbered by first appearance, so the address order never reaches the results).
struct CardValueKey
{
    std::int32_t elementId = 0;
    std::int32_t slot = 0;
    std::uintptr_t card = 0; // the address of the text (std::less order of pointers, as integers)
    std::size_t cardSize = 0;
    std::int32_t frameId = 0;
    std::array<double, 3> direction{};
    std::uintptr_t loadType = 0;
    std::uintptr_t scaleType = 0;
    double x1 = 0.0;
    double x2 = 0.0;
    std::array<std::int32_t, 4> faceGrids{};
};

CardValueKey keyOf(const assembly::LoadVectorBuilder::CardValue& value)
{
    return {value.elementId,
            value.slot,
            reinterpret_cast<std::uintptr_t>(value.card.data()),
            value.card.size(),
            value.frameId,
            value.direction,
            reinterpret_cast<std::uintptr_t>(value.loadType.data()),
            reinterpret_cast<std::uintptr_t>(value.scaleType.data()),
            value.x1,
            value.x2,
            value.faceGrids};
}

auto tieOf(const CardValueKey& key)
{
    return std::tie(key.elementId, key.slot, key.card, key.cardSize, key.frameId, key.direction, key.loadType,
                    key.scaleType, key.x1, key.x2, key.faceGrids);
}

// The bits of a real for a hash, +0 and −0 alike (they compare equal).
std::uint64_t bitsOf(double value) noexcept
{
    return value == 0.0 ? 0 : std::bit_cast<std::uint64_t>(value);
}

// The compact rows of the distributed card values, numbered by first appearance as rowsByFirstAppearance numbers
// them (the same rows: a value equal under tieOf to one seen before takes its row), by an open-addressing table of
// the distinct keys only, budgeted and doubled when half full.
// PERFORMANCE: rowsByFirstAppearance holds every value's key and four indices (about 150 bytes per value): 16 GB of
// keys for 6 000 cases of 300 pressure shapes (38.4 M ShellPressureLoad values, in the cases and in their load sets),
// refused by the budget of the reference PC; here the memory follows the distinct rows. Measured: 6 000 cases of 300
// pressure shapes, the keys 16 GB → the distinct ones.
class CardValueRows
{
public:
    CardValueRows(std::size_t expectedRows, base::MemoryBudget& budget) : m_budget(&budget)
    {
        resize(std::max<std::size_t>(MINIMUM_SLOTS, std::bit_ceil(2 * std::max<std::size_t>(expectedRows, 1))));
    }

    // The row of `key`: the row of the equal key seen first, or the next row.
    std::int64_t rowOf(const CardValueKey& key)
    {
        std::size_t slot = hashOf(key) & (m_slots.size() - 1);
        while (m_slots[slot] >= 0)
        {
            if (tieOf(m_keys[static_cast<std::size_t>(m_slots[slot])]) == tieOf(key))
                return m_slots[slot];
            slot = (slot + 1) & (m_slots.size() - 1);
        }
        if (2 * (m_count + 1) > m_slots.size())
        {
            resize(2 * m_slots.size());
            return rowOf(key);
        }
        if (m_count == m_keys.size())
        {
            base::BudgetedBuffer<CardValueKey> grown(*m_budget, 2 * m_keys.size(), "distributed value keys");
            std::copy(m_keys.data(), m_keys.data() + m_count, grown.data());
            m_keys = std::move(grown);
        }
        m_keys[m_count] = key;
        m_slots[slot] = static_cast<std::int64_t>(m_count);
        return static_cast<std::int64_t>(m_count++);
    }

    [[nodiscard]] std::int64_t rowCount() const noexcept { return static_cast<std::int64_t>(m_count); }
    // The key of the first value of a row.
    [[nodiscard]] const CardValueKey& keyOfRow(std::int64_t row) const { return m_keys[static_cast<std::size_t>(row)]; }

private:
    static constexpr std::size_t MINIMUM_SLOTS = 64;

    static std::uint64_t hashOf(const CardValueKey& key) noexcept
    {
        // FNV-1a over the fields compared by tieOf, then a final mix (the table takes the low bits).
        std::uint64_t hash = 14695981039346656037ULL;
        const auto mix = [&hash](std::uint64_t word)
        {
            hash ^= word;
            hash *= 1099511628211ULL;
        };
        mix(static_cast<std::uint32_t>(key.elementId));
        mix(static_cast<std::uint32_t>(key.slot));
        mix(key.card);
        mix(key.cardSize);
        mix(static_cast<std::uint32_t>(key.frameId));
        for (const double component : key.direction)
            mix(bitsOf(component));
        mix(key.loadType);
        mix(key.scaleType);
        mix(bitsOf(key.x1));
        mix(bitsOf(key.x2));
        for (const std::int32_t grid : key.faceGrids)
            mix(static_cast<std::uint32_t>(grid));
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        return hash;
    }

    void resize(std::size_t slotCount)
    {
        m_slots = base::BudgetedBuffer<std::int64_t>(*m_budget, slotCount, "distributed value rows");
        std::fill(m_slots.data(), m_slots.data() + slotCount, -1);
        if (m_keys.size() == 0)
            m_keys = base::BudgetedBuffer<CardValueKey>(*m_budget, slotCount / 2, "distributed value keys");
        for (std::size_t row = 0; row < m_count; ++row)
        {
            std::size_t slot = hashOf(m_keys[row]) & (slotCount - 1);
            while (m_slots[slot] >= 0)
                slot = (slot + 1) & (slotCount - 1);
            m_slots[slot] = static_cast<std::int64_t>(row);
        }
    }

    base::MemoryBudget* m_budget;
    base::BudgetedBuffer<CardValueKey> m_keys;  // by row
    base::BudgetedBuffer<std::int64_t> m_slots; // the row of each slot, −1 when free
    std::size_t m_count = 0;
};

// The load set of an inertial load, or the sets (TEMPERATURE(LOAD), TEMPERATURE(INITIAL), DEFORM) of a thermal load
// (the initial strains of a subcase).
struct SetPair
{
    base::EntityId first = 0;
    base::EntityId second = 0;
    base::EntityId third = 0;
};

constexpr double UNKNOWN_PRECISION = std::numeric_limits<double>::quiet_NaN();

// The elementary sets a LOAD combination brings whole, sorted; nothing for a set of its own or a combination checked
// whole (the combinations are checked through their sets). With the combination table they are its terms, every one
// whole; a LOAD entry that holds cards of its own besides its terms is checked whole (its own cards are not a set the
// others share). With tables that expand the combinations they are read from
// the rows (loadsTable: its rows' source_sid, each source's rows a multiple of its own); nothing for a combination
// with line loads (lineLoadsTable names no source) or one that brings a part of a set.
std::optional<std::vector<base::EntityId>> wholeSourcesOf(const assembly::LoadVectorBuilder& loads,
                                                          const input::ModelSnapshot& snapshot, base::EntityId loadSet)
{
    if (!snapshot.hasTable(input::TableRole::Loads))
        return std::nullopt;
    if (snapshot.loads().hasCombinationTerms())
    {
        if (!snapshot.loads().isCombination(loadSet) || loads.holdsOwnCards(loadSet))
            return std::nullopt;
        std::vector<base::EntityId> sources; // bounded by the sets of the combination
        for (const input::LoadTerm& term : snapshot.loads().termsOf(loadSet))
            if (term.set != loadSet)
                sources.push_back(term.set);
        std::sort(sources.begin(), sources.end());
        return sources;
    }
    if (loads.hasLineLoads(loadSet))
        return std::nullopt;
    const std::span<const std::int32_t> sourceSetIds =
        snapshot.loads().loads().values<std::int32_t>("sourceSetId", base::NullPolicy::Allow);
    std::vector<std::pair<base::EntityId, std::size_t>> counts; // bounded by the sets of the combination
    for (const std::int64_t row : loads.loadTableRowsOf(loadSet))
    {
        const base::EntityId source = sourceSetIds[static_cast<std::size_t>(row)];
        auto found =
            std::find_if(counts.begin(), counts.end(), [&](const auto& entry) { return entry.first == source; });
        if (found == counts.end())
            counts.emplace_back(source, 1);
        else
            ++found->second;
    }
    if (counts.empty() || (counts.size() == 1 && counts.front().first == loadSet))
        return std::nullopt;
    std::vector<base::EntityId> sources; // bounded by the sets of the combination
    for (const auto& [source, count] : counts)
    {
        if (source == loadSet || loads.hasLineLoads(source))
            return std::nullopt;
        std::size_t own = 0;
        for (const std::int64_t row : loads.loadTableRowsOf(source))
            own += sourceSetIds[static_cast<std::size_t>(row)] == source ? 1 : 0;
        if (own == 0 || count % own != 0)
            return std::nullopt;
        sources.push_back(source);
    }
    std::sort(sources.begin(), sources.end());
    return sources;
}

// The position of `id` in the sorted, unique `ids`.
std::size_t positionOf(const base::BudgetedBuffer<base::EntityId>& ids, base::EntityId id)
{
    return static_cast<std::size_t>(std::lower_bound(ids.data(), ids.data() + ids.size(), id) - ids.data());
}

} // namespace

StaticLoadCases::StaticLoadCases(Inputs inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_inputs(std::move(inputs)), m_rows(assembly::ALL_LOAD_FAMILIES.size())
{
    // Every load set of the cases is checked once, in the order of the cases, whatever basis the families take (a case
    // that becomes a combination is never assembled on its own).
    const std::size_t cases = caseCount();
    const ItemRows loadSetRows = rowsByFirstAppearance(
        cases,
        [&](std::int64_t left, std::int64_t right)
        {
            const std::optional<base::EntityId>& a = m_inputs.cases[static_cast<std::size_t>(left)]->load;
            const std::optional<base::EntityId>& b = m_inputs.cases[static_cast<std::size_t>(right)]->load;
            return a < b;
        },
        budget);
    // A LOAD combination that brings its sets whole is checked through them, each once: its rows are those sets'
    // rows scaled, and checking each combination whole would check the skin 6 000 times for 6 000 combinations of a
    // pressure on the whole skin. A set of its own, or a combination with line loads (not in the loads table) or with
    // a part of a set, is checked whole.
    std::set<base::EntityId> checkedSources;
    std::int64_t nextRow = 0;
    for (std::size_t c = 0; c < cases; ++c)
    {
        if (loadSetRows.rowOfItem[c] != nextRow)
            continue;
        ++nextRow;
        if (!m_inputs.cases[c]->load)
            continue;
        const base::EntityId loadSet = *m_inputs.cases[c]->load;
        const std::optional<std::vector<base::EntityId>> sources =
            wholeSourcesOf(*m_inputs.loads, *m_inputs.snapshot, loadSet);
        if (!sources)
        {
            m_inputs.loads->checkLoads(loadSet, m_inputs.mass, budget, sink, m_inputs.checksInertialLoads);
            continue;
        }
        for (const base::EntityId source : *sources)
            if (checkedSources.insert(source).second)
                m_inputs.loads->checkLoads(source, m_inputs.mass, budget, sink, m_inputs.checksInertialLoads);
    }
    m_familyCoefficients.resize(assembly::ALL_LOAD_FAMILIES.size());
    if (m_inputs.keepsFamilies)
        m_descriptions.resize(assembly::ALL_LOAD_FAMILIES.size());
    for (const LoadFamily family : assembly::ALL_LOAD_FAMILIES)
        addFamily(family, budget);
    m_coefficients = linalg::DenseMatrix(budget, m_vectors.size(), caseCount(), "load basis coefficients");
    std::size_t first = 0;
    for (std::size_t f = 0; f < m_familyCoefficients.size(); ++f)
    {
        const linalg::DenseMatrix& alpha = m_familyCoefficients[f];
        for (std::size_t c = 0; c < alpha.columns(); ++c)
            for (std::size_t i = 0; i < alpha.rows(); ++i)
                m_coefficients(first + i, c) = alpha(i, c);
        first += alpha.rows();
    }
}

bool StaticLoadCases::hasThermalLoads() const noexcept
{
    return std::any_of(m_vectors.begin(), m_vectors.end(),
                       [](const Vector& vector) { return vector.family == LoadFamily::Thermal; });
}

void StaticLoadCases::addFamily(LoadFamily family, base::MemoryBudget& budget)
{
    const std::size_t familyIndex = static_cast<std::size_t>(family);
    const assembly::LoadVectorBuilder& loads = *m_inputs.loads;
    const input::ModelSnapshot& snapshot = *m_inputs.snapshot;
    const std::size_t cases = caseCount();
    const auto gSize = static_cast<std::size_t>(m_inputs.dofMap->gSetSize());
    const bool usesWrittenPrecision = m_inputs.tolerance == assembly::LoadBasisTolerance::WrittenPrecision;
    std::optional<assembly::FieldPrecision> fieldPrecision;
    if (usesWrittenPrecision && !snapshot.loadCardTables().empty())
        fieldPrecision.emplace(snapshot);
    const assembly::FieldPrecision* precision = fieldPrecision ? &*fieldPrecision : nullptr;
    // Without the field formats of the load cards, the deck tolerance has no precision to use: NaN makes
    // LoadBasis say so (and so does the NaN of a card whose format the tables do not give).
    const double missingPrecision = usesWrittenPrecision && precision == nullptr ? UNKNOWN_PRECISION : 0.0;
    const auto precisionOf = [&](double given) { return precision != nullptr ? given : missingPrecision; };

    FamilyRows& rows = m_rows[familyIndex];
    std::int64_t rowCount = 0;
    bool offersLoadedDofs = false;
    base::BudgetedBuffer<DistributedRowPlace> places;
    assembly::CompactLoadMatrix caseMatrix;
    std::optional<assembly::CompactLoadMatrix> setMatrix;
    std::optional<linalg::DenseMatrix> factors;
    // The cases of the family: those whose load set has loads of the family.
    const auto hasCase = [&](std::size_t c)
    {
        const StaticSubcase& subcase = *m_inputs.cases[c];
        return subcase.load && loads.hasFamilyLoads(*subcase.load, family);
    };

    // The elementary load sets of a load set of the family, each with its factor: the terms of the combination table
    // (below), or, with tables that expand the combinations, read from loadsTable (source_sid and scale): every use of
    // a set in a combination brings all of its rows with the product of the LOAD factors it was reached through, so a
    // set used k times with the scale v has k × (its own rows) rows of scale v. The factor is Σ v·k over the distinct
    // scales, exact for a set used once (its scale itself, the factor O_o_Reader multiplied each of its values by).
    // Taking it as (Σ of the rows' scales) / (its own rows) would round more as the rows grow (a set of 8 700
    // FacePressureLoad summed 8 700 times: 1e-13 relative), so that the cases of a pressure on the whole skin would
    // fail the reconstruction check and join the basis (222 vectors for 600 cases of 30 families). A count that is not
    // a multiple of the set's own rows (a table that does not bring whole sets) keeps that quotient.
    const input::LoadTable* loadTable = snapshot.hasTable(input::TableRole::Loads) ? &snapshot.loads() : nullptr;
    // `isWhole` (when given) is cleared when a source does not come whole (its row count not a multiple of its own).
    const auto sourcesOf = [&](base::EntityId loadSet, bool* isWhole = nullptr)
    {
        std::vector<std::pair<base::EntityId, double>> sources; // bounded by the sets of one combination
        if (loadTable == nullptr)
            return sources;
        // The terms of the combination table, the factors as the deck gives them (combined_factor), never read back
        // from rows, each set whole by construction, line loads and every card type included. A LOAD entry that holds
        // cards of its own besides its terms (which a valid deck does not have) is not a combination of whole sets:
        // its own cards are no set column (the case joins the basis by its check).
        if (loadTable->hasCombinationTerms())
        {
            for (const input::LoadTerm& term : loadTable->termsOf(loadSet))
            {
                const bool isOwn = term.set == loadSet;
                if (isOwn && loadTable->isCombination(loadSet))
                {
                    if (isWhole != nullptr && loads.holdsOwnCards(loadSet))
                        *isWhole = false;
                    continue;
                }
                if (loads.hasFamilyLoads(term.set, family))
                    sources.emplace_back(term.set, term.factor);
            }
            return sources;
        }
        // The line loads of a combination are in lineLoadsTable under its own SID, which names no source set: the
        // combination is a set of its own in the distributed family (its column all its pressures and line loads).
        // Taking its sources from loadsTable alone would leave its line loads out of every combination when other
        // cases combine pressure sets (a LOAD of a BeamDistributedLoad set alone would solve to zero).
        if (family == LoadFamily::Distributed && loads.hasLineLoads(loadSet))
        {
            sources.emplace_back(loadSet, 1.0);
            return sources;
        }
        const std::span<const std::int32_t> sourceSetIds =
            loadTable->loads().values<std::int32_t>("sourceSetId", base::NullPolicy::Allow);
        const std::span<const double> scales = loadTable->loadScales();
        const base::DictionaryView& cards = loads.loadTableCards();
        // Per source: its distinct scales with their row counts, in order of appearance.
        std::vector<std::vector<std::pair<double, std::size_t>>> scalesOf; // bounded by the sets of one combination
        for (const std::int64_t row : loads.loadTableRowsOf(loadSet))
        {
            const std::optional<LoadFamily> rowFamily = assembly::LoadVectorBuilder::familyOfCard(cards[row]);
            if (!rowFamily || *rowFamily != family)
                continue;
            const base::EntityId source = sourceSetIds[static_cast<std::size_t>(row)];
            const double scale = scales[static_cast<std::size_t>(row)];
            auto found =
                std::find_if(sources.begin(), sources.end(),
                             [&](const std::pair<base::EntityId, double>& entry) { return entry.first == source; });
            if (found == sources.end())
            {
                sources.emplace_back(source, 0.0);
                scalesOf.emplace_back();
                found = sources.end() - 1;
            }
            std::vector<std::pair<double, std::size_t>>& distinct =
                scalesOf[static_cast<std::size_t>(found - sources.begin())];
            auto same = std::find_if(distinct.begin(), distinct.end(),
                                     [&](const std::pair<double, std::size_t>& entry) { return entry.first == scale; });
            if (same == distinct.end())
                distinct.emplace_back(scale, 1);
            else
                ++same->second;
        }
        for (std::size_t k = 0; k < sources.size(); ++k)
        {
            std::size_t own = 0;
            for (const std::int64_t row : loads.loadTableRowsOf(sources[k].first))
            {
                const std::optional<LoadFamily> rowFamily = assembly::LoadVectorBuilder::familyOfCard(cards[row]);
                if (rowFamily && *rowFamily == family &&
                    sourceSetIds[static_cast<std::size_t>(row)] == sources[k].first)
                    ++own;
            }
            const bool isSourceWhole = own > 0 && std::all_of(scalesOf[k].begin(), scalesOf[k].end(),
                                                              [&](const std::pair<double, std::size_t>& entry)
                                                              { return entry.second % own == 0; });
            if (isWhole != nullptr && !isSourceWhole)
                *isWhole = false;
            if (own == 0)
                continue;
            double factor = 0.0;
            if (isSourceWhole)
                for (const auto& [scale, count] : scalesOf[k])
                    factor += count == own ? scale : scale * static_cast<double>(count / own);
            else
            {
                for (const auto& [scale, count] : scalesOf[k])
                    for (std::size_t r = 0; r < count; ++r)
                        factor += scale;
                factor /= static_cast<double>(own);
            }
            sources[k].second = factor;
        }
        return sources;
    };

    // The load sets as written (concentrated and distributed loads) are a basis of their own only when some case
    // combines them (a LOAD card): when every case is its own and only set, they are the cases. Their IDs are sorted
    // (a set's column is found by a binary search), and the factors of every case are filled once they are known.
    // When every case of the family brings its sets whole (a LOAD of elementary sets, or a set of its own), the cases
    // are exact combinations of those sets with their factors: the basis is made from the sets alone
    // (assembly::LoadBasis::Input::isCombinationOfSets), never from a column per case, which would hold every value of
    // every case (6 000 cases of a pressure on the whole skin: the skin's values 6 000 times, read three times each,
    // and the rank test and the check on them). With the deck tolerance the cases are taken one column each.
    bool isCombinationOfSets = false;
    if ((family == LoadFamily::Concentrated || family == LoadFamily::Distributed) && loadTable != nullptr)
    {
        std::size_t sourceCount = 0;
        bool hasCombination = false;
        bool isEveryCaseWhole = true;
        std::vector<std::vector<std::pair<base::EntityId, double>>> sourcesOfCase(cases); // bounded by the cases
        for (std::size_t c = 0; c < cases; ++c)
        {
            if (!hasCase(c))
                continue;
            sourcesOfCase[c] = sourcesOf(*m_inputs.cases[c]->load, &isEveryCaseWhole);
            const auto& sources = sourcesOfCase[c];
            sourceCount += sources.size();
            hasCombination = hasCombination || sources.size() > 1 ||
                             (sources.size() == 1 && sources.front().first != *m_inputs.cases[c]->load);
        }
        if (hasCombination)
        {
            base::BudgetedBuffer<base::EntityId> ids(budget, sourceCount, "load set IDs");
            std::size_t next = 0;
            for (std::size_t c = 0; c < cases; ++c)
                for (const auto& [source, factor] : sourcesOfCase[c])
                    ids[next++] = source;
            std::sort(ids.data(), ids.data() + sourceCount);
            const auto setCount =
                static_cast<std::size_t>(std::unique(ids.data(), ids.data() + sourceCount) - ids.data());
            rows.setOfColumn = base::BudgetedBuffer<base::EntityId>(budget, setCount, "load set IDs");
            std::copy(ids.data(), ids.data() + setCount, rows.setOfColumn.data());
            factors.emplace(budget, setCount, cases, "load set factors");
            for (std::size_t c = 0; c < cases; ++c)
                for (const auto& [source, factor] : sourcesOfCase[c])
                    (*factors)(positionOf(rows.setOfColumn, source), c) += factor;
            isCombinationOfSets = isEveryCaseWhole && !usesWrittenPrecision;
        }
    }
    const std::size_t setCount = rows.setOfColumn.size();
    // The columns whose values make the compact rows: the cases, then the sets; the sets alone for combinations.
    const std::size_t firstColumn = isCombinationOfSets ? cases : 0;

    if (family == LoadFamily::Concentrated || family == LoadFamily::Enforced)
    {
        // One compact row per loaded DOF, numbered by first appearance (the cases, then the load sets).
        offersLoadedDofs = true;
        base::BudgetedBuffer<std::int64_t> rowOfDof(budget, gSize, "load rows of DOFs");
        std::fill(rowOfDof.data(), rowOfDof.data() + gSize, -1);
        const auto rowOf = [&](base::DofIndex dof)
        {
            std::int64_t& row = rowOfDof[static_cast<std::size_t>(dof)];
            if (row < 0)
                row = rowCount++;
            return row;
        };
        const auto entriesOf = [&](base::EntityId loadSet)
        { return loads.concentratedEntries(loadSet, budget, precision); };
        std::int64_t caseNonzeros = 0;
        std::int64_t setNonzeros = 0;
        std::size_t largestColumn = 0;
        for (std::size_t c = firstColumn; c < cases; ++c)
        {
            std::size_t count = 0;
            if (family == LoadFamily::Enforced)
            {
                for (const assembly::EnforcedDisplacement& value : m_inputs.spcs[c]->enforcedDisplacements())
                    if (value.value != 0.0)
                    {
                        rowOf(value.dof);
                        ++count;
                    }
            }
            else if (hasCase(c))
            {
                const auto entries = entriesOf(*m_inputs.cases[c]->load);
                for (std::size_t k = 0; k < entries.size(); ++k)
                    if (entries[k].value != 0.0)
                    {
                        rowOf(entries[k].dof);
                        ++count;
                    }
            }
            caseNonzeros += static_cast<std::int64_t>(count);
            largestColumn = std::max(largestColumn, count);
        }
        for (std::size_t s = 0; s < setCount; ++s)
        {
            const auto entries = entriesOf(rows.setOfColumn[s]);
            std::size_t count = 0;
            for (std::size_t k = 0; k < entries.size(); ++k)
                if (entries[k].value != 0.0)
                {
                    rowOf(entries[k].dof);
                    ++count;
                }
            setNonzeros += static_cast<std::int64_t>(count);
            largestColumn = std::max(largestColumn, count);
        }
        rows.dofOfRow = base::BudgetedBuffer<base::DofIndex>(budget, static_cast<std::size_t>(rowCount), "loaded DOFs");
        for (std::size_t dof = 0; dof < gSize; ++dof)
            if (rowOfDof[dof] >= 0)
                rows.dofOfRow[static_cast<std::size_t>(rowOfDof[dof])] = static_cast<base::DofIndex>(dof);

        ColumnBuilder builder(rowCount, largestColumn, budget);
        const auto appendSet = [&](base::EntityId loadSet, assembly::CompactLoadMatrix& matrix)
        {
            const auto entries = entriesOf(loadSet);
            for (std::size_t k = 0; k < entries.size(); ++k)
                if (entries[k].value != 0.0)
                    builder.add(rowOfDof[static_cast<std::size_t>(entries[k].dof)], entries[k].value,
                                precisionOf(entries[k].precision));
            builder.appendTo(matrix);
        };
        if (!isCombinationOfSets)
            caseMatrix = assembly::CompactLoadMatrix(budget, rowCount, static_cast<std::int64_t>(cases), caseNonzeros);
        for (std::size_t c = firstColumn; c < cases; ++c)
        {
            if (family == LoadFamily::Enforced)
            {
                for (const assembly::EnforcedDisplacement& value : m_inputs.spcs[c]->enforcedDisplacements())
                    if (value.value != 0.0)
                        builder.add(rowOfDof[static_cast<std::size_t>(value.dof)], value.value, 0.0);
                builder.appendTo(caseMatrix);
            }
            else if (hasCase(c))
                appendSet(*m_inputs.cases[c]->load, caseMatrix);
            else
                builder.appendTo(caseMatrix);
        }
        if (setCount > 0)
        {
            setMatrix.emplace(budget, rowCount, static_cast<std::int64_t>(setCount), setNonzeros);
            for (std::size_t s = 0; s < setCount; ++s)
                appendSet(rows.setOfColumn[s], *setMatrix);
        }
    }
    else if (family == LoadFamily::Distributed)
    {
        // One compact row per card value (element, corner and variant), numbered by first appearance (the cases, then
        // the load sets). Three passes over the columns: their values counted, their rows numbered (CardValueRows,
        // the distinct keys only), the columns filled.
        const auto valuesOf = [&](base::EntityId loadSet)
        { return loads.distributedValues(loadSet, budget, precision); };
        const auto loadSetOf = [&](std::size_t column)
        { return column >= cases ? rows.setOfColumn[column - cases] : *m_inputs.cases[column]->load; };
        base::BudgetedBuffer<std::int64_t> valueCount(budget, cases + setCount, "distributed values per column");
        std::size_t total = 0;
        std::int64_t caseNonzeros = 0;
        std::size_t largestColumn = 0;
        for (std::size_t column = firstColumn; column < cases + setCount; ++column)
        {
            std::size_t count = 0;
            if (column >= cases)
                count = valuesOf(rows.setOfColumn[column - cases]).size();
            else if (hasCase(column))
                count = valuesOf(*m_inputs.cases[column]->load).size();
            valueCount[column] = static_cast<std::int64_t>(count);
            total += count;
            if (column < cases)
                caseNonzeros += static_cast<std::int64_t>(count);
            largestColumn = std::max(largestColumn, count);
        }
        CardValueRows valueRows(largestColumn, budget);
        for (std::size_t column = firstColumn; column < cases + setCount; ++column)
        {
            if (valueCount[column] == 0)
                continue;
            const auto columnValues = valuesOf(loadSetOf(column));
            for (const assembly::LoadVectorBuilder::CardValue& value : columnValues.span())
                (void)valueRows.rowOf(keyOf(value));
        }
        rowCount = valueRows.rowCount();
        if (m_inputs.keepsFamilies)
        {
            // Where each row acts: the element (or PointsPressureLoad face) of its first value.
            places = base::BudgetedBuffer<DistributedRowPlace>(budget, static_cast<std::size_t>(rowCount),
                                                               "distributed row places");
            for (std::int64_t row = 0; row < rowCount; ++row)
                places[static_cast<std::size_t>(row)] =
                    DistributedRowPlace{valueRows.keyOfRow(row).elementId, valueRows.keyOfRow(row).faceGrids};
        }

        ColumnBuilder builder(rowCount, largestColumn, budget);
        if (!isCombinationOfSets)
            caseMatrix = assembly::CompactLoadMatrix(budget, rowCount, static_cast<std::int64_t>(cases), caseNonzeros);
        if (setCount > 0)
            setMatrix.emplace(budget, rowCount, static_cast<std::int64_t>(setCount),
                              static_cast<std::int64_t>(total) - caseNonzeros);
        for (std::size_t column = firstColumn; column < cases + setCount; ++column)
        {
            if (valueCount[column] != 0)
            {
                const auto columnValues = valuesOf(loadSetOf(column));
                for (const assembly::LoadVectorBuilder::CardValue& value : columnValues.span())
                    builder.add(valueRows.rowOf(keyOf(value)), value.value, precisionOf(value.precision));
            }
            builder.appendTo(column < cases ? caseMatrix : *setMatrix);
        }
    }
    else if (family == LoadFamily::Inertial)
    {
        // Inertial loads by the generators of their fields (assembly::InertialTerm): one compact row
        // per component of each distinct generator (the uniform translations along the basic axes that every
        // GravityLoad row shares, the points of an AccelNodeLoad, the table of an ACCEL, the field of an
        // RotationalForceLoad), numbered by first appearance, a case's column its coefficients summed over its rows (a
        // case's inertial loads are Mgg times the sum of its rows' fields, linear in the coefficients). The rank is
        // that of the accelerations the cases combine: three for any number of GravityLoad cards, where a row per load
        // set made every case its own vector (6 000 cases of 300 families with a GravityLoad in every LOAD gave 6 000
        // vectors). The rows' problems are reported when the loads are made (addFamilyCase).
        base::DiagnosticSink unreported;
        std::map<std::vector<double>, std::int64_t> firstRowOf; // bounded by the distinct generators
        std::int64_t caseNonzeros = 0;
        std::size_t largestColumn = 0;
        for (std::size_t c = 0; c < cases; ++c)
        {
            if (!hasCase(c))
                continue;
            std::size_t count = 0;
            for (const assembly::InertialTerm& term : loads.inertialTermsOf(*m_inputs.cases[c]->load, unreported))
            {
                if (firstRowOf.emplace(term.generator, rowCount).second)
                    rowCount += static_cast<std::int64_t>(term.coefficients.size());
                count += term.coefficients.size();
            }
            caseNonzeros += static_cast<std::int64_t>(count);
            largestColumn = std::max(largestColumn, count);
        }
        ColumnBuilder builder(rowCount, largestColumn, budget);
        caseMatrix = assembly::CompactLoadMatrix(budget, rowCount, static_cast<std::int64_t>(cases), caseNonzeros);
        constexpr double EXACT = 0.0;
        for (std::size_t c = 0; c < cases; ++c)
        {
            if (hasCase(c))
                for (const assembly::InertialTerm& term : loads.inertialTermsOf(*m_inputs.cases[c]->load, unreported))
                {
                    const std::int64_t first = firstRowOf.at(term.generator);
                    for (std::size_t k = 0; k < term.coefficients.size(); ++k)
                        builder.add(first + static_cast<std::int64_t>(k), term.coefficients[k], EXACT);
                }
            builder.appendTo(caseMatrix);
        }
    }
    else
    {
        // Thermal loads: one compact row per distinct key of temperature and deformation sets (InitialStrainKey):
        // two cases on the same sets have the same load, the others are independent; the rank then takes one case
        // per key.
        const auto keyOfCase = [&](std::size_t c) -> std::optional<SetPair>
        {
            const StaticSubcase& subcase = *m_inputs.cases[c];
            if (!subcase.hasInitialStrains())
                return std::nullopt;
            const InitialStrainKey key = initialStrainKeyOf(subcase);
            return SetPair{key.temperatureLoad.value_or(0), key.temperatureInitial.value_or(-1),
                           key.deformation.value_or(0)};
        };
        // The cases without a load share the row of "no key", which is not used.
        base::BudgetedBuffer<SetPair> keys(budget, cases, "load set keys");
        base::BudgetedBuffer<std::uint8_t> hasKey(budget, cases, "load set keys present");
        std::int64_t keyedCases = 0;
        for (std::size_t c = 0; c < cases; ++c)
            if (const auto key = keyOfCase(c))
            {
                keys[c] = *key;
                hasKey[c] = 1;
                ++keyedCases;
            }
        const ItemRows caseRows = rowsByFirstAppearance(
            cases,
            [&](std::int64_t left, std::int64_t right)
            {
                const auto l = static_cast<std::size_t>(left);
                const auto r = static_cast<std::size_t>(right);
                return std::tie(hasKey[l], keys[l].first, keys[l].second, keys[l].third) <
                       std::tie(hasKey[r], keys[r].first, keys[r].second, keys[r].third);
            },
            budget);
        // The rows of the keyed cases, numbered by first appearance among them.
        base::BudgetedBuffer<std::int64_t> rowOfCaseRow(budget, static_cast<std::size_t>(caseRows.rowCount),
                                                        "load rows of keys");
        std::fill(rowOfCaseRow.data(), rowOfCaseRow.data() + rowOfCaseRow.size(), -1);
        for (std::size_t c = 0; c < cases; ++c)
        {
            std::int64_t& row = rowOfCaseRow[static_cast<std::size_t>(caseRows.rowOfItem[c])];
            if (hasKey[c] != 0 && row < 0)
                row = rowCount++;
        }
        caseMatrix = assembly::CompactLoadMatrix(budget, rowCount, static_cast<std::int64_t>(cases), keyedCases);
        constexpr double UNIT = 1.0;
        constexpr double EXACT = 0.0;
        for (std::size_t c = 0; c < cases; ++c)
        {
            if (hasKey[c] == 0)
            {
                caseMatrix.appendColumn({}, {}, {});
                continue;
            }
            const std::int64_t row = rowOfCaseRow[static_cast<std::size_t>(caseRows.rowOfItem[c])];
            caseMatrix.appendColumn({&row, 1}, {&UNIT, 1}, {&EXACT, 1});
        }
    }

    assembly::LoadBasis::Input input;
    input.family = family;
    input.cases = isCombinationOfSets ? nullptr : &caseMatrix;
    input.isCombinationOfSets = isCombinationOfSets;
    input.caseCount = cases;
    input.loadSets = setMatrix ? &*setMatrix : nullptr;
    input.setCoefficients = setMatrix ? &*factors : nullptr;
    input.offersLoadedDofs = offersLoadedDofs;
    input.tolerance = (family == LoadFamily::Concentrated || family == LoadFamily::Distributed)
                          ? m_inputs.tolerance
                          : assembly::LoadBasisTolerance::Roundoff;
    input.pool = m_inputs.pool;
    input.kernel = m_inputs.kernel;
    assembly::LoadBasis basis = assembly::LoadBasis::choose(input, budget);
    if (basis.kind() != assembly::LoadBasisKind::Empty)
        m_reports.push_back(basis.report());
    if (m_inputs.keepsFamilies && basis.kind() != assembly::LoadBasisKind::Empty)
    {
        FamilyDescription description;
        description.family = family;
        description.offersLoadedDofs = offersLoadedDofs;
        description.tolerance = input.tolerance;
        description.cases = std::move(caseMatrix);
        description.isCombinationOfSets = isCombinationOfSets;
        description.rowCount = rowCount;
        description.loadSets = std::move(setMatrix);
        description.setFactors = std::move(factors);
        description.dofOfRow = rows.dofOfRow.span();
        description.places = std::move(places);
        m_descriptions[familyIndex] = std::move(description);
    }
    for (const assembly::BasisVector& vector : basis.vectors())
        m_vectors.push_back({family, vector});
    m_familyCoefficients[familyIndex] = linalg::DenseMatrix(budget, basis.coefficients().rows(),
                                                            basis.coefficients().columns(), "load basis coefficients");
    for (std::size_t c = 0; c < basis.coefficients().columns(); ++c)
        for (std::size_t i = 0; i < basis.coefficients().rows(); ++i)
            m_familyCoefficients[familyIndex](i, c) = basis.coefficients()(i, c);
}

void StaticLoadCases::addFamilyCase(LoadFamily family, std::size_t c, std::span<double> loadsG,
                                    std::span<double> enforcedG, base::MemoryBudget& budget,
                                    base::DiagnosticSink& sink) const
{
    const StaticSubcase& subcase = *m_inputs.cases[c];
    switch (family)
    {
    case LoadFamily::Concentrated:
    case LoadFamily::Distributed:
    case LoadFamily::Inertial:
        if (subcase.load)
            m_inputs.loads->addFamilyLoads(*subcase.load, family, m_inputs.mass, loadsG, budget, sink);
        return;
    case LoadFamily::Thermal:
    {
        // The thermal and element deformation loads join the applied loads (in the input format, TEMPERATURE(LOAD) and
        // DEFORM are loads of the subcase), so that the SPC forces balance them.
        const std::unique_ptr<assembly::TemperatureField> field =
            initialStrainsOf(*m_inputs.snapshot, *m_inputs.points, subcase, budget, sink);
        if (!field)
            return;
        base::DiagnosticSink thermalSink;
        m_inputs.loads->checkThermalLoads(field->temperatures(), thermalSink);
        const bool isApplicable = !thermalSink.hasErrors();
        for (base::Diagnostic& diagnostic : thermalSink.diagnostics())
            sink.report(std::move(diagnostic));
        if (isApplicable)
            m_inputs.loads->addThermalLoads(field->temperatures(), loadsG);
        return;
    }
    case LoadFamily::Enforced:
        for (const assembly::EnforcedDisplacement& value : m_inputs.spcs[c]->enforcedDisplacements())
            enforcedG[static_cast<std::size_t>(value.dof)] = value.value;
        return;
    }
}

StaticLoadCases::VectorName StaticLoadCases::nameOf(std::size_t vector) const
{
    const Vector& entry = m_vectors.at(vector);
    const FamilyRows& rows = m_rows[static_cast<std::size_t>(entry.family)];
    VectorName name;
    name.family = entry.family;
    name.source = entry.source.source;
    for (const assembly::LoadBasisReport& report : m_reports)
        if (report.family == entry.family)
            name.kind = report.kind;
    const auto index = static_cast<std::size_t>(entry.source.index);
    switch (entry.source.source)
    {
    case assembly::BasisVector::Source::LoadSet:
        name.loadSet = rows.setOfColumn[index];
        break;
    case assembly::BasisVector::Source::LoadedRow:
        name.dof = rows.dofOfRow[index];
        break;
    case assembly::BasisVector::Source::Case:
        name.subcase = m_inputs.cases.at(index)->label;
        break;
    }
    return name;
}

void StaticLoadCases::basisVector(std::size_t vector, std::span<double> loadsG, std::span<double> enforcedG,
                                  base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    std::fill(loadsG.begin(), loadsG.end(), 0.0);
    std::fill(enforcedG.begin(), enforcedG.end(), 0.0);
    const Vector& entry = m_vectors[vector];
    const FamilyRows& rows = m_rows[static_cast<std::size_t>(entry.family)];
    switch (entry.source.source)
    {
    case assembly::BasisVector::Source::LoadSet:
        m_inputs.loads->addFamilyLoads(rows.setOfColumn[static_cast<std::size_t>(entry.source.index)], entry.family,
                                       m_inputs.mass, loadsG, budget, sink);
        return;
    case assembly::BasisVector::Source::LoadedRow:
    {
        const auto dof = static_cast<std::size_t>(rows.dofOfRow[static_cast<std::size_t>(entry.source.index)]);
        if (entry.family == LoadFamily::Enforced)
            enforcedG[dof] = 1.0;
        else
            loadsG[dof] = 1.0;
        return;
    }
    case assembly::BasisVector::Source::Case:
        addFamilyCase(entry.family, static_cast<std::size_t>(entry.source.index), loadsG, enforcedG, budget, sink);
        return;
    }
}

const StaticLoadCases::FamilyDescription* StaticLoadCases::familyDescription(LoadFamily family) const noexcept
{
    const auto index = static_cast<std::size_t>(family);
    if (index >= m_descriptions.size() || !m_descriptions[index])
        return nullptr;
    return &*m_descriptions[index];
}

void StaticLoadCases::familyVector(LoadFamily family, const assembly::BasisVector& source, std::span<double> loadsG,
                                   std::span<double> enforcedG, base::MemoryBudget& budget,
                                   base::DiagnosticSink& sink) const
{
    std::fill(loadsG.begin(), loadsG.end(), 0.0);
    std::fill(enforcedG.begin(), enforcedG.end(), 0.0);
    const FamilyRows& rows = m_rows[static_cast<std::size_t>(family)];
    switch (source.source)
    {
    case assembly::BasisVector::Source::LoadSet:
        m_inputs.loads->addFamilyLoads(rows.setOfColumn[static_cast<std::size_t>(source.index)], family, m_inputs.mass,
                                       loadsG, budget, sink);
        return;
    case assembly::BasisVector::Source::LoadedRow:
    {
        const auto dof = static_cast<std::size_t>(rows.dofOfRow[static_cast<std::size_t>(source.index)]);
        if (family == LoadFamily::Enforced)
            enforcedG[dof] = 1.0;
        else
            loadsG[dof] = 1.0;
        return;
    }
    case assembly::BasisVector::Source::Case:
        addFamilyCase(family, static_cast<std::size_t>(source.index), loadsG, enforcedG, budget, sink);
        return;
    }
}

void StaticLoadCases::exactCase(std::size_t c, std::span<double> loadsG, std::span<double> enforcedG,
                                base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    std::fill(loadsG.begin(), loadsG.end(), 0.0);
    std::fill(enforcedG.begin(), enforcedG.end(), 0.0);
    for (const LoadFamily family : assembly::ALL_LOAD_FAMILIES)
        addFamilyCase(family, c, loadsG, enforcedG, budget, sink);
}

} // namespace oos::solutions
