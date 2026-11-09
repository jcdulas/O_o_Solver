#include "oos/input/tables/ContactTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <string>
#include <utility>

namespace oos::input
{

namespace
{

// The columns of contactNodePairs and contactPairsTable a glued contact needs: the face of each row and the side
// it was searched from (the glue ties the grid points of the secondary body only), and IGLUE (glued or touching).
constexpr std::array<const char*, 2> NODE_PAIR_GLUE_COLUMNS{"primaryFace", "searchedFrom"};
constexpr std::array<const char*, 1> PAIR_GLUE_COLUMNS{"glueOption"};

// A text of a column that O_o_Reader may give as utf8 or as dictionary<utf8>; empty for a null.
std::string_view textAt(const base::ContractReader& reader, const std::string& column, std::int64_t row)
{
    if (!reader.validity(column).isValid(row))
        return {};
    if (reader.columnType(column) == "utf8")
        return reader.texts(column, base::NullPolicy::Allow)[row];
    return reader.dictionaryTexts(column, base::NullPolicy::Allow)[row];
}

// The cards that pair contact bodies, as the `table` texts of the contact tables name them with their ID
// ("ContactTable 5", "ContactPair 3"): BCTABLE, and BCONECT, which only a BCTABL1 selects.
constexpr std::string_view CONTACT_TABLE_CARD = "ContactTable";
constexpr std::string_view CONTACT_PAIR_PREFIX = "ContactPair ";

template <typename T>
std::optional<T> optionalValue(const base::ContractReader& reader, const std::string& column, std::int64_t row)
{
    if (!reader.hasColumn(column) || !reader.validity(column).isValid(row))
        return std::nullopt;
    return reader.values<T>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

} // namespace

std::optional<ContactTableName> parseContactTableName(std::string_view text)
{
    const std::size_t space = text.find(' ');
    if (space == std::string_view::npos || space == 0)
        return std::nullopt;
    ContactTableName name;
    name.card = std::string(text.substr(0, space));
    const std::string_view number = text.substr(space + 1);
    const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), name.id);
    if (error != std::errc() || end != number.data() + number.size())
        return std::nullopt;
    return name;
}

ContactTable::ContactTable(base::ImportedTable nodePairs, base::ImportedTable pairs,
                           std::optional<base::ImportedTable> tables)
    : m_nodePairs(readTable(TableRole::ContactNodePairs, std::move(nodePairs))),
      m_pairs(readTable(TableRole::ContactPairs, std::move(pairs)))
{
    if (tables)
        m_tables.emplace(readTable(TableRole::ContactTables, std::move(*tables)));
}

std::vector<std::string> ContactTable::missingGlueColumns() const
{
    std::vector<std::string> missing;
    for (const char* column : NODE_PAIR_GLUE_COLUMNS)
        if (!m_nodePairs.hasColumn(column))
            missing.push_back(std::string("contactNodePairs.").append(column));
    for (const char* column : PAIR_GLUE_COLUMNS)
        if (!m_pairs.hasColumn(column))
            missing.push_back(std::string("contactPairsTable.").append(column));
    return missing;
}

std::string_view ContactTable::pairTable(std::int64_t row) const
{
    if (!m_pairs.validity("table").isValid(row))
        return {};
    return m_pairs.texts("table", base::NullPolicy::Allow)[row];
}

std::int32_t ContactTable::pairSecondaryBody(std::int64_t row) const
{
    return m_pairs.values<std::int32_t>("secondaryBody")[static_cast<std::size_t>(row)];
}

std::int32_t ContactTable::pairPrimaryBody(std::int64_t row) const
{
    return m_pairs.values<std::int32_t>("primaryBody")[static_cast<std::size_t>(row)];
}

std::string_view ContactTable::pairKind(std::int64_t row) const
{
    return m_pairs.dictionaryTexts("kind")[row];
}

std::int64_t ContactTable::pairNodeCount(std::int64_t row) const
{
    return m_pairs.values<std::int64_t>("nodePairCount")[static_cast<std::size_t>(row)];
}

GlueParameters ContactTable::glueParameters(std::int64_t row) const
{
    GlueParameters parameters;
    parameters.glueOption = optionalValue<std::int32_t>(m_pairs, "glueOption", row);
    parameters.glueSeparation = optionalValue<std::int32_t>(m_pairs, "glueSeparation", row);
    parameters.coordinateCheck = optionalValue<std::int32_t>(m_pairs, "coordinateCheck", row);
    parameters.searchOption = optionalValue<std::int32_t>(m_pairs, "searchOption", row);
    parameters.errorTolerance = optionalValue<double>(m_pairs, "errorTolerance", row);
    parameters.toleranceBias = optionalValue<double>(m_pairs, "toleranceBias", row);
    // O_o_Reader writes "not given" as NaN or null (the glue then takes its default tolerance).
    if (parameters.errorTolerance && std::isnan(*parameters.errorTolerance))
        parameters.errorTolerance.reset();
    if (parameters.toleranceBias && std::isnan(*parameters.toleranceBias))
        parameters.toleranceBias.reset();
    if (m_pairs.hasColumn("glueOptionSource"))
        parameters.glueOptionSource = std::string(textAt(m_pairs, "glueOptionSource", row));
    return parameters;
}

ContactNodePair ContactTable::nodePair(std::int64_t row) const
{
    const auto r = static_cast<std::size_t>(row);
    ContactNodePair pair;
    pair.row = row;
    if (m_nodePairs.validity("table").isValid(row))
        pair.table = m_nodePairs.texts("table", base::NullPolicy::Allow)[row];
    pair.secondaryBody = m_nodePairs.values<std::int32_t>("secondaryBody")[r];
    pair.primaryBody = m_nodePairs.values<std::int32_t>("primaryBody")[r];
    pair.nodeId = m_nodePairs.values<std::int32_t>("nodeId")[r];
    pair.primaryElementId = optionalValue<std::int32_t>(m_nodePairs, "primaryElementId", row);
    if (hasFace(row))
        pair.face = m_nodePairs.lists<std::int32_t>("primaryFace", base::NullPolicy::Allow)[row];
    if (m_nodePairs.hasColumn("searchedFrom"))
        pair.isSearchedFromPrimary = textAt(m_nodePairs, "searchedFrom", row) == "primary";
    return pair;
}

bool ContactTable::hasFace(std::int64_t row) const
{
    return m_nodePairs.hasColumn("primaryFace") && m_nodePairs.validity("primaryFace").isValid(row);
}

std::vector<std::int64_t> ContactTable::selectedPairRows(std::int32_t bcontactId) const
{
    // The `table` texts the ID selects: the pair cards contactTablesTable gives for that table ID (a BCTABLE itself,
    // the BCONECTs of a BCTABL1), else the BCTABLE of that ID.
    std::vector<std::string> selected; // bounded by the pairs of one contact table
    if (m_tables)
    {
        const std::span<const std::int32_t> ids = m_tables->values<std::int32_t>("tableId");
        for (std::size_t row = 0; row < ids.size(); ++row)
        {
            const auto r = static_cast<std::int64_t>(row);
            const std::string_view card = textAt(*m_tables, "pairCard", r);
            if (ids[row] == bcontactId && !card.empty())
                selected.emplace_back(card);
        }
    }
    // Without contactTablesTable, or with an empty one (O_o_Reader's for a model built without contact cards, whose
    // pairs a caller hands over), the BCTABLE of that ID (BCONTACT names a BCTABLE or a BCTABL1). A
    // non-empty table lists every contact table of the model: an ID it does not name selects nothing.
    if (!m_tables || m_tables->rowCount() == 0)
    {
        selected.push_back(base::concatenate({CONTACT_TABLE_CARD, " ", std::to_string(bcontactId)}));
        bool hasBctable = false;
        bool hasBconect = false;
        for (std::int64_t row = 0; row < pairCount(); ++row)
        {
            const std::string_view table = pairTable(row);
            hasBctable = hasBctable || table == selected.front();
            hasBconect = hasBconect || table.starts_with(CONTACT_PAIR_PREFIX);
        }
        if (!hasBctable && hasBconect)
            throw base::ContractError(base::concatenate(
                {"contact: ContactSelection = ", std::to_string(bcontactId),
                 " names no ContactTable and the model pairs bodies by ContactPair, which only a ContactTable1 "
                 "selects; expected the table contactTablesTable of O_o_Reader (the pairs of each ContactTable1), "
                 "found none; "
                 "install a O_o_Reader that gives it or hand it over to the "
                 "snapshot"}));
    }
    std::vector<std::int64_t> rows;
    for (std::int64_t row = 0; row < pairCount(); ++row)
    {
        const std::string_view table = pairTable(row);
        for (const std::string& name : selected)
            if (!table.empty() && table == name)
            {
                rows.push_back(row);
                break;
            }
    }
    return rows;
}

} // namespace oos::input
