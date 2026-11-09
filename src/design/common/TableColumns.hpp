#pragma once

// Column readers shared by the design table readers (model/DesignTables, responses/ResponseTables,
// constraints/ConstraintTables): the contract helpers of the design tables of O_o_Reader. Private to src/design.

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::design::table_columns
{

inline constexpr const char* INT32 = "int32";
inline constexpr const char* FLOAT64 = "float64";
inline constexpr const char* UTF8 = "utf8";
inline constexpr const char* TEXTS = "dictionary<utf8>";
inline constexpr const char* BOOLEAN = "bool";
inline constexpr const char* INT32_LIST = "list<int32>";

[[nodiscard]] inline base::ColumnContract required(std::string name, std::string type)
{
    return {std::move(name), std::move(type), false, {}};
}

[[nodiscard]] inline base::ColumnContract optional(std::string name, std::string type)
{
    return {std::move(name), std::move(type), true, {}};
}

// A text column, utf8 or dictionary<utf8>.
[[nodiscard]] inline base::ColumnContract text(std::string name)
{
    return {std::move(name), UTF8, false, {TEXTS}};
}

[[nodiscard]] inline base::ColumnContract optionalText(std::string name)
{
    return {std::move(name), UTF8, true, {TEXTS}};
}

// A text column read in either layout; a null or an absent optional column reads "".
class TextColumn
{
public:
    TextColumn(const base::ContractReader& reader, const std::string& name)
    {
        if (!reader.hasColumn(name))
            return;
        m_validity.emplace(reader.validity(name));
        if (reader.columnType(name) == UTF8)
            m_utf8.emplace(reader.texts(name, base::NullPolicy::Allow));
        else
            m_dictionary.emplace(reader.dictionaryTexts(name, base::NullPolicy::Allow));
    }
    [[nodiscard]] std::string operator[](std::int64_t row) const
    {
        if (!m_validity || !m_validity->isValid(row))
            return {};
        return std::string(m_utf8 ? (*m_utf8)[row] : (*m_dictionary)[row]);
    }

private:
    std::optional<base::ValidityView> m_validity;
    std::optional<base::Utf8View> m_utf8;
    std::optional<base::DictionaryView> m_dictionary;
};

// A nullable column of numbers: the values and their validity, or nothing when the optional column is absent.
template <typename T>
class NullableColumn
{
public:
    NullableColumn(const base::ContractReader& reader, const std::string& name)
    {
        if (!reader.hasColumn(name))
            return;
        m_values = reader.values<T>(name, base::NullPolicy::Allow);
        m_validity = reader.validity(name);
        m_isPresent = true;
    }
    [[nodiscard]] std::optional<T> operator[](std::int64_t row) const
    {
        if (!m_isPresent || !m_validity.isValid(row))
            return std::nullopt;
        return m_values[static_cast<std::size_t>(row)];
    }

private:
    std::span<const T> m_values;
    // A flag beside a plain view rather than an optional view: GCC 12 reports the optional's payload as maybe
    // uninitialized (-Wmaybe-uninitialized, a false positive) once several columns sit in a std::array.
    base::ValidityView m_validity;
    bool m_isPresent = false;
};

// A nullable list<int32> column (empty for a null row or an absent optional column).
class IntListColumn
{
public:
    IntListColumn(const base::ContractReader& reader, const std::string& name)
    {
        if (!reader.hasColumn(name))
            return;
        m_lists.emplace(reader.lists<std::int32_t>(name, base::NullPolicy::Allow));
        m_validity.emplace(reader.validity(name));
    }
    [[nodiscard]] std::vector<std::int32_t> operator[](std::int64_t row) const
    {
        if (!m_lists || !m_validity->isValid(row))
            return {};
        const std::span<const std::int32_t> values = (*m_lists)[row];
        return {values.begin(), values.end()};
    }

private:
    std::optional<base::ListView<std::int32_t>> m_lists;
    std::optional<base::ValidityView> m_validity;
};

[[noreturn]] inline void refuseRow(const base::ContractReader& table, std::int64_t row, const std::string& what)
{
    throw base::ContractError(
        base::concatenate({"table '", table.tableName(), "', row ", std::to_string(row), ": ", what,
                           "; the producer breaks the design table contract "
                           "(src/design/README.md): take the snapshot again with a "
                           "O_o_Reader that follows it"}));
}

} // namespace oos::design::table_columns
