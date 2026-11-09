#pragma once

#include "oos/base/arrow/ColumnViews.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::base
{

// One column a table must have: its name and its type in the words of the data contract ("int32", "float64",
// "dictionary<utf8>", "list<int32>", "fixed_list<float64>[3]", as contractTypeOf names them). An optional column may
// be absent; if present, its type is checked all the same.
//
// Some columns come in more than one layout: O_o_Reader gives the grid points of the elements of one type as
// "fixed_list<int32>[4]" and those of mixed types as "list<int32>". `otherTypes` lists the layouts accepted besides
// `type`; a fixed list written with the width "[*]" ("fixed_list<int32>[*]") accepts any width.
struct ColumnContract
{
    std::string name;
    std::string type;
    bool isOptional = false;
    std::vector<std::string> otherTypes = {};
};

// Whether `found` (a type as contractTypeOf names it) is one of the types the column accepts.
[[nodiscard]] bool acceptsType(const ColumnContract& column, const std::string& found);

// What a table of O_o_Reader must look like, as O_o_Solver reads it (the table contracts of input/contract/).
struct TableContract
{
    std::string table; // the name used in messages, e.g. "spc_table"
    std::vector<ColumnContract> columns;
};

// Whether a typed accessor accepts a column with null values. The solver never guesses a missing value: by default a
// column with nulls is refused; a caller that handles them asks with Allow and reads validity() — the value of a null
// row is whatever the producer left there.
enum class NullPolicy
{
    Refuse,
    Allow
};

// The data contract name of the C++ type of a value ("int32" for std::int32_t, "float64" for double).
template <typename T>
struct ArrowValueType;
template <>
struct ArrowValueType<std::int8_t>
{
    static constexpr std::string_view NAME = "int8";
};
template <>
struct ArrowValueType<std::uint8_t>
{
    static constexpr std::string_view NAME = "uint8";
};
template <>
struct ArrowValueType<std::int16_t>
{
    static constexpr std::string_view NAME = "int16";
};
template <>
struct ArrowValueType<std::uint16_t>
{
    static constexpr std::string_view NAME = "uint16";
};
template <>
struct ArrowValueType<std::int32_t>
{
    static constexpr std::string_view NAME = "int32";
};
template <>
struct ArrowValueType<std::uint32_t>
{
    static constexpr std::string_view NAME = "uint32";
};
template <>
struct ArrowValueType<std::int64_t>
{
    static constexpr std::string_view NAME = "int64";
};
template <>
struct ArrowValueType<std::uint64_t>
{
    static constexpr std::string_view NAME = "uint64";
};
template <>
struct ArrowValueType<float>
{
    static constexpr std::string_view NAME = "float32";
};
template <>
struct ArrowValueType<double>
{
    static constexpr std::string_view NAME = "float64";
};

// A fixed-width value type that Arrow stores as is, in place.
template <typename T>
concept ArrowPrimitive = requires { ArrowValueType<T>::NAME; };

// Reads a table of O_o_Reader by its contract. At construction it checks that every column of the
// contract is present with its type, and refuses the table otherwise with ContractError naming the table, the column,
// the type expected and the type found (or, for a missing column, the columns there are). The typed accessors check
// the type again (they may read a column outside the contract), refuse nulls unless asked (NullPolicy), check the
// offsets and indices they will follow, and return views into the producer's buffers: no copy at all (booleans are
// read as bits, narrow offsets and indices in place), so nothing the reader hands out grows the memory with the model;
// the views are valid as long as the reader lives.
//
// Every Arrow type O_o_Reader exports is read: integers, reals, booleans, texts, dictionaries of texts, lists and
// fixed-size lists of numbers, lists of fixed-size lists, and nulls (validity bitmaps, and the null type).
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: owns the table.
class ContractReader
{
public:
    ContractReader(ImportedTable table, TableContract contract);

    [[nodiscard]] const std::string& tableName() const noexcept { return m_contract.table; }
    [[nodiscard]] const TableContract& contract() const noexcept { return m_contract; }
    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_table.rowCount(); }
    [[nodiscard]] bool hasColumn(const std::string& name) const { return m_table.findColumn(name) >= 0; }
    // The data contract type of a column (contractTypeOf); ContractError when there is no such column.
    [[nodiscard]] std::string columnType(const std::string& name) const;

    [[nodiscard]] ValidityView validity(const std::string& name) const;
    [[nodiscard]] std::int64_t nullCount(const std::string& name) const { return validity(name).nullCount(); }

    // A numeric column ("int32", "float64", ...), in place.
    template <ArrowPrimitive T>
    [[nodiscard]] std::span<const T> values(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const
    {
        const void* data = primitiveValues(name, ArrowValueType<T>::NAME, sizeof(T), nulls);
        return {static_cast<const T*>(data), static_cast<std::size_t>(rowCount())};
    }

    // A "bool" column, in place (Arrow packs booleans in bits).
    [[nodiscard]] BooleanView booleans(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const;
    // A "utf8" column, in place.
    [[nodiscard]] Utf8View texts(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const;
    // A "dictionary<utf8>" column, in place whatever the width of its indices.
    [[nodiscard]] DictionaryView dictionaryTexts(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const;
    // A "list<utf8>" column, in place for both offset widths. The nulls policy applies to the rows and to the texts.
    [[nodiscard]] TextListView textLists(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const;

    // A "list<T>" column, in place for both offset widths. The nulls policy applies to the rows and to the values.
    template <ArrowPrimitive T>
    [[nodiscard]] ListView<T> lists(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const
    {
        const ListParts parts = listParts(name, ArrowValueType<T>::NAME, sizeof(T), nulls);
        return {parts.offsets, static_cast<const T*>(parts.values), rowCount()};
    }

    // A "list<fixed_list<T>[n]>" column, in place, whatever n (width() gives it). The nulls policy applies to the
    // rows, the items and the values.
    template <ArrowPrimitive T>
    [[nodiscard]] NestedListView<T> nestedLists(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const
    {
        const NestedListParts parts = nestedListParts(name, ArrowValueType<T>::NAME, sizeof(T), nulls);
        return {parts.offsets, static_cast<const T*>(parts.values), rowCount(), parts.width};
    }

    // A "fixed_list<T>[n]" column, in place, whatever n (width() gives it). The nulls policy applies to the rows and
    // to the values.
    template <ArrowPrimitive T>
    [[nodiscard]] FixedListView<T> fixedLists(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const
    {
        const FixedListParts parts = fixedListParts(name, ArrowValueType<T>::NAME, sizeof(T), nulls);
        return {static_cast<const T*>(parts.values), rowCount(), parts.width};
    }

    // A column of reals as rows of values, in place: a "float64" column gives one value per row (width 1), a
    // "fixed_list<float64>[W]" column W values per row. ContractError for another type.
    [[nodiscard]] FixedListView<double> realRows(const std::string& name, NullPolicy nulls = NullPolicy::Refuse) const;

    [[nodiscard]] const ImportedTable& table() const noexcept { return m_table; }

private:
    struct ListParts
    {
        ListOffsets offsets;
        const void* values = nullptr;
    };
    struct NestedListParts
    {
        ListOffsets offsets; // in items
        const void* values = nullptr;
        std::int64_t width = 0;
    };
    struct FixedListParts
    {
        const void* values = nullptr;
        std::int64_t width = 0;
    };

    [[nodiscard]] std::int64_t columnIndex(const std::string& name) const;
    // The index of a column of the expected type; ContractError naming both types otherwise.
    [[nodiscard]] std::int64_t typedColumn(const std::string& name, std::string_view expected) const;
    // ContractError when the rows [firstRow, firstRow + rows) of `array` (column `index` itself, or the part of it
    // `check` names) hold a null; a check passed once is recorded in the table and not run again.
    void requireNoNulls(const std::string& name, std::int64_t index, ImportedTable::ColumnCheck check,
                        const ArrowArray& array, std::int64_t firstRow, std::int64_t rows) const;
    [[nodiscard]] const void* primitiveValues(const std::string& name, std::string_view type, std::size_t valueSize,
                                              NullPolicy nulls) const;
    // The offsets of a list column of this table, checked (non-decreasing, from 0 or more, within childLength):
    // the one validator of every list layout.
    [[nodiscard]] ListOffsets checkedListOffsets(const std::string& name, std::int64_t index, const ArrowSchema& schema,
                                                 const ArrowArray& array, std::int64_t childLength) const;
    [[nodiscard]] ListParts listParts(const std::string& name, std::string_view valueType, std::size_t valueSize,
                                      NullPolicy nulls) const;
    [[nodiscard]] NestedListParts nestedListParts(const std::string& name, std::string_view valueType,
                                                  std::size_t valueSize, NullPolicy nulls) const;
    [[nodiscard]] FixedListParts fixedListParts(const std::string& name, std::string_view valueType,
                                                std::size_t valueSize, NullPolicy nulls) const;

    ImportedTable m_table;
    TableContract m_contract;
};

} // namespace oos::base
