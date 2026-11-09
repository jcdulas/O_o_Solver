#include "oos/base/arrow/ContractReader.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace oos::base
{

namespace
{

constexpr std::int64_t MAX_INT32 = std::numeric_limits<std::int32_t>::max();

const char* const WHAT_TO_DO =
    "; install a compatible pair of O_o_Reader and O_o_Solver (see the compatibility table "
    "of the user guide)";

const std::byte* bytesOf(const void* buffer)
{
    return static_cast<const std::byte*>(buffer);
}

// The first row of a column in its own buffers: its offset plus the offset of the table (the struct array).
std::int64_t firstRowOf(const ArrowArray& column, std::int64_t tableOffset)
{
    return column.offset + tableOffset;
}

// True when the offsets [offsets[0], offsets[count]] never decrease, start at 0 or more and end within `limit`.
template <typename Offset>
bool areOffsetsValid(const Offset* offsets, std::int64_t count, std::int64_t limit)
{
    if (offsets[0] < 0)
        return false;
    for (std::int64_t row = 0; row < count; ++row)
    {
        if (offsets[row + 1] < offsets[row])
            return false;
    }
    return static_cast<std::int64_t>(offsets[count]) <= limit;
}

} // namespace

ContractReader::ContractReader(ImportedTable table, TableContract contract)
    : m_table(std::move(table)), m_contract(std::move(contract))
{
    for (const ColumnContract& column : m_contract.columns)
    {
        const std::int64_t index = m_table.findColumn(column.name);
        if (index < 0)
        {
            if (column.isOptional)
                continue;
            std::string found;
            for (const std::string& name : m_table.columnNames())
                found.append(found.empty() ? "" : ", ").append(name);
            throw ContractError(
                concatenate({"table '", m_contract.table, "': expected the column '", column.name, "' (", column.type,
                             ") of the data contract, found only the columns [", found, "]", WHAT_TO_DO}));
        }
        const std::string found = contractTypeOf(m_table.columnSchema(index));
        if (!acceptsType(column, found))
        {
            std::string expected = column.type;
            for (const std::string& other : column.otherTypes)
                expected.append(" or ").append(other);
            throw ContractError(
                concatenate({"table '", m_contract.table, "', column '", column.name, "': expected the type ", expected,
                             ", found the type ", found, WHAT_TO_DO}));
        }
    }
}

bool acceptsType(const ColumnContract& column, const std::string& found)
{
    // "fixed_list<int32>[*]" matches "fixed_list<int32>[4]": the same text up to the opening bracket of the width.
    const auto matches = [&found](const std::string& accepted)
    {
        if (accepted.ends_with("[*]"))
        {
            const std::string_view prefix(accepted.data(), accepted.size() - 2);
            return found.starts_with(prefix) && found.ends_with("]") && found.size() > prefix.size() + 1;
        }
        return found == accepted;
    };
    if (matches(column.type))
        return true;
    for (const std::string& other : column.otherTypes)
    {
        if (matches(other))
            return true;
    }
    return false;
}

std::int64_t ContractReader::columnIndex(const std::string& name) const
{
    const std::int64_t index = m_table.findColumn(name);
    if (index < 0)
        throw ContractError(concatenate(
            {"table '", m_contract.table, "' has no column '", name, "', which the solver reads", WHAT_TO_DO}));
    return index;
}

std::string ContractReader::columnType(const std::string& name) const
{
    return contractTypeOf(m_table.columnSchema(columnIndex(name)));
}

std::int64_t ContractReader::typedColumn(const std::string& name, std::string_view expected) const
{
    const std::int64_t index = columnIndex(name);
    const std::string found = contractTypeOf(m_table.columnSchema(index));
    if (found != expected)
        throw ContractError(concatenate({"table '", m_contract.table, "', column '", name, "': expected the type ",
                                         expected, ", found the type ", found, WHAT_TO_DO}));
    return index;
}

void ContractReader::requireNoNulls(const std::string& name, std::int64_t index, ImportedTable::ColumnCheck check,
                                    const ArrowArray& array, std::int64_t firstRow, std::int64_t rows) const
{
    if (array.n_buffers == 0 || array.buffers[0] == nullptr || array.null_count == 0 || m_table.hasPassed(index, check))
        return;
    // null_count may be -1 (not computed) or count rows outside the table's slice: the bits decide.
    const std::int64_t nulls =
        ValidityView(static_cast<const std::uint8_t*>(array.buffers[0]), firstRow, rows).nullCount();
    if (nulls > 0)
        throw ContractError(
            concatenate({"table '", m_contract.table, "', column '", name,
                         "': expected a value in "
                         "every row, found ",
                         std::to_string(nulls), " nulls; the data contract promises a value here", WHAT_TO_DO}));
    m_table.markPassed(index, check);
}

ValidityView ContractReader::validity(const std::string& name) const
{
    const std::int64_t index = columnIndex(name);
    const ArrowArray& array = m_table.columnArray(index);
    if (contractTypeOf(m_table.columnSchema(index)) == "null")
        return {nullptr, 0, rowCount(), true};
    if (array.n_buffers == 0 || array.buffers[0] == nullptr || array.null_count == 0)
        return {nullptr, 0, rowCount()};
    return {static_cast<const std::uint8_t*>(array.buffers[0]), firstRowOf(array, m_table.offset()), rowCount()};
}

const void* ContractReader::primitiveValues(const std::string& name, std::string_view type, std::size_t valueSize,
                                            NullPolicy nulls) const
{
    const std::int64_t index = typedColumn(name, type);
    const ArrowArray& array = m_table.columnArray(index);
    const std::int64_t first = firstRowOf(array, m_table.offset());
    if (nulls == NullPolicy::Refuse)
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rowCount());
    return bytesOf(array.buffers[1]) + static_cast<std::size_t>(first) * valueSize;
}

BooleanView ContractReader::booleans(const std::string& name, NullPolicy nulls) const
{
    const std::int64_t index = typedColumn(name, "bool");
    const ArrowArray& array = m_table.columnArray(index);
    const std::int64_t first = firstRowOf(array, m_table.offset());
    if (nulls == NullPolicy::Refuse)
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rowCount());
    // The values of a boolean column are a bitmap laid out as a validity bitmap.
    return {static_cast<const std::uint8_t*>(array.buffers[1]), first, rowCount()};
}

Utf8View ContractReader::texts(const std::string& name, NullPolicy nulls) const
{
    const std::int64_t index = typedColumn(name, "utf8");
    const ArrowArray& array = m_table.columnArray(index);
    const std::int64_t first = firstRowOf(array, m_table.offset());
    if (nulls == NullPolicy::Refuse)
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rowCount());
    const auto* offsets = static_cast<const std::int32_t*>(array.buffers[1]) + first;
    if (!m_table.hasPassed(index, ImportedTable::ColumnCheck::TextOffsets))
    {
        if (rowCount() > 0 && !areOffsetsValid(offsets, rowCount(), MAX_INT32))
            throw ContractError(concatenate({"table '", m_contract.table, "', column '", name,
                                             "': expected non-decreasing text offsets, found others", WHAT_TO_DO}));
        m_table.markPassed(index, ImportedTable::ColumnCheck::TextOffsets);
    }
    return {offsets, static_cast<const char*>(array.buffers[2]), rowCount()};
}

DictionaryView ContractReader::dictionaryTexts(const std::string& name, NullPolicy nulls) const
{
    const std::int64_t index = typedColumn(name, "dictionary<utf8>");
    const ArrowSchema& schema = m_table.columnSchema(index);
    const ArrowArray& array = m_table.columnArray(index);
    const std::int64_t first = firstRowOf(array, m_table.offset());
    const std::int64_t rows = rowCount();
    if (nulls == NullPolicy::Refuse)
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rows);

    const ArrowArray& dictionary = *array.dictionary;
    requireNoNulls(name, index, ImportedTable::ColumnCheck::DictionaryNulls, dictionary, dictionary.offset,
                   dictionary.length);
    const auto* dictionaryOffsets = static_cast<const std::int32_t*>(dictionary.buffers[1]) + dictionary.offset;
    if (!m_table.hasPassed(index, ImportedTable::ColumnCheck::DictionaryOffsets))
    {
        if (dictionary.length > 0 && !areOffsetsValid(dictionaryOffsets, dictionary.length, MAX_INT32))
            throw ContractError(
                concatenate({"table '", m_contract.table, "', column '", name,
                             "': expected non-decreasing offsets in the dictionary, found others", WHAT_TO_DO}));
        m_table.markPassed(index, ImportedTable::ColumnCheck::DictionaryOffsets);
    }
    const Utf8View values(dictionaryOffsets, static_cast<const char*>(dictionary.buffers[2]), dictionary.length);

    // The index type is the format of the column itself (the dictionary holds the value type); the indices are read
    // in place whatever their width.
    const std::string_view indexFormat = schema.format;
    std::size_t indexBytes = 0;
    if (indexFormat == "c")
        indexBytes = 1;
    else if (indexFormat == "s")
        indexBytes = 2;
    else if (indexFormat == "i")
        indexBytes = 4;
    else if (indexFormat == "l")
        indexBytes = 8;
    else
        throw ContractError(concatenate({"table '", m_contract.table, "', column '", name,
                                         "': expected dictionary indices of a signed integer type (int8, int16, "
                                         "int32, int64), found the format '",
                                         indexFormat, "'", WHAT_TO_DO}));
    const DictionaryView view(bytesOf(array.buffers[1]) + static_cast<std::size_t>(first) * indexBytes,
                              static_cast<int>(indexBytes), rows, values);

    // An index outside int32 reads as -1 (DictionaryView::index), so this check refuses it too.
    if (m_table.hasPassed(index, ImportedTable::ColumnCheck::DictionaryIndices))
        return view;
    const ValidityView valid = validity(name);
    for (std::int64_t row = 0; row < rows; ++row)
    {
        const std::int32_t position = view.index(row);
        if (valid.isValid(row) && (position < 0 || position >= dictionary.length))
            throw ContractError(
                concatenate({"table '", m_contract.table, "', column '", name, "', row ", std::to_string(row),
                             ": expected a dictionary index below ", std::to_string(dictionary.length), ", found ",
                             std::to_string(position), WHAT_TO_DO}));
    }
    m_table.markPassed(index, ImportedTable::ColumnCheck::DictionaryIndices);
    return view;
}

TextListView ContractReader::textLists(const std::string& name, NullPolicy nulls) const
{
    const std::int64_t index = typedColumn(name, "list<utf8>");
    const ArrowSchema& schema = m_table.columnSchema(index);
    const ArrowArray& array = m_table.columnArray(index);
    const ArrowArray& child = *array.children[0];
    const std::int64_t first = firstRowOf(array, m_table.offset());
    const std::int64_t rows = rowCount();
    const ListOffsets offsets = checkedListOffsets(name, index, schema, array, child.length);
    if (nulls == NullPolicy::Refuse)
    {
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rows);
        requireNoNulls(name, index, ImportedTable::ColumnCheck::ChildNulls, child, child.offset + offsets[0],
                       offsets[rows] - offsets[0]);
    }
    // The list offsets count texts from the first of the child (its own offset applied here, as listParts does).
    const auto* textOffsets = static_cast<const std::int32_t*>(child.buffers[1]) + child.offset;
    if (!m_table.hasPassed(index, ImportedTable::ColumnCheck::TextOffsets))
    {
        if (child.length > 0 && !areOffsetsValid(textOffsets, child.length, MAX_INT32))
            throw ContractError(
                concatenate({"table '", m_contract.table, "', column '", name,
                             "': expected non-decreasing text offsets in the list, found others", WHAT_TO_DO}));
        m_table.markPassed(index, ImportedTable::ColumnCheck::TextOffsets);
    }
    return {offsets, Utf8View(textOffsets, static_cast<const char*>(child.buffers[2]), child.length), rows};
}

ListOffsets ContractReader::checkedListOffsets(const std::string& name, std::int64_t index, const ArrowSchema& schema,
                                               const ArrowArray& array, std::int64_t childLength) const
{
    const bool isChecked = m_table.hasPassed(index, ImportedTable::ColumnCheck::ListOffsets);
    const std::int64_t first = firstRowOf(array, m_table.offset());
    const std::int64_t rows = rowCount();
    bool isValid = false;
    ListOffsets offsets;
    if (std::string_view(schema.format) == "+L")
    {
        const auto* wide = static_cast<const std::int64_t*>(array.buffers[1]) + first;
        isValid = isChecked || areOffsetsValid(wide, rows, childLength);
        offsets = ListOffsets(nullptr, wide);
    }
    else
    {
        const auto* narrow = static_cast<const std::int32_t*>(array.buffers[1]) + first;
        isValid = isChecked || areOffsetsValid(narrow, rows, childLength);
        offsets = ListOffsets(narrow, nullptr);
    }
    if (!isValid)
        throw ContractError(concatenate(
            {"table '", m_contract.table, "', column '", name, "': expected non-decreasing list offsets within the ",
             std::to_string(childLength), " entries of the list, found others", WHAT_TO_DO}));
    m_table.markPassed(index, ImportedTable::ColumnCheck::ListOffsets);
    return offsets;
}

ContractReader::ListParts ContractReader::listParts(const std::string& name, std::string_view valueType,
                                                    std::size_t valueSize, NullPolicy nulls) const
{
    const std::int64_t index = typedColumn(name, concatenate({"list<", valueType, ">"}));
    const ArrowSchema& schema = m_table.columnSchema(index);
    const ArrowArray& array = m_table.columnArray(index);
    const ArrowArray& child = *array.children[0];
    const std::int64_t first = firstRowOf(array, m_table.offset());
    const std::int64_t rows = rowCount();

    ListParts parts;
    parts.offsets = checkedListOffsets(name, index, schema, array, child.length);
    if (nulls == NullPolicy::Refuse)
    {
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rows);
        requireNoNulls(name, index, ImportedTable::ColumnCheck::ChildNulls, child, child.offset + parts.offsets[0],
                       parts.offsets[rows] - parts.offsets[0]);
    }
    parts.values = bytesOf(child.buffers[1]) + static_cast<std::size_t>(child.offset) * valueSize;
    return parts;
}

ContractReader::NestedListParts ContractReader::nestedListParts(const std::string& name, std::string_view valueType,
                                                                std::size_t valueSize, NullPolicy nulls) const
{
    const ArrowSchema& columnSchema = m_table.columnSchema(columnIndex(name));
    // The width is the item type's: "list<fixed_list<float64>[6]>" is expected when the items have 6 values.
    const bool hasItem =
        columnSchema.n_children == 1 && columnSchema.children != nullptr && columnSchema.children[0] != nullptr;
    const std::int64_t width = hasItem ? fixedListWidth(*columnSchema.children[0]) : 0;
    const std::string widthText = width > 0 ? std::to_string(width) : std::string("n");
    const std::int64_t index = typedColumn(name, concatenate({"list<fixed_list<", valueType, ">[", widthText, "]>"}));
    const ArrowArray& array = m_table.columnArray(index);
    const ArrowArray& items = *array.children[0];
    const ArrowArray& values = *items.children[0];
    const std::int64_t first = firstRowOf(array, m_table.offset());
    const std::int64_t rows = rowCount();

    NestedListParts parts;
    parts.width = width;
    parts.offsets = checkedListOffsets(name, index, m_table.columnSchema(index), array, items.length);
    const std::int64_t firstItem = parts.offsets[0];
    const std::int64_t itemTotal = parts.offsets[rows] - firstItem;
    if (nulls == NullPolicy::Refuse)
    {
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rows);
        requireNoNulls(name, index, ImportedTable::ColumnCheck::ChildNulls, items, items.offset + firstItem, itemTotal);
        requireNoNulls(name, index, ImportedTable::ColumnCheck::GrandchildNulls, values,
                       values.offset + (items.offset + firstItem) * width, itemTotal * width);
    }
    parts.values =
        bytesOf(values.buffers[1]) + static_cast<std::size_t>(values.offset + items.offset * width) * valueSize;
    return parts;
}

FixedListView<double> ContractReader::realRows(const std::string& name, NullPolicy nulls) const
{
    const std::string type = columnType(name);
    if (type == "float64")
        return {values<double>(name, nulls).data(), rowCount(), 1};
    if (type.starts_with("fixed_list<float64>["))
        return fixedLists<double>(name, nulls);
    throw ContractError(
        concatenate({"table '", m_contract.table, "', column '", name,
                     "': expected a float64 or fixed_list<float64>[W] column, found the type ", type, WHAT_TO_DO}));
}

ContractReader::FixedListParts ContractReader::fixedListParts(const std::string& name, std::string_view valueType,
                                                              std::size_t valueSize, NullPolicy nulls) const
{
    const std::int64_t width = fixedListWidth(m_table.columnSchema(columnIndex(name)));
    const std::string widthText = width > 0 ? std::to_string(width) : std::string("n");
    const std::int64_t index = typedColumn(name, concatenate({"fixed_list<", valueType, ">[", widthText, "]"}));
    const ArrowArray& array = m_table.columnArray(index);
    const ArrowArray& child = *array.children[0];
    const std::int64_t first = firstRowOf(array, m_table.offset());
    if (nulls == NullPolicy::Refuse)
    {
        requireNoNulls(name, index, ImportedTable::ColumnCheck::Nulls, array, first, rowCount());
        requireNoNulls(name, index, ImportedTable::ColumnCheck::ChildNulls, child, child.offset + first * width,
                       rowCount() * width);
    }
    return {bytesOf(child.buffers[1]) + static_cast<std::size_t>(child.offset + first * width) * valueSize, width};
}

} // namespace oos::base
