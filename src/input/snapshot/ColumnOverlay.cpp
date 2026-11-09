#include "oos/input/snapshot/ColumnOverlay.hpp"

#include "oos/base/arrow/ArrowStructures.hpp"
#include "oos/base/arrow/ImportedTable.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/ContractVersion.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/input/snapshot/SnapshotBuilder.hpp"
#include "oos/input/tables/BorrowedArray.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::input
{

namespace
{

const std::string ELEMENTS_PREFIX = "elements/";
const std::string PROPERTIES_PREFIX = "properties/";
const std::string METHODS_PREFIX = "methods/";
const std::string MATERIALS_PREFIX = "materials/";
const std::string LOAD_CARDS_PREFIX = "loadCards/";
const std::string AERO_CARDS_PREFIX = "aeroCards/";

// Arrow validity bitmaps: bit i of the array is bit (i % 8) of byte i / 8, least significant bit first.
[[nodiscard]] bool isBitSet(const void* bitmap, std::int64_t index) noexcept
{
    const auto* bytes = static_cast<const std::uint8_t*>(bitmap);
    return ((bytes[index / 8] >> (index % 8)) & 1U) != 0;
}

void setBit(std::vector<std::uint8_t>& bitmap, std::int64_t index) noexcept
{
    bitmap[static_cast<std::size_t>(index / 8)] |= static_cast<std::uint8_t>(1U << (index % 8));
}

// A column array whose buffers the overlay owns: the copy of a "float64" column, or of a fixed list with its child of
// values. The structure keeps its address (held by unique_ptr) because its parent points to it.
struct OwnedColumn
{
    ArrowArray array{};
    std::vector<const void*> buffers;
    std::vector<std::uint8_t> validity;
    std::vector<double> values;
    std::unique_ptr<OwnedColumn> child;
    ArrowArray* childPointer = nullptr;
};

// What a shared table owns: the child pointer arrays of its two top-level structures, the copied columns, and the
// reference that keeps the base snapshot (and so the borrowed children) alive. Held by both structures.
struct SharedTable
{
    std::shared_ptr<const void> keepAlive;
    std::vector<ArrowSchema*> schemaChildren;
    std::vector<ArrowArray*> arrayChildren;
    std::vector<std::unique_ptr<OwnedColumn>> columns;
    // The schemas of the columns added to the source (appendRealColumns) and their names, at fixed addresses.
    std::vector<std::unique_ptr<ArrowSchema>> addedSchemas;
    std::deque<std::string> addedNames;
    std::vector<char> metadata;
    std::vector<const void*> topBuffers{nullptr};
};

// The metadata key by which O_o_Reader stamps its tables with their contract version.
const std::string CONTRACT_VERSION_KEY(DATA_CONTRACT_VERSION_KEY);

// Arrow C Data Interface metadata of one pair: an int32 pair count, then the int32 length and bytes of the key and of
// the value (native byte order).
std::vector<char> encodeMetadata(const std::string& key, const std::string& value)
{
    std::vector<char> bytes;
    const auto appendInt32 = [&bytes](std::int32_t number)
    {
        const auto* raw = reinterpret_cast<const char*>(&number);
        bytes.insert(bytes.end(), raw, raw + sizeof(number));
    };
    appendInt32(1);
    appendInt32(static_cast<std::int32_t>(key.size()));
    bytes.insert(bytes.end(), key.begin(), key.end());
    appendInt32(static_cast<std::int32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
    return bytes;
}

using SharedPointer = std::shared_ptr<SharedTable>;

void releaseSharedSchema(ArrowSchema* schema)
{
    delete static_cast<SharedPointer*>(schema->private_data);
    schema->private_data = nullptr;
    schema->release = nullptr;
}

void releaseSharedArray(ArrowArray* array)
{
    delete static_cast<SharedPointer*>(array->private_data);
    array->private_data = nullptr;
    array->release = nullptr;
}

// The validity of `rows` rows of a source array starting at its row `first` (its offset included), as a bitmap of
// rows `lead` + r of a new array of offset 0; empty when the source has no nulls. `extra` reserves rows after them.
std::vector<std::uint8_t> copyValidity(const ArrowArray& source, std::int64_t first, std::int64_t lead,
                                       std::int64_t rows)
{
    if (source.null_count == 0 || source.n_buffers == 0 || source.buffers[0] == nullptr)
        return {};
    std::vector<std::uint8_t> bitmap(static_cast<std::size_t>((lead + rows + 7) / 8), 0);
    for (std::int64_t r = 0; r < rows; ++r)
    {
        if (isBitSet(source.buffers[0], first + r))
            setBit(bitmap, lead + r);
    }
    return bitmap;
}

// Sets the null count and the validity buffer of an owned array of `length` rows, the first `lead` of which are not
// read (rows before the table's offset): an all-valid bitmap is dropped.
void finishValidity(OwnedColumn& column, std::int64_t lead, std::int64_t length)
{
    std::int64_t nulls = 0;
    if (!column.validity.empty())
    {
        for (std::int64_t r = lead; r < length; ++r)
            nulls += isBitSet(column.validity.data(), r) ? 0 : 1;
        if (nulls == 0)
            column.validity.clear();
    }
    column.array.null_count = nulls;
    column.buffers[0] = column.validity.empty() ? nullptr : column.validity.data();
}

// Marks row `row` of an owned array valid (a design value given to a field that was blank).
void markValid(OwnedColumn& column, std::int64_t row)
{
    if (!column.validity.empty())
        setBit(column.validity, row);
}

// A "float64" array of `values` (offset 0), with `validity`.
std::unique_ptr<OwnedColumn> makeRealArray(std::vector<double> values, std::vector<std::uint8_t> validity)
{
    auto column = std::make_unique<OwnedColumn>();
    column->values = std::move(values);
    column->validity = std::move(validity);
    column->buffers = {nullptr, column->values.data()};
    column->array.length = static_cast<std::int64_t>(column->values.size());
    column->array.offset = 0;
    column->array.n_buffers = 2;
    column->array.buffers = column->buffers.data();
    column->array.n_children = 0;
    column->array.children = nullptr;
    column->array.dictionary = nullptr;
    // The release of an owned column is never called: its parent's release (releaseSharedArray) frees it.
    column->array.release = nullptr;
    column->array.private_data = nullptr;
    return column;
}

// The copy of a "float64" column of a table of `rows` rows at struct offset `lead`, with the cells set.
std::unique_ptr<OwnedColumn> copyRealColumn(const ArrowArray& source, std::int64_t lead, std::int64_t rows,
                                            const std::vector<CellValue>& cells)
{
    const std::int64_t first = source.offset + lead;
    const auto* data = static_cast<const double*>(source.buffers[1]);
    std::vector<double> values(static_cast<std::size_t>(lead + rows), 0.0);
    std::copy(data + first, data + first + rows, values.begin() + lead);
    auto column = makeRealArray(std::move(values), copyValidity(source, first, lead, rows));
    for (const CellValue& cell : cells)
    {
        column->values[static_cast<std::size_t>(lead + cell.row)] = cell.value;
        markValid(*column, lead + cell.row);
    }
    finishValidity(*column, lead, lead + rows);
    return column;
}

// The copy of a "fixed_list<float64>[width]" column: the list array (its row validity) and its child of values.
std::unique_ptr<OwnedColumn> copyFixedListColumn(const ArrowArray& source, std::int64_t width, std::int64_t lead,
                                                 std::int64_t rows, const std::vector<CellValue>& cells)
{
    const ArrowArray& child = *source.children[0];
    const std::int64_t firstRow = source.offset + lead;
    const std::int64_t firstValue = child.offset + firstRow * width;
    const auto* data = static_cast<const double*>(child.buffers[1]);
    std::vector<double> values(static_cast<std::size_t>((lead + rows) * width), 0.0);
    std::copy(data + firstValue, data + firstValue + rows * width, values.begin() + lead * width);
    auto valueArray = makeRealArray(std::move(values), copyValidity(child, firstValue, lead * width, rows * width));

    auto column = std::make_unique<OwnedColumn>();
    column->validity = copyValidity(source, firstRow, lead, rows);
    column->buffers = {nullptr};
    column->array.length = lead + rows;
    column->array.offset = 0;
    column->array.n_buffers = 1;
    column->array.buffers = column->buffers.data();
    column->array.n_children = 1;
    column->array.dictionary = nullptr;
    column->array.release = nullptr;
    column->array.private_data = nullptr;
    for (const CellValue& cell : cells)
    {
        const std::int64_t valueIndex = (lead + cell.row) * width + cell.item;
        valueArray->values[static_cast<std::size_t>(valueIndex)] = cell.value;
        markValid(*valueArray, valueIndex);
        markValid(*column, lead + cell.row);
    }
    finishValidity(*valueArray, lead * width, (lead + rows) * width);
    finishValidity(*column, lead, lead + rows);
    column->child = std::move(valueArray);
    column->childPointer = &column->child->array;
    column->array.children = &column->childPointer;
    return column;
}

// The offset of list slot `slot` of a "+l" (int32) or "+L" (int64) list array: the position of its first item in the
// child array (the child's own offset still to add).
[[nodiscard]] std::int64_t readListOffset(const ArrowArray& list, bool isLarge, std::int64_t slot) noexcept
{
    if (isLarge)
        return static_cast<const std::int64_t*>(list.buffers[1])[slot];
    return static_cast<const std::int32_t*>(list.buffers[1])[slot];
}

// The copy of a "list<fixed_list<float64>[width]>" column (the stations of a PBEAM): the list array keeps the base's
// validity and offsets (borrowed: `keepAlive` holds the base), its child of fixed lists and their values are copied
// whole (the child keeps its offset, so that the list offsets stay right) with the cells set. The item of a cell is
// station · width + component, as O_o_Reader's design_relations_table flattens a list of vectors (field_item).
std::unique_ptr<OwnedColumn> copyNestedListColumn(const ArrowArray& source, bool isLarge, std::int64_t width,
                                                  std::int64_t lead, const std::vector<CellValue>& cells)
{
    const ArrowArray& items = *source.children[0];
    const ArrowArray& values = *items.children[0];
    // Every item slot up to the last one of the child, its own offset included, with its `width` values.
    const std::int64_t slots = items.offset + items.length;
    const auto* data = static_cast<const double*>(values.buffers[1]);
    std::vector<double> copied(static_cast<std::size_t>(slots * width), 0.0);
    std::copy(data + values.offset, data + values.offset + slots * width, copied.begin());
    auto valueArray = makeRealArray(std::move(copied), copyValidity(values, values.offset, 0, slots * width));

    auto itemArray = std::make_unique<OwnedColumn>();
    itemArray->validity = copyValidity(items, 0, 0, slots);
    itemArray->buffers = {nullptr};
    itemArray->array.length = items.length;
    itemArray->array.offset = items.offset;
    itemArray->array.n_buffers = 1;
    itemArray->array.buffers = itemArray->buffers.data();
    itemArray->array.n_children = 1;
    itemArray->array.dictionary = nullptr;
    itemArray->array.release = nullptr;
    itemArray->array.private_data = nullptr;

    for (const CellValue& cell : cells)
    {
        // The row of the table is slot lead + row of the list array (its own offset added); its first item is the
        // child slot items.offset + offset[slot].
        const std::int64_t slot = source.offset + lead + cell.row;
        const std::int64_t station = cell.item / width;
        const std::int64_t itemSlot = items.offset + readListOffset(source, isLarge, slot) + station;
        const std::int64_t valueIndex = itemSlot * width + cell.item % width;
        valueArray->values[static_cast<std::size_t>(valueIndex)] = cell.value;
        markValid(*valueArray, valueIndex);
        markValid(*itemArray, itemSlot);
    }
    finishValidity(*valueArray, 0, slots * width);
    finishValidity(*itemArray, items.offset, slots);
    itemArray->child = std::move(valueArray);
    itemArray->childPointer = &itemArray->child->array;
    itemArray->array.children = &itemArray->childPointer;

    // The list array: the base's validity bitmap and offsets, borrowed, and the copied child.
    auto column = std::make_unique<OwnedColumn>();
    column->buffers = {source.buffers[0], source.buffers[1]};
    column->array.length = source.length;
    column->array.offset = source.offset;
    column->array.null_count = source.null_count;
    column->array.n_buffers = 2;
    column->array.buffers = column->buffers.data();
    column->array.n_children = 1;
    column->array.dictionary = nullptr;
    column->array.release = nullptr;
    column->array.private_data = nullptr;
    column->child = std::move(itemArray);
    column->childPointer = &column->child->array;
    column->array.children = &column->childPointer;
    return column;
}

// The copy of a "list<float64>" column (the plies of a PCOMP, the dimensions of a PBARL or of the stations of a
// PBEAML): the list array keeps the base's validity and offsets (borrowed: `keepAlive` holds the base), its child of
// values is copied whole from its logical start (offset 0 in the copy, so that the list offsets stay right) with the
// cells set. The item of a cell is its position in the list of its row.
std::unique_ptr<OwnedColumn> copyListColumn(const ArrowArray& source, bool isLarge, std::int64_t lead,
                                            const std::vector<CellValue>& cells)
{
    const ArrowArray& values = *source.children[0];
    const auto* data = static_cast<const double*>(values.buffers[1]);
    std::vector<double> copied(static_cast<std::size_t>(values.length), 0.0);
    std::copy(data + values.offset, data + values.offset + values.length, copied.begin());
    auto valueArray = makeRealArray(std::move(copied), copyValidity(values, values.offset, 0, values.length));
    for (const CellValue& cell : cells)
    {
        const std::int64_t slot = source.offset + lead + cell.row;
        const std::int64_t valueIndex = readListOffset(source, isLarge, slot) + cell.item;
        valueArray->values[static_cast<std::size_t>(valueIndex)] = cell.value;
        markValid(*valueArray, valueIndex);
    }
    finishValidity(*valueArray, 0, values.length);

    auto column = std::make_unique<OwnedColumn>();
    column->buffers = {source.buffers[0], source.buffers[1]};
    column->array.length = source.length;
    column->array.offset = source.offset;
    column->array.null_count = source.null_count;
    column->array.n_buffers = 2;
    column->array.buffers = column->buffers.data();
    column->array.n_children = 1;
    column->array.dictionary = nullptr;
    column->array.release = nullptr;
    column->array.private_data = nullptr;
    column->child = std::move(valueArray);
    column->childPointer = &column->child->array;
    column->array.children = &column->childPointer;
    return column;
}

[[noreturn]] void refuseCell(const ColumnOverlay::ColumnEdits& edits, const std::string& what)
{
    throw std::invalid_argument(base::concatenate({"ColumnOverlay: column '", edits.column, "' of table '", edits.table,
                                                   "': ", what, "; overlay only existing cells of real columns"}));
}

// Checks the cells of one column against the table (row range, item range, one value per cell).
void checkCells(const ColumnOverlay::ColumnEdits& edits, std::int64_t rows, std::int64_t width)
{
    std::vector<std::pair<std::int64_t, std::int32_t>> keys;
    keys.reserve(edits.cells.size());
    for (const CellValue& cell : edits.cells)
    {
        if (cell.row < 0 || cell.row >= rows)
            refuseCell(edits, base::concatenate({"expected a row below ", std::to_string(rows), ", found ",
                                                 std::to_string(cell.row)}));
        const bool isItemValid = width == 0 ? cell.item == CellValue::SCALAR_ITEM : cell.item >= 0 && cell.item < width;
        if (!isItemValid)
            refuseCell(edits, base::concatenate({"expected ",
                                                 width == 0 ? std::string("no item (a float64 column)")
                                                            : "an item below " + std::to_string(width),
                                                 ", found item ", std::to_string(cell.item)}));
        keys.emplace_back(cell.row, cell.item);
    }
    std::sort(keys.begin(), keys.end());
    const auto twice = std::adjacent_find(keys.begin(), keys.end());
    if (twice != keys.end())
        refuseCell(edits, base::concatenate({"expected one value per cell, found two for row ",
                                             std::to_string(twice->first), " item ", std::to_string(twice->second)}));
}

// Checks the cells of a nested list column: the row in the table, the item within the stations of its row, one value
// per cell.
void checkNestedCells(const ColumnOverlay::ColumnEdits& edits, const ArrowArray& source, bool isLarge,
                      std::int64_t width, std::int64_t lead, std::int64_t rows)
{
    checkCells(edits, rows, std::numeric_limits<std::int32_t>::max()); // rows, items from 0, one value per cell
    for (const CellValue& cell : edits.cells)
    {
        const std::int64_t slot = source.offset + lead + cell.row;
        const std::int64_t stations = readListOffset(source, isLarge, slot + 1) - readListOffset(source, isLarge, slot);
        if (cell.item >= stations * width)
            refuseCell(edits, base::concatenate({"expected an item below ", std::to_string(stations * width), " (",
                                                 std::to_string(stations), " stations of ", std::to_string(width),
                                                 " values) in row ", std::to_string(cell.row), ", found item ",
                                                 std::to_string(cell.item)}));
    }
}

// A table sharing every column of `source` except those of `edits`, which are copied with their new values, and with
// the float64 columns `added` after them (appendRealColumns).
base::ImportedTable shareTable(const std::string& name, const base::ImportedTable& source,
                               const std::shared_ptr<const void>& keepAlive,
                               const std::vector<const ColumnOverlay::ColumnEdits*>& edits,
                               std::vector<AddedRealColumn> added = {})
{
    auto shared = std::make_shared<SharedTable>();
    shared->keepAlive = keepAlive;
    const std::int64_t sourceColumnCount = source.columnCount();
    for (std::int64_t index = 0; index < sourceColumnCount; ++index)
    {
        shared->schemaChildren.push_back(const_cast<ArrowSchema*>(&source.columnSchema(index)));
        shared->arrayChildren.push_back(const_cast<ArrowArray*>(&source.columnArray(index)));
    }
    for (AddedRealColumn& column : added)
    {
        if (source.findColumn(column.name) >= 0)
            throw std::invalid_argument(base::concatenate(
                {"appendRealColumns: table '", name, "' has a column '", column.name, "' already; name a new column"}));
        if (static_cast<std::int64_t>(column.values.size()) != source.rowCount())
            throw std::invalid_argument(base::concatenate(
                {"appendRealColumns: expected ", std::to_string(source.rowCount()), " values for column '", column.name,
                 "' of table '", name, "', found ", std::to_string(column.values.size())}));
        // The rows before the table's offset are not read: zeros.
        std::vector<double> values(static_cast<std::size_t>(source.offset()), 0.0);
        values.insert(values.end(), column.values.begin(), column.values.end());
        auto array = makeRealArray(std::move(values), {});
        finishValidity(*array, source.offset(), source.offset() + source.rowCount());
        auto schema = std::make_unique<ArrowSchema>();
        shared->addedNames.push_back(std::move(column.name));
        schema->format = "g";
        schema->name = shared->addedNames.back().c_str();
        schema->metadata = nullptr;
        schema->flags = 0;
        schema->n_children = 0;
        schema->children = nullptr;
        schema->dictionary = nullptr;
        // As an owned column's array: the parent's release (releaseSharedSchema) frees it.
        schema->release = nullptr;
        schema->private_data = nullptr;
        shared->schemaChildren.push_back(schema.get());
        shared->arrayChildren.push_back(&array->array);
        shared->addedSchemas.push_back(std::move(schema));
        shared->columns.push_back(std::move(array));
    }
    const auto columnCount = static_cast<std::int64_t>(shared->schemaChildren.size());
    std::vector<std::uint8_t> isBorrowed(static_cast<std::size_t>(columnCount), 1); // bounded by the columns
    for (std::size_t index = static_cast<std::size_t>(sourceColumnCount); index < isBorrowed.size(); ++index)
        isBorrowed[index] = 0;
    for (const ColumnOverlay::ColumnEdits* columnEdits : edits)
    {
        const std::int64_t index = source.findColumn(columnEdits->column);
        if (index < 0)
            refuseCell(*columnEdits, "the table has no such column");
        isBorrowed[static_cast<std::size_t>(index)] = 0;
        const std::string type = base::contractTypeOf(source.columnSchema(index));
        const ArrowArray& array = source.columnArray(index);
        std::unique_ptr<OwnedColumn> copy;
        if (type == "float64")
        {
            checkCells(*columnEdits, source.rowCount(), 0);
            copy = copyRealColumn(array, source.offset(), source.rowCount(), columnEdits->cells);
        }
        else if (type.starts_with("fixed_list<float64>"))
        {
            const std::int64_t width = base::fixedListWidth(source.columnSchema(index));
            checkCells(*columnEdits, source.rowCount(), width);
            copy = copyFixedListColumn(array, width, source.offset(), source.rowCount(), columnEdits->cells);
        }
        else if (type.starts_with("list<fixed_list<float64>"))
        {
            // The stations of a PBEAM (design/model/FieldTargets): "+L" (O_o_Reader) or "+l" offsets.
            const ArrowSchema& schema = source.columnSchema(index);
            const bool isLarge = std::string_view(schema.format) == "+L";
            const std::int64_t width = base::fixedListWidth(*schema.children[0]);
            if (width <= 0)
                refuseCell(*columnEdits, "expected a list of fixed-size lists of a positive width, found " + type);
            checkNestedCells(*columnEdits, array, isLarge, width, source.offset(), source.rowCount());
            copy = copyNestedListColumn(array, isLarge, width, source.offset(), columnEdits->cells);
        }
        else if (type == "list<float64>")
        {
            // The plies of a PCOMP, the dimensions of a PBARL or of the stations of a PBEAML (design/model/
            // FieldTargets): "+L" (O_o_Reader) or "+l" offsets; the item within the list of its row.
            const bool isLarge = std::string_view(source.columnSchema(index).format) == "+L";
            checkNestedCells(*columnEdits, array, isLarge, 1, source.offset(), source.rowCount());
            copy = copyListColumn(array, isLarge, source.offset(), columnEdits->cells);
        }
        else
        {
            refuseCell(*columnEdits,
                       "expected a float64, fixed_list<float64>[n], list<float64> or list<fixed_list<float64>[n]> "
                       "column, found " +
                           type);
        }
        shared->arrayChildren[static_cast<std::size_t>(index)] = &copy->array;
        shared->columns.push_back(std::move(copy));
    }

    // The top-level structures: a struct of the same rows and offset (a record batch has no struct-level nulls), the
    // contract version of the base carried over in the metadata, the new children arrays.
    const std::string version = source.metadataValue(CONTRACT_VERSION_KEY);
    if (!version.empty())
        shared->metadata = encodeMetadata(CONTRACT_VERSION_KEY, version);
    ArrowSchema schema{};
    schema.format = "+s";
    schema.name = "";
    schema.metadata = shared->metadata.empty() ? nullptr : shared->metadata.data();
    schema.flags = 0;
    schema.n_children = columnCount;
    schema.children = shared->schemaChildren.data();
    schema.dictionary = nullptr;
    schema.release = &releaseSharedSchema;
    schema.private_data = new SharedPointer(shared);
    ArrowArray array{};
    array.length = source.rowCount();
    array.null_count = 0;
    array.offset = source.offset();
    array.n_buffers = 1;
    array.buffers = shared->topBuffers.data();
    array.n_children = columnCount;
    array.children = shared->arrayChildren.data();
    array.dictionary = nullptr;
    array.release = &releaseSharedArray;
    array.private_data = new SharedPointer(shared);
    try
    {
        base::ImportedTable table(schema, array);
        // The borrowed columns are the source's own structures: the contract checks they passed there hold here.
        table.inheritPassedChecks(source, isBorrowed);
        return table;
    }
    catch (const std::exception& error)
    {
        throw std::invalid_argument(
            base::concatenate({"ColumnOverlay: the overlaid table '", name, "' is refused: ", error.what()}));
    }
}

} // namespace

base::ImportedTable appendRealColumns(const std::string& name, const base::ImportedTable& source,
                                      const std::shared_ptr<const void>& keepAlive, std::vector<AddedRealColumn> added)
{
    return shareTable(name, source, keepAlive, {}, std::move(added));
}

void ColumnOverlay::setValue(const std::string& table, const std::string& column, const CellValue& cell)
{
    if (cell.item < CellValue::SCALAR_ITEM)
        throw std::invalid_argument(
            base::concatenate({"ColumnOverlay: column '", column, "' of table '", table,
                               "': expected an item of 0 or more, or SCALAR_ITEM, found ", std::to_string(cell.item)}));
    for (ColumnEdits& edits : m_columns)
    {
        if (edits.table == table && edits.column == column)
        {
            edits.cells.push_back(cell);
            return;
        }
    }
    m_columns.push_back({table, column, {cell}});
}

std::size_t ColumnOverlay::cellCount() const noexcept
{
    std::size_t count = 0;
    for (const ColumnEdits& edits : m_columns)
        count += edits.cells.size();
    return count;
}

ModelSnapshot ColumnOverlay::applyTo(const ModelSnapshot& base, const std::shared_ptr<const void>& keepAlive) const
{
    for (const ColumnEdits& edits : m_columns)
    {
        if (base.findTable(edits.table) == nullptr)
            throw std::invalid_argument(base::concatenate(
                {"ColumnOverlay: expected a table of the snapshot (", std::to_string(base.tableNames().size()),
                 " tables, e.g. 'properties/ShellProp'), found '", edits.table, "', which it does not hold"}));
    }
    const auto share = [&](const std::string& name, const base::ContractReader& reader)
    {
        std::vector<const ColumnEdits*> edits;
        for (const ColumnEdits& columnEdits : m_columns)
        {
            if (columnEdits.table == name)
                edits.push_back(&columnEdits);
        }
        return shareTable(name, reader.table(), keepAlive, edits);
    };

    SnapshotBuilder builder;
    builder.setVersions(base.versions().structureVersion, base.versions().contentVersion);
    if (base.unsupportedOffsetDefinition().empty())
        builder.setOffsetDefinition(base.offsetDefinition());
    else
        builder.setOffsetDefinitionValue(base.unsupportedOffsetDefinition());
    for (const ElementFormulationChoice& choice : base.elementFormulations())
        builder.setElementFormulation(choice.card, choice.formulation);
    for (const TableRole role : ALL_TABLE_ROLES)
    {
        if (!base.hasTable(role))
            continue;
        const std::string name(tableName(role));
        builder.setTable(role, share(name, *base.findTable(name)));
    }
    if (base.nodes().hasCdRotations())
        builder.setCdRotations(BorrowedArray<double>(base.nodes().cdRotations(), keepAlive));
    for (const ElementBatch& batch : base.elementBatches())
        builder.addElementBatch(batch.card(), share(ELEMENTS_PREFIX + batch.card(), batch.reader()));
    for (const CardTable& table : base.propertyTables())
        builder.addPropertyTable(table.card(), share(PROPERTIES_PREFIX + table.card(), table.reader()));
    for (const CardTable& table : base.methodTables())
        builder.addMethodTable(table.card(), table.idColumn(), share(METHODS_PREFIX + table.card(), table.reader()));
    for (const CardTable& table : base.materialCardTables())
        builder.addMaterialCardTable(table.card(), share(MATERIALS_PREFIX + table.card(), table.reader()));
    for (const CardTable& table : base.loadCardTables())
        builder.addLoadCardTable(table.card(), share(LOAD_CARDS_PREFIX + table.card(), table.reader()));
    // Every card table of the aeroelastic model: the AESTAT IDs of the trim variables a SAERO response names among
    // them, not only the splines' and the lists'.
    for (const CardTable& table : base.aeroCardTables())
        builder.addAeroCardTable(table.card(), table.idColumn(),
                                 share(AERO_CARDS_PREFIX + table.card(), table.reader()));
    return builder.build();
}

} // namespace oos::input
