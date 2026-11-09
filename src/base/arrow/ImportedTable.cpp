#include "oos/base/arrow/ImportedTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <charconv>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace oos::base
{

namespace
{

// The buffers an array of each format has (Arrow columnar format, "Buffer listing for each layout"); -1 for a format
// the solver does not read, whose buffers are not checked (a reader refuses its type anyway).
std::int64_t bufferCountOf(std::string_view format)
{
    if (format == "n")
        return 0;
    if (format == "u" || format == "U")
        return 3;
    if (format == "+s" || format.starts_with("+w:"))
        return 1;
    if (format == "+l" || format == "+L")
        return 2;
    constexpr std::string_view FIXED_WIDTH_FORMATS = "bcCsSiIlLefg";
    if (format.size() == 1 && FIXED_WIDTH_FORMATS.find(format[0]) != std::string_view::npos)
        return 2;
    return -1;
}

std::int64_t childCountOf(std::string_view format)
{
    if (format == "+l" || format == "+L" || format.starts_with("+w:"))
        return 1;
    return 0;
}

// What to do about a table that breaks the Arrow C Data Interface.
constexpr std::string_view NOT_ARROW_COMPLIANT =
    "; the producer did not follow the Arrow C Data Interface: report it to the producer (O_o_Reader)";

[[noreturn]] void refuseLayout(const std::string& where, std::string_view problem)
{
    throw ContractError(concatenate({"Arrow table, column '", where, "': ", problem, NOT_ARROW_COMPLIANT}));
}

// Checks one column, recursively: what a reader dereferences must be there. `minimumLength` is the number of values
// the parent needs from it.
void validateColumn(const ArrowSchema& schema, const ArrowArray& array, const std::string& where,
                    std::int64_t minimumLength)
{
    if (schema.format == nullptr)
        refuseLayout(where, "the schema has no format");
    const std::string_view format = schema.format;
    if (array.length < 0 || array.offset < 0)
        refuseLayout(where, "negative length or offset");
    if (array.length < minimumLength)
        refuseLayout(where, concatenate({"expected at least ", std::to_string(minimumLength), " values, found ",
                                         std::to_string(array.length)}));
    if (schema.n_children != array.n_children)
        refuseLayout(where, concatenate({"the schema has ", std::to_string(schema.n_children),
                                         " children and the array ", std::to_string(array.n_children)}));
    const std::int64_t buffers = bufferCountOf(format);
    if (buffers >= 0 && (array.n_buffers < buffers || (buffers > 0 && array.buffers == nullptr)))
        refuseLayout(where, concatenate({"format '", format, "' needs ", std::to_string(buffers), " buffers, found ",
                                         std::to_string(array.n_buffers)}));
    // A buffer may be null only for an empty array, or as the validity bitmap (buffer 0) of an array without nulls.
    for (std::int64_t index = 1; index < buffers; ++index)
    {
        if (array.buffers[index] == nullptr && array.length > 0)
            refuseLayout(where, concatenate({"buffer ", std::to_string(index), " is null"}));
    }
    if (buffers > 0 && array.buffers[0] == nullptr && array.null_count > 0)
        refuseLayout(where, "nulls are counted but there is no validity bitmap");
    if (array.n_children < childCountOf(format))
        refuseLayout(where, concatenate({"format '", format, "' needs a child array"}));
    if (array.n_children > 0 && (schema.children == nullptr || array.children == nullptr))
        refuseLayout(where, "the children are missing");
    for (std::int64_t child = 0; child < array.n_children; ++child)
    {
        if (schema.children[child] == nullptr || array.children[child] == nullptr)
            refuseLayout(where, "a child is missing");
        // The values a fixed-size list needs from its child are known here; a variable list checks its offsets when
        // it is read.
        std::int64_t childLength = 0;
        if (format.starts_with("+w:"))
        {
            const std::int64_t width = fixedListWidth(schema);
            if (width <= 0)
                refuseLayout(where, concatenate({"format '", format, "' has no positive width"}));
            // A hostile or broken producer could give lengths whose product overflows 64 bits.
            const std::int64_t rows = array.offset + array.length; // both checked non-negative above
            if (rows < 0 || rows > std::numeric_limits<std::int64_t>::max() / width)
                refuseLayout(where, concatenate({"offset + length (", std::to_string(array.offset), " + ",
                                                 std::to_string(array.length), ") times the width ",
                                                 std::to_string(width), " overflows 64 bits"}));
            childLength = rows * width;
        }
        validateColumn(*schema.children[child], *array.children[child], where + ".item", childLength);
    }
    if ((schema.dictionary == nullptr) != (array.dictionary == nullptr))
        refuseLayout(where, "a dictionary is on one side only (schema or array)");
    if (schema.dictionary != nullptr)
        validateColumn(*schema.dictionary, *array.dictionary, where + ".dictionary", 0);
}

std::string nameOf(const ArrowSchema& schema)
{
    return schema.name != nullptr ? std::string(schema.name) : std::string();
}

} // namespace

ImportedTable::Owned::~Owned()
{
    if (array.release != nullptr)
        array.release(&array);
    if (schema.release != nullptr)
        schema.release(&schema);
}

ImportedTable::ImportedTable(ArrowSchema schema, ArrowArray array) : m_owned(std::make_unique<Owned>())
{
    m_owned->schema = schema;
    m_owned->array = array;
    if (schema.release == nullptr || array.release == nullptr)
        throw ContractError(
            "an Arrow table was handed over already released; expected live structures (a capsule "
            "may be consumed only once): ask O_o_Reader for the table again");
    if (schema.format == nullptr || std::strcmp(schema.format, "+s") != 0)
        throw ContractError(concatenate({"an Arrow table must be a struct array (format '+s'), found format '",
                                         schema.format != nullptr ? schema.format : "(none)",
                                         "'; hand over a table (one record batch of columns), as the query tables of "
                                         "O_o_Reader give it"}));
    if (schema.n_children != array.n_children || (schema.n_children > 0 && (!schema.children || !array.children)))
        throw ContractError(
            concatenate({"an Arrow table has ", std::to_string(schema.n_children), " columns in its schema and ",
                         std::to_string(array.n_children), " in its array; expected the same columns on both sides",
                         NOT_ARROW_COMPLIANT}));
    if (array.length < 0 || array.offset < 0)
        throw ContractError(concatenate({"an Arrow table has a negative length (", std::to_string(array.length),
                                         ") or offset (", std::to_string(array.offset), ")", NOT_ARROW_COMPLIANT}));
    for (std::int64_t index = 0; index < schema.n_children; ++index)
    {
        if (schema.children[index] == nullptr || array.children[index] == nullptr)
            throw ContractError(concatenate({"an Arrow table has no column ", std::to_string(index),
                                             " (a null child pointer) although it counts ",
                                             std::to_string(schema.n_children), NOT_ARROW_COMPLIANT}));
        const std::string where = schema.children[index]->name != nullptr ? std::string(schema.children[index]->name)
                                                                          : concatenate({"#", std::to_string(index)});
        validateColumn(*schema.children[index], *array.children[index], where, array.offset + array.length);
    }
    const auto columns = static_cast<std::size_t>(schema.n_children);
    m_owned->passedChecks = std::make_unique<std::atomic<std::uint16_t>[]>(columns);
    for (std::size_t index = 0; index < columns; ++index)
        m_owned->passedChecks[index].store(0, std::memory_order_relaxed);
}

bool ImportedTable::hasPassed(std::int64_t index, ColumnCheck check) const noexcept
{
    const auto bit = static_cast<std::uint16_t>(1U << static_cast<unsigned>(check));
    return (m_owned->passedChecks[static_cast<std::size_t>(index)].load(std::memory_order_relaxed) & bit) != 0;
}

void ImportedTable::markPassed(std::int64_t index, ColumnCheck check) const noexcept
{
    const auto bit = static_cast<std::uint16_t>(1U << static_cast<unsigned>(check));
    m_owned->passedChecks[static_cast<std::size_t>(index)].fetch_or(bit, std::memory_order_relaxed);
}

void ImportedTable::inheritPassedChecks(const ImportedTable& source, std::span<const std::uint8_t> isBorrowed) const
{
    if (!m_owned || !source.m_owned || source.offset() != offset() || source.rowCount() != rowCount() ||
        source.columnCount() != columnCount() || static_cast<std::int64_t>(isBorrowed.size()) != columnCount())
        return;
    for (std::int64_t index = 0; index < columnCount(); ++index)
        if (isBorrowed[static_cast<std::size_t>(index)] != 0 && &source.columnArray(index) == &columnArray(index))
            m_owned->passedChecks[static_cast<std::size_t>(index)].fetch_or(
                source.m_owned->passedChecks[static_cast<std::size_t>(index)].load(std::memory_order_relaxed),
                std::memory_order_relaxed);
}

ImportedTable::ImportedTable(ImportedTable&& other) noexcept = default;
ImportedTable& ImportedTable::operator=(ImportedTable&& other) noexcept = default;
ImportedTable::~ImportedTable() = default;

ImportedTable ImportedTable::takeOver(ArrowSchema* schema, ArrowArray* array)
{
    if (schema == nullptr || array == nullptr)
        throw ContractError(
            "an Arrow table was handed over without its schema or its array; hand over both, as "
            "the query tables of O_o_Reader give them");
    const ArrowSchema schemaCopy = *schema;
    const ArrowArray arrayCopy = *array;
    // The Arrow move: the source keeps its fields but is marked released, so that its owner does not free it again.
    schema->release = nullptr;
    array->release = nullptr;
    return {schemaCopy, arrayCopy};
}

ImportedTable ImportedTable::fromStream(ArrowArrayStream* stream)
{
    if (stream == nullptr || stream->release == nullptr)
        throw ContractError(
            "an Arrow stream was handed over already released; expected a live stream (a capsule "
            "may be consumed only once): ask O_o_Reader for the table again");
    auto streamError = [stream](int code, std::string_view what)
    {
        const char* detail = stream->get_last_error != nullptr ? stream->get_last_error(stream) : nullptr;
        return ContractError(
            concatenate({"the Arrow stream failed to give its ", what, " (error ", std::to_string(code),
                         detail != nullptr ? ": " : "", detail != nullptr ? detail : "", ")"}));
    };
    ArrowSchema schema{};
    if (const int code = stream->get_schema(stream, &schema); code != 0)
        throw streamError(code, "schema");
    ArrowArray array{};
    if (const int code = stream->get_next(stream, &array); code != 0)
    {
        schema.release(&schema);
        throw streamError(code, "first batch");
    }
    if (array.release == nullptr)
    {
        schema.release(&schema);
        throw ContractError(
            "the Arrow stream ended before its first batch; expected a table as one record batch: "
            "hand over the table as O_o_Reader gives it");
    }
    ImportedTable table(schema, array);
    ArrowArray next{};
    if (const int code = stream->get_next(stream, &next); code != 0)
        throw streamError(code, "end");
    if (next.release != nullptr)
    {
        next.release(&next);
        throw ContractError(
            "the Arrow stream has more than one batch; expected a table as one record batch: hand over the "
            "table as O_o_Reader gives it");
    }
    return table;
}

std::vector<std::string> ImportedTable::columnNames() const
{
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(columnCount()));
    for (std::int64_t i = 0; i < columnCount(); ++i)
        names.push_back(nameOf(columnSchema(i)));
    return names;
}

std::int64_t ImportedTable::findColumn(const std::string& name) const
{
    for (std::int64_t i = 0; i < columnCount(); ++i)
    {
        if (columnSchema(i).name != nullptr && name == columnSchema(i).name)
            return i;
    }
    return -1;
}

std::string ImportedTable::metadataValue(const std::string& key) const
{
    // Arrow metadata: an int32 pair count, then for each pair an int32 length and the key bytes, an int32 length and
    // the value bytes (C Data Interface, "metadata"). The total length is not given: the format is trusted, except
    // that a negative count or length stops the search.
    if (!m_owned || m_owned->schema.metadata == nullptr)
        return "";
    const char* cursor = m_owned->schema.metadata;
    auto readInt32 = [&cursor]
    {
        std::int32_t value = 0;
        std::memcpy(&value, cursor, sizeof(value));
        cursor += sizeof(value);
        return value;
    };
    const std::int32_t pairs = readInt32();
    for (std::int32_t i = 0; i < pairs; ++i)
    {
        const std::int32_t keyLength = readInt32();
        if (keyLength < 0)
            return "";
        const std::string_view currentKey(cursor, static_cast<std::size_t>(keyLength));
        cursor += keyLength;
        const std::int32_t valueLength = readInt32();
        if (valueLength < 0)
            return "";
        const std::string_view value(cursor, static_cast<std::size_t>(valueLength));
        cursor += valueLength;
        if (currentKey == key)
            return std::string(value);
    }
    return "";
}

std::int64_t fixedListWidth(const ArrowSchema& column)
{
    const std::string_view format = column.format != nullptr ? column.format : "";
    if (!format.starts_with("+w:"))
        return 0;
    const std::string_view digits = format.substr(3);
    std::int64_t width = 0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), width);
    if (error != std::errc() || end != digits.data() + digits.size() || width <= 0)
        return 0;
    return width;
}

std::string contractTypeOf(const ArrowSchema& column)
{
    if (column.dictionary != nullptr)
        return concatenate({"dictionary<", contractTypeOf(*column.dictionary), ">"});
    const std::string_view format = column.format != nullptr ? column.format : "";
    struct Name
    {
        std::string_view format;
        std::string_view type;
    };
    static constexpr Name PRIMITIVE_NAMES[] = {{"n", "null"},    {"b", "bool"},   {"c", "int8"},      {"C", "uint8"},
                                               {"s", "int16"},   {"S", "uint16"}, {"i", "int32"},     {"I", "uint32"},
                                               {"l", "int64"},   {"L", "uint64"}, {"e", "float16"},   {"f", "float32"},
                                               {"g", "float64"}, {"u", "utf8"},   {"U", "large_utf8"}};
    for (const Name& name : PRIMITIVE_NAMES)
    {
        if (format == name.format)
            return std::string(name.type);
    }
    const bool hasOneChild = column.n_children == 1 && column.children != nullptr && column.children[0] != nullptr;
    if ((format == "+l" || format == "+L") && hasOneChild)
        return concatenate({"list<", contractTypeOf(*column.children[0]), ">"});
    if (format.starts_with("+w:") && hasOneChild)
        return concatenate({"fixed_list<", contractTypeOf(*column.children[0]), ">[", format.substr(3), "]"});
    return concatenate({"unknown(", format, ")"});
}

} // namespace oos::base
