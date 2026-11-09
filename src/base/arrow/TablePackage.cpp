#include "oos/base/arrow/TablePackage.hpp"

#include "oos/base/errors/ContractError.hpp"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace oos::base
{

namespace
{

constexpr char MAGIC[4] = {'O', 'O', 'T', 'P'};
constexpr std::uint32_t PACKAGE_VERSION = 1;
constexpr std::uint64_t NULL_BUFFER = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t BUFFER_ALIGNMENT = 8;
constexpr std::uint64_t PREAMBLE_BYTES = 16; // magic, version, header length

std::uint64_t alignUp(std::uint64_t value)
{
    return (value + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT * BUFFER_ALIGNMENT;
}

[[noreturn]] void throwMalformed(const std::string& what)
{
    throw ContractError("table package: " + what +
                        "; expected the bytes of QueryTable.toPackage() of O_o_Reader, unchanged");
}

// The bytes of a package, shared by every structure built over them: the last structure released frees them.
using SharedBytes = std::shared_ptr<const std::vector<std::uint8_t>>;

// What a rebuilt ArrowSchema owns: its texts, its children and its dictionary.
struct SchemaPrivate
{
    std::string name;
    std::string format;
    std::string metadata; // the Arrow binary form of the key/value metadata; empty for none
    std::vector<ArrowSchema*> children;
    ArrowSchema* dictionary = nullptr;
};

// What a rebuilt ArrowArray owns: its buffer pointers, its children, its dictionary, and a share of the bytes.
struct ArrayPrivate
{
    SharedBytes bytes;
    std::vector<const void*> buffers;
    std::vector<ArrowArray*> children;
    ArrowArray* dictionary = nullptr;
};

void releaseSchema(ArrowSchema* schema)
{
    if (schema == nullptr || schema->release == nullptr)
        return;
    auto* owned = static_cast<SchemaPrivate*>(schema->private_data);
    for (ArrowSchema* child : owned->children)
    {
        releaseSchema(child);
        delete child;
    }
    if (owned->dictionary != nullptr)
    {
        releaseSchema(owned->dictionary);
        delete owned->dictionary;
    }
    delete owned;
    schema->release = nullptr;
}

void releaseArray(ArrowArray* array)
{
    if (array == nullptr || array->release == nullptr)
        return;
    auto* owned = static_cast<ArrayPrivate*>(array->private_data);
    for (ArrowArray* child : owned->children)
    {
        releaseArray(child);
        delete child;
    }
    if (owned->dictionary != nullptr)
    {
        releaseArray(owned->dictionary);
        delete owned->dictionary;
    }
    delete owned;
    array->release = nullptr;
}

// Reads the header, node by node, and fills the structures. Every read is checked against the end of the header.
class PackageReader
{
public:
    PackageReader(SharedBytes bytes, std::uint64_t headerStart, std::uint64_t headerEnd, std::uint64_t bufferStart)
        : m_bytes(std::move(bytes)), m_cursor(headerStart), m_headerEnd(headerEnd), m_bufferStart(bufferStart)
    {
    }

    // Fills `schema` and `array` (which the caller owns) with the next node and its children.
    void readNode(ArrowSchema& schema, ArrowArray& array)
    {
        auto schemaOwned = std::make_unique<SchemaPrivate>();
        auto arrayOwned = std::make_unique<ArrayPrivate>();
        arrayOwned->bytes = m_bytes;
        schemaOwned->name = readText();
        schemaOwned->format = readText();
        const auto flags = readInteger<std::int64_t>();
        schemaOwned->metadata = readMetadata();
        const auto length = readInteger<std::int64_t>();
        const auto nullCount = readInteger<std::int64_t>();
        const auto offset = readInteger<std::int64_t>();
        const auto bufferCount = readInteger<std::uint32_t>();
        for (std::uint32_t index = 0; index < bufferCount; ++index)
        {
            const auto start = readInteger<std::uint64_t>();
            const auto byteLength = readInteger<std::uint64_t>();
            arrayOwned->buffers.push_back(bufferAt(start, byteLength));
        }

        // The structures are given their release callbacks first, so that a failure while reading the children
        // releases what was built so far.
        schema = ArrowSchema{};
        array = ArrowArray{};
        schema.private_data = schemaOwned.release();
        schema.release = &releaseSchema;
        array.private_data = arrayOwned.release();
        array.release = &releaseArray;
        auto& schemaPrivate = *static_cast<SchemaPrivate*>(schema.private_data);
        auto& arrayPrivate = *static_cast<ArrayPrivate*>(array.private_data);

        const auto childCount = readInteger<std::uint32_t>();
        for (std::uint32_t index = 0; index < childCount; ++index)
        {
            schemaPrivate.children.push_back(new ArrowSchema{});
            arrayPrivate.children.push_back(new ArrowArray{});
            readNode(*schemaPrivate.children.back(), *arrayPrivate.children.back());
        }
        if (readInteger<std::uint8_t>() != 0)
        {
            schemaPrivate.dictionary = new ArrowSchema{};
            arrayPrivate.dictionary = new ArrowArray{};
            readNode(*schemaPrivate.dictionary, *arrayPrivate.dictionary);
        }

        schema.format = schemaPrivate.format.c_str();
        schema.name = schemaPrivate.name.c_str();
        schema.metadata = schemaPrivate.metadata.empty() ? nullptr : schemaPrivate.metadata.data();
        schema.flags = flags;
        schema.n_children = static_cast<std::int64_t>(schemaPrivate.children.size());
        schema.children = schemaPrivate.children.empty() ? nullptr : schemaPrivate.children.data();
        schema.dictionary = schemaPrivate.dictionary;
        array.length = length;
        array.null_count = nullCount;
        array.offset = offset;
        array.n_buffers = static_cast<std::int64_t>(arrayPrivate.buffers.size());
        array.buffers = arrayPrivate.buffers.empty() ? nullptr : arrayPrivate.buffers.data();
        array.n_children = static_cast<std::int64_t>(arrayPrivate.children.size());
        array.children = arrayPrivate.children.empty() ? nullptr : arrayPrivate.children.data();
        array.dictionary = arrayPrivate.dictionary;
    }

    [[nodiscard]] bool isAtHeaderEnd() const noexcept { return m_cursor == m_headerEnd; }

private:
    template <typename T>
    T readInteger()
    {
        if (m_headerEnd - m_cursor < sizeof(T))
            throwMalformed("the header ends in the middle of a node");
        T value{};
        // WebAssembly and x86-64 are little-endian: the bytes are the value.
        std::memcpy(&value, m_bytes->data() + m_cursor, sizeof(T));
        m_cursor += sizeof(T);
        return value;
    }

    std::string readText()
    {
        const auto length = readInteger<std::uint32_t>();
        if (m_headerEnd - m_cursor < length)
            throwMalformed("a text runs past the end of the header");
        std::string text(reinterpret_cast<const char*>(m_bytes->data() + m_cursor), length);
        m_cursor += length;
        return text;
    }

    // The key/value pairs, back in the binary form of the Arrow C Data Interface: an int32 count, then each key and
    // value as an int32 length and its bytes.
    std::string readMetadata()
    {
        const auto count = readInteger<std::uint32_t>();
        if (count == 0)
            return {};
        std::string metadata;
        const auto append = [&metadata](std::int32_t value)
        { metadata.append(reinterpret_cast<const char*>(&value), sizeof(value)); };
        append(static_cast<std::int32_t>(count));
        for (std::uint32_t index = 0; index < 2 * count; ++index)
        {
            const std::string text = readText();
            append(static_cast<std::int32_t>(text.size()));
            metadata += text;
        }
        return metadata;
    }

    const void* bufferAt(std::uint64_t start, std::uint64_t byteLength) const
    {
        if (start == NULL_BUFFER)
            return nullptr;
        const std::uint64_t areaBytes = m_bytes->size() - m_bufferStart;
        if (start % BUFFER_ALIGNMENT != 0 || start > areaBytes || byteLength > areaBytes - start)
            throwMalformed("a buffer lies outside the buffer area or is not aligned on 8 bytes");
        return m_bytes->data() + m_bufferStart + start;
    }

    SharedBytes m_bytes;
    std::uint64_t m_cursor;
    std::uint64_t m_headerEnd;
    std::uint64_t m_bufferStart;
};

} // namespace

ImportedTable importTablePackage(std::vector<std::uint8_t> bytes)
{
    if (bytes.size() < PREAMBLE_BYTES || std::memcmp(bytes.data(), MAGIC, sizeof(MAGIC)) != 0)
        throwMalformed("the bytes do not start with \"OOTP\"");
    std::uint32_t version = 0;
    std::memcpy(&version, bytes.data() + 4, sizeof(version));
    if (version != PACKAGE_VERSION)
        throwMalformed("version " + std::to_string(version) + " is not read by this build (it reads version " +
                       std::to_string(PACKAGE_VERSION) + ")");
    std::uint64_t headerBytes = 0;
    std::memcpy(&headerBytes, bytes.data() + 8, sizeof(headerBytes));
    if (headerBytes > bytes.size() - PREAMBLE_BYTES)
        throwMalformed("the header is longer than the package");
    const std::uint64_t headerEnd = PREAMBLE_BYTES + headerBytes;
    const std::uint64_t bufferStart = alignUp(headerEnd);
    if (bufferStart > bytes.size())
        throwMalformed("the package ends before its buffer area");

    // The vector's storage comes from operator new, aligned for any scalar type (at least 8 bytes), and the buffer area
    // starts at a multiple of 8 from it: every buffer is aligned for the values it holds.
    auto shared = std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
    PackageReader reader(shared, PREAMBLE_BYTES, headerEnd, bufferStart);
    ArrowSchema schema{};
    ArrowArray array{};
    try
    {
        reader.readNode(schema, array);
        if (!reader.isAtHeaderEnd())
            throwMalformed("the header holds more than one table");
    }
    catch (...)
    {
        releaseSchema(&schema);
        releaseArray(&array);
        throw;
    }
    // ImportedTable takes the structures over and releases them (even when it refuses the table).
    return ImportedTable(schema, array);
}

} // namespace oos::base
