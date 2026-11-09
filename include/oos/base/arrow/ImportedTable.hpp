#pragma once

#include "oos/base/arrow/ArrowStructures.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace oos::base
{

// A table handed over by its producer (O_o_Reader) through the Arrow C Data Interface: a struct array (one record
// batch) and its schema, each child being a column. The table owns the two structures and releases them when it dies,
// which lets the producer free its buffers; the buffers themselves are never copied.
//
// Invariants, checked at construction (ContractError otherwise, so that a malformed table never crashes a reader): the
// schema is a struct ("+s") with as many children as the array; every column, recursively, has a format, an array with
// the buffers and children its format needs, a dictionary on both sides or on neither, and at least the rows of the
// table. What the Arrow structures cannot tell (the byte length of a buffer) is trusted.
//
// Thread-safety: immutable once built, so concurrent reads are safe. Ownership: owns the structures; movable, not
// copyable.
class ImportedTable
{
public:
    ImportedTable() = default;
    // Takes the structures over (the caller has already marked its own copies as released, as the Arrow C Data
    // Interface says for a move). The structures are released even when the constructor throws.
    ImportedTable(ArrowSchema schema, ArrowArray array);
    ImportedTable(const ImportedTable&) = delete;
    ImportedTable& operator=(const ImportedTable&) = delete;
    ImportedTable(ImportedTable&& other) noexcept;
    ImportedTable& operator=(ImportedTable&& other) noexcept;
    ~ImportedTable();

    // Moves the structures out of the producer's (an ArrowSchema and an ArrowArray PyCapsule of the Arrow PyCapsule
    // interface): copies them and marks the sources released, so that the capsules no longer free them.
    [[nodiscard]] static ImportedTable takeOver(ArrowSchema* schema, ArrowArray* array);

    // Reads the one record batch of a stream (O_o_Reader's __arrow_c_stream__ gives a table as one batch). Throws
    // ContractError when the stream fails, is empty, or has a second batch. The stream stays the caller's: the
    // structures taken from it live on after its release, as the Arrow stream interface says.
    [[nodiscard]] static ImportedTable fromStream(ArrowArrayStream* stream);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_owned ? m_owned->array.length : 0; }
    [[nodiscard]] std::int64_t columnCount() const noexcept { return m_owned ? m_owned->schema.n_children : 0; }
    [[nodiscard]] std::vector<std::string> columnNames() const;

    // The index of a column, or -1 when there is none of that name.
    [[nodiscard]] std::int64_t findColumn(const std::string& name) const;

    // Precondition: 0 <= index < columnCount().
    [[nodiscard]] const ArrowSchema& columnSchema(std::int64_t index) const { return *m_owned->schema.children[index]; }
    [[nodiscard]] const ArrowArray& columnArray(std::int64_t index) const { return *m_owned->array.children[index]; }

    // The row offset of the struct array itself, added to every column's own offset.
    [[nodiscard]] std::int64_t offset() const noexcept { return m_owned ? m_owned->array.offset : 0; }

    // A value of the schema's key/value metadata (O_o_Reader puts "oor_reader.data_contract_version" there), "" when
    // absent.
    [[nodiscard]] std::string metadataValue(const std::string& key) const;

    // The contract checks of a column whose cost grows with the rows (ContractReader: no nulls in the column, its list
    // items or its dictionary; dictionary indices in range; non-decreasing list and text offsets). The buffers of an
    // imported table never change, so a check a column has passed always passes: it is recorded and not run again.
    enum class ColumnCheck : std::uint8_t
    {
        Nulls,
        ChildNulls,
        GrandchildNulls,
        DictionaryNulls,
        DictionaryIndices,
        DictionaryOffsets,
        ListOffsets,
        TextOffsets
    };
    // PERFORMANCE: the element families read their columns at every facet call (ElementBatchView, PropertyRows), and
    // a design sensitivity calls a facet for one element at a time; the checks cost O(rows of the batch) per call
    // (the dictionary of pid_type checked index by index, null bitmaps counted without POPCNT): 136 µs of the 155 µs
    // of a one-element CQUAD4 stiffness on a batch of 40 000 CQUAD4, so that the derivatives of a field cost O(model)
    // per element. Recorded, a check runs once per table (and once per base table for its overlays:
    // inheritPassedChecks). Measured: the setup of a CQUAD4 facet call 136 µs → 6.7 µs; the full Jacobian of 200 × 200
    // on 250 000 DOFs 32.6 s → 5.0 s (the same bits), of 2 000 × 2 000 on 1 M DOFs 423 s → 108 s.
    // Precondition (both): 0 <= index < columnCount(). Thread-safe: a check two threads run at once is recorded twice.
    [[nodiscard]] bool hasPassed(std::int64_t index, ColumnCheck check) const noexcept;
    void markPassed(std::int64_t index, ColumnCheck check) const noexcept;
    // Records the checks `source` has passed on the columns this table borrows from it unchanged (`isBorrowed`, one
    // flag per column: the same column structures, a ColumnOverlay of `source`), when both tables have the same rows
    // and offset; does nothing otherwise.
    void inheritPassedChecks(const ImportedTable& source, std::span<const std::uint8_t> isBorrowed) const;

private:
    // Heap-held so that the addresses of the structures stay valid when the table moves (the producer's release
    // callbacks may keep pointers to them).
    struct Owned
    {
        ArrowSchema schema{};
        ArrowArray array{};
        // One bit per ColumnCheck passed, per column (a cache of the immutable buffers' checks).
        std::unique_ptr<std::atomic<std::uint16_t>[]> passedChecks;
        Owned() = default;
        Owned(const Owned&) = delete;
        Owned& operator=(const Owned&) = delete;
        ~Owned();
    };
    std::unique_ptr<Owned> m_owned;
};

// The type of a column as the data contract of O_o_Reader names it ("int32", "float64", "utf8", "bool",
// "dictionary<utf8>", "list<int32>", "fixed_list<float64>[3]", "null"; "unknown(<format>)" for a format the solver
// does not read), from its Arrow schema alone. Both list layouts ("+l" and "+L") are "list<...>", as O_o_Reader names
// its "+L" lists.
[[nodiscard]] std::string contractTypeOf(const ArrowSchema& column);

// The width of a fixed-size list column ("+w:3" gives 3); 0 when the column is not one or its width is not a positive
// number.
[[nodiscard]] std::int64_t fixedListWidth(const ArrowSchema& column);

} // namespace oos::base
