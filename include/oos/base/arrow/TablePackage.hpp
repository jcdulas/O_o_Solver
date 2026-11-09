#pragma once

#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <vector>

namespace oos::base
{

// Reads a "table package": one block of bytes that holds a table of O_o_Reader with its exact Arrow layout (its
// `QueryTable.toPackage()` in JavaScript). It is how a table crosses from the O_o_Reader WebAssembly module to this
// one, which cannot read the other module's memory. The Arrow C Data Interface structures are rebuilt over the bytes,
// which the table keeps: the buffers are not copied again.
//
// Layout (all integers little-endian; docs/table-package.md):
//
//   "OOTP"  u32 version (1)  u64 headerBytes
//   header: the root node (the table: a struct array with one child per column), written depth first
//   padding to a multiple of 8 bytes
//   the buffers, each starting at a multiple of 8 bytes from the start of this area
//
//   node: string name, string format, i64 flags, u32 metadataCount, (string key, string value) * metadataCount,
//         i64 length, i64 nullCount, i64 offset,
//         u32 bufferCount, (u64 start, u64 byteLength) * bufferCount   (start = 2^64 - 1 for a null buffer),
//         u32 childCount, node * childCount,
//         u8 hasDictionary, node (when hasDictionary is 1)
//   string: u32 byteLength, UTF-8 bytes
//
// Throws ContractError when the bytes are not a package of a version this build reads, or when a node or a buffer runs
// past the end of the bytes; the contract of the table itself (its columns and types) is checked later, by its reader.
[[nodiscard]] ImportedTable importTablePackage(std::vector<std::uint8_t> bytes);

} // namespace oos::base
