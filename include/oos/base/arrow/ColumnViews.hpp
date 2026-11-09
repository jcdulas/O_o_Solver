#pragma once

// The typed views that ContractReader gives on the columns of an Arrow table, one class per header. They all point
// into the producer's buffers (no copy: narrow offsets and indices are read in place, booleans as bits) and are valid
// as long as the ContractReader that made them lives. Row indices are 64-bit; a view does not check its row argument
// (0 <= row < size()), as a span does not. Copyable, immutable, safe to read from several threads.
//
// This header gathers them for the readers that use several.

#include "oos/base/arrow/BooleanView.hpp"
#include "oos/base/arrow/DictionaryView.hpp"
#include "oos/base/arrow/FixedListView.hpp"
#include "oos/base/arrow/ListOffsets.hpp"
#include "oos/base/arrow/ListView.hpp"
#include "oos/base/arrow/NestedListView.hpp"
#include "oos/base/arrow/TextListView.hpp"
#include "oos/base/arrow/Utf8View.hpp"
#include "oos/base/arrow/ValidityView.hpp"
