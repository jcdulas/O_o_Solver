#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/output/sections/SectionLoadTable.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace oos::output
{

// One batch of load cases of section or cut loads, as the sink receives it: the labels of the rows (the table, without
// the values of the batch), the subcases of the batch and their values (rows × cases of the batch). The views are valid
// during the call only.
struct SectionLoadBatch
{
    const SectionLoadTable* table = nullptr;
    std::span<const std::int32_t> cases;
    linalg::ConstDenseMatrixView values;
};

// The values of the section or cut loads of an analysis, batch by batch of load cases: each batch
// updates the envelope of every row, goes to the sink when there is one, and is kept when the values are kept, so
// that without them the memory does not grow with the number of load cases (one batch at a time, the envelope four
// numbers a row).
//
// Thread-safety: none (one accumulator per analysis, fed in case order, so that the case kept on a tie is the first).
// Ownership: owns its table; calls the sink on the thread that offers.
class SectionLoadAccumulator
{
public:
    // `labels`: the table with its labels (no values, no cases). With `keepsValues`, rows × caseCount values are
    // reserved now (MemoryBudgetError before any computation when they do not fit). The sink may be empty.
    SectionLoadAccumulator(SectionLoadTable labels, std::size_t caseCount, bool keepsValues,
                           std::function<void(const SectionLoadBatch&)> sink, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t rowCount() const noexcept { return m_table.rowCount(); }
    [[nodiscard]] const SectionLoadTable& table() const noexcept { return m_table; }

    // Offers the values of `cases` (rows × cases.size()). Throws std::invalid_argument for sizes that disagree or more
    // cases than announced.
    void offer(std::span<const std::int32_t> cases, linalg::ConstDenseMatrixView values);

    // The table with every case offered.
    [[nodiscard]] SectionLoadTable finish() &&;

private:
    SectionLoadTable m_table;
    std::size_t m_caseCount = 0;
    std::function<void(const SectionLoadBatch&)> m_sink;
};

} // namespace oos::output
