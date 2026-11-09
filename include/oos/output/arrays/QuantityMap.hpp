#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/output/arrays/ElementResultTable.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::output
{

// A linear map from the element tables of some elements to declared quantities: quantity q = Σ coefficient · value,
// each term one value of an element's table, named by the element, the row among that element's rows of the table
// (its locations in the order the table gives them: the centre first, the corners, the stations of a beam) and the
// column (an index into the table's components). The caller makes the terms from O_o_Reader's reads on probe tables
// (the pockets' mean fluxes in their frames, the members' end forces), so that a quantity is O_o_Reader's read of the
// same tables, to round-off.
//
// The terms are kept sorted by element, row and column; reduce() sums a vector's terms in the order of its tables
// (their element types, then their rows), each term into its quantity: the same bits for the same tables.
//
// Thread-safety: immutable once built; reduce() is const. Ownership: owns its terms (budgeted).
class QuantityMap
{
public:
    struct Term
    {
        std::int32_t elementId = 0;
        std::int32_t row = 0;    // among the element's rows of its table
        std::int32_t column = 0; // into the table's components
        std::int32_t quantity = 0;
        double coefficient = 0.0;
    };

    // Throws std::invalid_argument for a term outside the quantities (or with a negative row or column).
    QuantityMap(std::vector<Term> terms, std::size_t quantityCount, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t quantityCount() const noexcept { return m_quantityCount; }
    [[nodiscard]] std::size_t termCount() const noexcept { return m_terms.size(); }
    // The elements the terms read, increasing, each once.
    [[nodiscard]] std::span<const std::int32_t> elements() const noexcept { return m_elements.span(); }
    // Adds the quantities of one vector's tables into `quantities` (quantityCount() values). A term whose element,
    // row or column the tables lack adds nothing (the element is not a member of them). Throws std::invalid_argument
    // for `quantities` of another size.
    void reduce(std::span<const ElementResultTable> tables, std::span<double> quantities) const;

private:
    base::BudgetedBuffer<Term> m_terms;
    base::BudgetedBuffer<std::int32_t> m_elements;
    std::size_t m_quantityCount = 0;
};

// The values of some cases from the units' values (the units of a load basis): out[i] = Σ data[k] · values[indices[k]]
// over the terms k of the combination matrix's row rows[i] (CSR: indptr, indices, data), each row of `values` and of
// `out` `width` values. Each case sums its units in their order in the row, every product and sum rounded on its own
// (no contraction): the same bits for a case whatever the cases asked with it, and those of the plain loop
// `out[i] += data[k] * values[indices[k]]`. Blocks of columns run on `pool` (null: the calling thread), each over every
// case, so that the units' values of a block stay in cache. Throws std::invalid_argument for a row, a unit or a size
// outside the arrays.
void combineUnitValues(std::span<const std::int64_t> indptr, std::span<const std::int64_t> indices,
                       std::span<const double> data, std::span<const std::int64_t> rows, std::span<const double> values,
                       std::size_t width, std::span<double> out, base::ThreadPool* pool);

} // namespace oos::output
