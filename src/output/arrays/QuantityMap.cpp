#include "oos/output/arrays/QuantityMap.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::output
{

QuantityMap::QuantityMap(std::vector<Term> terms, std::size_t quantityCount, base::MemoryBudget& budget)
    : m_quantityCount(quantityCount)
{
    for (const Term& term : terms)
        if (term.quantity < 0 || static_cast<std::size_t>(term.quantity) >= quantityCount || term.row < 0 ||
            term.column < 0)
            throw std::invalid_argument(
                base::concatenate({"QuantityMap: expected terms on quantities 0 to ", std::to_string(quantityCount),
                                   " with rows and columns from 0, found quantity ", std::to_string(term.quantity),
                                   ", row ", std::to_string(term.row), ", column ", std::to_string(term.column),
                                   " (a bug of O_o_Solver: report it)"}));
    // By element, row and column; the quantities of one value in the order given (stable).
    std::stable_sort(terms.begin(), terms.end(),
                     [](const Term& a, const Term& b)
                     {
                         if (a.elementId != b.elementId)
                             return a.elementId < b.elementId;
                         if (a.row != b.row)
                             return a.row < b.row;
                         return a.column < b.column;
                     });
    m_terms = base::BudgetedBuffer<Term>(budget, terms.size(), "quantity map terms");
    std::copy(terms.begin(), terms.end(), m_terms.data());
    std::size_t elementCount = 0;
    for (std::size_t i = 0; i < terms.size(); ++i)
        elementCount += i == 0 || terms[i].elementId != terms[i - 1].elementId ? 1 : 0;
    m_elements = base::BudgetedBuffer<std::int32_t>(budget, elementCount, "quantity map elements");
    std::size_t next = 0;
    for (std::size_t i = 0; i < terms.size(); ++i)
        if (i == 0 || terms[i].elementId != terms[i - 1].elementId)
            m_elements[next++] = terms[i].elementId;
}

void QuantityMap::reduce(std::span<const ElementResultTable> tables, std::span<double> quantities) const
{
    if (quantities.size() != m_quantityCount)
        throw std::invalid_argument(
            base::concatenate({"QuantityMap::reduce: expected ", std::to_string(m_quantityCount), " quantities, found ",
                               std::to_string(quantities.size())}));
    const Term* begin = m_terms.data();
    const Term* end = begin + m_terms.size();
    for (const ElementResultTable& table : tables)
    {
        const std::size_t width = table.componentCount();
        std::size_t row = 0;
        while (row < table.rowCount())
        {
            // The rows of one element are consecutive in its table.
            const std::int32_t elementId = table.ids[row];
            std::size_t last = row;
            while (last < table.rowCount() && table.ids[last] == elementId)
                ++last;
            const Term* term = std::lower_bound(begin, end, elementId,
                                                [](const Term& t, std::int32_t id) { return t.elementId < id; });
            for (; term != end && term->elementId == elementId; ++term)
            {
                const auto at = row + static_cast<std::size_t>(term->row);
                if (at >= last || static_cast<std::size_t>(term->column) >= width)
                    continue;
                quantities[static_cast<std::size_t>(term->quantity)] +=
                    term->coefficient * table.values[at * width + static_cast<std::size_t>(term->column)];
            }
            row = last;
        }
    }
}

namespace
{

// The columns of one block of combineUnitValues: the units' values of a block (8 KB a unit) stay in cache.
constexpr std::size_t COMBINE_COLUMN_BLOCK = 1024;

} // namespace

void combineUnitValues(std::span<const std::int64_t> indptr, std::span<const std::int64_t> indices,
                       std::span<const double> data, std::span<const std::int64_t> rows, std::span<const double> values,
                       std::size_t width, std::span<double> out, base::ThreadPool* pool)
{
    const std::size_t unitCount = width == 0 ? 0 : values.size() / width;
    const std::size_t caseCount = indptr.empty() ? 0 : indptr.size() - 1;
    const auto invalid = [](const std::string& what)
    {
        throw std::invalid_argument(
            base::concatenate({"combineUnitValues: ", what, " (a bug of O_o_Solver: report it)"}));
    };
    if (out.size() != rows.size() * width || (width != 0 && values.size() % width != 0) ||
        indices.size() != data.size() || (!indptr.empty() && static_cast<std::size_t>(indptr.back()) > data.size()))
        invalid("expected arrays of consistent sizes");
    for (const std::int64_t row : rows)
        if (row < 0 || static_cast<std::size_t>(row) >= caseCount || indptr[row] < 0 || indptr[row] > indptr[row + 1])
            invalid("expected rows of the combination matrix, found " + std::to_string(row));
    for (const std::int64_t unit : indices)
        if (unit < 0 || static_cast<std::size_t>(unit) >= unitCount)
            invalid("expected units below " + std::to_string(unitCount) + ", found " + std::to_string(unit));
    std::fill(out.begin(), out.end(), 0.0);
    const std::size_t blockCount = (width + COMBINE_COLUMN_BLOCK - 1) / COMBINE_COLUMN_BLOCK;
    const auto block = [&](std::size_t b)
    {
        const std::size_t first = b * COMBINE_COLUMN_BLOCK;
        const std::size_t count = std::min(COMBINE_COLUMN_BLOCK, width - first);
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            double* target = out.data() + i * width + first;
            for (std::int64_t k = indptr[rows[i]]; k < indptr[rows[i] + 1]; ++k)
            {
                const double coefficient = data[k];
                const double* source = values.data() + static_cast<std::size_t>(indices[k]) * width + first;
                for (std::size_t c = 0; c < count; ++c)
                    target[c] += coefficient * source[c];
            }
        }
    };
    if (pool != nullptr && blockCount > 1)
        pool->runBatch(blockCount, block);
    else
        for (std::size_t b = 0; b < blockCount; ++b)
            block(b);
}

} // namespace oos::output
