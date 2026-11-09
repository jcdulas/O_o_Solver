#include "oos/design/sensitivity/ResponseLinearisation.hpp"

#include <algorithm>

namespace oos::design
{

namespace
{

// compressSparse sorts a vector of at least this many entries by counting, when its DOFs span at most
// COUNTING_SORT_SPAN_RATIO times its entries (the counts then take no more room than the vector).
constexpr std::size_t COUNTING_SORT_MIN_ENTRIES = 4096;
constexpr std::size_t COUNTING_SORT_SPAN_RATIO = 2;

// PERFORMANCE: a stable counting sort by DOF: the order std::stable_sort gives (the entries of a DOF in the order
// they were given), in O(entries + span) instead of O(entries·log(entries)); a field's pseudo-load at 1 M DOFs holds
// 4 M entries (24 per ShellQuad4), whose sort took some 0.35 s of the 0.83 s of the field loop on six threads.
// Measured: the threaded evaluation of a design point.
bool sortByCounting(SparseVector& vector)
{
    if (vector.size() < COUNTING_SORT_MIN_ENTRIES)
        return false;
    const auto [lowest, highest] = std::minmax_element(
        vector.begin(), vector.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
    const auto span = static_cast<std::size_t>(highest->first - lowest->first) + 1;
    if (span > COUNTING_SORT_SPAN_RATIO * vector.size())
        return false;
    const base::DofIndex first = lowest->first;
    std::vector<std::size_t> start(span + 1, 0); // bounded by the vector (the span test above)
    for (const auto& entry : vector)
        ++start[static_cast<std::size_t>(entry.first - first) + 1];
    for (std::size_t k = 0; k < span; ++k)
        start[k + 1] += start[k];
    SparseVector sorted(vector.size());
    for (const auto& entry : vector)
        sorted[start[static_cast<std::size_t>(entry.first - first)]++] = entry;
    vector.swap(sorted);
    return true;
}

} // namespace

void compressSparse(SparseVector& vector)
{
    if (!sortByCounting(vector))
        std::stable_sort(vector.begin(), vector.end(),
                         [](const auto& left, const auto& right) { return left.first < right.first; });
    std::size_t kept = 0;
    for (std::size_t i = 0; i < vector.size();)
    {
        const base::DofIndex dof = vector[i].first;
        double sum = 0.0;
        for (; i < vector.size() && vector[i].first == dof; ++i)
            sum += vector[i].second;
        if (sum != 0.0)
            vector[kept++] = {dof, sum};
    }
    vector.resize(kept);
}

SparseVector addScaledSparse(const SparseVector& y, double a, const SparseVector& x)
{
    SparseVector sum;
    sum.reserve(x.size() + y.size());
    std::size_t i = 0;
    std::size_t k = 0;
    while (i < y.size() || k < x.size())
    {
        if (k == x.size() || (i < y.size() && y[i].first < x[k].first))
            sum.push_back(y[i++]);
        else if (i == y.size() || x[k].first < y[i].first)
        {
            sum.emplace_back(x[k].first, a * x[k].second);
            ++k;
        }
        else
        {
            sum.emplace_back(y[i].first, y[i].second + a * x[k].second);
            ++i;
            ++k;
        }
    }
    return sum;
}

double dotSparse(const SparseVector& x, std::span<const double> v) noexcept
{
    double sum = 0.0;
    for (const auto& [dof, value] : x)
        sum += value * v[static_cast<std::size_t>(dof)];
    return sum;
}

void compressRow(VariableRow& row)
{
    std::stable_sort(row.begin(), row.end(),
                     [](const auto& left, const auto& right) { return left.first < right.first; });
    std::size_t kept = 0;
    for (std::size_t i = 0; i < row.size();)
    {
        const std::size_t column = row[i].first;
        // From 0.0, as the dense accumulation it replaces: 0.0 + a + b gives the bits of the dense row.
        double sum = 0.0;
        for (; i < row.size() && row[i].first == column; ++i)
            sum += row[i].second;
        if (sum != 0.0)
            row[kept++] = {column, sum};
    }
    row.resize(kept);
}

VariableRow addScaledRow(const VariableRow& y, double a, const VariableRow& x)
{
    VariableRow sum;
    sum.reserve(x.size() + y.size());
    std::size_t i = 0;
    std::size_t k = 0;
    while (i < y.size() || k < x.size())
    {
        if (k == x.size() || (i < y.size() && y[i].first < x[k].first))
            sum.push_back(y[i++]);
        else if (i == y.size() || x[k].first < y[i].first)
        {
            sum.emplace_back(x[k].first, a * x[k].second);
            ++k;
        }
        else
        {
            sum.emplace_back(y[i].first, y[i].second + a * x[k].second);
            ++i;
            ++k;
        }
    }
    return sum;
}

void addRowTo(const VariableRow& row, std::span<double> dense) noexcept
{
    for (const auto& [column, value] : row)
        dense[column] += value;
}

void rowSlice(const VariableRow& row, std::size_t first, std::span<double> block) noexcept
{
    std::fill(block.begin(), block.end(), 0.0);
    const auto begin = std::lower_bound(row.begin(), row.end(), first,
                                        [](const auto& entry, std::size_t column) { return entry.first < column; });
    for (auto entry = begin; entry != row.end() && entry->first < first + block.size(); ++entry)
        block[entry->first - first] = entry->second;
}

} // namespace oos::design
