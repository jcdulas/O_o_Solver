#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace oos::output
{

// The section loads or the cut loads of an analysis: the labels of the rows, the cases in the order they were computed,
// the values of every case when they are kept, and the envelope over the cases.
//
// Thread-safety: immutable once built. Ownership: owns its labels and values (the values budgeted: rows × cases).
struct SectionLoadTable
{
    std::string kind; // "sectionLoads" or "cutLoads"
    // Per section or cut, in the order of the definitions.
    std::vector<std::int32_t> ids;
    std::vector<std::string> names;
    std::vector<std::int32_t> groups;
    std::vector<std::string> families; // "shell", "bar", "beam", "rod"; "cut" for a cut
    std::vector<base::Vec3> points;    // a cut: its point in basic (empty for sections)
    std::vector<base::Mat3> axes;      // a cut: the axes of its CD in basic, as columns (empty for sections)
    // Per row: its section or cut (an index into the lists above) and its component.
    std::vector<std::int32_t> rowItems;
    std::vector<std::string_view> rowComponents; // static names (sectionComponents, CUT_COMPONENTS)
    // The subcases, in the order they were computed (constraint set by constraint set, then the SUBCOMs).
    std::vector<std::int32_t> cases;
    // The values of every case, case after case (value of row r in case c at c·rowCount() + r); empty when they were
    // not kept.
    base::BudgetedBuffer<double> values;
    bool hasValues = false;
    // The envelope over the cases, per row: the largest and the smallest value and the subcase that gives each (the
    // first one on a tie).
    std::vector<double> maxima;
    std::vector<std::int32_t> maximumCases;
    std::vector<double> minima;
    std::vector<std::int32_t> minimumCases;
    std::int64_t batchCount = 0; // the batches of cases the values came in (and the sink was called with)

    [[nodiscard]] std::size_t rowCount() const noexcept { return rowItems.size(); }
};

} // namespace oos::output
