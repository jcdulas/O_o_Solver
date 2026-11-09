#pragma once

#include "oos/design/responses/stress_strain_force/ItemCodes.hpp"
#include "oos/output/arrays/ResultArrays.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace oos::design
{

// The result tables of an analysis indexed for the design responses: a nodal value by (kind, subcase, point,
// component), an element value by (kind, element type, subcase, element, place, layer, column).
//
// Thread-safety: none (the row indices are built on first use). Ownership: borrows the arrays, which must outlive it.
class ResultLookup
{
public:
    explicit ResultLookup(const output::ResultArrays& arrays);

    // The component (1 to 6) of point `id` in the nodal table of `kind` ("displacements", "singlePointConstrForces") of
    // `subcase`, or nothing when the table or the point is not there.
    [[nodiscard]] std::optional<double> nodalValue(std::string_view kind, std::int32_t subcase, std::int32_t id,
                                                   std::int32_t component);

    // The value at `location` of element `elementId` in the table of (location.tableKind, `elementType`, `subcase`):
    // the row of the element at the place (centre: grid point 0 or the only row; end A, end B: positions 0 and 1) and
    // of the layer (`layer` 0: the first row of the place), or nothing when the table, the element or the row is not
    // there. The column is found by its name (location.column) in the table's components: a table without it gives
    // nothing.
    [[nodiscard]] std::optional<double> elementValue(const ItemLocation& location, std::string_view elementType,
                                                     std::int32_t subcase, std::int32_t elementId, std::int32_t layer);

    // The row values (all columns) of the same row, empty when not there (for the strength ratio, which reads the ply
    // stresses).
    [[nodiscard]] std::span<const double> elementRow(std::string_view tableKind, std::string_view elementType,
                                                     std::int32_t subcase, std::int32_t elementId, ItemPlace place,
                                                     std::int32_t layer);

private:
    // The table of (`tableKind`, `elementType`, `subcase`), or null.
    [[nodiscard]] const output::ElementResultTable* elementTable(std::string_view tableKind,
                                                                 std::string_view elementType,
                                                                 std::int32_t subcase) const;

    // The first and past-the-last rows of each element of a table (its rows are contiguous).
    using ElementRows = std::unordered_map<std::int32_t, std::pair<std::size_t, std::size_t>>;

    const output::ResultArrays* m_arrays;
    std::map<std::pair<std::string, std::int32_t>, std::size_t> m_nodalTables;
    std::map<std::tuple<std::string, std::string, std::int32_t>, std::size_t> m_elementTables;
    std::unordered_map<std::size_t, std::unordered_map<std::int32_t, std::size_t>> m_nodalRows;
    std::unordered_map<std::size_t, ElementRows> m_elementRows;
};

} // namespace oos::design
