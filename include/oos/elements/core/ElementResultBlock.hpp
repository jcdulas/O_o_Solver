#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::elements
{

// Where a row of results lies in its element, as the result tables of O_o_Reader name the rows (node_ids, layers,
// positions): the grid point ID of a corner (0 for the centre or for an element with a single place), the layer (1 for
// an element with a single row per place; for a shell 1 the fibre Z1 or the membrane strains, 2 the fibre Z2 or the
// curvatures; for a beam the number of the station, from 1 at end A) and the position of a station of a bar or beam
// along its length, as a fraction from end A (0) to end B (1).
struct ResultPlace
{
    std::int32_t nodeId = 0;
    std::int32_t layer = 1;
    double position = 0.0;
};

// The recovered quantities of a range of elements of one batch for one output kind (forces, stresses): one or more rows
// of values per element (a rod has one, a ShellQuad4 one per corner and fibre), each with its place, in the columns the
// family names for that kind. The column names are those of the O_o_Reader data
// contract ("springForce", "axial", "sma", ...), so that the output
// module copies a block into a result table without renaming.
//
// Bounded by the range length the caller chooses (std::vector).
//
// Thread-safety: none for writes; one block per thread. Ownership: owns its values.
class ElementResultBlock
{
public:
    // Forgets the rows held and sets the columns of the rows to come.
    void reset(std::vector<std::string> columns);

    // Appends the row of the element of batch row `elementRow`, ID `elementId`, at the default place (centre, layer 1).
    // Throws std::invalid_argument when the number of values is not the number of columns.
    void append(std::int64_t elementRow, std::int32_t elementId, std::span<const double> values)
    {
        append(elementRow, elementId, ResultPlace{}, values);
    }
    // Appends one row of the element at `place`; same precondition.
    void append(std::int64_t elementRow, std::int32_t elementId, ResultPlace place, std::span<const double> values);

    [[nodiscard]] const std::vector<std::string>& columns() const noexcept { return m_columns; }
    [[nodiscard]] std::size_t size() const noexcept { return m_elementRows.size(); }
    [[nodiscard]] std::int64_t elementRow(std::size_t i) const { return m_elementRows[i]; }
    [[nodiscard]] std::int32_t elementId(std::size_t i) const { return m_elementIds[i]; }
    [[nodiscard]] ResultPlace place(std::size_t i) const { return m_places[i]; }
    [[nodiscard]] std::span<const double> values(std::size_t i) const;
    // The values of row i, to be set in place (a family completing the rows of the vectorised recovery:
    // ElementFamily::completeRecoveredRows).
    [[nodiscard]] std::span<double> mutableValues(std::size_t i);
    // Room for `rows` rows without reallocation.
    void reserve(std::size_t rows);
    // The value of a named column for element i; throws std::invalid_argument for an unknown column.
    [[nodiscard]] double value(std::size_t i, const std::string& column) const;

private:
    std::vector<std::string> m_columns;
    std::vector<std::int64_t> m_elementRows;
    std::vector<std::int32_t> m_elementIds;
    std::vector<ResultPlace> m_places;
    std::vector<double> m_values;
};

} // namespace oos::elements
