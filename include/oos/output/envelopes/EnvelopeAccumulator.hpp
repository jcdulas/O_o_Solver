#pragma once

#include "oos/output/arrays/ElementResultTable.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oos::output
{

// One envelope asked for (for example kind "stress", quantity "vonMises", mode "max"): a table kind (a nodal kind
// "displacements", "singlePointConstrForces", "multiPointConstrForces", "appliedLoads" — the singular "displacement" is
// accepted —, or an element kind "stress", "strain", "forces", "strainEnergy", for every element type that has the
// quantity), a quantity (a component of the tables, or "magnitude", the length of the translation t1, t2, t3 of a nodal
// table) and the extreme kept ("max", "min", "absMax": the largest absolute value, kept with its sign).
struct EnvelopeSpec
{
    std::string kind;
    std::string quantity;
    std::string mode;
};

// The envelope of one table: per row (grid point, or element with its grid point, layer and position), the extreme
// over the load cases and the load case (subcase) that gives it.
struct EnvelopeTable
{
    std::string kind;        // the table kind, plural as the data contract names it ("displacements", "stress")
    std::string elementType; // empty for a nodal table
    std::string quantity;
    std::string mode;
    std::vector<std::int32_t> ids; // bounded by the rows of one table
    std::vector<std::int32_t> nodeIds;
    std::vector<std::int32_t> layers;
    std::vector<double> positions; // empty unless the tables have stations
    std::vector<double> values;
    std::vector<std::int32_t> cases; // the subcase of each extreme
};

// The envelopes computed on the fly over the load cases (one row per item instead of one per load case):
// each table of a case is offered once (offer), the accumulator keeping, per row, the extreme and its case. The
// rows are those of the first table of each kind and element type; a later table with other rows is matched row by row
// on its labels (its rows that the first table does not have are left out, and counted).
//
// A spec whose kind is unknown, whose mode is not max, min or absMax, or whose quantity no table has, is refused by
// the constructor (std::invalid_argument naming it and what is accepted).
//
// Thread-safety: none (one accumulator per analysis, fed in case order, so that the case kept on a tie is the first).
// Ownership: owns its tables.
class EnvelopeAccumulator
{
public:
    explicit EnvelopeAccumulator(std::vector<EnvelopeSpec> specs);

    [[nodiscard]] bool isEmpty() const noexcept { return m_specs.empty(); }
    // Whether some envelope takes tables of this kind (the per-case tables of such a kind may then be left out).
    [[nodiscard]] bool takes(const std::string& kind) const;

    void offer(const NodalResultTable& table);
    void offer(const ElementResultTable& table);
    // The rows of `table` for the envelopes that take it, when none was offered a table of its kind and element type
    // yet: those of the first table, which its values do not touch (the condensed path recovering its
    // units unit by unit offers each unit's rows of every case, after the rows of the whole table are defined, so that
    // the envelopes have the rows, their order and their extremes of the cases offered whole). Nothing otherwise.
    void defineRows(const NodalResultTable& table);
    void defineRows(const ElementResultTable& table);

    // The envelopes, in the order of the specs, then of the element types met. Throws std::invalid_argument naming a
    // spec whose quantity no table offered had (a component name to correct).
    [[nodiscard]] std::vector<EnvelopeTable> tables() const;
    // The rows of later tables not found in the first table of their kind.
    [[nodiscard]] std::int64_t unmatchedRowCount() const noexcept { return m_unmatchedRows; }

private:
    struct Accumulated
    {
        std::size_t spec = 0;
        EnvelopeTable table;
        // The rows by label, made at the first table offered with other rows (the rows never change once made).
        std::vector<std::size_t> sortedRows;
    };

    [[nodiscard]] Accumulated& entryFor(std::size_t spec, const std::string& kind, const std::string& elementType,
                                        bool& isNew);
    // The specs that take a nodal table (its quantity, a component or the magnitude), or an element table (its
    // quantity among the table's columns).
    [[nodiscard]] bool takesNodal(std::size_t spec, const NodalResultTable& table) const;
    [[nodiscard]] bool takesElement(std::size_t spec, const ElementResultTable& table) const;
    static void initialiseRows(Accumulated& entry, const NodalResultTable& table);
    static void initialiseRows(Accumulated& entry, const ElementResultTable& table);
    void update(Accumulated& entry, std::size_t row, double value, std::int32_t subcase) const;

    std::vector<EnvelopeSpec> m_specs;           // as given, kinds made plural
    std::vector<Accumulated> m_entries;          // bounded by the specs × the element types
    std::vector<std::uint8_t> m_isQuantityFound; // per spec
    std::int64_t m_unmatchedRows = 0;
};

} // namespace oos::output
