#include "oos/design/aids/ColumnPerturbation.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"

#include <array>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

constexpr std::array<double, FIELD_SIDE_COUNT> SIDE_SCALES{1.0, -1.0, 0.5, -0.5}; // Plus, Minus, HalfPlus, HalfMinus
constexpr const char* MATERIALS_TABLE = "materialsTable";
// The bytes of one entry of a std::map besides its value (the node's links and colour): for the memory estimate.
constexpr std::uint64_t MAP_NODE_OVERHEAD_BYTES = 32;

// One table that holds the column: its name for the overlay, its reader, the IDs of its rows, and the rows to move
// (all of them, or those of the card's type in materialsTable).
struct ColumnSource
{
    std::string table;
    const base::ContractReader* reader = nullptr;
    std::span<const std::int32_t> ids;
    std::vector<std::int64_t> rows;
};

[[noreturn]] void refuse(const std::string& card, const std::string& field, const std::string& what)
{
    throw base::OptimisationError(
        base::concatenate({"the sensitivity map of ", card, " '", field, "': ", what,
                           "; give a real field of a property, material or element card, as the data contract "
                           "names it (\"ShellProp\", \"t\")"}));
}

std::vector<std::int64_t> everyRow(std::int64_t count)
{
    std::vector<std::int64_t> rows(static_cast<std::size_t>(count));
    for (std::int64_t row = 0; row < count; ++row)
        rows[static_cast<std::size_t>(row)] = row;
    return rows;
}

} // namespace

ColumnPerturbation::ColumnPerturbation(const std::shared_ptr<const input::ModelSnapshot>& snapshot, std::string card,
                                       std::string field, const elements::ElementRegistry& registry,
                                       base::MemoryBudget& budget)
    : m_card(std::move(card)), m_field(std::move(field))
{
    std::vector<ColumnSource> sources;
    if (const input::CardTable* table = snapshot->findPropertyTable(m_card))
    {
        m_kind = TargetKind::Property;
        sources.push_back({"properties/" + m_card, &table->reader(), table->ids(), everyRow(table->rowCount())});
    }
    else if (const input::ElementBatch* batch = snapshot->findElementBatch(m_card))
    {
        m_kind = TargetKind::Element;
        sources.push_back({"elements/" + m_card, &batch->reader(), batch->ids(), everyRow(batch->rowCount())});
    }
    else
    {
        m_kind = TargetKind::Material;
        if (const input::CardTable* table = snapshot->findMaterialCardTable(m_card))
            sources.push_back({"materials/" + m_card, &table->reader(), table->ids(), everyRow(table->rowCount())});
        if (snapshot->hasTable(input::TableRole::Materials))
        {
            const input::MaterialTable& materials = snapshot->materials();
            const base::DictionaryView types = materials.types();
            ColumnSource source{MATERIALS_TABLE, &materials.reader(), materials.ids(), {}};
            for (std::int64_t row = 0; row < materials.rowCount(); ++row)
                if (types[row] == m_card)
                    source.rows.push_back(row);
            if (!source.rows.empty())
                sources.push_back(std::move(source));
        }
    }
    if (sources.empty())
        refuse(m_card, m_field, "the model has no such card");
    // Per moved row: its index, a cell of each side's overlay, a row of each side's copy of the column, its widths.
    std::uint64_t rowCount = 0;
    for (const ColumnSource& source : sources)
        rowCount += source.rows.size();
    m_reservation = budget.reserve(
        rowCount * (sizeof(std::int64_t) + FIELD_SIDE_COUNT * (sizeof(input::CellValue) + sizeof(double)) +
                    sizeof(std::int32_t) + sizeof(Widths) + MAP_NODE_OVERHEAD_BYTES),
        "sensitivity map moved column");
    std::array<input::ColumnOverlay, FIELD_SIDE_COUNT> overlay;
    bool hasColumn = false;
    for (const ColumnSource& source : sources)
    {
        if (!source.reader->hasColumn(m_field))
            continue;
        if (source.reader->columnType(m_field) != "float64")
            refuse(m_card, m_field,
                   base::concatenate({"the column is of type ", source.reader->columnType(m_field),
                                      ", not a real scalar (a list column: design one item with a relation)"}));
        hasColumn = true;
        const std::span<const double> values = source.reader->values<double>(m_field, base::NullPolicy::Allow);
        const base::ValidityView valid = source.reader->validity(m_field);
        for (const std::int64_t row : source.rows)
        {
            if (!valid.isValid(row))
                continue;
            const double value = values[static_cast<std::size_t>(row)];
            const double step = fieldStep(value);
            std::array<double, FIELD_SIDE_COUNT> moved{};
            for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
            {
                moved[side] = value + SIDE_SCALES[side] * step;
                overlay[side].setValue(source.table, m_field, {row, input::CellValue::SCALAR_ITEM, moved[side]});
            }
            // The widths as the cells store them (the same for the two tables of a material).
            m_widths[source.ids[static_cast<std::size_t>(row)]] = {moved[0] - moved[1], moved[2] - moved[3]};
        }
    }
    if (!hasColumn)
        refuse(m_card, m_field, "its table has no such column");
    for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
    {
        Side& made = m_sides[side];
        made.snapshot = std::make_shared<const input::ModelSnapshot>(overlay[side].applyTo(*snapshot, snapshot));
        made.index = std::make_unique<elements::SnapshotIndex>(*made.snapshot, budget);
        made.set = std::make_unique<assembly::ElementSet>(*made.index, registry);
    }
}

double ColumnPerturbation::derivative(std::int32_t id, double plus, double minus, double halfPlus,
                                      double halfMinus) const
{
    const Widths& widths = m_widths.at(id);
    // Richardson on the central differences at h and h/2 (FieldPerturbation): (4·D(h/2) − D(h)) / 3.
    const double coarse = (plus - minus) / widths.full;
    const double fine = (halfPlus - halfMinus) / widths.half;
    return (4.0 * fine - coarse) / 3.0;
}

void ColumnPerturbation::derivatives(std::int32_t id, std::span<const double> plus, std::span<const double> minus,
                                     std::span<const double> halfPlus, std::span<const double> halfMinus,
                                     std::span<double> derivative) const
{
    for (std::size_t k = 0; k < derivative.size(); ++k)
        derivative[k] = this->derivative(id, plus[k], minus[k], halfPlus[k], halfMinus[k]);
}

} // namespace oos::design
