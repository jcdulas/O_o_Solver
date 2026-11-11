#include "oos/design/model/DerivedMaterials.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

namespace oos::design
{
void DerivedMaterials::bind(const DesignModel& model, const input::ModelSnapshot& snapshot)
{
    m_groups.clear();
    m_relationGroups.assign(model.relations().size(), {});
    const auto* card = snapshot.findMaterialCardTable("IsoMat");
    if (card == nullptr || !card->reader().hasColumn("sourceBlankFields"))
        return;
    const auto& reader = card->reader();
    const auto blanks = reader.values<std::uint8_t>("sourceBlankFields");
    const auto young = reader.values<double>("youngsModulus");
    const auto shear = reader.values<double>("shearModulus");
    const auto poisson = reader.values<double>("poissonRatio");
    constexpr const char* NAMES[]{"youngsModulus", "shearModulus", "poissonRatio"};
    std::map<std::int32_t, std::array<std::optional<std::size_t>, 3>> related;
    for (std::size_t r = 0; r < model.relations().size(); ++r)
    {
        const auto& relation = model.relations()[r];
        if (relation.targetKind != TargetKind::Material || relation.targetCard != "IsoMat")
            continue;
        for (std::size_t field = 0; field < 3; ++field)
            if (relation.field == NAMES[field])
                related[relation.targetId][field] = r;
    }
    std::map<std::int32_t, std::int64_t> summaryRows;
    if (snapshot.hasTable(input::TableRole::Materials))
    {
        const auto& table = snapshot.materials();
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
            if (table.types()[row] == "IsoMat")
                summaryRows.emplace(table.ids()[row], row);
    }
    for (std::int64_t row = 0; row < card->rowCount(); ++row)
    {
        const auto found = related.find(card->ids()[row]);
        if (found == related.end() || (blanks[row] != 1 && blanks[row] != 2 && blanks[row] != 4))
            continue;
        const auto dependent = blanks[row] == 1 ? 0U : blanks[row] == 2 ? 1U : 2U;
        if (found->second[dependent])
            continue; // An explicit relation replaces the original default.
        Group group;
        group.cardRow = row;
        group.blank = blanks[row];
        group.base = {young[row], shear[row], poisson[row]};
        group.relations = found->second;
        if (const auto summary = summaryRows.find(card->ids()[row]); summary != summaryRows.end())
            group.summaryRow = summary->second;
        const auto index = m_groups.size();
        m_groups.push_back(group);
        for (const auto relation : group.relations)
            if (relation)
                m_relationGroups[*relation] = index;
    }
}

bool DerivedMaterials::reaches(std::size_t relation) const noexcept
{
    return relation < m_relationGroups.size() && m_relationGroups[relation].has_value();
}

void DerivedMaterials::addCells(input::ColumnOverlay& overlay, std::span<const double> fields,
                                std::optional<std::size_t> relation) const
{
    constexpr const char* COLUMNS[]{"youngsModulus", "shearModulus", "poissonRatio"};
    constexpr const char* SUMMARY[]{"e", "g", "nu"};
    for (std::size_t index = 0; index < m_groups.size(); ++index)
    {
        if (relation && (!reaches(*relation) || m_relationGroups[*relation] != index))
            continue;
        const auto& group = m_groups[index];
        auto values = group.base;
        for (std::size_t field = 0; field < 3; ++field)
            if (group.relations[field])
                values[field] = fields[*group.relations[field]];
        const auto dependent = group.blank == 1 ? 0U : group.blank == 2 ? 1U : 2U;
        if (dependent == 0)
            values[0] = 2.0 * values[1] * (1.0 + values[2]);
        else if (dependent == 1)
            values[1] = values[0] / (2.0 * (1.0 + values[2]));
        else
            values[2] = values[0] / (2.0 * values[1]) - 1.0;
        if (!std::isfinite(values[dependent]))
            throw std::invalid_argument("a designed IsoMat default is not finite");
        overlay.setValue("materials/IsoMat", COLUMNS[dependent], {group.cardRow, -1, values[dependent]});
        if (group.summaryRow >= 0)
            overlay.setValue("materialsTable", SUMMARY[dependent], {group.summaryRow, -1, values[dependent]});
    }
}
} // namespace oos::design
