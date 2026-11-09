#include "oos/design/sensitivity/AffectedElements.hpp"

#include "oos/design/responses/ElementCatalog.hpp"
#include "oos/elements/fasteners/weld/WeldElement.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

[[nodiscard]] bool isMaterialIdColumn(const std::string& name)
{
    // matId, matId1 ... and the materials a shell names by their role (membraneMatId, bendingMatId, ...); not
    // matFrameId, a coordinate system.
    return name == "matId" || name.ends_with("MatId") ||
           (name.size() == 6 && name.starts_with("matId") && name[5] >= '1' && name[5] <= '9');
}

// The rows of the int32 material ID columns of a table that name `matId`, as the IDs of those rows.
void collectNamers(const base::ContractReader& reader, std::span<const std::int32_t> ids, std::int32_t matId,
                   std::set<std::int32_t>& found)
{
    for (const std::string& column : reader.table().columnNames())
    {
        if (!isMaterialIdColumn(column) || reader.columnType(column) != "int32")
            continue;
        const std::span<const std::int32_t> matIds = reader.values<std::int32_t>(column, base::NullPolicy::Allow);
        const base::ValidityView valid = reader.validity(column);
        for (std::int64_t row = 0; row < reader.rowCount(); ++row)
            if (valid.isValid(row) && matIds[static_cast<std::size_t>(row)] == matId)
                found.insert(ids[static_cast<std::size_t>(row)]);
    }
}

// The Weld of `snapshot` that read the thickness of shell property `propId` (a side on that property or on a shell
// of it, elements::weldSideProperty): a designed thickness changes their beam length when the mean thickness sets it.
void collectWeldsOnProperty(const input::ModelSnapshot& snapshot, std::int32_t propId, std::set<std::int32_t>& found)
{
    const input::ElementBatch* welds = snapshot.findElementBatch("Weld");
    if (welds == nullptr)
        return;
    const base::ContractReader& reader = welds->reader();
    const base::Utf8View types = reader.texts("kind");
    const std::span<const std::int32_t> ida = reader.values<std::int32_t>("patchIdA");
    const std::span<const std::int32_t> idb = reader.values<std::int32_t>("patchIdB");
    for (std::int64_t row = 0; row < welds->rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        for (const std::int32_t side : {ida[r], idb[r]})
            if (elements::weldSideProperty(snapshot, types[row], side) == propId)
                found.insert(welds->ids()[r]);
    }
}

[[nodiscard]] bool isShellThicknessCard(std::string_view card)
{
    return card == "ShellProp" || card == "CompProp" || card == "CompPropGlobal";
}

} // namespace

AffectedElements::AffectedElements(const DesignModel& model, const assembly::ElementSet& set)
{
    const input::ModelSnapshot& snapshot = set.index().snapshot();
    const ElementCatalog catalog(snapshot);
    // The batch of the element set of each card.
    std::map<std::string, std::size_t, std::less<>> batchOfCard;
    for (std::size_t b = 0; b < set.batches().size(); ++b)
        batchOfCard.emplace(set.batches()[b].view.card(), b);
    // PERFORMANCE: the elements of a relation gathered, then sorted and made unique (the order of a std::set of them),
    // where a set made a node per element: 0.08 s for the 167 000 shells of one ShellProp at 1 M DOFs.
    // Measured: the threaded evaluation of a design point.
    const auto add = [&](const ElementEntry& entry, std::vector<ElementRef>& refs)
    {
        const auto found = batchOfCard.find(entry.card);
        if (found != batchOfCard.end())
            refs.push_back({found->second, entry.row});
    };

    m_elements.resize(model.relations().size());
    for (std::size_t r = 0; r < model.relations().size(); ++r)
    {
        const DesignRelation& relation = model.relations()[r];
        std::vector<ElementRef> refs; // bounded by the elements the relation reaches
        switch (relation.targetKind)
        {
        case TargetKind::Property:
        {
            for (const ElementEntry* entry : catalog.elementsOfProperty(relation.targetCard, relation.targetId))
                add(*entry, refs);
            // A shell property also reaches the welds whose length its thickness may set (WeldDerivatives).
            if (isShellThicknessCard(relation.targetCard))
            {
                std::set<std::int32_t> welds;
                collectWeldsOnProperty(snapshot, relation.targetId, welds);
                for (const std::int32_t elementId : welds)
                    if (const ElementEntry* entry = catalog.find(elementId);
                        entry != nullptr && entry->card == "Weld")
                        add(*entry, refs);
            }
            break;
        }
        case TargetKind::Element:
            if (const ElementEntry* entry = catalog.find(relation.targetId);
                entry != nullptr && entry->card == relation.targetCard)
                add(*entry, refs);
            break;
        case TargetKind::Material:
        {
            for (const input::CardTable& table : snapshot.propertyTables())
            {
                std::set<std::int32_t> propIds;
                collectNamers(table.reader(), table.ids(), relation.targetId, propIds);
                for (const std::int32_t propId : propIds)
                    for (const ElementEntry* entry : catalog.elementsOfProperty(table.card(), propId))
                        add(*entry, refs);
            }
            for (const input::ElementBatch& batch : snapshot.elementBatches())
            {
                std::set<std::int32_t> elementIds;
                collectNamers(batch.reader(), batch.ids(), relation.targetId, elementIds);
                for (const std::int32_t elementId : elementIds)
                    if (const ElementEntry* entry = catalog.find(elementId);
                        entry != nullptr && entry->card == batch.card())
                        add(*entry, refs);
            }
            if (snapshot.hasTable(input::TableRole::LaminatePlies))
            {
                const base::ContractReader& plies = snapshot.laminatePlies();
                std::set<std::int32_t> propIds;
                collectNamers(plies, plies.values<std::int32_t>("propId"), relation.targetId, propIds);
                for (const ElementEntry& entry : catalog.entries())
                    if (propIds.count(entry.propId) != 0 &&
                        (entry.propertyCard == "CompProp" || entry.propertyCard == "CompPropGlobal"))
                        add(entry, refs);
            }
            break;
        }
        }
        std::sort(refs.begin(), refs.end());
        refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
        m_elements[r] = std::move(refs);
    }
}

} // namespace oos::design
