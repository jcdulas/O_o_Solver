#include "oos/input/snapshot/SupportCatalog.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace oos::input
{

namespace
{

using base::Diagnostic;

// Where a card type is used in the model: how many entries (cards, or rows of an expanded table) and its first IDs.
struct CardUse
{
    std::string kind; // "element", "property", "material", "load", "constraint", "method"
    std::int64_t entryCount = 0;
    std::vector<std::int32_t> firstIds; // distinct, in order of appearance, at most MAX_LISTED_IDS
    bool hasMoreIds = false;

    void add(std::int32_t id)
    {
        ++entryCount;
        if (std::find(firstIds.begin(), firstIds.end(), id) != firstIds.end())
            return;
        if (firstIds.size() < SupportCatalog::MAX_LISTED_IDS)
            firstIds.push_back(id);
        else
            hasMoreIds = true;
    }
};

using CardUses = std::map<std::string, CardUse, std::less<>>;

// The card of the multipoint constraint equations, whose rows multiPointConstrTable gives without a card column.
constexpr std::string_view MULTIPOINT_CONSTRAINT_CARD = "MultiPointConstr";

// The card of the model parameters (MDLPRM), and its parameter OFFDEF by its neutral name (the card table's name).
constexpr std::string_view MODEL_PARAMS_CARD = "ModelParams";
constexpr std::string_view OFFSET_DEFINITION_PARAM = "OffsetDefinition";

// The refusal of an MDLPRM OFFDEF value O_o_Solver does not take (input::OffsetDefinition).
Diagnostic unsupportedOffsetDefinition(const std::string& value)
{
    const std::string keyword = std::string(OFFSET_DEFINITION_PARAM);
    const std::string parameter = base::concatenate({std::string(MODEL_PARAMS_CARD), " ", keyword});
    Diagnostic diagnostic;
    diagnostic.code = "unsupported-parameter";
    diagnostic.card = parameter;
    diagnostic.field = keyword;
    diagnostic.message = base::concatenate(
        {"expected ", parameter,
         " ElemOffsetsOnly (the default: the mass of offset elements at their grid points), LoadRecoveryOffsets or "
         "ElemOffsetZ (the mass on the offsets), found '",
         value,
         "': O_o_Solver does not form apart the parts of LoadRecoveryOffsets that value turns off; use one of "
         "those three, or remove the parameter"});
    return diagnostic;
}

CardUse& useOf(CardUses& uses, std::string_view card, std::string_view kind)
{
    auto found = uses.find(card);
    if (found == uses.end())
        found = uses.emplace(std::string(card), CardUse{std::string(kind), 0, {}, false}).first;
    return found->second;
}

void addRows(CardUses& uses, std::string_view card, std::string_view kind, std::span<const std::int32_t> ids)
{
    if (ids.empty())
        return;
    CardUse& use = useOf(uses, card, kind);
    for (const std::int32_t id : ids)
        use.add(id);
}

// The rows of a table whose card type is a dictionary column, grouped by card; the ID of a row is `ids[row]` when
// valid, otherwise `fallbackIds[row]` (the grid point of a permanent constraint, whose set is null).
void addDictionaryRows(CardUses& uses, std::string_view kind, const base::DictionaryView& cards,
                       std::span<const std::int32_t> ids, const base::ValidityView& validIds,
                       std::span<const std::int32_t> fallbackIds = {})
{
    for (std::int64_t row = 0; row < cards.size(); ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        const std::int32_t id = validIds.isValid(row) || fallbackIds.empty() ? ids[index] : fallbackIds[index];
        useOf(uses, cards[row], kind).add(id);
    }
}

void addLoadTable(CardUses& uses, const base::ContractReader& table)
{
    addDictionaryRows(uses, "load", table.dictionaryTexts("card"), table.values<std::int32_t>("setId"),
                      table.validity("setId"));
}

CardUses cardUsesOf(const ModelSnapshot& snapshot)
{
    CardUses uses;
    for (const ElementBatch& batch : snapshot.elementBatches())
        addRows(uses, batch.card(), "element", batch.ids());
    for (const CardTable& table : snapshot.propertyTables())
        addRows(uses, table.card(), "property", table.ids());
    for (const CardTable& table : snapshot.methodTables())
        addRows(uses, table.card(), "method", table.ids());
    if (snapshot.hasTable(TableRole::Materials))
    {
        const MaterialTable& materials = snapshot.materials();
        addDictionaryRows(uses, "material", materials.types(), materials.ids(), base::ValidityView());
    }
    if (snapshot.hasTable(TableRole::Loads))
    {
        const LoadTable& loads = snapshot.loads();
        addLoadTable(uses, loads.loads());
        addLoadTable(uses, loads.pressures());
        addLoadTable(uses, loads.lineLoads());
        addLoadTable(uses, loads.temperatures());
        if (loads.hasInertialLoads())
            addLoadTable(uses, loads.inertialLoads());
    }
    if (snapshot.hasTable(TableRole::Spc))
    {
        const ConstraintTable& constraints = snapshot.constraints();
        addDictionaryRows(uses, "constraint", constraints.spcCards(), constraints.spcSetIds(),
                          constraints.spcReader().validity("setId"), constraints.spcNodeIds());
        addRows(uses, MULTIPOINT_CONSTRAINT_CARD, "constraint", constraints.mpcSetIds());
    }
    return uses;
}

std::string idList(const CardUse& use)
{
    std::string text;
    for (const std::int32_t id : use.firstIds)
        text.append(text.empty() ? "" : ", ").append(std::to_string(id));
    if (use.hasMoreIds)
        text.append(", ...");
    return text;
}

Diagnostic unsupportedCard(const std::string& card, const CardUse& use)
{
    Diagnostic diagnostic;
    diagnostic.code = "unsupported-card";
    diagnostic.card = std::string(card);
    diagnostic.id = use.firstIds.empty() ? 0 : use.firstIds.front();
    diagnostic.message = base::concatenate(
        {"the ", use.kind, " card ", diagnostic.card, " is not supported by this version of O_o_Solver (",
         std::to_string(use.entryCount), use.entryCount == 1 ? " entry" : " entries", ", IDs ", idList(use),
         "); remove these cards from the model, or use a version of O_o_Solver that supports them"});
    return diagnostic;
}

std::string formatNumber(double value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

// The refusals of one field rule on one table (an element batch or a property table of the rule's card).
void checkFieldRule(const FieldRule& rule, const base::ContractReader& table, std::span<const std::int32_t> ids,
                    std::vector<Diagnostic>& refusals)
{
    if (!table.hasColumn(rule.field))
        return;
    const std::string type = table.columnType(rule.field);
    const base::ValidityView valid = table.validity(rule.field);
    const auto refuse = [&](std::int64_t row, double found)
    {
        Diagnostic diagnostic;
        diagnostic.code = "unsupported-field";
        diagnostic.card = std::string(rule.card);
        diagnostic.id = ids[static_cast<std::size_t>(row)];
        diagnostic.field = std::string(rule.field);
        diagnostic.message =
            base::concatenate({"expected ", formatNumber(rule.acceptedValue), " (", rule.reason, "), found ",
                               formatNumber(found), "; set the field to its default or remove the card"});
        refusals.push_back(std::move(diagnostic));
    };
    if (type == "float64")
    {
        const std::span<const double> values = table.values<double>(rule.field, base::NullPolicy::Allow);
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            if (valid.isValid(row) && values[static_cast<std::size_t>(row)] != rule.acceptedValue)
                refuse(row, values[static_cast<std::size_t>(row)]);
        }
    }
    else if (type == "int32")
    {
        const std::span<const std::int32_t> values = table.values<std::int32_t>(rule.field, base::NullPolicy::Allow);
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            const auto value = static_cast<double>(values[static_cast<std::size_t>(row)]);
            if (valid.isValid(row) && value != rule.acceptedValue)
                refuse(row, value);
        }
    }
}

} // namespace

SupportCatalog SupportCatalog::current()
{
    // The base catalog supports nothing: each module that implements cards adds them and the field rules of what it
    // does not implement yet.
    return {};
}

void SupportCatalog::addCard(std::string card)
{
    m_cards.insert(std::move(card));
}

void SupportCatalog::addFieldRule(FieldRule rule)
{
    m_fieldRules.push_back(std::move(rule));
}

bool SupportCatalog::isSupported(std::string_view card) const
{
    return m_cards.find(card) != m_cards.end();
}

base::SupportReport SupportCatalog::check(const ModelSnapshot& snapshot) const
{
    base::SupportReport report;
    report.module = "input";
    for (const auto& [card, use] : cardUsesOf(snapshot))
    {
        if (isSupported(card))
            report.supported.push_back(
                base::concatenate({std::string(card), " (", std::to_string(use.entryCount), ")"}));
        else
            report.refusals.push_back(unsupportedCard(card, use));
    }
    for (const FieldRule& rule : m_fieldRules)
    {
        if (!isSupported(rule.card))
            continue; // the card itself is refused already
        if (const ElementBatch* batch = snapshot.findElementBatch(rule.card))
            checkFieldRule(rule, batch->reader(), batch->ids(), report.refusals);
        if (const CardTable* table = snapshot.findPropertyTable(rule.card))
            checkFieldRule(rule, table->reader(), table->ids(), report.refusals);
    }
    if (const std::string& value = snapshot.unsupportedOffsetDefinition(); !value.empty())
        report.refusals.push_back(unsupportedOffsetDefinition(value));
    std::stable_sort(report.refusals.begin(), report.refusals.end(), [](const Diagnostic& a, const Diagnostic& b)
                     { return std::tie(a.card, a.id, a.field) < std::tie(b.card, b.id, b.field); });
    return report;
}

} // namespace oos::input
