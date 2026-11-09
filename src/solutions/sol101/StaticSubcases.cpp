#include "oos/solutions/sol101/StaticSubcases.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

namespace oos::solutions
{

namespace
{

// The results of a deck without SUBCASE are numbered as subcase 1; O_o_Reader gives its selections as the global
// section, subcase 0 of kind GLOBAL.
constexpr base::EntityId GLOBAL_SUBCASE = 0;
constexpr base::EntityId SUBCASE_OF_A_DECK_WITHOUT_SUBCASES = 1;

base::Diagnostic subcaseProblem(base::EntityId subcase, std::string code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = "SubcaseStart";
    diagnostic.id = subcase;
    diagnostic.message = std::move(message);
    return diagnostic;
}

StaticSubcase subcaseAtRow(const input::SubcaseTable& table, std::int64_t row, base::EntityId label,
                           base::EntityId tableId)
{
    StaticSubcase subcase;
    subcase.label = label;
    subcase.tableId = tableId;
    subcase.spc = table.selectedSetAtRow(row, "spc");
    subcase.mpc = table.selectedSetAtRow(row, "mpc");
    subcase.load = table.selectedSetAtRow(row, "load");
    subcase.temperatureLoad = table.selectedSetAtRow(row, "temperatureLoad");
    subcase.temperatureInitial = table.selectedSetAtRow(row, "temperatureInitial");
    subcase.deformation = table.deformationSetAtRow(row);
    subcase.contact = table.contactTableAtRow(row);
    subcase.nonStructuralMass = table.selectedSetAtRow(row, "nonStructMass");
    return subcase;
}

} // namespace

InitialStrainKey initialStrainKeyOf(const StaticSubcase& subcase)
{
    InitialStrainKey key;
    key.temperatureLoad = subcase.temperatureLoad;
    // TEMPERATURE(INITIAL) without TEMPERATURE(LOAD) makes no thermal strain (a rule of the input format).
    key.temperatureInitial = subcase.temperatureLoad ? subcase.temperatureInitial : std::nullopt;
    key.deformation = subcase.deformation;
    return key;
}

std::unique_ptr<assembly::TemperatureField> initialStrainsOf(const input::ModelSnapshot& snapshot,
                                                             const input::PointIndex& points,
                                                             const InitialStrainKey& key, base::MemoryBudget& budget,
                                                             base::DiagnosticSink& sink)
{
    if (key.isEmpty())
        return nullptr;
    // A thermal load alone takes the constructor it always took (the same field, the same bits).
    if (!key.deformation)
        return std::make_unique<assembly::TemperatureField>(snapshot, points, *key.temperatureLoad,
                                                            key.temperatureInitial, budget, sink);
    return std::make_unique<assembly::TemperatureField>(snapshot, points, key.temperatureLoad, key.temperatureInitial,
                                                        key.deformation, budget, sink);
}

std::unique_ptr<assembly::TemperatureField> initialStrainsOf(const input::ModelSnapshot& snapshot,
                                                             const input::PointIndex& points,
                                                             const StaticSubcase& subcase, base::MemoryBudget& budget,
                                                             base::DiagnosticSink& sink)
{
    return initialStrainsOf(snapshot, points, initialStrainKeyOf(subcase), budget, sink);
}

std::vector<StaticSubcase> staticSubcasesOf(const input::ModelSnapshot& snapshot,
                                            const std::vector<base::EntityId>& asked, base::DiagnosticSink& sink)
{
    std::vector<StaticSubcase> result; // bounded by the subcases of the deck
    if (!snapshot.hasTable(input::TableRole::Subcases))
    {
        // A snapshot without Case Control (built by hand): one subcase that selects nothing.
        result.push_back({});
        return result;
    }
    const input::SubcaseTable& table = snapshot.subcases();
    const std::span<const std::int32_t> ids = table.ids();
    const base::DictionaryView kinds = table.kinds();
    std::vector<StaticSubcase> all; // every SUBCASE and SUBCOM of the deck, in order
    bool hasGlobal = false;
    std::int64_t globalRow = -1;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const std::string_view kind = kinds[row];
        const base::EntityId id = ids[static_cast<std::size_t>(row)];
        if (kind == "global")
        {
            hasGlobal = true;
            globalRow = row;
            continue;
        }
        if (kind == "subcase")
        {
            all.push_back(subcaseAtRow(table, row, id, id));
            continue;
        }
        if (kind == "auxiliary")
            continue; // AUXCASE: a subcase of an auxiliary model (SOL 200 shape basis), not a static load case
        if (kind == "subcaseCombination")
        {
            StaticSubcase combination = subcaseAtRow(table, row, id, id);
            combination.isCombination = true;
            const input::SubcaseTable::Combination terms = table.combinationAtRow(row);
            if (terms.subcases.empty())
            {
                sink.report(subcaseProblem(
                    id, "subcase-combination-without-terms",
                    base::concatenate({"SubcaseCombination ", std::to_string(id),
                                       " gives no subcases to combine (no SubcaseCombinationFactors, or a "
                                       "subcasesTable without combined_subcases); write its "
                                       "SubcaseCombinationFactors coefficients"})));
                continue;
            }
            combination.combined.assign(terms.subcases.begin(), terms.subcases.end());
            combination.coefficients.assign(terms.coefficients.begin(), terms.coefficients.end());
            all.push_back(std::move(combination));
            continue;
        }
        sink.report(subcaseProblem(
            id, "unsupported-subcase",
            base::concatenate({"subcase ", std::to_string(id), " is a ", kind,
                               ", which this version does not solve (SubcaseStart and SubcaseCombination are); "
                               "remove it or ask for the other subcases"})));
    }
    if (all.empty() && hasGlobal)
        all.push_back(subcaseAtRow(table, globalRow, SUBCASE_OF_A_DECK_WITHOUT_SUBCASES, GLOBAL_SUBCASE));

    const auto find = [&](base::EntityId label) -> const StaticSubcase*
    {
        for (const StaticSubcase& subcase : all)
            if (subcase.label == label)
                return &subcase;
        return nullptr;
    };
    // The SUBCOM terms must be SUBCASEs of the deck.
    for (const StaticSubcase& subcase : all)
        for (const base::EntityId term : subcase.combined)
        {
            const StaticSubcase* found = find(term);
            if (found == nullptr || found->isCombination)
                sink.report(subcaseProblem(subcase.label, "subcase-combination-unknown-term",
                                           base::concatenate({"SubcaseCombination ", std::to_string(subcase.label),
                                                              " combines subcase ", std::to_string(term),
                                                              ", which is not a SubcaseStart of the deck; correct "
                                                              "the SubcaseCombinationFactors or the subcase "
                                                              "order"})));
        }
    if (asked.empty())
        return all;
    std::vector<base::EntityId> wanted; // bounded by the subcases of the deck
    for (const base::EntityId label : asked)
    {
        const StaticSubcase* subcase = find(label);
        if (subcase == nullptr)
            throw std::invalid_argument(base::concatenate(
                {"subcases: expected subcases of the Case Control of the model, found ", std::to_string(label),
                 "; ask for one of the SubcaseStart or SubcaseCombination IDs of the deck (or 1 "
                 "for a "
                 "deck without SubcaseStart)"}));
        for (const base::EntityId term : subcase->combined)
            if (std::find(wanted.begin(), wanted.end(), term) == wanted.end() &&
                std::find(asked.begin(), asked.end(), term) == asked.end())
                wanted.push_back(term);
        if (std::find(wanted.begin(), wanted.end(), label) == wanted.end())
            wanted.push_back(label);
    }
    for (const base::EntityId label : wanted)
        if (const StaticSubcase* subcase = find(label))
            result.push_back(*subcase);
    return result;
}

} // namespace oos::solutions
