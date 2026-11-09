#include "oos/aero/trim/TrimCase.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <algorithm>

namespace oos::aero
{

namespace
{

base::Diagnostic trimProblem(std::int32_t setId, std::string code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = "AeroTrim";
    diagnostic.id = setId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

TrimVariableKind kindOf(std::string_view kind)
{
    if (kind == "rigidBody")
        return TrimVariableKind::RigidBody;
    if (kind == "controlSurface")
        return TrimVariableKind::ControlSurface;
    if (kind == "user")
        return TrimVariableKind::User;
    return TrimVariableKind::Undefined;
}

TrimVariableState stateOf(std::string_view state, bool isFixed)
{
    if (state == "linked")
        return TrimVariableState::Linked;
    if (state == "scheduled")
        return TrimVariableState::Scheduled;
    if (state == "fixed" || (state.empty() && isFixed))
        return TrimVariableState::Fixed;
    return TrimVariableState::Free;
}

bool containsSubcase(std::span<const std::int32_t> subcases, base::EntityId label, base::EntityId tableId)
{
    return std::find(subcases.begin(), subcases.end(), label) != subcases.end() ||
           std::find(subcases.begin(), subcases.end(), tableId) != subcases.end();
}

} // namespace

std::optional<std::size_t> TrimCase::find(std::string_view label) const
{
    for (std::size_t i = 0; i < settings.size(); ++i)
        if (std::string_view(settings[i].label) == label)
            return i;
    return std::nullopt;
}

std::optional<TrimCase> TrimCase::forSubcase(const input::ModelSnapshot& snapshot, base::EntityId subcase,
                                             base::EntityId tableId, std::span<const AeroLink> links,
                                             base::DiagnosticSink& sink)
{
    if (!snapshot.hasTable(input::TableRole::TrimCases))
        return std::nullopt;
    const base::ContractReader& table = snapshot.aeroTable(input::TableRole::TrimCases);
    const std::span<const std::int32_t> setIds = table.values<std::int32_t>("setId");
    const base::ListView<std::int32_t> subcases = table.lists<std::int32_t>("subcases", base::NullPolicy::Allow);
    const base::ValidityView hasSubcases = table.validity("subcases");
    const std::span<const double> machs = table.values<double>("mach");
    const std::span<const double> pressures = table.values<double>("dynamicPressure");
    const bool hasAeqrColumn = table.hasColumn("equivalentAspectRatio");
    const std::span<const double> aeqrs = hasAeqrColumn
                                              ? table.values<double>("equivalentAspectRatio", base::NullPolicy::Allow)
                                              : std::span<const double>{};
    const base::ValidityView hasAeqr = hasAeqrColumn ? table.validity("equivalentAspectRatio") : base::ValidityView{};
    const base::Utf8View labels = table.texts("variable");
    const std::span<const double> values = table.values<double>("value", base::NullPolicy::Allow);
    const base::ValidityView hasValue = table.validity("value");
    const base::BooleanView isFixed = table.booleans("isFixed", base::NullPolicy::Allow);
    const base::DictionaryView states = table.dictionaryTexts("state", base::NullPolicy::Allow);
    const base::ValidityView hasState = table.validity("state");
    const base::DictionaryView kinds = table.dictionaryTexts("variableKind", base::NullPolicy::Allow);
    const base::ValidityView hasKind = table.validity("variableKind");

    std::optional<TrimCase> found;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        if (!hasSubcases.isValid(row) || !containsSubcase(subcases[row], subcase, tableId))
            continue;
        if (found && found->setId != setIds[index])
        {
            sink.report(
                trimProblem(setIds[index], "trim-two-sets",
                            base::concatenate({"subcase ", std::to_string(subcase), " is selected by TrimSelection ",
                                               std::to_string(found->setId), " and by TrimSelection ",
                                               std::to_string(setIds[index]),
                                               "; a subcase has one trim case: keep one TrimSelection selection"})));
            continue;
        }
        if (!found)
        {
            found.emplace();
            found->setId = setIds[index];
            found->subcase = subcase;
            found->mach = machs[index];
            found->dynamicPressure = pressures[index];
            found->aeqr = hasAeqrColumn && hasAeqr.isValid(row) ? aeqrs[index] : 1.0;
        }
        TrimSetting setting;
        setting.label = std::string(labels[row]);
        setting.kind = hasKind.isValid(row) ? kindOf(kinds[row]) : TrimVariableKind::Undefined;
        setting.state = stateOf(hasState.isValid(row) ? states[row] : std::string_view{}, isFixed[row]);
        setting.value = hasValue.isValid(row) ? values[index] : 0.0;
        found->settings.push_back(std::move(setting));
    }
    if (!found)
        return std::nullopt;

    TrimCase& trimCase = *found;
    const std::string name = base::concatenate({"AeroTrim ", std::to_string(trimCase.setId)});
    // The doublet lattice at k = 0 is the subsonic steady method; the supersonic methods are out of scope.
    if (!(trimCase.mach >= 0.0 && trimCase.mach < 1.0))
        sink.report(trimProblem(
            trimCase.setId, "trim-mach",
            base::concatenate({name,
                               ": expected a subsonic Mach number (0 <= AeroTrim.machNumber < 1, the "
                               "doublet-lattice method), found ",
                               std::to_string(trimCase.mach), "; supersonic trim is not supported by this version"})));
    if (!(trimCase.dynamicPressure > 0.0))
        sink.report(
            trimProblem(trimCase.setId, "trim-dynamic-pressure",
                        base::concatenate({name, ": expected a dynamic pressure AeroTrim.dynamicPressure > 0, found ",
                                           std::to_string(trimCase.dynamicPressure)})));
    if (!(trimCase.aeqr >= 0.0 && trimCase.aeqr <= 1.0))
        sink.report(
            trimProblem(trimCase.setId, "trim-aeqr",
                        base::concatenate({name, ": expected AeroTrim.equivalentAspectRatio between 0 and 1, found ",
                                           std::to_string(trimCase.aeqr)})));
    for (const TrimSetting& setting : trimCase.settings)
    {
        if (setting.state == TrimVariableState::Scheduled)
            sink.report(
                trimProblem(trimCase.setId, "trim-scheduled",
                            base::concatenate({name, ": the variable ", setting.label,
                                               " is scheduled by a AeroControlSchedule, which this version does not "
                                               "support; fix it on the AeroTrim or leave it free"})));
        if (setting.state != TrimVariableState::Linked)
            continue;
        std::vector<const AeroLink*> applying;
        for (const AeroLink& link : links)
            if ((link.id == 0 || link.id == trimCase.setId) && std::string(link.dependent) == setting.label)
                applying.push_back(&link);
        if (applying.size() != 1)
        {
            sink.report(trimProblem(
                trimCase.setId, applying.empty() ? "trim-link-missing" : "trim-link-repeated",
                base::concatenate({name, ": the variable ", setting.label,
                                   " is linked, expected one AeroTrimLink of ID ", std::to_string(trimCase.setId),
                                   " or ALWAYS that makes it depend on others, found ", std::to_string(applying.size()),
                                   applying.empty() ? " (the AeroTrimLink cards are read by the caller from the "
                                                      "model and passed to the solver)"
                                                    : ""})));
            continue;
        }
        AeroLink link = *applying.front();
        link.dependent = setting.label;
        for (LinkTerm& term : link.terms)
        {
            const std::optional<std::size_t> independent = trimCase.find(term.label);
            if (!independent || trimCase.settings[*independent].state == TrimVariableState::Linked)
                sink.report(trimProblem(
                    trimCase.setId, "trim-link-label",
                    base::concatenate({name, ": the AeroTrimLink of ", setting.label, " names ", term.label,
                                       independent
                                           ? ", which is itself linked; an independent variable of an AeroTrimLink "
                                             "must be free or fixed"
                                           : ", which is not a trim variable of the model (AeroRigidTrimVariable, "
                                             "AeroControlSurface, AeroController)"})));
        }
        trimCase.links.push_back(std::move(link));
    }
    return found;
}

} // namespace oos::aero
