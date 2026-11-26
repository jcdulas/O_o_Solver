#include "oos/assembly/loads/TemperatureField.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace oos::assembly
{

namespace
{

constexpr double NO_TEMPERATURE = std::numeric_limits<double>::quiet_NaN();
// The Case Control commands that select the temperature sets of a subcase: what the diagnostic of a set no card
// defines names.
constexpr std::string_view LOAD_SELECTION_COMMAND = "TemperatureSelection(LOAD)";
constexpr std::string_view INITIAL_SELECTION_COMMAND = "TemperatureSelection(INITIAL)";

base::Diagnostic temperatureProblem(std::string code, std::string_view card, base::EntityId setId, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = setId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The gradients of a LineElemTemperatureLoad across the section of its line, the columns gradient1a, gradient1b
// (along the element y axis at ends A and B), gradient2a and gradient2b (along z) of temperatureTable, in the order of
// ElementTemperature::lineGradients. A gradient is 0 where the table has no such column (a table built by hand), on
// the row of another card (null) and where the value is not a number.
class LineGradientColumns
{
public:
    explicit LineGradientColumns(const base::ContractReader& table)
    {
        static constexpr std::array<const char*, 4> NAMES{"gradient1a", "gradient1b", "gradient2a", "gradient2b"};
        for (std::size_t k = 0; k < NAMES.size(); ++k)
        {
            if (!table.hasColumn(NAMES[k]))
                continue;
            m_values[k] = table.values<double>(NAMES[k], base::NullPolicy::Allow);
            m_isGiven[k] = table.validity(NAMES[k]);
        }
    }

    [[nodiscard]] std::array<double, 4> at(std::int64_t row) const
    {
        std::array<double, 4> gradients{};
        for (std::size_t k = 0; k < gradients.size(); ++k)
        {
            if (m_values[k].empty() || !m_isGiven[k].isValid(row))
                continue;
            const double value = m_values[k][static_cast<std::size_t>(row)];
            gradients[k] = std::isfinite(value) ? value : 0.0;
        }
        return gradients;
    }

private:
    std::array<std::span<const double>, 4> m_values{}; // empty: the table has no such column
    std::array<base::ValidityView, 4> m_isGiven{};
};

} // namespace

TemperatureField::Set TemperatureField::gather(const input::ModelSnapshot& snapshot, const input::PointIndex& points,
                                               base::EntityId setId, std::string_view selection,
                                               base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const auto gridCount = static_cast<std::size_t>(snapshot.nodes().rowCount());
    Set set;
    set.atGrid = base::BudgetedBuffer<double>(budget, gridCount, "grid point temperatures");
    std::fill(set.atGrid.data(), set.atGrid.data() + gridCount, NO_TEMPERATURE);
    if (!snapshot.hasTable(input::TableRole::Temperatures))
        return set;
    const base::ContractReader& table = snapshot.loads().temperatures();
    const std::span<const std::int32_t> setIds = table.values<std::int32_t>("setId");
    const base::DictionaryView cards = table.dictionaryTexts("card");
    const base::DictionaryView targetTypes = table.dictionaryTexts("targetType");
    const std::span<const std::int32_t> targets = table.values<std::int32_t>("targetId", base::NullPolicy::Allow);
    const base::ValidityView isTargetGiven = table.validity("targetId");
    const std::span<const double> values = table.values<double>("temperature", base::NullPolicy::Allow);
    const std::span<const double> gradients = table.values<double>("gradient", base::NullPolicy::Allow);
    const base::ValidityView isGradientGiven = table.validity("gradient");
    const LineGradientColumns lineGradients(table);

    double defaultTemperature = NO_TEMPERATURE;
    // The element rows the loop below keeps (those with a target): the buffer holds exactly them, so that the view
    // the elements search by ID is sorted to its end.
    std::size_t elementCount = 0;
    bool isSetDefined = false; // a row of the table names the set
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        if (setIds[static_cast<std::size_t>(row)] != setId)
            continue;
        isSetDefined = true;
        if (targetTypes[row] == "element" && isTargetGiven.isValid(row))
            ++elementCount;
    }
    // A selected set that no card defines (a mistyped ID) would give no temperature without a word.
    if (!isSetDefined)
        sink.report(temperatureProblem(
            "load-unknown-set", selection, setId,
            base::concatenate({"expected the temperature cards of set ", std::to_string(setId), ", which a ", selection,
                               " selects, found none (no NodeTemperatureLoad, DefaultTemperatureLoad, "
                               "PlateTemperatureLoad or LineElemTemperatureLoad row of that set in temperatureTable); "
                               "correct the set ID or add its temperature cards"})));
    set.ofElements = base::BudgetedBuffer<elements::ElementTemperature>(budget, elementCount, "element temperatures");
    std::size_t next = 0;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        if (setIds[index] != setId)
            continue;
        const std::string_view card = cards[row];
        const std::string_view targetType = targetTypes[row];
        if (targetType == "default")
        {
            defaultTemperature = values[index];
            continue;
        }
        if (targetType == "element" && isTargetGiven.isValid(row))
        {
            const double gradient = isGradientGiven.isValid(row) ? gradients[index] : 0.0;
            set.ofElements[next++] = {targets[index], values[index], std::isfinite(gradient) ? gradient : 0.0,
                                      lineGradients.at(row)};
            continue;
        }
        if (targetType == "node" && isTargetGiven.isValid(row))
        {
            const std::optional<std::int32_t> grid = points.findGrid(targets[index]);
            if (!grid)
            {
                sink.report(temperatureProblem(
                    "load-unknown-point", card, setId,
                    base::concatenate({"expected a grid point of the model, found ID ", std::to_string(targets[index]),
                                       " in temperature set ", std::to_string(setId),
                                       "; correct the NodeTemperatureLoad card"})));
                continue;
            }
            set.atGrid[static_cast<std::size_t>(*grid)] = values[index];
            continue;
        }
        sink.report(temperatureProblem(
            "load-unsupported", card, setId,
            base::concatenate(
                {"the temperature card ", card, " of set ", std::to_string(setId),
                 " is not supported by this version of O_o_Solver (NodeTemperatureLoad, "
                 "DefaultTemperatureLoad, PlateTemperatureLoad, LineElemTemperatureLoad are); remove it"})));
    }
    if (std::isfinite(defaultTemperature))
        for (std::size_t grid = 0; grid < gridCount; ++grid)
            if (!std::isfinite(set.atGrid[grid]))
                set.atGrid[grid] = defaultTemperature;
    // The element temperatures sorted by ID (stable: the last card of an element is not decided here, the first
    // found wins in the lookup, as a PlateTemperatureLoad names an element once in a valid deck).
    std::stable_sort(set.ofElements.data(), set.ofElements.data() + next,
                     [](const elements::ElementTemperature& a, const elements::ElementTemperature& b)
                     { return a.elementId < b.elementId; });
    return set;
}

base::BudgetedBuffer<elements::ElementDeformation> TemperatureField::gatherDeformations(
    const input::ModelSnapshot& snapshot, base::EntityId setId, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    std::span<const std::int32_t> setIds;
    base::ListView<std::int32_t> elementLists;
    base::ListView<double> valueLists;
    if (snapshot.hasTable(input::TableRole::Deformations))
    {
        const base::ContractReader& table = snapshot.deformations();
        setIds = table.values<std::int32_t>("id");
        elementLists = table.lists<std::int32_t>("element");
        valueLists = table.lists<double>("deformation");
    }
    std::size_t count = 0;
    for (std::size_t row = 0; row < setIds.size(); ++row)
        if (setIds[row] == setId)
            count += elementLists[static_cast<std::int64_t>(row)].size();
    base::BudgetedBuffer<elements::ElementDeformation> deformations(budget, count, "element deformations");
    std::size_t next = 0;
    for (std::size_t row = 0; row < setIds.size(); ++row)
    {
        if (setIds[row] != setId)
            continue;
        const std::span<const std::int32_t> elementIds = elementLists[static_cast<std::int64_t>(row)];
        const std::span<const double> values = valueLists[static_cast<std::int64_t>(row)];
        if (elementIds.size() != values.size())
            throw base::ContractError(base::concatenate(
                {"deform, row ", std::to_string(row), ": expected as many deformations as elements, found ",
                 std::to_string(values.size()), " for ", std::to_string(elementIds.size()),
                 "; the table does not follow the data contract of O_o_Model: take the snapshot again"}));
        for (std::size_t k = 0; k < elementIds.size(); ++k)
            deformations[next++] = {elementIds[k], values[k]};
    }
    if (count == 0)
    {
        sink.report(temperatureProblem(
            "load-unknown-set", "AxialDeformationLoad", setId,
            base::concatenate({"expected the AxialDeformationLoad cards of set ", std::to_string(setId),
                               ", which a subcase selects with DeformationLoadSelection = ", std::to_string(setId),
                               ", found none; add them or correct the DeformationLoadSelection command"})));
        return deformations;
    }
    std::stable_sort(deformations.data(), deformations.data() + count,
                     [](const elements::ElementDeformation& a, const elements::ElementDeformation& b)
                     { return a.elementId < b.elementId; });
    for (std::size_t k = 1; k < count; ++k)
        if (deformations[k].elementId == deformations[k - 1].elementId)
            sink.report(temperatureProblem(
                "load-duplicate", "AxialDeformationLoad", setId,
                base::concatenate({"expected each element once in AxialDeformationLoad set ", std::to_string(setId),
                                   ", found element ", std::to_string(deformations[k].elementId),
                                   " twice (which deformation applies is ambiguous); give it one "
                                   "AxialDeformationLoad.deformation"})));
    return deformations;
}

TemperatureField::TemperatureField(const input::ModelSnapshot& snapshot, const input::PointIndex& points,
                                   base::EntityId loadSet, std::optional<base::EntityId> initialSet,
                                   base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_load(gather(snapshot, points, loadSet, LOAD_SELECTION_COMMAND, budget, sink))
{
    m_view.load = {m_load.atGrid.span(), m_load.ofElements.span()};
    if (initialSet)
    {
        m_initial = gather(snapshot, points, *initialSet, INITIAL_SELECTION_COMMAND, budget, sink);
        m_view.initial = {m_initial.atGrid.span(), m_initial.ofElements.span()};
    }
}

TemperatureField::TemperatureField(const input::ModelSnapshot& snapshot, const input::PointIndex& points,
                                   std::optional<base::EntityId> loadSet, std::optional<base::EntityId> initialSet,
                                   std::optional<base::EntityId> deformationSet, base::MemoryBudget& budget,
                                   base::DiagnosticSink& sink)
{
    if (!loadSet && !deformationSet)
        throw std::invalid_argument(
            "TemperatureField: expected a TemperatureSelection(LOAD) or a DeformationLoadSelection "
            "set, found neither");
    if (loadSet)
    {
        m_load = gather(snapshot, points, *loadSet, LOAD_SELECTION_COMMAND, budget, sink);
        m_view.load = {m_load.atGrid.span(), m_load.ofElements.span()};
        if (initialSet)
        {
            m_initial = gather(snapshot, points, *initialSet, INITIAL_SELECTION_COMMAND, budget, sink);
            m_view.initial = {m_initial.atGrid.span(), m_initial.ofElements.span()};
        }
    }
    m_view.hasTemperatureLoad = loadSet.has_value();
    if (deformationSet)
    {
        m_deformations = gatherDeformations(snapshot, *deformationSet, budget, sink);
        m_view.deformations = m_deformations.span();
        m_view.deformationSet = *deformationSet;
    }
}

} // namespace oos::assembly
