#include "oos/elements/bars_beams/BarStations.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

constexpr std::string_view ACTUAL_LENGTH = "ActualLength";
constexpr const char* IGNORED_CARD = "ignored-card";

base::Diagnostic stationProblem(std::int32_t elementId, const char* field, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = problem_code::INVALID_FIELD;
    diagnostic.card = BAR_STATIONS_CARD;
    diagnostic.id = elementId;
    diagnostic.field = field;
    diagnostic.message = std::move(message);
    return diagnostic;
}

} // namespace

BarStations::BarStations(const ElementBatchView& view)
{
    const base::ContractReader* table = view.snapshot().barStations();
    if (table == nullptr || table->rowCount() == 0)
        return;
    m_ids = table->values<std::int32_t>("id");
    m_scale = table->texts("stationScale");
    m_listed = table->lists<double>("listedStations");
    m_count = table->values<std::int32_t>("stationCount");
    m_first = table->values<double>("station1");
    m_spacing = table->values<double>("stationSpacing");
    m_rowById.reserve(m_ids.size());
    for (std::size_t row = 0; row < m_ids.size(); ++row)
        m_rowById.emplace_back(m_ids[row], static_cast<std::int64_t>(row));
    // The first card of an element is the one read: the sort keeps the order of the cards of one element.
    std::stable_sort(m_rowById.begin(), m_rowById.end(),
                     [](const auto& left, const auto& right) { return left.first < right.first; });
}

std::int64_t BarStations::cardRowOf(std::int32_t elementId) const
{
    const auto found = std::lower_bound(m_rowById.begin(), m_rowById.end(), elementId,
                                        [](const auto& entry, std::int32_t id) { return entry.first < id; });
    return found != m_rowById.end() && found->first == elementId ? found->second : -1;
}

Resolution<std::vector<double>> BarStations::fractionsOf(std::int32_t elementId, double length) const
{
    using Result = Resolution<std::vector<double>>;
    std::vector<double> fractions; // bounded by the stations of one card
    const std::int64_t cardRow = cardRowOf(elementId);
    if (cardRow < 0)
        return Result::success(std::move(fractions));
    const auto r = static_cast<std::size_t>(cardRow);
    const double scale = m_scale[cardRow] == ACTUAL_LENGTH ? 1.0 / length : 1.0;
    const bool isSpaced = m_count[r] > 0;
    const std::span<const double> listed = m_listed[cardRow];
    const std::size_t count = isSpaced ? static_cast<std::size_t>(m_count[r]) : listed.size();
    for (std::size_t k = 0; k < count; ++k)
    {
        const double station = isSpaced ? m_first[r] + m_spacing[r] * static_cast<double>(k) : listed[k];
        const double fraction = station * scale;
        if (!std::isfinite(fraction) || fraction < -BAR_STATION_END_TOLERANCE ||
            fraction > 1.0 + BAR_STATION_END_TOLERANCE)
            return Result::failure(stationProblem(
                elementId, isSpaced ? "stationSpacing" : "listedStations",
                base::concatenate({"expected the stations of Bar2Stations ", std::to_string(elementId),
                                   " between the ends of its Bar2 (0 to ", std::to_string(1.0 / scale), " in ",
                                   std::string(m_scale[cardRow]), "), found ", std::to_string(station),
                                   "; give stations within the length of the element"})));
        // An end is always written: a station on it adds no row.
        if (fraction > BAR_STATION_END_TOLERANCE && fraction < 1.0 - BAR_STATION_END_TOLERANCE)
            fractions.push_back(fraction);
    }
    std::sort(fractions.begin(), fractions.end());
    const auto isSame = [](double left, double right) { return right - left <= BAR_STATION_END_TOLERANCE; };
    fractions.erase(std::unique(fractions.begin(), fractions.end(), isSame), fractions.end());
    return Result::success(std::move(fractions));
}

std::vector<std::int64_t> BarStations::rowsOf(const ElementBatchView& view) const
{
    std::vector<std::int64_t> rows; // bounded by the cards
    if (isEmpty())
        return rows;
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
        if (has(view.elementId(row)))
            rows.push_back(row);
    return rows;
}

void BarStations::reportUnusedCards(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    if (isEmpty())
        return;
    std::vector<std::int32_t> elementIds; // bounded by the batch
    elementIds.reserve(static_cast<std::size_t>(view.rowCount()));
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
        elementIds.push_back(view.elementId(row));
    std::sort(elementIds.begin(), elementIds.end());
    for (const auto& [elementId, cardRow] : m_rowById)
    {
        if (std::binary_search(elementIds.begin(), elementIds.end(), elementId))
            continue;
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = IGNORED_CARD;
        diagnostic.card = BAR_STATIONS_CARD;
        diagnostic.id = elementId;
        diagnostic.field = "id";
        diagnostic.message = base::concatenate(
            {"expected Bar2Stations ", std::to_string(elementId), " to name a ", std::string(view.card()),
             " of the model, found none with that ID: the card is not used; give the ID of the bar whose stations it "
             "lists, or remove the card"});
        sink.report(std::move(diagnostic));
    }
}

void reportBarStationsNotUsed(const ElementBatchView& view, std::string_view formulation, base::DiagnosticSink& sink)
{
    const BarStations stations(view);
    if (stations.rowsOf(view).empty())
        return;
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Warning;
    diagnostic.code = IGNORED_CARD;
    diagnostic.card = BAR_STATIONS_CARD;
    diagnostic.message = base::concatenate(
        {"the Bar2Stations cards are not used by the ", std::string(view.card()), " elements of ",
         std::string(formulation), ": their forces and stresses are written at their two ends; nothing to do, or "
                                   "remove the cards to drop this warning"});
    sink.report(std::move(diagnostic));
}

} // namespace oos::elements
