#pragma once

// The output stations of the bars: the Bar2Stations cards of the model (the table barStations of the snapshot,
// input/contract/TableContracts), each giving a Bar2 stations along its length where its forces, stresses and strains
// are written besides its two ends.
//
// A card lists its stations or gives them equally spaced (a count, the first one and their spacing), as fractions of
// the length of the bar ("FractionalLength") or as distances from end A ("ActualLength"); the length is that of the
// element between its ends A and B, offsets included. A station on an end is that end, which is always written; a
// station beyond end B is refused.
//
// Thread-safety: immutable once built. Ownership: borrows the table of the snapshot of the view it was built from.

#include "oos/base/arrow/ListView.hpp"
#include "oos/base/arrow/Utf8View.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::elements
{

// The card of the output stations, as the diagnostics name it.
inline constexpr const char* BAR_STATIONS_CARD = "Bar2Stations";
// A station within this fraction of the length of an end of the bar is that end.
inline constexpr double BAR_STATION_END_TOLERANCE = 1.0e-9;

class BarStations
{
public:
    // The cards of the snapshot of `view`; a model without the table has none.
    explicit BarStations(const ElementBatchView& view);

    [[nodiscard]] bool isEmpty() const noexcept { return m_rowById.empty(); }
    // Whether a card gives stations to element `elementId`.
    [[nodiscard]] bool has(std::int32_t elementId) const { return cardRowOf(elementId) >= 0; }
    // The stations of element `elementId` strictly between its ends, as fractions of its length `length`, increasing
    // and once each; none for an element without card. Fails with the diagnostic of its card when a station is not a
    // finite position within the element.
    [[nodiscard]] Resolution<std::vector<double>> fractionsOf(std::int32_t elementId, double length) const;
    // The batch rows of the elements of `view` with a card, increasing.
    [[nodiscard]] std::vector<std::int64_t> rowsOf(const ElementBatchView& view) const;
    // Reports, as warnings, the cards that name no element of `view` (the card is then not used).
    void reportUnusedCards(const ElementBatchView& view, base::DiagnosticSink& sink) const;

private:
    // The row of the card of element `elementId` in the table (the first one given), or −1.
    [[nodiscard]] std::int64_t cardRowOf(std::int32_t elementId) const;

    std::span<const std::int32_t> m_ids;
    base::Utf8View m_scale;
    base::ListView<double> m_listed;
    std::span<const std::int32_t> m_count;
    std::span<const double> m_first;
    std::span<const double> m_spacing;
    // (element ID, card row), sorted by element; bounded by the cards.
    std::vector<std::pair<std::int32_t, std::int64_t>> m_rowById;
};

// Reports, as one warning, that the output stations of the model are not used by the formulation `formulation` of
// the bars of `view` (their forces and stresses are then written at their ends); nothing for a model without cards.
void reportBarStationsNotUsed(const ElementBatchView& view, std::string_view formulation, base::DiagnosticSink& sink);

} // namespace oos::elements
