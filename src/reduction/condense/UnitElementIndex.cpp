#include "oos/reduction/condense/UnitElementIndex.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementFamily.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace oos::reduction
{

namespace
{

bool hasMatrix(const assembly::AssemblyBatch& batch) noexcept
{
    const elements::FacetSet facets = batch.family->facets();
    return facets.has(elements::Facet::Stiffness) || facets.has(elements::Facet::Mass);
}

} // namespace

UnitElementIndex::UnitElementIndex(const assembly::ElementSet& elements, base::MemoryBudget& budget)
{
    const std::span<const assembly::AssemblyBatch> batches = elements.batches();
    std::size_t count = 0;
    for (const assembly::AssemblyBatch& batch : batches)
        count += hasMatrix(batch) ? static_cast<std::size_t>(batch.view.rowCount()) : 0;
    m_places = base::BudgetedBuffer<ElementPlace>(budget, count, "unit element index");
    std::size_t next = 0;
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        if (!hasMatrix(batches[b]))
            continue;
        for (std::int64_t row = 0; row < batches[b].view.rowCount(); ++row)
            m_places[next++] = ElementPlace{batches[b].view.elementId(row), static_cast<std::uint32_t>(b), row};
    }
    std::sort(m_places.data(), m_places.data() + count,
              [](const ElementPlace& a, const ElementPlace& b) { return a.elementId < b.elementId; });
    for (std::size_t k = 1; k < count; ++k)
        if (m_places[k].elementId == m_places[k - 1].elementId)
            throw std::invalid_argument(base::concatenate(
                {"UnitElementIndex: element ", std::to_string(m_places[k].elementId), " is in two element batches (",
                 elements.batches()[m_places[k - 1].batchIndex].view.card(), " and ",
                 elements.batches()[m_places[k].batchIndex].view.card(),
                 "); element IDs must be unique across the element cards"}));
}

std::optional<ElementPlace> UnitElementIndex::find(base::EntityId elementId) const noexcept
{
    const ElementPlace* first = m_places.data();
    const ElementPlace* last = first + m_places.size();
    const ElementPlace* found = std::lower_bound(
        first, last, elementId, [](const ElementPlace& place, base::EntityId id) { return place.elementId < id; });
    if (found == last || found->elementId != elementId)
        return std::nullopt;
    return *found;
}

} // namespace oos::reduction
