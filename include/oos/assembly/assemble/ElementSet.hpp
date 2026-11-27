#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::assembly
{

// One element batch of the snapshot with the family that computes it.
struct AssemblyBatch
{
    const elements::ElementFamily* family;
    elements::ElementBatchView view;
};

// The element batches of a snapshot that the assemblers process: every batch whose card has a family in the registry,
// in the order of the snapshot (the order of the sums of the assembly). A batch whose card has no family
// is left out: the support catalog of the run refuses it before any assembly.
//
// The views carry the entities of the analysis (EntityLayout, those its DofMap numbers; null: none) and, with them,
// the model of their formulation (elements::formulationModelOf).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: borrows the index (and through it the snapshot), the
// registry and the layout, which must outlive it.
class ElementSet
{
public:
    ElementSet(const elements::SnapshotIndex& index, const elements::ElementRegistry& registry,
               const elements::EntityLayout* entities = nullptr);

    [[nodiscard]] std::span<const AssemblyBatch> batches() const noexcept { return m_batches; }
    [[nodiscard]] const elements::SnapshotIndex& index() const noexcept { return *m_index; }

    // Runs the check of every family on its batch: every element the facets cannot process becomes a diagnostic. With
    // a pool, a long batch of a family checked row by row is checked by ranges of CHECK_RANGE_ROWS on its threads (the
    // sink orders its diagnostics whatever the timing: the same report).
    void check(base::DiagnosticSink& sink, base::ThreadPool* pool = nullptr) const;
    static constexpr std::int64_t CHECK_RANGE_ROWS = 2048;

    // The batches whose family offers `facet`, by index in batches().
    [[nodiscard]] std::vector<std::size_t> batchesWith(elements::Facet facet) const;

    // Whether the elements are those of a formulation with entities (the mixed hybrid one), whose matrices may cost
    // tens of milliseconds each: the pass that forms them cuts its work finer then (ElementConnectivity).
    [[nodiscard]] bool hasEntities() const noexcept { return m_hasEntities; }

private:
    const elements::SnapshotIndex* m_index;
    bool m_hasEntities = false; // the views carry the entities of a formulation (the mixed hybrid one)
    std::vector<AssemblyBatch> m_batches; // bounded by the number of card types
};

} // namespace oos::assembly
