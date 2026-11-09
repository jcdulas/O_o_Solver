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
// Thread-safety: immutable; concurrent reads are safe. Ownership: borrows the index (and through it the snapshot) and
// the registry, which must outlive it.
class ElementSet
{
public:
    ElementSet(const elements::SnapshotIndex& index, const elements::ElementRegistry& registry);

    [[nodiscard]] std::span<const AssemblyBatch> batches() const noexcept { return m_batches; }
    [[nodiscard]] const elements::SnapshotIndex& index() const noexcept { return *m_index; }

    // Runs the check of every family on its batch: every element the facets cannot process becomes a diagnostic. With
    // a pool, a long batch of a family checked row by row is checked by ranges of CHECK_RANGE_ROWS on its threads (the
    // sink orders its diagnostics whatever the timing: the same report).
    void check(base::DiagnosticSink& sink, base::ThreadPool* pool = nullptr) const;
    static constexpr std::int64_t CHECK_RANGE_ROWS = 2048;

    // The batches whose family offers `facet`, by index in batches().
    [[nodiscard]] std::vector<std::size_t> batchesWith(elements::Facet facet) const;

private:
    const elements::SnapshotIndex* m_index;
    std::vector<AssemblyBatch> m_batches; // bounded by the number of card types
};

} // namespace oos::assembly
