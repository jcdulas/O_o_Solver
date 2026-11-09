#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/MassTable.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace oos::assembly
{

// The non-structural mass that the NSM and NonStructMass1 cards of one NSM set add to each element (MassOptions::
// addedNonStructuralMass of the element families), from the nonStructMassTable of O_o_Reader (NSMADD expanded): per
// unit length for the line elements (CROD, Rod2InlineSection, Tube2, Bar2, Beam2: kind per_length), per unit
// area for the shells (CQUAD4, ShellTria3: kind per_area). A row names a property (every element of that PID: on an NSM
// card the property IDs are unique across the property types) or an element; the values of several rows on one element
// add up.
//
// Refused with an error Diagnostic: the lumped kinds NonStructMassLumped and NonStructMassLumped1
// ("nsm-lumped-unsupported": their mass is shared in proportion to the lengths or areas of the elements, which the
// table does not give per element), a row whose kind does not fit the element ("nsm-kind-mismatch"), a target the model
// does not have ("nsm-unknown-target").
//
// Thread-safety: immutable. Ownership: owns its per-batch arrays (budgeted); the budget must outlive it.
class NonStructuralMassShares
{
public:
    // The shares of the set `setId` (nothing when `table` is null). `set` gives the batches; the arrays follow its
    // batch order.
    NonStructuralMassShares(const ElementSet& set, const input::MassTable* table, base::EntityId setId,
                            base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // No added mass at all.
    explicit NonStructuralMassShares(const ElementSet& set);

    // The added mass of each row of batch `batchIndex` (an index in ElementSet::batches()), or an empty span when no
    // row of that batch receives any.
    [[nodiscard]] std::span<const double> sharesOf(std::size_t batchIndex) const noexcept
    {
        return m_shares[batchIndex].span();
    }

private:
    std::vector<base::BudgetedBuffer<double>> m_shares; // one per batch of the set
};

} // namespace oos::assembly
