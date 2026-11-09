#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/CardTable.hpp"
#include "oos/input/tables/ElementBatch.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace oos::elements
{

// One element batch of a snapshot (all the elements of one card type), with what its element family reads besides
// the batch's own columns: the grid point positions in the basic system and the rotations of their displacement
// systems, the property and material tables, and the lookups of SnapshotIndex. The family reads the card's fields
// through columns() (the table of model.query("<CARD>")), each with its column name in the data contract.
//
// A view is cheap (references only) and is made once per batch; the facets of the family take it with a range of
// rows.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: borrows the snapshot, the batch and the index, which
// must outlive it.
class ElementBatchView
{
public:
    // Precondition: `batch` belongs to the snapshot of `index`.
    ElementBatchView(const input::ElementBatch& batch, const SnapshotIndex& index) noexcept
        : m_batch(&batch), m_index(&index)
    {
    }

    [[nodiscard]] const input::ElementBatch& batch() const noexcept { return *m_batch; }
    [[nodiscard]] const SnapshotIndex& index() const noexcept { return *m_index; }
    [[nodiscard]] const input::ModelSnapshot& snapshot() const noexcept { return m_index->snapshot(); }
    [[nodiscard]] const std::string& card() const noexcept { return m_batch->card(); }
    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_batch->rowCount(); }
    [[nodiscard]] const base::ContractReader& columns() const noexcept { return m_batch->reader(); }
    [[nodiscard]] std::int32_t elementId(std::int64_t row) const
    {
        return m_batch->ids()[static_cast<std::size_t>(row)];
    }
    // Whether the mass of an offset element is on its offset (the model's ModelParams OFFDEF LROFF or ELMZ) rather than
    // at its grid points (ELMOFF, the default): input::OffsetDefinition.
    [[nodiscard]] bool isMassOnOffsets() const noexcept
    {
        return input::carriesMassOnOffsets(snapshot().offsetDefinition());
    }
    // Whether the element loads of an offset element (thermal, pressure, BeamDistributedLoad) act through its offset
    // (LROFF, ELMZ) rather than at its grid points (ELMOFF, the default): input::OffsetDefinition.
    [[nodiscard]] bool areLoadsOnOffsets() const noexcept
    {
        return input::carriesLoadsOnOffsets(snapshot().offsetDefinition());
    }
    // Whether the differential stiffness of an offset element is formed (LROFF, ELMZ); under ELMOFF it is not and an
    // offset element is refused in a buckling analysis: input::OffsetDefinition.
    [[nodiscard]] bool hasOffsetDifferentialStiffness() const noexcept
    {
        return input::carriesLoadsOnOffsets(snapshot().offsetDefinition());
    }

    // The position of the grid point of a node table row, in the basic system.
    [[nodiscard]] base::Vec3 positionInBasic(std::int32_t gridRow) const;
    // The rotation from the displacement system CD of a grid point to the basic system, evaluated at the point: its
    // columns are the directions 1, 2, 3 of CD in the basic system, so that u_basic = R · u_CD for the translations
    // and the rotations alike. The identity for CD 0. Throws ContractError when the snapshot was taken without the CD
    // rotations and the grid point has a CD other than 0.
    [[nodiscard]] base::Mat3 displacementFrame(std::int32_t gridRow) const;

    // The table of a property card (PROD, SpringProp...), or nullptr when the model has none.
    [[nodiscard]] const input::CardTable* propertyTable(std::string_view card) const noexcept
    {
        return snapshot().findPropertyTable(card);
    }

private:
    const input::ElementBatch* m_batch;
    const SnapshotIndex* m_index;
};

} // namespace oos::elements
