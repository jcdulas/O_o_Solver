#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>

namespace oos::input
{

// The non-structural mass of the model, expanded by O_o_Reader (nonStructMassTable): one row per NonStructMass,
// NonStructMass1, NonStructMassLumped, NonStructMassLumped1 target (NonStructMassUnion resolved) with the value as the
// card gives it and the mass it adds. The concentrated masses (PointMassMatrix, PointMassInertia, CMASSi) are element
// batches of the snapshot, and the mass parameters (WeightToMass, CoupledMass) PARAM values.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class MassTable
{
public:
    explicit MassTable(base::ImportedTable nonStructuralMass);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    base::ContractReader m_reader;
};

} // namespace oos::input
