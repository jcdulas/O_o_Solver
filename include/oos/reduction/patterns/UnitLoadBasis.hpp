#pragma once

#include "oos/assembly/loads/CompactLoadMatrix.hpp"
#include "oos/assembly/loads/LoadBasis.hpp"
#include "oos/assembly/loads/LoadFamily.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/reduction/patterns/BasisCoefficients.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// The loads of one family of the model's load cases, as the static solution describes them for its own load basis
// (assembly::LoadBasis::Input: the compact load matrix of the cases and, where the deck gives them, the load sets
// as written with their factors), and the rows of that description that load the unit.
struct UnitFamilyLoads
{
    assembly::LoadFamily family = assembly::LoadFamily::Concentrated;
    const assembly::CompactLoadMatrix* cases = nullptr;
    const assembly::CompactLoadMatrix* loadSets = nullptr; // null: no load sets as written
    const linalg::DenseMatrix* setCoefficients = nullptr;  // with loadSets: the factors (sets × cases)
    bool offersLoadedDofs = false;                         // the compact rows are DOFs (concentrated, enforced)
    // With offersLoadedDofs: the g-set DOF of each compact row (the stamp of a loaded-DOF basis names DOFs, never the
    // row numbers, which follow the order in which the cases load them). Empty otherwise.
    std::span<const base::DofIndex> dofOfRow;
    assembly::LoadBasisTolerance tolerance = assembly::LoadBasisTolerance::Roundoff;
    // One flag per compact row: nonzero when the row loads a DOF the unit's patterns carry (its interior, and its
    // boundary when the caller gives the unit the loads of its boundary points). Empty: every row (a field over the
    // whole model: the inertial and thermal families, one row per load set).
    std::span<const std::uint8_t> isUnitRow;
    // Step 10.24: the cases are exact combinations of the load sets with their factors
    // (assembly::LoadBasis::Input::isCombinationOfSets): `cases` may be null, the rows those of `loadSets`.
    bool isCombinationOfSets = false;
};

// One vector of a unit's load basis, named in the model's description of its family (the compact rows and columns of
// UnitFamilyLoads), so that the caller makes it as the full path makes its own basis vectors: a load set as written, a
// unit load (or a unit enforced displacement) on the DOF of a compact row, or the family part of a load case.
struct UnitBasisVector
{
    assembly::LoadFamily family = assembly::LoadFamily::Concentrated;
    assembly::BasisVector source; // LoadedRow: the model's compact row; LoadSet, Case: the column, as in the model
};

// The load basis of one unit: the load basis (assembly::LoadBasis) chosen on
// the loads of the unit's interior, exact and never truncated, with the coefficients α of every load case.
//
// Method: for each family, the compact matrices of the model are restricted to the rows that load the unit (the other
// rows give no load on its DOFs: the loads of a family are linear in its compact values, each row acting on its own
// DOFs or element), renumbered in their order, and assembly::LoadBasis chooses the smallest exact basis among the load
// sets as written, the loaded DOFs, the rank at round-off and the cases, with its reconstruction check of every case
// (LOAD_BASIS_RESIDUAL_LIMIT): f_case,unit = Σ_k α_k,case f_k,unit holds to round-off for every case, so the patterns
// g_k = K_oo⁻¹·f_k and f*_k condensed from the basis vectors give every case exactly. A case that does not load the
// unit is a zero column: no vector, zero coefficients. The basis vectors are named in the model's numbering
// (UnitBasisVector): a loaded DOF by its compact row of the model, a load set or a case by its column.
//
// With the loaded-DOF basis the patterns depend only on which DOFs are loaded, not on the values ("new
// load values on the same points reuse the condensation"): its vectors are put in the order of their DOFs (not of the
// compact rows, which follow the order in which the cases first load them), and patternStamp() is a stamp of the
// family and of those DOFs, the same for any values, which the condensation cache keys the patterns with;
// for the other bases the patterns depend on the loads themselves and the stamp mixes the values of the sets and cases
// taken.
//
// Thread-safety: an immutable value once chosen. Ownership: owns its vectors, reports and coefficients (budgeted).
class UnitLoadBasis
{
public:
    // caseCount: the load cases (the columns of every family's cases matrix). Throws std::invalid_argument when a
    // family's matrices disagree with caseCount or with its row flags, or a family is given twice; MemoryBudgetError.
    [[nodiscard]] static UnitLoadBasis choose(std::span<const UnitFamilyLoads> families, std::size_t caseCount,
                                              base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);

    [[nodiscard]] std::size_t size() const noexcept { return m_vectors.size(); }
    [[nodiscard]] std::size_t caseCount() const noexcept { return m_coefficients.caseCount(); }
    [[nodiscard]] std::span<const UnitBasisVector> vectors() const noexcept { return m_vectors; }
    // α (size() × caseCount()), the families stacked in the order given.
    [[nodiscard]] const BasisCoefficients& coefficients() const noexcept { return m_coefficients; }
    // The report of each family that loads the unit ("per load family and per unit on the condensed
    // path, the basis taken and its size"), in the order given.
    [[nodiscard]] const std::vector<assembly::LoadBasisReport>& reports() const noexcept { return m_reports; }
    // The stamp of what the patterns depend on (FNV-1a, StampHasher): for a loaded-DOF family its DOFs only, for the
    // other bases the compact values of the sets and cases taken.
    [[nodiscard]] std::int64_t patternStamp() const noexcept { return m_patternStamp; }

private:
    std::vector<UnitBasisVector> m_vectors; // bounded by the loaded DOFs and the cases of the unit
    std::vector<assembly::LoadBasisReport> m_reports;
    BasisCoefficients m_coefficients;
    std::int64_t m_patternStamp = 0;
};

} // namespace oos::reduction
