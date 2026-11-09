#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/ConstraintTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::assembly
{

// A term of a constraint equation or of a row of Gm: a coefficient on a g-set DOF.
struct ConstraintTerm
{
    base::DofIndex dof = 0;
    double coefficient = 0.0;
};

// The MPC equations of one MPC set (MPC: Σ_j A_j·u_j = 0, the first term's DOF dependent, put in the m set), from
// the multiPointConstrTable of O_o_Reader (MPCADD resolved: the rows of an MultiPointConstrUnion set carry its ID in
// `setId`; one row per term, the equations in the order of the cards). A term on a scalar point has component 0.
//
// An equation with a problem is dropped and reported: a point the model does not define ("mpc-unknown-point"), a
// component the point does not have ("mpc-bad-component"), a dependent coefficient of zero
// ("mpc-zero-dependent-coefficient"), the dependent DOF repeated among its own terms ("mpc-dependent-repeated"). The
// diagnostics name the card MPC, the set ID and, in the field, the equation number within the set.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its terms (budgeted); the budget must outlive
// it. Movable, not copyable.
//
// The MPC card of each equation: the equations of an MultiPointConstrUnion set are those of the MPC
// cards whose SIDs it lists. multiPointConstrTable names the card's own SID of each row in an optional `source_sid`
// column (asked of O_o_Reader, as singlePointConstrTable and loadsTable have it); without it the selected set stands
// for the card (sourceSetIdOf() is setId()), which is exact for a set of MPC cards and keys an MultiPointConstrUnion by
// its own ID.
class MpcEquations
{
public:
    // The equations of the set `mpcSetId` (none when it is absent or `constraints` is null).
    MpcEquations(const input::ConstraintTable* constraints, const DofMap& dofMap,
                 std::optional<base::EntityId> mpcSetId, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // The equations of the set `mpcSetId` whose MPC card is one of `keptSourceSetIds` (sorted): the part of an
    // MultiPointConstrUnion that every subcase of a condensation plan selects. Without the source_sid column, the set
    // is kept whole when its own ID is listed. Throws std::invalid_argument when keptSourceSetIds is not sorted.
    MpcEquations(const input::ConstraintTable* constraints, const DofMap& dofMap,
                 std::optional<base::EntityId> mpcSetId, std::span<const base::EntityId> keptSourceSetIds,
                 base::MemoryBudget& budget, base::DiagnosticSink& sink);

    // Whether multiPointConstrTable gives the SID of the MPC card of every row (its optional source_sid column).
    [[nodiscard]] static bool hasSourceSetIds(const input::ConstraintTable* constraints);
    // The SIDs of the MPC cards whose equations the set `mpcSetId` holds, sorted and distinct: the members of an
    // MultiPointConstrUnion, or the set itself; {mpcSetId} without the source_sid column; empty when `constraints` is
    // null.
    [[nodiscard]] static std::vector<base::EntityId> sourceSetIdsOf(const input::ConstraintTable* constraints,
                                                                    base::EntityId mpcSetId);

    [[nodiscard]] std::optional<base::EntityId> setId() const noexcept { return m_setId; }
    [[nodiscard]] std::size_t size() const noexcept { return m_count; }
    // The terms of equation i; term 0 is the dependent one.
    [[nodiscard]] std::span<const ConstraintTerm> terms(std::size_t i) const noexcept
    {
        return {m_terms.data() + m_offsets[i], static_cast<std::size_t>(m_offsets[i + 1] - m_offsets[i])};
    }
    [[nodiscard]] base::DofIndex dependentDof(std::size_t i) const noexcept { return terms(i)[0].dof; }
    // The position of equation i among the equations of its set (from 1), as multiPointConstrTable numbers them.
    [[nodiscard]] std::int32_t equationNumber(std::size_t i) const noexcept { return m_equationNumbers[i]; }
    // The SID of the MPC card equation i comes from (source_sid), or setId() when multiPointConstrTable does not give
    // it.
    [[nodiscard]] base::EntityId sourceSetIdOf(std::size_t i) const noexcept { return m_sourceSetIds[i]; }

private:
    // Reads the equations of the set; keptSourceSetIds null keeps every equation.
    void readEquations(const input::ConstraintTable& constraints, const DofMap& dofMap,
                       const std::span<const base::EntityId>* keptSourceSetIds, base::MemoryBudget& budget,
                       base::DiagnosticSink& sink);

    std::optional<base::EntityId> m_setId;
    std::size_t m_count = 0;
    // Sized for every equation of the set; the first m_count are the accepted ones.
    base::BudgetedBuffer<std::int64_t> m_offsets; // m_count + 1 used
    base::BudgetedBuffer<ConstraintTerm> m_terms;
    base::BudgetedBuffer<std::int32_t> m_equationNumbers;
    base::BudgetedBuffer<base::EntityId> m_sourceSetIds;
};

} // namespace oos::assembly
