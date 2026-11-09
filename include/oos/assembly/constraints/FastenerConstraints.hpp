#pragma once

#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElementEquations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/FastenerPiercingTable.hpp"

#include <cstdint>
#include <span>

namespace oos::assembly
{

// The internal constraints of the Fastener and Weld whose GA or GB is a grid point of the model (the
// internal constraints of Fastener/Weld are eliminated by default, the dependent DOFs in the m set through
// Gm). The fastener elements condense their auxiliary points into their own matrices (elements/fasteners/
// FastenerConnection: the elimination done at element level, which is all a generated GA or GB needs); a GA or GB
// that the model defines also has DOFs of its own, which these rows make dependent on the grid points of the pierced
// element by the same interpolation, so that the point is recovered (its displacements follow the pierced element)
// and whatever else is attached to it (a mass, a load, another element) acts on the pierced element:
//     u_GA = R_GAᵀ·(u_P + θ_P × (x_GA − x_P)),   θ_GA = R_GAᵀ·θ_P,   (u_P, θ_P) = M·u_grids,
// M the interpolation of the piercing point P of the side from the grid points of the element it pierces (the
// RigidInterpolation of elements::interpolatePiercing, through elements::interpolateSide), u_grids their DOFs, R_GA the
// rotation of the displacement system of GA. The rows are handed to RigidElimination, which resolves them with the
// rigid elements and MPCs (a grid point of the pierced element that is itself dependent is replaced by its row). A grid
// point that several fasteners name (the middle point of a stack of three sheets) is constrained once, by the first
// side that names it in table order.
//
// Problems are Diagnostics on the card and ID of the fastener (field GA or GB, IDA or IDB for the pierced element): a
// grid point of the table that the model does not define, shape functions that are not an interpolation.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its rows (budgeted); the budget must outlive it.
class FastenerConstraints
{
public:
    // `piercings` may be null (no fastener piercing table: no rows).
    FastenerConstraints(const input::FastenerPiercingTable* piercings, const DofMap& dofMap, base::MemoryBudget& budget,
                        base::DiagnosticSink& sink);

    [[nodiscard]] std::size_t rowCount() const noexcept { return m_rowCount; }
    [[nodiscard]] std::int64_t termCount() const noexcept { return m_termCount; }
    // Row i: its dependent DOF, its terms on the DOFs of the pierced element's grid points, its fastener card and ID.
    [[nodiscard]] base::DofIndex dependentDof(std::size_t i) const noexcept { return m_dependents[i]; }
    [[nodiscard]] std::span<const ConstraintTerm> row(std::size_t i) const noexcept
    {
        return {m_terms.data() + m_offsets[i], static_cast<std::size_t>(m_offsets[i + 1] - m_offsets[i])};
    }
    [[nodiscard]] ConstraintSource sourceOf(std::size_t i) const noexcept
    {
        return static_cast<ConstraintSource>(m_sources[i]);
    }
    [[nodiscard]] base::EntityId sourceIdOf(std::size_t i) const noexcept { return m_sourceIds[i]; }

private:
    base::BudgetedBuffer<base::DofIndex> m_dependents;
    base::BudgetedBuffer<std::int64_t> m_offsets;
    base::BudgetedBuffer<ConstraintTerm> m_terms;
    base::BudgetedBuffer<std::uint8_t> m_sources;
    base::BudgetedBuffer<base::EntityId> m_sourceIds;
    std::size_t m_rowCount = 0;
    std::int64_t m_termCount = 0;
};

} // namespace oos::assembly
