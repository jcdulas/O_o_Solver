#pragma once

#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElementEquations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/RigidTable.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace oos::assembly
{

// Where a row of the constraints comes from, for a filter of the rows an elimination takes (the condensation of a unit
// takes the relations eliminated inside units only).
struct ConstraintRowOrigin
{
    ConstraintSource source = ConstraintSource::RigidBodyStar;
    // The element ID of a rigid element, a Fastener or a Weld; the SID of the MPC card of an MPC equation
    // (MpcEquations::sourceSetIdOf); the ID of the contact table of a glue row.
    base::EntityId id = 0;
    base::DofIndex dependent = 0;
};

// The force one relation pushes on one DOF (RigidElimination::relationForces): the DOF, the relation
// (RigidElimination::relationOf) and the force.
struct RelationForce
{
    base::DofIndex dof = 0;
    std::int32_t relation = 0;
    double force = 0.0;
};

// Whether an elimination takes a row (true) or leaves it out as if its element or equation were not in the model.
using ConstraintRowFilter = std::function<bool(const ConstraintRowOrigin&)>;

// The elimination of the dependent DOFs of the rigid elements and of the MPC equations of a subcase (linear
// elimination, the default of the format): the m set and its matrix Gm,
//     u_m = Gm·u_n,
// with every row of Gm on n-set DOFs only. Each element or equation gives its dependent DOFs as combinations of its
// other DOFs (RigidElementEquations, MpcEquations); a term on a dependent DOF of another element (an RigidBodyStar on
// the dependent point of an RigidBar, an MPC on the reference point of an RigidInterpolation) is replaced by that DOF's
// row, in dependency order, so that chains of any length resolve to the n set.
//
// Problems are Diagnostics, and the DOFs involved are left out of the m set: a DOF made dependent by two elements or
// equations ("dof-dependent-twice", naming both; the first in table order keeps it, rigid elements before MPCs), a
// cycle of dependencies ("dependency-cycle", naming the elements of the cycle; every row of the cycle is dropped), and
// the element-level problems of RigidElementEquations and MpcEquations. An analysis with an error diagnostic stops, so
// the remaining rows only need to be consistent, not meaningful.
//
// What the static solution does with it: Kgg is reduced to Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm, the loads to
// Pn = P̄n + Gmᵀ·Pm (reduceToN), the solved u_n is expanded with expandToG, and the MPC forces are computed from the
// element forces Kgg·ug and the loads (computeMpcForces).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns Gm (budgeted); the budget must outlive it.
// Movable, not copyable.
class RigidElimination
{
public:
    // `rigidElements` may be null (no rigidElementsTable), `mpc` null (no MPC set selected), `fasteners` null (no
    // fastener whose GA or GB is a grid point of the model), `glue` null (no glued contact). The rows come in this
    // order: rigid elements, fasteners, MPC equations, glue (the first of two rows making one DOF dependent keeps it;
    // GlueConstraints leaves out the points the others make dependent, so its rows never meet them).
    // `filter`, when given, is asked for every row before the DOFs are checked and the chains resolved: a row it
    // refuses is not taken (the element diagnostics are reported whatever it says).
    RigidElimination(const input::RigidTable* rigidElements, const MpcEquations* mpc, const DofMap& dofMap,
                     base::MemoryBudget& budget, base::DiagnosticSink& sink,
                     const FastenerConstraints* fasteners = nullptr, const GlueConstraints* glue = nullptr,
                     const ConstraintRowFilter& filter = {});

    // The m-set DOFs, increasing.
    [[nodiscard]] std::span<const base::DofIndex> mDofs() const noexcept { return m_mDofs.span(); }
    [[nodiscard]] std::size_t rowCount() const noexcept { return m_mDofs.size(); }
    // Row i of Gm (for mDofs()[i]): its terms on n-set DOFs, by increasing DOF.
    [[nodiscard]] std::span<const ConstraintTerm> row(std::size_t i) const noexcept
    {
        return {m_terms.data() + m_offsets[i], static_cast<std::size_t>(m_offsets[i + 1] - m_offsets[i])};
    }
    // The row of a DOF of the m set, or nothing for an n-set DOF.
    [[nodiscard]] std::optional<std::size_t> findRow(base::DofIndex dof) const noexcept
    {
        const std::int32_t found = m_rowOfDof.empty() ? -1 : m_rowOfDof[static_cast<std::size_t>(dof)];
        return found < 0 ? std::nullopt : std::optional<std::size_t>(static_cast<std::size_t>(found));
    }
    [[nodiscard]] bool isDependent(base::DofIndex dof) const noexcept { return findRow(dof).has_value(); }
    // Where row i comes from: the card and its ID (the set ID for an MPC).
    [[nodiscard]] ConstraintSource sourceOf(std::size_t i) const noexcept
    {
        return static_cast<ConstraintSource>(m_sources[i]);
    }
    [[nodiscard]] base::EntityId sourceIdOf(std::size_t i) const noexcept { return m_sourceIds[i]; }

    // u_m = Gm·u_n on a g-set vector: the m entries of `displacementsG` are overwritten from its n entries.
    void expandToG(std::span<double> displacementsG) const noexcept;
    // P_n += Gmᵀ·P_m, then P_m = 0, on a g-set vector (the loads applied to dependent DOFs moved to the n set, by
    // virtual work: δu_mᵀ·P_m = δu_nᵀ·Gmᵀ·P_m).
    void reduceToN(std::span<double> loadsG) const noexcept;
    // The forces of multipoint constraint (MPCFORCES) on the g set: the forces the MPCs and rigid elements apply to the
    // grid points, such that Kgg·ug = Pg + q_mpc + q_spc at every DOF. On the m set q_m = (Kgg·ug − Pg)_m (no SPC
    // acts there, the sets being exclusive); on the n set q_n = −Gmᵀ·q_m (the constraint forces do no work on any
    // motion the constraints allow: δu_nᵀ·q_n + (Gm·δu_n)ᵀ·q_m = 0); zero elsewhere. `elementForcesG` is Kgg·ug with
    // the expanded displacements, `loadsG` the applied loads Pg before reduceToN. The SPC forces are then
    // q_s = (Kgg·ug − Pg − q_mpc)_s.
    void computeMpcForces(std::span<const double> elementForcesG, std::span<const double> loadsG,
                          std::span<double> mpcForcesG) const noexcept;

    // The relation of row i as the split multiPointConstrForce rows of a grid point force table name it (O_o_Reader's
    // data contract): the element ID of a rigid element (RBE1, RigidBodyStar, RigidInterpolation, RigidBar, RigidRod),
    // minus the SID of the MPC card of an MPC equation (the equations of one card set are one relation at a grid
    // point), 0 for the other rows (the internal constraints of a Fastener or Weld, the glue of a contact pair:
    // no relation a cut counts).
    [[nodiscard]] std::int32_t relationOf(std::size_t i) const noexcept { return m_relationIds[i]; }

    // The forces of each relation on the DOFs it acts at: with λ_i the multiplier of row i, the
    // relation of row i pushes λ_i·(e_{m_i} − Σ_j c_ij·e_j) on the g set (c_ij its own terms, before the chains are
    // resolved), so that the sum over the rows is q_mpc (computeMpcForces). Without a chain λ_i = q_m(i) and the
    // terms are row(i)'s; through a chain (a term of row ρ on the m DOF d of row i) the total force at d is
    // λ_d − Σ_ρ c_ρd·λ_ρ, so λ_d = q_m(d) + Σ_ρ c_ρd·λ_ρ, the parents of a chain first. `mpcForcesG` is
    // computeMpcForces' result; visit(row, dof, force) is called for every DOF of every row, a zero force included
    // (the points a relation acts at are those of its terms): the rows in order, each row's own DOF first, then its
    // terms in order (a fixed order of every sum).
    template <typename Visit>
    void forEachRelationForce(std::span<const double> mpcForcesG, base::MemoryBudget& budget, Visit visit) const
    {
        base::BudgetedBuffer<double> multipliers(budget, m_mDofs.size(), "MPC relation multipliers");
        for (std::size_t i = 0; i < m_mDofs.size(); ++i)
            multipliers[i] = mpcForcesG[static_cast<std::size_t>(m_mDofs[i])];
        for (std::size_t k = 0; k < m_chainOrder.size(); ++k)
        {
            const auto parent = static_cast<std::size_t>(m_chainOrder[k]);
            for (const ConstraintTerm& term : relationTerms(parent))
                if (const std::optional<std::size_t> child = findRow(term.dof))
                    multipliers[*child] += term.coefficient * multipliers[parent];
        }
        for (std::size_t i = 0; i < m_mDofs.size(); ++i)
        {
            visit(i, m_mDofs[i], multipliers[i]);
            for (const ConstraintTerm& term : relationTerms(i))
                visit(i, term.dof, -term.coefficient * multipliers[i]);
        }
    }
    // forEachRelationForce as a list, in its order (one entry per DOF of every row: as many as Gm's terms and rows).
    [[nodiscard]] base::BudgetedBuffer<RelationForce> relationForces(std::span<const double> mpcForcesG,
                                                                     base::MemoryBudget& budget) const;
    // The terms of row i as its element or equation gives them, before the chains are resolved (row(i) when the
    // elimination has no chain).
    [[nodiscard]] std::span<const ConstraintTerm> relationTerms(std::size_t i) const noexcept
    {
        if (m_directOffsets.size() == 0)
            return row(i);
        return {m_directTerms.data() + m_directOffsets[i],
                static_cast<std::size_t>(m_directOffsets[i + 1] - m_directOffsets[i])};
    }
    // Whether a row's own terms reach a DOF another row makes dependent (a chain, resolved in Gm).
    [[nodiscard]] bool hasChains() const noexcept { return m_directOffsets.size() != 0; }
    // With chains, the rows of Gm, parents (a row whose own term is another row's m DOF) before their children; empty
    // without.
    [[nodiscard]] std::span<const std::int64_t> chainOrder() const noexcept { return m_chainOrder.span(); }

private:
    base::BudgetedBuffer<base::DofIndex> m_mDofs;
    base::BudgetedBuffer<std::int64_t> m_offsets; // rowCount() + 1
    base::BudgetedBuffer<ConstraintTerm> m_terms;
    base::BudgetedBuffer<std::uint8_t> m_sources;     // ConstraintSource of each row
    base::BudgetedBuffer<base::EntityId> m_sourceIds; // elementId, or the MPC set ID
    base::BudgetedBuffer<std::int32_t> m_rowOfDof;    // one per g-set DOF, -1 for the n set
    base::BudgetedBuffer<std::int32_t> m_relationIds; // relationOf() of each row
    // With chains only (empty otherwise): each row's own terms (relationTerms) and chainOrder().
    base::BudgetedBuffer<std::int64_t> m_directOffsets;
    base::BudgetedBuffer<ConstraintTerm> m_directTerms;
    base::BudgetedBuffer<std::int64_t> m_chainOrder;
};

} // namespace oos::assembly
