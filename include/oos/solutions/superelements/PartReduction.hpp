#pragma once

#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/reduction/condense/CraigBampton.hpp"
#include "oos/reduction/condense/ModalReduction.hpp"
#include "oos/reduction/condense/StaticCondensation.hpp"
#include "oos/reduction/condense/UnitElementIndex.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/EigenMethod.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"
#include "oos/solutions/superelements/SuperelementPart.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::solutions
{

// Two components of a fixed-interface mode tie for its sign (PartReduction: the component of largest magnitude made
// positive, the first one in the order of the part's DOFs on a tie) when the smaller is within this fraction of the
// larger: a symmetric part gives pairs of equal magnitudes that the round-off of the Lanczos solve separates by about
// 1e-14 of the value; 1e-8 keeps such a pair a tie, far below the difference of two distinct values of a mode.
inline constexpr double FIXED_INTERFACE_TIE_TOLERANCE = 1.0e-8;

// The reduction of one part superelement (BEGIN SUPER) onto its boundary (deck superelements are mapped onto the
// condensation engine): the part is one condensation unit of its own model (reduction::UnitSystem on
// every element and point of the part; its boundary points, those O_o_Reader's part_boundaries gives, the unit's
// boundary), reduced by the unit's static condensation,
//     K* = K_bb − K_bo·K_oo⁻¹·K_ob,    f*_s = f_b,s − K_bo·K_oo⁻¹·f_o,s    (one f* per subcase of the part),
// and, in a modal analysis, by Craig-Bampton (reduction::CraigBampton) on the modes of its q set: the
// scalar points of its GeneralizedDofSet (its generalised coordinates, which SuperelementConnection joins to scalar
// points of the residual structure) take the N lowest fixed-interface modes of the METHOD of its subcases, N the number
// of those points, and
//     K_CB = [K* 0; 0 K_qq],    M_CB = [M_bb M_bq; M_qb M_qq]    (ModalReduction);
// without a q set the part is reduced statically (Guyan: K* and M_bb).
//
// The rules of a part superelement in the format:
// - its constraints are those of its own Bulk Data that the subcases applying to it select (SPC, MPC, the PS of its
//   grid points) with the grid point singularities of its interior (AUTOSPC); they must be the same in every one of
//   those subcases (one reduction serves them all), and a boundary DOF held in the part is held in the residual
//   structure too (the permanent constraints are united, heldBoundaryLabels);
// - its loads are the loads of its own Bulk Data the subcases select, at every point (its boundary points included:
//   f_b), condensed onto the boundary;
// - the fixed-interface modes are normalised by the NORM of the part's method (MASS: unit generalised mass, the sign
//   making the component of largest magnitude positive, the first in the part's DOF order on a tie: the format fixes
//   no sign, so this rule makes the generalised coordinates reproducible; MAX: that component +1).
//
// What it refuses (errors of the sink): constraints or MPC sets that differ between the part's subcases, glued contact
// in a part, a boundary point that is no point of the part or whose DOF a relation of the part makes dependent, a q
// set point that is not a scalar point of the part or that an element uses, a q set larger than the modes its method
// gives (the rest would need residual vectors, which this version does not compute) or without a METHOD, a method
// whose range starts above 0.
//
// The recovery (recoverStatic, recoverModes) gives the part's g-set displacements from its boundary values (and its
// generalised coordinates): the interior by the unit's recovery u_o = K_oo⁻¹·(f_o − K_ob·u_b) (statics), or
// u_o = Ψ·u_b + Φ·q (Craig-Bampton), the q set points holding q.
//
// Thread-safety: const methods may run concurrently. Ownership: owns the part's analysis model and its reduction
// (budgeted: the budget must outlive it); borrows the part's snapshot, which must outlive it, and the kernels. Not
// movable (its members refer to each other).
class PartReduction
{
public:
    struct Inputs
    {
        const SuperelementPart* part = nullptr; // a part (snapshot not null)
        AnalysisKind analysis = AnalysisKind::Statics;
        // The labels of the subcases that apply to the part (SUPER), in the order of the Case Control.
        std::vector<base::EntityId> subcases;
        const linalg::DenseKernels* kernels = nullptr;
        base::ThreadPool* pool = nullptr;
        const base::CancellationToken* cancellation = nullptr;
    };

    // Reports the problems of the part to `sink` and stops on them (SolverError); throws as the analysis model
    // (UnsupportedCardError), the factorisation (SingularityError: the part is not held by its boundary),
    // MemoryBudgetError, OperationCancelledError.
    PartReduction(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    PartReduction(const PartReduction&) = delete;
    PartReduction& operator=(const PartReduction&) = delete;
    PartReduction(PartReduction&&) = delete;
    PartReduction& operator=(PartReduction&&) = delete;
    ~PartReduction();

    [[nodiscard]] std::int32_t superelementId() const noexcept { return m_superelementId; }
    [[nodiscard]] const AnalysisModel& model() const noexcept { return *m_model; }
    // The part's subcases (its view of the Case Control: the selections of its own Bulk Data), in the order given.
    [[nodiscard]] std::span<const StaticSubcase> subcases() const noexcept { return m_subcases; }

    // The rows of K* (the part's boundary DOFs, its own IDs) and K*, f* (boundary DOFs × subcases(): no column in a
    // modal analysis).
    [[nodiscard]] std::span<const reduction::UnitDofLabel> boundaryLabels() const noexcept;
    [[nodiscard]] const linalg::DenseMatrix& stiffness() const noexcept;
    [[nodiscard]] const linalg::DenseMatrix& loads() const noexcept;
    // The boundary DOFs of its boundary points held in the part (PS, SPC, AUTOSPC), which the residual structure holds
    // too.
    [[nodiscard]] std::span<const reduction::UnitDofLabel> heldBoundaryLabels() const noexcept
    {
        return m_heldBoundary;
    }

    // A modal analysis: the reduced boundary mass M_bb, and the q part (null without a q set) with its points.
    [[nodiscard]] const linalg::DenseMatrix& boundaryMass() const noexcept { return m_boundaryMass; }
    [[nodiscard]] const reduction::ModalReduction* modal() const noexcept { return m_modal.get(); }
    [[nodiscard]] std::span<const base::EntityId> modalPoints() const noexcept { return m_modalPoints; }

    [[nodiscard]] std::int64_t interiorDofCount() const noexcept;
    // The applied loads of subcase s (an index into subcases()) on the part's g set: OLOAD.
    [[nodiscard]] std::span<const double> appliedLoads(std::size_t s) const noexcept;

    // The part's g-set displacements (g × columns) of boundary values u_b (boundaryLabels() rows × columns) for the
    // subcases subcaseOfColumn (indices into subcases(), one per column): u_o = K_oo⁻¹·(f_o − K_ob·u_b) with f_o the
    // loads of the column's subcase, the held DOFs at their enforced values, the dependent DOFs by Gm. Throws
    // std::invalid_argument on a size mismatch.
    [[nodiscard]] linalg::DenseMatrix recoverStatic(linalg::ConstDenseMatrixView boundaryValues,
                                                    std::span<const std::size_t> subcaseOfColumn,
                                                    base::MemoryBudget& budget) const;
    // The part's g-set mode shapes of boundary values u_b and generalised coordinates q (modalPoints() rows; no row
    // without a q set): u_o = Ψ·u_b + Φ·q, the q set points holding q.
    [[nodiscard]] linalg::DenseMatrix recoverModes(linalg::ConstDenseMatrixView boundaryValues,
                                                   linalg::ConstDenseMatrixView modalValues,
                                                   base::MemoryBudget& budget) const;

private:
    void selectSubcases(const Inputs& inputs, base::DiagnosticSink& sink);
    void buildConstraints(base::MemoryBudget& budget, base::DiagnosticSink& sink);
    void describe(const SuperelementPart& part, base::DiagnosticSink& sink);
    void checkBoundaryRoles(base::DiagnosticSink& sink);
    void buildLoads(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    void reduceModes(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // The g-set vector (g × columns) of the unit's local vectors, with the q set points from `modalValues`.
    [[nodiscard]] linalg::DenseMatrix toGSet(linalg::ConstDenseMatrixView localValues,
                                             linalg::ConstDenseMatrixView modalValues,
                                             base::MemoryBudget& budget) const;

    std::int32_t m_superelementId = 0;
    const linalg::DenseKernels* m_kernels = nullptr;
    const base::CancellationToken* m_cancellation = nullptr;
    std::unique_ptr<AnalysisModel> m_model;
    std::vector<StaticSubcase> m_subcases;
    std::unique_ptr<assembly::MpcEquations> m_mpc;
    std::unique_ptr<assembly::FastenerConstraints> m_fasteners;
    std::unique_ptr<assembly::RigidElimination> m_relations;
    std::unique_ptr<reduction::UnitElementIndex> m_elementIndex;
    std::unique_ptr<assembly::NonStructuralMassShares> m_shares;
    base::BudgetedBuffer<std::uint8_t> m_isHeld; // per g-set DOF of the part: held in every subcase of the part
    reduction::UnitModel m_unitModel;
    reduction::UnitDescription m_description;
    std::vector<base::EntityId> m_modalPoints; // the q set, increasing
    std::vector<base::DofIndex> m_modalDofs;   // their g-set DOFs
    std::shared_ptr<const reduction::UnitSystem> m_system;
    linalg::FactorOptions m_factorOptions;
    std::unique_ptr<reduction::StaticCondensation> m_condensation;
    linalg::DenseMatrix m_patternLoads;    // local DOFs × subcases (statics)
    linalg::DenseMatrix m_patternEnforced; // local DOFs × subcases, or no column without enforced displacements
    linalg::DenseMatrix m_appliedLoads;    // g × subcases (statics): OLOAD
    linalg::DenseMatrix m_noLoads;         // boundary DOFs × 0: the loads of a modal analysis
    linalg::DenseMatrix m_boundaryMass;    // M_bb (modal analysis)
    linalg::DenseMatrix m_modes;           // Φ: interior equations × N (Craig-Bampton), normalised
    std::unique_ptr<reduction::ModalReduction> m_modal;
    std::vector<reduction::UnitDofLabel> m_heldBoundary;
};

} // namespace oos::solutions
