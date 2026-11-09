#pragma once

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/singularities/NodeDiagonalBlockSet.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/sol101/StreamedElements.hpp"
#include "oos/solutions/sol101/StreamedLoads.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::solutions
{

// The front source of the full path streamed (linalg::FrontSource): each front takes the elements
// placed at its supernode (StreamedElements), their stiffness made then (the family's stiffness facet, a run of
// consecutive rows at a time), carried onto the equations (StreamedReduction: K_red = Tᵀ·K·T), and the loads of every
// basis vector (StreamedLoads): the elements' pressures, their mass times each vector's GravityLoad acceleration less
// its rigid relief, the point loads placed there. Each term is added in a fixed order (the elements increasing, each
// element's terms in its DOFs' order), so a front's terms do not depend on the threads: the streamed solve gives the
// same bits whatever they are.
//
// The diagonal of the matrix (the MAXRATIO ratio of the pivot diagnostics) is the n-set diagonal the singularity
// pass summed (nodeDiagonals), held in single precision: the ratio is compared with MAXRATIO and reported, its
// digits beyond the seventh are not read.
//
// Thread-safety: addTerms() and diagonal() are safe concurrently (each call has its own scratch). Ownership: borrows
// everything it is given, which outlives it.
class StreamedFront final : public linalg::FrontSource
{
public:
    struct Inputs
    {
        const AnalysisModel* model = nullptr;
        const StreamedElements* elements = nullptr;
        const StreamedLoads* loads = nullptr;
        const assembly::LoadVectorBuilder* loadBuilder = nullptr;
        assembly::MassAssemblyOptions massOptions;
        const assembly::NonStructuralMassShares* shares = nullptr;
        std::span<const float> diagonalOfDof; // per g-set DOF: Knn's diagonal (nodeDiagonals)
        // Inertia relief (the full path's D = Φ·C, zero on the held DOFs, Gm on the dependent ones): Gm and the sets
        // whose s set D is zero on (the solved sets before the conditioned supports). Null without relief.
        const assembly::RigidElimination* gm = nullptr;
        const assembly::DofSets* reliefSets = nullptr;
        // The refinement of a relieved solution: equations × columns subtracted from the
        // right-hand sides of the equations at their own supernodes (K·û, so that the solve gives the correction of
        // the residual P'_l − K·û). Empty: none.
        linalg::ConstDenseMatrixView subtractedLoads;
    };

    explicit StreamedFront(const Inputs& inputs);

    void addTerms(linalg::FrontTerms& front) override;
    void diagonal(std::int32_t supernode, std::span<double> values) const override;

    // What inertia relief on rigid support motions sums over the model before the solve: Φᵀ·M·Φ and the
    // resultants Φᵀ·P of every vector, P on the g set (the element loads and the point loads; RigidBodyModes'
    // convention about the basic origin), element by element in their order.
    struct ReliefSums
    {
        base::MatN<6> rigidBodyMass{};
        std::vector<std::array<double, 6>> resultants; // per vector
    };
    // PERFORMANCE: the elements in runs of a fixed size, each run's sums on the pool (when given) and added in run
    // order: the same values whatever the threads; each element's support motions made once, its loads summed onto them
    // before their factors.
    // Measured: the profile of 6 000 cases in 300 families at the section's 1/16, free, 30.5 s on one thread.
    [[nodiscard]] ReliefSums reliefSums(base::ThreadPool* pool, base::MemoryBudget& budget,
                                        const base::CancellationToken* cancellation) const;
    // The loads of every vector at the g-set DOFs `dofs` (increasing), as the solve takes them (P' = P − M·Φ·w with
    // inertia relief): the pressure loads and the mass loads of the elements with a DOF there, the concentrated loads
    // there; dofs × vectors (the appliedLoad of the fixed rows of a load basis, and its loads at the
    // relations' dependent DOFs). Throws MemoryBudgetError.
    [[nodiscard]] linalg::DenseMatrix loadsAt(std::span<const base::DofIndex> dofs, base::MemoryBudget& budget) const;

    // The row of the support motions of inertia relief at a g-set DOF before C (D = Φ·C: Φ on a free independent
    // DOF, zero on a held one, Gm on a dependent one), the gauge StreamedOutputs removes.
    [[nodiscard]] std::array<double, 6> reliefRowOf(base::DofIndex dof) const;

    // The refinement of a relieved solution (linalg::IterativeRefinement), element by element (their
    // mass or stiffness made again, in pieces of a fixed size on the pool, each piece's terms added in element order
    // after each wave: the same values whatever the threads). `solution`: equations × columns, the held DOFs at rest.
    // Φᵀ·M·u of each column (RigidBodyModes' convention about the basic origin, the rows of reliefRowOf; WTMASS
    // applied, as reliefSums' Φᵀ·M·Φ).
    [[nodiscard]] std::vector<std::array<double, 6>> rigidMotionSums(linalg::ConstDenseMatrixView solution,
                                                                     base::ThreadPool* pool,
                                                                     const base::CancellationToken* cancellation) const;
    // products ← K·û on the equations (equations × columns), û = u − Φ·a the solution in the gauge of the amplitudes
    // a of each column: each element's K_e·(T·u − Φ_e·a) carried back by Tᵀ (the relative motion across a stiff
    // element from the displacements of its own points, StreamedReduction's T).
    void multiplyStiffness(linalg::ConstDenseMatrixView solution, std::span<const std::array<double, 6>> amplitudes,
                           linalg::DenseMatrixView products, base::ThreadPool* pool,
                           const base::CancellationToken* cancellation) const;
    // An upper bound of the work memory of those two passes (a wave of pieces' terms).
    [[nodiscard]] static std::uint64_t refinementWorkBytes(std::size_t threads, std::size_t columns) noexcept;

    // The n-set diagonal blocks of the grid point singularity processor (ConstrainedSystemOptions::diagonalBlocks),
    // summed element by element (the stiffness of each element carried onto the n set through Gm, Knn's terms), and
    // the diagonal of every g-set DOF in single precision (zero for a dependent DOF). With a pool, the elements' blocks
    // are made in pieces of a fixed size on it and summed in element order: the same values whatever the threads.
    // `largestRigidForce`: the largest |K_e·Φ| of an element's rigid-body motions on the n set (free everywhere): the
    // round-off of a free element's matrix, the stiffness of a grounded element (a free body is solved on
    // rigid support motions only when no element resists them, inertia relief's RIGID_SUPPORT_TOLERANCE).
    struct NodeDiagonals
    {
        assembly::NodeDiagonalBlockSet blocks;
        base::BudgetedBuffer<float> diagonalOfDof;
        double largestRigidForce = 0.0;
    };
    [[nodiscard]] static NodeDiagonals nodeDiagonals(const AnalysisModel& model, const assembly::RigidElimination& gm,
                                                     base::MemoryBudget& budget, base::ThreadPool* pool,
                                                     const base::CancellationToken* cancellation);

private:
    Inputs m_inputs;
};

} // namespace oos::solutions
