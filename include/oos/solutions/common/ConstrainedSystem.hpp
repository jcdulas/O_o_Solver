#pragma once

#include "oos/assembly/assemble/ElementMatrixStore.hpp"
#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/assembly/singularities/NodeDiagonalBlockSet.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/AnalysisTimings.hpp"
#include "oos/solutions/common/StructureCache.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::solutions
{

// The constraints of a subcase (Case Control SPC, MPC, the EnforcedMotionConstr of its LOAD set, and the glued contact
// of its BCONTACT).
struct ConstraintSelection
{
    base::EntityId subcase = 1; // its ID for the messages and the PARAMs of its Case Control
    std::optional<base::EntityId> spc;
    std::optional<base::EntityId> mpc;
    std::optional<base::EntityId> load;
    std::optional<base::EntityId> contact;
};

// What the static solution adds to a constraint set (the eigenvalue solutions take the defaults).
struct ConstrainedSystemOptions
{
    // The SpcSet of the selection when the caller built it already (the static solution builds those of every
    // subcase to group them): borrowed, it must outlive the system; null: built from the selection's SPC and LOAD.
    const assembly::SpcSet* spc = nullptr;
    // The r set of FreeBodySupport (SUPORT, SUPORT1), out of the l set.
    std::vector<base::DofIndex> supportDofs;
    // PARAM INREL = −2: the supports are chosen among the free DOFs left by the grid point singularity processor
    // (assembly::chooseAutomaticSupports on the rigid-body modes about referencePoint); supportDofs is then ignored.
    bool choosesSupports = false;
    base::Vec3 referencePoint{};
    // The element stiffness matrices the connectivity pass kept (AnalysisModel::keptStiffness()), read by the assembly
    // of Kgg and released right after it when releasesKeptStiffness (the last constraint set of a run): before Knn is
    // allocated, so that the peak of the budget does not hold both.
    assembly::ElementMatrixStore* keptStiffness = nullptr;
    bool releasesKeptStiffness = false;
    // The external units of the analysis (solutions/sol101/ExternalUnitPath): the interior DOFs of their zones, held as
    // permanent constraints (sorted; nothing acts on them once their elements are replaced), the node groups their
    // dense K* couples (one per unit: its contour nodes), and the callable that adds their K* to Kgg once the elements
    // are assembled, before the m set is eliminated. Empty: none.
    std::span<const base::DofIndex> heldDofs;
    linalg::NodeGroupList extraCouplings;
    std::function<void(linalg::BlockSparseMatrix&)> addStiffness;
    // DOFs the grid point singularity processor leaves free whatever their stiffness, sorted: the generalised
    // coordinates of the superelements of a deck (the q set), whose stiffness K_qq (the eigenvalues of their
    // fixed-interface modes) may lie far below the largest diagonal term without being singular. Empty: none.
    std::span<const base::DofIndex> uncheckedDofs;
    // The structure kept between the analyses of the model: the sparse pattern taken from it when an earlier analysis
    // built it from the same node graph, else built and kept there; its bytes reserved in the system's budget while the
    // pattern is held. Null: built for this system alone. Not owned.
    StructureCache* structures = nullptr;
    // The full path streamed: no matrix assembled (no pattern, Kgg nor Knn: kgg(), knn(), pattern() and reduce() are
    // not to be called), the diagonal blocks of Knn the grid point singularity processor reads given by this callable
    // instead, from Gm and the sets of the constraints (summed element by element). Null: the matrices assembled.
    std::function<assembly::NodeDiagonalBlockSet(const assembly::RigidElimination&, const assembly::DofSets&)>
        diagonalBlocks;
};

// The stiffness of one constraint set (the steps of a solution that come before its factorisation,
// src/solutions/README.md): Gm of the MPCs, rigid elements, fasteners and glued contact, the DOF sets (SPC, GRID PS,
// the r set of FreeBodySupport), the node graph and pattern with the m-set couplings, Kgg, Knn = the m set eliminated
// (MsetElimination), the grid point singularities constrained by AUTOSPC (PARAM AUTOSPC, EPZERO, PRGPST), and the
// solved sets. reduce() brings any other g-set matrix assembled on the pattern (the mass, the differential stiffness)
// to the n set the same way.
//
// The eigenvalue solutions give no r set: in SOL 103 the rigid-body modes of a free structure are found by the
// Lanczos solver from the free stiffness (a warning says so); in SOL 105 the stiffness must be positive definite. The
// static solution gives the r set of FreeBodySupport, or asks for the supports of PARAM INREL = −2.
//
// Thread-safety: immutable once built. Ownership: owns its parts (budgeted); Gm, the solved sets, the pattern and Kgg
// are shared with what outlives the system (a FactorisedSet kept by the static solution); borrows the model, which
// must outlive it. Not movable.
class ConstrainedSystem
{
public:
    // Throws SolverError listing the problems of the constraints (stopOnErrors), MemoryBudgetError,
    // OperationCancelledError.
    ConstrainedSystem(const AnalysisModel& model, const ConstraintSelection& selection, base::MemoryBudget& budget,
                      base::ThreadPool* pool, const base::CancellationToken* cancellation, base::DiagnosticSink& sink,
                      const ConstrainedSystemOptions& options = {});
    ConstrainedSystem(const ConstrainedSystem&) = delete;
    ConstrainedSystem& operator=(const ConstrainedSystem&) = delete;
    ConstrainedSystem(ConstrainedSystem&&) = delete;
    ConstrainedSystem& operator=(ConstrainedSystem&&) = delete;
    ~ConstrainedSystem() = default;

    [[nodiscard]] const assembly::RigidElimination& gm() const noexcept { return *m_gm; }
    [[nodiscard]] const assembly::SpcSet& spc() const noexcept { return *m_spc; }
    // The glued contact of the selection (its pairs and what it tied; no pair when the subcase selects none).
    [[nodiscard]] const assembly::GlueConstraints& glue() const noexcept { return *m_glue; }
    // The sets solved: those of the constraints with the AUTOSPC DOFs constrained.
    [[nodiscard]] const assembly::DofSets& solveSets() const noexcept { return *m_solveSets; }
    // The r set: the FreeBodySupport DOFs, or the supports chosen for PARAM INREL = −2.
    [[nodiscard]] const std::vector<base::DofIndex>& supportDofs() const noexcept { return m_supportDofs; }
    [[nodiscard]] const linalg::NodeGraph& graph() const noexcept { return *m_graph; }
    [[nodiscard]] const linalg::SparsePattern& pattern() const noexcept { return *m_pattern; }
    [[nodiscard]] const linalg::BlockSparseMatrix& kgg() const noexcept { return *m_kgg; }
    [[nodiscard]] const linalg::BlockSparseMatrix& knn() const noexcept { return *m_knn; }
    // The table of the singular components as text (empty when nothing was constrained or listed).
    [[nodiscard]] const std::string& singularityTable() const noexcept { return m_singularityTable; }
    // One flag per g-set DOF: held (the s set, AUTOSPC included, and the r set): its SPC force is a result.
    [[nodiscard]] std::span<const std::uint8_t> isConstrained() const noexcept { return m_isConstrained.span(); }
    // The time of its parts (AnalysisPartTimings::constraints, graph, stiffness, singularities).
    [[nodiscard]] const AnalysisPartTimings& timings() const noexcept { return m_timings; }
    // Whether its pattern came from the structure cache of its options, built by an earlier analysis.
    [[nodiscard]] bool isPatternReused() const noexcept { return m_isPatternReused; }

    // The parts a FactorisedSet keeps (Kgg refers to the pattern).
    [[nodiscard]] std::shared_ptr<const assembly::RigidElimination> sharedGm() const noexcept { return m_gm; }
    [[nodiscard]] std::shared_ptr<const assembly::DofSets> sharedSolveSets() const noexcept { return m_solveSets; }
    [[nodiscard]] std::shared_ptr<const linalg::SparsePattern> sharedPattern() const noexcept { return m_pattern; }
    [[nodiscard]] std::shared_ptr<const linalg::BlockSparseMatrix> sharedKgg() const noexcept { return m_kgg; }

    // The n-set matrix of a g-set matrix assembled on pattern() (MsetElimination::reduce).
    [[nodiscard]] linalg::BlockSparseMatrix reduce(const linalg::BlockSparseMatrix& gg, base::MemoryBudget& budget,
                                                   const char* label) const;

private:
    const AnalysisModel* m_model;
    std::unique_ptr<assembly::MpcEquations> m_mpc;
    std::unique_ptr<assembly::FastenerConstraints> m_fasteners;
    std::unique_ptr<assembly::GlueConstraints> m_glue;
    std::shared_ptr<assembly::RigidElimination> m_gm;
    std::unique_ptr<assembly::SpcSet> m_ownSpc;
    const assembly::SpcSet* m_spc = nullptr;     // m_ownSpc, or the caller's
    std::vector<base::DofIndex> m_permanentDofs; // the sg DOFs with the external units' held DOFs (empty: sgDofs())
    std::vector<base::DofIndex> m_supportDofs;
    std::unique_ptr<assembly::DofSets> m_sets;
    std::shared_ptr<assembly::DofSets> m_solveSets;
    std::unique_ptr<linalg::NodeGraph> m_graph;
    std::shared_ptr<const linalg::SparsePattern> m_pattern;
    std::shared_ptr<linalg::BlockSparseMatrix> m_kgg;
    std::unique_ptr<linalg::BlockSparseMatrix> m_knn;
    std::string m_singularityTable;
    base::BudgetedBuffer<std::uint8_t> m_isConstrained;
    AnalysisPartTimings m_timings;
    bool m_isPatternReused = false;
};

} // namespace oos::solutions
