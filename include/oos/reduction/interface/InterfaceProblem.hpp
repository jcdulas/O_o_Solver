#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"
#include "oos/linalg/sparse/SparseAssembler.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/reduction/absorption/SpiderChangeOfBasis.hpp"
#include "oos/reduction/condense/LocalRelations.hpp"
#include "oos/reduction/condense/ModalReduction.hpp"
#include "oos/reduction/condense/ReducedUnit.hpp"
#include "oos/reduction/condense/UnitElementIndex.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace oos::reduction
{

// A point of the interface problem: a point of the model, or the point generated on the absorbed side of a Fastener
// or Weld (six DOFs in the basic system at the piercing point: a fastener side's generated GA/GB stands as a
// virtual boundary point).
struct InterfacePoint
{
    std::int32_t modelNode = -1; // the node of assembly::DofMap; -1 for a generated fastener point
    UnitDofLabel label;          // its name (component 1): a grid or scalar point, or the fastener and side
    std::uint8_t dofCount = 6;
    base::Vec3 position{}; // in the basic system (a scalar point takes the position of a grid point it is coupled to)
};

// A condensed unit as the interface problem takes it: its reduced unit (K* on its labels) and its equations (the raw
// blocks of its boundary points for the singularity check).
//
// In an eigenvalue analysis (InterfaceInputs::second), a Craig-Bampton unit also brings its q set: `modal`
// (its N modal coordinates, whose stiffness K_qq the interface adds on them; null: none), and `second`, its part of the
// second matrix of the pencil on its boundary DOFs then its modal coordinates ((a + N) square: the Craig-Bampton mass
// M_CB of a vibration analysis, the reduced differential stiffness of a buckling analysis; null: none).
struct InterfaceUnit
{
    const ReducedUnit* reduced = nullptr;
    const UnitSystem* system = nullptr;
    const ModalReduction* modal = nullptr;
    const linalg::DenseMatrix* second = nullptr;
};

// The second matrix of the pencil of an eigenvalue analysis on the interface: none for a static analysis;
// the mass (SOL 103: K·φ = λ·M·φ); the differential stiffness under a preload (SOL 105: (K + λ·K_d)·φ = 0).
enum class InterfaceSecondMatrix : std::uint8_t
{
    None,
    Mass,
    DifferentialStiffness
};

// An absorbed spider whose legs an element of the interface acts on: the fastener of an absorbed fastener side (its
// stiffness between P_A and P_B belongs to the interface problem). The unit's change of basis
// u_legs = W⁺·u_ref + N·q carries the element onto the reference DOFs: K ← Pᵀ·K·P with P = W⁺ on the legs. The
// N·q part is dropped, which is exact because the fastener's matrix on its legs is Mᵀ·k·M with M = W (a generated
// point) or W = A·M (a given GA), so that its rows on the legs lie in the row space of W and N, orthogonal to it,
// sees nothing (W·N = 0): the interface checks it (FASTENER_FACTOR_TOLERANCE).
struct InterfaceSpider
{
    SpiderKey key;
    SpiderWeightMatrix weights;                // W (m × n)
    std::vector<base::DofIndex> legDofs;       // the g-set DOFs of W's columns, increasing
    std::vector<base::DofIndex> referenceDofs; // the g-set DOFs of W's rows; empty: the generated point of `key`
    base::Vec3 generatedPointPosition{};       // the piercing point (basic) of a generated point
};

// The stiffness of an external unit on the condensed path (there a unit of the interface problem): the plugin's K* on
// g-set DOFs of interface points (the contour DOFs of its
// zone, whose elements belong to units kept full or to the connection system), added as a condensed unit's K* is, and
// summed with the other contributions for the singularity check (the full path adds it to Kgg before its singularity
// processor).
struct InterfaceExternalBlock
{
    std::int32_t zoneId = 0;                        // named in the messages
    std::span<const base::DofIndex> dofs;           // the rows and columns of K* (g set)
    const linalg::DenseMatrix* stiffness = nullptr; // dofs × dofs, symmetric (ExternalUnitAssembly::stiffness)
};

// What the interface problem of one group of subcases is made of.
struct InterfaceInputs
{
    const assembly::ElementSet* elements = nullptr;
    const assembly::DofMap* dofMap = nullptr;
    const UnitElementIndex* elementIndex = nullptr;
    // One flag per node of the DOF map: nonzero for the interior points of the condensed units, which the interface
    // problem does not hold; every other point of the model is one of its points.
    std::span<const std::uint8_t> isInteriorNode;
    // The elements of the units kept full and of the connection system, sorted (an ID no element batch holds, a
    // relation, is skipped).
    std::span<const base::EntityId> elementIds;
    std::span<const InterfaceUnit> units;
    std::span<const InterfaceSpider> spiders;
    // The external units placed at the interface, in their order; none by default.
    std::span<const InterfaceExternalBlock> externalBlocks;
    // The relations eliminated at the interface (those of the connection system and of the units kept full, and the
    // rows a condensed unit shares with it at its boundary points), every row's dependent DOF at an interface point.
    const assembly::RigidElimination* relations = nullptr;
    // The DOF sets of the full path for these subcases (m, sb, sg, r), for the held DOFs (the s and r sets) and the
    // DOFs the singularity processor checks (the f set).
    const assembly::DofSets* modelSets = nullptr;
    // Step 10.24: the g-set DOFs held in place of the r set of modelSets (a free body solved on conditioned
    // supports, assembly::chooseConditionedSupports; its FreeBodySupport DOFs then equations); empty: the r set held.
    std::span<const base::DofIndex> conditionedSupports;
    assembly::SingularityOptions singularity;
    linalg::PivotPolicy pivotPolicy;
    linalg::AssemblyOptions assembly;
    // The mass of the interface elements (inertia relief): computed when true, with these options and shares.
    bool computesMass = false;
    assembly::MassAssemblyOptions massOptions;
    const assembly::NonStructuralMassShares* nonStructuralMass = nullptr;
    // Eigenvalue analyses: the second matrix of the pencil (the mass with massOptions and
    // nonStructuralMass, or the differential stiffness of the interface elements under `preload`, the g-set
    // displacements of the preload), assembled with the units' parts (InterfaceUnit::second) on the pattern of the
    // stiffness, the relations eliminated as on it; and whether the stiffness is factorised (false: an eigenvalue
    // solver factorises K − σ·B at its shifts, linalg::SparsePencil or BucklingPencil on symbolic(), matrix() and
    // secondMatrix()).
    InterfaceSecondMatrix second = InterfaceSecondMatrix::None;
    const elements::NodalDisplacements* preload = nullptr;
    bool factorises = true;
    // Whether the factor of the stiffness is kept (true: made
    // here, every solve a sweep of it) or not (false: each solve factorises the stiffness again, streamed, each panel
    // released once used, linalg::StreamedSolve), its backward sweep by regions of at most regionValues panel values
    // (every panel kept from the forward run when the factor fits it: the factor made for the solve and released after
    // it). A static analysis whose interface factor does not fit its memory budget beside the rest (the memory plan's
    // strategy, reduction::MemoryStrategy::isInterfaceWithoutFactor).
    bool keepsFactor = true;
    std::int64_t regionValues = 0;
};

// What the solves of an interface problem without its factor did (InterfaceProblem::solveFigures): their
// count, the right-hand sides they took, the subtrees their regions factorised again, the flops of those and the most
// panel values a region kept at once. Zero with the factor kept.
struct InterfaceSolveFigures
{
    std::int64_t solveCount = 0;
    std::int64_t columnCount = 0;
    std::int64_t regionRunCount = 0;
    double recomputedFlops = 0.0;
    std::int64_t keptPanelValues = 0;
};

// An interface element acting on the legs of an absorbed spider must be the fastener of that side, whose rows on the
// legs lie in the row space of W (InterfaceSpider): their part outside it, (I − (W⁺·W)ᵀ)·K_L, stays below this
// fraction of the element's largest term. The element's map M and the spider's W come from the same piercing rows
// through two computations of the interpolation (elements::interpolatePiercing), so they agree to round-off; 1e-8 is
// far above it and far below any stiffness that would matter, so a plan inconsistent with its fastener is refused
// instead of being solved with a stiffness lost.
inline constexpr double FASTENER_FACTOR_TOLERANCE = 1.0e-8;

// The interface problem of a condensed static analysis: one
// node-block system of the K* of the condensed units (dense blocks), the elements of the units kept full and of the
// connection system (sparse), with its relations eliminated and the constraints of a group of subcases applied,
// assembled in a fixed order and factorised in memory.
//
// Its points: every point of the model but the interior points of the condensed units (the boundary points, the points
// of the full units, the connection-only points), then a point per generated fastener point. Its DOFs: node-major, six
// per grid or generated point, one per scalar point (the local DOFs).
//
// Steps, in this order (each the full path's on these points):
// 1. Matrices: the elements of the interface (InterfaceInputs::elementIds) in the model's order of batches and rows,
//    those on the legs of an absorbed spider carried to its reference (InterfaceSpider), through
//    linalg::SparseAssembler; then the K* of each unit, in the order of the units, added onto its labels.
// 2. Singularities ("the AUTOSPC of boundary nodes decided on the summed contributions"): a second
//    matrix on the same pattern sums what the full path's Knn holds at these points, the interface elements on their
//    own DOFs and each unit's raw boundary blocks (UnitSystem::rawBoundaryBlocks: its stiffness before the
//    condensation); its relations are eliminated and the grid point singularity processor (assembly::SingularBlock)
//    runs on its diagonal blocks, the DOFs of the full path's f set only (a spider's reference is in its m set; a
//    generated point is not a point of the model). A scalar point is checked against the largest free diagonal of the
//    interface and of every unit (UnitSystem::largestFreeDiagonal). The components found are held (AUTOSPC YES).
// 3. Relations: the rows of InterfaceInputs::relations on the local DOFs, eliminated in place by linear elimination
//    (LocalRelations: Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm), chains resolved in dependency order by
//    assembly::RigidElimination (a cycle or a DOF dependent twice is its error, reported before).
// 4. Constraints: the held DOFs (the s and r sets of the full path, the AUTOSPC of step 2) removed, their enforced
//    values loads −K_ls·u_s; their rows kept for their reactions (heldReactions).
// 5. Ordering: a nested dissection on the quotient graph of the units (each condensed unit's boundary one vertex,
//    InterfaceEstimate::orderOnUnits, the rule of the plan's estimate), the symbolic factor, the SymmetricCsc and the
//    supernodal factorisation (positive definite, MAXRATIO: a mechanism of the assembly is a SingularityError naming
//    its points; a generated fastener point is named by its fastener's ID).
//
// Eigenvalue analyses (Craig-Bampton for SOL 103 and SOL 105): the modal coordinates of the
// Craig-Bampton units are points of their own after the others (up to six coordinates a point, the q set; no point
// of the model: no singularity check, no rigid-body motion), in their unit's clique of the graph and its vertex of the
// quotient graph; K_qq is added on them; the second matrix (the interface elements' mass or differential stiffness
// and each unit's (a + N)-square part) is assembled on the same pattern and its relations eliminated as the
// stiffness's; without factorisation the pencil is symbolic(), matrix(), secondMatrix().
//
// Without its factor (InterfaceInputs::keepsFactor false): steps 1 to 5 but the factorisation; each
// solve factorises the stiffness streamed (linalg::StreamedSolve: the forward sweep while the panels are made, the
// backward sweep by regions of the elimination tree, each region factorised again), with the bits of the solve with
// the kept factor; the pivot diagnostics and a mechanism come with the first solve. The interface problem of the
// units of the dissection is the separators above them, so this is the substructuring of the units carried up the
// elimination tree: each level of the separators condensed onto its parent without its factor, the top solved, the
// levels recovered top-down region by region.
//
// Determinism: assembly in element order and units in order, eliminations serial, factorisation and solves
// deterministic: the same bits with any number of threads, with the factor kept or not.
//
// Thread-safety: immutable once built (but the figures of the solves without the factor, under a lock); const methods
// may run concurrently (the solves without the factor one at a time). Ownership: owns its matrices and factor
// (budgeted); borrows the model's element set, DOF map and the units' reduced units, which must outlive it, and,
// without its factor, the budget it was built with, which its solves reserve in: it must outlive it. Not movable.
class InterfaceProblem
{
public:
    // Throws std::invalid_argument when the inputs do not fit (a label of K* that is not an interface DOF, an element
    // reaching an interior point that is not an absorbed leg, a relation reaching an interior point), std::logic_error
    // when a fastener does not factor through its spider (FASTENER_FACTOR_TOLERANCE), SingularityError for a mechanism,
    // MemoryBudgetError, OperationCancelledError.
    InterfaceProblem(const InterfaceInputs& inputs, const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                     base::ThreadPool* pool = nullptr, const base::CancellationToken* cancellation = nullptr);
    InterfaceProblem(const InterfaceProblem&) = delete;
    InterfaceProblem& operator=(const InterfaceProblem&) = delete;
    InterfaceProblem(InterfaceProblem&&) = delete;
    InterfaceProblem& operator=(InterfaceProblem&&) = delete;
    ~InterfaceProblem();

    // The points and the local DOFs.
    [[nodiscard]] std::span<const InterfacePoint> points() const noexcept { return m_points.span(); }
    [[nodiscard]] std::int64_t localDofCount() const noexcept { return m_firstDof[m_firstDof.size() - 1]; }
    [[nodiscard]] std::int64_t firstDof(std::int32_t point) const noexcept
    {
        return m_firstDof[static_cast<std::size_t>(point)];
    }
    // The local DOF of a g-set DOF, nothing for a DOF of an interior point.
    [[nodiscard]] std::optional<std::int64_t> localDofOf(base::DofIndex dof) const noexcept;
    // The g-set DOF of a local DOF, nothing for a DOF of a generated fastener point.
    [[nodiscard]] std::optional<base::DofIndex> modelDofOf(std::int64_t localDof) const noexcept;
    // The local DOFs of the rows of a unit's K* (in the order of its labels).
    [[nodiscard]] std::span<const std::int64_t> unitLocalDofs(std::size_t unit) const noexcept
    {
        const auto first = static_cast<std::size_t>(m_unitDofOffsets[unit]);
        return m_unitDofs.span().subspan(first, static_cast<std::size_t>(m_unitDofOffsets[unit + 1]) - first);
    }
    // The local DOFs of the rows of an external block's K* (InterfaceInputs::externalBlocks order).
    [[nodiscard]] std::size_t externalBlockCount() const noexcept
    {
        return m_externalDofOffsets.size() == 0 ? 0 : m_externalDofOffsets.size() - 1;
    }
    [[nodiscard]] std::span<const std::int64_t> externalLocalDofs(std::size_t block) const noexcept
    {
        const auto first = static_cast<std::size_t>(m_externalDofOffsets[block]);
        return m_externalDofs.span().subspan(first, static_cast<std::size_t>(m_externalDofOffsets[block + 1]) - first);
    }
    // The first local DOF of a unit's modal coordinates and their count (InterfaceUnit::modal): the local
    // DOFs firstModalDof(u), ..., firstModalDof(u) + modalDofCount(u) − 1, in the order of its modes.
    [[nodiscard]] std::int64_t firstModalDof(std::size_t unit) const noexcept { return m_modalDofOffsets[unit]; }
    [[nodiscard]] std::int64_t modalDofCount(std::size_t unit) const noexcept
    {
        return m_modalDofOffsets[unit + 1] - m_modalDofOffsets[unit];
    }

    // The equations (the local DOFs neither dependent nor held), in the order of the factor.
    [[nodiscard]] std::int32_t equationCount() const noexcept { return m_selection.keptDofCount(); }
    [[nodiscard]] std::int64_t localDofOfEquation(std::int32_t equation) const noexcept
    {
        return m_selection.dofOfEquation(equation);
    }
    // The held local DOFs (increasing), and whether a local DOF is held or dependent.
    [[nodiscard]] std::span<const std::int64_t> heldDofs() const noexcept { return m_heldDofs.span(); }
    [[nodiscard]] bool isHeld(std::int64_t localDof) const noexcept
    {
        return m_role[static_cast<std::size_t>(localDof)] == HELD;
    }
    [[nodiscard]] bool isDependent(std::int64_t localDof) const noexcept
    {
        return m_role[static_cast<std::size_t>(localDof)] == DEPENDENT;
    }
    // The relations eliminated at the interface, u_m = Gm·u_n on the local DOFs (each term on an independent DOF: an
    // equation or a held DOF), as expand() applies them (the expansion handed to the caller).
    [[nodiscard]] const LocalRelations& relations() const noexcept { return m_relations; }

    // P_n = P̄_n + Gmᵀ·P_m on local vectors (local DOFs × columns), in place.
    void reduceLoads(linalg::DenseMatrixView loadsLocal) const;
    // The right-hand sides of the equations from reduced local loads (P_n, reduceLoads) and the enforced values of the
    // held DOFs (local DOFs × the same columns, read on the held DOFs; no column: none): f = P_n,l − K_ls·u_s. Throws
    // std::invalid_argument on a size mismatch.
    void equationLoads(linalg::ConstDenseMatrixView reducedLocal, linalg::ConstDenseMatrixView enforcedLocal,
                       linalg::DenseMatrixView equationValues) const;
    // B ← K_ll⁻¹·B (B of equationCount() rows). Throws OperationCancelledError between blocks, std::logic_error when
    // the stiffness was not factorised (InterfaceInputs::factorises). Without the factor: a factorisation streamed
    // for these columns (put every column of the analysis in one call), MemoryBudgetError before any numeric work,
    // SingularityError for a mechanism.
    void solve(linalg::DenseMatrixView rightHandSides, const base::CancellationToken* cancellation = nullptr) const;
    // The same with the caller's kernels: an interface problem kept after its analysis (the sensitivities of a design
    // point) outlives the kernels it was factorised with; the same kernel kind gives the same bits whatever
    // its threads.
    void solve(linalg::DenseMatrixView rightHandSides, const linalg::DenseKernels& kernels,
               const base::CancellationToken* cancellation) const;

    // The pencil of an eigenvalue analysis: the symbolic factor of the equations (the nested dissection on
    // the quotient graph of the units, a unit's modal coordinates in its vertex), the stiffness on the equations, and
    // the second matrix (InterfaceInputs::second) with the same structure; empty matrices without a second matrix.
    [[nodiscard]] const linalg::SymbolicFactor& symbolic() const noexcept { return m_symbolic; }
    [[nodiscard]] const linalg::SymmetricCsc& matrix() const noexcept { return m_matrix; }
    [[nodiscard]] const linalg::SymmetricCsc& secondMatrix() const noexcept { return m_secondMatrix; }
    // The local displacements from the solution of the equations and the enforced values: the held DOFs at their
    // values, u_m = Gm·u_n. Throws std::invalid_argument on a size mismatch.
    void expand(linalg::ConstDenseMatrixView equationValues, linalg::ConstDenseMatrixView enforcedLocal,
                linalg::DenseMatrixView displacementsLocal) const;
    // f − K_ll·u on the equations (EPSILON), column by column.
    void residuals(linalg::ConstDenseMatrixView equationValues, linalg::ConstDenseMatrixView rightHandSides,
                   linalg::DenseMatrixView residuals) const;
    // The reactions at the held DOFs (their SPC forces as the interface sees them), Knn,s·u_n − P_n,s, from the local
    // displacements and the reduced local loads; heldDofs().size() rows.
    void heldReactions(linalg::ConstDenseMatrixView displacementsLocal, linalg::ConstDenseMatrixView reducedLocal,
                       linalg::DenseMatrixView reactions) const;

    // The grid point singularity table rows of the interface points (step 2), by point ID then component.
    [[nodiscard]] std::span<const assembly::SingularityRow> singularities() const noexcept
    {
        return m_singularities.span();
    }
    // Without the factor: those of the first solve's factorisation (empty before it; read them after it).
    [[nodiscard]] const linalg::PivotDiagnostics& pivotDiagnostics() const noexcept
    {
        return m_isStreamed ? m_streamedDiagnostics : m_factor.pivotDiagnostics();
    }
    // The name of a local DOF for the messages: its point ID and component (a generated fastener point named by its
    // fastener's ID).
    [[nodiscard]] base::GridComponent nameOf(std::int64_t localDof) const;
    // The threads its factorisation ran on: those of the kernels given, or one when their working storage did not fit
    // the memory budget (the same bits either way). Without the factor: its last solve's.
    [[nodiscard]] std::size_t factorisationThreadCount() const noexcept
    {
        return m_isStreamed ? m_streamedThreadCount : m_factorisationThreadCount;
    }
    // The values of its factor's panels (the interface factor of the report; without the factor, those it would have).
    [[nodiscard]] std::int64_t factorValueCount() const noexcept
    {
        return m_isStreamed ? m_symbolic.panelValueCount() : m_factor.allocatedPanelValueCount();
    }
    // Step 10.25: whether its factor is kept (InterfaceInputs::keepsFactor), and without it the region bound and what
    // its solves did: their count, the subtrees factorised again for their regions and their flops, the most panel
    // values a region kept.
    [[nodiscard]] bool isFactorKept() const noexcept { return !m_isStreamed; }
    [[nodiscard]] std::int64_t regionValues() const noexcept { return m_regionValues; }
    [[nodiscard]] InterfaceSolveFigures solveFigures() const;

    // The rigid-body motions of the local DOFs about the basic origin (local DOFs × 6, RigidBodyModes' convention).
    [[nodiscard]] const linalg::DenseMatrix& rigidBodyModes() const noexcept { return m_rigidBodyModes; }
    // With InterfaceInputs::computesMass: Φ_rᵀ·M·Φ_r of the interface elements and their inertial loads −M·Φ_r (local
    // DOFs × 6), M carried to the references of the absorbed spiders as the stiffness is.
    [[nodiscard]] const base::MatN<6>& rigidBodyMass() const noexcept { return m_rigidBodyMass; }
    [[nodiscard]] const linalg::DenseMatrix& inertialLoads() const noexcept { return m_inertialLoads; }

private:
    static constexpr std::uint8_t FREE = 0;
    static constexpr std::uint8_t DEPENDENT = 1;
    static constexpr std::uint8_t HELD = 2;

    struct LocalSpider;

    void numberPoints(const InterfaceInputs& inputs, base::MemoryBudget& budget);
    void mapUnits(const InterfaceInputs& inputs, base::MemoryBudget& budget);
    void readRelations(const InterfaceInputs& inputs, base::MemoryBudget& budget);
    void buildGraph(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool);
    void assemble(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool,
                  const base::CancellationToken* cancellation);
    void checkSingularities(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool);
    void computeMassProperties(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool);
    // The second matrix of an eigenvalue analysis (InterfaceInputs::second): the interface elements' differential
    // stiffness (the mass is computeMassProperties's), then the units' parts.
    void assembleSecond(const InterfaceInputs& inputs, base::MemoryBudget& budget, base::ThreadPool* pool);
    void order(const InterfaceInputs& inputs, const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
               base::ThreadPool* pool, const base::CancellationToken* cancellation);
    void keepHeldRows(base::MemoryBudget& budget);
    // A solve without the factor.
    void solveStreamed(linalg::DenseMatrixView rightHandSides, const linalg::DenseKernels& kernels,
                       const base::CancellationToken* cancellation) const;

    // The contributions of the interface elements on the local DOFs, run by run of the element list (facet: stiffness
    // or mass), carried to the spiders' references (isRaw false) or restricted to the interface DOFs (isRaw true).
    template <typename Facet>
    void addElementContributions(const InterfaceInputs& inputs, std::size_t begin, std::size_t end, Facet&& facet,
                                 bool isRaw, linalg::ContributionBlock& contributions) const;

    [[nodiscard]] std::int32_t pointOfLocalDof(std::int64_t localDof) const noexcept;
    [[nodiscard]] linalg::NodeDof nodeDofOfLocal(std::int64_t localDof) const noexcept;

    const assembly::DofMap* m_dofMap = nullptr;
    base::BudgetedBuffer<InterfacePoint> m_points;
    base::BudgetedBuffer<std::int64_t> m_firstDof;        // points + 1
    base::BudgetedBuffer<std::int32_t> m_pointOfNode;     // per node of the DOF map: its point, -1 for an interior one
    base::BudgetedBuffer<std::int64_t> m_unitDofOffsets;  // units + 1
    base::BudgetedBuffer<std::int64_t> m_unitDofs;        // per unit: the local DOF of each K* row
    base::BudgetedBuffer<std::int64_t> m_modalDofOffsets; // units + 1: the first local DOF of each unit's q set
    base::BudgetedBuffer<std::int64_t> m_externalDofOffsets; // external blocks + 1
    base::BudgetedBuffer<std::int64_t> m_externalDofs;       // per external block: the local DOF of each K* row
    std::unique_ptr<linalg::BlockSparseMatrix> m_second;     // the second matrix while it is built
    linalg::SymmetricCsc m_secondMatrix;
    bool m_isFactorised = false;
    // A leg DOF of an absorbed spider: its spider and its column in W.
    struct LegEntry
    {
        base::DofIndex dof = 0;
        std::int32_t spider = 0;
        std::int32_t slot = 0;
    };
    std::vector<std::unique_ptr<LocalSpider>> m_spiders; // bounded by the absorbed fastener sides
    base::BudgetedBuffer<LegEntry> m_legs;               // sorted by DOF
    base::BudgetedBuffer<ElementPlace> m_elementPlaces;  // the interface elements with a matrix, in model order
    LocalRelations m_relations;
    base::BudgetedBuffer<std::uint8_t> m_role; // per local DOF: FREE, DEPENDENT or HELD
    linalg::NodeGraph m_graph;
    std::unique_ptr<linalg::SparsePattern> m_pattern;
    std::unique_ptr<linalg::BlockSparseMatrix> m_stiffness;
    base::BudgetedBuffer<assembly::SingularityRow> m_singularities;
    base::BudgetedBuffer<std::int64_t> m_heldDofs;
    // Knn on the held DOFs: per held DOF (as m_heldDofs), its terms on every local DOF (CSR).
    base::BudgetedBuffer<std::int64_t> m_heldOffsets;
    base::BudgetedBuffer<std::int64_t> m_heldColumns;
    base::BudgetedBuffer<double> m_heldValues;
    linalg::DofSelection m_selection;
    linalg::SymbolicFactor m_symbolic;
    linalg::SymmetricCsc m_matrix;
    linalg::SupernodalFactor m_factor;
    std::size_t m_factorisationThreadCount = 1;
    // Without the factor: the bound, the options of its factorisations, the budget its solves reserve in,
    // and, under the lock, the diagnostics of the first solve and the figures of every solve.
    bool m_isStreamed = false;
    std::int64_t m_regionValues = 0;
    linalg::PivotPolicy m_pivotPolicy;
    base::MemoryBudget* m_budget = nullptr;
    mutable std::mutex m_streamMutex;
    mutable linalg::PivotDiagnostics m_streamedDiagnostics;
    mutable InterfaceSolveFigures m_solveFigures;
    mutable std::size_t m_streamedThreadCount = 1;
    const linalg::DenseKernels* m_kernels = nullptr;
    linalg::DenseMatrix m_rigidBodyModes;
    base::MatN<6> m_rigidBodyMass{};
    linalg::DenseMatrix m_inertialLoads;
};

} // namespace oos::reduction
