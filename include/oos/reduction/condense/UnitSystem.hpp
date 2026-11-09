#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"
#include "oos/linalg/sparse/SparseAssembler.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/reduction/absorption/SpiderChangeOfBasis.hpp"
#include "oos/reduction/condense/LocalRelations.hpp"
#include "oos/reduction/condense/UnitDescription.hpp"
#include "oos/reduction/condense/UnitElementIndex.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::reduction
{

class ReducedUnit; // reduction/condense/ReducedUnit.hpp (which names the labels of this header)

// The role of a DOF of a unit in its condensation.
enum class UnitDofRole : std::uint8_t
{
    Interior,    // an interior equation: eliminated by the partial factorisation (the o set of the unit)
    Boundary,    // a boundary equation: kept in K* (the a set of the unit)
    Constrained, // held in every subcase (SPC, GRID PS, AUTOSPC of the full path aside): removed, its enforced value a
                 // load
    Dependent,   // made dependent by a relation eliminated inside the unit (a row of the units' Gm)
    Absent       // a leg slot an absorbed spider frees: its m reference values live at the reference point
};

// What a point of a unit is.
enum class UnitPointKind : std::uint8_t
{
    Grid,
    Scalar,
    FastenerPoint,  // the point generated on one side of a Fastener or Weld (no point of the model)
    ModalCoordinate // the generalised coordinates of a Craig-Bampton unit's fixed-interface modes (the q set; the
                    // interface problem of an eigenvalue analysis): its ID is the unit's
};

// The name of a DOF of a unit (grid or reference point, component).
struct UnitDofLabel
{
    UnitPointKind kind = UnitPointKind::Grid;
    base::EntityId id = 0; // the grid or scalar point ID; the Fastener or Weld ID of a fastener point
    FastenerSide side = FastenerSide::None; // the side of a fastener point
    std::uint8_t component = 1;             // 1 to 6 (0 for a scalar point); a fastener point's in the basic system

    friend constexpr bool operator==(const UnitDofLabel&, const UnitDofLabel&) = default;
};

// The model the units are cut from, shared by all of them (built once per analysis).
struct UnitModel
{
    const assembly::ElementSet* elements = nullptr; // checked
    const assembly::DofMap* dofMap = nullptr;
    const UnitElementIndex* elementIndex = nullptr;
    // The relations eliminated inside the units: the Gm of the rigid elements, fastener constraints,
    // MPC equations and glue that clusters and spiders brought into units, every relation of the connection system
    // left out (assembly::RigidElimination with a filter), so that chains resolve inside each unit.
    const assembly::RigidElimination* unitRelations = nullptr;
    // One flag per g-set DOF: nonzero when held in every subcase (a DOF held in some subcases only is on a boundary).
    std::span<const std::uint8_t> isConstrainedEverywhere;
    // The non-structural mass of the NSM set (null: none) and the mass options (COUPMASS, WTMASS).
    const assembly::NonStructuralMassShares* nonStructuralMass = nullptr;
    assembly::MassAssemblyOptions massOptions;
    linalg::AssemblyOptions assembly;
    // The grid point singularity processor of the unit's interior points (PARAM AUTOSPC, EPZERO).
    assembly::SingularityOptions singularity;
    // Craig-Bampton: keep the unit's mass on its equations (UnitSystem::mass()), its relations eliminated
    // and its spiders absorbed as the stiffness's, for its fixed-interface modes and its reduced mass.
    bool keepsMass = false;
};

// A block of the unit's stiffness between two of its boundary points (or a boundary point's diagonal block), as the
// assembly and the relations eliminated inside the unit leave it, before the spider change of basis and the
// condensation: the unit's contribution to the blocks of the full model's Knn at its boundary, which the interface
// problem sums with the others to check the singularities of the boundary points on the blocks the full path sees
// ("the AUTOSPC of boundary nodes decided on the summed contributions"). The nodes are those of the
// model (assembly::DofMap); the values rowDofCount × columnDofCount, column-major, at valueOffset in
// UnitSystem::rawBoundaryValues(); the rows and columns of the DOFs folded by the unit's relations (dependent DOFs,
// the reference DOFs of its spiders) are zero.
struct RawBoundaryBlock
{
    std::int32_t rowNode = 0;
    std::int32_t columnNode = 0; // rowNode >= columnNode
    std::int64_t valueOffset = 0;
};

// What an analysis reads of a unit's build beside its equations, for a unit taken from the condensation
// cache whose equations the analysis does not read (UnitSystem's constructor without equations): the singularity rows
// of its interior points (and so the DOFs AUTOSPC holds), its largest free diagonal, its raw boundary blocks, its
// elements with a point on the boundary, whether each DOF held in every subcase is coupled, its boundary labels and the
// local DOFs of its boundary equations, its interior equation count, its rigid-body mass properties. Made by a build
// with the equations (UnitSystem::boundaryData), a deterministic function of the unit's inputs: the build of an equal
// cache key gives the same values to the bit. Never in the condensation cache (it may be large): kept in memory
// between analyses by the structure cache of the solutions.
//
// Thread-safety: immutable once made. Ownership: owns its buffers (budgeted in the budget given, which must outlive
// it).
struct UnitBoundaryData
{
    base::BudgetedBuffer<assembly::SingularityRow> singularities;
    double largestFreeDiagonal = 0.0;
    base::BudgetedBuffer<RawBoundaryBlock> rawBlocks;
    base::BudgetedBuffer<double> rawValues;
    base::BudgetedBuffer<base::EntityId> boundaryElements;
    base::BudgetedBuffer<std::uint8_t> isHeldCoupled; // per DOF held in every subcase, in UnitSystem::heldDofs() order
    base::BudgetedBuffer<UnitDofLabel> boundaryLabels;
    base::BudgetedBuffer<std::int64_t> boundaryLocalDofs; // the local DOF of each boundary equation, in their order
    std::int32_t interiorDofCount = 0;
    base::MatN<6> rigidBodyMass{};

    // Whether `other` holds the same values, bit for bit (a verification of the cache key: CondensedStatics).
    [[nodiscard]] bool isEqual(const UnitBoundaryData& other) const noexcept;
};

// The equations of one unit, ready for its partial factorisation: the unit's own
// element matrices assembled on its own points, its constraints applied first, the spiders absorbed by change of
// basis, its DOFs numbered with the boundary last.
//
// Steps, in this order (each one the full path's on the unit alone, so that the interior equations are the full
// model's):
// 1. Numbering: the unit's points in increasing order of their node in the model, then a point of six DOFs per
//    generated fastener point; each DOF gets its role (UnitDofRole). A boundary point brings all its DOFs that are
//    neither held in every subcase nor dependent inside the unit (point granularity).
// 2. Assembly: the stiffness K and the mass M of the unit's elements only (the connections of the connection system
//    are the interface problem's), in the model's order of batches and rows, through linalg::SparseAssembler: an
//    interior term of K sums the same element terms in the same order as the full model's Kgg.
// 3. Constraints: the dependent DOFs of the units' Gm on the unit's points eliminated by linear elimination
//    (MsetElimination): with u_m = Gm·u_n,
//        Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm,
//    computed in place (the reads touch terms with a dependent index, the writes terms between independent ones);
//    the DOFs held in every subcase removed, their enforced values loads −K_fs·u_s.
// 3b. Singularities: the grid point singularity processor of the full path (assembly::SingularBlock, the
//    rules of GridPointSingularities) on the diagonal blocks of the interior points, which hold every contribution the
//    full model's Knn has there (no element or relation outside the unit reaches an interior point): with PARAM
//    AUTOSPC YES the singular components are held (role Constrained, value 0), as the full path holds them. A scalar
//    point is singular when its diagonal is at most EPZERO times the largest free diagonal of the unit (the full path
//    takes the largest of the whole model: the one difference, documented in the step report). The boundary points
//    are checked by the interface problem on the summed blocks (rawBoundaryBlocks()), and an absorbed spider's leg
//    found singular is refused (the spider is then kept on its legs).
// 4. Spiders (only by change of basis, applied to the assembled rows of the legs, equal to the
//    element-level form, without a dense element fill): for each absorbed spider of W (m reference DOFs, n leg DOFs),
//    u_legs = W⁺·u_ref + N·q = T·[u_ref; q] (SpiderChangeOfBasis). After step 3 the reference DOFs are dependent
//    (u_ref = W·u_legs, their stiffness folded onto the legs), so the unknowns (legs) become (u_ref, q) by
//        K ← Tᵀ·K·T on the leg rows and columns,
//    the m reference rows going to the reference point's DOFs (boundary) and the n − m deformation rows to the last
//    n − m leg slots (interior), the first m leg slots left empty (Absent). The relation u_ref = W·u_legs then holds
//    for every (u_ref, q) (W·W⁺ = I, W·N = 0), so this is exact; T is square and invertible, so the interior stays
//    positive definite (no flexibility form, no Lagrange multiplier). Loads: P_legs ← Tᵀ·P_legs (after
//    Gmᵀ moved the reference loads onto the legs: P'_ref = W⁺ᵀ·(P_legs + Wᵀ·P_ref) = W⁺ᵀ·P_legs + P_ref). Recovery:
//    u_legs = T·[u_ref; q], the reference DOFs keeping their boundary values.
// 5. Ordering: the points with an equation ordered by linalg::ConstrainedOrdering (the rule of the full path and of
//    the cost estimate, UnitCostEstimate) with the boundary points last; SymbolicFactor with the boundary as trailing
//    set; the SymmetricCsc of K in that order.
//
// The node graph holds every coupling these steps create: the element groups, each group extended with the points of
// the Gm rows of its dependent points (MsetCouplings), then with every leg and the reference point of each spider any
// of whose legs or reference it holds (the rows Tᵀ·K mix the rows of all legs).
//
// A unit taken from the condensation cache whose equations its analysis does not read (not recovered, no
// Craig-Bampton modes, no inertia relief, no system kept) is built without them: steps 1 and the roles of 3b from a
// UnitBoundaryData that an earlier build of the same cache key gave, no element matrix computed. Such a system answers
// every query of its numbering, its roles, its boundary and its singularities as the full build would, to the bit;
// what needs the equations or the mass (matrix(), symbolic(), mass(), equationLoads(), expandSolution(),
// heldReactions(), inertialLoads(), the interior equations' numbering) throws base::InternalError (hasEquations()).
//
// Thread-safety: immutable once built; const methods may run concurrently. Ownership: owns its matrices and maps
// (budgeted); borrows the model's DOF map, which must outlive it. Not movable (shared by its condensation).
class UnitSystem
{
public:
    // Throws std::invalid_argument when the description does not fit the model (a point or a boundary point it does not
    // have, an element with a point outside the unit, a relation of the units' Gm reaching outside the unit, a spider
    // whose leg is on the boundary, constrained or dependent, whose reference is not on the boundary, or whose W is
    // rank-deficient), base::MemoryBudgetError, and as the assembly.
    UnitSystem(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
               base::ThreadPool* pool = nullptr);
    // What a build of a unit's equations again for its recovery takes beside them: the rigid-body
    // mass properties of its first build (no mass is assembled again).
    struct WithoutMass
    {
        base::MatN<6> rigidBodyMass{};
    };
    // The equations of a unit built again for its recovery when they were released after its condensation: the steps
    // of the constructor above without the mass, which no recovery reads (the inertial patterns of
    // inertia relief are among the unit's patterns already), its rigid-body mass properties given. The same equations
    // to the bit: the stiffness is assembled apart from the mass (as boundaryDataOf's build). inertialLoads() throws
    // base::InternalError on it. Throws as the constructor above; std::invalid_argument for a model that keeps the mass
    // (Craig-Bampton: its equations are never released).
    UnitSystem(const UnitModel& model, const UnitDescription& description, const WithoutMass& withoutMass,
               base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    // The unit without its equations, from what a build of the same cache key gave (`data`). Throws as the constructor
    // above for the numbering, and base::InternalError when `data` does not fit the unit (its held DOFs or its boundary
    // equations: a key that missed an input).
    UnitSystem(const UnitModel& model, const UnitDescription& description, const UnitBoundaryData& data,
               base::MemoryBudget& budget);
    UnitSystem(const UnitSystem&) = delete;
    UnitSystem& operator=(const UnitSystem&) = delete;
    UnitSystem(UnitSystem&&) = delete;
    UnitSystem& operator=(UnitSystem&&) = delete;
    ~UnitSystem();

    [[nodiscard]] std::int64_t unitId() const noexcept { return m_unitId; }
    // False for a unit built without its equations (from a UnitBoundaryData).
    [[nodiscard]] bool hasEquations() const noexcept { return m_hasEquations; }
    // What a build without equations of the same unit needs (UnitBoundaryData), in `budget`. Throws base::InternalError
    // without equations (hasEquations()), MemoryBudgetError.
    [[nodiscard]] UnitBoundaryData boundaryData(base::MemoryBudget& budget) const;
    // The boundary data of a unit taken from the cache (its `reduced` unit: the rows of K*, the rigid-body mass
    // properties) when no build of its key left them (another session): steps 1 to 4 on its stiffness alone, as the
    // full build makes them (the same bits), without its mass, its ordering or its matrix. Throws as the full build,
    // and base::InternalError when `reduced` does not fit the unit (a key that missed an input).
    // PERFORMANCE: a unit read from the cache's folder skips its mass, ordering, symbolic analysis and conversion
    // (about 0.6 of the 1.6 ms of a unit of 744 DOFs on one thread), which no part of the analysis reads. Measured on
    // the reanalysis of a condensed model from a cache on disk.
    [[nodiscard]] static UnitBoundaryData boundaryDataOf(const UnitModel& model, const UnitDescription& description,
                                                         const ReducedUnit& reduced, base::MemoryBudget& budget,
                                                         base::ThreadPool* pool = nullptr);

    // The local numbering: node-major DOFs of the unit's points (6 per grid or fastener point, 1 per scalar point).
    [[nodiscard]] std::int32_t pointCount() const noexcept
    {
        return static_cast<std::int32_t>(m_pointDofCounts.size());
    }
    [[nodiscard]] std::int64_t localDofCount() const noexcept { return m_firstDof[m_firstDof.size() - 1]; }
    [[nodiscard]] std::span<const UnitDofRole> roles() const noexcept { return m_roles.span(); }
    [[nodiscard]] UnitDofLabel labelOf(std::int64_t localDof) const noexcept;
    // The g-set DOF of a local DOF, nothing for a DOF of a generated fastener point.
    [[nodiscard]] std::optional<base::DofIndex> modelDofOf(std::int64_t localDof) const noexcept;
    // The local DOF of a g-set DOF, nothing when its point is not the unit's.
    [[nodiscard]] std::optional<std::int64_t> localDofOf(base::DofIndex dof) const noexcept;

    // The equations: the interior ones first, then the boundary ones (the trailing set).
    [[nodiscard]] std::int32_t equationCount() const noexcept { return interiorDofCount() + boundaryDofCount(); }
    [[nodiscard]] std::int32_t boundaryDofCount() const noexcept
    {
        return static_cast<std::int32_t>(m_boundaryLabels.size());
    }
    [[nodiscard]] std::int32_t interiorDofCount() const noexcept { return m_interiorEquationCount; }
    // The local DOF of an equation; without equations, of a boundary equation only (base::InternalError otherwise).
    [[nodiscard]] std::int64_t localDofOfEquation(std::int32_t equation) const;
    // The equation of a local DOF, or linalg::DofSelection::ELIMINATED. Throws base::InternalError without equations.
    [[nodiscard]] std::int32_t equationOfLocalDof(std::int64_t localDof) const;
    // The labels of the boundary equations, in their order (the rows of K*).
    [[nodiscard]] std::span<const UnitDofLabel> boundaryLabels() const noexcept { return m_boundaryLabels.span(); }

    // Throw base::InternalError without equations (hasEquations()).
    [[nodiscard]] const linalg::SymbolicFactor& symbolic() const;
    [[nodiscard]] const linalg::SymmetricCsc& matrix() const;

    // Craig-Bampton (UnitModel::keepsMass). The mass M on the equations, in the order and with the structure
    // of matrix(): the unit's elements' mass (WTMASS applied), its relations eliminated like the m set
    // (Mnn = M̄nn + M̄nm·Gm + Gmᵀ·M̄mn + Gmᵀ·M̄mm·Gm) and its spiders absorbed (M ← Tᵀ·M·T), as the stiffness.
    [[nodiscard]] bool hasMass() const noexcept { return m_hasMass; }
    // Throws std::logic_error when the mass was not kept.
    [[nodiscard]] const linalg::SymmetricCsc& mass() const;
    // The symbolic factor of the interior equations alone (the leading interiorDofCount() equations of matrix()),
    // whose equations are numbered as those of matrix(): the interior part of the unit's elimination order is a
    // postorder of the interior's own elimination tree, so the analysis keeps it (checked: base::InternalError
    // otherwise). With SymmetricCsc::leadingBlock of matrix() and mass(), the pencil (K_oo, M_oo) of the
    // fixed-interface modes. Throws MemoryBudgetError.
    [[nodiscard]] linalg::SymbolicFactor interiorSymbolic(base::MemoryBudget& budget) const;
    // The differential stiffness K_d of the unit's elements under a preload (SOL 105: the differential stiffness of a
    // condensed unit is reduced with the same basis), on the equations as matrix(): each element's
    // ElementFamily::computeDifferentialStiffness from the g-set preload, assembled on the unit's points in the
    // model's order, its relations eliminated and its spiders absorbed as the stiffness. `model` and `description` must
    // be those the unit was built from. Throws as the assembly, MemoryBudgetError.
    [[nodiscard]] linalg::SymmetricCsc differentialStiffness(const UnitModel& model, const UnitDescription& description,
                                                             const elements::NodalDisplacements& preload,
                                                             base::MemoryBudget& budget,
                                                             base::ThreadPool* pool = nullptr) const;

    // The loads of the equations (equationCount() rows) from loads on the local DOFs (localDofCount() rows, in the
    // displacement systems of the points: a fastener point's in the basic system) and the enforced displacements of
    // the DOFs held in every subcase (localDofCount() rows read on the Constrained DOFs only; no column: none), column
    // by column: f = Tᵀ·(P̄_n + Gmᵀ·P_m) − K_fs·u_s. Throws std::invalid_argument on a size mismatch;
    // MemoryBudgetError for its work vector.
    void equationLoads(linalg::ConstDenseMatrixView loadsLocal, linalg::ConstDenseMatrixView enforcedLocal,
                       linalg::DenseMatrixView equationValues, base::MemoryBudget& budget) const;
    // The displacements of the local DOFs (localDofCount() rows) from the solution of the equations (equationCount()
    // rows) and the enforced displacements (no column: none), column by column: the legs of the spiders from
    // T·[u_ref; q], the dependent DOFs by Gm, the held DOFs at their enforced values. Throws std::invalid_argument on a
    // size mismatch.
    void expandSolution(linalg::ConstDenseMatrixView equationValues, linalg::ConstDenseMatrixView enforcedLocal,
                        linalg::DenseMatrixView displacementsLocal) const;

    // The values of a g-set vector on the local DOFs (0 on a generated fastener point). Throws std::invalid_argument
    // on a size mismatch.
    void gatherLocal(std::span<const double> valuesG, std::span<double> valuesLocal) const;

    // The rigid-body motions of the unit about the basic origin (localDofCount() × 6; columns: translations along the
    // basic axes, rotations about them; RigidBodyModes' convention: u_cd = Rᵀ·(t + θ × x), θ_cd = Rᵀ·θ), made from the
    // points' positions and displacement systems each time they are asked (the same bits; not kept,
    // 48 bytes a local DOF), with or without the equations. Throws MemoryBudgetError.
    [[nodiscard]] linalg::DenseMatrix makeRigidBodyModes(base::MemoryBudget& budget) const;
    // Φ_rᵀ·M·Φ_r about the basic origin, M the unit's mass (WTMASS applied): its rigid-body mass properties.
    [[nodiscard]] const base::MatN<6>& rigidBodyMass() const noexcept { return m_rigidBodyMass; }
    // −M·Φ_r on the local DOFs (localDofCount() × 6): the loads of a unit rigid-body acceleration (d'Alembert), the six
    // inertial load patterns of inertia relief. Throws base::InternalError without equations or built again without
    // the mass (no mass assembled).
    [[nodiscard]] const linalg::DenseMatrix& inertialLoads() const;

    // The spiders absorbed by change of basis.
    [[nodiscard]] std::size_t spiderCount() const noexcept { return m_spiders.size(); }

    // The rows of the grid point singularity table of the interior points (step 3b), by point ID then component: the
    // DOFs AUTOSPC holds (isConstrained) or, with PARAM AUTOSPC NO, reports.
    [[nodiscard]] std::span<const assembly::SingularityRow> interiorSingularities() const noexcept
    {
        return m_singularities.span();
    }
    // The largest diagonal term of the unit's free DOFs (interior and boundary, before the condensation): the interface
    // problem checks its scalar points against the largest of every unit and its own.
    [[nodiscard]] double largestFreeDiagonal() const noexcept { return m_largestFreeDiagonal; }
    // The unit's blocks at its boundary points (RawBoundaryBlock), by column node then row node.
    [[nodiscard]] std::span<const RawBoundaryBlock> rawBoundaryBlocks() const noexcept { return m_rawBlocks.span(); }
    [[nodiscard]] std::span<const double> rawBoundaryValues() const noexcept { return m_rawValues.span(); }
    // The unit's elements with a point on its boundary, sorted: a distributed load on one of them loads the boundary
    // too.
    [[nodiscard]] std::span<const base::EntityId> boundaryElementIds() const noexcept
    {
        return m_boundaryElements.span();
    }

    // The local DOFs held in every subcase (role Constrained: SPC, GRID PS, the AUTOSPC of step 3b), increasing.
    [[nodiscard]] std::span<const std::int64_t> heldDofs() const noexcept { return m_constrainedDofs.span(); }
    // Whether the held DOF heldDofs()[k] has a stiffness term in the unit (its reaction then comes from the unit too).
    [[nodiscard]] bool isHeldDofCoupled(std::size_t k) const noexcept { return m_isHeldCoupled[k] != 0; }
    // The reactions of the unit at its held DOFs (its part of their SPC forces), column by column:
    //     q_s = K_sf·u_f + K_ss·u_s − P_n,s,
    // from the solution of the equations (equationCount() rows), the enforced values (localDofCount() rows read on
    // the held DOFs; no column: none) and the unit's loads on its local DOFs (localDofCount() rows; P_n the loads with
    // the dependent DOFs' moved by Gmᵀ), into `reactions` (heldDofs().size() rows). Throws std::invalid_argument on a
    // size mismatch; MemoryBudgetError for its work vector.
    void heldReactions(linalg::ConstDenseMatrixView equationValues, linalg::ConstDenseMatrixView enforcedLocal,
                       linalg::ConstDenseMatrixView loadsLocal, linalg::DenseMatrixView reactions,
                       base::MemoryBudget& budget) const;

private:
    // A spider absorbed by change of basis, on the local DOFs.
    struct LocalSpider
    {
        SpiderKey key;
        SpiderChangeOfBasis changeOfBasis;
        std::vector<std::int64_t> legs;       // in the order of W's columns (n), bounded by the spider
        std::vector<std::int64_t> references; // in the order of W's rows (m)
        std::vector<std::int32_t> points;     // its leg points and its reference point (local), sorted
    };

    void numberPoints(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget);
    void readRelations(const UnitModel& model, base::MemoryBudget& budget);
    void readSpiders(const UnitDescription& description, base::MemoryBudget& budget);
    void assignRoles(const UnitModel& model, base::MemoryBudget& budget);
    // The element stiffness matrices buildGraph() computes, kept for assemble() (UnitSystem.cpp).
    struct KeptStiffness;
    void buildGraph(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                    base::ThreadPool* pool, KeptStiffness& kept);
    // The tag of the build of boundaryDataOf: the stiffness alone, no mass, no ordering.
    struct StiffnessOnly
    {
    };
    UnitSystem(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
               base::ThreadPool* pool, StiffnessOnly);
    // Steps 1 to 4 (the mass with `assemblesMass`).
    void buildStiffness(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                        base::ThreadPool* pool, bool assemblesMass);
    void assemble(const UnitModel& model, const UnitDescription& description, base::MemoryBudget& budget,
                  base::ThreadPool* pool, const KeptStiffness& kept, bool assemblesMass);
    // m_isHeldCoupled from the roles and the stiffness after step 4, as keepEnforcedCouplings decides it.
    void findHeldCouplings(base::MemoryBudget& budget);
    void eliminateDependents(base::MemoryBudget& budget);
    void keepRawBoundaryStiffness(base::MemoryBudget& budget);
    void constrainSingularities(const UnitModel& model, base::MemoryBudget& budget);
    // K ← Tᵀ·K·T on the leg rows of every absorbed spider, for a matrix on `pattern` (the unit's: the stiffness, the
    // mass, the differential stiffness).
    void transformSpiders(const linalg::SparsePattern& pattern, std::span<double> values) const;
    void computeMassProperties(const linalg::BlockSparseMatrix& mass, base::MemoryBudget& budget);
    // Throws base::InternalError naming `what` when the system has no equations.
    void requireEquations(const char* what) const;
    void order(base::MemoryBudget& budget, base::ThreadPool* pool);
    void keepEnforcedCouplings(base::MemoryBudget& budget);

    // The local DOF of a model point's component (from 0), nothing when the point is not the unit's.
    [[nodiscard]] std::optional<std::int64_t> localDofOfNode(std::int32_t modelNode, int component) const noexcept;
    [[nodiscard]] std::optional<std::int32_t> localPointOf(std::int32_t modelNode) const noexcept;
    // The local point of a local DOF.
    [[nodiscard]] std::int32_t pointOfLocalDof(std::int64_t localDof) const noexcept;
    // The ID of a local point: its grid or scalar point ID, the fastener's ID for a generated point.
    [[nodiscard]] base::EntityId pointIdOf(std::int32_t localPoint) const noexcept;
    // The g-set DOF of a component (from 0) of a point of the model.
    [[nodiscard]] base::DofIndex modelDofOfNode(std::int32_t modelNode, int component) const noexcept;
    // Calls visit(localDof) for the DOFs of a point and of its neighbours in the graph, by increasing point.
    void forEachCoupledDof(std::int32_t point, const std::function<void(std::int64_t)>& visit) const;

    std::int64_t m_unitId = 0;
    const assembly::DofMap* m_dofMap = nullptr;
    // Per local point: its node in the model (-1 for a generated fastener point), its DOF count, its position.
    base::BudgetedBuffer<std::int32_t> m_modelNodes;
    base::BudgetedBuffer<std::uint8_t> m_pointDofCounts;
    base::BudgetedBuffer<std::int64_t> m_firstDof; // pointCount() + 1: the node-major numbering of the local DOFs
    base::BudgetedBuffer<base::Vec3> m_positions;
    // The model nodes of the unit's points, sorted, with their local point (for the lookups).
    base::BudgetedBuffer<std::int32_t> m_sortedModelNodes;
    std::vector<UnitDofLabel> m_fastenerPoints; // per generated point: its label at component 1 (bounded by spiders)
    base::BudgetedBuffer<std::uint8_t> m_isBoundaryPoint;
    base::BudgetedBuffer<UnitDofRole> m_roles;
    // The rows of the units' Gm whose dependent DOF is the unit's, by increasing dependent DOF (CSR); a row of a
    // spider's reference DOF is kept for the elimination and skipped by the expansion (the reference keeps its
    // boundary value).
    LocalRelations m_relations;
    base::BudgetedBuffer<std::uint8_t> m_isSpiderReferenceRow;
    std::vector<LocalSpider> m_spiders; // bounded by the spiders of the unit
    linalg::NodeGraph m_graph;
    // The pattern and K on it while the system is built (released once K is in m_matrix), and M with UnitModel::
    // keepsMass (released once in m_mass).
    std::unique_ptr<linalg::SparsePattern> m_pattern;
    std::unique_ptr<linalg::BlockSparseMatrix> m_stiffness;
    std::unique_ptr<linalg::BlockSparseMatrix> m_massBlocks;
    bool m_hasMass = false;
    linalg::SymmetricCsc m_mass;
    // K_fs: for each DOF held in every subcase (increasing), its terms on the equations (CSR), for the enforced loads.
    base::BudgetedBuffer<std::int64_t> m_constrainedDofs;
    base::BudgetedBuffer<std::int64_t> m_constrainedOffsets;
    base::BudgetedBuffer<std::int32_t> m_constrainedEquations;
    base::BudgetedBuffer<double> m_constrainedValues;
    // K_ss: for each held DOF (as m_constrainedDofs), its terms on the held DOFs (CSR, by index in m_constrainedDofs).
    base::BudgetedBuffer<std::int64_t> m_heldOffsets;
    base::BudgetedBuffer<std::int32_t> m_heldIndices;
    base::BudgetedBuffer<double> m_heldValues;
    // Per held DOF (as m_constrainedDofs): whether it has a term in K_fs or K_ss (isHeldDofCoupled).
    base::BudgetedBuffer<std::uint8_t> m_isHeldCoupled;
    base::BudgetedBuffer<assembly::SingularityRow> m_singularities;
    double m_largestFreeDiagonal = 0.0;
    base::BudgetedBuffer<RawBoundaryBlock> m_rawBlocks;
    base::BudgetedBuffer<double> m_rawValues;
    base::BudgetedBuffer<base::EntityId> m_boundaryElements;
    bool m_hasEquations = true;
    std::int32_t m_interiorEquationCount = 0;
    linalg::DofSelection m_selection;
    linalg::SymbolicFactor m_symbolic;
    linalg::SymmetricCsc m_matrix;
    base::BudgetedBuffer<UnitDofLabel> m_boundaryLabels;
    // Without equations: the local DOF of each boundary equation (UnitBoundaryData::boundaryLocalDofs).
    base::BudgetedBuffer<std::int64_t> m_boundaryLocalDofs;
    base::MatN<6> m_rigidBodyMass{};
    linalg::DenseMatrix m_inertialLoads;
};

// The nodes in the model of a unit's points, increasing: the order of UnitSystem's local points (step 1; its generated
// fastener points follow them). Throws std::invalid_argument for a point the model does not have, or listed twice.
[[nodiscard]] base::BudgetedBuffer<std::int32_t> unitPointNodes(const assembly::DofMap& dofMap,
                                                                const UnitDescription& description,
                                                                base::MemoryBudget& budget);

// The g-set DOF of each local DOF of a unit in UnitSystem's numbering (step 1: node-major, 6 DOFs a grid point and a
// generated fastener point, 1 a scalar point), −1 for a DOF of a generated point: what UnitSystem::modelDofOf gives,
// without the unit built (the patterns and the cache key of a unit come before its equations). Throws as
// unitPointNodes.
[[nodiscard]] base::BudgetedBuffer<base::DofIndex> unitLocalModelDofs(const assembly::DofMap& dofMap,
                                                                      const UnitDescription& description,
                                                                      base::MemoryBudget& budget);

} // namespace oos::reduction
