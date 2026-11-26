#pragma once

// The eigenvalue items of the sensitivities: EIGN, FREQ and LAMA, a part of the internal computation of
// SensitivityComputation (not a public header; included by the sources of the design module only).

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/ResponseValues.hpp"
#include "oos/design/sensitivity/DesignGradients.hpp"
#include "oos/design/sensitivity/ElementDerivatives.hpp"
#include "oos/design/sensitivity/FieldPerturbation.hpp"
#include "oos/design/sensitivity/PseudoLoads.hpp"
#include "oos/design/sensitivity/ResponseLinearisation.hpp"
#include "oos/design/sensitivity/SensitivitySystem.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/input/tables/ParamTable.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/sol103/ModesSolution.hpp"
#include "oos/solutions/sol105/BucklingSolution.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace oos::design::detail
{

// The derivatives of the eigenvalue responses with respect to the design: the eigenvalues of vibration and the buckling
// load factors, the latter with the derivative of the preload state.
//
// Vibration (EIGN, FREQ). K·φ = λ·M·φ with φᵀ·M·φ = 1: differentiating and multiplying by φᵀ (Fox & Kapoor, "Rates of
// change of eigenvalues and eigenvectors", AIAA J. 6(12), 1968),
//     dλ/dv = φᵀ·(∂K/∂v − λ·∂M/∂v)·φ,
// summed element by element over the elements the designed quantity v reaches (ElementDerivatives: the semi-analytic
// differences of their stiffness and mass facets), then through d(v)/dx. FREQ f = √λ / 2π has
// df/dλ = 1 / (4π·√λ) (8π²·f).
//
// Buckling (LAMA). (K + λ·K_d(u))·φ = 0 with φᵀ·K·φ = 1, so φᵀ·K_d·φ = −1/λ, K_d linear in the preload u = K⁻¹·P:
//     dλ/dv = λ·φᵀ·(∂K/∂v + λ·dK_d/dv)·φ,   dK_d/dv = ∂K_d/∂v|_u + K_d(du/dv).
// The explicit part ∂K_d/∂v|_u is the element's differential stiffness differenced at fixed u; the preload part is an
// item linear in u, λ²·φᵀ·K_d(du)·φ = cᵀ·du with the weights c = λ²·∇_u(φᵀ·K_d(u)·φ) on the g set of the preload
// subcase (formed element by element with the differential stiffness of unit displacements, exact: K_d is linear in
// u), so that the direct, adjoint and chained orders of StaticSensitivity solve it as any static item: the item's
// subcase becomes its preload's.
//
// Repeated roots (the subspace eigenproblem of the derivatives; Haftka & Gürdal, "Elements of
// Structural Optimization", §7.3.2; Seyranian, Lund & Olhoff, Struct. Optim. 8, 1994). The roots of a solve within
// REPEATED_ROOT_TOLERANCE of each other form a cluster Φ (m modes): along one variable x_j the m roots split with the
// derivatives the eigenvalues of
//     D_j = Φᵀ·(∂K/∂x_j − λ·∂M/∂x_j)·Φ          (vibration),
//     D_j = λ·Φᵀ·(∂K/∂x_j + λ·dK_d/dx_j)·Φ     (buckling, the preload part K_d(du_j) by the direct method),
// which are not linear in the direction: they are the one-sided derivatives along +x_j, each variable on its own, the
// sorted eigenvalues given to the cluster's modes in the order of their roots (the smallest derivative to the lowest
// root; for negative load factors, ordered by magnitude, the largest). A cluster of one mode is the formula above.
//
// Determinism: the clusters, the quantities, the elements and the variables in a fixed order, every sum serial.
//
// Thread-safety: none (filled while the sensitivities are linearised). Ownership: borrows the results, the DOF map,
// the element set and the budget, which must outlive it.
class EigenItems
{
public:
    // Two roots of one solve are taken as one repeated root when they differ by less than this share of their
    // magnitude: the Lanczos roots of a symmetric structure's double root agree to about 1e-12, and two distinct roots
    // closer than 1e-6 are, for a derivative, one root split by the design (their own derivatives, valid in a step
    // below their gap, would not survive any finite step).
    static constexpr double REPEATED_ROOT_TOLERANCE = 1.0e-6;
    // The rigid-body roots are rounded zeros of either sign, about 1e-12 of the largest root found: two roots below
    // this share of it are one (repeated) zero root.
    static constexpr double ZERO_ROOT_FRACTION = 1.0e-9;

    struct Inputs
    {
        const solutions::ModesResults* modes = nullptr;
        const solutions::BucklingResults* buckling = nullptr;
        const assembly::DofMap* dofMap = nullptr;
        const assembly::DofMap* modalDofMap = nullptr;
        const assembly::ElementSet* set = nullptr; // the design point's
        // the design point's PARAMs (CoupledMass, WeightToMass of the MODES subcases)
        const input::ParamTable* params = nullptr;
        std::size_t variableCount = 0;
        base::MemoryBudget* budget = nullptr;
        // The design point's snapshot and the pool, for the non-structural mass set a MODES subcase selects (the mass
        // of its derivatives) and for the constraints of the solves under a shape that moves the relations of an
        // element formulation (addFormulationShapeTerms).
        const input::ModelSnapshot* snapshot = nullptr;
        base::ThreadPool* pool = nullptr;
    };

    explicit EigenItems(const Inputs& inputs);

    [[nodiscard]] bool isEmpty() const noexcept { return m_items.empty() && m_shapeItems.empty(); }
    [[nodiscard]] bool hasModalItems() const noexcept { return m_hasModal; }
    [[nodiscard]] bool hasBucklingItems() const noexcept { return m_hasBuckling; }
    [[nodiscard]] std::size_t itemCount() const noexcept { return m_items.size() + m_shapeItems.size(); }
    [[nodiscard]] std::size_t repeatedClusterCount() const noexcept;

    // The static subcase whose displacements the LAMA items of a BUCK subcase read (its preload), or nothing.
    [[nodiscard]] std::optional<std::int32_t> preloadOf(std::int32_t bucklingSubcase) const;

    // Registers an EIGN, FREQ or LAMA item: `label` its subcase, its layer the mode taken (1-based), `value` the
    // evaluation's (checked against the solve's root). Throws OptimisationError when the item does not match the
    // analysis (no such subcase or mode, another root), or for a FREQ of a zero root (no derivative).
    void add(const ItemKey& key, ResponseType type, const ResponseItem& label, double value);
    // Groups the roots of the items into clusters; call once every item is added.
    void formClusters();
    void addBucklingShape(ItemKey key, const ResponseItem& label, double value);
    void prepareBucklingShapes(linalg::KernelKind kernel, bool hasTrim);
    [[nodiscard]] std::size_t shapeFactorisations() const { return m_shapeFactorisations; }
    [[nodiscard]] std::size_t shapeSolves() const { return m_shapeItems.size(); }

    // The preload displacements (and temperatures) of a static subcase, for the differential stiffness at fixed u, and
    // the same displacements on the g set (the static solution's, which must outlive the items).
    void setPreload(std::int32_t subcase, const elements::NodalDisplacements& preload,
                    std::span<const double> displacementsG);
    // The weights c = λ²·∇_u(φᵀ·K_d(u)·φ) of the LAMA items whose cluster has one mode (the others take their preload
    // part by preloadTerms()), on the g set, by item.
    [[nodiscard]] std::vector<std::pair<ItemKey, SparseVector>> bucklingWeights() const;

    // The derivatives of a mass matrix (for a mass formulation) and of a differential stiffness (at a preload) of one
    // element, as a caller differentiates them (the family's closed forms, else the differences of its facets).
    using MassOf = std::function<ElementMatrixDerivative(const elements::MassOptions&)>;
    using DifferentialOf = std::function<ElementMatrixDerivative(const elements::NodalDisplacements&)>;

    // The contribution of one element of `family` a designed quantity reaches: `use` what the quantity changes in it,
    // `stiffness` its dK/dv as the caller differentiated it (empty when the quantity does not change the stiffness),
    // `massOf` and `differentialOf` its dM/dv and dK_d/dv.
    void addElement(std::size_t quantity, const elements::ElementFamily& family, const elements::FieldUse& use,
                    const ElementMatrixDerivative& stiffness, const ElementMatrixDerivative& modalStiffness,
                    const MassOf& massOf, const DifferentialOf& differentialOf);
    // The derivative of every item with respect to one element's own value (a sensitivity map, design/aids): its
    // matrices' derivatives, `stiffness` and those `massOf`, `differentialOf` give, without the preload part of the
    // LAMA items (their weights' adjoint gives it). A repeated vibration root takes the eigenvalues of its subspace
    // problem along that one direction; a repeated buckling root is refused (OptimisationError).
    [[nodiscard]] std::vector<std::pair<ItemKey, double>> elementDerivatives(
        const elements::ElementFamily& family, const elements::FieldUse& use, const ElementMatrixDerivative& stiffness,
        const ElementMatrixDerivative& modalStiffness, const MassOf& massOf, const DifferentialOf& differentialOf);

    // The preload part of the LAMA clusters of several modes (the direct method): for every variable with a
    // pseudo-load in the preload subcase (`loads`, its index `subcase`), du_j = K⁻¹·b_j on `system`, then
    // Φᵀ·K_d(du_j)·Φ. Returns the solves made.
    [[nodiscard]] bool needsPreloadRate(std::int32_t subcase) const;
    void projectPreloadRate(std::int32_t subcase, std::span<const double> displacements, std::size_t column,
                            bool byQuantity);
    std::int64_t addPreloadTerms(std::int32_t preloadSubcase, const SensitivitySystem& system, const PseudoLoads& loads,
                                 std::size_t subcase);

    // A shape (designed quantity `quantity`, its perturbation and the formulation model of each side) that moves the
    // relations of an element formulation (its ties and observer relations, functions of the positions: u = T(v)·u_n).
    // The roots of the reduced problem then change with T too: with D_a = ∂(T·φ_a,n)/∂v at fixed φ_a,n
    // (dependentDerivative), the projection of the cluster along the quantity takes
    //     D_aᵀ·R_b + D_bᵀ·R_a,   R = (K − λ·M)·φ (vibration), (K + λ·K_d)·φ (buckling),
    // the derivative of φ_nᵀ·Tᵀ·A·T·φ_n beyond φᵀ·∂A·φ (R is the force of the relations on the mode: zero off their
    // dependent DOFs); and the Rayleigh parts of a single vibration root 2·D·K·φ and 2·D·M·φ. The constraints of each
    // solve (its constraint set with the formulation's, Kgg, Mgg or K_d) are made at the first shape.
    void addRigidShapeTerms(std::size_t quantity, std::span<const assembly::LocalConstraintRows> rates);
    void addFormulationShapeTerms(std::size_t quantity, const FieldPerturbation& perturbation,
                                  const std::array<const elements::FormulationModel*, FIELD_SIDE_COUNT>& models);

    // The explicit parts of the items, appended to `linearised` (sparse rows over the independent variables):
    // `quantities` the designed quantities with d(v)/dx (the fields, then the shapes, as PseudoLoads indexes them).
    void addExplicitParts(const std::vector<FieldValue>& quantities, LinearisedItems& linearised) const;
    // The Rayleigh quotient parts of the EIGN and FREQ items of single roots among `wanted` (their response ID and item
    // index by key): ∇(φᵀ·K·φ) and ∇(φᵀ·M·φ) over the independent variables.
    [[nodiscard]] std::vector<RayleighGradient> rayleighParts(
        const std::vector<FieldValue>& quantities,
        const std::map<ItemKey, std::pair<std::int32_t, std::size_t>>& wanted) const;

private:
    // A solve of the items: a MODES subcase's (its roots, shapes and scales) or a BUCK subcase's.
    struct Solve
    {
        bool isBuckling = false;
        std::int32_t subcase = 0;
        std::int32_t preloadSubcase = 0;
        std::vector<double> roots; // λ in the solve's order (bounded by the modes found)
        const linalg::DenseMatrix* shapes = nullptr;
        const std::vector<double>* scales = nullptr; // φ_normalised = shapes(:, m) / scales[m]
        // The mass formulation of a MODES subcase; the added non-structural mass is per element batch: massOptionsOf.
        elements::MassOptions massOptions;
        double weightToMass = 1.0;
    };
    struct Item
    {
        ItemKey key;
        ResponseType type = ResponseType::Eigenvalue;
        std::size_t solve = 0;
        std::size_t mode = 0;    // 0-based column of the solve
        std::size_t cluster = 0; // index into m_clusters
        std::size_t rank = 0;    // its place in the cluster
        double scale = 1.0;      // dr/dλ
    };
    struct Cluster
    {
        std::size_t solve = 0;
        std::vector<std::size_t> modes; // the solve's columns, in the order of the roots
        double root = 0.0;              // λ (their mean)
        // Per designed quantity, Φᵀ·(∂K/∂v − λ·∂M/∂v)·Φ or Φᵀ·(∂K/∂v + λ·∂K_d/∂v|_u)·Φ (m × m row-major), empty when
        // the quantity reaches none of its elements; per variable the preload part Φᵀ·K_d(du_j)·Φ of a buckling
        // cluster of several modes.
        std::map<std::size_t, std::vector<double>> byQuantity;
        std::map<std::size_t, std::vector<double>> preloadByVariable;
        // A vibration root of one mode: per designed quantity φᵀ·∂K/∂v·φ and φᵀ·∂M/∂v·φ apart (the Rayleigh quotient).
        std::map<std::size_t, std::pair<double, double>> energiesByQuantity;
    };

    [[nodiscard]] double shapeValue(const Solve& solve, std::size_t mode, base::DofIndex dof) const;
    // Reads the non-structural mass set the MODES subcase `subcase` selects (NonStructMassSelection; the subcases of
    // one solve select the same) and keeps the mass it adds to each element for the solve `solveIndex` (m_addedMass):
    // the modal solution assembles its mass with it, so the pencil K − λ·M of the derivatives must too.
    void readAddedMass(std::size_t solveIndex, std::int32_t subcase);
    // The mass options of an element of `family` for a modal solve: its formulation and the mass the selected set adds
    // to the rows of the element's batch (per unit length or area: the same on every side of a perturbation, each
    // side multiplying it by its own length or area, so that dM_e/dv carries the share's change with a shape).
    [[nodiscard]] elements::MassOptions massOptionsOf(std::size_t solveIndex,
                                                      const elements::ElementFamily& family) const;
    // Φᵀ·A·Φ of an element matrix A (row-major over `dofs`) into `target` (m × m), scaled by `factor`.
    void addProjection(const Cluster& cluster, std::span<const base::DofIndex> dofs, std::span<const double> matrix,
                       double factor, std::vector<double>& target) const;
    [[nodiscard]] std::vector<base::DofIndex> dofsOf(std::span<const elements::ElementDof> dofs,
                                                     bool modal = false) const;
    // The share of the element of batch row `row` in the weights of the LAMA items (bucklingWeights) from its
    // family's product differentialStiffnessPreloadGradient, `shapes` the items' columns as displacements; false
    // (nothing added) when the family has none.
    bool addProductWeights(const elements::ElementFamily& family, const elements::ElementBatchView& view,
                           std::int64_t row, std::span<const Item* const> items,
                           std::span<const elements::NodalDisplacements> shapes,
                           std::vector<std::vector<std::pair<base::DofIndex, double>>>& weights) const;
    // Projects one element's derivatives on every cluster: Φᵀ·(dK − λ·dM)·Φ (vibration) or Φᵀ·(dK + λ·dK_d)·Φ
    // (buckling) added to targetOf(cluster), and the strain and kinetic energy rates of the single vibration roots
    // given to energiesOf(cluster, strain, kinetic) when it is set.
    using TargetOf = std::function<std::vector<double>&(std::size_t)>;
    using EnergiesOf = std::function<void(std::size_t, double, double)>;
    void projectElement(const elements::ElementFamily& family, const elements::FieldUse& use,
                        const ElementMatrixDerivative& stiffness, const ElementMatrixDerivative& modalStiffness,
                        const MassOf& massOf, const DifferentialOf& differentialOf, const TargetOf& targetOf,
                        const EnergiesOf& energiesOf);

    // The constraints of a solve for addFormulationShapeTerms: its analysis model and constraint set, and per mode
    // column K·φ and M·φ (vibration) or K_d·φ (buckling) on the g set; no system when the formulation has no relation
    // in that set.
    struct SolveConstraints
    {
        std::unique_ptr<solutions::AnalysisModel> model;
        std::unique_ptr<solutions::ConstrainedSystem> system;
        std::map<std::size_t, std::pair<std::vector<double>, std::vector<double>>> products;
    };
    [[nodiscard]] SolveConstraints& constraintsOf(std::size_t solve);
    // φ of a mode on the g set, normalised.
    [[nodiscard]] std::vector<double> normalisedShape(const Solve& solve, std::size_t mode) const;

    struct ShapeItem
    {
        ItemKey key;
        const solutions::BucklingSolve* solve;
        std::size_t mode;
        base::DofIndex dof;
        std::span<const double> shape;
        base::BudgetedBuffer<double> adjoint;
        double metricMultiplier = 0.0;
        std::map<std::size_t, double> byQuantity, byVariable;
    };
    [[nodiscard]] double shapeProduct(const ShapeItem& item, std::span<const base::DofIndex> dofs,
                                      std::span<const double> matrix, bool metric = false) const;
    std::vector<ShapeItem> m_shapeItems;
    bool m_hasTrim = false;
    std::size_t m_shapeFactorisations = 0;
    Inputs m_in;
    std::vector<Solve> m_solves;
    std::vector<Item> m_items;
    std::vector<Cluster> m_clusters;
    std::map<std::int32_t, elements::NodalDisplacements> m_preloads;
    std::map<std::int32_t, std::span<const double>> m_preloadVectors;
    std::map<std::size_t, SolveConstraints> m_solveConstraints; // by solve
    // By modal solve whose subcase selects a non-structural mass set: the mass it adds to the rows of each batch of
    // the design point's element set (Inputs::set).
    std::map<std::size_t, assembly::NonStructuralMassShares> m_addedMass;
    bool m_hasModal = false;
    bool m_hasBuckling = false;
};

} // namespace oos::design::detail
