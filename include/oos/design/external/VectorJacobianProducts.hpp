#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/external/ExternalEvaluator.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/responses/BatchCallable.hpp"
#include "oos/design/responses/equation/EquationResponse.hpp"
#include "oos/design/sensitivity/DesignGradients.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// The weights of one response in compressed sparse rows (rows × items): the entries of row r at
// [rowStart[r], rowStart[r + 1]), their items strictly increasing within the row. The form of a reserve factor
// constraint, whose row touches the items of one zone only.
struct SparseWeights
{
    std::vector<std::size_t> rowStart; // rows + 1
    std::vector<std::size_t> items;
    std::vector<double> values;
};

// Weights on the items of responses for a vector-Jacobian product: per response ID, `rows` rows of one weight per item
// of the response, the same rows for every response; dense (`weights`: rows × items, row-major) or sparse
// (`sparseWeights`), one form per response. The two forms give the same bits for the same non-zero weights: the
// products read the non-zero weights of a row in the order of the responses and of their items either way.
struct ProductWeights
{
    std::size_t rows = 0;
    std::map<std::int32_t, std::vector<double>> weights;
    std::map<std::int32_t, SparseWeights> sparseWeights;
};

// What a jvp computes (for instance, the members of a stiffened panel model linearised in a bay's components on the
// loads of the bay's points): some responses, some of their items, some subcases. Empty (the default): every item of
// every response in every subcase, as before. A selection that is not empty makes the call restricted: its rows are
// the items of the selected responses only (in the order of the evaluation), the items it leaves out NaN; it
// linearises only the items
// it computes (the responses an equation of them reads included) and only the designed fields that the directions'
// variables drive (the cost of a linearisation is that of the fields' pseudo-loads: some 4.3 s of 4.8 s for the 923
// variables of the demo's model, against a few variables a bay); it solves only the subcases its items are in. Each
// item computed has the bits of the unrestricted call.
struct ProductSelection
{
    // The responses (IDs) whose items are returned; empty: every response, or those of `itemMasks` when given.
    std::vector<std::int32_t> responseIds;
    // Per selected response some of its items (one flag per item); a response without a mask: every item.
    std::vector<ItemMask> itemMasks;
    // The subcases (labels) whose items are computed; empty: every subcase. An item of no subcase (a global item, no
    // displacement) is computed whatever the subcases.
    std::vector<std::int32_t> subcases;

    [[nodiscard]] bool isEmpty() const noexcept { return responseIds.empty() && itemMasks.empty() && subcases.empty(); }
};

// One block of products handed to the sink: rows [first, first + count) of W·dr/dx (count × variables) for a vjp; the
// directions [first, first + count) of dr/dx·V (items × count, the items of every evaluated response in the order of
// the evaluation, or of the selected responses of a restricted jvp) for a jvp. Row-major; valid during consume() only.
struct ProductBlock
{
    std::size_t first = 0;
    std::size_t count = 0;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::span<const double> values;
};

// Where the blocks go (a plug-in point, called once per block on the calling thread); an exception it throws stops the
// computation and goes to the caller.
class ProductSink
{
public:
    ProductSink() = default;
    ProductSink(const ProductSink&) = delete;
    ProductSink& operator=(const ProductSink&) = delete;
    virtual ~ProductSink() = default;

    virtual void consume(const ProductBlock& block) = 0;
};

// `solves`: the right-hand sides solved (a row of a vjp counts once per subcase it weights items of; the rows of a
// block that weight no item of a subcase are not solved on its factor). `weightsRead`: the weights the products of a
// vjp read, each time they read one (every entry of a dense row, the stored entries of a sparse row; once for the
// explicit parts and once per subcase): a sparse row on the items of one zone reads none of the other zones'
// `lineariseSeconds`: the wall time of the linearisation of the items (their derivatives with respect to
// the displacements and the explicit parts, the pseudo-loads of the variables); `productSeconds`: the rest (the
// right-hand sides, the solves, the products), the two parts of the cost, which scale differently.
//
// With a kept linearisation (ProductLinearisation), `isLinearisationReused` says whether the call took the
// linearisation an earlier call at the same evaluation made (its `lineariseSeconds` then only the check that it
// covers the call), and `linearisationBytes` the memory the kept linearisation holds in the budget (0 when it is not
// kept: no ProductLinearisation given, or the budget could not hold it).
struct ProductReport
{
    std::int64_t solves = 0;
    std::int64_t blocks = 0;
    std::int32_t factorisations = 0;
    std::int64_t weightsRead = 0;
    double lineariseSeconds = 0.0;
    double productSeconds = 0.0;
    // The parts of productSeconds a vjp's solves take (timed around each phase on the calling thread): the
    // right-hand sides, the solves, the combination of shared item adjoints (item_adjoints) and the adjoint vectors'
    // expansion and dots with the pseudo-loads.
    double rightHandSideSeconds = 0.0;
    double solveSeconds = 0.0;
    double combineSeconds = 0.0;
    double dotSeconds = 0.0;
    // The two largest parts of a linearisation made by the call (SensitivityReport::itemSeconds, fieldSeconds; 0 when
    // the call took a kept one).
    double lineariseItemSeconds = 0.0;
    double lineariseFieldSeconds = 0.0;
    bool isLinearisationReused = false;
    std::uint64_t linearisationBytes = 0;
    // The Hessian-vector products: `solves` counts δu and δλ, two per subcase; the adjoint of Σ yᵢ·rᵢ is
    // solved once per weights (`adjointSolves`, 0 when it was kept from a product with the same weights:
    // `isAdjointReused`); the linearisations at the perturbed states (two per product).
    std::int64_t adjointSolves = 0;
    bool isAdjointReused = false;
    std::int32_t perturbedLinearisations = 0;
    // The element Hessian blocks of (x, y): whether the product was made from them (no perturbed
    // linearisation), whether they were kept from a product with the same weights, their count, memory and making
    // time (0 when kept), or why the closed forms do not cover the weights (the products then difference the
    // linearisation).
    bool hasHessianBlocks = false;
    bool usesModalTangents = false; // analytic vibration root/field curvature from bordered tangent and adjoint solves
    bool usesTrimTangents = false; // analytic trim curvature from mixed state and coefficient tangents
    bool usesBucklingTangents = false; // analytic load-factor curvature including the mechanical preload
    bool areHessianBlocksReused = false;
    std::int64_t hessianBlockCount = 0;
    std::uint64_t hessianBlockBytes = 0;
    double hessianBlockSeconds = 0.0;
    std::string hessianBlocksReason;
};

// A sparse matrix as triplets (row, column, value), rows × columns; a reader sums the triplets of an entry.
struct SparseTriplets
{
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<std::int64_t> row;
    std::vector<std::int64_t> column;
    std::vector<double> value;

    void add(std::int64_t r, std::int64_t c, double v)
    {
        row.push_back(r);
        column.push_back(c);
        value.push_back(v);
    }
};

// The structural support of a sparse matrix: the entries its structure allows, whatever the values
// of a design point (the node graph, the elements the variables' fields reach, the items' elements), each (row, column)
// at most once; a superset of the non-zero entries of the matrix it supports. A reader builds the pattern of the
// matrix from it, so that the pattern is the same at every point of the same structure.
struct SparseSupport
{
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<std::int64_t> row;
    std::vector<std::int64_t> column;

    void add(std::int64_t r, std::int64_t c)
    {
        row.push_back(r);
        column.push_back(c);
    }
};

// The blocks of one constraint set: the subcases of the set share them.
struct ConstraintSetBlocks
{
    // The lower triangle of Kgg (g × g), the stiffness the analysis factorised (its diagonal blocks' lower halves).
    SparseTriplets stiffness;
    // Every entry of Kgg's lower blocks (the node graph's), zeros included: the support of `stiffness` and
    // of the subcases' `hessianDisplacement`, in that order.
    SparseSupport stiffnessSupport;
    // T (g × equations): u_g = T·u_l + u_e, the l set to the g set (1 on the l DOFs, Gm on the m set, 0 on the s and
    // r sets), so that K_ll = Tᵀ·Kgg·T and the loads of the equations are Tᵀ·P_g (SensitivitySystem).
    SparseTriplets transform;
    // Per equation (column of T, row and column of K_ll) its g-set DOF.
    std::vector<std::int64_t> equationDofs;
    // Whether T was made from the l DOFs and Gm (checked against the solver's expansion), not column by column (the
    // fallback, O(equations · g)).
    bool isTransformFromGm = false;
};

// The blocks of one subcase.
struct SubcaseBlocks
{
    std::int32_t subcase = 0;
    std::size_t set = 0; // its constraint set in DesignBlocks::sets
    // B (g × variables): b_j = ∂P/∂x_j − ∂K/∂x_j·u, the pseudo-loads; the residual R = K·u − P has ∂R/∂x = −Tᵀ·B on
    // the equations.
    SparseTriplets pseudoLoads;
    // The structural support of B and of `hessianMixed`: per variable the DOFs of the nodes of the
    // elements its fields reach, with B's non-zero entries.
    SparseSupport pseudoLoadSupport;
    // With weights y (DesignBlocks::hasHessian): the second derivatives of the Lagrangian
    // ℒ = Σᵢ yᵢ·rᵢ − Σ_s λ_sᵀ·(K·u_s − P_s) on the g set, from the element Hessian blocks of (x, y) (HessianBlocks):
    // Σ_e A_uu on Kgg's lower blocks (one value per entry of the set's `stiffnessSupport`, in its order, zeros
    // included: the lower entry of each pair), Σ_e A_up·∂p/∂x (g × variables, one value per entry of
    // `pseudoLoadSupport`, zeros included) and the adjoint λ_s (g values; empty when no weighted item reads the
    // subcase).
    std::vector<double> hessianDisplacement;
    std::vector<double> hessianMixed;
    std::vector<double> adjoint;
};

// The blocks of a design point, for a partner that assembles its own system: with the l-set equations of each subcase,
// K_ll·u_l = Tᵀ·(P − K·u_e), every item r_i(u, x), J = ∂g/∂p − ∂g/∂u·K⁻¹·∂R/∂p with ∂g/∂u = C·T (C the items'
// weights on the g set, `itemWeights`), ∂R/∂p = −Tᵀ·B and ∂g/∂p the explicit parts. The items in the order of the
// evaluation's responses and of their items (jvp's rows).
//
// The supports are the matrices' structural patterns: unions of the non-zero entries with what the
// structure allows (an element item over its element's nodes, an SPC force item over its point's neighbours in the node
// graph, an element item's explicit part over the variables of its element's fields); the other items' supports are
// their non-zero entries.
//
// With weights y (`hasHessian`): the subcases' second derivatives (SubcaseBlocks) and `hessianDesign` (variables ×
// variables, both triangles: Σ_e A_pp·(∂p/∂x)ᵀ·(∂p/∂x), the relations linear in x; one value per entry of
// `designSupport`, the pairs of variables that drive a common field, zeros included), and `hessianReport` (the
// blocks' count, memory and time, the adjoint's solves, whether the adjoint and the blocks were kept).
struct DesignBlocks
{
    std::vector<ConstraintSetBlocks> sets;
    std::vector<SubcaseBlocks> subcases;
    SparseTriplets itemWeights;             // items × g
    SparseSupport itemWeightSupport;        // items × g
    std::vector<std::int32_t> itemSubcases; // per item its subcase (0: a global item, no displacement)
    SparseTriplets explicitParts;           // items × variables
    SparseSupport explicitPartSupport;      // items × variables
    bool hasHessian = false;
    SparseSupport designSupport;       // variables × variables
    std::vector<double> hessianDesign; // per entry of designSupport
    ProductReport hessianReport;
    // The largest |A_uu(d, e) − A_uu(e, d)| of the element blocks relative to their largest |A_uu| (0 without
    // displacement blocks): the round-off of their symmetry, the lower entry of each pair being the one kept.
    double hessianAsymmetry = 0.0;
    // Per g-set DOF its point ID and component (1 to 6 for a grid point, 0 for a scalar point).
    std::vector<std::int32_t> dofPoints;
    std::vector<std::int8_t> dofComponents;
};

// What the products read besides the evaluation: the callables and external evaluators of the equation and external
// responses (their partials), the kernel and threads of a factorisation made again, the items of a batch call.
struct ProductContext
{
    const CallableRegistry* callables = nullptr;
    const ExternalRegistry* externals = nullptr;
    solutions::StaticOptions analysisOptions;
    std::size_t blockItems = EQUATION_BATCH_ITEMS;
    // Whether a Hessian-vector product may take the element Hessian blocks (false: the differences of the
    // linearisation along v, the reference the blocks are checked against).
    bool usesHessianBlocks = true;
    // The values and gradients of the relations given by functions at a design point (the moved states of a
    // Hessian-vector product), called on the calling thread; empty: such relations are refused.
    std::function<std::vector<SuppliedField>(std::span<const double>)> suppliedAt;
    // A vjp's adjoint vectors of the items shared by the rows: in a block, the
    // distinct derivative vectors (with respect to the displacements) of the items its rows weight on a factor of the
    // full path solved once each when they are fewer than the rows' (subcase, row) columns, each column's adjoint
    // vector their combination; equal to the rows' own adjoints to round-off, not their bits. False: one solve per
    // column. The condensed path, inertia relief and one row of weights (its adjoint vectors kept) solve the columns.
    bool sharesItemAdjoints = false;
};

// The linearisation of the items of one evaluation kept between products: the
// items' derivatives with respect to the displacements and their explicit parts, the pseudo-loads of the variables,
// the subcases with their constraint sets. The linearisation is the whole cost of a product at a point (some 2 ms
// per shell element and subcase, against microseconds for the solves on the kept factor), so an optimiser's many
// products at one point (an interior point method's Hessian-vector and Jacobian products) linearise once.
//
// The holder of the evaluation keeps it beside the evaluation and drops it with it (a new point, clear_analysis(), the
// responses evaluated again): VectorJacobianProducts checks that a kept linearisation is that of the evaluation, of
// the same problem and batch size, and of the responses the call reads (else it linearises again, the responses of
// the earlier call included, and keeps the new one). Its memory is reserved in the budget of the evaluation, so the
// kept analysis reports it (DesignAnalysis.memory_bytes); one that does not fit is not kept. Reusing it gives the bits
// of a call that linearises: the products read the same linearised items in the same order.
//
// Thread-safety: none (one product at a time). Ownership: owns the linearisation; borrows the problem and the
// evaluation, which must outlive it (or clear() it before they go).
class ProductLinearisation
{
public:
    ProductLinearisation();
    ProductLinearisation(const ProductLinearisation&) = delete;
    ProductLinearisation& operator=(const ProductLinearisation&) = delete;
    ProductLinearisation(ProductLinearisation&&) noexcept;
    ProductLinearisation& operator=(ProductLinearisation&&) noexcept;
    ~ProductLinearisation();

    // Drops the kept linearisation and gives its memory back.
    void clear() noexcept;
    [[nodiscard]] bool isEmpty() const noexcept { return m_state == nullptr; }
    // The bytes reserved for it in the evaluation's budget (0 when empty).
    [[nodiscard]] std::uint64_t bytes() const noexcept;

    struct State; // the linearisation, its inputs and its memory (VectorJacobianProducts.cpp)

private:
    friend class VectorJacobianProducts;
    std::unique_ptr<State> m_state;
};

// The vector-Jacobian and Jacobian-vector products of a design point: problem.vjp(x, weights) exposes the chained
// adjoint to any external computation, W·dr/dx with one adjoint solve per row of weights whatever the number of items
// and of variables; problem.jvp(x, directions) returns dr/dx·V for every response with one solve per direction. Both
// work by blocks and with a sink, as the full gradient mode does.
//
// Every item is linearised as for the gradients (StaticSensitivity: c = ∂r/∂u, the explicit part ∂r/∂x, the chained
// weights of the equations and the external items), then, with b_j the pseudo-loads of the variables and K the
// factorised constraint set of a subcase:
//     vjp: row r of W·dr/dx = Σ_i W_ri·∂r_i/∂x + λ_rᵀ·b,   λ_r = K⁻¹·Σ_{i of the subcase} W_ri·c_i   (one solve per row
//          and subcase that has weighted items);
//     jvp: dr_i/dx·v = ∂r_i/∂x·v + c_i·δu,   δu = K⁻¹·Σ_j v_j·b_j                                     (one solve per
//          direction and subcase).
// The items that failed or were not evaluated are excluded: their weights are ignored and their rows of a
// jvp are NaN. The products are formed block by block (`block` rows or directions), each block handed to the sink:
// the derivatives of the displacements are never stored. The right-hand sides of a block and subcase are solved
// together on its factor (linalg::FactorSolve: each column has the bits it has alone), so the products do not
// depend on `block`; a vjp solves only the rows that weight items of the subcase.
//
// `kept`: the linearisation kept with the evaluation, used when it is the evaluation's and covers the
// call, else made and kept there (null: linearised for this call alone, as before). `budget` must be the budget of
// the evaluation when a linearisation is kept.
//
// Thread-safety: run on the calling thread (the callables and evaluators are called from it). Ownership: none kept
// but in `kept`.
class VectorJacobianProducts
{
public:
    // Throws std::invalid_argument for a block of 0, weights of a response the evaluation does not hold, of the
    // wrong size, given in both forms, or sparse with row starts that do not run from 0 to the entries, an item out
    // of range or items not increasing within a row; throws as StaticSensitivity.
    [[nodiscard]] static ProductReport vjp(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                           const ProductWeights& weights, std::size_t block, ProductSink& sink,
                                           base::MemoryBudget& budget, const ProductContext& context,
                                           ProductLinearisation* kept = nullptr);
    // `directions`: variables × count, row-major (one column per direction). Throws as vjp(), and
    // std::invalid_argument for a selection of a response the evaluation does not hold or a mask of the wrong size.
    //
    // `selection`: a restricted call (ProductSelection) takes the kept linearisation when it is the
    // evaluation's and holds the selected responses (a kept one holds every variable's pseudo-loads); else it
    // linearises what it computes for itself alone (the items selected, the fields of the directions' variables) and
    // leaves the kept one in place: a partial linearisation is never kept, so that a later product along other
    // directions or on other responses never reads one.
    [[nodiscard]] static ProductReport jvp(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                           std::span<const double> directions, std::size_t count, std::size_t block,
                                           ProductSink& sink, base::MemoryBudget& budget, const ProductContext& context,
                                           ProductLinearisation* kept = nullptr,
                                           const ProductSelection& selection = {});

    // H·v of the weighted sum Σᵢ yᵢ·rᵢ of the responses' items: `weights` one
    // row (the multipliers y, dense or sparse per response, as vjp takes them), `direction` v (one value per
    // independent variable); `values` gets H·v (one value per variable). The second-order adjoint (Haftka and Gürdal,
    // Elements of Structural Optimization, 3rd ed., §7.3; Nocedal and Wright, Numerical Optimization, 2nd ed., §8.4),
    // with K(x)·u = P(x), K·λ = Σ yᵢ·cᵢ (cᵢ = ∂rᵢ/∂u), b_j = ∂P/∂x_j − ∂K/∂x_j·u and the gradient
    // g_j = Σ yᵢ·∂rᵢ/∂x_j + λᵀ·b_j: along x + σ·v, u moves by σ·δu (δu = K⁻¹·Σ_j v_j·b_j) and λ by σ·δλ, with
    // K·δλ = d/dσ (Σ yᵢ·cᵢ) − (∂K/∂v)·λ, so that
    //     (H·v)_j = d/dσ [Σ yᵢ·∂rᵢ/∂x_j + λᵀ·b_j](x + σ·v, u + σ·δu) + δλᵀ·b_j,
    // the derivatives in σ of the linearisation's own terms at the states (x ± t·v, u ± t·δu), one central difference
    // (HESSIAN_STEP_RELATIVE): the items not linear in u (von Mises) enter by their own curvature, through their
    // weights cᵢ at the moved displacements. Two solves per subcase and product (δu and δλ); λ once per weights, kept
    // with the linearisation for the next products at the same (x, y); two linearisations at the moved states per
    // product. Also covered: the section and cut loads; inertia relief (the linearisation at a moved state
    // takes the relief of the moved point); the condensed path (the solves through the boundaries, the units
    // read recovered); the relations given by functions (ProductContext::suppliedAt); SUBCOM subcases; the equation
    // responses whose callable has hessianVector (their weights y moved onto their arguments, μ = Σ y·∂f/∂L, the
    // callable's (w_x, w_L) at (v, δL) added to the product and Σ w_L·∂L/∂u to the right-hand side of δλ: still two
    // solves per subcase). Only the subcases a weight reads are solved; λ is taken from a vjp of the same one row at
    // the same evaluation (kept with the linearisation). Throws std::invalid_argument for more than one row or a
    // direction of the wrong size, and OptimisationError for what requireSecondOrder refuses (the external responses,
    // an equation without hessian_vector, the other extension kinds, shape variables, SAERO subcases).
    //
    // `count` directions at once: `direction` variables × count, row-major (one column per
    // direction), `values` variables × count the same way; the adjoint and the blocks once, then each column as the
    // product of that direction alone (its bits), the solves counted together.
    [[nodiscard]] static ProductReport hessianVector(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                     const ProductWeights& weights, std::span<const double> direction,
                                                     std::vector<double>& values, base::MemoryBudget& budget,
                                                     const ProductContext& context,
                                                     ProductLinearisation* kept = nullptr, std::size_t count = 1);

    // The sparse blocks of the evaluation: Kgg, T, the pseudo-loads, the items' weights and explicit
    // parts, from the kept linearisation of every response (made when it is not), with their structural supports.
    // The full path only; refused (OptimisationError): the condensed path, external units, inertia
    // relief, SAERO subcases, shape variables.
    //
    // With `weights` (one row, as hessianVector takes them): also the second
    // derivatives of the Lagrangian of (x, y) (SubcaseBlocks::hessian*, DesignBlocks::hessianDesign), from the element
    // Hessian blocks (HessianBlocks), the same objects hessianVector applies: made by the first of the two at these
    // weights and kept with the linearisation for the other, with the adjoint λ. Refused (OptimisationError) where the
    // closed forms stop (the reason hessianVector reports in hessianBlocksReason: its differences of the linearisation
    // give products, not blocks), for the equation responses of the second order (their callables give products) and
    // for what hessianVector refuses; throws std::invalid_argument as hessianVector.
    [[nodiscard]] static DesignBlocks blocks(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                             base::MemoryBudget& budget, const ProductContext& context,
                                             ProductLinearisation* kept = nullptr,
                                             const ProductWeights* weights = nullptr);

private:
    // The products `compute` makes on the linearisation of the responses `read` (every one when `readsEvery`), kept
    // or made (with `restricted`, made of those options for the call alone when the kept one does not cover it).
    template <typename Products>
    [[nodiscard]] static ProductReport products(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                bool readsEvery, const std::set<std::int32_t>& read,
                                                base::MemoryBudget& budget, const ProductContext& context,
                                                ProductLinearisation* kept, const Products& compute,
                                                const GradientOptions* restricted = nullptr);
};

} // namespace oos::design
