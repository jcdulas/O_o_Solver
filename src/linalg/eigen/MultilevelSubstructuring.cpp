#include "oos/linalg/eigen/MultilevelSubstructuring.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/base/threading/TaskGraph.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/SmallEigenSolver.hpp"
#include "oos/linalg/eigen/BucklingPencil.hpp"
#include "oos/linalg/eigen/DensePencil.hpp"
#include "oos/linalg/eigen/ShiftStrategy.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/FactorSolve.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{
namespace
{

constexpr double INFINITE = std::numeric_limits<double>::infinity();
// A node eigenvalue μ = 1/λ this small relative to the largest of its node is an infinite root (a massless DOF of a
// lumped mass, a DOF the preload leaves unloaded): not a mode (LanczosRun::NEGLIGIBLE_RITZ_VALUE's reason).
constexpr double NEGLIGIBLE_INVERSE_EIGENVALUE = 1.0e-14;

// The Lanczos solver of a unit: 120 modes a shift (a basis of 2·120 + 21 columns: 25 MB for a unit of 12 000 DOFs) and
// up to 200 shifts, as a unit asks for every mode below the cut-off.
constexpr std::size_t UNIT_MODES_PER_SHIFT = 120;
constexpr std::size_t UNIT_MAXIMUM_SHIFTS = 200;
// A unit whose Lanczos solver did not find every mode below the cut-off takes its dense pencil up to this many DOFs
// (SmallEigenSolver, 9n³ flops: 10 GFLOP at 1 024, a few seconds).
constexpr std::size_t DENSE_FALLBACK_UNIT_DOFS = 1024;

std::size_t toSize(std::int64_t value) noexcept
{
    return static_cast<std::size_t>(value);
}

// A matrix entry gathered from a front: rows and columns of the unit (local) or a boundary row (an equation). The
// right-hand value is the basis's matrix (the nodes' modes are those of (K, it)); the projected value that of the
// matrix projected on them (B = −K_d on the vibration basis of a buckling solve; else zero).
struct Entry
{
    std::int32_t row = 0;
    std::int32_t column = 0;
    double stiffness = 0.0;
    double rightHand = 0.0;
    double projected = 0.0;
};

// The source terms of a unit's supernodes: its interior entries (row >= column, local equations) and its boundary
// entries (row: the global equation beyond the unit; column: a local equation). Bounded by the unit.
struct UnitEntries
{
    std::vector<Entry> interior;
    std::vector<Entry> boundary;
};

// The modes of a node: its vectors (rows: the node's equations, in their order) and eigenvalues.
struct NodeModes
{
    DenseMatrix vectors;
    std::vector<double> eigenvalues; // bounded by the node's modes
};

// The result of a supernode waiting for its parent: the update matrices of K, the basis's matrix and the projected
// one (r × r, lower triangles; the last empty without projection) and the coupling block (r × the modes kept in its
// subtree, in the postorder of their nodes: the reduced pencil's right-hand matrix, the projected one's when there is
// one).
struct Pending
{
    std::int32_t supernode = 0;
    DenseMatrix stiffness;
    DenseMatrix rightHand;
    DenseMatrix projected;
    DenseMatrix coupling;
};

// The front of one supernode for one matrix: the panel (the c columns over the c + r rows) and the trailing block
// (r × r), both lower, square storage.
struct Front
{
    DenseMatrix panel;
    DenseMatrix contribution;

    [[nodiscard]] double& at(std::size_t columns, std::size_t i, std::size_t j) noexcept
    {
        if (i < j)
            std::swap(i, j);
        if (j < columns)
            return panel(i, j);
        return contribution(i - columns, j - columns);
    }
};

// The cut-off of a pass: an eigenvalue (V2, or the second pass of a count: each node keeps its modes below it), or a
// count (the first pass of a count: each node keeps its `count` lowest modes, which make the multilevel substructuring
// subspace in which the count-th root is an upper bound of the true one).
class Cutoff
{
public:
    static Cutoff below(double value)
    {
        Cutoff cutoff;
        cutoff.m_value = value;
        return cutoff;
    }
    static Cutoff lowest(std::size_t count)
    {
        Cutoff cutoff;
        cutoff.m_count = count;
        return cutoff;
    }
    [[nodiscard]] bool isCount() const noexcept { return m_count > 0; }
    [[nodiscard]] std::size_t count() const noexcept { return m_count; }
    [[nodiscard]] double value() const noexcept { return m_value; }

private:
    double m_value = INFINITE;
    std::size_t m_count = 0;
};

// The nodes of the tree for a unit bound (MultilevelSubstructuring::unitsOf) and its shape.
struct Tree
{
    std::vector<std::int32_t> unitOf;       // per supernode
    std::vector<std::int32_t> subtreeFirst; // per supernode
    std::vector<std::vector<std::int32_t>> children;
    std::vector<std::int32_t> unitRoots; // the units' roots, increasing
};

Tree treeOf(const SymbolicFactor& symbolic, std::int64_t unitDofs)
{
    const std::int32_t count = symbolic.supernodeCount();
    Tree tree;
    tree.subtreeFirst.resize(toSize(count));
    tree.children.resize(toSize(count));
    std::vector<std::int64_t> dofs(toSize(count), 0); // per supernode: its subtree's DOFs
    for (std::int32_t s = 0; s < count; ++s)
        tree.subtreeFirst[toSize(s)] = s;
    for (std::int32_t s = 0; s < count; ++s)
    {
        dofs[toSize(s)] += symbolic.columnCount(s);
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (parent == Supernodes::ROOT)
            continue;
        dofs[toSize(parent)] += dofs[toSize(s)];
        tree.subtreeFirst[toSize(parent)] = std::min(tree.subtreeFirst[toSize(parent)], tree.subtreeFirst[toSize(s)]);
        tree.children[toSize(parent)].push_back(s);
    }
    tree.unitOf.assign(toSize(count), -1);
    for (std::int32_t s = count - 1; s >= 0; --s)
    {
        const std::int32_t parent = symbolic.supernodes().parent(s);
        if (parent != Supernodes::ROOT && tree.unitOf[toSize(parent)] >= 0)
        {
            tree.unitOf[toSize(s)] = tree.unitOf[toSize(parent)];
            continue;
        }
        if (dofs[toSize(s)] <= unitDofs)
        {
            tree.unitOf[toSize(s)] = static_cast<std::int32_t>(tree.unitRoots.size());
            tree.unitRoots.push_back(s);
        }
    }
    // The units numbered by their roots' order.
    std::vector<std::int32_t> order(tree.unitRoots.size());
    std::vector<std::int32_t> sortedRoots = tree.unitRoots;
    std::sort(sortedRoots.begin(), sortedRoots.end());
    for (std::size_t u = 0; u < sortedRoots.size(); ++u)
    {
        const auto at = std::find(tree.unitRoots.begin(), tree.unitRoots.end(), sortedRoots[u]);
        order[toSize(at - tree.unitRoots.begin())] = static_cast<std::int32_t>(u);
    }
    for (std::int32_t& unit : tree.unitOf)
        if (unit >= 0)
            unit = order[toSize(unit)];
    tree.unitRoots = std::move(sortedRoots);
    return tree;
}

// A view of a 0-column block (the right-hand sides a front of multilevel substructuring does not have).
DenseMatrixView emptyColumns(std::size_t rows) noexcept
{
    return {nullptr, rows, 0, std::max<std::size_t>(rows, 1)};
}

// C ← C + A·B as C − (−A)·B would need −A: the product's terms subtracted from a zero block, then negated into C.
void addProduct(const DenseKernels& kernels, DenseMatrixView c, ConstDenseMatrixView a, ConstDenseMatrixView b,
                base::MemoryBudget& budget)
{
    DenseMatrix product(budget, c.rows, c.columns, "Substructuring: product");
    kernels.subtractProduct(product.view(), a, b);
    for (std::size_t j = 0; j < c.columns; ++j)
        for (std::size_t i = 0; i < c.rows; ++i)
            c(i, j) -= product(i, j);
}

// C ← Aᵀ·B (C zeroed first).
void setTransposedProduct(const DenseKernels& kernels, DenseMatrixView c, ConstDenseMatrixView a,
                          ConstDenseMatrixView b)
{
    for (std::size_t j = 0; j < c.columns; ++j)
        std::fill_n(&c(0, j), c.rows, 0.0);
    kernels.subtractTransposedProduct(c, a, b);
    for (std::size_t j = 0; j < c.columns; ++j)
        for (std::size_t i = 0; i < c.rows; ++i)
            c(i, j) = -c(i, j);
}

} // namespace

std::vector<std::int32_t> MultilevelSubstructuring::unitsOf(const SymbolicFactor& symbolic, std::int64_t unitDofs)
{
    return treeOf(symbolic, unitDofs).unitOf;
}

SubstructuringPlan MultilevelSubstructuring::plan(const SymbolicFactor& symbolic, const SubstructuringOptions& options,
                                                  std::size_t matrices, std::size_t threads, std::size_t vectors,
                                                  std::uint64_t availableBytes)
{
    constexpr std::uint64_t VALUE = sizeof(double);
    // The modes of one unit the plan allows (its Lanczos solver's two shifts' worth): a unit keeps 0 to 200 on a
    // series of barrel models of growing size.
    constexpr std::uint64_t UNIT_MODE_ALLOWANCE = 2 * UNIT_MODES_PER_SHIFT;
    // A unit's gathered entries at most this share of its panels' values (each (i, j) of its matrices lies in its
    // factor's structure; the barrel's factor holds six to ten times its stiffness's entries, the preflight's
    // stiffness and factor values): one in PANEL_VALUES_PER_ENTRY.
    constexpr std::uint64_t PANEL_VALUES_PER_ENTRY = 4;
    const Tree tree = treeOf(symbolic, options.unitDofs);
    const std::int32_t count = symbolic.supernodeCount();
    const auto k = static_cast<std::uint64_t>(matrices);
    const auto columnsOf = [&](std::int32_t s) { return static_cast<std::uint64_t>(symbolic.columnCount(s)); };
    const auto rowsOf = [&](std::int32_t s) { return static_cast<std::uint64_t>(symbolic.rowCount(s)); };
    // A supernode's fronts of every matrix with its temporaries (X, Yᵀ of each right-hand matrix, B_cc), and the
    // update matrices it leaves.
    const auto frontBytes = [&](std::int32_t s)
    {
        const std::uint64_t c = columnsOf(s);
        const std::uint64_t r = rowsOf(s);
        return VALUE * (k * ((c + r) * c + r * r) + k * c * r + c * c);
    };
    const auto updateBytes = [&](std::int32_t s) { return VALUE * k * rowsOf(s) * rowsOf(s); };

    SubstructuringPlan plan;
    plan.availableBytes = availableBytes;
    std::uint64_t largestRows = 0;  // the rows of one supernode's result, at most (its coupling block's)
    std::uint64_t largestOrder = 0; // the largest front
    // Liu's stack along the postorder of [first, last] (a unit).
    const auto stackPeak = [&](std::int32_t first, std::int32_t last)
    {
        std::vector<std::int32_t> alive; // bounded by the tree's depth
        std::uint64_t bytes = 0;
        std::uint64_t peak = 0;
        for (std::int32_t s = first; s <= last; ++s)
        {
            peak = std::max(peak, bytes + frontBytes(s));
            largestOrder = std::max(largestOrder, columnsOf(s) + rowsOf(s));
            for (const std::int32_t child : tree.children[toSize(s)])
                if (std::find(alive.begin(), alive.end(), child) != alive.end())
                {
                    alive.erase(std::find(alive.begin(), alive.end(), child));
                    bytes -= updateBytes(child);
                }
            alive.push_back(s);
            bytes += updateBytes(s);
            largestRows = std::max(largestRows, rowsOf(s));
        }
        return peak;
    };
    // The units: each its own stack and its eigensolution (gathered entries, factor and pencils, Lanczos workspace and
    // modes), the largest `threads` at once when they run as tasks.
    std::vector<std::uint64_t> unitPeaks; // per unit
    std::uint64_t unitResults = 0;
    for (const std::int32_t root : tree.unitRoots)
    {
        const std::int32_t first = tree.subtreeFirst[toSize(root)];
        std::uint64_t panels = 0;
        std::uint64_t dofs = 0;
        for (std::int32_t s = first; s <= root; ++s)
        {
            panels += columnsOf(s) * (columnsOf(s) + rowsOf(s));
            dofs += columnsOf(s);
        }
        const std::uint64_t entries = panels / PANEL_VALUES_PER_ENTRY * sizeof(Entry);
        const std::uint64_t factor = VALUE * panels + 2 * panels * (sizeof(double) + sizeof(std::int32_t));
        const std::uint64_t lanczos = VALUE * dofs * (2 * UNIT_MODES_PER_SHIFT + 49 + UNIT_MODE_ALLOWANCE);
        unitPeaks.push_back(stackPeak(first, root) + entries + factor + lanczos);
        unitResults += updateBytes(root);
    }
    std::sort(unitPeaks.begin(), unitPeaks.end(), std::greater<>());
    const bool isParallel = threads > 1;
    for (std::size_t u = 0; u < unitPeaks.size() && u < (isParallel ? threads : 1); ++u)
        plan.unitBytes += unitPeaks[u];
    // The separators in postorder after every unit's result (the units' results all alive: an upper bound of both
    // orders).
    std::uint64_t separatorPeak = 0;
    {
        std::uint64_t bytes = unitResults;
        for (std::int32_t s = 0; s < count; ++s)
        {
            if (tree.unitOf[toSize(s)] >= 0)
                continue;
            separatorPeak = std::max(separatorPeak, bytes + frontBytes(s));
            largestOrder = std::max(largestOrder, columnsOf(s) + rowsOf(s));
            for (const std::int32_t child : tree.children[toSize(s)])
                bytes -= updateBytes(child);
            bytes += updateBytes(s);
            largestRows = std::max(largestRows, rowsOf(s));
        }
    }
    plan.stackBytes = std::max(separatorPeak, unitResults);
    plan.vectorBytes = VALUE * static_cast<std::uint64_t>(symbolic.dofCount()) * static_cast<std::uint64_t>(vectors);
    // The spool: a block for each writer at once (the units' tasks and a separator), and in memory
    // its records: the separators' panels, and per mode its node's vector (a unit's DOFs at most).
    std::uint64_t separatorPanels = 0;
    for (std::int32_t s = 0; s < count; ++s)
        if (tree.unitOf[toSize(s)] < 0)
            separatorPanels += VALUE * columnsOf(s) * (columnsOf(s) + rowsOf(s));
    const bool isSpoolInMemory = !options.spool.isOnDisk;
    plan.spoolBytes = static_cast<std::uint64_t>((isParallel ? threads : 1) + 1) * options.spool.blockBytes +
                      (isSpoolInMemory ? separatorPanels : 0);
    plan.fixedBytes = plan.stackBytes + plan.unitBytes + plan.vectorBytes + plan.spoolBytes;
    // Per mode: its coupling blocks (the results alive hold Σ r_i·m_i ≤ max r · m, the modes below each disjoint; the
    // front made its coupling and its result's, order × m each at most), its vector in the spool in memory (a unit's
    // DOFs at most); per mode squared: the reduced right-hand matrix (the dense pencil takes its storage) and the
    // pencil's factor, the Z blocks (half of m² at most) while the matrix is assembled.
    const std::uint64_t largestNode =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(options.unitDofs), largestOrder);
    plan.perModeBytes = VALUE * (largestRows + 2 * largestOrder + (isSpoolInMemory ? largestNode : 0));
    plan.perModeSquaredBytes = VALUE * 5 / 2;
    if (availableBytes > plan.fixedBytes)
    {
        // The largest m with a·m² + b·m ≤ A.
        const double a = static_cast<double>(plan.perModeSquaredBytes);
        const double b = static_cast<double>(plan.perModeBytes);
        const auto room = static_cast<double>(availableBytes - plan.fixedBytes);
        plan.modeAllowance = static_cast<std::size_t>((-b + std::sqrt(b * b + 4.0 * a * room)) / (2.0 * a));
    }
    return plan;
}

namespace
{

// The memory plan of a solve, refused before any work.
SubstructuringPlan planOrRefuse(const SymbolicFactor& symbolic, const SubstructuringOptions& options,
                                std::size_t matrices, std::size_t threads, const EigenRequest& request,
                                const base::MemoryBudget& budget)
{
    const SubstructuringPlan plan =
        MultilevelSubstructuring::plan(symbolic, options, matrices, threads, request.modeCount, budget.available());
    const std::size_t asked = std::max<std::size_t>(request.modeCount, 1);
    if (plan.fixedBytes < plan.availableBytes && plan.modeAllowance >= asked)
        return plan;
    base::MemoryShortfall shortfall;
    shortfall.requestedBytes = plan.bytesWith(asked);
    shortfall.availableBytes = plan.availableBytes;
    shortfall.limitBytes = budget.limit();
    shortfall.largestConsumers = {{"Substructuring: fronts and update matrices", plan.stackBytes},
                                  {"Substructuring: units in flight", plan.unitBytes},
                                  {"Substructuring: scratch spool", plan.spoolBytes},
                                  {"Substructuring: recovered vectors", plan.vectorBytes}};
    throw base::MemoryBudgetError(
        base::concatenate({"Substructuring: the memory plan needs ", base::formatBytes(plan.fixedBytes),
                           " before any mode (fronts and update matrices ", base::formatBytes(plan.stackBytes),
                           ", the units in flight ", base::formatBytes(plan.unitBytes), ", the scratch spool ",
                           base::formatBytes(plan.spoolBytes), ", the recovered vectors ",
                           base::formatBytes(plan.vectorBytes), ") and ", base::formatBytes(plan.perModeBytes),
                           " per mode kept beside the reduced pencil; ", base::formatBytes(plan.availableBytes),
                           " are available, room for ", std::to_string(plan.modeAllowance),
                           " modes; give more "
                           "memory, smaller units (substructuring_unit_dofs=), fewer threads, or ask fewer modes"}),
        shortfall);
}

// A first pass by count (each node its `count` lowest modes) whose modes the plan cannot hold is refused before
// any work: its reduced pencil grows with the nodes (count × nodes), the second pass's with the spectrum.
void refuseLargeFirstPass(const SubstructuringPlan& plan, std::size_t count, std::size_t nodes, bool isBuckling,
                          const base::MemoryBudget& budget)
{
    const std::size_t modes = count * nodes;
    if (modes <= plan.modeAllowance)
        return;
    base::MemoryShortfall shortfall;
    shortfall.requestedBytes = plan.bytesWith(modes);
    shortfall.availableBytes = plan.availableBytes;
    shortfall.limitBytes = budget.limit();
    throw base::MemoryBudgetError(
        base::concatenate({"Substructuring: ", isBuckling ? "the buckling solution's" : "a request by count's",
                           " first pass keeps the ", std::to_string(count), " lowest modes of each of its ",
                           std::to_string(nodes), " nodes, up to ", std::to_string(modes), " modes, above the ",
                           std::to_string(plan.modeAllowance), " the memory plan allows; ",
                           isBuckling
                               ? "use condensation=\"out_of_core\" (exact), give more memory, or larger units "
                                 "(substructuring_unit_dofs=)"
                               : "give a frequency range (an upper bound: one pass), more memory, or larger units "
                                 "(substructuring_unit_dofs=)"}),
        shortfall);
}

// A pass that keeps more modes than the plan allows stops before its reduced pencil.
void checkAllowance(const SubstructuringReport& report, const base::MemoryBudget& budget)
{
    if (report.reducedOrder <= report.plan.modeAllowance)
        return;
    base::MemoryShortfall shortfall;
    shortfall.requestedBytes = report.plan.bytesWith(report.reducedOrder);
    shortfall.availableBytes = report.plan.availableBytes;
    shortfall.limitBytes = budget.limit();
    throw base::MemoryBudgetError(
        base::concatenate(
            {"Substructuring: the upward pass keeps ", std::to_string(report.reducedOrder), " modes (",
             std::to_string(report.unitModes), " of the units, ", std::to_string(report.separatorModes),
             " of the separators), above the ", std::to_string(report.plan.modeAllowance),
             " the memory plan allows; lower the cut-off (substructuring_cutoff=), ask fewer modes, or give more "
             "memory"}),
        shortfall);
}

} // namespace

namespace
{

// One multilevel substructuring pass (its cut-off) and, when kept, its recovery. `rightHand` is the basis's matrix (the
// nodes' modes are those of (K, it): M, or B = −K_d for the buckling basis); `projected`, when given, the matrix of the
// reduced pencil projected on them (B = −K_d on the vibration basis of a buckling solve), the reduced pencil being then
// K̄ = I and B̄ = Tᵀ·B·T.
class SubstructuringPass
{
public:
    SubstructuringPass(const SymbolicFactor& symbolic, FrontSource& stiffness, FrontSource& rightHand,
                       FrontSource* projected, SpectralTransform transform, const DenseKernels& kernels,
                       const SubstructuringOptions& options, const Tree& tree, Cutoff cutoff,
                       base::MemoryBudget& budget)
        : m_symbolic(symbolic),
          m_stiffness(stiffness),
          m_rightHand(rightHand),
          m_projected(projected),
          m_transform(transform),
          m_basisTransform(projected != nullptr ? SpectralTransform::Vibration : transform),
          m_kernels(kernels),
          m_options(options),
          m_tree(tree),
          m_cutoff(std::move(cutoff)),
          m_budget(budget),
          m_spool(options.spool, budget)
    {
        const auto count = toSize(symbolic.supernodeCount());
        m_panelRecords.resize(count);
        m_modeRecords.resize(count);
        m_ownFirstMode.assign(count, 0);
        m_ownModeCount.assign(count, 0);
        m_subtreeFirstMode.assign(count, 0);
        m_couplingBlocks.resize(count);
        m_ownBlocks.resize(count);
        m_nodeEigenvalues.resize(count);
        m_pending.resize(count);
        m_nodeSeconds.assign(count, 0.0);
    }

    void runUpward(SubstructuringReport& report);
    // The reduced pencil's roots and vectors (m × k).
    [[nodiscard]] EigenSolution solveReduced(const EigenRequest& request, SubstructuringReport& report);
    // The vectors on the pencil's equations from the reduced vectors.
    [[nodiscard]] DenseMatrix recover(ConstDenseMatrixView reduced, SubstructuringReport& report);
    [[nodiscard]] std::uint64_t spoolBytes() const noexcept { return m_spool.heldBytes(); }
    [[nodiscard]] bool isSpoolOnDisk() const noexcept { return m_spool.isOnDisk(); }

private:
    struct UnitShape
    {
        std::int32_t first = 0; // its supernodes [first, root]
        std::int32_t root = 0;
        std::int32_t firstColumn = 0;
        std::int32_t columnCount = 0;
        std::int32_t firstPosition = 0;
        std::int32_t positionCount = 0;
    };
    [[nodiscard]] UnitShape unitShape(std::int32_t unit) const;
    // The fronts of supernode s: the stiffness's, and with needsRightHand the basis's and the projected one's.
    void makeFronts(std::int32_t s, Front& stiffness, Front& rightHand, Front& projected, bool needsRightHand);
    // A supernode's fronts, its elimination, its node's modes and couplings, from its children's results; in a unit
    // its source terms are gathered into `entries`.
    [[nodiscard]] Pending processSupernode(std::int32_t s, std::vector<Pending>& children, const DenseKernels& kernels,
                                           ScratchSpool::Writer& writer, const UnitShape* unit, UnitEntries* entries);
    // A unit's supernodes in postorder (its root's result in m_pending), and a separator's.
    void runUnit(std::int32_t unit, const DenseKernels& kernels);
    void runSeparator(std::int32_t s, const DenseKernels& kernels);
    // The congruence of a right-hand front on [Ψ; I] (X = −Ψ): its update matrix and Yᵀ = B_cr − B_cc·X.
    struct Congruence
    {
        DenseMatrix yt;
        DenseMatrix update;
    };
    [[nodiscard]] Congruence congruenceOf(Front& front, const DenseMatrix& x, const DenseKernels& kernels);
    // A node's vibration modes scaled K-orthonormal (the projected reduced pencil's K̄ = I).
    void scaleToStiffness(NodeModes& modes) const;
    void gather(std::int32_t s, const UnitShape& unit, Front& stiffness, Front* rightHand, Front* projected,
                UnitEntries& entries) const;
    // B ← K_ii⁻¹·B on a unit's interior (its equations × columns, in its local order): the unit's stiffness gathered in
    // `entries` factorised (dense when small, else on its own symbolic factor).
    void solveUnitStiffness(const UnitShape& unit, const UnitEntries& entries, DenseMatrixView block,
                            const DenseKernels& kernels);
    // On the vibration basis of a buckling solve, a unit's projected blocks: its own block Φᵀ·B_ii·Φ and its modes'
    // coupling with its boundary rows, B_bi·Φ − K_bi·K_ii⁻¹·(B_ii·Φ), added to `coupling` (rows R(root) × modes).
    void projectUnit(const UnitShape& unit, const UnitEntries& entries, const NodeModes& own, DenseMatrixView coupling,
                     DenseMatrix& ownBlock, const DenseKernels& kernels);
    // The unit's interior pencil in the order of its own symbolic factor (the global order restricted, postordered).
    struct UnitSystem
    {
        SymbolicFactor symbolic;
        SymmetricCsc stiffness;
        SymmetricCsc rightHand;
        std::vector<std::int32_t> symbolicEquationOf; // per local equation (bounded by the unit)
    };
    [[nodiscard]] UnitSystem unitSystem(const UnitShape& unit, const UnitEntries& entries) const;
    [[nodiscard]] NodeModes unitModes(const UnitShape& unit, const UnitEntries& entries, const DenseKernels& kernels);
    // A node's modes of a dense pencil below the cut-off (a separator's below its own: separatorCutoffFactor).
    [[nodiscard]] NodeModes denseModes(ConstDenseMatrixView a, ConstDenseMatrixView b, bool isFactored, double shift,
                                       const DenseKernels& kernels, bool isSeparator = false) const;
    static void spoolMatrix(ConstDenseMatrixView matrix, SpoolRecord& record, ScratchSpool::Writer& writer);
    [[nodiscard]] DenseMatrix readMatrix(const SpoolRecord& record, std::size_t rows, std::size_t columns);
    void recoverUnit(std::int32_t unit, ConstDenseMatrixView reduced, DenseMatrixView vectors,
                     const DenseKernels& kernels);

    const SymbolicFactor& m_symbolic;
    FrontSource& m_stiffness;
    FrontSource& m_rightHand;
    FrontSource* m_projected;
    SpectralTransform m_transform;      // the reduced pencil's
    SpectralTransform m_basisTransform; // the nodes' modes'
    const DenseKernels& m_kernels;
    const SubstructuringOptions& m_options;
    const Tree& m_tree;
    Cutoff m_cutoff;
    base::MemoryBudget& m_budget;
    ScratchSpool m_spool;
    std::vector<SpoolRecord> m_panelRecords; // per supernode (bounded by the supernodes)
    std::vector<SpoolRecord> m_modeRecords;
    std::vector<std::size_t> m_ownFirstMode;
    std::vector<std::size_t> m_ownModeCount;
    std::vector<std::size_t> m_subtreeFirstMode;
    std::vector<DenseMatrix> m_couplingBlocks; // per separator: Z (the modes below it × its own)
    // Per node with projection: its own block of B̄ (its modes × its modes, lower triangle).
    std::vector<DenseMatrix> m_ownBlocks;
    std::vector<double> m_eigenvalues; // per mode kept (bounded by the modes)
    // Per supernode (each written by the task that makes it): its node's eigenvalues, its result for its parent, the
    // seconds of its node's eigensolution.
    std::vector<std::vector<double>> m_nodeEigenvalues;
    std::vector<Pending> m_pending;
    std::vector<double> m_nodeSeconds;
};

SubstructuringPass::UnitShape SubstructuringPass::unitShape(std::int32_t unit) const
{
    UnitShape shape;
    shape.root = m_tree.unitRoots[toSize(unit)];
    shape.first = m_tree.subtreeFirst[toSize(shape.root)];
    shape.firstColumn = m_symbolic.firstColumn(shape.first);
    shape.columnCount = m_symbolic.firstColumn(shape.root) + m_symbolic.columnCount(shape.root) - shape.firstColumn;
    shape.firstPosition = m_symbolic.supernodes().firstPosition(shape.first);
    shape.positionCount = m_symbolic.supernodes().lastPosition(shape.root) + 1 - shape.firstPosition;
    return shape;
}

void SubstructuringPass::makeFronts(std::int32_t s, Front& stiffness, Front& rightHand, Front& projected,
                                    bool needsRightHand)
{
    const auto columns = toSize(m_symbolic.columnCount(s));
    const auto rows = toSize(m_symbolic.rowCount(s));
    stiffness.panel = DenseMatrix(m_budget, columns + rows, columns, "Substructuring: stiffness front");
    stiffness.contribution = DenseMatrix(m_budget, rows, rows, "Substructuring: stiffness front");
    FrontTerms terms;
    terms.supernode = s;
    terms.firstColumn = m_symbolic.firstColumn(s);
    terms.columnCount = static_cast<std::int32_t>(columns);
    terms.rows = m_symbolic.rows(s);
    terms.panel = stiffness.panel.view();
    terms.contribution = stiffness.contribution.view();
    terms.contributionBlockColumns = 0;
    terms.rightHandColumns = emptyColumns(columns);
    terms.rightHandRows = emptyColumns(rows);
    m_stiffness.addTerms(terms);
    if (!needsRightHand)
        return;
    rightHand.panel = DenseMatrix(m_budget, columns + rows, columns, "Substructuring: right-hand front");
    rightHand.contribution = DenseMatrix(m_budget, rows, rows, "Substructuring: right-hand front");
    terms.panel = rightHand.panel.view();
    terms.contribution = rightHand.contribution.view();
    m_rightHand.addTerms(terms);
    if (m_projected == nullptr)
        return;
    projected.panel = DenseMatrix(m_budget, columns + rows, columns, "Substructuring: projected front");
    projected.contribution = DenseMatrix(m_budget, rows, rows, "Substructuring: projected front");
    terms.panel = projected.panel.view();
    terms.contribution = projected.contribution.view();
    m_projected->addTerms(terms);
}

void SubstructuringPass::gather(std::int32_t s, const UnitShape& unit, Front& stiffness, Front* rightHand,
                                Front* projected, UnitEntries& entries) const
{
    const auto columns = toSize(m_symbolic.columnCount(s));
    const std::span<const std::int32_t> rows = m_symbolic.rows(s);
    const std::size_t order = columns + rows.size();
    const std::int32_t firstColumn = m_symbolic.firstColumn(s);
    const auto equationAt = [&](std::size_t p)
    { return p < columns ? firstColumn + static_cast<std::int32_t>(p) : rows[p - columns]; };
    const std::int32_t unitEnd = unit.firstColumn + unit.columnCount;
    for (std::size_t q = 0; q < order; ++q)
    {
        const std::int32_t eq = equationAt(q);
        const bool isQInterior = eq >= unit.firstColumn && eq < unitEnd;
        for (std::size_t p = q; p < order; ++p)
        {
            const double k = stiffness.at(columns, p, q);
            const double b = rightHand != nullptr ? rightHand->at(columns, p, q) : 0.0;
            const double d =
                projected != nullptr && projected->panel.columns() > 0 ? projected->at(columns, p, q) : 0.0;
            if (k == 0.0 && b == 0.0 && d == 0.0)
                continue;
            const std::int32_t ep = equationAt(p);
            const bool isPInterior = ep >= unit.firstColumn && ep < unitEnd;
            if (isPInterior && isQInterior)
                entries.interior.push_back({ep - unit.firstColumn, eq - unit.firstColumn, k, b, d});
            else if (isPInterior)
                entries.boundary.push_back({eq, ep - unit.firstColumn, k, b, d});
            else if (isQInterior)
                entries.boundary.push_back({ep, eq - unit.firstColumn, k, b, d});
        }
    }
}

SubstructuringPass::UnitSystem SubstructuringPass::unitSystem(const UnitShape& unit, const UnitEntries& entries) const
{
    // The unit's nodes are the positions [firstPosition, firstPosition + positionCount): each local equation's node.
    const auto nodeCount = toSize(unit.positionCount);
    std::vector<std::int32_t> columnStart(nodeCount + 1); // local first column of each node
    for (std::size_t v = 0; v <= nodeCount; ++v)
        columnStart[v] =
            m_symbolic.firstColumnOfPosition(unit.firstPosition + static_cast<std::int32_t>(v)) - unit.firstColumn;
    const auto nodeOf = [&](std::int32_t equation)
    {
        const auto at = std::upper_bound(columnStart.begin(), columnStart.end(), equation);
        return static_cast<std::int32_t>(at - columnStart.begin()) - 1;
    };
    std::vector<std::vector<std::int32_t>> adjacency(nodeCount);
    for (const Entry& entry : entries.interior)
    {
        const std::int32_t a = nodeOf(entry.row);
        const std::int32_t b = nodeOf(entry.column);
        if (a == b)
            continue;
        adjacency[toSize(a)].push_back(b);
        adjacency[toSize(b)].push_back(a);
    }
    std::vector<std::int64_t> offsets(nodeCount + 1, 0);
    std::vector<std::int32_t> neighbours;
    for (std::size_t v = 0; v < nodeCount; ++v)
    {
        std::vector<std::int32_t>& list = adjacency[v];
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        neighbours.insert(neighbours.end(), list.begin(), list.end());
        offsets[v + 1] = static_cast<std::int64_t>(neighbours.size());
    }
    const NodeGraph graph =
        NodeGraph::fromAdjacency(static_cast<std::int32_t>(nodeCount), offsets, neighbours, m_budget);
    std::vector<std::int32_t> weights(nodeCount);
    for (std::size_t v = 0; v < nodeCount; ++v)
        weights[v] = columnStart[v + 1] - columnStart[v];
    UnitSystem system;
    system.symbolic = SymbolicFactor::analyse(graph, weights,
                                              NodePermutation::identity(static_cast<std::int32_t>(nodeCount), m_budget),
                                              0, AmalgamationThresholds{}, m_budget);
    const auto equations = toSize(unit.columnCount);
    system.symbolicEquationOf.resize(equations);
    for (std::size_t v = 0; v < nodeCount; ++v)
    {
        const std::int32_t position = system.symbolic.nodeOrder().positionOf(static_cast<std::int32_t>(v));
        const std::int32_t first = system.symbolic.firstColumnOfPosition(position);
        for (std::int32_t e = columnStart[v]; e < columnStart[v + 1]; ++e)
            system.symbolicEquationOf[toSize(e)] = first + (e - columnStart[v]);
    }
    // The entries in the symbolic order, lower triangle, sorted by column then row, duplicates summed in the order
    // gathered (stable sort).
    std::vector<Entry> mapped;
    mapped.reserve(entries.interior.size() + equations);
    for (const Entry& entry : entries.interior)
    {
        std::int32_t row = system.symbolicEquationOf[toSize(entry.row)];
        std::int32_t column = system.symbolicEquationOf[toSize(entry.column)];
        if (row < column)
            std::swap(row, column);
        mapped.push_back({row, column, entry.stiffness, entry.rightHand});
    }
    // Every diagonal present (a zero one included): the factorisation reads the diagonal first in each column.
    for (std::size_t e = 0; e < equations; ++e)
        mapped.push_back({static_cast<std::int32_t>(e), static_cast<std::int32_t>(e), 0.0, 0.0});
    std::stable_sort(mapped.begin(), mapped.end(), [](const Entry& a, const Entry& b)
                     { return a.column < b.column || (a.column == b.column && a.row < b.row); });
    std::vector<std::int64_t> columnOffsets(equations + 1, 0);
    std::vector<std::int32_t> rowIndices;
    std::vector<double> stiffnessValues;
    std::vector<double> rightHandValues;
    for (std::size_t k = 0; k < mapped.size(); ++k)
    {
        const Entry& entry = mapped[k];
        if (k > 0 && mapped[k - 1].row == entry.row && mapped[k - 1].column == entry.column)
        {
            stiffnessValues.back() += entry.stiffness;
            rightHandValues.back() += entry.rightHand;
            continue;
        }
        rowIndices.push_back(entry.row);
        stiffnessValues.push_back(entry.stiffness);
        rightHandValues.push_back(entry.rightHand);
        columnOffsets[toSize(entry.column) + 1] = static_cast<std::int64_t>(rowIndices.size());
    }
    for (std::size_t j = 0; j < equations; ++j)
        columnOffsets[j + 1] = std::max(columnOffsets[j + 1], columnOffsets[j]);
    const auto dimension = static_cast<std::int32_t>(equations);
    system.stiffness =
        SymmetricCsc::fromCompressedColumns(dimension, columnOffsets, rowIndices, stiffnessValues, m_budget);
    system.rightHand =
        SymmetricCsc::fromCompressedColumns(dimension, columnOffsets, rowIndices, rightHandValues, m_budget);
    return system;
}

NodeModes SubstructuringPass::unitModes(const UnitShape& unit, const UnitEntries& entries, const DenseKernels& kernels)
{
    const auto equations = toSize(unit.columnCount);
    const auto dense = [&]
    {
        DenseMatrix a(m_budget, equations, equations, "Substructuring: dense unit");
        DenseMatrix b(m_budget, equations, equations, "Substructuring: dense unit");
        for (const Entry& entry : entries.interior)
        {
            a(toSize(entry.row), toSize(entry.column)) += entry.stiffness;
            b(toSize(entry.row), toSize(entry.column)) += entry.rightHand;
        }
        return denseModes(a.view(), b.view(), false, 0.0, kernels);
    };
    // Small: the dense pencil of the unit.
    if (equations <= toSize(std::max<std::int64_t>(m_options.denseUnitDofs, 0)) || m_options.keepsEveryMode)
        return dense();
    const UnitSystem system = unitSystem(unit, entries);
    FactorOptions factorOptions;
    factorOptions.cancellation = m_options.lanczos.control.cancellation;
    EigenRequest request;
    const bool isBuckling = m_basisTransform == SpectralTransform::Buckling;
    if (m_cutoff.isCount())
        request.modeCount = m_cutoff.count();
    else
    {
        request.upperBound = m_cutoff.value();
        if (isBuckling)
            request.lowerBound = -m_cutoff.value();
    }
    // A unit asks for every mode below the cut-off, often several hundred: more modes a shift (a larger basis, on the
    // unit's small dimension) and more shifts than an analysis's request.
    LanczosOptions unitLanczos = m_options.lanczos;
    unitLanczos.modesPerShift = std::max(unitLanczos.modesPerShift, UNIT_MODES_PER_SHIFT);
    unitLanczos.maximumShifts = std::max(unitLanczos.maximumShifts, UNIT_MAXIMUM_SHIFTS);
    const BlockLanczos lanczos(unitLanczos);
    EigenSolution solution;
    if (isBuckling)
    {
        // BucklingPencil takes K_d (B = −K_d).
        const SymmetricCsc differential =
            SymmetricCsc::combine(0.0, system.rightHand, -1.0, system.rightHand, m_budget);
        BucklingPencil pencil(system.symbolic, system.stiffness, differential, kernels, factorOptions, m_budget);
        solution = lanczos.solve(pencil, request, kernels, m_budget);
    }
    else
    {
        SparsePencil pencil(system.symbolic, system.stiffness, system.rightHand, kernels, factorOptions, m_budget);
        solution = lanczos.solve(pencil, request, kernels, m_budget);
    }
    // A unit asked for most of its modes may exhaust the Lanczos solver's shifts: a small one then takes its dense
    // pencil (the same modes below the cut-off).
    if (!solution.isComplete && equations <= DENSE_FALLBACK_UNIT_DOFS)
        return dense();
    if (!solution.isComplete)
        throw base::SolverError(base::concatenate(
            {"Substructuring: the fixed-interface modes of a unit of ", std::to_string(equations),
             " DOFs below the cut-off were not all found within the Lanczos solver's shifts; lower the cut-off "
             "(substructuring_cutoff=) or use the full path"}));
    NodeModes modes;
    const std::size_t count = solution.eigenvalues.size();
    modes.vectors = DenseMatrix(m_budget, equations, count, "Substructuring: unit modes");
    for (std::size_t j = 0; j < count; ++j)
        for (std::size_t e = 0; e < equations; ++e)
            modes.vectors(e, j) = solution.eigenvectors(toSize(system.symbolicEquationOf[e]), j);
    modes.eigenvalues = solution.eigenvalues;
    return modes;
}

NodeModes SubstructuringPass::denseModes(ConstDenseMatrixView a, ConstDenseMatrixView b, bool isFactored, double shift,
                                         const DenseKernels& kernels, bool isSeparator) const
{
    const double separatorRatio =
        m_options.separatorCutoffFactor > 0.0 ? m_options.separatorCutoffFactor / m_options.cutoffFactor : 1.0;
    const double limit = isSeparator ? m_cutoff.value() * separatorRatio * separatorRatio : m_cutoff.value();
    // (A − σ₀B) = L·D·Lᵀ, then S = D^(−1/2)·L⁻¹·B·L⁻ᵀ·D^(−1/2) = V·diag(μ)·Vᵀ: the roots of (A, B) are λ = 1/μ + σ₀
    // and Φ = L⁻ᵀ·D^(−1/2)·V is (A − σ₀B)-orthonormal (Φᵀ·(A − σ₀B)·Φ = I, Φᵀ·B·Φ = diag(μ)).
    const std::size_t n = a.rows;
    DenseMatrix factor(m_budget, n, n, "Substructuring: node factor");
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j; i < n; ++i)
            factor(i, j) = isFactored ? a(i, j) : a(i, j) - shift * b(i, j);
    if (!isFactored)
    {
        const LdltOutcome outcome = kernels.factorWithoutPivoting(factor.view(), n);
        if (!outcome.isComplete || outcome.inertia.negative > 0)
            throw base::SolverError(base::concatenate(
                {"Substructuring: a node's stiffness of ", std::to_string(n),
                 " DOFs is not positive definite (a part of the model held by nothing below the root, or a "
                 "negative stiffness); constrain the mechanism (single-point constraints) or solve on the full path"}));
    }
    for (std::size_t j = 0; j < n; ++j)
        if (!(factor(j, j) > 0.0))
            throw base::SolverError(base::concatenate({"Substructuring: a node's stiffness of ", std::to_string(n),
                                                       " DOFs has a pivot that is not positive; constrain the "
                                                       "mechanism (single-point constraints) or solve on the "
                                                       "full path"}));
    DenseMatrix g(m_budget, n, n, "Substructuring: node pencil");
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j; i < n; ++i)
        {
            g(i, j) = b(i, j);
            g(j, i) = b(i, j);
        }
    const ConstDenseMatrixView l = factor.view();
    kernels.solveUnitLower(l, g.view()); // L⁻¹·B
    DenseMatrix transposed(m_budget, n, n, "Substructuring: node pencil");
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            transposed(i, j) = g(j, i);           // B·L⁻ᵀ
    kernels.solveUnitLower(l, transposed.view()); // L⁻¹·B·L⁻ᵀ
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j; i < n; ++i)
            transposed(i, j) /= std::sqrt(factor(i, i) * factor(j, j));
    const SymmetricEigenDecomposition decomposition = SmallEigenSolver::decompose(transposed.view(), m_budget);
    double largest = 0.0;
    for (const double mu : decomposition.eigenvalues)
        largest = std::max(largest, std::abs(mu));
    const bool isBuckling = m_basisTransform == SpectralTransform::Buckling;
    // Kept: λ below the cut-off, in increasing λ (vibration: μ decreasing) or increasing |λ| (buckling: |μ|
    // decreasing, the positive first on a tie).
    std::vector<std::size_t> kept;
    for (std::size_t k = 0; k < n; ++k)
    {
        const double mu = decomposition.eigenvalues[k];
        if (std::abs(mu) <= NEGLIGIBLE_INVERSE_EIGENVALUE * largest || (!isBuckling && mu <= 0.0))
            continue;
        const double lambda = 1.0 / mu + shift;
        if (m_options.keepsEveryMode || m_cutoff.isCount() || std::abs(lambda) < limit)
            kept.push_back(k);
    }
    std::stable_sort(kept.begin(), kept.end(),
                     [&](std::size_t x, std::size_t y)
                     {
                         const double mx = decomposition.eigenvalues[x];
                         const double my = decomposition.eigenvalues[y];
                         return std::abs(mx) > std::abs(my) || (std::abs(mx) == std::abs(my) && mx > my);
                     });
    if (m_cutoff.isCount() && !m_options.keepsEveryMode && kept.size() > m_cutoff.count())
        kept.resize(m_cutoff.count());
    NodeModes modes;
    modes.vectors = DenseMatrix(m_budget, n, kept.size(), "Substructuring: node modes");
    for (std::size_t j = 0; j < kept.size(); ++j)
        for (std::size_t i = 0; i < n; ++i)
            modes.vectors(i, j) = decomposition.eigenvectors(i, kept[j]) / std::sqrt(factor(i, i));
    kernels.solveUnitLowerTransposed(l, modes.vectors.view());
    for (std::size_t j = 0; j < kept.size(); ++j)
    {
        const double mu = decomposition.eigenvalues[kept[j]];
        modes.eigenvalues.push_back(1.0 / mu + shift);
        if (!isBuckling)
        {
            // B-orthonormal: φ/√μ (φᵀ·B·φ = μ before).
            const double scale = 1.0 / std::sqrt(mu);
            for (std::size_t i = 0; i < n; ++i)
                modes.vectors(i, j) *= scale;
        }
    }
    return modes;
}

void SubstructuringPass::spoolMatrix(ConstDenseMatrixView matrix, SpoolRecord& record, ScratchSpool::Writer& writer)
{
    std::vector<double> packed(matrix.rows * matrix.columns); // bounded by a node's block
    for (std::size_t j = 0; j < matrix.columns; ++j)
        std::copy_n(matrix.data + j * matrix.leadingDimension, matrix.rows, packed.data() + j * matrix.rows);
    writer.append(packed.data(), packed.size() * sizeof(double), record);
}

DenseMatrix SubstructuringPass::readMatrix(const SpoolRecord& record, std::size_t rows, std::size_t columns)
{
    std::vector<double> packed(rows * columns); // bounded by a node's block
    if (record.bytes != packed.size() * sizeof(double))
        throw base::InternalError(
            "Substructuring: a spooled block has another size than its node's (a bug of O_o_Solver: "
            "report it)");
    m_spool.read(record, packed.data());
    DenseMatrix matrix(m_budget, rows, columns, "Substructuring: spooled block");
    for (std::size_t j = 0; j < columns; ++j)
        std::copy_n(packed.data() + j * rows, rows, &matrix(0, j));
    return matrix;
}

SubstructuringPass::Congruence SubstructuringPass::congruenceOf(Front& front, const DenseMatrix& x,
                                                                const DenseKernels& kernels)
{
    // U = B_rr − Xᵀ·Yᵀ − B_rc·X with Yᵀ = B_cr − B_cc·X (X = −Ψ): [Ψ; I]ᵀ·B·[Ψ; I] expanded.
    const std::size_t columns = x.rows();
    const std::size_t r = x.columns();
    DenseMatrix bcc(m_budget, columns, columns, "Substructuring: right-hand block");
    for (std::size_t j = 0; j < columns; ++j)
        for (std::size_t i = j; i < columns; ++i)
        {
            bcc(i, j) = front.panel(i, j);
            bcc(j, i) = front.panel(i, j);
        }
    const ConstDenseMatrixView brc{front.panel.data() + columns, r, columns, front.panel.leadingDimension()};
    Congruence result;
    result.yt = DenseMatrix(m_budget, columns, r, "Substructuring: right-hand coupling");
    for (std::size_t j = 0; j < r; ++j)
        for (std::size_t i = 0; i < columns; ++i)
            result.yt(i, j) = brc(j, i);
    kernels.subtractProduct(result.yt.view(), bcc.view(), x.view());
    result.update = std::move(front.contribution);
    kernels.subtractTransposedProduct(result.update.view(), x.view(), result.yt.view());
    kernels.subtractProduct(result.update.view(), brc, x.view());
    return result;
}

Pending SubstructuringPass::processSupernode(std::int32_t s, std::vector<Pending>& children,
                                             const DenseKernels& kernels, ScratchSpool::Writer& writer,
                                             const UnitShape* unit, UnitEntries* entries)
{
    m_options.lanczos.control.throwIfCancelled();
    const auto at = toSize(s);
    const auto columns = toSize(m_symbolic.columnCount(s));
    const std::span<const std::int32_t> rows = m_symbolic.rows(s);
    const std::size_t r = rows.size();
    const std::size_t order = columns + r;
    const bool isInUnit = unit != nullptr;
    const bool isUnitRoot = isInUnit && unit->root == s;
    const bool isRoot = m_symbolic.supernodes().parent(s) == Supernodes::ROOT;
    const bool isBuckling = m_basisTransform == SpectralTransform::Buckling;
    const bool isProjected = m_projected != nullptr;

    Front stiffness;
    Front rightHand;
    Front projected;
    makeFronts(s, stiffness, rightHand, projected, true);
    if (isInUnit)
        gather(s, *unit, stiffness, &rightHand, isProjected ? &projected : nullptr, *entries);

    // The children's results, in increasing order: their update matrices extend-added, their coupling blocks side by
    // side (each child's subtree modes after those of the children before it: the postorder of the nodes).
    std::size_t belowModes = 0;
    for (const Pending& child : children)
        belowModes += child.coupling.columns();
    DenseMatrix coupling(m_budget, order, belowModes, "Substructuring: coupling block");
    const std::int32_t firstColumn = m_symbolic.firstColumn(s);
    const auto positionOf = [&](std::int32_t equation)
    {
        if (equation >= firstColumn && equation < firstColumn + static_cast<std::int32_t>(columns))
            return static_cast<std::size_t>(equation - firstColumn);
        const auto found = std::lower_bound(rows.begin(), rows.end(), equation);
        if (found == rows.end() || *found != equation)
            throw base::InternalError(
                "Substructuring: a child's row is not in its parent's front (a bug of O_o_Solver: "
                "report it)");
        return columns + static_cast<std::size_t>(found - rows.begin());
    };
    std::size_t offset = 0;
    for (Pending& child : children)
    {
        const std::span<const std::int32_t> childRows = m_symbolic.rows(child.supernode);
        std::vector<std::size_t> to(childRows.size()); // bounded by a front
        for (std::size_t i = 0; i < childRows.size(); ++i)
            to[i] = positionOf(childRows[i]);
        for (std::size_t j = 0; j < childRows.size(); ++j)
            for (std::size_t i = j; i < childRows.size(); ++i)
            {
                stiffness.at(columns, to[i], to[j]) += child.stiffness(i, j);
                rightHand.at(columns, to[i], to[j]) += child.rightHand(i, j);
                if (isProjected)
                    projected.at(columns, to[i], to[j]) += child.projected(i, j);
            }
        for (std::size_t m = 0; m < child.coupling.columns(); ++m)
            for (std::size_t i = 0; i < childRows.size(); ++i)
                coupling(to[i], offset + m) += child.coupling(i, m);
        offset += child.coupling.columns();
        child = Pending{};
    }

    Pending result;
    result.supernode = s;
    // The roots of a separator node solve their dense pencil unfactorised (a free model's root front is only
    // positive semi-definite); every other front is eliminated (U_K, Ψ).
    const bool isEliminated = !(isRoot && !isInUnit) && r > 0;
    DenseMatrix x;  // X = L_cc⁻ᵀ·L_rcᵀ = −Ψ (c × r)
    DenseMatrix yt; // Yᵀ = B_cr − B_cc·X (c × r) of the coupling's matrix (the projected one when there is one)
    if (isEliminated)
    {
        const LdltOutcome outcome =
            kernels.factorFrontWithoutPivoting(stiffness.panel.view(), stiffness.contribution.view());
        if (!outcome.isComplete || outcome.inertia.negative > 0)
            throw base::SolverError(base::concatenate(
                {"Substructuring: the front of supernode ", std::to_string(s),
                 " is not positive definite (a part of the model held by nothing below the root, or a negative "
                 "stiffness); constrain the mechanism (single-point constraints) or solve on the full path"}));
        const ConstDenseMatrixView lcc{stiffness.panel.data(), columns, columns, stiffness.panel.leadingDimension()};
        const ConstDenseMatrixView lrc{stiffness.panel.data() + columns, r, columns,
                                       stiffness.panel.leadingDimension()};
        x = DenseMatrix(m_budget, columns, r, "Substructuring: static modes");
        for (std::size_t j = 0; j < r; ++j)
            for (std::size_t i = 0; i < columns; ++i)
                x(i, j) = lrc(j, i);
        kernels.solveUnitLowerTransposed(lcc, x.view());
        Congruence basis = congruenceOf(rightHand, x, kernels);
        result.rightHand = std::move(basis.update);
        if (isProjected)
        {
            Congruence carried = congruenceOf(projected, x, kernels);
            result.projected = std::move(carried.update);
            yt = std::move(carried.yt);
        }
        else
            yt = std::move(basis.yt);
        result.stiffness = std::move(stiffness.contribution);
    }

    // The node's own modes.
    NodeModes own;
    base::Stopwatch nodeWatch;
    if (isUnitRoot)
        own = unitModes(*unit, *entries, kernels);
    else if (!isInUnit)
    {
        const ConstDenseMatrixView bcc{rightHand.panel.data(), columns, columns, rightHand.panel.leadingDimension()};
        const ConstDenseMatrixView acc{stiffness.panel.data(), columns, columns, stiffness.panel.leadingDimension()};
        double shift = 0.0;
        if (!isEliminated && !isBuckling)
        {
            // A root: σ₀ = −RIGID_SHIFT_FRACTION·min A_ii/B_ii (ShiftStrategy's rigid shift), below its rigid modes if
            // the model is free.
            double bound = INFINITE;
            for (std::size_t i = 0; i < columns; ++i)
                if (bcc(i, i) > 0.0 && acc(i, i) > 0.0)
                    bound = std::min(bound, acc(i, i) / bcc(i, i));
            shift = -ShiftStrategy::RIGID_SHIFT_FRACTION * (bound < INFINITE ? bound : 1.0);
        }
        own = denseModes(acc, bcc, isEliminated, shift, kernels, true);
    }
    const std::size_t ownCount = own.eigenvalues.size();
    if (isProjected)
        scaleToStiffness(own);
    m_ownModeCount[at] = ownCount;
    m_nodeEigenvalues[at] = own.eigenvalues;
    if (!isInUnit)
    {
        // Z = C_cᵀ·Φ_s: the reduced right-hand matrix between the modes below and the node's own.
        const ConstDenseMatrixView couplingColumns{coupling.data(), columns, belowModes, coupling.leadingDimension()};
        m_couplingBlocks[at] = DenseMatrix(m_budget, belowModes, ownCount, "Substructuring: reduced coupling");
        if (belowModes > 0 && ownCount > 0)
            setTransposedProduct(kernels, m_couplingBlocks[at].view(), couplingColumns, own.vectors.view());
        if (isProjected && ownCount > 0)
        {
            // Its own block Φ_sᵀ·B̃_cc·Φ_s on its transformed pivot block.
            DenseMatrix pcc(m_budget, columns, columns, "Substructuring: projected block");
            for (std::size_t j = 0; j < columns; ++j)
                for (std::size_t i = j; i < columns; ++i)
                {
                    pcc(i, j) = projected.panel(i, j);
                    pcc(j, i) = projected.panel(i, j);
                }
            DenseMatrix product(m_budget, columns, ownCount, "Substructuring: projected block");
            addProduct(kernels, product.view(), pcc.view(), own.vectors.view(), m_budget);
            m_ownBlocks[at] = DenseMatrix(m_budget, ownCount, ownCount, "Substructuring: reduced own block");
            setTransposedProduct(kernels, m_ownBlocks[at].view(), own.vectors.view(), product.view());
        }
        spoolMatrix(own.vectors.view(), m_modeRecords[at], writer);
        spoolMatrix(stiffness.panel.view(), m_panelRecords[at], writer);
    }
    else if (isUnitRoot)
        spoolMatrix(own.vectors.view(), m_modeRecords[at], writer);

    // The coupling carried up: C_r − Xᵀ·C_c, then the own modes' (B_rc − Xᵀ·B_cc)·Φ = (Yᵀ)ᵀ·Φ, or a unit's
    // B_bi·Φ − K_bi·Φ·Λ⁻¹ (its own modes of (K, B)), or B_bi·Φ − K_bi·K_ii⁻¹·(B_ii·Φ) (projected).
    const std::size_t outModes = belowModes + ownCount;
    result.coupling = DenseMatrix(m_budget, r, outModes, "Substructuring: coupling block");
    for (std::size_t m = 0; m < belowModes; ++m)
        for (std::size_t i = 0; i < r; ++i)
            result.coupling(i, m) = coupling(columns + i, m);
    if (isEliminated && belowModes > 0)
    {
        const ConstDenseMatrixView couplingColumns{coupling.data(), columns, belowModes, coupling.leadingDimension()};
        kernels.subtractTransposedProduct(
            DenseMatrixView{result.coupling.data(), r, belowModes, result.coupling.leadingDimension()}, x.view(),
            couplingColumns);
    }
    const DenseMatrixView ownColumns{result.coupling.data() + belowModes * result.coupling.leadingDimension(), r,
                                     ownCount, std::max<std::size_t>(result.coupling.leadingDimension(), 1)};
    if (ownCount > 0 && isUnitRoot && isProjected)
        projectUnit(*unit, *entries, own, ownColumns, m_ownBlocks[at], kernels);
    else if (ownCount > 0 && isUnitRoot && r > 0)
    {
        for (const Entry& entry : entries->boundary)
        {
            const auto found = std::lower_bound(rows.begin(), rows.end(), entry.row);
            if (found == rows.end() || *found != entry.row)
                throw base::InternalError(
                    "Substructuring: a unit's boundary entry is not on its root's rows (a bug of "
                    "O_o_Solver: report it)");
            const auto row = static_cast<std::size_t>(found - rows.begin());
            for (std::size_t m = 0; m < ownCount; ++m)
            {
                const double phi = own.vectors(toSize(entry.column), m);
                result.coupling(row, belowModes + m) +=
                    entry.rightHand * phi - entry.stiffness * phi / own.eigenvalues[m];
            }
        }
    }
    else if (ownCount > 0 && r > 0)
        setTransposedProduct(kernels, ownColumns, yt.view(), own.vectors.view());
    m_nodeSeconds[at] = nodeWatch.lap();
    return result;
}

void SubstructuringPass::scaleToStiffness(NodeModes& modes) const
{
    // Φ M-orthonormal with Φᵀ·K·Φ = Λ: Φ·Λ^(−1/2) is K-orthonormal (K positive definite: SOL 105 asks it).
    for (std::size_t j = 0; j < modes.eigenvalues.size(); ++j)
    {
        const double lambda = modes.eigenvalues[j];
        if (!(lambda > 0.0))
            throw base::SolverError(base::concatenate(
                {"Substructuring: a node's vibration mode has the eigenvalue ", std::to_string(lambda),
                 " (a part of the model free of every constraint: its stiffness is not positive definite); the "
                 "buckling solution asks a constrained model: constrain it (single-point constraints)"}));
        const double scale = 1.0 / std::sqrt(lambda);
        for (std::size_t i = 0; i < modes.vectors.rows(); ++i)
            modes.vectors(i, j) *= scale;
    }
}

void SubstructuringPass::projectUnit(const UnitShape& unit, const UnitEntries& entries, const NodeModes& own,
                                     DenseMatrixView coupling, DenseMatrix& ownBlock, const DenseKernels& kernels)
{
    const auto equations = toSize(unit.columnCount);
    const std::size_t count = own.eigenvalues.size();
    // B_ii·Φ, the interior entries in the order gathered (an entry off the diagonal on both sides).
    DenseMatrix product(m_budget, equations, count, "Substructuring: projected unit");
    for (const Entry& entry : entries.interior)
    {
        if (entry.projected == 0.0)
            continue;
        const auto row = toSize(entry.row);
        const auto column = toSize(entry.column);
        for (std::size_t m = 0; m < count; ++m)
        {
            product(row, m) += entry.projected * own.vectors(column, m);
            if (row != column)
                product(column, m) += entry.projected * own.vectors(row, m);
        }
    }
    ownBlock = DenseMatrix(m_budget, count, count, "Substructuring: reduced own block");
    setTransposedProduct(kernels, ownBlock.view(), own.vectors.view(), product.view());
    if (coupling.rows == 0)
        return;
    // W = K_ii⁻¹·(B_ii·Φ), then B_bi·Φ − K_bi·W on the boundary rows.
    solveUnitStiffness(unit, entries, product.view(), kernels);
    const std::span<const std::int32_t> rows = m_symbolic.rows(unit.root);
    for (const Entry& entry : entries.boundary)
    {
        const auto found = std::lower_bound(rows.begin(), rows.end(), entry.row);
        if (found == rows.end() || *found != entry.row)
            throw base::InternalError(
                "Substructuring: a unit's boundary entry is not on its root's rows (a bug of O_o_Solver: report it)");
        const auto row = static_cast<std::size_t>(found - rows.begin());
        const auto column = toSize(entry.column);
        for (std::size_t m = 0; m < count; ++m)
            coupling(row, m) += entry.projected * own.vectors(column, m) - entry.stiffness * product(column, m);
    }
}

void SubstructuringPass::solveUnitStiffness(const UnitShape& unit, const UnitEntries& entries, DenseMatrixView block,
                                            const DenseKernels& kernels)
{
    const auto equations = toSize(unit.columnCount);
    const std::size_t columns = block.columns;
    if (equations <= toSize(std::max<std::int64_t>(m_options.denseUnitDofs, 0)) || m_options.keepsEveryMode)
    {
        DenseMatrix a(m_budget, equations, equations, "Substructuring: dense unit");
        for (const Entry& entry : entries.interior)
            a(toSize(entry.row), toSize(entry.column)) += entry.stiffness;
        const LdltOutcome outcome = kernels.factorWithoutPivoting(a.view(), equations);
        if (!outcome.isComplete)
            throw base::SolverError(
                "Substructuring: a unit's stiffness is not positive definite; constrain the mechanism (single-point "
                "constraints) "
                "or solve "
                "on the full path");
        kernels.solveUnitLower(a.view(), block);
        kernels.solveBlockDiagonal(blockDiagonalOf(a.view()), block);
        kernels.solveUnitLowerTransposed(a.view(), block);
        return;
    }
    const UnitSystem system = unitSystem(unit, entries);
    FactorOptions options;
    options.cancellation = m_options.lanczos.control.cancellation;
    const SupernodalFactor factor =
        SupernodalFactor::factorise(system.symbolic, system.stiffness, kernels, options, m_budget);
    DenseMatrix permuted(m_budget, equations, columns, "Substructuring: unit solve");
    for (std::size_t j = 0; j < columns; ++j)
        for (std::size_t e = 0; e < equations; ++e)
            permuted(toSize(system.symbolicEquationOf[e]), j) = block(e, j);
    FactorSolve(factor, kernels).solve(permuted.view(), options.cancellation);
    for (std::size_t j = 0; j < columns; ++j)
        for (std::size_t e = 0; e < equations; ++e)
            block(e, j) = permuted(toSize(system.symbolicEquationOf[e]), j);
}

void SubstructuringPass::runUnit(std::int32_t unit, const DenseKernels& kernels)
{
    const UnitShape shape = unitShape(unit);
    UnitEntries entries;
    ScratchSpool::Writer writer(m_spool);
    std::vector<Pending> stack; // Liu's stack of the unit: bounded by its depth
    std::vector<Pending> children;
    for (std::int32_t s = shape.first; s <= shape.root; ++s)
    {
        const std::size_t childCount = m_tree.children[toSize(s)].size();
        children.clear();
        for (std::size_t k = stack.size() - childCount; k < stack.size(); ++k)
            children.push_back(std::move(stack[k]));
        stack.resize(stack.size() - childCount);
        Pending result = processSupernode(s, children, kernels, writer, &shape, &entries);
        if (s == shape.root)
            m_pending[toSize(s)] = std::move(result);
        else
            stack.push_back(std::move(result));
    }
    writer.flush();
}

void SubstructuringPass::runSeparator(std::int32_t s, const DenseKernels& kernels)
{
    std::vector<Pending> children; // bounded by the children
    for (const std::int32_t child : m_tree.children[toSize(s)])
        children.push_back(std::move(m_pending[toSize(child)]));
    ScratchSpool::Writer writer(m_spool);
    m_pending[toSize(s)] = processSupernode(s, children, kernels, writer, nullptr, nullptr);
    writer.flush();
}

// The budget's use and largest consumers into the report when above its earlier samples.
void sampleBudget(const base::MemoryBudget& budget, SubstructuringReport& report)
{
    constexpr std::size_t SAMPLED_CONSUMERS = 12;
    const std::uint64_t used = budget.used();
    if (used <= report.sampledPeakBytes)
        return;
    report.sampledPeakBytes = used;
    report.sampledConsumers = budget.largestConsumers(SAMPLED_CONSUMERS);
}

void SubstructuringPass::runUpward(SubstructuringReport& report)
{
    base::Stopwatch watch;
    const std::int32_t count = m_symbolic.supernodeCount();
    base::ThreadPool* pool = m_kernels.options().pool;
    if (pool != nullptr && pool->threadCount() > 1)
    {
        // PERFORMANCE: the units as tasks of their own (each on one thread with the kernels on that thread, its
        // Lanczos solver too), the separators after their children (the kernels on every thread): the work of each
        // node is the same whatever runs beside it, so the bits are those of one thread.
        DenseKernelOptions serialOptions = m_kernels.options();
        serialOptions.pool = nullptr;
        serialOptions.workspace = nullptr;
        const DenseKernels serial(m_kernels.kind(), serialOptions);
        base::TaskGraph graph;
        std::vector<base::TaskGraph::TaskId> taskOf(toSize(count));
        for (std::size_t u = 0; u < m_tree.unitRoots.size(); ++u)
            taskOf[toSize(m_tree.unitRoots[u])] =
                graph.add([this, u, &serial] { runUnit(static_cast<std::int32_t>(u), serial); });
        for (std::int32_t s = 0; s < count; ++s)
        {
            if (m_tree.unitOf[toSize(s)] >= 0)
                continue;
            std::vector<base::TaskGraph::TaskId> dependencies; // bounded by the children
            for (const std::int32_t child : m_tree.children[toSize(s)])
                dependencies.push_back(taskOf[toSize(child)]);
            taskOf[toSize(s)] = graph.add([this, s] { runSeparator(s, m_kernels); }, dependencies);
        }
        graph.run(pool, m_options.lanczos.control.cancellation);
    }
    else
    {
        for (std::int32_t s = 0; s < count; ++s)
        {
            const std::int32_t unit = m_tree.unitOf[toSize(s)];
            if (unit < 0)
                runSeparator(s, m_kernels);
            else if (m_tree.unitRoots[toSize(unit)] == s)
                runUnit(unit, m_kernels);
        }
    }
    // The modes numbered in the postorder of their nodes.
    m_eigenvalues.clear();
    for (std::int32_t s = 0; s < count; ++s)
    {
        const auto at = toSize(s);
        m_ownFirstMode[at] = m_eigenvalues.size();
        m_eigenvalues.insert(m_eigenvalues.end(), m_nodeEigenvalues[at].begin(), m_nodeEigenvalues[at].end());
        m_nodeEigenvalues[at] = std::vector<double>();
    }
    for (std::int32_t s = 0; s < count; ++s)
        m_subtreeFirstMode[toSize(s)] = m_ownFirstMode[toSize(m_tree.subtreeFirst[toSize(s)])];
    m_pending.clear();
    for (std::size_t u = 0; u < m_tree.unitRoots.size(); ++u)
    {
        const UnitShape shape = unitShape(static_cast<std::int32_t>(u));
        const auto root = toSize(shape.root);
        ++report.unitCount;
        report.unitModes += m_ownModeCount[root];
        report.unitSeconds += m_nodeSeconds[root];
        report.largestUnitDofs = std::max<std::int64_t>(report.largestUnitDofs, shape.columnCount);
        report.largestBoundary = std::max<std::int64_t>(report.largestBoundary, m_symbolic.rowCount(shape.root));
    }
    for (std::int32_t s = 0; s < count; ++s)
    {
        if (m_tree.unitOf[toSize(s)] >= 0)
            continue;
        ++report.separatorCount;
        report.separatorModes += m_ownModeCount[toSize(s)];
        report.largestCouplingValues = std::max<std::int64_t>(
            report.largestCouplingValues,
            static_cast<std::int64_t>((m_ownFirstMode[toSize(s)] - m_subtreeFirstMode[toSize(s)]) *
                                      static_cast<std::size_t>(m_symbolic.columnCount(s) + m_symbolic.rowCount(s))));
    }
    report.reducedOrder = m_eigenvalues.size();
    sampleBudget(m_budget, report);
    report.upwardSeconds += watch.lap();
}

EigenSolution SubstructuringPass::solveReduced(const EigenRequest& request, SubstructuringReport& report)
{
    base::Stopwatch watch;
    const std::size_t m = m_eigenvalues.size();
    const bool isBuckling = m_transform == SpectralTransform::Buckling;
    const bool isProjected = m_projected != nullptr;
    std::vector<double> stiffness(m); // bounded by the modes
    DenseMatrix rightHand(m_budget, m, m, "Substructuring: reduced pencil");
    // K̄ = diag(Λ), B̄'s diagonal I (vibration); K̄ = I, B̄'s diagonal diag(1/λ) (the buckling basis); K̄ = I, B̄'s
    // diagonal blocks the nodes' own blocks (projected).
    for (std::size_t i = 0; i < m; ++i)
    {
        stiffness[i] = isBuckling ? 1.0 : m_eigenvalues[i];
        if (!isProjected)
            rightHand(i, i) = isBuckling ? 1.0 / m_eigenvalues[i] : 1.0;
    }
    for (std::int32_t s = 0; s < m_symbolic.supernodeCount(); ++s)
    {
        const auto at = toSize(s);
        const DenseMatrix& block = m_couplingBlocks[at];
        for (std::size_t j = 0; j < block.columns(); ++j)
            for (std::size_t i = 0; i < block.rows(); ++i)
                rightHand(m_ownFirstMode[at] + j, m_subtreeFirstMode[at] + i) = block(i, j);
        const DenseMatrix& own = m_ownBlocks[at];
        for (std::size_t j = 0; j < own.columns(); ++j)
            for (std::size_t i = j; i < own.rows(); ++i)
                rightHand(m_ownFirstMode[at] + i, m_ownFirstMode[at] + j) = own(i, j);
    }
    m_couplingBlocks.clear();
    m_ownBlocks.clear();
    DensePencil pencil(stiffness, std::move(rightHand),
                       isBuckling ? SpectralTransform::Buckling : SpectralTransform::Vibration, m_kernels, m_budget);
    EigenSolution solution = BlockLanczos(m_options.lanczos).solve(pencil, request, m_kernels, m_budget);
    report.reducedFactorisations += solution.factorisationCount;
    sampleBudget(m_budget, report);
    report.reducedSeconds += watch.lap();
    return solution;
}

void SubstructuringPass::recoverUnit(std::int32_t unit, ConstDenseMatrixView reduced, DenseMatrixView vectors,
                                     const DenseKernels& kernels)
{
    const UnitShape shape = unitShape(unit);
    const auto equations = toSize(shape.columnCount);
    const std::size_t columns = vectors.columns;
    const std::span<const std::int32_t> rows = m_symbolic.rows(shape.root);
    const auto root = toSize(shape.root);
    // The unit's entries again (its stiffness only), then x_i = Φ_u·η_u − K_ii⁻¹·K_ib·x_b.
    UnitEntries entries;
    for (std::int32_t s = shape.first; s <= shape.root; ++s)
    {
        Front stiffness;
        Front unused;
        Front unusedProjected;
        makeFronts(s, stiffness, unused, unusedProjected, false);
        gather(s, shape, stiffness, nullptr, nullptr, entries);
    }
    DenseMatrix interior(m_budget, equations, columns, "Substructuring: unit recovery");
    if (!rows.empty())
    {
        // −K_ib·x_b, the boundary entries in the order gathered.
        for (const Entry& entry : entries.boundary)
            for (std::size_t j = 0; j < columns; ++j)
                interior(toSize(entry.column), j) -= entry.stiffness * vectors(toSize(entry.row), j);
        solveUnitStiffness(shape, entries, interior.view(), kernels);
    }
    // + Φ_u·η_u.
    const std::size_t ownCount = m_ownModeCount[root];
    if (ownCount > 0)
    {
        const DenseMatrix phi = readMatrix(m_modeRecords[root], equations, ownCount);
        const ConstDenseMatrixView eta{reduced.data + m_ownFirstMode[root], ownCount, columns,
                                       reduced.leadingDimension};
        addProduct(kernels, interior.view(), phi.view(), eta, m_budget);
    }
    for (std::size_t j = 0; j < columns; ++j)
        for (std::size_t e = 0; e < equations; ++e)
            vectors(toSize(shape.firstColumn) + e, j) = interior(e, j);
}

DenseMatrix SubstructuringPass::recover(ConstDenseMatrixView reduced, SubstructuringReport& report)
{
    base::Stopwatch watch;
    const auto n = toSize(m_symbolic.dofCount());
    const std::size_t columns = reduced.columns;
    DenseMatrix vectors(m_budget, n, columns, "Substructuring: eigenvectors");
    // The separators top-down, then the units (each reads only its boundary, the separators' rows, and writes only its
    // interior: on the threads, each on one).
    for (std::int32_t s = m_symbolic.supernodeCount() - 1; s >= 0; --s)
    {
        m_options.lanczos.control.throwIfCancelled();
        const auto at = toSize(s);
        if (m_tree.unitOf[at] >= 0)
            continue;
        const auto c = toSize(m_symbolic.columnCount(s));
        const std::span<const std::int32_t> rows = m_symbolic.rows(s);
        const std::size_t r = rows.size();
        DenseMatrix values(m_budget, c, columns, "Substructuring: separator recovery");
        const bool isRoot = m_symbolic.supernodes().parent(s) == Supernodes::ROOT;
        if (!isRoot && r > 0)
        {
            // −L_cc⁻ᵀ·L_rcᵀ·x_r.
            const DenseMatrix panel = readMatrix(m_panelRecords[at], c + r, c);
            DenseMatrix known(m_budget, r, columns, "Substructuring: separator recovery");
            for (std::size_t j = 0; j < columns; ++j)
                for (std::size_t i = 0; i < r; ++i)
                    known(i, j) = vectors(toSize(rows[i]), j);
            const ConstDenseMatrixView lrc{panel.data() + c, r, c, panel.leadingDimension()};
            m_kernels.subtractTransposedProduct(values.view(), lrc, known.view());
            m_kernels.solveUnitLowerTransposed(ConstDenseMatrixView{panel.data(), c, c, panel.leadingDimension()},
                                               values.view());
        }
        if (m_ownModeCount[at] > 0)
        {
            const DenseMatrix phi = readMatrix(m_modeRecords[at], c, m_ownModeCount[at]);
            const ConstDenseMatrixView eta{reduced.data + m_ownFirstMode[at], m_ownModeCount[at], columns,
                                           reduced.leadingDimension};
            addProduct(m_kernels, values.view(), phi.view(), eta, m_budget);
        }
        const auto first = toSize(m_symbolic.firstColumn(s));
        for (std::size_t j = 0; j < columns; ++j)
            for (std::size_t i = 0; i < c; ++i)
                vectors(first + i, j) = values(i, j);
    }
    base::ThreadPool* pool = m_kernels.options().pool;
    DenseKernelOptions serialOptions = m_kernels.options();
    serialOptions.pool = nullptr;
    serialOptions.workspace = nullptr;
    const DenseKernels serial(m_kernels.kind(), serialOptions);
    const bool isParallel = pool != nullptr && pool->threadCount() > 1;
    const DenseMatrixView view = vectors.view();
    base::parallelFor(
        isParallel ? pool : nullptr, m_tree.unitRoots.size(), 1,
        [&](std::size_t first, std::size_t end)
        {
            for (std::size_t u = first; u < end; ++u)
                recoverUnit(static_cast<std::int32_t>(u), reduced, view, isParallel ? serial : m_kernels);
        },
        m_options.lanczos.control.cancellation);
    sampleBudget(m_budget, report);
    report.recoverySeconds += watch.lap();
    return vectors;
}

} // namespace

MultilevelSubstructuring::MultilevelSubstructuring(const SymbolicFactor& symbolic, FrontSource& stiffness,
                                                   FrontSource& rightHand, SpectralTransform transform,
                                                   const DenseKernels& kernels, SubstructuringOptions options,
                                                   FrontSource* mass)
    : m_symbolic(symbolic),
      m_stiffness(stiffness),
      m_rightHand(rightHand),
      m_transform(transform),
      m_kernels(kernels),
      m_options(std::move(options)),
      m_mass(transform == SpectralTransform::Buckling ? mass : nullptr)
{
    if (!(m_options.cutoffFactor >= 1.0))
        throw std::invalid_argument(
            base::concatenate({"Substructuring: the cut-off factor must be at least 1, found ",
                               std::to_string(m_options.cutoffFactor), "; give substructuring_cutoff= of 1 or more"}));
    if (m_options.unitDofs < 1)
        throw std::invalid_argument("Substructuring: the units' DOFs must be at least 1; give a positive unit bound");
}

namespace
{

// The roots and vectors of the kept pass into the solution and its report.
void keepPass(SubstructuringPass& substructuring, EigenSolution reduced, SubstructuringReport passReport,
              SubstructuringSolution& result)
{
    result.solution.eigenvectors = substructuring.recover(reduced.eigenvectors.view(), passReport);
    reduced.eigenvectors = DenseMatrix();
    result.solution.eigenvalues = std::move(reduced.eigenvalues);
    result.solution.residualBounds = std::move(reduced.residualBounds);
    result.solution.shifts = std::move(reduced.shifts);
    result.solution.factorisationCount = reduced.factorisationCount;
    result.solution.verifiedCount = reduced.verifiedCount;
    result.solution.isComplete = reduced.isComplete;
    passReport.spoolBytes = substructuring.spoolBytes();
    passReport.isSpoolOnDisk = substructuring.isSpoolOnDisk();
    result.report = std::move(passReport);
}

// A pass's report from the earlier passes' (their seconds and samples), its counts cleared.
SubstructuringReport nextPassReport(const SubstructuringReport& report, std::size_t pass)
{
    SubstructuringReport next = report;
    next.passes = pass;
    next.unitCount = next.separatorCount = next.unitModes = next.separatorModes = 0;
    return next;
}

} // namespace

SubstructuringSolution MultilevelSubstructuring::solve(const EigenRequest& request, base::MemoryBudget& budget)
{
    const bool isBuckling = m_transform == SpectralTransform::Buckling;
    const bool hasLower = request.lowerBound > -INFINITE;
    // Bounded: below V2 (vibration), or within [V1, V2] both finite (buckling: a side open to infinity is asked by its
    // count).
    const bool hasBound = request.hasUpperBound() && (!isBuckling || hasLower);
    if (request.modeCount == 0 && !hasBound)
        throw std::invalid_argument(
            "Substructuring: the request has neither a mode count nor a bound; give a mode count, or an upper bound");
    const Tree tree = treeOf(m_symbolic, m_options.unitDofs);
    const double factorSquared = m_options.cutoffFactor * m_options.cutoffFactor;
    SubstructuringSolution result;
    SubstructuringReport& report = result.report;
    report.cutoffFactor = m_options.cutoffFactor;
    const base::ThreadPool* pool = m_kernels.options().pool;
    report.plan = planOrRefuse(m_symbolic, m_options, m_mass != nullptr ? 3 : 2,
                               pool != nullptr ? pool->threadCount() : 1, request, budget);
    const std::size_t nodes =
        tree.unitRoots.size() + static_cast<std::size_t>(std::count(tree.unitOf.begin(), tree.unitOf.end(), -1));
    if (!m_options.keepsEveryMode && m_mass != nullptr)
        refuseLargeFirstPass(report.plan, SUBSTRUCTURING_BUCKLING_REFERENCE_MODES, nodes, true, budget);
    else if (!m_options.keepsEveryMode && !hasBound)
        refuseLargeFirstPass(report.plan, request.modeCount, nodes, false, budget);

    if (m_mass != nullptr)
    {
        // Buckling on the vibration basis: a first pass of vibration (each node its
        // SUBSTRUCTURING_BUCKLING_REFERENCE_MODES lowest modes of (K, M)) whose highest root sets λ_c = f²·λ_ref, then
        // the second pass with −K_d projected on the modes below λ_c.
        report.basis = "vibration";
        Cutoff cutoff = Cutoff::below(INFINITE);
        SubstructuringOptions passOptions = m_options;
        if (!(passOptions.separatorCutoffFactor > 0.0))
            passOptions.separatorCutoffFactor = SUBSTRUCTURING_BUCKLING_SEPARATOR_RATIO * m_options.cutoffFactor;
        std::size_t pass = 1;
        if (m_options.keepsEveryMode)
            report.reference = "every mode";
        else
        {
            report.reference = "vibration first pass";
            EigenRequest reference;
            reference.modeCount = SUBSTRUCTURING_BUCKLING_REFERENCE_MODES;
            SubstructuringPass first(m_symbolic, m_stiffness, *m_mass, nullptr, SpectralTransform::Vibration, m_kernels,
                                     m_options, tree, Cutoff::lowest(SUBSTRUCTURING_BUCKLING_REFERENCE_MODES), budget);
            SubstructuringReport firstReport = nextPassReport(report, pass);
            first.runUpward(firstReport);
            checkAllowance(firstReport, budget);
            const EigenSolution roots = first.solveReduced(reference, firstReport);
            const bool hasEveryRoot = roots.eigenvalues.size() >= SUBSTRUCTURING_BUCKLING_REFERENCE_MODES;
            firstReport.referenceEigenvalue = roots.eigenvalues.empty() ? INFINITE : roots.eigenvalues.back();
            report = firstReport;
            cutoff = Cutoff::below(hasEveryRoot ? factorSquared * report.referenceEigenvalue : INFINITE);
            passOptions.keepsEveryMode = !hasEveryRoot;
            ++pass;
        }
        SubstructuringPass substructuring(m_symbolic, m_stiffness, *m_mass, &m_rightHand, m_transform, m_kernels,
                                          passOptions, tree, cutoff, budget);
        SubstructuringReport passReport = nextPassReport(report, pass);
        substructuring.runUpward(passReport);
        checkAllowance(passReport, budget);
        EigenSolution reduced = substructuring.solveReduced(request, passReport);
        passReport.cutoffEigenvalue = cutoff.value();
        keepPass(substructuring, std::move(reduced), std::move(passReport), result);
        return result;
    }

    report.basis = isBuckling ? "buckling" : "vibration";
    Cutoff cutoff = Cutoff::lowest(request.modeCount);
    if (m_options.keepsEveryMode)
    {
        cutoff = Cutoff::below(INFINITE);
        report.reference = "every mode";
    }
    else if (hasBound)
    {
        double reference = request.hasUpperBound() ? std::abs(request.upperBound) : 0.0;
        if (isBuckling && hasLower)
            reference = std::max(reference, std::abs(request.lowerBound));
        cutoff = Cutoff::below(factorSquared * reference);
        report.reference = "V2";
        report.referenceEigenvalue = reference;
    }
    else
        report.reference = "first pass";

    SubstructuringOptions passOptions = m_options;
    for (std::size_t pass = 1;; ++pass)
    {
        SubstructuringPass substructuring(m_symbolic, m_stiffness, m_rightHand, nullptr, m_transform, m_kernels,
                                          passOptions, tree, cutoff, budget);
        SubstructuringReport passReport = nextPassReport(report, pass);
        substructuring.runUpward(passReport);
        checkAllowance(passReport, budget);
        EigenSolution reduced = substructuring.solveReduced(request, passReport);
        if (cutoff.isCount())
        {
            // The first pass: its count-th root, an upper bound of the true one, sets the second pass's cut-off
            // (never too low). A pencil with fewer roots than the count: every one of its nodes' modes.
            const double reference = reduced.eigenvalues.empty() ? INFINITE : std::abs(reduced.eigenvalues.back());
            passReport.referenceEigenvalue = reference;
            report = passReport;
            cutoff =
                Cutoff::below(reduced.eigenvalues.size() < request.modeCount ? INFINITE : factorSquared * reference);
            passOptions.keepsEveryMode = cutoff.value() == INFINITE;
            continue;
        }
        passReport.cutoffEigenvalue = cutoff.value();
        for (const double value : reduced.eigenvalues)
            passReport.cutoffRatios.push_back(cutoff.value() < INFINITE ? std::abs(value) / cutoff.value() : 0.0);
        keepPass(substructuring, std::move(reduced), std::move(passReport), result);
        return result;
    }
}

} // namespace oos::linalg
