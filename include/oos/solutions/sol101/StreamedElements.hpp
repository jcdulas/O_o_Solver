#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::solutions
{

// The g-set DOFs of an element (or a point, or a load) carried onto the equations of the full path streamed:
// u_dof = Σ coefficient · x_equation, the independent DOFs one term each, a DOF of the m set its row of Gm on
// the n set (u_m = Gm·u_n, so K_red = Tᵀ·K·T and f_red = Tᵀ·f give the usual
// Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm and Pn = P̄n + Gmᵀ·Pm element by element), the held DOFs (zero: the
// streamed path holds no enforced value) without a term. `equations` are the distinct equations reached, ascending;
// each term names the position of its DOF in the caller's list and the position of its equation in `equations`.
struct StreamedReduction
{
    struct Term
    {
        std::int32_t dof = 0;      // position in the DOFs reduced
        std::int32_t equation = 0; // position in `equations`
        double coefficient = 0.0;
    };
    std::vector<std::int32_t> equations; // bounded by the element's DOFs and their relations
    std::vector<Term> terms;
};

// What the full path streamed places on its fronts: every
// element of the model and every point whose displacements are an output, each at the supernode that eliminates its
// first equation (through the relations: an element on the reference point of a spider is placed by the legs its Gm
// row reaches). Every equation such an item reaches is then in the front of that supernode (the item's nodes, extended
// by the relations' couplings, are a clique of the node graph the symbolic factor was made on, MsetCouplings), so the
// front takes the whole item: its stiffness and loads (FrontSource), and, in the downward pass, its displacements.
//
// The elements are numbered over the batches of the element set (batch b's rows from batchFirst(b)); an element whose
// DOFs reach no equation (every DOF held) is placed at the root. A point that reaches no equation (held) too: its
// displacements are zero.
//
// Thread-safety: immutable once built; const methods are safe concurrently. Ownership: owns its placement (budgeted);
// borrows the model, Gm, the selection and the symbolic factor, which must outlive it.
class StreamedElements
{
public:
    struct Inputs
    {
        const AnalysisModel* model = nullptr;
        const assembly::RigidElimination* gm = nullptr;
        const linalg::DofSelection* selection = nullptr; // the equations (the l set, in the factor's order)
        const linalg::SymbolicFactor* symbolic = nullptr;
        std::span<const std::uint8_t> isOutputNode; // per node of the DOF map: its displacements are asked for
    };
    // An element's batch (an index of ElementSet::batches()) and row.
    struct Place
    {
        std::size_t batch = 0;
        std::int64_t row = 0;
    };

    // Throws MemoryBudgetError.
    StreamedElements(const Inputs& inputs, base::MemoryBudget& budget);

    [[nodiscard]] std::int64_t elementCount() const noexcept { return m_batchFirst.back(); }
    [[nodiscard]] std::int64_t batchFirst(std::size_t batch) const noexcept { return m_batchFirst[batch]; }
    [[nodiscard]] Place placeOf(std::int32_t element) const noexcept;
    // The elements placed at supernode s, increasing, and the points (nodes of the DOF map).
    [[nodiscard]] std::span<const std::int32_t> elementsAt(std::int32_t s) const noexcept;
    [[nodiscard]] std::span<const std::int32_t> pointsAt(std::int32_t s) const noexcept;
    // The nodes of an element (ElementConnectivity's: those of its matrices), empty for an element without any.
    [[nodiscard]] std::span<const std::int32_t> nodesOf(std::int32_t element) const noexcept;
    // The supernode whose columns hold an equation.
    [[nodiscard]] std::int32_t supernodeOf(std::int32_t equation) const noexcept;
    // The g-set DOFs of a node of the DOF map (six for a grid point, one for a scalar point), into `dofs`.
    void nodeDofs(std::int32_t node, std::vector<base::DofIndex>& dofs) const;
    // StreamedReduction of g-set DOFs.
    void reduce(std::span<const base::DofIndex> dofs, StreamedReduction& out) const;
    // The supernode of the first equation of a reduction, or the root when it has none.
    [[nodiscard]] std::int32_t placeOf(const StreamedReduction& reduction) const noexcept;

    [[nodiscard]] const AnalysisModel& model() const noexcept { return *m_inputs.model; }
    [[nodiscard]] const linalg::DofSelection& selection() const noexcept { return *m_inputs.selection; }
    [[nodiscard]] const linalg::SymbolicFactor& symbolic() const noexcept { return *m_inputs.symbolic; }
    // The bytes of its arrays (the memory plan's).
    [[nodiscard]] std::uint64_t bytes() const noexcept;
    // What the placement of a model of `elements` elements, `points` points asked for and `supernodes` supernodes
    // holds, for the memory plan before the placement is made.
    [[nodiscard]] static std::uint64_t estimateBytes(std::int64_t elements, std::int64_t points,
                                                     std::int64_t supernodes, std::size_t batches) noexcept;

private:
    Inputs m_inputs;
    std::vector<std::int64_t> m_batchFirst; // batches + 1 (bounded by the card types)
    // Per element: its group of ElementConnectivity (>= 0), its single-node element (−2 − index), or −1 (none).
    base::BudgetedBuffer<std::int32_t> m_nodeSource;
    base::BudgetedBuffer<std::int64_t> m_elementOffsets; // supernodes + 1
    base::BudgetedBuffer<std::int32_t> m_elements;
    base::BudgetedBuffer<std::int64_t> m_pointOffsets; // supernodes + 1
    base::BudgetedBuffer<std::int32_t> m_points;
};

} // namespace oos::solutions
