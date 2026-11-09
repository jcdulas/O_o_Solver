#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/ElementResultTable.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/QuantityMap.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/output/envelopes/EnvelopeAccumulator.hpp"
#include "oos/solutions/sol101/KeptLoadBasis.hpp"
#include "oos/solutions/sol101/StreamedElements.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::solutions
{

// The outputs of the full path streamed, made region by region (linalg::SolutionSink): as the
// downward pass solves a supernode, the displacements of the nodes of the elements and points placed there
// (StreamedElements) are made from the front's solution (u = T·x through Gm, zero on the held DOFs) for every vector of
// the basis and gathered in a chunk; once the chunk holds its bound, it is handed over: each case's displacements the
// basis's combination (u_c = Σ_b α(b, c)·u_b), the displacement rows of its points and the element tables of its
// elements (output::ElementResultBuilder::buildWindow: the rows the whole model's recovery gives them) offered to the
// envelopes and, when the case tables are kept, joined to them; then the chunk is released. Nothing of a region is
// kept once its chunk is out.
//
// A free body solved on conditioned supports is brought back to the gauge of its FreeBodySupport DOFs as the full path
// brings it (u ← u − D·u_r, D = Φ·C on the independent DOFs, the conditioned supports included, zero on the held ones,
// Gm on the dependent ones): each chunk node's DOFs keep their row of D before C (Inputs::gaugeRowOf), and the motion
// is removed once the FreeBodySupport DOFs' values are known (the order puts them in the first supernodes the downward
// pass solves, or a refined solution gives them first, presetSupportValues; a chunk waits for them), from the
// displacement rows only: the element tables are made from the displacements as solved on the conditioned supports,
// whose element results do not depend on the FreeBodySupport DOFs.
//
// The case tables are made in the order of the full path's (by element ID within an element type, by node), so that the
// results are the full path's tables, to round-off.
//
// Thread-safety: take() on one thread (the downward pass's); the recovery of a chunk on the pool. Ownership: owns its
// chunk and the tables (budgeted); borrows what Inputs names, which outlives it.
class StreamedOutputs final : public linalg::SolutionSink
{
public:
    struct Inputs
    {
        const StreamedElements* elements = nullptr;
        const output::ElementResultBuilder* builder = nullptr;
        const linalg::DenseMatrix* coefficients = nullptr; // α: basis × cases
        std::span<const base::EntityId> caseLabels;       // the cases, in α's order
        const output::OutputPlan* plan = nullptr;          // the plan every case shares
        output::EnvelopeAccumulator* envelopes = nullptr;  // null: none
        bool keepsCaseTables = true;                       // the tables of the kinds no envelope takes are kept anyway
        // Free bodies on conditioned supports: the FreeBodySupport DOFs and C (rigidSupportCoefficients: column j at
        // 6·j).
        std::span<const base::DofIndex> supports;
        std::optional<std::array<double, 36>> supportCoefficients;
        // The row of D before C at a g-set DOF (StreamedFront::reliefRowOf), with the supports.
        std::function<std::array<double, 6>(base::DofIndex)> gaugeRowOf;
        // The most values a chunk holds (its nodes' displacements of every vector) before it is handed over.
        std::int64_t chunkValues = 0;
        // The load basis with declared quantities: the vectors are the cases (α the identity,
        // their displacements taken as they are), the force tables of the map's elements (the describers of
        // `basisSelection`) reduced by the map as each chunk makes them, and the grid point forces of the elements at
        // the nodes `isForceNode` flags gathered for every vector; `fixedRows` (its loads those of the vectors) gives
        // the fixed rows at its nodes (BasisFixedRows). Null map and empty flags: the outputs of the cases.
        const output::QuantityMap* basisQuantities = nullptr;
        output::ElementOutputSelection basisSelection;
        std::span<const std::uint8_t> isForceNode;
        const KeptFixedRows* fixedRows = nullptr;
        base::ThreadPool* pool = nullptr;
        const base::CancellationToken* cancellation = nullptr;
    };

    StreamedOutputs(const Inputs& inputs, base::MemoryBudget& budget);

    void take(std::int32_t supernode, linalg::ConstDenseMatrixView columnValues,
              linalg::ConstDenseMatrixView rowValues) override;
    // The FreeBodySupport DOFs' values given before any take() (supports × vectors: a refined solution handed over from
    // a store): the chunks then go out at their bound whatever the order. Throws std::invalid_argument for another
    // shape.
    void presetSupportValues(linalg::ConstDenseMatrixView values);
    // Hands over the last chunk and makes the case tables. Throws std::logic_error when the FreeBodySupport DOFs were
    // never solved (a bug).
    void finish();

    // The tables of each case (finish() made them).
    [[nodiscard]] std::vector<output::NodalResultTable>& nodalTables(std::size_t c) { return m_nodal[c]; }
    [[nodiscard]] std::vector<output::ElementResultTable>& elementTables(std::size_t c) { return m_element[c]; }
    // The load basis with declared quantities: what finish() made of the vectors (KeptStreamedBasis), moved out.
    [[nodiscard]] std::shared_ptr<KeptStreamedBasis> takeBasis();
    // The chunks handed over and the largest (values).
    [[nodiscard]] std::int64_t chunkCount() const noexcept { return m_chunkCount; }
    [[nodiscard]] std::int64_t largestChunkValues() const noexcept { return m_largestChunkValues; }
    // The bytes a chunk of `values` values and `cases` cases holds while it is handed over (the memory plan's).
    [[nodiscard]] static std::uint64_t chunkBytes(std::int64_t values, std::size_t columns, std::size_t cases) noexcept;
    // The values of a chunk whose chunkBytes() is about `bytes` (at least one node's).
    [[nodiscard]] static std::int64_t chunkValuesFor(std::uint64_t bytes, std::size_t columns,
                                                     std::size_t cases) noexcept;

private:
    // The nodes of an item into the chunk, their displacements from the front's solution.
    void addNode(std::int32_t node, std::int32_t supernode, linalg::ConstDenseMatrixView columnValues,
                 linalg::ConstDenseMatrixView rowValues);
    void flush();
    [[nodiscard]] bool isGaugeReady() const noexcept;

    Inputs m_inputs;
    base::MemoryBudget& m_budget;
    std::size_t m_columns = 0;
    std::size_t m_caseCount = 0;
    bool m_hasStress = false;
    bool m_asksDisplacements = false;
    // The chunk.
    base::BudgetedBuffer<std::int32_t> m_slotOfNode;  // per node of the DOF map: its slot in the chunk, or −1
    base::BudgetedBuffer<std::int32_t> m_gridSlots;   // per grid point row: its slot in a window, or −1
    base::BudgetedBuffer<std::int32_t> m_scalarSlots; // per scalar point row
    std::vector<std::int32_t> m_slotNodes;            // per slot (bounded by the chunk)
    std::vector<double> m_values;                     // per slot: its DOFs (6, or 1) × the vectors, DOF-major
    std::vector<double> m_gauge;                      // per slot: its DOFs × 6 (Σ coefficient·Φ)
    std::vector<std::int64_t> m_valueOffset;          // per slot
    std::vector<std::vector<elements::ElementRange>> m_rowsOfBatch; // per batch: the chunk's elements' rows
    std::vector<std::int32_t> m_points;                             // the output points of the chunk
    // The FreeBodySupport DOFs' values (per vector), once solved.
    std::vector<std::vector<double>> m_supportValues;
    std::vector<std::uint8_t> m_isSupportSolved;
    // The load basis with declared quantities.
    bool m_isBasis = false;
    std::vector<std::vector<elements::ElementRange>>
        m_forceRowsOfBatch;                // per batch: the chunk's elements at force nodes
    std::vector<std::string> m_forceNames; // the grid point forces' names
    std::vector<std::vector<output::StiffnessRecovery::PointContribution>> m_contributions; // per vector
    std::shared_ptr<KeptStreamedBasis> m_basis;
    // The tables kept, per case.
    std::vector<std::vector<output::NodalResultTable>> m_nodal;
    std::vector<std::vector<output::ElementResultTable>> m_element;
    std::int64_t m_chunkCount = 0;
    std::int64_t m_largestChunkValues = 0;
};

} // namespace oos::solutions
