#pragma once

#include "FactorSchedule.hpp"
#include "FrontAssembly.hpp"
#include "WorkBuffer.hpp"
#include "WorkspacePool.hpp"

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/KernelWorkspace.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/ScratchSpool.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::linalg
{

class FrontSource;

// The numeric multifrontal factorisation behind SupernodalFactor and PartialFactor (internal; the method is described
// on SupernodalFactor). One object per factorisation.
//
// Working storage: the bound of FactorSchedule is reserved once, up front; the WorkBuffers it covers are measured
// live (WorkspaceMeter) and the measured peak is reported with the factor, so that tests check the bound against an
// independent measure. Their blocks come from a WorkspacePool of that bound, which keeps the large ones released for
// the next fronts. The bound assumes at most threadCount subtrees at once; a thread outside the pool helping the
// subtree batch (ThreadPool lets a caller waiting for an older batch help newer ones) would run one more, so a subtree
// that starts while threadCount are running reserves its own front and stack in the budget instead (admission: the
// bound stays an upper bound of what the reservation covers, whatever calls the pool).
//
// The matrix is a SymmetricCsc, or a FrontSource (the pencils from the elements: each front's own terms
// added by the source, assembleFrontInPhases, in indefinite mode only: the positive definite factorisation of a source
// is StreamedSolve's, and its MAXRATIO diagonal is not read here).
//
// Thread-safety: run() once; the tasks it starts write only the supernodes they own. Ownership: references its
// arguments, which outlive it.
class MultifrontalDriver
{
public:
    MultifrontalDriver(const SymbolicFactor& symbolic, const SymmetricCsc& matrix, const DenseKernels& kernels,
                       const FactorOptions& options, bool excludesTrailing, base::MemoryBudget& budget);
    // Throws std::invalid_argument for options that are not of indefinite mode. The source's addTerms() is called
    // concurrently for distinct supernodes (the subtrees' tasks).
    MultifrontalDriver(const SymbolicFactor& symbolic, FrontSource& source, const DenseKernels& kernels,
                       const FactorOptions& options, base::MemoryBudget& budget);

    // Allocates the panels of `factor`, reserves the working storage, factorises, then checks the pivots. With
    // excludesTrailing, the trailing front is assembled into `schurComplement` (both triangles).
    void run(SupernodalFactor& factor, DenseMatrix* schurComplement);
    // The panels written to `spool` as they are made (indefinite mode): each supernode's
    // record in records[s] (its eliminated columns over its rows, as a panel holds them), no panel value kept in the
    // factor, its rows, subdiagonal and arrays kept. Before run(); the spool and records outlive it.
    void writePanelsTo(ScratchSpool& spool, std::span<SpoolRecord> records) noexcept
    {
        m_spool = &spool;
        m_records = records;
    }

private:
    enum class Status : std::uint8_t
    {
        Pending,
        Done,
        ZeroPivot, // positive definite mode: a zero or non-finite pivot; the front stops there
        Skipped    // a descendant has a zero pivot
    };

    // The update matrix of a subtree root or a top supernode, waiting for its parent.
    struct PendingUpdate
    {
        WorkBuffer values;
        std::int32_t order = 0;
        std::vector<std::int32_t> delayedEquations;
    };

    // An update matrix on the stack of a subtree, in its own buffer (a positive definite front leaves it there; it is
    // handed to the parent, then to the pending slot of a subtree root, without a copy).
    struct StackEntry
    {
        WorkBuffer values;
        std::int32_t order = 0;
        std::vector<std::int32_t> delayedEquations;
    };

    void allocateFactor(SupernodalFactor& factor);
    void runSubtree(const FactorSubtree& subtree, SupernodalFactor& factor);
    void runTopSupernode(std::int32_t supernode, SupernodalFactor& factor);
    void assembleTrailing(std::int32_t supernode, DenseMatrix* schurComplement);

    // An assembled front: positive definite, its panel in place in the factor and its trailing block in `update`;
    // indefinite, the whole front in the caller's front buffer.
    struct AssembledFront
    {
        FrontStorage storage;
        WorkBuffer update;
    };
    // isPlanned: the buffers are covered by the reservation of the working storage (not a subtree beyond the plan).
    // pool: the threads of a top front's assembly (null in a subtree).
    [[nodiscard]] AssembledFront assemble(const FrontLayout& layout, std::span<const UpdateMatrix> children,
                                          SupernodalFactor& factor, WorkBuffer& front, bool isPlanned,
                                          base::ThreadPool* pool);

    // Eliminates the candidate columns of an assembled front and stores the panel; returns the update matrix (its
    // buffer, order and delayed equations), or sets a status other than Done.
    struct EliminationResult
    {
        WorkBuffer update;
        std::int32_t updateOrder = 0;
        std::vector<std::int32_t> delayedEquations;
    };
    [[nodiscard]] EliminationResult eliminate(const FrontLayout& layout, AssembledFront& front,
                                              const DenseKernels& kernels, SupernodalFactor& factor, bool isPlanned,
                                              ScratchSpool::Writer* writer);
    [[nodiscard]] EliminationResult eliminatePivoted(const FrontLayout& layout, DenseMatrixView front,
                                                     const DenseKernels& kernels, SupernodalFactor& factor,
                                                     bool isPlanned, ScratchSpool::Writer* writer);

    [[nodiscard]] FrontLayout layoutOf(std::int32_t supernode, std::span<const UpdateMatrix> children) const;
    [[nodiscard]] bool areChildrenDone(std::int32_t supernode) const;
    void finish(SupernodalFactor& factor);

    const SymbolicFactor& m_symbolic;
    const SymmetricCsc* m_matrix = nullptr; // the matrix, or
    FrontSource* m_source = nullptr;        // the front source
    // The packing buffers of the dense kernels kept from one front to the next: declared before the
    // kernels that name it.
    KernelWorkspace m_kernelWorkspace;
    // The kernels of the top supernodes: the caller's, with the factorisation's cancellation token, so that a large
    // front is interrupted between its tiles and blocks, not only between supernodes, and with the kernel workspace
    // (the subtrees' kernels take both).
    const DenseKernels m_kernels;
    const FactorOptions& m_options;
    base::MemoryBudget& m_budget;
    FactorSchedule m_schedule;
    std::size_t m_plannedConcurrency = 1;              // the subtrees at once the bound of the schedule assumes
    std::atomic<std::size_t> m_runningSubtrees{0};     // the subtree tasks running now
    std::atomic<std::int64_t> m_subtreesBeyondPlan{0}; // the subtrees that reserved their own storage
    WorkspaceMeter m_meter;
    WorkspacePool m_pool;                             // declared before every WorkBuffer member: it outlives them
    base::BudgetedBuffer<Status> m_status;            // per supernode
    base::BudgetedBuffer<std::int32_t> m_zeroPivotAt; // per supernode: the equation of its zero pivot
    base::BudgetedBuffer<std::int32_t> m_delayedOut;  // per supernode: the pivots it delayed to its parent
    base::MemoryReservation m_pendingReservation;     // the bytes of m_pending (declared first: reserved before)
    std::vector<PendingUpdate> m_pending;             // per slot of the schedule
    DenseMatrix* m_schurComplement = nullptr;
    // Out of core (writePanelsTo): the spool, the records, and per supernode the inertia of its pivots (taken as its
    // panel is written: finish() has no panel to read) and its 2 × 2 pivots.
    ScratchSpool* m_spool = nullptr;
    std::span<SpoolRecord> m_records;
    base::BudgetedBuffer<Inertia> m_inertiaOf;
    base::BudgetedBuffer<std::int64_t> m_twoByTwoOf;
};

} // namespace oos::linalg
