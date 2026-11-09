#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/dense/Inertia.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/ScratchSpool.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace oos::linalg
{

// What the solves of an out-of-core factor read so far.
struct OutOfCoreTraffic
{
    std::uint64_t writtenBytes = 0; // the panels written by the factorisation
    std::uint64_t readBytes = 0;    // read back by the sweeps
    std::uint64_t readRequests = 0;
    double readSeconds = 0.0; // the sweeps' waits on the reads
};

// The LDLᵀ factor of an indefinite matrix with its panels on a scratch disk: the
// multifrontal factorisation of SupernodalFactor in indefinite mode (MultifrontalDriver unchanged in its arithmetic:
// the same fronts, pivots, delays and panels) hands each panel to a ScratchSpool once it is final and keeps only the
// factor's rows, subdiagonal and arrays and the solve plan in memory; the solves are FactorSolve's sweeps by the plan's
// units in FactorSolve's order (the subtrees, their accumulators added in subtree order, the top; backward the
// reverse), each unit's panels read from the spool into memory just before it is swept and released after.
//
// Bits: every panel is the in-core factor's, and every unit step is SolveSweeps' on the same panels in the order of
// FactorSolve's sweeps by units, whose bits FactorSolve's flow has too: solve() gives the bits of
// FactorSolve::solve on SupernodalFactor::factorise of the same matrix, whatever the threads.
//
// Memory: the factorisation's working storage (FactorSchedule's indefinite bound, the fronts grown by delays reserved
// as they grow), the factor's arrays and plan, one unit's panels at a time in the solves; the panels on the spool's
// disk (or in memory: the spool's options). Budgeted (MemoryBudgetError).
//
// Thread-safety: one solve at a time (a solve points the factor's panels at its buffers). Ownership: owns its factor
// and spool; references nothing after factorise().
class OutOfCoreFactor
{
public:
    OutOfCoreFactor() = default;
    OutOfCoreFactor(OutOfCoreFactor&&) noexcept = default;
    OutOfCoreFactor& operator=(OutOfCoreFactor&&) noexcept = default;

    // The factorisation in indefinite mode (the pivoting of the options replaced), its panels to a spool of
    // `spoolOptions`. Throws what SupernodalFactor::factorise throws, and FileError.
    [[nodiscard]] static OutOfCoreFactor factorise(const SymbolicFactor& symbolic, const SymmetricCsc& matrix,
                                                   const DenseKernels& kernels, const FactorOptions& options,
                                                   const ScratchSpool::Options& spoolOptions,
                                                   base::MemoryBudget& budget);
    // The same factorisation of a matrix given front by front (the pencils from the elements): each front's own terms
    // from the source (its addTerms() called concurrently for distinct supernodes),
    // then the children's update matrices. The factor of a source that adds a matrix's entries of its supernodes'
    // columns has the bits of the matrix's.
    [[nodiscard]] static OutOfCoreFactor factorise(const SymbolicFactor& symbolic, FrontSource& source,
                                                   const DenseKernels& kernels, const FactorOptions& options,
                                                   const ScratchSpool::Options& spoolOptions,
                                                   base::MemoryBudget& budget);

    // The factor's structure (its panels are not resident outside a solve).
    [[nodiscard]] const SupernodalFactor& factor() const noexcept { return m_factor; }
    [[nodiscard]] const Inertia& inertia() const noexcept { return m_factor.inertia(); }
    [[nodiscard]] bool isSingular() const noexcept { return m_factor.isSingular(); }
    // The values of the panels on the spool.
    [[nodiscard]] std::uint64_t spooledBytes() const noexcept { return m_spool ? m_spool->heldBytes() : 0; }
    [[nodiscard]] OutOfCoreTraffic traffic() const noexcept;

    // B ← A⁻¹·B (FactorSolve::solve's bits). Throws std::invalid_argument for B of another row count,
    // SingularityError for a singular factor, FileError, OperationCancelledError.
    void solve(DenseMatrixView rightHandSides, const DenseKernels& kernels,
               const base::CancellationToken* cancellation = nullptr);

private:
    // Points the panels of supernodes [first, last] at a buffer holding their records read from the spool.
    void load(std::int32_t first, std::int32_t last);
    void release(std::int32_t first, std::int32_t last) noexcept;

    SupernodalFactor m_factor;
    std::unique_ptr<ScratchSpool> m_spool;
    std::vector<SpoolRecord> m_records; // per supernode (bounded by the supernodes)
    SpoolBuffer m_window;               // the panels resident
    base::MemoryBudget* m_budget = nullptr;
    OutOfCoreTraffic m_traffic;
};

} // namespace oos::linalg
