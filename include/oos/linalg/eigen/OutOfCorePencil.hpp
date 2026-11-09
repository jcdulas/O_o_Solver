#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/eigen/BucklingPencil.hpp"
#include "oos/linalg/eigen/PencilSource.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/OutOfCoreFactor.hpp"
#include "oos/linalg/supernodal/ScratchSpool.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace oos::linalg
{

// The pencil of SparsePencil (vibration: K, M) or BucklingPencil (buckling: K, K_d) with its factor out of core (step
// 10.43): K − σB formed as theirs (SymmetricCsc::combine), factorised by OutOfCoreFactor (its panels
// on a scratch spool), solved by its sweeps; the products with B and with the inner product's matrix are the in-core
// pencil's own. The same factor's bits, solves and products: the block Lanczos solver gives the in-core path's bits.
//
// Given a PencilSource instead of the matrices (the pencils from the elements): K − σB factorised from the
// source's fronts (OutOfCoreFactor of a FrontSource), the products and the diagonals of the Rayleigh bound the
// source's; no matrix of the pencil is held. The bits then follow the source's sums.
//
// Memory: the shifted matrix (of the matrices) and the factor's arrays in the budget; the panels on the spool (one
// factor's at a time: the previous spool is removed before the next factorisation). Thread-safety and ownership: as
// SparsePencil (the matrices or the source, the symbolic factor and the kernels outlive it).
class OutOfCorePencil final : public ShiftedPencil
{
public:
    // `rightHand`: M for vibration, K_d for buckling. Throws what SparsePencil and BucklingPencil throw.
    OutOfCorePencil(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness, const SymmetricCsc& rightHand,
                    SpectralTransform transform, const DenseKernels& kernels, const FactorOptions& options,
                    const ScratchSpool::Options& spool, base::MemoryBudget& budget);
    // The pencil of a source (its fronts in the order of `symbolic`). Throws SolverError for a mass diagonal that is
    // negative or not finite (vibration), as SparsePencil.
    OutOfCorePencil(const SymbolicFactor& symbolic, PencilSource& source, SpectralTransform transform,
                    const DenseKernels& kernels, const FactorOptions& options, const ScratchSpool::Options& spool,
                    base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t dimension() const noexcept override { return m_symbolic.dofCount(); }
    [[nodiscard]] ShiftFactorisation factorise(double shift) override;
    // Throws std::logic_error before the first factorisation; SingularityError on a singular factor; FileError.
    void solve(DenseMatrixView block) const override;
    void multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const override;
    [[nodiscard]] SpectralTransform transform() const noexcept override { return m_transform; }
    void multiplyInnerProduct(ConstDenseMatrixView x, DenseMatrixView y) const override;
    [[nodiscard]] double diagonalRayleighBound() const override;

    [[nodiscard]] std::size_t factorisationCount() const noexcept { return m_factorisationCount; }
    // The disk's traffic so far, over every factorisation.
    [[nodiscard]] OutOfCoreTraffic traffic() const noexcept;
    // The bytes of one factor's panels on the spool at most so far.
    [[nodiscard]] std::uint64_t largestSpoolBytes() const noexcept { return m_largestSpoolBytes; }
    // The budget's largest consumers at the most used of its samples (after each factorisation and each solve), with
    // that use: what the memory of a run is made of besides the factorisation's transient fronts.
    [[nodiscard]] std::uint64_t sampledPeakBytes() const noexcept { return m_sampledPeakBytes; }
    [[nodiscard]] const std::vector<base::MemoryConsumer>& sampledConsumers() const noexcept
    {
        return m_sampledConsumers;
    }

private:
    const SymbolicFactor& m_symbolic;
    const SymmetricCsc* m_stiffness = nullptr; // the matrices, or
    const SymmetricCsc* m_rightHand = nullptr;
    PencilSource* m_source = nullptr; // the source
    double m_sourceRayleighBound = 0.0;
    SpectralTransform m_transform;
    const DenseKernels& m_kernels;
    FactorOptions m_options;
    ScratchSpool::Options m_spool;
    base::MemoryBudget& m_budget;
    // The in-core pencil of the products (never factorised).
    std::unique_ptr<SparsePencil> m_vibration;
    std::unique_ptr<BucklingPencil> m_buckling;
    SymmetricCsc m_shifted;
    mutable OutOfCoreFactor m_factor;
    bool m_hasFactor = false;
    std::size_t m_factorisationCount = 0;
    OutOfCoreTraffic m_earlier; // the traffic of the factors given back
    std::uint64_t m_largestSpoolBytes = 0;
    void sample() const;
    mutable std::uint64_t m_sampledPeakBytes = 0;
    mutable std::vector<base::MemoryConsumer> m_sampledConsumers; // bounded by SAMPLED_CONSUMERS
};

} // namespace oos::linalg
