#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/solutions/sol101/CondensedFactorisedSet.hpp"
#include "oos/solutions/sol101/KeptLoadBasis.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::solutions
{

// The units of a constraint set's load basis on the condensed path (the load basis handed over): the basis vectors of
// `cases` (the full path's basis of the set: the same units and coefficients) solved on the condensed set the analysis
// kept (`set`, CondensedStatics::GroupInputs::solvesCases false), in three steps:
//  1. each vector's g-set loads split by rows (so that each load acts once): the rows of the interface's points are its
//     loads, the interior rows of a condensed unit the unit's, put on its equations (UnitSystem::equationLoads: its
//     relations and spiders applied) and condensed onto its boundary, f* = f_a − K_ao·K_oo⁻¹·f_o, with its interior
//     factor (made again, to the bits of its condensation, when the analysis released it);
//  2. the interface problem solved once for the k vectors (its factor already made): k solves, never more than the
//     cases, whatever the units;
//  3. kept: the interface's displacements of the k vectors (the boundary basis solutions) and each loaded unit's
//     equation loads; the interiors are recovered on demand (CondensedBasisColumns).
// The full path solves the same equations in another order: the units' displacements agree with its to round-off.
//
// Inertia relief (the set's CondensedRelief): each vector relieved as a case is, its accelerations ü = (DᵀMD)⁻¹·Cᵀ·Φᵀ·P
// from its resultant (assembly::rigidResultantOf), its relief loads −M·Φ·a (a = C·ü) the units' condensed inertial
// patterns and the interface's inertial loads (KeptCondensedBasis::reliefAmplitudes, the units' inertial loads in their
// interiors at the recovery); with conditioned supports the rigid motion that brings each vector's FreeBodySupport DOFs
// back to rest (KeptCondensedBasis::gaugeAmplitudes, added when filled).
//
// The fixed rows of the units' grid point forces (`fixedRows` when given): each vector's g-set loads at the DOFs
// they keep, with inertia relief's loads at the interface's points (its inertial loads −M·Φ times the vector's
// amplitudes, as statics writes appliedLoad there); the units' relief loads are added when they are recovered
// (CondensedBasisColumns::addUnitReliefLoads).
//
// Refused before (StaticSolution): enforced displacements, PARAM INREL = −2 and external units (std::logic_error here
// for a vector that enforces a displacement). Throws MemoryBudgetError, OperationCancelledError.
[[nodiscard]] std::shared_ptr<const KeptCondensedBasis> solveCondensedBasis(
    std::shared_ptr<const CondensedFactorisedSet> set, const StaticLoadCases& cases,
    const linalg::DenseKernels& kernels, base::MemoryBudget& budget, base::DiagnosticSink& sink,
    const base::CancellationToken* cancellation = nullptr, KeptFixedRows* fixedRows = nullptr);

// The g-set displacements of some units of a condensed load basis set (`columns`: their positions in the set's units),
// the interiors of the condensed units `isUnitRead` flags (one flag per unit of the set) recovered for them, the other
// interiors left at zero (the caller reads no DOF there; with conditioned supports they hold the gauge's rigid motion):
//     u_o = K_oo⁻¹·(f_o − K_oa·u_a)    (with inertia relief f_o + Σ_m a_m·(−M·Φ)_o,m)
// with the regime of fewest interior solves per unit: the columns asked (C solves), or H = −K_oo⁻¹·K_oa and the columns
// that load the unit (n_a + L solves) when that is fewer, then u_c = H·u_a,c + g_c. The regime depends on the columns
// asked only: the same bits whatever the threads of the kernels.
//
// Thread-safety: the const methods may run concurrently once built. Ownership: owns the recovered interiors
// (budgeted: the local DOFs of the units read × the columns); borrows the basis, which must outlive it.
class CondensedBasisColumns
{
public:
    // Throws std::invalid_argument for a column outside the units or flags of another size; MemoryBudgetError;
    // OperationCancelledError.
    CondensedBasisColumns(const KeptCondensedBasis& basis, std::span<const std::size_t> columns,
                          std::span<const std::uint8_t> isUnitRead, const linalg::DenseKernels& kernels,
                          base::MemoryBudget& budget, const base::CancellationToken* cancellation = nullptr);

    [[nodiscard]] std::size_t columnCount() const noexcept { return m_columns.size(); }
    // The interior solves made (the statistics of the tests and of the benchmark).
    [[nodiscard]] std::int64_t interiorSolveCount() const noexcept { return m_interiorSolves; }
    // The g-set displacements of the i-th column asked (0 ≤ i < columnCount()) into `displacementsG`.
    void fill(std::size_t i, std::span<double> displacementsG) const;
    // Adds into `loadsG` (a g-set vector) the relief loads Σ_m a_m·(−M·Φ)_m of the units recovered for the i-th column
    // asked, on all their local DOFs (with inertia relief; nothing otherwise): the units' part of its appliedLoad (as
    // statics adds each unit's relief pattern at its points, the interface's own inertial loads beside).
    void addUnitReliefLoads(std::size_t i, std::span<double> loadsG) const;

private:
    const KeptCondensedBasis* m_basis;
    std::vector<std::size_t> m_columns;
    std::vector<std::optional<linalg::DenseMatrix>> m_interiors; // per unit read: local DOFs × columns
    std::vector<std::optional<linalg::DenseMatrix>> m_inertial;  // per unit read with relief: its −M·Φ, local × 6
    std::int64_t m_interiorSolves = 0;
};

} // namespace oos::solutions
