#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/reduction/condense/StaticCondensation.hpp"
#include "oos/reduction/patterns/BasisCoefficients.hpp"

#include <cstdint>
#include <span>

namespace oos::reduction
{

// The boundary displacements of every load case of a unit, as the interface problem gives them (the boundary basis):
// the solutions of its Q basis right-hand sides restricted to the unit's boundary DOFs (in the order
// of K*'s rows), and each case as a combination of them.
struct UnitBoundaryBasis
{
    linalg::ConstDenseMatrixView values;       // boundaryDofCount() × Q
    linalg::ConstDenseMatrixView coefficients; // Q × cases: β, the boundary displacements of a case are values·β
    // Per column of `values`: the unit's own load pattern whose condensed vector is that right-hand side (its
    // coefficients are then the pattern's α), or −1; empty: none of them is.
    std::span<const std::int64_t> patternOfColumn;
};

// How a recovery evaluates u_o = Σ_k α_k·g_k − K_oo⁻¹·(K_oa·u_a) (UnitRecovery).
enum class RecoveryRegime : std::uint8_t
{
    BasisColumns, // one interior solve per basis right-hand side: w_q = K_oo⁻¹·(f_o,k(q) − K_oa·V_q)
    LoadCases,    // one interior solve per load case: K_oo⁻¹·(Σ_k α_k·f_o,k − K_oa·u_a)
    BoundaryDofs  // one per boundary DOF, H = −K_oo⁻¹·K_oa, and one per load pattern, g_k = K_oo⁻¹·f_o,k
};

[[nodiscard]] const char* nameOf(RecoveryRegime regime) noexcept;

// The interior solves of a recovery (the number of interior solves of a recovered unit never exceeds
// min(n_a, load cases, Q): the solves of the boundary term, besides the k solves g_k = K_oo⁻¹·f_o,k of the unit's
// own patterns, which are counted with the condensation).
struct RecoveryStatistics
{
    RecoveryRegime regime = RecoveryRegime::BasisColumns;
    std::int64_t boundaryDofCount = 0; // n_a
    std::int64_t caseCount = 0;
    std::int64_t basisColumnCount = 0; // Q
    std::int64_t patternCount = 0;     // the unit's load patterns k
    // The solves of the boundary term K_oo⁻¹·K_oa·u_a: min(n_a, cases, Q) for the regime taken by it.
    std::int64_t boundarySolveCount = 0;
    // The solves of the patterns' own term g_k that the boundary solves do not carry (the condensation's g_k): the
    // patterns not among the basis columns, or all k with BoundaryDofs; 0 with LoadCases.
    std::int64_t patternSolveCount = 0;
    [[nodiscard]] std::int64_t interiorSolveCount() const noexcept { return boundarySolveCount + patternSolveCount; }
};

// The recovery of a condensed unit (one formula): for every load case,
//     u_o = K_oo⁻¹·(Σ_k α_k·f_o,k − K_oa·u_a),   u_a = V·β,
// evaluated on as few interior solves as the case permits, then combined per load case in a fixed order:
// - BasisColumns: w_q = K_oo⁻¹·(f_o,k(q) − K_oa·V_q) for each of the Q basis columns (a column that is the unit's own
//   pattern k carries its f_o,k, its β row being α_k; a pattern no column carries adds a column g_k of its own), then
//   u_c = Σ_q β_q,c·w_q: Q solves;
// - LoadCases: the right-hand side of each case solved, batch by batch: one solve per case;
// - BoundaryDofs: H = −K_oo⁻¹·K_oa (one solve per boundary DOF) and g_k, then u_c = Σ_k α_k,c·g_k + H·u_a,c: n_a + k
//   solves.
// The regime of fewest solves in all is taken (on a tie, in that order): never one per load case when the cases repeat
// (6 000 instead of 37 500 for the largest unit of a large model), and never more than k + min(n_a, cases, Q), the
// count whose condensation computes the k pattern solves g_k (here they ride on the boundary solves
// of the first two regimes and are made only for the third). Each solve is PartialFactor::recover, whose columns have
// the same bits alone or in any block; each case is
// combined over the basis in increasing order (BasisCoefficients::combine): a case has the same bits in any batch and
// with any number of threads. The displacements of the unit's DOFs follow (UnitSystem::expandSolution: the legs of
// the spiders, the dependent DOFs, the held DOFs at their enforced values).
//
// The element results are linear in the displacements: computed on basisDisplacements() and combined with
// basisCoefficients(), or from the displacements of a batch of cases (recoverCases).
//
// Thread-safety: immutable once built; recoverCases may run concurrently. Ownership: owns the basis solutions
// (budgeted: equationCount() × basis columns, released with the object); borrows the condensation, which must outlive
// it and keep its interior factor while this is built (and, for LoadCases, while cases are recovered).
class UnitRecovery
{
public:
    // patterns: the unit's load patterns as given to the condensation; alpha: their coefficients (patterns × cases);
    // boundary: the boundary basis (Q × cases coefficients). Throws std::invalid_argument on a size mismatch or when
    // a column named as the pattern k has coefficients other than α_k; std::logic_error when the interior factor was
    // released; MemoryBudgetError; OperationCancelledError.
    UnitRecovery(const StaticCondensation& condensation, const UnitPatterns& patterns, const BasisCoefficients& alpha,
                 const UnitBoundaryBasis& boundary, const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                 const base::CancellationToken* cancellation = nullptr);

    [[nodiscard]] const RecoveryStatistics& statistics() const noexcept { return m_statistics; }
    [[nodiscard]] std::size_t caseCount() const noexcept { return static_cast<std::size_t>(m_statistics.caseCount); }

    // The displacements of the unit's local DOFs (UnitSystem numbering) for the cases [firstCase, firstCase +
    // out.columns). Throws std::invalid_argument on a size mismatch or a batch beyond the cases; std::logic_error
    // with LoadCases when the interior factor was released meanwhile.
    void recoverCases(std::size_t firstCase, linalg::DenseMatrixView displacementsLocal,
                      const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                      const base::CancellationToken* cancellation = nullptr) const;
    // recoverCases before the expansion to the local DOFs: the solution of the unit's equations (equationCount() rows)
    // and the enforced values of the held DOFs (localDofCount() rows; no column when the patterns enforce none) for
    // the cases [firstCase, firstCase + equationValues.columns) — UnitSystem::expandSolution of them is recoverCases;
    // UnitSystem::heldReactions takes them. Throws as recoverCases.
    void recoverEquations(std::size_t firstCase, linalg::DenseMatrixView equationValues,
                          linalg::DenseMatrixView enforcedLocal, const linalg::DenseKernels& kernels,
                          base::MemoryBudget& budget, const base::CancellationToken* cancellation = nullptr) const;
    // Whether the patterns enforce displacements (recoverEquations then fills enforcedLocal).
    [[nodiscard]] bool hasEnforcedValues() const noexcept { return m_patterns.enforced.columns > 0; }

    // Whether the cases are combinations of basis displacements (BasisColumns, BoundaryDofs), and then the basis
    // displacements of the local DOFs (localDofCount() × basis) and the coefficients of every case on them. Throws
    // std::logic_error with LoadCases.
    [[nodiscard]] bool hasBasis() const noexcept { return m_statistics.regime != RecoveryRegime::LoadCases; }
    [[nodiscard]] linalg::DenseMatrix basisDisplacements(base::MemoryBudget& budget) const;
    [[nodiscard]] const BasisCoefficients& basisCoefficients() const;

private:
    const StaticCondensation* m_condensation;
    UnitPatterns m_patterns;
    const BasisCoefficients* m_alpha;
    UnitBoundaryBasis m_boundary;
    RecoveryStatistics m_statistics;
    // BasisColumns, BoundaryDofs: the solutions of the basis on the equations (equationCount() × B), the enforced
    // values of each basis column on the local DOFs (localDofCount() × B, no column when none), and the coefficients
    // of every case (B × cases).
    linalg::DenseMatrix m_solutions;
    linalg::DenseMatrix m_enforced;
    BasisCoefficients m_coefficients;
    // LoadCases: the loads of the patterns on the equations (equationCount() × patterns), combined per batch.
    linalg::DenseMatrix m_patternLoads;
};

} // namespace oos::reduction
