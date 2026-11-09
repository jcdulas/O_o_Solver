#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/solutions/common/FsetSolver.hpp"

#include <cstddef>
#include <vector>

namespace oos::solutions
{

// The zero-energy shapes of a stiffness matrix that cannot be factorised (a singularity or a mechanism also gives its
// shape): a near mechanism that fails the factor diagonal ratio test, or an exact one that meets a zero pivot (a
// straight chain of equal elements is singular to the last bit). Both are found the same way, by shifted subspace
// inverse iteration (Golub & Van Loan, Matrix Computations, 4th ed., §8.2.2 and §8.3.7):
//
//   K_s = K_ll + SHIFT·diag(K_ll),   X₀ = R (k fixed pseudo-random columns),   X_{i+1} = orth(K_s⁻¹·X_i).
//
// The shift makes an exact null space factorisable (its pivots become about SHIFT·a_jj instead of 0) and changes the
// other directions by a relative SHIFT only; K_s is factorised with the failed pivots kept (the PARAM BAILOUT −1
// behaviour). A zero-energy direction then has an eigenvalue of K_s below SHIFT·max a_jj, every other one at least the
// smallest flexible eigenvalue λ₁ of K_ll, so each solve amplifies the null space against the rest by λ₁/(SHIFT·a);
// after INVERSE_ITERATIONS solves the columns span it to round-off: for a free structure the six rigid-body motions,
// for a mechanism its free motion. The shift moves the shapes off the exact null space by about SHIFT·a/λ₁, the order
// of the round-off of any factorisation of K_ll (a gap λ₁ is resolved to ε·a/λ₁, ε = 2.2e-16).
//
// The block has the failed pivots of K_s plus GUARD_COLUMNS (at most MAX_SHAPES): the ratio test finds the dimension
// of the near null space only approximately (a null pivot of a large model may pass it, round-off having grown its
// value). The shapes are then chosen by their energy, by the Rayleigh-Ritz projection of the block on K_ll and
// D = diag(K_ll) (Golub & Van Loan §8.1.2; Parlett, The Symmetric Eigenvalue Problem, §11.4):
//
//   (Xᵀ K X) y = θ (Xᵀ D X) y,   shapes = X·y for θ ≤ ZERO_ENERGY_RATIO,
//
// θ = xᵀKx / xᵀDx being the energy of a shape relative to that of its DOFs taken one by one: below 1/MAXRATIO, the
// near mechanism the factor diagonal ratio test flags; at round-off for an exact one.
//
// Each shape is expanded to the g set (zero on the s set, u_m = Gm·u_n), scaled so that its largest amplitude is 1
// with that amplitude positive (a fixed sign, so that the shapes are reproducible), and listed as the DOFs whose
// amplitude is above AMPLITUDE_FLOOR in g-set order (base::MechanismShape). The start columns, the orthonormalisation
// (modified Gram-Schmidt, twice, column by column in a fixed order) and the solves are deterministic, so the shapes
// have the same bits whatever the number of threads.
//
// Thread-safety: the functions are pure functions of their arguments. Ownership: the work buffers and the shifted
// matrix are budgeted; the shapes returned are bounded by MAX_LISTED_ENTRIES.
class MechanismShapes
{
public:
    // At most this many shapes (a free body has six; more failed pivots mean a badly connected model, whose first
    // shapes are enough to find the fault).
    static constexpr std::size_t MAX_SHAPES = 24;
    // The relative shift of the diagonal: about 64 ε, the smallest that turns an exact zero pivot into a small
    // non-zero one whatever the round-off of the elimination, so that the shapes stay within a few dozen ε·a/λ₁ of
    // the exact null space.
    static constexpr double SHIFT = 1.0e-14;
    // Three solves: each amplifies the null space by at least λ₁/(SHIFT·a), 1e2 even for a model whose flexible
    // eigenvalues start 1e12 below its largest diagonal term (a 1000 × 1000 free plate), so three leave the other
    // directions below 1e-6 of the shapes on such a model and at round-off on ordinary ones.
    static constexpr std::size_t INVERSE_ITERATIONS = 3;
    // The columns added to the failed pivot count (a free body has six zero-energy motions).
    static constexpr std::size_t GUARD_COLUMNS = 6;
    // A Ritz vector of energy ratio θ = xᵀKx / xᵀDx at most this is a zero-energy shape: the inverse of the default
    // PARAM MAXRATIO (10⁷), the near-mechanism threshold of the factor diagonal ratio test.
    static constexpr double ZERO_ENERGY_RATIO = 1.0e-7;
    // A DOF whose amplitude is below this fraction of the largest one is left out of the shape (the round-off of the
    // iteration is far below; a real motion is never left out).
    static constexpr double AMPLITUDE_FLOOR = 1.0e-10;
    // A column whose norm falls below this fraction of its norm before the orthogonalisation is a combination of the
    // previous ones and is dropped (a near null space smaller than the failed pivot count).
    static constexpr double DEPENDENT_COLUMN_RATIO = 1.0e-8;
    // The entries (DOF and amplitude) of all shapes together: the shapes carried by a SingularityError are not
    // budgeted, so their number of shapes is reduced until they fit this bound (2²² entries, 64 MB).
    static constexpr std::size_t MAX_LISTED_ENTRIES = std::size_t{1} << 22U;

    // K_s = K + SHIFT·diag(K): a copy of `knn` on its pattern with every diagonal term scaled by 1 + SHIFT. Throws
    // MemoryBudgetError.
    [[nodiscard]] static linalg::BlockSparseMatrix shiftedStiffness(const linalg::BlockSparseMatrix& knn,
                                                                    base::MemoryBudget& budget);

    // The shapes of the near null space of K_ll (`knn` restricted to the equations of `solver`), from the factorised
    // K_s of `solver` (a factor kept despite failed pivots): a block of `failedPivots` + GUARD_COLUMNS columns (reduced
    // to MAX_SHAPES, to the equation count and to MAX_LISTED_ENTRIES), the shapes in increasing energy. `gm` expands
    // the l-set motion to the m set. Throws MemoryBudgetError, OperationCancelledError.
    [[nodiscard]] static std::vector<base::MechanismShape> compute(const FsetSolver& solver,
                                                                   const linalg::BlockSparseMatrix& knn,
                                                                   const assembly::RigidElimination& gm,
                                                                   const assembly::DofMap& dofMap,
                                                                   std::size_t failedPivots, base::MemoryBudget& budget,
                                                                   const base::CancellationToken* cancellation);
};

} // namespace oos::solutions
