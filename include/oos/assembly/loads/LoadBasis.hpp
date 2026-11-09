#pragma once

#include "oos/assembly/loads/CompactLoadMatrix.hpp"
#include "oos/assembly/loads/LoadFamily.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oos::assembly
{

// The rank at round-off of the column-pivoted QR of the load matrix: a column whose remaining norm,
// once projected out of the pivots taken, is at most LOAD_RANK_TOLERANCE times its own norm is not a new direction.
// Decimal load values carry the rounding of their terms. The 1e-12 threshold gives the preliminary rank selection
// room above double-precision arithmetic round-off. A column wrongly taken as dependent is caught
// by the reconstruction check (LOAD_BASIS_RESIDUAL_LIMIT) and joins the basis: the tolerance only decides the gain,
// never a result.
inline constexpr double LOAD_RANK_TOLERANCE = 1.0e-12;

// The largest relative residual ‖f_case − Σ α_i f_i‖ / ‖f_case‖ with which a load case counts as reconstructed by the
// basis ("of the order of 1e-13"): above the residual of an exact combination computed in double (a
// few 1e-16), while allowing accumulated decimal rounding;
// the results then equal those of a case-by-case solution to round-off (the solve itself loses more digits through
// the condition number of K). A case above it joins the basis.
inline constexpr double LOAD_BASIS_RESIDUAL_LIMIT = 1.0e-13;

// With loadBasisTolerance="writtenPrecision", the cases with the largest residuals that the solution also solves on
// their own to report the deviation of their results: ten cases cost ten solves and show the worst deviation when the
// residuals are close; the deviation of a case is proportional to its load residual, so the cases with the largest
// residuals bound the others.
inline constexpr std::size_t WRITTEN_PRECISION_CHECK_CASES = 10;

// The full-rank certificate of the rank test: before the column-pivoted QR, when the cases are the smallest
// candidate, the unpivoted QR of the scaled load matrix (linalg::FullRankCertificate) may prove that the pivoted QR
// would take every case but one, so that it is not run. Why the proof holds: scaled by the references of the columns
// (their norms, or the norms of their precisions with the deck tolerance), the pivoted QR stops at step k when every
// remaining norm is at most the threshold τ; its remaining block R22 then has ‖R22‖₂ ≤ ‖R22‖_F ≤ τ·√(n − k), and
// σ_{k+1} ≤ ‖R22‖₂ (Eckart-Young: the first k columns of R give a matrix of rank k at that distance). So
// σ_min ≥ RANK_CERTIFICATE_MARGIN·τ·√n proves that it takes n pivots. The margin covers the rounding of the remaining
// norms of the pivoted QR (a few ε relative, against τ ≥ 1e-12), and the rounding of the unpivoted QR itself is added:
// RANK_CERTIFICATE_ROUNDING·n·ε·‖Â‖_F (Higham §19.3, with a factor that covers the constant of the bound in practice).
// A certificate that fails costs the unpivoted QR up to its first dependent column (it stops there) and the pivoted QR
// runs as before; one that holds decides the same as the pivoted QR would, at the rate of the matrix products.
inline constexpr double RANK_CERTIFICATE_MARGIN = 10.0;
inline constexpr double RANK_CERTIFICATE_ROUNDING = 10.0;
// Below this many cases the pivoted QR is cheap: no certificate is tried.
inline constexpr std::size_t RANK_CERTIFICATE_MIN_CASES = 256;

// The precision the rank test and the reconstruction check use: the round-off of the values
// (default), or the precision each value is written with in the deck (loadBasisTolerance="writtenPrecision").
enum class LoadBasisTolerance : std::uint8_t
{
    Roundoff,
    WrittenPrecision
};

// The basis a family takes.
enum class LoadBasisKind : std::uint8_t
{
    Empty,      // no case of the family has a load
    LoadSets,   // the load sets as written (LOAD and SUBCOM give the coefficients)
    LoadedDofs, // the loaded DOFs, unit vectors
    Rank,       // a subset of the cases spanning the others to round-off (or to the deck precision)
    Cases       // the load cases themselves
};

[[nodiscard]] const char* nameOf(LoadBasisKind kind) noexcept;

// What one basis vector is: an elementary load set (its column in the load sets matrix), a compact row (a loaded DOF,
// for LoadedDofs), or a load case (its column in the cases matrix: a pivot of the rank basis, a case of the Cases
// basis, or a case that joined the basis because it was not reconstructed).
struct BasisVector
{
    enum class Source : std::uint8_t
    {
        LoadSet,
        LoadedRow,
        Case
    };
    Source source = Source::Case;
    std::int64_t index = 0;
};

// The report of the choice of one family's basis (the report the solution hands to the user, per family and per
// constraint set).
struct LoadBasisReport
{
    LoadFamily family = LoadFamily::Concentrated;
    LoadBasisKind kind = LoadBasisKind::Empty;
    LoadBasisTolerance tolerance = LoadBasisTolerance::Roundoff;
    std::int64_t size = 0;                // the basis vectors
    std::int64_t caseCount = 0;           // the cases with a load of the family
    std::int64_t loadSetCount = -1;       // the size of the LoadSets candidate (with its joined cases), -1 if none
    std::int64_t loadedDofCount = -1;     // the size of the LoadedDofs candidate, -1 if not offered
    std::int64_t rank = -1;               // the size of the Rank candidate (with its joined cases), -1 if not computed
    std::int64_t joinedCaseCount = 0;     // the cases of the chosen basis that joined it by the reconstruction check
    double largestResidual = 0.0;         // the largest relative load residual of the chosen basis
    std::vector<std::int64_t> worstCases; // deck tolerance: the WRITTEN_PRECISION_CHECK_CASES cases with the largest
                                          // residuals (columns of the cases matrix), largest first
    std::vector<double> worstResiduals;   // their relative residuals
    std::string explanation;              // why this basis, and why not the others (the rank skipped, no gain)
};

// The smallest exact basis of the load cases of one family: among the load sets as written, the
// loaded DOFs, the rank of the load matrix and the cases themselves, the one with the fewest vectors (on a tie, in that
// order: the exact constructions before the rank), each case f_c then being Σ α_ic f_i. Nothing is truncated:
//
// - the rank is the numerical rank of the column-pivoted Householder QR of the compact load matrix (Businger and
//   Golub, Numer. Math. 7, 1965; Golub and Van Loan, Matrix Computations, §5.4.1), the pivot being the column of the
//   largest remaining norm relative to its own norm (or to the norm of its precisions, with the deck tolerance), the
//   elimination stopping when that ratio is at most LOAD_RANK_TOLERANCE (1 with the deck tolerance), or as soon as the
//   rank can no longer be the smallest basis (no work beyond what can win);
// - every case is then reconstructed from its basis and checked in the compact space: its relative residual must stay
//   below LOAD_BASIS_RESIDUAL_LIMIT (with the deck tolerance: every value within its written precision, plus that
//   limit of the column's largest value for the values written as exact zeros); a case that fails joins the basis and
//   the coefficients are computed again, until every case passes;
// - the coefficients of the rank basis are the least-squares solution on its vectors (Householder QR of the basis
//   columns, applied to each case); those of the load sets are the factors of the LOAD combinations (checked the same
//   way); those of the loaded DOFs are the values themselves; those of the cases the identity.
// The rank test needs a dense copy of the compact matrix: when it does not fit the budget it is skipped and the report
// says so. The choice depends on the counts only (and the order of the columns), so it is deterministic.
//
// Thread-safety: an immutable value once chosen. Ownership: owns its vectors and coefficients (budgeted).
class LoadBasis
{
public:
    // What the choice works from. `cases`: one column per load case of the family (a zero column: no load of the
    // family in that case, which takes no basis vector and zero coefficients). `loadSets` and `setCoefficients` (rows
    // the load sets, columns the cases): the elementary load sets and the factors each case takes them with, or null
    // when the family has no such description. `offersLoadedDofs`: the rows of `cases` are DOFs (the concentrated
    // loads), so that the unit vectors are a candidate. With LoadBasisTolerance::WrittenPrecision the precisions of
    // `cases` are used; a NaN precision (the format of a card not known) makes the rank candidate fall back to the
    // round-off.
    struct Input
    {
        LoadFamily family = LoadFamily::Concentrated;
        const CompactLoadMatrix* cases = nullptr;
        const CompactLoadMatrix* loadSets = nullptr;
        const linalg::DenseMatrix* setCoefficients = nullptr;
        bool offersLoadedDofs = false;
        LoadBasisTolerance tolerance = LoadBasisTolerance::Roundoff;
        base::ThreadPool* pool = nullptr; // shares the columns of the QR updates; null: one thread
        // The dense kernels of the full-rank certificate of the rank test (the analysis' family; its matrix products
        // run on `pool`).
        linalg::KernelKind kernel = linalg::KernelKind::Reference;
        // The cases are exact combinations of the load sets with the factors
        // `setCoefficients` (every case a LOAD of elementary sets it brings whole, each with the product of the
        // factors it is reached through: StaticLoadCases), so that the matrix of the cases need not be made: `cases`
        // may be null, `caseCount` then gives the cases. The basis is the load sets some case takes that have a value
        // on the rows (the others left out), or the loaded DOFs when they are fewer, exact by construction (a case
        // differs from Σ factor × set by the order of its sums only): no rank test, no reconstruction check.
        bool isCombinationOfSets = false;
        std::size_t caseCount = 0;
    };

    // Throws std::invalid_argument when the matrices disagree in size; MemoryBudgetError for the coefficients and the
    // check (the rank test alone is skipped when it does not fit).
    [[nodiscard]] static LoadBasis choose(const Input& input, base::MemoryBudget& budget);

    [[nodiscard]] LoadBasisKind kind() const noexcept { return m_report.kind; }
    [[nodiscard]] std::int64_t size() const noexcept { return static_cast<std::int64_t>(m_vectors.size()); }
    [[nodiscard]] std::span<const BasisVector> vectors() const noexcept { return m_vectors.span(); }
    // α (size() × the case count): case c is Σ_i α(i, c) times basis vector i.
    [[nodiscard]] const linalg::DenseMatrix& coefficients() const noexcept { return m_coefficients; }
    [[nodiscard]] const LoadBasisReport& report() const noexcept { return m_report; }

private:
    // The basis of cases given as combinations of the load sets (Input::isCombinationOfSets).
    [[nodiscard]] static LoadBasis chooseFromSets(const Input& input, base::MemoryBudget& budget);

    base::BudgetedBuffer<BasisVector> m_vectors;
    linalg::DenseMatrix m_coefficients;
    LoadBasisReport m_report;
};

} // namespace oos::assembly
