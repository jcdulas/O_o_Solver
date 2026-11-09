#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/responses/BatchCallable.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// The default tolerance of a fast reanalysis: the relative energy norm of the residual
// below which the combined approximations are taken, η = (ρᵀ·K⁻¹·ρ / f'ᵀ·u')^½. A tolerance of 1e-6 on η keeps the
// responses of the clamped plate within 1e-6 of the full analysis on steps of 1 %, 10 % and 30 % of the thickness
// (measured), and η reaches it in 2 to 5 basis vectors there.
inline constexpr double FAST_REANALYSIS_TOLERANCE = 1.0e-6;
// The largest basis of the combined approximations: each vector costs one solve on the current factor; a change too
// large to be reached in so many is analysed in full (Kirsch's examples use 2 to 6 terms).
inline constexpr std::size_t FAST_REANALYSIS_MAX_BASIS = 8;

struct FastReanalysisOptions
{
    double tolerance = FAST_REANALYSIS_TOLERANCE;
    std::size_t maxBasis = FAST_REANALYSIS_MAX_BASIS;
};

// One subcase of a fast reanalysis: the basis it took and its residual η.
struct FastReanalysisSubcase
{
    std::int32_t subcase = 0;
    std::size_t basis = 0;
    double residual = 0.0;
};

// What a fast reanalysis gave: the evaluation of the trial point when every subcase is within the tolerance, or the
// reason a full analysis is needed (the evaluation is then empty: the caller analyses the point).
struct FastReanalysisResult
{
    bool isWithinTolerance = false;
    std::string fallback; // empty when the approximation was taken
    std::vector<FastReanalysisSubcase> subcases;
    double residual = 0.0; // the largest η
    std::int64_t solves = 0;
    std::unique_ptr<DesignEvaluation> evaluation;
};

// The fast reanalysis of a trial design point on the factor of the current one (an optimisation aid beyond
// gradients: an evaluation asked as approximate, with a tolerance), by combined approximations:
// Kirsch, "Combined approximations — a general reanalysis approach for structural optimization", Struct. Multidisc.
// Optim. 20 (2000) 97-106; Kirsch and Papalambros, AIAA J. 39 (2001) 2242-2246. With K the factorised stiffness of
// the current point and K' = K + ΔK, f' the stiffness and loads of the trial point (assembled, never factorised):
//   - the basis is the binomial series of K'⁻¹ = (I + K⁻¹·ΔK)⁻¹·K⁻¹ applied to f', r_1 = K⁻¹·f',
//     r_{k+1} = −K⁻¹·ΔK·r_k; here each new vector is the preconditioned residual K⁻¹·(f' − K'·u'_k), which spans the
//     same space (a Krylov space of K⁻¹·K') and gives the residual at once; the vectors are orthonormalised (modified
//     Gram-Schmidt, twice) for the conditioning;
//   - the displacements are the Ritz solution on the basis R: (Rᵀ·K'·R)·y = Rᵀ·f', u' = R·y (Cholesky of the small
//     system);
//   - the residual ρ = f' − K'·u' is measured in the energy norm, η = (ρᵀ·K⁻¹·ρ / f'ᵀ·u')^½ (K⁻¹ for K'⁻¹, which it
//     approaches as the change gets small), a bound of the relative energy error of u'.
// Vectors are added until η ≤ tolerance (one solve each), at most maxBasis; a subcase that does not get there sends
// the point to a full analysis, named in `fallback`. Within the tolerance, the displacements are recovered as the
// static solution does (the SPC forces K'·u' − P' less the MPC forces, the element tables, the strain energies, the
// external work of the checks), and the responses, constraints and objective evaluated on them.
//
// Statics only. A full analysis is asked (fallback, named) for a subcase with enforced displacements, under inertia
// relief or a SUBCOM, and for a trial point whose shape distorts an element (the full analysis refuses or warns).
//
// Thread-safety: run on the calling thread. Ownership: none kept.
class FastReanalysis
{
public:
    // The trial point x (`supplied`: its relations given by functions) reanalysed on the kept evaluation `current`.
    [[nodiscard]] static FastReanalysisResult evaluate(
        const DesignProblem& problem, const DesignEvaluation& current, std::span<const double> x,
        std::span<const SuppliedField> supplied, std::span<const SuppliedField> initialSupplied,
        const CallableRegistry& callables, base::MemoryBudget& budget, const solutions::StaticOptions& analysisOptions,
        const FastReanalysisOptions& options, std::size_t blockItems = EQUATION_BATCH_ITEMS);
};

} // namespace oos::design
