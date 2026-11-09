#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace oos::design
{

// One block of the displacement derivatives handed to the sink: du/dx_j of the variables [firstVariable, firstVariable
// + variableCount) in one subcase, one g-set vector per variable (variableCount × gSize, row-major; the DOFs in the
// order of DisplacementDerivatives::labels). The values are valid during consume() only.
struct DisplacementDerivativeBlock
{
    std::int32_t subcase = 0;
    std::size_t firstVariable = 0;
    std::size_t variableCount = 0;
    std::size_t gSize = 0;
    std::span<const double> values;
};

// Where the blocks go (a plug-in point, called once per block, on the calling thread, subcase by subcase and the
// variables in increasing order); an exception it throws stops the computation and goes to the caller.
class DisplacementDerivativeSink
{
public:
    DisplacementDerivativeSink() = default;
    DisplacementDerivativeSink(const DisplacementDerivativeSink&) = delete;
    DisplacementDerivativeSink& operator=(const DisplacementDerivativeSink&) = delete;
    virtual ~DisplacementDerivativeSink() = default;

    virtual void consume(const DisplacementDerivativeBlock& block) = 0;
};

// The size of the whole field of derivatives and what one block holds, known before any computation.
struct DisplacementDerivativeSize
{
    std::size_t subcaseCount = 0;
    std::size_t variableCount = 0;
    std::size_t gSize = 0;
    std::uint64_t bytes = 0;          // subcases × variables × gSize doubles: what a destination keeping all must hold
    std::uint64_t blockPeakBytes = 0; // the largest memory of one block (its vectors and right-hand sides)
};

struct DisplacementDerivativeReport
{
    DisplacementDerivativeSize size;
    std::int64_t solves = 0;
    std::int64_t blocks = 0;
    std::int32_t factorisations = 0;
};

// The displacement derivatives of a design point (an optimisation aid beyond gradients): the whole field du/dx_j
// per variable, so that a partner approximates any derived quantity itself:
//     K_ll·du_l/dx_j = (Tᵀ·b_j)_l,  du_g/dx_j = T·du_l/dx_j,   b_j = ∂P/∂x_j − ∂K/∂x_j·u (PseudoLoads),
// the solves of the direct method (DirectSensitivity), one per variable and subcase, made by blocks of `block`
// variables, each block handed to the sink and discarded, as the full gradient mode streams G (GradientStream): the
// field of 1 000 variables on 1 M DOF would be 8 GB per subcase. The memory of a block is estimated first
// (size()), and a destination that cannot hold the whole field is refused before any solve (requireDestination).
//
// Thread-safety: run on the calling thread (the sink is called from it). Ownership: none kept.
class DisplacementDerivatives
{
public:
    // The g-set DOFs of the blocks: (point ID, component: 1-6 for a grid point, 0 for a scalar point).
    [[nodiscard]] static std::vector<std::pair<std::int32_t, std::int32_t>> labels(const DesignProblem& problem);

    // The size for `variableCount` variables by blocks of `block`, on the analysed subcases of `evaluation`.
    [[nodiscard]] static DisplacementDerivativeSize size(const DesignProblem& problem,
                                                         const DesignEvaluation& evaluation, std::size_t variableCount,
                                                         std::size_t block);
    // OutputVolumeError naming the destination, the size and the remedies when it cannot hold the field.
    static void requireDestination(const DisplacementDerivativeSize& size, std::string_view destination,
                                   std::uint64_t capacityBytes);

    // The derivatives of the variables `variables` (columns of x; empty: every one) of a kept evaluation, block by
    // block to `sink`. std::invalid_argument for a block of 0 or a variable out of range; throws as StaticSensitivity.
    [[nodiscard]] static DisplacementDerivativeReport stream(const DesignProblem& problem,
                                                             const DesignEvaluation& evaluation,
                                                             std::span<const std::size_t> variables, std::size_t block,
                                                             DisplacementDerivativeSink& sink,
                                                             base::MemoryBudget& budget,
                                                             const solutions::StaticOptions& analysisOptions = {});
};

} // namespace oos::design
