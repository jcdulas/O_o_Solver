#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/symbolic/Supernodes.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::linalg
{

// The fixed thresholds of the relaxed amalgamation, in DOF columns. The rule is CHOLMOD's
// (T. A. Davis, cholmod_super.c, after Ashcraft and Grimes, "The influence of relaxed supernode partitions on the
// multifrontal method", ACM TOMS 15, 1989): a child is merged into its parent when the merged supernode has
//   at most columnLimits[0] columns, whatever its zeros, or
//   at most columnLimits[1] columns and a fraction of explicit zeros below zeroFractions[0], or
//   at most columnLimits[2] columns and a fraction below zeroFractions[1], or
//   any width and a fraction below zeroFractions[2].
// CHOLMOD counts scalar columns (4, 16, 48); a grid point here is 6 columns, so the column limits are CHOLMOD's in
// nodes of 6 DOFs: the dense kernels then work on blocks large enough to pay their call overhead.
struct AmalgamationThresholds
{
    std::array<std::int64_t, 3> columnLimits = {24, 96, 288};
    std::array<double, 3> zeroFractions = {0.8, 0.1, 0.05};
    // The most columns of a supernode once amalgamated (0: no bound): a wider one is cut into a chain of supernodes of
    // at most this many (the full path streamed: a front holds its panel, c·(c + r) values, beside the
    // update matrices waiting; a separator of several thousand columns made the top's panel the largest part of the
    // working storage). The trailing supernode is never cut. Without fill: the same factor, by smaller panels.
    std::int64_t maxColumns = 0;

    // No amalgamation: the fundamental supernodes are kept (the exact structure of the factor, for the tests).
    [[nodiscard]] static constexpr AmalgamationThresholds none() { return {{0, 0, 0}, {-1.0, -1.0, -1.0}}; }
};

// Merges supernodes into their parents by the rule of AmalgamationThresholds (relaxed supernodes): a few explicit
// zeros in exchange for larger dense blocks. Only a child placed just before its parent (its last child in the
// postorder) can merge, so supernodes stay ranges of consecutive positions; the trailing supernode never merges.
//
// The zeros are counted exactly: merging child c (cc columns, rc rows below) into parent p (cp columns, rp rows
// below) gives the columns of c the rows of p, so the merged supernode has the zeros of both plus cc·(cp + rp − rc).
class Amalgamation
{
public:
    explicit Amalgamation(AmalgamationThresholds thresholds = {}) : m_thresholds(thresholds) {}

    // positionWeight: the DOFs of each position; belowWeight: the DOF rows below each position's diagonal block
    // (ColumnCounts); hasTrailingSupernode: the last supernode is the trailing block.
    [[nodiscard]] Supernodes relax(const Supernodes& fundamental, std::span<const std::int32_t> positionWeight,
                                   const ColumnCounts& counts, bool hasTrailingSupernode,
                                   base::MemoryBudget& budget) const;

    [[nodiscard]] const AmalgamationThresholds& thresholds() const noexcept { return m_thresholds; }

private:
    [[nodiscard]] bool isMergeWorthIt(std::int64_t columns, std::int64_t zeros, std::int64_t rowsBelow) const noexcept;

    AmalgamationThresholds m_thresholds;
};

} // namespace oos::linalg
