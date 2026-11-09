#pragma once

#include <cstdint>
#include <vector>

namespace oos::design
{

// What the sensitivity solves did in one condensed unit (the statistics of its solves;
// BoundaryReducedSensitivity). Each column condensed or recovered is one interior solve (a forward sweep, or a forward
// and a backward sweep, of the interior factor); a unit neither driven nor read gets none.
struct UnitSensitivityStatistics
{
    std::int64_t unitId = 0;
    std::int64_t boundaryDofCount = 0; // n_a
    bool isDriven = false;             // a design point drives it (the analysis' design stamp)
    std::int64_t condensedColumns = 0; // right-hand sides condensed onto its boundary: f* = f_a − K_ao·K_oo⁻¹·f_o
    std::int64_t recoveredColumns = 0; // solutions recovered in its interior: u_o = K_oo⁻¹·(f_o − K_oa·u_a)
    // Interface solves of its boundary basis: n_a each time the basis is made (a block whose columns on the unit
    // exceed n_a; again in a later such block when the basis was released for the memory budget).
    std::int64_t basisSolves = 0;
    std::int32_t refactorisations = 0; // its interior factor computed again (the analysis had released it)
};

// What the sensitivity solves of one constraint set did on the condensed path (BoundaryReducedSensitivity).
struct BoundaryReducedStatistics
{
    std::vector<std::int32_t> subcases;           // the subcases of the constraint set
    std::vector<UnitSensitivityStatistics> units; // the condensed units, in the plan's order
    std::int64_t columns = 0;                     // the right-hand sides given (on the g set)
    std::int64_t interfaceSolves = 0;             // the interface right-hand sides solved: columns and unit bases
    // The bound of the interface solves (the interface problem is solved for at most min(variables of the unit, n_a)
    // vectors per unit): Σ over the condensed units of min(columns that load its interior or its boundary, n_a),
    // plus the columns that load interface rows on no condensed unit's boundary (a full unit's interior, an external
    // unit's contour), solved as they are.
    std::int64_t interfaceSolveBound = 0;
    // The boundary bases kept between blocks: the most bytes they held at once, and how many were released
    // to stay within their share of the budget (each made again, to the same bits, by the next block that needs it).
    std::uint64_t retainedBasisPeakBytes = 0;
    std::int64_t basisReleases = 0;
};

} // namespace oos::design
