#include "oos/reduction/scheduling/UnitFootprint.hpp"

#include "oos/reduction/condense/CraigBampton.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace oos::reduction
{

namespace
{

constexpr std::uint64_t VALUE_BYTES = sizeof(double);
constexpr std::uint64_t INDEX_BYTES = sizeof(std::int32_t);
constexpr std::uint64_t OFFSET_BYTES = sizeof(std::int64_t);
// A grid point's DOFs: its node blocks are 6 × 6.
constexpr std::uint64_t BLOCK_VALUES = 36;
constexpr std::uint64_t RIGID_BODY_MODE_COUNT = 6;
// Per point of the unit, what its arrays hold: the local numbering (model node, DOF count, position, first DOF, sorted
// node, boundary flag: 46 bytes), the offsets of the node graph (8), the node order and its inverse (8), the DOF
// selection's first DOF (8), the symbolic factor's node arrays (about 16); rounded up.
constexpr std::uint64_t POINT_BYTES = 96;
// Per local DOF: its role (1), the DofSelection (equation of a DOF and DOF of an equation: 12), the inertial patterns
// −M·Φ_r (6 doubles: 48; the rigid-body modes are made when read), the condensed run's maps (model
// DOF, interior flag: 5), the column offset of its equation in the SymmetricCsc (8); rounded up.
constexpr std::uint64_t DOF_BYTES = 80;
// Per point of a unit whose equations are released (the UnitSystem made from its boundary data):
// the local numbering (46 bytes; no graph, no ordering, no symbolic factor).
constexpr std::uint64_t LIGHT_POINT_BYTES = 48;
// Per local DOF of a released unit: its role (1), the condensed run's maps (5); rounded up. Its rigid-body modes are
// made from the points' positions when they are read (UnitSystem::rigidBodyModesOf), never kept.
constexpr std::uint64_t LIGHT_DOF_BYTES = 8;
// The rows of the grid point singularity table of a point (assembly::SingularityRow, 24 bytes): three, the rotations a
// solid's grid point leaves without stiffness (a shell's drilling DOF is one); more is within the margin.
constexpr std::uint64_t SINGULARITY_POINT_BYTES = 3 * 24;
// The symbolic factor per supernode (starts, counts, the tree) and the factor per supernode (offsets and counts, the
// driver's status arrays: SupernodalFactor's PER_SUPERNODE_BYTES, 80).
constexpr std::uint64_t SUPERNODE_BYTES = 80;
// Per row of the interior panels, the factor's solve plan (each panel row's place in its unit, the
// accumulator rows, the outside rows): three indices, rounded up.
constexpr std::uint64_t SOLVE_PLAN_ROW_BYTES = 16;
// A DOF held in every subcase keeps its couplings to the equations and to the other held DOFs (K_fs, K_ss: an index and
// a value each) for its enforced loads and its reaction: at most its point's 6 DOFs times the 9 points a shell point
// touches, rounded up to 64 couplings of 12 bytes.
constexpr std::uint64_t HELD_DOF_BYTES = 64 * (INDEX_BYTES + VALUE_BYTES);
// A held DOF of a released unit: its index and its coupling flag (the couplings are the equations').
constexpr std::uint64_t LIGHT_HELD_DOF_BYTES = 16;
// A raw boundary block (RawBoundaryBlock, 16 bytes, and its 6 × 6 values) and a boundary label (UnitDofLabel padded),
// with the boundary equation's local DOF of a released unit (8) and its share of the boundary elements (16).
constexpr std::uint64_t RAW_BLOCK_BYTES = 16 + BLOCK_VALUES * VALUE_BYTES;
constexpr std::uint64_t LABEL_BYTES = 16;
constexpr std::uint64_t LIGHT_BOUNDARY_DOF_BYTES = LABEL_BYTES + 8 + 16;
// The relations of a released unit's spiders (LocalRelations: a row per reference DOF with a term per leg DOF, an index
// and a value) and their changes of basis (the reflectors and the right inverse: two values per term).
constexpr std::uint64_t SPIDER_TERM_BYTES = INDEX_BYTES + 3 * VALUE_BYTES;
// The build per point beyond its blocks: the block pattern's two offset arrays (16), the element groups and the Gm
// couplings (about 64), the ordering's arrays and its workspace (about 120: keptPoints, weights, positions, trailing,
// the subgraph's offsets, the nested dissection's levels); rounded up.
constexpr std::uint64_t BUILD_POINT_BYTES = 256;
// The build per block: the pattern's row and value offset (12), the subgraph's adjacency (8), and the stiffness and the
// mass in blocks (2 × 36 values), which coexist in the assembly.
constexpr std::uint64_t BUILD_BLOCK_BYTES = 12 + 8 + 2 * BLOCK_VALUES * VALUE_BYTES;
// The build per local DOF: the singularity processor's flags and ratios (10), the mass products (16), the held DOF
// index (4), the eliminated flags (1); rounded up.
constexpr std::uint64_t BUILD_DOF_BYTES = 40;
// The vectors of one case of a recovered unit's pass (CondensedStatics::solveGroup): the solution, the loads and K·u on
// its equations; its enforced values, displacements, loads and applied loads on its local DOFs.
constexpr std::uint64_t PASS_EQUATION_VECTORS = 3;
constexpr std::uint64_t PASS_LOCAL_VECTORS = 4;
// The margins on what the plan's graph does not see exactly: the unit's own graph keeps its points held in every
// subcase and the couplings of its spiders' change of basis (counted below by their blocks); the plan's factor is
// within 10 % of the unit's, twice that. The arrays were measured against the budget's accounting on
// assemblies of 3 000 plates (12 × 12 and 23 × 23 plates, absorbed spiders): the matrix of a unit within 1 % of the
// plan's entries and its spiders' blocks, the factor's
// panels within 2 % of the plan's; the arrays' margin, 1.5 before (a unit's equations estimated at 1.7 to 3 times
// what they held), is a tenth.
constexpr double FACTOR_MARGIN = 1.2;
constexpr double ARRAY_MARGIN = 1.1;

std::uint64_t withMargin(std::uint64_t bytes, double margin)
{
    return static_cast<std::uint64_t>(std::ceil(static_cast<double>(bytes) * margin));
}

std::uint64_t asBytes(std::int64_t count)
{
    return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

} // namespace

UnitFootprint UnitFootprint::estimate(std::int64_t unitId, const UnitFootprintInputs& inputs)
{
    if (inputs.cost == nullptr)
        throw std::invalid_argument("UnitFootprint::estimate: the unit's cost estimate is needed; give UnitPlan::cost");
    const UnitCostEstimate& cost = *inputs.cost;
    const std::uint64_t points = asBytes(inputs.pointCount);
    const std::uint64_t localDofs = asBytes(inputs.localDofCount);
    const std::uint64_t boundary = asBytes(cost.boundaryDofCount);
    const std::uint64_t equations = asBytes(cost.interiorDofCount) + boundary;
    const std::uint64_t patterns = asBytes(inputs.patternCount);
    const std::uint64_t matrixValues = asBytes(cost.matrixValueCount);
    const std::uint64_t freePoints = std::max<std::uint64_t>(asBytes(cost.nodeCount), 1);
    // The blocks over every point of the unit (the held ones too): the free points' blocks scaled by the points, and
    // those the spiders' change of basis couples.
    const std::uint64_t blocks = asBytes(cost.blockCount) * std::max<std::uint64_t>(points, freePoints) / freePoints +
                                 points + asBytes(inputs.spiderBlockCount);
    const std::uint64_t heldDofs = localDofs > equations ? localDofs - equations : 0;
    // The boundary points (6 DOFs each, at least one) and the raw blocks they hold: each with itself and its
    // neighbours among them, at the unit's average degree, and no more than the pairs of boundary points (the lower
    // triangle with the diagonal).
    const std::uint64_t boundaryPoints = (boundary + 5) / 6;
    const std::uint64_t averageDegree = 2 * asBytes(cost.blockCount) / freePoints;
    const std::uint64_t rawBlocks =
        std::min(boundaryPoints * (1 + averageDegree), boundaryPoints * (boundaryPoints + 1) / 2);
    const std::uint64_t spiderValues = asBytes(inputs.spiderValueCount);

    UnitFootprint footprint;
    footprint.unitId = unitId;
    footprint.flopCount = cost.flopCount;
    footprint.isRecovered = inputs.isRecovered;

    // The equations (UnitSystem), with the margin of the arrays: the matrix holds the plan's entries and the blocks its
    // spiders couple (a full 6 × 6 block each, an upper bound of their lower triangle).
    const std::uint64_t system =
        points * (POINT_BYTES + SINGULARITY_POINT_BYTES) + localDofs * DOF_BYTES + blocks * 2 * INDEX_BYTES +
        (equations + 1) * OFFSET_BYTES + matrixValues * (INDEX_BYTES + VALUE_BYTES) +
        asBytes(cost.factorRowCount) * 2 * INDEX_BYTES + asBytes(cost.supernodeCount) * SUPERNODE_BYTES +
        boundary * INDEX_BYTES + heldDofs * HELD_DOF_BYTES + rawBlocks * RAW_BLOCK_BYTES + boundary * LABEL_BYTES +
        asBytes(inputs.spiderBlockCount) * BLOCK_VALUES * (INDEX_BYTES + VALUE_BYTES) +
        spiderValues * SPIDER_TERM_BYTES;
    // The same unit without its equations (released): its numbering and roles, the rows and changes of basis of
    // its spiders, its singularity rows, its held DOFs, its raw boundary blocks and boundary equations.
    const std::uint64_t light = points * (LIGHT_POINT_BYTES + SINGULARITY_POINT_BYTES) + localDofs * LIGHT_DOF_BYTES +
                                spiderValues * SPIDER_TERM_BYTES + heldDofs * LIGHT_HELD_DOF_BYTES +
                                rawBlocks * RAW_BLOCK_BYTES + boundary * LIGHT_BOUNDARY_DOF_BYTES;
    // The patterns on the local DOFs (and their enforced values), the reduced unit (K*, f* and the six inertial
    // patterns, the labels) and the pattern work: exact sizes.
    const std::uint64_t patternBytes = localDofs * patterns * VALUE_BYTES * (inputs.hasEnforcedPatterns ? 2 : 1);
    const std::uint64_t reduced = boundary * boundary * VALUE_BYTES +
                                  boundary * (patterns + RIGID_BODY_MODE_COUNT) * VALUE_BYTES + boundary * LABEL_BYTES;
    const std::uint64_t work = patterns * patterns * VALUE_BYTES;
    footprint.keptBytes = withMargin(system, ARRAY_MARGIN) + patternBytes + reduced + work;
    footprint.lightBytes = withMargin(light, ARRAY_MARGIN) + patternBytes + reduced + work;
    // Craig-Bampton: the mass on the equations (the structure of the stiffness: its values and row indices) and the
    // reduced boundary mass M_aa. A Craig-Bampton unit keeps its equations (never released).
    if (inputs.keepsMass)
    {
        footprint.keptBytes +=
            withMargin(matrixValues * (INDEX_BYTES + VALUE_BYTES), ARRAY_MARGIN) + boundary * boundary * VALUE_BYTES;
        footprint.lightBytes = footprint.keptBytes;
    }

    footprint.buildBytes =
        withMargin(points * BUILD_POINT_BYTES + blocks * BUILD_BLOCK_BYTES + localDofs * BUILD_DOF_BYTES, ARRAY_MARGIN);

    // The partial factorisation, the loads of the patterns and of the inertial patterns on the equations with the
    // copies their forward sweep works on, the two sweeps of the pattern work, two work vectors of the local DOFs.
    const std::uint64_t loadColumns = patterns + RIGID_BODY_MODE_COUNT;
    footprint.factorisationBytes = withMargin(cost.factorisationBytes, FACTOR_MARGIN) +
                                   2 * equations * loadColumns * VALUE_BYTES + 2 * equations * patterns * VALUE_BYTES +
                                   2 * localDofs * VALUE_BYTES;
    // Craig-Bampton: the blocks of constraint modes, their products and condensations (three arrays of
    // CONSTRAINT_MODE_BLOCK columns on the equations, one on the boundary) and the projection M_aa.
    if (inputs.keepsMass)
    {
        footprint.factorisationBytes += (3 * equations + boundary) * CraigBampton::CONSTRAINT_MODE_BLOCK * VALUE_BYTES +
                                        boundary * boundary * VALUE_BYTES;
    }

    // Without its factor: the streamed factorisation with the forward sweep of its load columns, the loads
    // on the equations swept in place, the pattern work (its two sweeps from the swept loads and the pivots), two work
    // vectors of the local DOFs.
    footprint.withoutFactorBytes = withMargin(cost.withoutFactor.bytes(loadColumns), FACTOR_MARGIN) +
                                   equations * loadColumns * VALUE_BYTES + patterns * patterns * VALUE_BYTES +
                                   2 * localDofs * VALUE_BYTES;
    footprint.largestPanelBytes =
        withMargin(asBytes(cost.withoutFactor.largestPanelValues) * VALUE_BYTES, FACTOR_MARGIN);

    // The interior factor once made: its panels and row indices, its per-supernode arrays, its solve plan, the
    // factorisation's K*.
    footprint.factorBytes = withMargin(asBytes(cost.interiorFactorValueCount) * VALUE_BYTES +
                                           asBytes(cost.factorRowCount) * (INDEX_BYTES + SOLVE_PLAN_ROW_BYTES) +
                                           asBytes(cost.supernodeCount) * SUPERNODE_BYTES,
                                       FACTOR_MARGIN) +
                            boundary * boundary * VALUE_BYTES;

    // The recovery (UnitRecovery): B basis columns, B ≤ min(Q, n_a) + k, their solutions on the equations, their
    // boundary values, their enforced values on the local DOFs, their coefficients for every case, and the loads of the
    // patterns on the equations (the recovery's, and the run's for the interior residuals); with as many cases as
    // columns or fewer and patterns of its own it may solve the cases instead (regime "load cases"), holding the loads
    // of its patterns and its interior factor through the cases: the larger of the two. Without a pattern of its own
    // that regime is never taken (its basis columns, no more than the cases, cost no more than the cases),
    // and its factor is released once the basis is solved.
    if (inputs.isRecovered)
    {
        const std::uint64_t columns = std::min(asBytes(inputs.basisColumnCount), boundary) + patterns; // the basis B
        const std::uint64_t cases = asBytes(inputs.caseCount);
        const std::uint64_t basisData = (equations * (columns + 2 * patterns) + boundary * columns +
                                         (inputs.hasEnforcedPatterns ? localDofs * columns : 0) + columns * cases) *
                                        VALUE_BYTES;
        const std::uint64_t caseData = 2 * equations * patterns * VALUE_BYTES;
        footprint.recoveryBytes = basisData;
        if (cases <= columns && patterns > 0)
            footprint.recoveryBytes = std::max(footprint.recoveryBytes, caseData + footprint.factorBytes);
        // Its recovery without its factor: the streamed factorisation of the forward sweep on the columns
        // with an interior load and their copy on its equations, the regions' panels apart. Those columns: a basis
        // column that is one of its patterns and a pattern no column carries (UnitRecovery's BasisColumns: 2k at most),
        // the cases (LoadCases), its patterns (BoundaryDofs: the boundary DOFs' columns have none); never more than B.
        const std::uint64_t swept = std::min(columns, 2 * patterns + cases);
        footprint.regionFixedBytes =
            withMargin(cost.withoutFactor.bytes(swept), FACTOR_MARGIN) + equations * swept * VALUE_BYTES;
        // In sequence the factor is counted apart (passPeakBytes); one case's vectors beside the recovery data.
        footprint.passBytes =
            std::max(basisData, caseData) +
            (PASS_EQUATION_VECTORS * equations + PASS_LOCAL_VECTORS * localDofs + heldDofs + boundary) * VALUE_BYTES;
    }
    // The load basis handed over: a block's recovery of C columns. The direct regime solves the C
    // columns on its equations, the regime of H only when its boundary, the loaded columns and the six inertial ones
    // are fewer than C (so its motions and own columns hold fewer than C columns on the equations); the boundary values
    // and their negation, its interiors kept for the block on its local DOFs, its six inertial loads.
    if (inputs.readColumnCount > 0)
    {
        const std::uint64_t columns = asBytes(inputs.readColumnCount);
        footprint.isRecovered = true;
        footprint.recoveryBytes = footprint.factorBytes;
        footprint.passBytes = (2 * equations * columns + 2 * boundary * columns + localDofs * columns +
                               RIGID_BODY_MODE_COUNT * equations) *
                              VALUE_BYTES;
        footprint.regionFixedBytes =
            withMargin(cost.withoutFactor.bytes(columns), FACTOR_MARGIN) + equations * columns * VALUE_BYTES;
    }
    return footprint;
}

} // namespace oos::reduction
