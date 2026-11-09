#include "oos/reduction/scheduling/InterfaceFootprint.hpp"

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
constexpr std::uint64_t BLOCK_VALUES = 36;
constexpr std::uint64_t RIGID_BODY_MODE_COUNT = 6;
// Per DOF of the g set, what a constraint set holds there beside its relations (InterfaceFootprintInputs::
// relationBytes, measured, their Gm's row of each DOF included): the exclusive set of the DOF sets (1) and the held
// flags (1); with an m set, the vector of the MPC forces (8), and the vector of the element forces they and the grid
// point forces of the units not recovered are computed from (8). Step 10.21 measured them on the budget's labels of the
// 3 000-plate assemblies (128 bytes a DOF before, the case vectors counted with them).
constexpr std::uint64_t MODEL_DOF_BYTES = 2;
constexpr std::uint64_t FORCE_VECTOR_DOF_BYTES = sizeof(double);
// Per point of the model: the interface's point of each node (4) and the interior flags of the plan (1); rounded up.
constexpr std::uint64_t MODEL_NODE_BYTES = 8;
// Per interface point: InterfacePoint (its node, kind, DOF count, position: 40), first DOF (8), the graph's offsets
// (8), the ordering's arrays (about 40); rounded up.
constexpr std::uint64_t POINT_BYTES = 128;
// Per local DOF: its role (1), the DofSelection (12), the rigid-body modes and inertial loads (96), the held rows
// (their offsets: 8); rounded up.
constexpr std::uint64_t DOF_BYTES = 128;
constexpr std::uint64_t SUPERNODE_BYTES = 80;
// The plan's factor within 10 % of the measured one, and within 2 % on the interfaces of the 3 000-plate
// assemblies (444 MB planned, 454 MB made at 1.5 M DOFs): half again of that; a quarter for the arrays,
// whose sizes the plan approximates (the points with no free DOF are not in its graph). Step 10.21 measured the
// interface phase within 20 % of the estimate (1.2 and 1.5 before: 30 %).
constexpr double FACTOR_MARGIN = 1.15;
constexpr double ARRAY_MARGIN = 1.25;
// Per row of the factor's panels, its solve plan (each panel row's place in its unit, the accumulator
// rows, the outside rows): three indices, rounded up.
constexpr std::uint64_t SOLVE_PLAN_ROW_BYTES = 16;

std::uint64_t withMargin(std::uint64_t bytes, double margin)
{
    return static_cast<std::uint64_t>(std::ceil(static_cast<double>(bytes) * margin));
}

std::uint64_t asBytes(std::int64_t count)
{
    return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

} // namespace

std::uint64_t InterfaceFootprint::solveBytes(std::uint64_t regionBytes) const noexcept
{
    const auto values = static_cast<std::int64_t>(std::min<std::uint64_t>(regionBytes, factorBytes) / VALUE_BYTES);
    return withMargin(
        streamedSolve.bytes(static_cast<std::size_t>(std::max<std::int64_t>(basisColumnCount, 1)), values),
        FACTOR_MARGIN);
}

double InterfaceFootprint::recomputedFlops(std::uint64_t regionBytes) const noexcept
{
    if (regionBytes >= factorBytes)
        return 0.0;
    double flops = regionCosts.empty() ? 0.0 : regionCosts.front().recomputedFlops;
    for (const linalg::RegionSweepCost& cost : regionCosts)
        if (static_cast<std::uint64_t>(cost.regionValues) * VALUE_BYTES <= regionBytes)
            flops = cost.recomputedFlops;
    return flops;
}

InterfaceFootprint InterfaceFootprint::estimate(const InterfaceFootprintInputs& inputs)
{
    if (inputs.estimate == nullptr)
        throw std::invalid_argument(
            "InterfaceFootprint::estimate: the plan's interface estimate is needed; give "
            "CondensationPlan::interfaceEstimate");
    const InterfaceEstimate& plan = *inputs.estimate;
    const std::uint64_t points = asBytes(inputs.pointCount);
    const std::uint64_t localDofs = asBytes(inputs.localDofCount);
    const std::uint64_t equations = asBytes(plan.dofCount);
    const std::uint64_t cases = asBytes(inputs.caseCount);
    const std::uint64_t freePoints = std::max<std::uint64_t>(asBytes(plan.nodeCount), 1);
    const std::uint64_t blocks =
        asBytes(plan.blockCount) * std::max<std::uint64_t>(points, freePoints) / freePoints + points;
    const std::uint64_t matrixValues = asBytes(plan.matrixValueCount);

    InterfaceFootprint footprint;
    footprint.modelBytes =
        asBytes(inputs.modelDofCount) * (MODEL_DOF_BYTES + (inputs.hasMset ? FORCE_VECTOR_DOF_BYTES : 0) +
                                         (inputs.needsElementForces ? FORCE_VECTOR_DOF_BYTES : 0)) +
        asBytes(inputs.modelNodeCount) * MODEL_NODE_BYTES + inputs.relationBytes;

    // The interface problem: what it keeps, and the largest of its three stages.
    const std::uint64_t pattern = points * 2 * OFFSET_BYTES + blocks * (INDEX_BYTES + OFFSET_BYTES);
    const std::uint64_t graph = (points + 1) * OFFSET_BYTES + 2 * blocks * INDEX_BYTES;
    const std::uint64_t blockValues = blocks * BLOCK_VALUES * VALUE_BYTES;
    const std::uint64_t csc = (equations + 1) * OFFSET_BYTES + matrixValues * (INDEX_BYTES + VALUE_BYTES);
    const std::uint64_t symbolic =
        asBytes(plan.factorRowCount) * 2 * INDEX_BYTES + asBytes(plan.supernodeCount) * SUPERNODE_BYTES;
    const std::uint64_t kept =
        withMargin(points * POINT_BYTES + localDofs * DOF_BYTES + graph + symbolic, ARRAY_MARGIN);
    const std::uint64_t assembly = withMargin(pattern + 2 * blockValues, ARRAY_MARGIN);
    const std::uint64_t conversion = withMargin(pattern + blockValues + csc, ARRAY_MARGIN);
    const std::uint64_t factorisation =
        withMargin(csc, ARRAY_MARGIN) + withMargin(plan.factorisationBytes, FACTOR_MARGIN);
    footprint.problemBytes = kept + std::max({assembly, conversion, factorisation});
    footprint.heldBytes = kept + withMargin(csc, ARRAY_MARGIN) +
                          withMargin(asBytes(plan.factorValueCount) * VALUE_BYTES +
                                         asBytes(plan.factorRowCount) * (INDEX_BYTES + SOLVE_PLAN_ROW_BYTES) +
                                         asBytes(plan.supernodeCount) * SUPERNODE_BYTES,
                                     FACTOR_MARGIN);
    // Without the factor: no factorisation while it is made, no factor held; its solves' streamed
    // factorisation (its skeleton, pivots, fronts and regions) in solveBytes.
    footprint.problemWithoutFactorBytes = kept + std::max(assembly, conversion);
    footprint.heldWithoutFactorBytes = kept + withMargin(csc, ARRAY_MARGIN);
    footprint.streamedSolve = plan.streamedSolve;
    footprint.regionCosts = plan.regionCosts;
    footprint.factorBytes = asBytes(plan.streamedSolve.panelValues) * VALUE_BYTES;
    footprint.leastRegionBytes = asBytes(plan.streamedSolve.run.largestPanelValues) * VALUE_BYTES;

    // The boundary basis on its columns C = min(Q, cases): V and P_n on the local DOFs, F, R and the solutions on the
    // equations, the loads and enforced values of the Q right-hand sides (and of the cases when they are solved
    // instead), β; each unit's boundary rows on the columns, the coefficients of the units and of the interface, the
    // interface's load vectors (loads, enforced values, mechanical ones).
    std::uint64_t distinct =
        asBytes(inputs.interfaceVectorCount) + (inputs.hasInertiaRelief ? RIGID_BODY_MODE_COUNT : 0);
    std::uint64_t unitRows = 0;
    std::uint64_t coefficients = asBytes(inputs.interfaceVectorCount) * cases;
    for (const InterfaceUnitSize& unit : inputs.units)
    {
        distinct += asBytes(unit.patternCount);
        coefficients += (2 * asBytes(unit.patternCount) + RIGID_BODY_MODE_COUNT) * cases;
    }
    const std::uint64_t columns = std::min(distinct, cases);
    for (const InterfaceUnitSize& unit : inputs.units)
        unitRows += asBytes(unit.boundaryDofCount) * columns;
    footprint.basisColumnCount = static_cast<std::int64_t>(columns);
    const std::uint64_t basis = (2 * localDofs + 3 * equations) * columns + 2 * localDofs * distinct +
                                (distinct > cases ? 2 * localDofs * cases : 0) + distinct * cases;
    const std::uint64_t vectors = 3 * localDofs * asBytes(inputs.interfaceVectorCount);
    const std::uint64_t relief = inputs.hasInertiaRelief ? (4 * localDofs + 2 * equations) * RIGID_BODY_MODE_COUNT +
                                                               2 * RIGID_BODY_MODE_COUNT * cases
                                                         : 0;
    footprint.basisBytes = (basis + unitRows + coefficients + vectors + relief) * VALUE_BYTES;
    return footprint;
}

} // namespace oos::reduction
