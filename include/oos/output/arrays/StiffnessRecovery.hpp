#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/ElementResultTable.hpp"
#include "oos/output/arrays/GridPointForceTable.hpp"
#include "oos/output/arrays/OutputPlan.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::output
{

// The outputs of a static subcase that come from the element stiffness matrices themselves:
// - GPFORCE, the grid point forces (GridPointForceTable): the force each element exerts on each of its grid points,
//   −(K_e·u_e − f_e,T) with f_e,T the element's thermal load under the subcase's temperatures, with the applied loads,
//   SPC and MPC forces of the grid point and their total, which the equilibrium makes zero (appliedLoad +
//   singlePointConstrForce + multiPointConstrForce = Kgg·ug − Σ f_T = Σ_e (K_e·u_e − f_e,T));
// - ESE, the strain energy of each element, E = ½·u_eᵀ·K_e·u_e, its percentage of the total of every element of the
//   model, and its density, the energy over the element volume (assembly::ElementVolumes; NaN for an element without
//   a volume, a spring or a bush); each element type's table ends with its subtotal row (element ID 100000000, as the
//   result tables carry it, with no density). With a thermal load the energy is that of the total strain (the energy
//   of the mechanical strain would need the thermal strain field of each element).
// The element matrices are those of the assembly (the stiffness facet, in the displacement systems of the grid
// points), computed again range by range (a bounded block each), the ranges of a wave on the pool given, and their
// forces K_e·u_e used in element order: the same bits for any thread count.
//
// Thread-safety: const methods are safe concurrently. Ownership: borrows the element set, the DOF map and the pool.
class StiffnessRecovery
{
public:
    static constexpr std::int64_t RANGE_ELEMENTS = 2048;
    // With a pool, a batch is cut into ranges of at least MIN_RANGE_ELEMENTS, about RANGES_PER_THREAD per thread, and
    // the ranges are computed a wave of one per thread at a time (a wave holds their element matrices).
    static constexpr std::int64_t MIN_RANGE_ELEMENTS = 64;
    static constexpr std::int64_t RANGES_PER_THREAD = 4;
    // strainEnergies() holds the energies of a group of vectors (one value per element and vector) within this
    // fraction of the memory left (1/8): the modes of an eigenvalue solution take few groups, a huge model one vector
    // at a time.
    static constexpr std::uint64_t STRAIN_ENERGY_MEMORY_DIVISOR = 8;

    // `pool`: the threads that compute the element matrices (null: the caller's).
    StiffnessRecovery(const assembly::ElementSet& set, const assembly::DofMap& dofMap, base::ThreadPool* pool = nullptr)
        : m_set(&set), m_dofMap(&dofMap), m_pool(pool)
    {
    }

    // The g-set vectors of one subcase: displacements, applied loads (OLOAD), SPC forces, MPC forces (empty when the
    // model has no m set), and the flags of the constrained DOFs (nonzero for the s set and the r set, one per g-set
    // DOF). `request` the GPFORCE request (its grid points). Throws std::invalid_argument for a vector not of the
    // g-set size. `isMpcSplit` (for an O_o_Reader of data contract 56 or later): the multiPointConstrForce row of each
    // grid point split by relation from `mpcRelationForces` (RigidElimination::relationForces of `mpcForces`, or a
    // combination of such lists): a row per rigid element or MPC set acting at the point, a zero one included, its
    // element ID the relation's (RigidElimination::relationOf), and one row of ID 0 for the other constraint rows (a
    // CFAST's, the glue) when nonzero; total the sum with the point's MPC force as before, so its bits do not change.
    // Without: one multiPointConstrForce row a point.
    struct NodalState
    {
        std::span<const double> displacements;
        std::span<const double> appliedLoads;
        std::span<const double> spcForces;
        std::span<const double> mpcForces;
        std::span<const std::uint8_t> isConstrained;
        const elements::ElementTemperatures* temperatures = nullptr;
        bool isMpcSplit = false;
        std::span<const assembly::RelationForce> mpcRelationForces;
    };
    [[nodiscard]] GridPointForceTable gridPointForces(std::int32_t subcase, const NodalState& state,
                                                      const OutputRequestPlan& request, base::MemoryBudget& budget,
                                                      const base::CancellationToken* cancellation = nullptr) const;

    // What the condensed path changes in the grid point forces: the elements of the condensed units not recovered,
    // whose displacements are not known, are left out, and the force of each such unit on its boundary points,
    // −(K*·u_a − f*·α) (known without recovering the unit's interior), is a row of its own, named UNIT_ROW_NAME with
    // the unit's ID in the element column, after the element rows of its point.
    static constexpr const char* UNIT_ROW_NAME = "condensedUnit";
    struct UnitRow
    {
        std::int32_t node = 0; // the DofMap node of the boundary point
        std::int32_t unitId = 0;
        std::array<double, 6> values{}; // the force of the unit on the point, in its displacement system
    };
    struct CondensedUnits
    {
        std::span<const std::int32_t> excludedElements; // sorted element IDs
        std::span<const UnitRow> unitRows;              // sorted by node
    };
    // gridPointForces with the condensed units not recovered (CondensedUnits).
    [[nodiscard]] GridPointForceTable gridPointForces(std::int32_t subcase, const NodalState& state,
                                                      const OutputRequestPlan& request, const CondensedUnits& units,
                                                      base::MemoryBudget& budget,
                                                      const base::CancellationToken* cancellation = nullptr) const;

    // One element's force on one grid point, −K_e·u_e in the point's displacement system (pointForces).
    struct PointContribution
    {
        std::int32_t node = 0; // the DofMap node
        std::int32_t elementId = 0;
        std::int32_t name = 0; // into PointForces::names
        std::array<double, 6> values{};
    };
    // The element forces of several vectors at some grid points (the units of a load basis).
    struct PointForces
    {
        // The names of a grid point force table: the fixed rows' (appliedLoad, singlePointConstrForce,
        // multiPointConstrForce, total), then the element types' in the order they come.
        std::vector<std::string> names;
        std::vector<std::vector<PointContribution>> columns; // per vector: by node, the elements' order within a node
    };
    // The names of the fixed rows come first in every table (indices 0 to 3): appliedLoad, singlePointConstrForce,
    // multiPointConstrForce, total.
    static constexpr std::int32_t FIRST_ELEMENT_NAME = 4;
    // The forces −K_e·u_e at the nodes `isNodeTaken` flags (one flag per DofMap node) of the elements `rowsOfBatch`
    // lists (per batch of the set, its rows in increasing ranges: the elements with a grid point there; the others
    // are not computed), for each column of `displacements` (g × n): each element's matrix made once for every column
    // (the ranges of a wave on the pool), each column's forces the chain of gridPointForces, visited in element order:
    // the same bits as gridPointForces for each column, whatever the threads. Throws std::invalid_argument for
    // displacements not of the g-set size, flags not one per node or ranges not one list per batch. With `window` (the
    // chunks of the full path streamed), `displacements` are on the window's slots (6 rows a grid point's
    // slot, then one a scalar point's), every node of the elements given holding one (std::invalid_argument
    // otherwise).
    [[nodiscard]] PointForces pointForces(std::span<const std::uint8_t> isNodeTaken,
                                          std::span<const std::vector<elements::ElementRange>> rowsOfBatch,
                                          linalg::ConstDenseMatrixView displacements, base::MemoryBudget& budget,
                                          const base::CancellationToken* cancellation = nullptr,
                                          const RecoveryWindow* window = nullptr) const;
    // The multiPointConstrForce row of one relation at a grid point (NodalState::isMpcSplit): its element ID (a rigid
    // element's, minus an MPC set's, 0 for the other constraint rows) and its force in the point's displacement system.
    struct RelationRow
    {
        std::int32_t relation = 0;
        std::array<double, 6> values{};
    };
    // The fixed rows of one vector at one grid point (pointForceTable), in its displacement system: appliedLoad,
    // singlePointConstrForce (written when the point has a held DOF) and multiPointConstrForce (written when nonzero),
    // as gridPointForces writes them; with `isMpcSplit` the multiPointConstrForce rows are `mpcByRelation` (by
    // increasing relation; the one of ID 0 written when nonzero), total still adding `mpc`.
    struct PointFixedRows
    {
        std::int32_t node = 0;
        std::array<double, 6> applied{};
        std::array<double, 6> spc{};
        std::array<double, 6> mpc{};
        bool isHeld = false;
        bool isMpcSplit = false;
        std::vector<RelationRow> mpcByRelation; // bounded by the relations at the point
    };
    // The grid point force table of column `column` of `forces` at the nodes `rowNodes` (increasing): a node of
    // `fixedRows` (sorted by node) has the rows of gridPointForces (appliedLoad, singlePointConstrForce,
    // multiPointConstrForce, its element rows, total); any other node its element rows alone (the units of a load basis
    // that keep no fixed row there).
    [[nodiscard]] GridPointForceTable pointForceTable(std::int32_t subcase, const PointForces& forces,
                                                      std::size_t column, std::span<const std::int32_t> rowNodes,
                                                      std::span<const PointFixedRows> fixedRows,
                                                      base::MemoryBudget& budget) const;

    // Appends to `tables` the strain energy tables of one subcase, one per element type with a stiffness, for the
    // elements of `request` (the ESE request).
    void strainEnergy(std::int32_t subcase, std::span<const double> displacements, const OutputRequestPlan& request,
                      base::MemoryBudget& budget, std::vector<ElementResultTable>& tables,
                      const base::CancellationToken* cancellation = nullptr) const;
    // strainEnergy() for every column of `vectors` (g × columns: the modes of an eigenvalue solution, the cases of a
    // subcase), into tables[j] for column j (resized to the columns): each element's stiffness computed once for a
    // group of columns (STRAIN_ENERGY_MEMORY_DIVISOR; at most `largestGroup` columns when it is not 0), the energies
    // of a column the chains of strainEnergy(), so the same tables to the bit whatever the group or the threads.
    // Throws std::invalid_argument for vectors not of the g-set size.
    void strainEnergies(std::int32_t subcase, linalg::ConstDenseMatrixView vectors, const OutputRequestPlan& request,
                        base::MemoryBudget& budget, std::vector<std::vector<ElementResultTable>>& tables,
                        const base::CancellationToken* cancellation = nullptr, std::size_t largestGroup = 0) const;

private:
    const assembly::ElementSet* m_set;
    const assembly::DofMap* m_dofMap;
    base::ThreadPool* m_pool;
};

} // namespace oos::output
