#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/loads/LoadBasis.hpp"
#include "oos/assembly/loads/LoadFamily.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/output/arrays/GridPointForceTable.hpp"
#include "oos/solutions/sol101/CondensedFactorisedSet.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace oos::solutions
{

// One unit of the load basis handed over: one vector of the load basis of a constraint set, named by its family, the
// basis kind its family took and its source: a load set as written (loadSet), a unit load or enforced displacement on
// one DOF (grid, component in the displacement system of the grid point; component 0 for a scalar point), or the family
// part of a subcase (subcase: a case of the Cases basis, a pivot of the Rank basis, or a case that joined the basis by
// the reconstruction check).
struct LoadBasisUnit
{
    assembly::LoadFamily family = assembly::LoadFamily::Concentrated;
    assembly::LoadBasisKind kind = assembly::LoadBasisKind::Empty;
    assembly::BasisVector::Source source = assembly::BasisVector::Source::Case;
    base::EntityId loadSet = 0;
    base::EntityId grid = 0;
    std::int32_t component = 0;
    base::EntityId subcase = 0;
    // A unit of the thermal family (StaticOptions::keepsThermalUnits): the sets of its subcase's initial strains,
    // TEMPERATURE(LOAD) and TEMPERATURE(INITIAL) and DEFORM, whose temperatures and element deformations its element
    // tables are recovered with (temperatureLoad and deformation 0: none of them; both 0: not a thermal unit).
    base::EntityId temperatureLoad = 0;
    std::optional<base::EntityId> temperatureInitial;
    base::EntityId deformation = 0;

    [[nodiscard]] bool isThermal() const noexcept { return temperatureLoad != 0 || deformation != 0; }
};

// The units of one constraint set's load basis on the condensed path (unit blocks, recovered lazily): the condensed
// constraint set kept with its interface factor and its units (CondensedFactorisedSet), the boundary basis solutions
// (the interface's displacements of every unit: local DOFs × units) and, per condensed unit that some unit loads
// inside, those units (increasing positions in the set's units) and their loads on its equations
// (UnitSystem::equationLoads: equationCount() × those units). A unit's displacements are those of the interface's
// points and, for the interiors of the condensed units a block reads, u_o = K_oo⁻¹·(f_o − K_oa·u_a), recovered on
// demand (CondensedBasisColumns).
struct KeptCondensedBasis
{
    struct UnitLoads
    {
        std::vector<std::size_t> columns;
        linalg::DenseMatrix equations;
    };
    std::shared_ptr<const CondensedFactorisedSet> set;
    linalg::DenseMatrix interfaceDisplacements;
    std::vector<UnitLoads> unitLoads; // per condensed unit of the set (CondensedFactorisedSet::units order)
    // Inertia relief (the set's CondensedRelief): the amplitudes a = C·ü of the rigid-body accelerations of
    // each unit (6 × units; no column without relief), whose loads −M·Φ·a load every unit's interior; and the rigid
    // motion that brings each unit's displacements back to the gauge of the FreeBodySupport DOFs, −Φ·C·u_r (6 × units;
    // no column unless the interface holds conditioned supports).
    linalg::DenseMatrix reliefAmplitudes;
    linalg::DenseMatrix gaugeAmplitudes;
};

// What a load basis is asked to keep of its units' fixed rows of grid point forces (StaticOptions::basisFixedRows):
// nothing (the element rows alone), every point (`isAll`: the rows of GPFORCE(ALL)), or the grid points `points` (IDs)
// and the points of the relations that reach one of them or straddle one of `planes` (a point and a normal in the basic
// system: the plane cuts declared; a relation straddles a plane when it has points strictly behind it and points not
// behind it).
struct BasisFixedRowRequest
{
    bool isRequested = false;
    bool isAll = false;
    // The multiPointConstrForce rows split by relation (a snapshot of O_o_Reader's data contract 56 or later): set by
    // the solution from the snapshot's contract, not by the caller.
    bool isMpcSplit = false;
    std::vector<base::EntityId> points;
    std::vector<std::array<double, 6>> planes;
};

// The fixed rows of the units' grid point forces kept with a constraint set's load basis: the units' applied loads at
// `dofs` (every DOF of the points where rows are given and of the dependent points of the relations kept: appliedLoad
// as statics writes it, inertia relief's loads included; on the condensed path the relief loads of the units are added
// when they are recovered, CondensedBasisColumns::addUnitReliefLoads), which of them are held (the s set and the r set:
// singlePointConstrForce), and the rows of Gm of the relations that reach the points (u_m = Gm·u_n: an m-set DOF and
// its terms), so that a recovery makes the rows at `nodes` as statics does:
//     q_m = (K·u − P)_m, q_n = −Gmᵀ·q_m (RigidElimination::computeMpcForces), q_s = (K·u − P − q_mpc)_s,
// K·u summed from the elements at those points (StiffnessRecovery::pointForces). With `isAll` every DOF is kept
// (`dofs` empty, `loads` g × units) and every row of Gm.
//
// Memory: budgeted in the run's budget (KeptLoadBasis). Thread-safety: immutable once built.
struct KeptFixedRows
{
    bool isAll = false;
    base::BudgetedBuffer<std::int32_t> nodes;      // the DofMap nodes whose rows are given, increasing (not with isAll)
    base::BudgetedBuffer<base::DofIndex> dofs;     // increasing (not with isAll)
    base::BudgetedBuffer<std::uint8_t> isHeld;     // per entry of dofs (with isAll per g-set DOF)
    linalg::DenseMatrix loads;                     // dofs × units (with isAll g × units)
    base::BudgetedBuffer<base::DofIndex> mDofs;    // the relations kept: their m-set DOFs, increasing
    base::BudgetedBuffer<std::int64_t> offsets;    // mDofs + 1
    base::BudgetedBuffer<base::DofIndex> termDofs; // the terms of each row on the n set
    base::BudgetedBuffer<double> termCoefficients;
    base::BudgetedBuffer<std::int32_t> mNodes; // the nodes of mDofs, increasing, each once
    // The multiPointConstrForce rows split by relation (BasisFixedRowRequest::isMpcSplit;
    // RigidElimination::forEachRelationForce): each kept row's relation, and with chains its own terms and the kept
    // rows of the chains, parents first (empty without: the terms are termDofs').
    bool isMpcSplit = false;
    base::BudgetedBuffer<std::int32_t> relations;  // per kept row
    base::BudgetedBuffer<std::int64_t> ownOffsets; // mDofs + 1, with chains
    base::BudgetedBuffer<base::DofIndex> ownTermDofs;
    base::BudgetedBuffer<double> ownTermCoefficients;
    base::BudgetedBuffer<std::int64_t> chainOrder; // kept rows, with chains

    // The position of a DOF in `dofs` (the DOF itself with isAll), or nothing when it is not kept.
    [[nodiscard]] std::optional<std::size_t> placeOf(base::DofIndex dof) const noexcept
    {
        if (isAll)
            return static_cast<std::size_t>(dof);
        const base::DofIndex* found = std::lower_bound(dofs.data(), dofs.data() + dofs.size(), dof);
        if (found == dofs.data() + dofs.size() || *found != dof)
            return std::nullopt;
        return static_cast<std::size_t>(found - dofs.data());
    }
    // The count of kept DOFs (the rows of `loads`).
    [[nodiscard]] std::size_t keptDofCount() const noexcept { return loads.rows(); }
    [[nodiscard]] std::uint64_t bytes() const noexcept
    {
        return loads.leadingDimension() * loads.columns() * sizeof(double) + nodes.size() * sizeof(std::int32_t) +
               dofs.size() * sizeof(base::DofIndex) + isHeld.size() + mDofs.size() * sizeof(base::DofIndex) +
               offsets.size() * sizeof(std::int64_t) + termDofs.size() * (sizeof(base::DofIndex) + sizeof(double)) +
               mNodes.size() * sizeof(std::int32_t) + relations.size() * sizeof(std::int32_t) +
               ownOffsets.size() * sizeof(std::int64_t) +
               ownTermDofs.size() * (sizeof(base::DofIndex) + sizeof(double)) +
               chainOrder.size() * sizeof(std::int64_t);
    }
};

// What the full path streamed keeps of a constraint set's load basis with declared quantities: no displacement (its
// regions are gone), the declared quantities of every unit (the force tables of the map's elements reduced as the
// downward pass made them: units × quantityCount, row-major) and the grid point forces of every unit at the declared
// points, with their fixed rows, as LoadBasisRecovery::recover gives them.
struct KeptStreamedBasis
{
    std::size_t quantityCount = 0;
    std::size_t termCount = 0; // the map's terms (its identity with quantityCount)
    base::BudgetedBuffer<double> quantities;
    std::vector<output::GridPointForceTable>
        gridPointForces; // per unit of the set, in its order (bounded by the basis)
    // The DofMap nodes whose rows the tables give, increasing.
    std::vector<std::int32_t> rowNodes; // bounded by the declared points
};

// The load basis of one constraint set: its solved subcases (the columns of the coefficients, in solve order), its
// units (the rows of the coefficients and the columns of the displacements, in the order of the basis: family, then
// index in the family), the coefficients α (units × cases: case j is Σ_i α(i, j) · unit i, the numbers the analysis
// combines) and the g-set displacements of the units (g × units, in the displacement systems of the grid points; with
// inertia relief, the relieved solution of each unit). On the condensed path the displacements are empty and
// `condensed` holds what gives them on demand.
struct KeptLoadBasisSet
{
    std::int32_t constraintSet = 0;
    std::vector<base::EntityId> cases;
    std::vector<LoadBasisUnit> units;
    linalg::DenseMatrix coefficients;
    linalg::DenseMatrix displacements;
    std::shared_ptr<const KeptCondensedBasis> condensed; // the condensed path (null on the full path)
    std::shared_ptr<const KeptFixedRows> fixedRows;      // null: the element rows alone
    std::shared_ptr<const KeptStreamedBasis> streamed;   // the full path streamed (null otherwise)
};

// A SUBCOM of the run (SUBCOM, SUBSEQ): its label and the subcases it combines with their coefficients.
struct KeptCombination
{
    base::EntityId label = 0;
    std::vector<base::EntityId> subcases;
    std::vector<double> coefficients;
};

// What a static run keeps of its load basis (StaticOptions::keepsLoadBasis): the bases of its constraint sets, its
// SUBCOMs, the subcases it left out (a thermal load, StaticOptions::excludesThermalCases) and what the recovery of the
// units needs beyond the snapshot (the index of the points and the DOF map, shared with the run).
//
// Memory: the displacements (on the condensed path: the kept sets with their units and interior factors, the boundary
// basis solutions and the units' loads) are budgeted in the run's memory budget until the last owner drops them (the
// budget must outlive them).
struct KeptLoadBasis
{
    std::vector<KeptLoadBasisSet> sets;
    std::vector<KeptCombination> combinations;
    std::vector<base::EntityId> excludedCases;
    std::shared_ptr<const elements::SnapshotIndex> index;
    std::shared_ptr<const assembly::DofMap> dofMap;
    // The condensed path: the elements of each condensed unit (by element ID, increasing), with the unit's index in
    // the units of the kept sets (CondensedFactorisedSet::units, the same for every set of the run).
    struct CondensedElement
    {
        base::EntityId elementId = 0;
        std::uint32_t unit = 0;
    };
    std::vector<CondensedElement> condensedElements;
    // The estimate checked before the analysis (StaticSolution: what the units keep and hold while they are made).
    std::uint64_t estimatedBytes = 0;

    // The bytes the units keep: their displacements, or on the condensed path the boundary basis solutions and the
    // units' loads (the kept sets' factors and units are the condensed analysis's own).
    [[nodiscard]] std::uint64_t keptBytes() const noexcept
    {
        std::uint64_t bytes = 0;
        for (const KeptLoadBasisSet& set : sets)
        {
            bytes += set.displacements.leadingDimension() * set.displacements.columns() * sizeof(double);
            if (!set.condensed)
                continue;
            const linalg::DenseMatrix& boundary = set.condensed->interfaceDisplacements;
            bytes += boundary.leadingDimension() * boundary.columns() * sizeof(double);
            for (const KeptCondensedBasis::UnitLoads& loads : set.condensed->unitLoads)
                bytes += loads.equations.leadingDimension() * loads.equations.columns() * sizeof(double);
        }
        for (const KeptLoadBasisSet& set : sets)
        {
            if (set.fixedRows)
                bytes += set.fixedRows->bytes();
            if (set.streamed)
            {
                bytes += set.streamed->quantities.size() * sizeof(double);
                for (const output::GridPointForceTable& table : set.streamed->gridPointForces)
                    bytes += table.rowCount() * (3 * sizeof(std::int32_t) + 6 * sizeof(double));
            }
        }
        return bytes;
    }
};

} // namespace oos::solutions
