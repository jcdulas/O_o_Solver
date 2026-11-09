#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/ordering/NodeGroupList.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/output/sections/ContourReport.hpp"
#include "oos/output/sections/ZoneDefinition.hpp"
#include "oos/reduction/external/ExternalUnit.hpp"
#include "oos/reduction/external/ExternalUnitAssembly.hpp"
#include "oos/reduction/external/ExternalUnitChecks.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::solutions
{

class StaticLoadCases;

// An external unit of a static analysis (StaticOptions::externalUnits): the zone whose elements it replaces, as for the
// contour data (its definition with O_o_Reader's boundary; its contour DOFs are taken in the displacement systems of
// the grid points whatever its frameId), and the plugin's matrices.
struct ExternalUnitRequest
{
    output::ZoneDefinition zone;
    std::shared_ptr<const reduction::ExternalUnit> unit;
};

// The load basis of one constraint set as an external unit saw it: the loads of its zone's interior DOFs per basis
// vector (what its load source was given), its condensed loads f* per basis vector, and the coefficients α of the
// solved subcases of the set (case j is Σ_i α(i, j) · basis vector i).
struct ExternalUnitBasis
{
    std::int32_t constraintSet = 0;
    std::vector<base::EntityId> cases;
    linalg::DenseMatrix interiorLoads;  // interior DOFs × k
    linalg::DenseMatrix condensedLoads; // contour DOFs × k
    linalg::DenseMatrix coefficients;   // k × cases
};

// What a static analysis keeps of an external unit (StaticResults::externalUnits): the unit on the model's DOFs (its
// zone and its symmetrised matrices), the contour report of its zone, what its checks measured, and per case (the
// subcases and SUBCOMs in the order of the results) its contour displacements u_a, its condensed loads f* and its
// contour forces K*·u_a − f* (the forces of the zone's elements at the contour, Σ_{e∈zone} K_e·u_e, as the contour data
// of output/sections give them for a zone of elements); and its load basis per constraint set, which the
// sensitivities read (the derivatives of f* through the unit's load source).
struct ExternalUnitRecord
{
    std::shared_ptr<const reduction::ExternalUnitAssembly> placement;
    output::ContourReport report;
    reduction::ExternalUnitCheckReport checks;
    std::uint64_t designStamp = 0;
    std::vector<base::EntityId> cases;
    linalg::DenseMatrix displacements;  // contour DOFs × cases
    linalg::DenseMatrix condensedLoads; // contour DOFs × cases
    linalg::DenseMatrix forces;         // contour DOFs × cases
    std::vector<ExternalUnitBasis> bases;
};

// What the checks of one external unit found before any analysis (checkExternalUnits): its zone, the contour report
// of the zone and the figures of ExternalUnitChecks.
struct ExternalUnitCheckOutcome
{
    std::int32_t zoneId = 0;
    output::ContourReport report;
    reduction::ExternalUnitCheckReport checks;
};

// The external units of a static analysis on the full path: what StaticSolution does with them, step by step.
//  1. Resolution (the constructor, on the model's plain element set, before the connectivity pass): each zone's contour
//     (output::ZoneContour with the constraints of the solved subcases), its replaced elements (its members: elements
//     with a stiffness or a mass), contour DOFs (the components the outside acts through, in CD) and interior DOFs;
//     then ExternalUnitChecks. Refused, naming the zone: a zone the contour refuses, a rigid element, an MPC equation
//     or a fastener inside it (their constraint rows would stay in the analysis while their stiffness is the plugin's),
//     a FreeBodySupport point inside it, two units sharing an element or an interior point, a subcase with a thermal
//     load (the zone's temperatures are not given to the plugin in this version), and every failed check.
//  2. The model without the zones: the element set on reduction::ReplacedElements (their stiffness, mass and results
//     left out), the interior DOFs held as permanent constraints (heldDofs(): nothing acts on them any more), the
//     contour nodes of each unit coupled in the node graph (couplings()), K* added to Kgg (addStiffness) and M* to Mgg
//     (addMass) on the contour DOFs: "a dense block assembled on the contour DOFs".
//  3. Mass: a unit without M* is refused for a subcase with inertial loads or inertia relief (checkMass), and named in
//     a warning for PARAM GRDPNT.
//  4. Loads, per constraint set (condenseLoads): the loads of each zone's interior DOFs for every vector of the load
//     basis go to the unit's load source, which gives f* per vector; each basis vector then has its interior loads
//     removed and f* added on the contour (applyToBasisVector). A zone loaded inside without a load source is refused.
//     The applied loads (OLOAD) of the contour points therefore hold f*.
//  5. Results (per case, recordCase): u_a, f* and K*·u_a − f* recorded; the outputs of the interior points and the
//     replaced elements left out of the plans (restrictPlan; ESE is not given, with a warning: its percentages need
//     the unit's energy); a GPFORCE row named UNIT per contour point, the unit's ID the zone's, holding −K*·u_a (the
//     applied loads there hold f*); the resultant of K*·u_a about the basic origin, −Σ r·K*·u_a, added to the SPC
//     force resultant (zero for a free unit; the force the zone's held constraints take otherwise), so that the OLOAD
//     and SPC resultants balance.
//
// Thread-safety: none (the static solution drives it on its thread). Ownership: owns the placements and the records
// (budgeted: the budget must outlive it); borrows nothing after construction.
class ExternalUnitPath
{
public:
    struct Inputs
    {
        const input::ModelSnapshot* snapshot = nullptr;
        const assembly::ElementSet* elements = nullptr; // the plain set (the zones are resolved on it)
        const assembly::DofMap* dofMap = nullptr;
        std::span<const ExternalUnitRequest> requests;
        std::span<const StaticSubcase> subcases; // every subcase of the run (the solved ones decide the contours)
    };

    // Reports every problem to `sink` (the caller stops on its errors). Throws MemoryBudgetError.
    ExternalUnitPath(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);

    [[nodiscard]] std::size_t unitCount() const noexcept { return m_units.size(); }
    // The elements of every zone, sorted.
    [[nodiscard]] const std::vector<base::EntityId>& replacedElements() const noexcept { return m_replaced; }
    // The interior DOFs of every zone (g set), sorted.
    [[nodiscard]] std::span<const base::DofIndex> heldDofs() const noexcept { return m_heldDofs; }
    // One group of nodes per unit: its contour nodes.
    [[nodiscard]] linalg::NodeGroupList couplings() const noexcept { return {m_groupOffsets, m_groupNodes}; }

    void addStiffness(linalg::BlockSparseMatrix& kgg) const;
    void addMass(linalg::BlockSparseMatrix& mgg) const;

    // Step 3: refuses (error) a unit without M* when `isInertial` (a subcase with inertial loads or inertia relief),
    // `reason` naming why; warns when only PARAM GRDPNT reads the mass.
    void checkMass(bool isInertial, bool isGridPointWeight, const std::string& reason,
                   base::DiagnosticSink& sink) const;

    // Step 4 for the constraint set `constraintSet` solving the subcases `cases` (indices into Inputs::subcases, the
    // columns of the coefficients of `loadCases`). Errors (a zone loaded inside without a load source, a source that
    // fails) go to `sink`.
    void condenseLoads(std::int32_t constraintSet, std::span<const std::size_t> cases, const StaticLoadCases& loadCases,
                       base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // The same from the loads the caller gives (the condensed path: the vectors of the interface's own load
    // basis): per unit, in their order, the loads of its zone's interior DOFs for each of the k vectors (interior × k,
    // ExternalZone::interiorDofs order), and the coefficients α (k × cases) of the solved subcases `cases` on them.
    // Throws std::invalid_argument for another number of units.
    void condenseLoads(std::int32_t constraintSet, std::span<const std::size_t> cases,
                       std::vector<linalg::DenseMatrix> interiorLoads, const linalg::DenseMatrix& coefficients,
                       base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // The condensed loads f* of the last condenseLoads, unit u (contour × k; no row for a unit whose zone was refused).
    [[nodiscard]] const linalg::DenseMatrix& currentLoads(std::size_t unit) const { return m_units[unit].currentLoads; }
    // The placement of unit u (null for a unit whose zone was refused).
    [[nodiscard]] std::shared_ptr<const reduction::ExternalUnitAssembly> placement(std::size_t unit) const
    {
        return m_units[unit].placement;
    }
    // The interior loads of basis vector b of the last condenseLoads removed, its f* added.
    void applyToBasisVector(std::size_t b, std::span<double> loadsG) const;
    // The same for the exact loads of one case (the deck tolerance's check): f* of its interior loads, from the source.
    void applyToExactCase(std::span<double> loadsG, base::MemoryBudget& budget) const;

    // The condensed loads of a SUBCOM: Σ coefficient · f* of its terms (indices into Inputs::subcases).
    void combineCase(std::size_t subcase, std::span<const std::size_t> terms, std::span<const double> coefficients);

    // Step 5 for the case `subcase` (an index into Inputs::subcases) with its g-set displacements: records u_a, f* and
    // K*·u_a − f*; returns the GPFORCE rows of the units (sorted by node) and adds −Σ r·K*·u_a to `reactions` (about
    // the basic origin).
    [[nodiscard]] std::vector<output::StiffnessRecovery::UnitRow> recordCase(std::size_t subcase,
                                                                             std::span<const double> displacements,
                                                                             std::array<double, 6>& reactions,
                                                                             base::MemoryBudget& budget);

    // The plan of a subcase without the interior points of the zones and the replaced elements (and without ESE).
    [[nodiscard]] output::OutputPlan restrictPlan(const output::OutputPlan& plan,
                                                  const assembly::ElementSet& elements) const;

    // The records, the cases in the order recordCase saw them; the path keeps nothing of them.
    [[nodiscard]] std::vector<ExternalUnitRecord> takeRecords(base::MemoryBudget& budget);
    // What step 1 found for each unit whose zone resolved (accepted or not).
    [[nodiscard]] std::vector<ExternalUnitCheckOutcome> outcomes() const;

private:
    struct Unit
    {
        std::int32_t zoneId = 0;
        std::shared_ptr<reduction::ExternalUnitAssembly> placement;
        output::ContourReport report;
        reduction::ExternalUnitCheckReport checks;
        linalg::DenseMatrix currentLoads;       // contour × k of the current constraint set
        linalg::DenseMatrix caseLoads;          // contour × subcases: f* of each subcase
        linalg::DenseMatrix caseDisplacements;  // contour × subcases: u_a of each recorded subcase
        linalg::DenseMatrix caseForces;         // contour × subcases: K*·u_a − f* of each recorded subcase
        std::vector<std::size_t> recordedCases; // the subcase indices recordCase saw, in order (bounded by them)
        std::vector<ExternalUnitBasis> bases;   // bounded by the constraint sets
    };

    void resolve(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);

    const assembly::DofMap* m_dofMap = nullptr;
    std::vector<StaticSubcase> m_subcases;
    std::vector<Unit> m_units;                   // bounded by the plugins
    std::vector<base::EntityId> m_replaced;      // bounded by the zones
    std::vector<base::DofIndex> m_heldDofs;      // bounded by the zones
    std::vector<std::int32_t> m_interiorNodes;   // sorted; bounded by the zones
    std::vector<std::int64_t> m_groupOffsets{0}; // couplings()
    std::vector<std::int32_t> m_groupNodes;      // couplings()
};

// The checks of external units without an analysis (a check of the model alone): step 1 of ExternalUnitPath on the
// model, the zones resolved with the constraints of every static subcase of the Case Control. The problems are
// diagnostics of `sink`. Throws MemoryBudgetError, and UnsupportedCardError or SolverError for a model the analysis
// would refuse before resolving its zones.
[[nodiscard]] std::vector<ExternalUnitCheckOutcome> checkExternalUnits(const input::ModelSnapshot& snapshot,
                                                                       std::span<const ExternalUnitRequest> requests,
                                                                       base::MemoryBudget& budget,
                                                                       base::DiagnosticSink& sink);

} // namespace oos::solutions
