#pragma once

#include "oos/aero/trim/TrimCase.hpp"
#include "oos/aero/trim/TrimVariables.hpp"
#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/output/monitors/MonitorPoint1.hpp"
#include "oos/output/monitors/MonitorPoint3.hpp"
#include "oos/reduction/cache/CondensationCache.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"
#include "oos/solutions/common/AnalysisPath.hpp"
#include "oos/solutions/common/ProgressMonitor.hpp"
#include "oos/solutions/sol101/CondensedStatics.hpp"
#include "oos/solutions/sol144/TrimLinearisation.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace oos::solutions
{

// What a static aeroelastic analysis runs and with what.
struct TrimOptions
{
    // The subcases to solve, in this order; empty: every SUBCASE of the Case Control (the global section of a deck
    // without SUBCASE, as subcase 1).
    std::vector<base::EntityId> subcases;
    // The AeroTrimLink cards of the model (O_o_Reader gives them without their fields: the caller reads them).
    std::vector<aero::AeroLink> links;
    base::ThreadPool* pool = nullptr; // null: one thread; not owned
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    const base::CancellationToken* cancellation = nullptr; // not owned, may be null
    ProgressMonitor* monitor = nullptr;                    // not owned, may be null: the stages of the run
    // The memory the structural result tables may take; nothing: half of the memory budget left at the start.
    std::optional<std::uint64_t> outputLimitBytes;
    // The MonitorIntegratedLoad1 of the model, their components resolved (O_o_Reader gives the AeroComponent and
    // AeroComponentUnion as card objects only: the caller resolves them). The trim adds its own points: the whole
    // vehicle, structural and aerodynamic (named after the configuration), and the hinge moment of each
    // AeroControlSurface.
    std::vector<output::MonitorPoint1Definition> monitorPoints;
    // The MonitorNodeForceSum3 of the model, their GRIDSET and ELEMSET expanded (output::MonitorPoint3).
    std::vector<output::MonitorPoint3Definition> monitorPoints3;
    // The aerodynamic configuration (Case Control AECONFIG; AEROSG2D by default).
    std::string configuration = "AEROSG2D";
    // The condensed path (static condensation with the spline nodes retained): a plan of the snapshot
    // (CondensationPlanner) whose condensed units keep the splines' grid points and the FreeBodySupport points on their
    // boundaries (trimRetainedPoints; null: the full path); the units to recover (nothing: those whose outputs are
    // asked for, and those of the elements of a MonitorNodeForceSum3); the condensation cache and the name of the model
    // in its entries.
    std::shared_ptr<const reduction::CondensationPlan> condensation;
    std::optional<std::vector<std::int64_t>> recoveredUnits;
    reduction::CondensationCache* cache = nullptr;
    std::string cacheModelTag;
    // Outputs added to the requests of every subcase, whatever its Case Control asks (the outputs the design responses
    // of a SOL 200 SAERO analysis read, as StaticOptions::requiredOutputs).
    std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> requiredOutputs{};
    // Whether the results keep the trim of every subcase as a function of the structure (TrimResults::linearisation):
    // the design derivatives of SOL 200 SAERO solve on it. The kept parts stay in the memory budget until the results
    // are dropped.
    bool keepsLinearisation = false;
};

// The grid points a condensed trim keeps on the boundaries of their units: the structural points of the splines (the
// aerodynamic loads act on them and their motions give the boxes' slopes), sorted. Throws as TrimSolution's reading of
// the aerodynamic model (SolverError, UnsupportedCardError).
[[nodiscard]] std::vector<base::EntityId> trimRetainedPoints(const input::ModelSnapshot& snapshot,
                                                             base::MemoryBudget& budget);

// The six coefficients CX, CY, CZ, CMX, CMY, CMZ (aero::ReferenceCoefficients) of one trim variable, the
// non-dimensional stability and control derivatives, in each of their columns: the aerodynamic loads of a unit
// value of the variable on the rigid boxes (unsplined) and through the splines on the structure (splined), the
// aerodynamic loads of the elastic structure supported at the FreeBodySupport DOFs (restrained) or free and balanced by
// its inertia (unrestrained), and the inertial loads of the supported motion in each. An acceleration (RigidAccelX ...)
// has aerodynamic loads only through the deformation its inertial loads give the restrained structure; a free structure
// is not deformed by a uniform acceleration (its unrestrained columns are zero).
struct StabilityDerivativeRow
{
    std::string label;
    std::array<double, 6> rigidUnsplined{};
    std::array<double, 6> rigidSplined{};
    std::array<double, 6> elasticRestrained{};
    std::array<double, 6> elasticUnrestrained{};
    std::array<double, 6> inertialRestrained{};
    std::array<double, 6> inertialUnrestrained{};
};

// The non-dimensional hinge moment coefficients of one control surface for a unit value of each variable of the trim
// case: rigid, elastic restrained and unrestrained.
struct HingeMomentDerivatives
{
    std::string surface;
    std::vector<std::array<double, 3>> perVariable; // in the order of the trim case's variables
    std::array<double, 3>
        atReference{}; // at the reference: the intercept's (DownwashMatrix, AeroPressureMatrix), zero without
};

// A trim variable of the trimmed state.
struct TrimVariableValue
{
    std::int32_t id = 0; // its AeroRigidTrimVariable or AeroControlSurface, 0 when the snapshot does not give it
    std::string label;
    aero::TrimVariableKind kind = aero::TrimVariableKind::RigidBody;
    aero::TrimVariableState state = aero::TrimVariableState::Free;
    double value = 0.0;
};

// The trim of one subcase.
struct TrimSubcaseResult
{
    base::EntityId subcase = 1;
    std::int32_t trimSet = 0;
    double mach = 0.0;
    double dynamicPressure = 0.0;
    std::vector<TrimVariableValue> variables;        // in the order of the trim case
    std::vector<StabilityDerivativeRow> derivatives; // one per variable, in the same order
    // The REF. COEFF. row: the coefficients of the intercept, the fixed variable of value 1 that DownwashMatrix and
    // AeroPressureMatrix make; zero without them.
    StabilityDerivativeRow reference;
    bool hasUnrestrained = false; // with FreeBodySupport DOFs (without, the columns stay zero)
    std::vector<HingeMomentDerivatives> hingeMomentDerivatives; // one per AeroControlSurface
    std::vector<double> pressures;                              // the trimmed Δcp of each box
    std::vector<double> boxForces;            // q·Δcp·A of each box along its normal, at its center (AEROF T3)
    std::vector<double> boxMoments;           // its moment about the spanwise axis of the box at its center (AEROF R2)
    std::vector<double> hingeMoments;         // the trimmed hinge moment of each AeroControlSurface
    std::vector<double> supportAccelerations; // ü_r of each FreeBodySupport DOF, in model units
    bool printsPressures = false;             // the Case Control asks for APRESSURE
    bool printsForces = false;                // and for AEROF
    // The loads of each monitor point (TrimResults::monitorPoints, in that order): per contribution
    // (output::MonitorContribution) the six loads along and about the axes of its CD.
    std::vector<std::array<std::array<double, 6>, output::MONITOR_CONTRIBUTION_COUNT>> monitorLoads;
    // The resultant of each MonitorNodeForceSum3 (TrimResults::monitorPoints3, in that order) in the trimmed state: the
    // six loads along and about the axes of its CD.
    std::vector<std::array<double, 6>> monitor3Loads;
    // The trimmed displacements, g set (one value per DOF of assembly::DofMap): relative to the supports (u_r = 0),
    // the constrained DOFs zero, the m set from Gm.
    base::BudgetedBuffer<double> displacements;
};

// The seconds spent in each stage (summed over the constraint sets and subcases).
struct TrimTimings
{
    double setup = 0.0;         // checks, DOF map, connectivity, trim cases, boxes and splines
    double assembly = 0.0;      // stiffness, mass, m-set elimination, singularities, ordering
    double factorisation = 0.0; // the l-set factor
    double flexibility = 0.0;   // the structure under unit box pressures: one solve per box
    double aerodynamics = 0.0;  // the doublet lattice and the pressures of the variables
    double trim = 0.0;          // the trim system, the derivatives, the trimmed displacements
};

struct TrimResults
{
    // The structural tables of the trimmed states, as each subcase's output plan asks (displacements, applied loads,
    // SPC and MPC forces, element results, grid point forces, strain energy).
    output::ResultArrays arrays;
    std::vector<TrimSubcaseResult> subcases; // in the order solved
    std::vector<std::int32_t> boxIds;        // the boxes of `pressures`, in that order
    std::vector<std::string> surfaces;       // the AeroControlSurface labels of the hinge moments, in that order
    // The AeroControlSurface of `surfaces`: their hinge reference values and limits (the results report them).
    std::vector<aero::TrimVariables::ControlSurface> surfaceDefinitions;
    // The AeroStaticParams: REFC, REFB, REFS, the symmetry keys, and {X}_REF = basicToReference·{X}_BAS +
    // referenceOffset.
    double referenceChord = 0.0;
    double referenceSpan = 0.0;
    double referenceArea = 0.0;
    std::int32_t symmetryXz = 0;
    std::int32_t symmetryXy = 0;
    base::Mat3 basicToReference = base::Mat3::identity();
    base::Vec3 referenceOffset;
    std::string configuration;
    // The monitor points: the caller's MonitorIntegratedLoad1, then the whole vehicle (structural, aerodynamic) and the
    // hinge moments.
    std::vector<output::MonitorPoint1Definition> monitorPoints;
    std::vector<output::MonitorPoint3Definition> monitorPoints3;
    std::vector<base::Diagnostic> diagnostics;
    std::vector<std::string> singularityTables;
    std::int64_t gSetSize = 0;
    std::int64_t largestLsetSize = 0; // the full path's l set, or the condensed path's interface equations
    TrimTimings timings;
    // The path taken, and on the condensed path its report (the units condensed and recovered, the interface).
    AnalysisPath path = AnalysisPath::Full;
    std::optional<CondensedRunRecord> condensed;
    // With TrimOptions::keepsLinearisation: the trim of every subcase as a function of the structure.
    std::shared_ptr<TrimLinearisation> linearisation;
};

// The static aeroelastic trim on the full model (SOL 144, subsonic). The structure K_ll (the l set: the f set without
// the r set of FreeBodySupport, SUPORT1) and the doublet lattice at k = 0 (aero::DoubletLatticeSteady: D·Δcp = w) are
// coupled through the splines (aero::SplineMatrix): the pressures load the structure by P = q·T·Δcp, and its
// deformation adds the normalwash of its slopes, w_e = S·u (w = +∂h/∂x: aero::SplineMatrix). With the structure held at
// the supports (u_r = 0),
//     u_l = K_ll⁻¹·(q·T·Δcp + f),    (D − AEQR·q·E)·Δcp = w_x·u_x + AEQR·S·K_ll⁻¹·f,    E = S·K_ll⁻¹·T,
// f = −M·D_r·ü_r the inertial loads of the rigid-body acceleration of the supports (ü_r of the rigid acceleration
// variables in the reference frame, divided by PARAM AUNITS; D_r the rigid-body motions of a unit motion of each
// FreeBodySupport DOF, from the geometry about the RCSID origin on the supported directions, zero on the s set and
// Gm·u_n on the m set: exact rigid-body motions, not −K_ll⁻¹·K_lr, which an improperly specified FreeBodySupport makes
// elastic; with them the inertial derivatives do not depend on the quality of the supports, and an unconnected grid
// point with a mass, held by AUTOSPC, carries no inertia), w_x the normalwash of each aerodynamic variable
// (aero::TrimVariables). The problem is solved in the space of the boxes: one structural solve per box gives E (n²
// terms for n boxes) and the products with T, then each variable's pressures come from one dense LU of D − AEQR·q·E
// (the formulation in the a set, K_aa − q·Q_aa, is the same system after the boxes are eliminated).
//
// The trim (aero::TrimSystem): the equilibrium of the supported motions, D_rᵀ·(q·T·Δcp + f) = 0, one equation per
// FreeBodySupport DOF, linear in the trim variables (a linear trim without redundant control surfaces). The stability
// derivatives (StabilityDerivativeRow) restrained are those of the same system; the unrestrained ones take the loads
// relieved by the inertia of the free body, R·P = P − M·D_r·(D_rᵀ·M·D_r)⁻¹·D_rᵀ·P (E_R = S·K_ll⁻¹·R·T); their
// coefficients are the resultants of the structural loads about the RCSID origin by the rigid-body modes of the
// structure (assembly::RigidBodyModes), the images of a half model added (aero::ReferenceCoefficients).
//
// On the condensed path (TrimOptions::condensation) the same algebra runs on the interface problem of the condensed
// units, the spline nodes on their boundaries: the box loads are interface loads, K_ll⁻¹·M·D_r·ü_r comes from the
// units' condensed inertial patterns −M·Φ_r with the rigid-body accelerations C·ü_r (D_r = Φ_r·C away from the held
// and dependent DOFs), and the trimmed state is recovered in the units its outputs need; it equals the full path to
// round-off where no mass couples a held DOF with a free one.
//
// Applied loads: the subcase's LOAD P enters the structural equilibrium beside the aerodynamic and inertial loads,
// u_l = K_ll⁻¹·(q·T·Δcp + f + P), the pressures gain (D − AEQR·q·E)⁻¹·AEQR·S·K_ll⁻¹·P and the trim equations the
// constant D_rᵀ·(q·T·Δcp_P + P); P is in the applied loads of the outputs and the applied columns of the monitor
// points. A model without FreeBodySupport takes no inertia from its rigid acceleration variables (the inertial loads
// are those of the supported motions): its weight is a GravityLoad in its LOAD.
//
// The direct matrix input (aero::TrimCorrections): DownwashMatrix's normalwash w_0 and AeroPressureMatrix's pressures
// Δcp_e, a single vector for every subcase, are a fixed variable INTERCEPT of value 1: one more column of the
// pressures, rigid D⁻¹·w_0 + Δcp_e, restrained (D − AEQR·q·E)⁻¹·(w_0 + AEQR·q·E·Δcp_e) + Δcp_e (unrestrained with E_R),
// its resultant q·D_rᵀ·T·Δcp_0 in the constant part of the trim equations, its coefficients the REF. COEFF. row
// (TrimSubcaseResult::reference); on both paths. Without them nothing of it is computed.
//
// Not supported by this version (each refused naming it): the design derivatives of a model with DownwashMatrix or
// AeroPressureMatrix, thermal loads in a trim subcase (TEMPERATURE(LOAD)), applied loads on the condensed path,
// the trim variables of AeroController or of other AeroRigidTrimVariable
// labels when free (they act only through user matrices), AeroControlSchedule, the redundant-surface trim, panels other
// than AeroLiftingPanel1, the splines aero::SplineMatrix refuses.
//
// Determinism: the assembly, factor, solves and the dense algebra (serial, fixed order) give the same bits for any
// number of threads, for a given kernel family.
//
// Thread-safety: run() is const; it uses the pool of the options. Ownership: borrows the snapshot, which must outlive
// the call.
class TrimSolution
{
public:
    TrimSolution(const input::ModelSnapshot& snapshot, TrimOptions options)
        : m_snapshot(&snapshot), m_options(std::move(options))
    {
    }

    // Throws UnsupportedCardError (cards not supported), SolverError (problems of the model and of its trim cases,
    // listed; a singular trim), SingularityError, MemoryBudgetError, OperationCancelledError, std::invalid_argument (a
    // subcase asked for that the Case Control does not have).
    [[nodiscard]] TrimResults run(base::MemoryBudget& budget) const;

private:
    const input::ModelSnapshot* m_snapshot;
    TrimOptions m_options;
};

} // namespace oos::solutions
