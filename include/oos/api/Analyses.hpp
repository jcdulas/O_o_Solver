#pragma once

#include "oos/api/Threading.hpp"
#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/arrays/EigenvalueTable.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/output/envelopes/EnvelopeAccumulator.hpp"
#include "oos/output/sections/ZoneDefinition.hpp"
#include "oos/reduction/cache/CondensationCache.hpp"
#include "oos/reduction/scheduling/MemoryMode.hpp"
#include "oos/solutions/sol144/TrimSolution.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::api
{

// The analyses a model can be run with.
enum class Analysis : std::uint8_t
{
    Statics,     // linear statics (SOL 101)
    Modes,       // real eigenvalues, normal modes (SOL 103)
    Buckling,    // linear buckling (SOL 105): a static preload, then the load factors
    Aeroelastic, // static aeroelastic trim (SOL 144)
};

// Which units of condensation an analysis condenses. The units are the components of the model as O_o_Model
// computes them (connectedComponents: elements joined through shared grid points; rigid elements, multipoint
// constraints, fasteners, springs and glued contacts between them are the connections), so the model must carry the
// component tables (in JavaScript: modelFromReader(oor, deck, { condensation: true })).
enum class CondensedUnits : std::uint8_t
{
    None,        // the full path (the default)
    All,         // every unit
    Recommended, // every unit but those whose condensed stiffness would hold more values than their interior factor
    Listed       // the units of CondensationSettings::condensedUnits
};

// The condensed path of an analysis (solutions::CondensationPlanner, reduction::CondensationPlan): each condensed unit
// reduced to its boundary, the interface problem solved, the units recovered. A static analysis and a trim condense
// their units statically, with the results of the full path to round-off; normal modes and buckling reduce them by
// Craig-Bampton (their fixed-interface modes kept beside their boundary), whose roots approach those of the full
// path as more modes are kept. A model of one unit and no connection takes the full path whatever the settings.
struct CondensationSettings
{
    CondensedUnits units = CondensedUnits::None;
    std::vector<std::int64_t> condensedUnits; // with Listed: the component IDs
    std::vector<std::int64_t> fullUnits;      // kept full whatever `units` says
    std::vector<std::int32_t> retainedNodes;  // grid points kept on the boundary of their unit
    // Normal modes and buckling: the fixed-interface modes kept per unit; 0: those below the cut-off of
    // reduction::CB_CUTOFF_FACTOR times the highest root asked for.
    std::int32_t modeCount = 0;
    // Statics: what the analysis keeps in memory (reduction::MemoryMode: every mode gives the same results to the bit;
    // LowMemory recovers the units in sequence and builds their equations again at their recovery).
    reduction::MemoryMode memoryMode = reduction::MemoryMode::Speed;
    // The condensation cache the units are taken from and stored in (null: none). It outlives the analysis; an analysis
    // that finds a unit in it condenses nothing of that unit, with the same results to the bit. `cacheModelTag` names
    // the model in the entries the analysis stores.
    std::shared_ptr<reduction::CondensationCache> cache;
    std::string cacheModelTag;
};

// An external unit of a static analysis given by its matrices (solutions::ExternalUnitPath): the zone whose elements
// it replaces (its elements and its boundary grid points, as O_o_Model's zone boundaries give them) and the matrices
// of the zone condensed onto its contour, on `labels` (grid point and component 1 to 6, in the displacement systems of
// the grid points), column by column. No load source: a zone loaded inside is refused.
struct ExternalUnitSettings
{
    output::ZoneDefinition zone;
    std::vector<reduction::UnitDofLabel> labels;
    std::vector<double> stiffness; // K*, labels × labels
    std::vector<double> mass;      // M*, labels × labels; empty: not given
};

// The eigenvalue solver of normal modes and buckling on the full path (solutions::EigenSolverOptions).
enum class EigenSolver : std::uint8_t
{
    // The block shift-invert Lanczos solver on a factor held in memory (the default): the roots of the request, each
    // confirmed by the Sturm counts of the shifts.
    Lanczos,
    // Multilevel substructuring on the elimination tree of the model: the tree's subtrees reduced to their lowest
    // modes, then the reduced problem solved. Its roots are approximate, as accurate as its cut-off allows (the lowest
    // ones best); it factorises no matrix of the whole model, so it takes far less memory and time on a large one.
    Substructuring,
    // The Lanczos solver with its factor on a scratch disk: the roots of Lanczos, the memory of the factor traded for
    // disk space and the time of its reads.
    OutOfCore
};

// The eigenvalue solver and what it needs. Substructuring and OutOfCore take their matrices from the elements, front
// by front: no matrix of the whole model is assembled, so the outputs that need one (the constraint forces, the modal
// effective mass, the grid point weight) are not produced; the run says so in an information diagnostic.
struct EigenSolverSettings
{
    EigenSolver solver = EigenSolver::Lanczos;
    // The folder of the scratch files. Empty: OutOfCore takes the folder named by the OOS_SCRATCH environment
    // variable, else a folder of the system's temporary one; Substructuring keeps its scratch data in memory.
    std::string scratchFolder;
    // Substructuring: the cut-off of the modes kept in each subtree, as a factor of the highest root asked for
    // (a higher factor keeps more modes: more accurate, slower). 0: the solver's default.
    double cutoffFactor = 0.0;
};

// What an analysis is asked to do. Every field has a default that runs the deck as it is written.
struct AnalysisSettings
{
    // The subcases to solve, in this order; empty: every subcase of the Case Control.
    std::vector<std::int32_t> subcases;
    // The threads of the run; AUTOMATIC_THREADS: every core of the machine (automaticThreadCount).
    std::size_t threads = AUTOMATIC_THREADS;
    // The memory the analysis may use, in bytes; 0: what is left in the module's memory.
    std::uint64_t memoryBytes = 0;
    // Additional links; links exported in the snapshot are read automatically.
    std::vector<aero::AeroLink> links;
    // The units to condense (none: the full path).
    CondensationSettings condensation;
    // Statics only: the full path streamed (solutions::StreamedStatics: the equations solved without a factor of the
    // whole model, the outputs made region by region), for a model whose factor would not fit the memory. Refused with
    // a condensation.
    bool isStreamed = false;
    // Normal modes and buckling only, on the full path: the eigenvalue solver (refused with a condensation).
    EigenSolverSettings eigenSolver;
    // Statics only: the external units whose matrices replace zones of elements (empty: none).
    std::vector<ExternalUnitSettings> externalUnits;
    // Statics only: the envelopes computed over the load cases (output::EnvelopeAccumulator; empty: none), and whether
    // the per-case tables of a kind that has an envelope are kept too (false: only the envelope; the condensed path
    // recovered in sequence may then recover its units unit by unit).
    std::vector<output::EnvelopeSpec> envelopes;
    bool keepsCaseTables = true;
};

// What the condensed path of an analysis did.
struct CondensationSummary
{
    std::vector<std::int64_t> condensedUnits; // by increasing ID
    std::vector<std::int64_t> fullUnits;      // the units kept full, by increasing ID
    std::int64_t interfaceDofs = 0;           // the DOFs of the interface problem, as the plan estimates them
    std::string note;                         // why the plan differs from the request (the one-unit rule), or empty
    // Statics: whether a constraint set recovered its units unit by unit (CondensedStatics::GroupInputs::visitUnit).
    bool isUnitByUnit = false;
};

// One stage of a run and its wall time, in seconds.
struct RunStage
{
    std::string name;
    double seconds = 0.0;
};

// What a run cost, for the caller to see without instrumenting it: the threads and dense kernels it ran on, the size of
// the problem, its wall time stage by stage and the largest memory its budget held. The stages are those the analysis
// measures (statics: setup, assembly, loadBasis, factorisation, solve, recovery; normal modes: setup, assembly,
// eigenvalues, recovery; buckling: preload, assembly, eigenvalues, recovery; trim: setup, assembly, factorisation,
// flexibility, aerodynamics, trim), in the order it runs them; their sum is below `seconds`, which also counts what
// lies between them (the condensation plan, the hand-over of the tables).
// The transverse shear of one laminate property (CompProp, CompPropGlobal) that a shell of the model uses, along the
// material axes x and y of the element, as the model's convention forms it (Parameter LaminateTransverseShear:
// Equilibrium, the default, or FiveSixths):
// - stiffness: H_x and H_y, a force per unit length per unit shear strain; 0 along an axis the laminate is rigid in;
// - factor: κ = H / Σ G_k t_k, the shear correction factor the stiffness amounts to (G_k the transverse shear modulus
//   of ply k along the axis, t_k its thickness): 5/6 for one material and under FiveSixths, another value for a
//   stack of plies of different moduli; 0 where it is not defined (the laminate, or one of its plies, rigid along the
//   axis).
// A program that takes 5/6 Σ G_k t_k for every stack differs from the default by the ratio of the factor to 5/6.
struct LaminateShear
{
    std::string card; // "CompProp" or "CompPropGlobal"
    std::int32_t propId = 0;
    std::array<double, 2> stiffness{};
    std::array<double, 2> factor{};
};

struct RunReport
{
    std::size_t threads = 1;   // the threads of the run, the calling thread included
    std::string kernel;        // the dense kernel family the factorisations ran on ("avx2", "reference", "wasm128")
    std::int64_t dofs = 0;     // the degrees of freedom of the model (six per grid point, one per scalar point)
    std::int64_t solvedDofs = 0; // the equations of the largest system factorised (the free set, or the interface's)
    std::size_t factorisations = 0; // the sparse factorisations made (0 when the analysis does not count them)
    // Normal modes and buckling: the eigenvalue solver that ran ("lanczos", "substructuring", "outOfCore"; "condensed"
    // on the condensed path); empty for the other analyses.
    std::string eigenSolver;
    double seconds = 0.0;           // the wall time of the run
    std::vector<RunStage> stages;
    // The largest memory the run's budget held at once, in bytes: the matrices, factors and result tables the solver
    // reserves in it (base::MemoryBudget::peak), not the memory of the process.
    std::uint64_t peakBudgetBytes = 0;
    // The laminates of the shells, by card then by ID (empty for a model without one).
    std::vector<LaminateShear> laminateShear;
};

// The outcome of an analysis: the result tables, the eigenvalues and the warnings.
struct AnalysisOutput
{
    // The memory budget of the run: the result buffers hold their reservations in it, so it is declared first and
    // destroyed last.
    std::shared_ptr<base::MemoryBudget> budget;
    // The applied quartic shell IDs chosen by Auto in full statics.
    std::optional<std::vector<std::int32_t>> selectedQuarticShells;
    // The nodal and element tables, in the layout of O_o_Model's results ({kind, subcase, step, ids, values}).
    output::ResultArrays tables;
    // The eigenvalue table of each modal or buckling subcase (empty for statics).
    std::vector<output::EigenvalueTable> eigenvalues;
    // Aerodynamic and trim results for SOL 144; its structural tables are in tables.
    std::shared_ptr<solutions::TrimResults> aeroelastic;
    // The warnings and information of the run (an error stops the run with an exception instead).
    std::vector<base::Diagnostic> diagnostics;
    // An analysis asked to condense: what its plan decided (empty otherwise).
    std::optional<CondensationSummary> condensation;
    // Statics: the envelopes asked for (AnalysisSettings::envelopes).
    std::vector<output::EnvelopeTable> envelopes;
    // What the run cost (threads, kernels, sizes, stage times, memory).
    RunReport report;
};

// The cards and fields of the model the analysis cannot take (refusals: errors) or will not use (ignored: warnings),
// all found in one pass before anything is computed.
// Supported cards and diagnostics use the neutral model card types ("ShellQuad4") and field names ("offset").
struct SupportCheck
{
    std::vector<std::string> supported;
    std::vector<base::Diagnostic> refusals;
    std::vector<base::Diagnostic> ignored;
};

// What the formulation that reads its geometry from the mesh (MixedHybridFormulation) made of every edge of the
// surface of the shells and of the skin of the solids (docs/mixed-hybrid-formulation-spec.md section 10.5), one row
// per edge and column by column, for a program that shows the model: the IDs of its two grid points (the lower
// first); the surface ("Shells", "SolidSkin"); its state ("Smooth", "Sharp", "Raw"); where the state comes from
// ("Declared", "RawElement", "Angle", "Junction", "Outline"); and, in degrees, the angle measured between the normals
// of its elements at its grid points, the angle left between the normals of the surface read on both sides, and how
// far the surface read turns from the plane of the corners of an element (the last two on the shells only).
struct SurfaceEdgeTable
{
    std::vector<std::int32_t> grid1;
    std::vector<std::int32_t> grid2;
    std::vector<std::string> surface;
    std::vector<std::string> state;
    std::vector<std::string> origin;
    std::vector<double> angle;
    std::vector<double> kink;
    std::vector<double> rounding;
};

// The edges read of a model: empty when no card of the model is in a formulation that reads its geometry from the
// mesh. The model of the formulation is built as an analysis builds it; the problems that would stop the analysis
// (checkSupport lists them: a model without geometry declarations...) do not stop the table, which then shows what
// the fold angle of the model alone decides.
[[nodiscard]] SurfaceEdgeTable surfaceEdges(const input::ModelSnapshot& snapshot);

// The formulations of one element card, the default first (elements/core/ElementFormulation.hpp).
struct ElementFormulations
{
    std::string card;
    std::vector<std::string> formulations;
};

// Every element card the solver computes, with its formulations, in the order of the element library.
[[nodiscard]] std::vector<ElementFormulations> elementFormulations();

// What the solver refuses of the model for the analysis, without running it: the cards, fields and selections it does
// not take and, when there is none, what the element formulation of the model refuses (the problems of its model, as
// a geometry declaration that names no edge, and the elements it cannot compute), which a run reports before it
// assembles anything.
[[nodiscard]] SupportCheck checkSupport(const input::ModelSnapshot& snapshot, Analysis analysis);
[[nodiscard]] SupportCheck checkTrimPreloadedBuckling(const input::ModelSnapshot& snapshot);

// Runs the analysis on the model. Throws UnsupportedCardError when a card is refused (checkSupport lists them all),
// SolverError for a problem of the model (a mechanism, a singular matrix, a missing card), MemoryBudgetError when the
// memory does not suffice.
[[nodiscard]] AnalysisOutput runAnalysis(const input::ModelSnapshot& snapshot, Analysis analysis,
                                         const AnalysisSettings& settings);

// One superelement upstream of the residual structure (solutions::SuperelementSolution): its own model, read with
// the Case Control of the residual structure (its subcases select the sets of its own model), and its boundary
// points, each a point of the part and the point of the residual structure it is.
struct SuperelementPartSettings
{
    std::int32_t superelementId = 1;
    std::shared_ptr<const input::ModelSnapshot> snapshot;
    std::vector<std::pair<base::EntityId, base::EntityId>> boundary; // (part point, residual point)
};

// The results of a run with superelements: the tables of each superelement (the residual structure, 0, first), the
// eigenvalues of a modal analysis, and the warnings.
struct SuperelementOutput
{
    std::int32_t superelementId = 0;
    output::ResultArrays tables;
    std::vector<output::EigenvalueTable> eigenvalues;
};
struct SuperelementAnalysisOutput
{
    std::shared_ptr<base::MemoryBudget> budget; // declared first: the result buffers hold their reservations in it
    std::vector<SuperelementOutput> superelements;
    std::vector<base::Diagnostic> diagnostics;
};

// Runs a static (Statics) or modal (Modes) analysis of the residual structure `residual` with the superelements
// `parts`, each reduced on its boundary (static condensation, or Craig-Bampton on its generalised coordinates in a
// modal analysis), the residual structure solved with them, each part recovered. Every subcase of the residual's Case
// Control applies to every superelement. Throws as runAnalysis, SolverError for another analysis.
[[nodiscard]] SuperelementAnalysisOutput runSuperelements(const input::ModelSnapshot& residual,
                                                          std::vector<SuperelementPartSettings> parts,
                                                          Analysis analysis, const AnalysisSettings& settings);

// The name and version of the solver: "O_o_Solver 0.2.6".
[[nodiscard]] std::string programName();

// A condensation cache that lives in memory, for the analyses of the process to share (CondensationSettings::cache):
// its entries at most `memoryBytes` (0: reduction::CacheOptions::DEFAULT_MEMORY_BYTES), the least recently used dropped
// first.
[[nodiscard]] std::shared_ptr<reduction::CondensationCache> makeCondensationCache(std::uint64_t memoryBytes);

} // namespace oos::api
