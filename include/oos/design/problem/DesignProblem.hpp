#pragma once

#include "oos/base/memory/KeptObject.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/aids/LinearityIndicator.hpp"
#include "oos/design/constraints/DesignConstraints.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/model/DesignPoint.hpp"
#include "oos/design/model/FieldTargets.hpp"
#include "oos/design/problem/DesignAnalyses.hpp"
#include "oos/design/responses/DesignResponses.hpp"
#include "oos/design/sensitivity/DesignGradients.hpp"
#include "oos/design/sensitivity/GradientStream.hpp"
#include "oos/design/sensitivity/StaticSensitivity.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"
#include "oos/solutions/sol103/ModesSolution.hpp"
#include "oos/solutions/sol105/BucklingSolution.hpp"
#include "oos/solutions/sol144/TrimSolution.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace oos::design
{

// A design point and the analyses made on it: the static analysis, and the eigenvalue analyses of its MODES
// and BUCK subcases.
struct StaticDesignEvaluation
{
    DesignPoint point;
    std::shared_ptr<const input::ModelSnapshot> snapshot; // the overlaid snapshot the analysis read
    // The static results: of the STATICS subcases, and with BUCK subcases of every static subcase of the deck (the
    // buckling solution's preloads, BucklingResults::preload moved here).
    solutions::StaticResults results;
    std::vector<std::int32_t> subcases; // the STATICS subcases the responses read, in order
    // The eigenvalue analyses (DesignAnalyses; none without MODES or BUCK subcases): the modes of the MODES subcases,
    // the buckling solves of the BUCK subcases (their preload in `results`), the subcases of each kind in their order,
    // and the static subcases that are preloads (the sensitivities of LAMA read their displacements and factors).
    std::optional<solutions::ModesResults> modes{};
    std::optional<solutions::BucklingResults> buckling{};
    std::vector<std::int32_t> modesSubcases{};
    std::vector<std::int32_t> bucklingSubcases{};
    std::vector<std::int32_t> preloadSubcases{};
    // The static subcases the sensitivities read: the STATICS ones, then the preloads that are not (empty: `subcases`).
    std::vector<std::int32_t> staticSubcasesRead{};
    // The trim of the SAERO subcases (none without them), kept as a function of the structure when the
    // gradients are wanted (TrimResults::linearisation), and those subcases in their order. The structural tables of
    // the trimmed states are in `results.arrays` with the static ones (the static responses of a SAERO subcase read
    // them).
    std::optional<solutions::TrimResults> trim{};
    std::vector<std::int32_t> aeroelasticSubcases{};
};

// The responses, constraints and objective of a design point, with the analysis they were read from.
struct DesignEvaluation
{
    StaticDesignEvaluation analysis;
    std::vector<std::int32_t> subcases;    // the analysed subcases, in order
    std::vector<ResponseValues> responses; // in the order of DesignResponses::ids(), the evaluated ones
    std::vector<ConstraintValue> constraints;
    std::optional<double> objective;
    // Per external response, the subcases it evaluated per zone (its selection).
    std::map<std::int32_t, ItemSelection> selections;
    // The modes the eigenvalue responses that track their mode took, and the warnings of an uncertain tracking.
    std::vector<TrackedMode> trackedModes{};
    std::vector<base::Diagnostic> trackingWarnings{};
};

// The core of a design problem driven by the caller: a design model bound to one snapshot of the model, which turns a
// vector x of the independent design variables into a design point (field values, derivatives, overlay, design stamps)
// and runs the analysis on the overlaid snapshot, never editing the model. The responses, the sensitivities and the
// optimisers build on it, and the SOL 200 driver uses it. The caller gathers the deck's design model and the
// declarations into DesignModelParts, builds this object, and writes a point into the model (card edits through
// O_o_Reader).
//
// Thread-safety: immutable but for what it keeps between design points (the geometry of the linearity check, under a
// mutex); designPoint() and evaluateStatics() may run concurrently. Ownership: shares the snapshot.
class DesignProblem
{
public:
    // Throws as DesignModel (problems of the design model) and FieldTargets (fields the snapshot lacks or that are
    // refused); std::invalid_argument for a null snapshot.
    DesignProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, DesignModelParts parts);
    // With responses and constraints. Throws, in addition, as DesignResponses and DesignConstraints.
    DesignProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, DesignModelParts parts, ResponseParts responses,
                  ConstraintParts constraints);
    // With the section library of the designed BarShapeProp and BeamShapeProp (O_o_Reader's formulas, which the caller
    // gives; FieldTargets).
    DesignProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, DesignModelParts parts, ResponseParts responses,
                  ConstraintParts constraints, std::shared_ptr<const SectionLibrary> sections);

    [[nodiscard]] const input::ModelSnapshot& snapshot() const noexcept { return *m_snapshot; }
    [[nodiscard]] const std::shared_ptr<const input::ModelSnapshot>& sharedSnapshot() const noexcept
    {
        return m_snapshot;
    }
    [[nodiscard]] const DesignModel& model() const noexcept { return m_model; }
    [[nodiscard]] const FieldTargets& targets() const noexcept { return m_targets; }
    [[nodiscard]] const DesignResponses& responses() const noexcept { return m_responses; }
    [[nodiscard]] const DesignConstraints& constraints() const noexcept { return m_constraints; }
    [[nodiscard]] std::vector<double> initialValues() const { return m_model.initialValues(); }
    // The units (component IDs of the snapshot's unit tables) the design variables can drive: those of every relation's
    // field and of every shape's moved grid points (FieldTargets), increasing. the automatic condensation keeps
    // them full (every unit condensed except those driven by a design variable).
    [[nodiscard]] std::vector<std::int64_t> drivenUnits() const;
    // Whether its analyses recover the outputs the deck's Case Control asks for beside those its responses read (true,
    // the default), or only the latter (false: StaticOptions::usesCaseControlOutputs).
    void setDeckOutputs(bool deckOutputs) noexcept { m_deckOutputs = deckOutputs; }
    [[nodiscard]] bool deckOutputs() const noexcept { return m_deckOutputs; }

    // The design point at x. Throws as DesignModel::evaluate.
    // `reference`: the field values of a point whose meshes the finite element sections take (DesignPoint).
    [[nodiscard]] DesignPoint designPoint(std::span<const double> x, std::span<const SuppliedField> supplied = {},
                                          std::span<const double> reference = {}) const
    {
        return {m_model, m_targets, x, supplied, reference};
    }
    // The snapshot overlaid with a design point of this problem; it keeps the base snapshot alive.
    [[nodiscard]] std::shared_ptr<const input::ModelSnapshot> overlaySnapshot(const DesignPoint& point) const;

    // The mesh distortions of a design point (ShapeDistortion: the element checks of its snapshot that the base model
    // does not fail), as SHAPE_DISTORTION diagnostics; none for a design model without shape variables. The checks of
    // the base snapshot are made once per problem, at the first call. Throws as ShapeDistortion::elementProblems.
    [[nodiscard]] std::vector<base::Diagnostic> distortions(const DesignPoint& point, base::MemoryBudget& budget) const;

    // SOL 101 on the design point x (solutions/sol101 on the overlaid snapshot), with the outputs the responses read
    // added to the requests of `options`; with `needsDisplacements`, the displacements of every point too and the
    // factorised constraint sets (StaticOptions::keepsSystems), what gradients() reads: the kept factors stay in
    // `budget` until the evaluation is dropped, and `budget` must outlive it. With shape variables, the distortions of
    // the point are checked first: an error is a base::SolverError naming the element (the point is refused, as a
    // degenerate element of a deck; with failures captured its status is "failed"), a warning is added to the
    // diagnostics of the results. Throws as designPoint() and StaticSolution::run.
    [[nodiscard]] StaticDesignEvaluation evaluateStatics(std::span<const double> x,
                                                         const solutions::StaticOptions& options,
                                                         base::MemoryBudget& budget,
                                                         std::span<const SuppliedField> supplied = {},
                                                         bool needsDisplacements = false) const;

    // The analyses of the design point x by subcase kind (DesignAnalyses): the static solution of the
    // STATICS subcases as evaluateStatics() makes it (`options` gives its subcases when analyses.statics is an empty
    // list), solutions::ModesSolution on the MODES subcases and solutions::BucklingSolution on the BUCK subcases (the
    // static subcases solved by it as their preloads, with the outputs of the responses and, with
    // `needsDisplacements`, their factorised constraint sets kept), all on the overlaid snapshot. On the condensed path
    // (options.condensation) the eigenvalue analyses take the same plan, the units the point drives recovered (their
    // eigenvalue derivatives read the mode shapes of their elements), and every unit for the buckling shapes when the
    // problem has a LAMA response (the derivative of a load factor reads the shape and the preload of every element).
    // Throws as evaluateStatics(), ModesSolution::run and BucklingSolution::run.
    [[nodiscard]] StaticDesignEvaluation evaluateAnalyses(std::span<const double> x, const DesignAnalyses& analyses,
                                                          const solutions::StaticOptions& options,
                                                          base::MemoryBudget& budget,
                                                          std::span<const SuppliedField> supplied = {},
                                                          bool needsDisplacements = false) const;
    // Whether a response of the problem is a LAMA.
    [[nodiscard]] bool hasBucklingResponses() const;

    // The responses, constraints and objective of an analysis made by evaluateStatics (the outputs the responses read
    // are added to the options' requests by evaluateStatics itself). Runs on the calling thread: the callables of the
    // equation responses are called from it, by batch (`blockItems` items per call). `initialSupplied`: the values of
    // the relations given by functions at the initial design (FRMASS reads the mass there). Throws as
    // DesignResponses::evaluate, DesignConstraints::evaluate and objectiveValue.
    //
    // The eigenvalue responses that track their mode take, in each subcase, the mode whose shape
    // matches the shape of `references` best (ModeTracking: the modal assurance criterion), the mode of their ATTA
    // without a reference; the evaluation records the modes taken and their shapes (DesignEvaluation::trackedModes).
    [[nodiscard]] DesignEvaluation evaluateResponses(StaticDesignEvaluation analysis, const CallableRegistry& callables,
                                                     base::MemoryBudget& budget,
                                                     std::span<const SuppliedField> initialSupplied = {},
                                                     std::size_t blockItems = EQUATION_BATCH_ITEMS,
                                                     const ExternalRegistry& externals = {},
                                                     const ModeReferences* references = nullptr) const;

    // The gradients of the responses of an evaluation with respect to x (StaticSensitivity), with the gradients of its
    // normalised constraints and of its objective. The analysis must have been made with needsDisplacements;
    // `analysisOptions` gives the kernel and the threads of the sensitivity factorisation. The callables of the
    // equation responses are called on the calling thread with their partials. Throws as StaticSensitivity::compute.
    [[nodiscard]] DesignGradients gradients(const DesignEvaluation& evaluation, const CallableRegistry& callables,
                                            base::MemoryBudget& budget, const GradientOptions& options = {},
                                            const solutions::StaticOptions& analysisOptions = {},
                                            std::size_t blockItems = EQUATION_BATCH_ITEMS,
                                            const ExternalRegistry* externals = nullptr) const;

    // The full gradient mode, computed in blocks and streamed: G of every item of every response of the evaluation
    // (rows in the order of evaluation.responses, item by item; columns the independent variables) handed to `sink`
    // block by block (GradientStream). The analysis must have been made with needsDisplacements, as for gradients().
    // Throws as StaticSensitivity::stream.
    [[nodiscard]] GradientStreamReport streamGradients(const DesignEvaluation& evaluation,
                                                       const CallableRegistry& callables, base::MemoryBudget& budget,
                                                       const GradientStreamOptions& options, GradientSink& sink,
                                                       const solutions::StaticOptions& analysisOptions = {},
                                                       std::size_t blockItems = EQUATION_BATCH_ITEMS,
                                                       const ExternalRegistry* externals = nullptr) const;

    // The size of G for an evaluation (every item of its responses, the independent variables): what a destination of
    // the full mode must hold, known before any sensitivity computation.
    [[nodiscard]] GradientSize gradientSize(const DesignEvaluation& evaluation) const;

    // The validity of the linear analysis of a design point (an optimisation aid; LinearityIndicator):
    // the deflection over thickness of every shell grid point and the rotation of every grid point of the displacement
    // tables of the analysis (all of them when it was made with needsDisplacements, or with the displacements of every
    // point requested), with a warning per subcase above the thresholds. No solve. The geometry of the shells is kept
    // between the points of a design without shape variables (built once from the problem's snapshot, its
    // bytes reserved in `budget` while the report is made; LinearityReport::geometry); the volumes on `pool`. Throws
    // as LinearityIndicator.
    [[nodiscard]] LinearityReport linearityReport(const StaticDesignEvaluation& analysis, base::MemoryBudget& budget,
                                                  base::ThreadPool* pool = nullptr) const;

    // The subcases a static analysis of `options` solves.
    [[nodiscard]] std::vector<std::int32_t> analysedSubcases(const solutions::StaticOptions& options) const;
    // Every subcase an evaluation analysed, of every kind (STATICS, MODES, BUCK), in increasing order.
    [[nodiscard]] static std::vector<std::int32_t> analysedSubcasesOf(const StaticDesignEvaluation& analysis);

    // The subcases each response of an evaluation was evaluated in (closed over the equations).
    [[nodiscard]] ResponseSubcases responseSubcases(const DesignEvaluation& evaluation) const;
    // What StaticSensitivity reads of an evaluation (gradients(), streamGradients() and the optimisation aids of
    // design/aids); `evaluation` and `subcases` must outlive it.
    [[nodiscard]] StaticSensitivityInputs sensitivityInputs(const DesignEvaluation& evaluation,
                                                            const ResponseSubcases& subcases,
                                                            const CallableRegistry& callables,
                                                            const solutions::StaticOptions& analysisOptions,
                                                            std::size_t blockItems,
                                                            const ExternalRegistry* externals = nullptr) const;

private:
    // The element problems of the base snapshot (ShapeDistortion), made once, at the first distortion check; shared by
    // the copies of the problem (the snapshot is shared and immutable).
    struct BaseProblems
    {
        std::once_flag once;
        std::vector<base::Diagnostic> problems;
    };

    [[nodiscard]] std::vector<base::Diagnostic> distortionsOf(const input::ModelSnapshot& overlaid,
                                                              base::MemoryBudget& budget) const;
    // The static options of a design point (the outputs, sections, zones, stamps and kept systems its responses and
    // gradients read), as evaluateStatics() gives them to the static solution, and the subcases they solve.
    struct PreparedStatics
    {
        solutions::StaticOptions options;
        std::vector<std::int32_t> subcases;
    };
    [[nodiscard]] PreparedStatics prepareStatics(const DesignPoint& point, const input::ModelSnapshot& overlaid,
                                                 const solutions::StaticOptions& options,
                                                 bool needsDisplacements) const;
    // The design point x, its overlaid snapshot and its distortions (an error refuses the point).
    struct PreparedPoint
    {
        DesignPoint point;
        std::shared_ptr<const input::ModelSnapshot> overlaid;
        std::vector<base::Diagnostic> distortions;
    };
    [[nodiscard]] PreparedPoint preparePoint(std::span<const double> x, std::span<const SuppliedField> supplied,
                                             base::MemoryBudget& budget) const;
    // The subcases of each response of an evaluation (DesignConstraints, closed over the equations), each response in
    // the subcases of its analysis only (DesignResponses::isEvaluatedIn).
    [[nodiscard]] ResponseSubcases subcasesOfEvaluation(const StaticDesignEvaluation& analysis) const;
    // The mode each eigenvalue response that tracks its mode takes in each of its subcases (ModeTracking against
    // `references`), into `tracked`, with a warning per uncertain match.
    void trackModes(const StaticDesignEvaluation& analysis, const ResponseSubcases& subcases,
                    const ModeReferences* references, std::vector<TrackedMode>& tracked,
                    std::vector<base::Diagnostic>& warnings) const;

    std::shared_ptr<const input::ModelSnapshot> m_snapshot;
    DesignModel m_model;
    FieldTargets m_targets;
    DesignResponses m_responses;
    DesignConstraints m_constraints;
    bool m_deckOutputs = true;
    std::shared_ptr<BaseProblems> m_baseProblems = std::make_shared<BaseProblems>();
    // The geometry of the linearity check kept between the design points, built at the first report of a
    // design that moves no grid point; shared by the copies of the problem (the snapshot is shared and immutable).
    struct KeptLinearityGeometry
    {
        std::mutex mutex;
        std::shared_ptr<const base::KeptObject<LinearityGeometry>> geometry; // guarded by mutex
    };
    std::shared_ptr<KeptLinearityGeometry> m_keptLinearity = std::make_shared<KeptLinearityGeometry>();
};

} // namespace oos::design
