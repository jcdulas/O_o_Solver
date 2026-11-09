#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/model/FieldTargets.hpp"
#include "oos/design/responses/DesignResponses.hpp"
#include "oos/design/sensitivity/DesignGradients.hpp"
#include "oos/design/sensitivity/GradientStream.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"
#include "oos/solutions/sol103/ModesSolution.hpp"
#include "oos/solutions/sol105/BucklingSolution.hpp"
#include "oos/solutions/sol144/TrimSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::design
{

// What the static sensitivities of one design point read: the design model and its binding, the responses, the base
// snapshot and the overlaid one the analysis ran on, the analysis results (the displacements of every point: the
// analysis must have been asked for them), the responses' values and the subcases they were evaluated in.
struct StaticSensitivityInputs
{
    const DesignModel* model = nullptr;
    const FieldTargets* targets = nullptr;
    const DesignResponses* responses = nullptr;
    std::shared_ptr<const input::ModelSnapshot> base;     // the snapshot the targets were bound to
    std::shared_ptr<const input::ModelSnapshot> overlaid; // the design point's snapshot (the analysis read it)
    const DesignValues* design = nullptr;
    // The field values of the point whose meshes the finite element sections of the linearisation take (empty: the
    // design's own; a moved state of the Hessian-vector products takes its point's).
    std::vector<double> sectionReference;
    const solutions::StaticResults* results = nullptr;
    std::span<const std::int32_t> subcases;              // the analysed static subcases (and preloads), in order
    const std::vector<ResponseValues>* values = nullptr; // the evaluated responses (DesignResponses::ids() order)
    const ResponseSubcases* responseSubcases = nullptr;  // the subcases each was evaluated in (closed)
    const CallableRegistry* callables = nullptr;
    const ExternalRegistry* externals = nullptr; // the evaluators of the external responses
    std::size_t blockItems = EQUATION_BATCH_ITEMS;
    elements::MassFormulation massFormulation = elements::MassFormulation::Lumped;
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    base::ThreadPool* pool = nullptr;
    // The eigenvalue analyses of the design point (null without MODES or BUCK subcases): the roots and shapes
    // the EIGN, FREQ and LAMA items read (their preloads among `subcases`).
    const solutions::ModesResults* modes = nullptr;
    const solutions::BucklingResults* buckling = nullptr;
    // The trim of the SAERO subcases (null without them), with its linearisation (TrimResults::linearisation:
    // the gradients solve on it).
    const solutions::TrimResults* trim = nullptr;
};

// The gradients of the static responses of a design point: the
// semi-analytic method.
//
//  1. The items are linearised: every item gets its displacement weights c = ∂r/∂u_g (a DISP item the unit vector of
//     its DOF; an SPC force wᵀ·(K·u − P) the weights K·w, w the constrained DOF and the m-set rows of Gm that read it;
//     an element item the central differences of its recovery with respect to the element's displacements,
//     ElementItemFunction; an equation item the chained weights of its arguments, ChainedAdjoint) and its explicit
//     part ∂r/∂x at fixed u.
//  2. For every designed field (each relation; the fields no variable moves are skipped), the snapshots with the field
//     at v ± h and v ± h/2 (FieldPerturbation): on the elements it reaches (AffectedElements), the derivatives of their
//     stiffness, thermal loads (ElementDerivatives), and of the mass (the inertial loads) and the distributed loads of
//     the subcases that have them, form the pseudo-loads q_rs = ∂P_s/∂v − ∂K/∂v·u_s (PseudoLoads); the element items on
//     those elements, the SPC forces (−wᵀ·q_rs) and the mass properties (WEIGHT, VOLUME, FRMASS: MassProperties on
//     the two snapshots, exact for their linear dependence on the fields) get their explicit parts, through
//     d(field)/dx.
//  3. Per subcase, MethodChoice counts the solves of the three orders and DirectSensitivity, AdjointSensitivity or
//     ChainedAdjoint forms the gradients on the factor of its constraint set (SensitivitySystem: the factor the
//     analysis kept, StaticResults::systems, else one factorisation per constraint set); the global items (no
//     displacement weights) keep their explicit parts.
//
// Checks: the values every linearised item recomputes at the design point equal the analysis' values (an element item
// recovered again, an equation evaluated again with its partials), else OptimisationError: a derivative of another
// function than the response is never returned.
//
// Refused (OptimisationError naming it): an element type without the Derivatives facet that a designed field reaches;
// inertia relief and SUBCOMs (SensitivitySystem); a DesignResp3 callable without gradient=.
//
// Determinism: every loop is serial in a fixed order (fields, elements, subcases, items); the factorisation and the
// block solves are deterministic for any number of threads.
//
// Thread-safety: run on the calling thread (the callables of the equations are called from it). Ownership: none kept.
class StaticSensitivity
{
public:
    // The gradients of the responses `options.responseIds` asks for (every evaluated response by default), in the
    // order of `values`, and the report. Throws OptimisationError (above), SolverError, SingularityError,
    // MemoryBudgetError.
    [[nodiscard]] static DesignGradients compute(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                                                 base::MemoryBudget& budget);

    // The full gradient mode: every item of every evaluated response linearised
    // as compute() does (steps 1 and 2 above, no item mask), then G streamed to `sink` block by block by GradientStream
    // (the rows of G are the items in the order of `values`, response by response; its columns the independent
    // variables). Throws as compute(), and what GradientStream::run and the sink throw.
    [[nodiscard]] static GradientStreamReport stream(const StaticSensitivityInputs& inputs,
                                                     const GradientStreamOptions& options, GradientSink& sink,
                                                     base::MemoryBudget& budget);
};

} // namespace oos::design
