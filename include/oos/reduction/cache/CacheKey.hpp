#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/core/ElementFormulationVersion.hpp"
#include "oos/reduction/condense/StaticCondensation.hpp"
#include "oos/reduction/condense/UnitDescription.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::reduction
{

// A PARAM that changes the matrices of a unit (WTMASS, COUPMASS, AUTOSPC and the like), with its value as text
// ("" when the model does not give it: the default).
struct CacheParameter
{
    std::string name;
    std::string value;

    friend bool operator==(const CacheParameter&, const CacheParameter&) = default;
};

// The PARAMs whose value changes what a unit is condensed from: WTMASS and COUPMASS (the mass of the inertial patterns
// and of the rigid-body mass properties), AUTOSPC and EPZERO (which interior DOFs the unit's singularity processor
// holds). MAXRATIO and BAILOUT only test the pivots (the same bits when the factorisation goes through) and stay out,
// as does K6ROT, which changes no matrix (solutions::checkSupport reports it ignored).
// By their neutral names (the names of the params table).
inline constexpr std::array<std::string_view, 4> MATRIX_PARAMETERS{"AutoSingularityConstr", "CoupledMass",
                                                                   "PivotZeroThreshold", "WeightToMass"};

// The model parameter beside them, under this name in CacheKey::parameters with the value
// input::offsetDefinitionName gives: ModelParams OFFDEF, where the mass of the offset elements is (the reduced mass,
// the inertial patterns and the rigid-body mass properties).
inline constexpr std::string_view OFFSET_DEFINITION_PARAMETER = "ModelParams OffsetDefinition";

// The key of one condensation in the condensation cache: its parts kept apart so that a key that differs from a stored
// one says why (differenceFrom), every one made from the unit's inputs without its equations (makeUnitKey: a hit then
// needs no build of them), and the checksum of the equations an entry was condensed from, kept as its verification:
// - solverVersion, kernel: the version of O_o_Solver and the dense kernel family the entry was condensed with (another
//   one gives other bits: refused, named);
// - elementFormulation: the version of the element formulations the unit's matrices and loads were computed with
//   (elements::ELEMENT_FORMULATION_VERSION): a change of an element's stiffness, mass or loads
//   between two builds of one version condenses every unit again (the stamps see the model, not the formulation);
// - unitIdentity: the hash of the unit's element IDs (a unit "identified by the hash of its element
//   IDs"): the same unit across edits, which tells a superseded entry and names the cause of a miss; unitId is the
//   plan's ID, for the reports only;
// - revision: O_o_Reader's revision stamp of the unit (its cards and what they depend on), the stamps of its absorbed
//   clusters and spiders mixed in by the plan (UnitPlan::revision);
// - designStamp: the hash of the values a design point gives to the unit's fields and grid points
//   (design::computeDesignStamps; EMPTY_DESIGN_STAMP, 0, when no variable drives it): a design point is applied as an
//   overlay, which O_o_Reader's stamps do not see;
// - constraintStamp: the DOFs of the unit held in every subcase (the constraints applied before reducing; the enforced
//   values are loads, in the patterns), before the singularity processor (whose AUTOSPC is a function of the rest);
// - boundaryStamp: its boundary points (the rows of K* follow from them, the DOFs held and the relations);
// - spiderStamp: its spiders absorbed by change of basis, their keys and the bits of their weighting matrices (the
//   absorption decisions and the stamps of the absorbed spiders);
// - reduction, modeCount, modeCutoff: static, or Craig-Bampton with its mode request: modeCount 0 for the
//   boundary part of a Craig-Bampton unit (K*, its reduced boundary mass), N > 0 for its N lowest fixed-interface
//   modes, MODE_CUTOFF_REQUEST with the bits of the cut-off eigenvalue in modeCutoff for its modes below a cut-off;
// - parameters: the MATRIX_PARAMETERS and OFFSET_DEFINITION_PARAMETER with their values, by name;
// - patternStamp: what the condensed load patterns depend on: UnitLoadBasis::patternStamp (the loaded DOFs for the
//   loaded-DOF basis, so that new load values on the same DOFs reuse the condensation; the load values otherwise) and
//   a checksum of the patterns on the unit's DOFs (the mass of the inertial patterns and of the rigid-body mass
//   properties is the elements', with inputStamp's non-structural mass);
// - inputStamp: a checksum of the unit's inputs as the solver reads them beside the stamps, so that a
//   change no stamp sees (an overlay, a selection, a parameter) condenses again: its points in the model's order (IDs,
//   positions, displacement systems), its elements in the order of the model's sums (batch, row, card), the relations
//   of the units' Gm on its DOFs, the non-structural mass of its elements and the options of its assembly, its mass
//   and its singularity processor (unitInputStamp);
// - equationStamp: the checksum of the equations the entry was condensed from (equationStampOf), stored with it and not
//   matched: whenever a hit's equations are built they are compared with it, a difference being a key that missed an
//   input.
//
// Thread-safety: a value. Ownership: owns its texts.
struct CacheKey
{
    // The modeCount of a request of the fixed-interface modes below a cut-off (its eigenvalue in modeCutoff).
    static constexpr std::int32_t MODE_CUTOFF_REQUEST = -1;

    std::string solverVersion;
    std::string kernel;
    std::uint32_t elementFormulation = elements::ELEMENT_FORMULATION_VERSION;
    std::int64_t unitId = 0;
    std::uint64_t unitIdentity = 0;
    std::int64_t revision = 0;
    std::uint64_t designStamp = 0;
    std::uint64_t constraintStamp = 0;
    std::uint64_t boundaryStamp = 0;
    std::uint64_t spiderStamp = 0;
    ReductionKind reduction = ReductionKind::Static;
    std::int32_t modeCount = 0;
    std::uint64_t modeCutoff = 0;           // the bits of the cut-off eigenvalue (std::bit_cast), with a cut-off
    std::vector<CacheParameter> parameters; // sorted by name
    std::uint64_t patternStamp = 0;
    std::uint64_t inputStamp = 0;
    std::uint64_t equationStamp = 0; // stored, not matched: 0 until the equations are built

    // The hash of every part but unitId and equationStamp (CacheHasher): the name of the entry in the cache.
    [[nodiscard]] std::uint64_t hash() const noexcept;

    // Why this key does not take an entry stored with `stored` (the same unit), as the report says it: the first part
    // that differs, in the order version, kernel, element formulation, parameters, then the unit's revision, design,
    // constraints, boundary, spiders, reduction, patterns and inputs ("parameter WTMASS changed (1.0 -> 0.00259)", "the
    // unit's revision stamp changed (an edit of its cards or of what they depend on)"); empty when the keys are equal.
    [[nodiscard]] std::string differenceFrom(const CacheKey& stored) const;

    // Whether an entry stored with `stored` serves this key: every part equal but unitId (the same unit may have
    // another ID in another plan) and equationStamp (a verification).
    [[nodiscard]] bool matches(const CacheKey& stored) const;
};

// What the key of a unit's condensation is made from: its plan (identity, revision), its description (its points,
// elements, boundary and absorbed spiders), the model it is cut from (UnitModel: the inputs its equations are built
// from), its load patterns, the reduction the analysis makes, and the analysis' values. No equation is read.
struct UnitKeySources
{
    const UnitPlan* plan = nullptr;
    const UnitDescription* description = nullptr;
    const UnitModel* model = nullptr;
    // On the unit's local DOFs (UnitSystem's numbering), the load patterns alone: the loads on the interior rows, the
    // enforced values on the DOFs held in every subcase (the inertial patterns follow from the mass: inputStamp).
    UnitPatterns patterns;
    // Whether the six inertial patterns of inertia relief follow them (an entry condensed with them has their columns).
    bool hasInertialPatterns = false;
    std::uint64_t designStamp = 0;
    std::span<const CacheParameter> parameters; // sorted by name
    std::string_view solverVersion;
    std::string_view kernel;
    // The version of the element formulations the unit's equations are built with (the build's own but in the tests).
    std::uint32_t elementFormulation = elements::ELEMENT_FORMULATION_VERSION;
    // The reduction of the entry: Static, or CraigBampton with its mode request (CacheKey::modeCount, modeCutoff); the
    // analysis decides it, not the plan (the static preload of a buckling analysis of a Craig-Bampton plan is static).
    ReductionKind reduction = ReductionKind::Static;
    std::int32_t modeCount = 0;
    double modeCutoff = 0.0;
};

// The key of the condensation of a unit, from its inputs (its equationStamp 0: equationStampOf once they are built),
// its work buffers (a unit's points) in `budget`. Throws std::invalid_argument when a source is missing or a point of
// the description is not the model's, MemoryBudgetError.
[[nodiscard]] CacheKey makeUnitKey(const UnitKeySources& sources, base::MemoryBudget& budget);

// The checksum of a unit's inputs beside the stamps (CacheKey::inputStamp), in O(points + elements + relation terms):
// what UnitSystem reads of `model` for `description` that O_o_Reader's revision stamps do not see. Throws
// std::invalid_argument for a point the model does not have, MemoryBudgetError.
[[nodiscard]] std::uint64_t unitInputStamp(const UnitModel& model, const UnitDescription& description,
                                           base::MemoryBudget& budget);

// The checksum of a unit's equations (CacheKey::equationStamp): the matrix condensed, in the order of its equations,
// and its trailing (boundary) count; with a Craig-Bampton reduction its mass too (UnitSystem::mass(), which the
// fixed-interface modes and the reduced mass are made from). Precondition: system.hasEquations().
[[nodiscard]] std::uint64_t equationStampOf(const UnitSystem& system, ReductionKind reduction);

// The identity of a unit: the hash of its element IDs (sorted).
[[nodiscard]] std::uint64_t unitIdentityOf(std::span<const base::EntityId> sortedElementIds) noexcept;

} // namespace oos::reduction
