#pragma once

#include "oos/base/diagnostics/SupportReport.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/snapshot/SupportCatalog.hpp"

#include <cstdint>
#include <string_view>

namespace oos::solutions
{

// The analysis a run makes, for the cards it reads: a linear static analysis (SOL 101), real eigenvalues (SOL 103:
// LanczosEigenMethod and RealEigenMethod read), linear buckling (SOL 105: LanczosEigenMethod and BucklingEigenMethod
// read, and the static cards of its preload), a static aeroelastic trim (SOL 144: the static cards; the aerodynamic
// cards are tables of their own, which the aero module reads and refuses itself).
enum class AnalysisKind : std::uint8_t
{
    Statics,
    Modes,
    Buckling,
    StaticAeroelastic
};

// The support catalog of a linear static run: input::SupportCatalog::current(), the element, property and material
// cards of the registry with their field rules (ElementRegistry::addSupportTo, which input cannot call itself, being
// below elements), and the constraint and load cards the assembly applies: SPC, SinglePointConstrList,
// EnforcedMotionConstr (singlePointConstrTable, SinglePointConstrUnion expanded), GRID.PS (the permanent constraints),
// MPC (multiPointConstrTable, MultiPointConstrUnion expanded), FORCE, PointForceLoad1, PointForceLoad2, MOMENT,
// PointMomentLoad1, PointMomentLoad2, ScalarPointLoad, GravityLoad (loadsTable, LOAD expanded), ACCEL, AccelNodeLoad,
// RotationalForceLoad (inertialLoadsTable), PointsPressureLoad, ShellPressureLoad, FacePressureLoad (pressureTable),
// BeamDistributedLoad (lineLoadsTable), the temperature cards, FreeBodySupport and SUPORT1 (dofSetsTable), and the
// rigid elements RigidBodyMulti, RigidBodyStar, RigidInterpolation, RigidBar, RigidRod (rigidElementsTable;
// RigidElimination refuses what it cannot eliminate).
[[nodiscard]] input::SupportCatalog staticSupportCatalog(const elements::ElementRegistry& registry);

// Whether a static run ignores the card (the cards with no effect on the analysis are ignored with a warning, never
// refused): the damping elements and properties, the frequency-dependent properties, the thermal-only materials, the
// plot elements, and the eigenvalue methods, which a static run does not read. The snapshot holds them (element batches
// without a family, property, material and method tables), and the assembly leaves them out (assembly::ElementSet keeps
// only the batches with a family).
[[nodiscard]] bool isIgnoredByStaticRun(std::string_view card) noexcept;

// The support check of a static run: staticSupportCatalog(registry).check(snapshot), with the refusals of the cards a
// static run ignores moved to SupportReport::ignored as warnings.
[[nodiscard]] base::SupportReport checkStaticSupport(const input::ModelSnapshot& snapshot,
                                                     const elements::ElementRegistry& registry);

// The support check of a run of `analysis`: that of a static run, with the eigenvalue method cards the analysis reads
// accepted (SOL 103: LanczosEigenMethod and RealEigenMethod; SOL 105: LanczosEigenMethod and BucklingEigenMethod) and
// the others ignored as a static run ignores them. The loads of a SOL 103 run are accepted and not applied (a vibration
// analysis has no load). The PARAMs that change nothing in this version (K6ROT: no shell has a drilling penalty) are
// listed in SupportReport::ignored as warnings
// ("ignored-parameter"), whether given in the Bulk Data or in the Case Control.
[[nodiscard]] base::SupportReport checkSupport(const input::ModelSnapshot& snapshot,
                                               const elements::ElementRegistry& registry, AnalysisKind analysis);

} // namespace oos::solutions
