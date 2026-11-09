#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/reduction/condense/StaticCondensation.hpp"
#include "oos/reduction/condense/UnitDescription.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/reduction/external/ExternalUnitAssembly.hpp"
#include "oos/reduction/interface/InterfaceProblem.hpp"
#include "oos/solutions/sol101/KeptUnitModel.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::solutions
{

// One condensed unit of a condensed static run kept for the sensitivities (CondensedFactorisedSet): its equations, its
// condensation (K*, and the interior factor while kept, computed again to the same bits otherwise: the cache keeps no
// factor, never one here beyond the analysis' budget), the map of its local DOFs (UnitSystem numbering) to the model's,
// and what the analysis did with it.
struct KeptCondensedUnit
{
    std::int64_t unitId = 0;
    std::shared_ptr<const reduction::UnitSystem> system;
    // Not const: the sensitivities recompute a released interior factor (StaticCondensation::recomputeInteriorFactor).
    std::shared_ptr<reduction::StaticCondensation> condensation;
    // Per local DOF: its g-set DOF (-1 for a point generated on a fastener side), and whether its point is interior
    // to the unit (a load on an interior row is the unit's, on a boundary row the interface's, so that each load acts
    // once).
    base::BudgetedBuffer<base::DofIndex> modelDofOfLocal;
    base::BudgetedBuffer<std::uint8_t> isInteriorLocal;
    bool isRecovered = false; // the analysis recovered it: its interior displacements are in the results
    bool isDriven = false;    // a design point drives it (its design stamp is not empty)
    // The load basis handed over in a memory mode that keeps no factor (the memory plan's strategy in sequence): its
    // interior factor released after each use (the condensation of the basis vectors' loads, each block's recovery of
    // its interior), recovered by regions of at most regionValues panel values when its factor exceeds the plan's bound
    // (0: factorised again whole); and when the plan releases its equations, `system` is the system without them, built
    // again for each use from `description` on `rebuild`.
    bool releasesFactor = false;
    std::int64_t regionValues = 0;
    reduction::UnitDescription description;
    std::shared_ptr<const KeptUnitModel> rebuild;
};

// The units of one condensed run, in the order of the plan's condensed units (the order of the units of every interface
// problem of the run, InterfaceProblem::unitLocalDofs), shared by its kept constraint sets.
using KeptCondensedUnits = std::vector<KeptCondensedUnit>;

// The inertia relief of a condensed constraint set (CondensedStatics: D = Φ·C, Φ the rigid-body motions about the basic
// origin, C fitted on the interface's support motions), kept for the sensitivities: C (6 × r, column j at
// fit[6·j ... 6·j + 5]) and the accelerations ü_r of each subcase solved on the set, in order.
struct CondensedRelief
{
    std::size_t supportCount = 0;
    std::vector<double> fit;
    std::vector<std::vector<double>> accelerations;
    // The load basis handed over: DᵀMD = Cᵀ·Mr·C (r × r, row-major), the relief loads of the interface for a unit
    // acceleration of each rigid-body motion (local DOFs × 6: the units' condensed inertial patterns and the
    // interface's own inertial loads), the FreeBodySupport DOFs' local DOFs, and whether the interface holds
    // conditioned supports in their place (assembly::chooseConditionedSupports: the displacements then brought back to
    // their gauge, u ← u − Φ·C·u_r).
    std::vector<double> reducedMass;
    // With six supports (C exact): Mr, the 6 × 6 rigid-body mass about the basic origin (row-major), the relief
    // amplitudes a = Mr⁻¹·Φᵀ·P made without C; empty otherwise.
    std::vector<double> rigidMass;
    linalg::DenseMatrix reliefLoads;
    std::vector<std::size_t> supportLocals;
    bool isConditioned = false;
};

// The parts a CondensedFactorisedSet keeps, as the condensed path hands them over.
struct CondensedFactorisedSetParts
{
    std::shared_ptr<const elements::SnapshotIndex> index; // the points the DOF map refers to
    std::shared_ptr<const assembly::DofMap> dofMap;
    std::shared_ptr<const KeptCondensedUnits> units;
    std::unique_ptr<const reduction::InterfaceProblem> interface; // factorised
    std::unique_ptr<const assembly::RigidElimination> gm;         // the full path's Gm of the set (SPC force weights)
    base::BudgetedBuffer<std::uint8_t> isConstrained; // per g-set DOF: held at the interface or inside a unit
    // The external units placed at the interface, in InterfaceProblem::externalLocalDofs order.
    std::vector<std::shared_ptr<const reduction::ExternalUnitAssembly>> externalUnits;
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    std::vector<base::EntityId> subcases; // the subcases solved on it, in solve order
    // The inertia relief of the set (PARAM INREL = −1), null without one.
    std::shared_ptr<const CondensedRelief> relief;
};

// One constraint set of a condensed static run kept after the run (StaticOptions::keepsSystems with a condensation
// plan, StaticResults::condensedSystems): the condensed path's counterpart of FactorisedSet. The static
// sensitivities of a design point solve on it through the boundaries (design::BoundaryReducedSensitivity: with
// a condensation, reduced by the boundaries): the pseudo-loads of the driven units condensed with their
// interior factors, the interface problem solved with its factor already made for the analysis, the units read
// recovered; a unit neither driven nor read is not touched.
//
// The solves take the caller's kernels: those of the run may be gone; the same kernel kind gives the same bits.
//
// Memory: every part stays reserved in the run's memory budget until the last owner drops the set (the budget must
// outlive it). The interior factors of the units stay as the analysis left them (kept or released by FactorRetention);
// a released one is computed again by the sensitivities, to the same bits, and stays in the run's budget from then on.
//
// Thread-safety: the parts are immutable, but the units' condensations are not (a factor computed again): one user at a
// time. Ownership: shares its parts; borrows, through the DOF map and the index, the snapshot of the run, which must
// outlive it.
class CondensedFactorisedSet
{
public:
    // Throws std::invalid_argument when a part is missing.
    explicit CondensedFactorisedSet(CondensedFactorisedSetParts parts);

    [[nodiscard]] const elements::SnapshotIndex& index() const noexcept { return *m_parts.index; }
    [[nodiscard]] const assembly::DofMap& dofMap() const noexcept { return *m_parts.dofMap; }
    [[nodiscard]] const KeptCondensedUnits& units() const noexcept { return *m_parts.units; }
    [[nodiscard]] const reduction::InterfaceProblem& interface() const noexcept { return *m_parts.interface; }
    [[nodiscard]] const assembly::RigidElimination& gm() const noexcept { return *m_parts.gm; }
    [[nodiscard]] std::span<const std::uint8_t> isConstrained() const noexcept { return m_parts.isConstrained.span(); }
    [[nodiscard]] const std::vector<std::shared_ptr<const reduction::ExternalUnitAssembly>>& externalUnits()
        const noexcept
    {
        return m_parts.externalUnits;
    }
    [[nodiscard]] linalg::KernelKind kernel() const noexcept { return m_parts.kernel; }
    [[nodiscard]] const std::vector<base::EntityId>& subcases() const noexcept { return m_parts.subcases; }
    // The inertia relief of the set (null without one).
    [[nodiscard]] const CondensedRelief* relief() const noexcept { return m_parts.relief.get(); }
    [[nodiscard]] bool solves(base::EntityId subcase) const
    {
        return std::find(m_parts.subcases.begin(), m_parts.subcases.end(), subcase) != m_parts.subcases.end();
    }

private:
    CondensedFactorisedSetParts m_parts;
};

} // namespace oos::solutions
