#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/FastenerPiercingTable.hpp"
#include "oos/reduction/absorption/SpiderWeights.hpp"
#include "oos/reduction/condense/UnitDescription.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace oos::reduction
{

// The units a plan condenses, as their condensation and the interface problem take them (a UnitDescription built from
// each UnitPlan):
// - the description of each condensed unit (UnitDescription): its points and elements as the plan has them; its
//   boundary, the plan's but for the pierced element's grid point standing for the point generated on an absorbed
// fastener
//   side (SpiderDecision::boundaryNodeId: the plan's estimates are made on model points) when nothing else keeps it
//   there, and with the GA (GB) a fastener card gives on an absorbed side (the reference of its change of basis); its
//   spiders absorbed by change of basis: a RigidInterpolation by its rows of the units' Gm
//   (SpiderWeights::rigidInterpolationRelation), a fastener side with a given GA by that point's rows
//   (SpiderWeights::fastenerPointRelation), a generated side by the interpolation of its pierced element
//   (FastenerPiercing::relation) at its piercing point (an absorbed RigidBodyStar needs
// none: its own elimination);
// - the relations eliminated inside the units (one model-wide assembly::RigidElimination with unitRowFilter()):
//   every relation that is not the connection system's (CondensationPlan's connection relations), and the rows of the
//   given GA (GB) of an absorbed fastener side, whose fastener stays in the connection system;
// - the relations eliminated at the interface (interfaceRowFilter()): every row but those a condensed unit eliminates
//   inside it, the rows of its interior dependent DOFs and of its spiders' references (those DOFs are its unknowns at
//   the interface); a row a unit eliminates at one of its boundary points is eliminated at the interface too, on the
//   interface's part of the stiffness (the elimination is a congruence: Tᵀ·(Σ K)·T = Σ Tᵀ·K·T);
// - the points an element of the interface reaches inside a condensed unit (reachedPoints: the DOFs of the matrices of
//   the elements of no condensed unit) put on its boundary when the plan's boundary misses them (the grid points
//   of a fastener's pierced element the connection tables do not give), as BoundaryPromotion does for the relations, so
// that no coupling
//   is lost; the legs of an absorbed fastener side, which its own fastener reaches through its change of basis, stay
//   interior;
// - which points of the model are interior to a condensed unit, and which elements are the condensed units'.
//
// Thread-safety: immutable once built; the filters may be called from any thread. Ownership: copies what it needs
// from the plan (lists bounded by the plan); borrows the DOF map, which must outlive it.
class PlannedUnits
{
public:
    // reachedPoints: the points the elements of no condensed unit reach, sorted. Throws std::invalid_argument for a
    // plan that does not fit the DOF map (a point it does not have).
    PlannedUnits(const CondensationPlan& plan, const assembly::DofMap& dofMap, base::MemoryBudget& budget,
                 std::span<const base::EntityId> reachedPoints = {});

    // The plan's condensed units (indices into CondensationPlan::units(), increasing).
    [[nodiscard]] std::span<const std::size_t> condensedUnits() const noexcept { return m_condensed; }

    [[nodiscard]] assembly::ConstraintRowFilter unitRowFilter() const;
    [[nodiscard]] assembly::ConstraintRowFilter interfaceRowFilter() const;

    // The description of the condensed unit at index `unit` of CondensationPlan::units(). unitRelations: the Gm of
    // the units (unitRowFilter()); piercings: the fastener piercing table (null: none). Throws std::invalid_argument
    // when a spider's relation cannot be made (a RigidInterpolation without rows, a fastener side whose shape
    // functions are not an interpolation).
    [[nodiscard]] UnitDescription describe(std::size_t unit, const SpiderWeights& unitRelations,
                                           const input::FastenerPiercingTable* piercings) const;

    // One flag per node of the DOF map: nonzero for the interior points of the condensed units.
    [[nodiscard]] std::span<const std::uint8_t> interiorNodes() const noexcept { return m_isInteriorNode.span(); }
    // The points of a condensed unit (index into CondensationPlan::units()) put on its boundary beyond the plan's
    // (reachedPoints), sorted.
    [[nodiscard]] std::span<const base::EntityId> promotedPoints(std::size_t unit) const noexcept;
    // Whether an element is one of a condensed unit's (its matrices are in that unit's K*).
    [[nodiscard]] bool isCondensedElement(base::EntityId elementId) const noexcept;
    // The condensed unit (index into CondensationPlan::units()) whose interior holds a node, or nothing.
    [[nodiscard]] std::optional<std::size_t> unitOfInteriorNode(std::int32_t node) const noexcept;

private:
    // The key of the relation of a row, as the plan names its relations (RelationKey).
    [[nodiscard]] RelationKey keyOf(const assembly::ConstraintRowOrigin& origin) const;
    [[nodiscard]] bool isGivenSideRow(const assembly::ConstraintRowOrigin& origin, bool condensedOnly) const;

    const CondensationPlan* m_plan;
    const assembly::DofMap* m_dofMap;
    std::vector<std::size_t> m_condensed; // bounded by the units
    std::vector<RelationKey> m_relations; // the connection system's, sorted
    // The absorbed fastener sides whose card gives GA (GB): (fastener ID, point ID), sorted; the condensed ones apart.
    std::vector<std::pair<base::EntityId, base::EntityId>> m_givenSides;
    std::vector<std::pair<base::EntityId, base::EntityId>> m_condensedGivenSides;
    // The RigidInterpolation elements absorbed into condensed units, sorted.
    std::vector<base::EntityId> m_condensedRigidInterpolations;
    base::BudgetedBuffer<std::uint8_t> m_isInteriorNode;
    base::BudgetedBuffer<std::int32_t> m_unitOfNode;          // per node: the condensed unit holding it inside, or -1
    base::BudgetedBuffer<base::EntityId> m_condensedElements; // sorted
    std::vector<std::vector<base::EntityId>> m_promoted;      // per unit of the plan (bounded by its points)
};

} // namespace oos::reduction
