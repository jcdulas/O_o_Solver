#pragma once

#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/ids/EntityId.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace oos::assembly
{

// Carries the forces on the entity DOFs of an element formulation (the unknowns of the edges, faces and interiors of
// the mixed hybrid formulation: docs/mixed-hybrid-formulation-spec.md section 5.1) to the grid points of their
// entities, for the tables that name grid points only (the grid point force balance, section 10.1).
//
// The forces on the DOFs of one entity are replaced by forces and moments at its grid points that have the same
// resultant force and moment: the work of the entity's forces in each of the six rigid motions (the values the
// motion gives its DOFs, DofMap::addAffineEntityField) is the work of the carried forces in that motion. Among the
// sets with that resultant the one of least norm is taken, Σ|F_i|² + Σ|M_i|² / (w·L²) with L the size of the entity
// and w = ROTATION_WEIGHT small: moments are used only where forces cannot give the resultant (the torque about an
// edge, which its two grid points carry as moments about it).
//
// The map depends on the entity alone, never on the element or the load the force comes from, so that what balances
// on an entity DOF still balances once carried: the forces of the elements, the loads, the forces of the relations
// and the reactions of the supports the formulation derives (DerivedSupportDof::reactionTo, made with termsOf).
//
// Thread-safety: immutable once built; concurrent calls are safe. Ownership: holds the six rigid motions of the
// entity DOFs and the grid points of the entities (std::vector: bounded by the entity DOFs, built once per table).
class EntityForceCarry
{
public:
    // The weight w of the moments against the forces.
    static constexpr double ROTATION_WEIGHT = 1.0e-6;

    // `dofMap` must outlive this.
    explicit EntityForceCarry(const DofMap& dofMap);

    // Whether the model has entity DOFs (without them carry() does nothing).
    [[nodiscard]] bool hasEntities() const noexcept { return !m_modes.empty(); }

    using Add = std::function<void(base::DofIndex gridDof, double value)>;
    // The forces `forces` on the g-DOFs `dofs` (an element's): add(gridDof, value) receives what the forces on its
    // entity DOFs carry to each grid component (the grid and scalar DOFs among `dofs` are left to the caller).
    void carry(std::span<const base::DofIndex> dofs, std::span<const double> forces, const Add& add) const;
    // On a g-set vector of forces: the forces on the entity DOFs added, carried, to the grid DOFs, then set to 0.
    void carry(std::span<double> forcesG) const;
    // What a unit force on the entity DOF `entityDof` carries: the grid DOFs and their shares.
    [[nodiscard]] std::vector<ConstraintTerm> termsOf(base::DofIndex entityDof) const;

private:
    // Adds to the grid points of entity `entity` the forces and moments of resultant `resultant` (force, then moment
    // about the origin of the basic system), in their displacement systems.
    void carryResultant(std::int32_t entity, const std::array<double, 6>& resultant, const Add& add) const;
    const DofMap* m_dofMap;
    std::vector<double> m_modes; // entityRigidModes: mode-major, the entity DOFs of each motion
    std::size_t m_entityDofCount = 0;
    std::vector<std::size_t> m_gridOffsets; // per entity, into m_gridRows
    std::vector<std::int32_t> m_gridRows;
};

} // namespace oos::assembly
