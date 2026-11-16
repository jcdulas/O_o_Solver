#pragma once

#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElementEquations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/core/ElementDof.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::assembly
{

// One term of a relation of the element formulation, on a DOF of an element (grid, scalar or entity).
struct FormulationTerm
{
    elements::ElementDof dof;
    double coefficient = 0.0;
};

// The linear relations an element formulation adds to the constraints of a model, in the form of the rigid elements
// and multipoint constraints: u_d = Σ_j c_j·u_j. The mixed hybrid formulation gives its ties between unknowns (the
// solids that follow the shells and the lines on edges and faces, the rotations of the lines in directions without
// stiffness: docs/mixed-hybrid-formulation-spec.md section 3.4) and its observer relations (a rotation of a shell grid
// point that another element, a load or a relation uses, made the mean rotation of the shells at their corners:
// section 5.4). RigidElimination takes them as one more source (ConstraintSource::Formulation), first, so that a user
// relation that makes one of their dependent DOFs dependent again is the one refused (dof-dependent-twice).
//
// The formulation rows are not reported among the multipoint constraint forces (they are no card of the model), but
// their forces reach the supports' forces: see RigidElimination::computeMpcForces.
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: owns its rows (bounded by the
// formulation's ties and the grid points it observes: std::vector is the right container).
class FormulationRelations
{
public:
    explicit FormulationRelations(const DofMap& dofMap) : m_dofMap(&dofMap) {}

    // Appends the relation u_dependent = Σ terms. The terms with a zero coefficient are left out.
    void add(const elements::ElementDof& dependent, std::span<const FormulationTerm> terms);

    [[nodiscard]] bool empty() const noexcept { return m_dependents.empty(); }
    [[nodiscard]] std::size_t rowCount() const noexcept { return m_dependents.size(); }
    [[nodiscard]] std::int64_t termCount() const noexcept { return static_cast<std::int64_t>(m_terms.size()); }
    // Row i: its dependent g-DOF and its terms on g-DOFs, in the order they were given.
    [[nodiscard]] base::DofIndex dependentDof(std::size_t i) const noexcept { return m_dependents[i]; }
    [[nodiscard]] std::span<const ConstraintTerm> row(std::size_t i) const noexcept
    {
        return {m_terms.data() + m_offsets[i], m_offsets[i + 1] - m_offsets[i]};
    }
    [[nodiscard]] static constexpr ConstraintSource sourceOf(std::size_t) noexcept
    {
        return ConstraintSource::Formulation;
    }
    [[nodiscard]] static constexpr base::EntityId sourceIdOf(std::size_t) noexcept { return 0; }

private:
    const DofMap* m_dofMap;
    std::vector<base::DofIndex> m_dependents;
    std::vector<std::size_t> m_offsets{0};
    std::vector<ConstraintTerm> m_terms;
};

} // namespace oos::assembly
