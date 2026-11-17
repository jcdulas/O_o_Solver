#pragma once

#include "oos/assembly/constraints/FormulationConstraints.hpp"
#include "oos/assembly/constraints/FormulationRelations.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/design/sensitivity/FieldPerturbation.hpp"
#include "oos/elements/core/FormulationModel.hpp"

#include <array>
#include <span>
#include <vector>

namespace oos::design::detail
{

// What a shape changes in the relations of an element formulation (MixedHybridFormulation: its ties and observer
// relations, u_m = Σ c_mj(v)·u_j, functions of the positions; docs/mixed-hybrid-formulation-spec.md section 5.15),
// by differences over the four sides of the shape's perturbation, each side's formulation model giving the rows.

// The formulation model of each side of `perturbation` (one per snapshot, shared by the batches of its set). Throws
// std::logic_error when a side has none (the design point has one: a bug).
[[nodiscard]] std::array<const elements::FormulationModel*, FIELD_SIDE_COUNT> sideFormulationModels(
    const FieldPerturbation& perturbation);

// The rows of the relations of `constraints` on each side's model (FormulationConstraints::relationsOn).
[[nodiscard]] std::array<assembly::FormulationRelations, FIELD_SIDE_COUNT> sideRelations(
    const assembly::FormulationConstraints& constraints,
    const std::array<const elements::FormulationModel*, FIELD_SIDE_COUNT>& models, base::DiagnosticSink& sink);

// The derivative of a g-set vector `values` that satisfies the relations of `gm` (a solution, a mode shape) with
// respect to the shape at fixed independent values: on the dependent DOF m of row i of Gm,
//     D_m = Σ_j (∂c_mj/∂v)·values_j + Σ_j c_mj·D_j,
// the first sum over the formulation's rows (`relations`, on their sides `sides`), the second over each row's own terms
// (RigidElimination::relationTerms: a relation of the model whose independent DOF a formulation row makes dependent
// follows it), by passes until it holds (one per level of the chains); zero on the other DOFs.
[[nodiscard]] std::vector<double> dependentDerivative(
    const assembly::RigidElimination& gm, const assembly::FormulationRelations& relations,
    const std::array<assembly::FormulationRelations, FIELD_SIDE_COUNT>& sides, const FieldPerturbation& perturbation,
    std::span<const double> values);

} // namespace oos::design::detail
