#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/model/FieldTargets.hpp"
#include "oos/design/sensitivity/AffectedElements.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::design
{

// The central-difference step of a shape value: h = SHAPE_STEP_RELATIVE·L/|T|max, L the
// size of the smallest element the shape variable reaches (the largest distance between two of its grid points) and
// |T|max the largest move of its basis per unit of the variable, so that no grid point moves by more than
// SHAPE_STEP_RELATIVE of the smallest element it belongs to; the derivative is extrapolated from the differences at h
// and h/2 as for a field (FieldPerturbation: truncation of order (h/ℓ)⁴, ℓ the length over which the element's
// computation changes, rounding of order ε·|K|/(h/ℓ) times the cancellation of the pseudo-load, K·u ≫ P).
//
// Why 3e-4, where a field takes 1e-2 (FieldPerturbation): here the truncation matters as much as the rounding. The
// element computations are rational functions of the positions (a length, a direction, a Jacobian and their inverses)
// whose high derivatives are large against the element (the fourth-order term of the extrapolation carries their fifth
// derivative), and ℓ can be much shorter than the element: the warp height of a warped ShellQuad4, the thickness of a
// thin shell whose plane tilts (the stresses at its fibres). The rounding, on the other hand, is that of the element's
// own computation, which a thin higher-order shell carries at about 1e-9 of its matrix (bending terms t³ beside
// membrane terms t). Measured, the worst relative error of the shape gradients against the extrapolated finite
// differences of the analysis on small test patches (rods; connectors; plates of unit cells 0.01 thick, a ShellQuad4 and
// a ShellTria3 one with a node 0.04 out of their plane, a ShellQuad8 and a ShellTria6 one stretched and sheared in their
// plane):
//   step    rods     connectors  ShellQuad4 (warped)  ShellTria3 (tilted)  ShellQuad8   ShellTria6
//   1e-3    1.4e-9   1.2e-9      8.5e-5           8.3e-4           6.1e-7   1.1e-6
//   3e-4    2.5e-10  1.1e-9      6.9e-7           6.4e-6           2.8e-7   1.8e-7
//   1e-4    2.7e-10  1.3e-9      8.6e-9           7.7e-8           1.9e-6   3.2e-6
//   3e-5    2.3e-10  4.9e-9      1.3e-9           2.4e-9           5.8e-6   2.5e-5
// (1e-2, the step of the fields, loses 1.7e-5 on the rods, 2.7e-4 on the higher-order shells and the warped ShellQuad4's
// stress entirely.) The truncation falls as h⁴ (the stresses of the tilted thin shells the last), the rounding of the
// higher-order shells grows as 1/h: 3e-4 is the best compromise, every displacement within 3e-7 and every element item
// within 1e-5 (the von Mises stress of the tilted thin ShellTria3, 6.4e-6, the last), where 1e-4 would lose 2e-6 on the
// displacements of the higher-order shells. A forward difference at 1e-4 would lose more (its
// truncation is of order h). A move of 3e-4 of the smallest element distorts no element of a valid mesh enough to
// change its formulation.
//
// When no element the basis reaches has a size (only concentrated masses, or springs between coincident grid points),
// L is the size of the box of every grid point of the model, and, for a model whose grid points all coincide,
// SHAPE_STEP_AT_ZERO_SIZE in the units of the model.
inline constexpr double SHAPE_STEP_RELATIVE = 3.0e-4;
inline constexpr double SHAPE_STEP_AT_ZERO_SIZE = 1.0e-3;

// A rigid element whose grid points a shape variable moves relative to each other: its equations
// (RigidElimination, the lever arms and the RigidInterpolation weights) change with the positions, a derivative of Gm
// the sensitivities do not form.
struct DeformedRigidElement
{
    std::string type; // RigidBodyStar, RigidInterpolation, RigidBar ...
    std::int32_t elementId = 0;
    std::int32_t gridId = 0; // a grid point that moves relative to another of the element
};

// The elements each shape variable reaches, and the step of its shape value (the element
// matrices are differenced with respect to the designed quantity, here the positions along the basis vector).
//
// An element is reached by a shape basis when a grid point whose position its computations read moves: the grid points
// of the DOFs of its stiffness matrix (of its mass matrix for an element without stiffness: PointMassInertia, CMASSi), and
// the orientation point of the line elements and connectors whose card gives one (G0 of Bar2 and Beam2, GO of
// Bushing and Gap2: the int32 column "orientationNode"), and the elements that read a coordinate system the basis
// moves (DerivedGeometry: their int32 columns "frameId", "offsetFrameId", "matFrameId", and the "frameId", "matFrameId"
// of their property). The matrices are those of ElementConnectivity (computed once, for every element of the set: about
// 1 µs for a light element, 7 µs for a ShellQuad4), made only for a design model with shape variables. The derivatives
// of every other element are zero: their computations read no moved position and no moved frame. The step of a basis is
// also bounded by the smallest distance between the grid points that define a direction or a frame it changes.
//
// Thread-safety: immutable once built. Ownership: owns its lists (bounded by the elements the shapes reach); borrows
// the design model and the snapshot of the set, which must outlive it.
class ShapeElements
{
public:
    // `set`: the element set of the design point's snapshot (the one the analysis ran on). Precondition: the set passed
    // its check (the facets throw on an element the check refuses).
    ShapeElements(const DesignModel& model, const FieldTargets& targets, const assembly::ElementSet& set);

    // The elements a shape basis reaches, in increasing (batch, row).
    [[nodiscard]] const std::vector<ElementRef>& of(std::size_t basis) const { return m_elements[basis]; }
    // The step h of the shape value of a basis; 0 when the basis moves nothing (every COEFF·N zero).
    [[nodiscard]] double step(std::size_t basis) const { return m_steps[basis]; }
    // The rigid elements of the snapshot whose grid points a basis moves relative to each other (rigidElementsTable).
    [[nodiscard]] std::vector<DeformedRigidElement> deformedRigidElements(std::size_t basis) const;

private:
    const DesignModel* m_model;
    const input::ModelSnapshot* m_snapshot;
    std::vector<std::vector<ElementRef>> m_elements;
    std::vector<double> m_steps;
};

} // namespace oos::design
