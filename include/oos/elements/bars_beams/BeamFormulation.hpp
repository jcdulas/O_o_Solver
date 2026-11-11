#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/elements/bars_beams/BeamGeometry.hpp"
#include "oos/elements/bars_beams/BeamLineLoad.hpp"
#include "oos/elements/bars_beams/BeamMass.hpp"
#include "oos/elements/bars_beams/BeamSection.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::elements
{

// The formulation shared by Bar2 and Beam2 (a Bar2 is a Beam2 of uniform section without warping): the two
// cards differ in where the section comes from (PBAR; BeamProp with its stations) and in their output layouts, so each
// family's facet files gather a BeamData and call this. The pieces:
//
// - BeamGeometry: end points, element system, offsets, pin flags, the map T from grid DOFs to end DOFs;
// - BeamSection: the section along the length;
// - BeamFlexibility: the end stiffness and the fixed-end loads (flexibility method, shear flexibility);
// - PinRelease: the condensation of the pin flags;
// - BeamMass: lumped and coupled mass;
// - BeamLineLoad: BeamDistributedLoad in the element system;
// - BeamResponse: the end forces and the stress resultants along the element from the displacements.
//
// Grid-level matrices are Tᵀ·K·T, Tᵀ·M·T and Tᵀ·f over the DOFs [GA 1-6, GB 1-6] in the displacement systems.

// Everything a bar or beam facet needs about one element.
struct BeamData
{
    BeamGeometry geometry;
    BeamSection section;
    IsotropicMaterial material;
};

// The problem of a bar or beam that its geometry and section allow but its stiffness does not: pin flags on DOFs
// without stiffness or leaving a mechanism. Checked by the families' resolve functions.
[[nodiscard]] bool hasStablePinFlags(const BeamData& beam);
// The diagnostic of pin flags that hasStablePinFlags() refuses, on the field PA (or PB when end A has no flag).
[[nodiscard]] base::Diagnostic unstablePinFlags(const ElementBatchView& view, std::int64_t row,
                                                const BeamGeometry& geometry);

// The stiffness over the grid DOFs (pin flags condensed, offsets and CD systems applied).
[[nodiscard]] base::MatN<12> beamStiffness(const BeamData& beam);
// The mass over the grid DOFs; `addedMassPerLength` is the NSM cards' share (MassOptions); a Beam2 carries the
// torsional inertia of its section, a Bar2 does not (BeamMass.hpp). The mass of the element between its end points
// (the offset ends), carried to the grid points through the offsets when `isMassOnOffsets` (MDLPRM OFFDEF LROFF,
// ELMZ), by the rotations alone otherwise (ELMOFF, the default: BeamGeometry::rotationTransform). A Beam2 whose
// neutral axis or NSM is off its axis has its structural mass on the neutral axis and its NSM (the NSM cards' share
// with it) at the NSM's centre of gravity, rigidly attached to the axis at each end (under every OFFDEF: they
// are offsets within the section, not WA, WB).
[[nodiscard]] base::MatN<12> beamGridMass(const BeamData& beam, double addedMassPerLength, MassFormulation formulation,
                                          TorsionalInertia torsionalInertia, bool isMassOnOffsets);
// The equivalent nodal loads of the line loads of one element over the grid DOFs (pin flags condensed).
// Through the offsets when `areLoadsOnOffsets` (MDLPRM OFFDEF LROFF, ELMZ); at the grid points by the rotations alone
// otherwise (ELMOFF; a design choice for BeamDistributedLoad, a distributed load: treated as the pressure loads,
// without the offsets, consistent with the default of ModelParams OFFDISLD that applies the distributed loads at the
// grid points).
[[nodiscard]] std::array<double, 12> beamGridLoads(const BeamData& beam, std::span<const BeamLineLoad> loads,
                                                   bool areLoadsOnOffsets);

// The batch loops of the two families: `resolve(row)` returns the BeamData of a row (or throws for an element check()
// refuses).
template <typename Resolve>
void appendBeamStiffness(ElementRange range, ElementMatrixBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const BeamData beam = resolve(row);
        const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
        const base::MatN<12> stiffness = beamStiffness(beam);
        block.append(row, dofs, stiffness.values);
    }
}

template <typename Resolve>
void appendBeamMass(ElementRange range, const MassOptions& options, TorsionalInertia torsionalInertia,
                    bool isMassOnOffsets, ElementMatrixBlock& block, Resolve&& resolve)
{
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const BeamData beam = resolve(row);
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
        const base::MatN<12> mass = beamGridMass(beam, added, options.formulation, torsionalInertia, isMassOnOffsets);
        block.append(row, dofs, mass.values);
    }
}

// Reports the problem of every line load the element cannot take (its element refused, or a field value the input
// format does not allow).
template <typename Resolve>
void checkBeamLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                        base::DiagnosticSink& sink, Resolve&& resolve)
{
    for (const ElementLineLoad& load : loads)
    {
        auto beam = resolve(load.elementRow);
        if (!beam)
            continue; // check() reports the element itself
        auto resolved = resolveLineLoad(load, beam.value->geometry, view.card(), view.elementId(load.elementRow));
        if (auto* problem = std::get_if<base::Diagnostic>(&resolved))
            sink.report(std::move(*problem));
    }
}

// The resolved loads of one element, or std::invalid_argument for a load checkBeamLineLoads() refuses.
[[nodiscard]] std::vector<BeamLineLoad> resolveLineLoadsOrThrow(const ElementBatchView& view,
                                                                std::span<const ElementLineLoad> loads,
                                                                  const BeamGeometry& geometry,
                                                                  std::int64_t row = -1);

template <typename Resolve>
void appendBeamLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                         ElementVectorBlock& block, Resolve&& resolve)
{
    for (const ElementLineLoad& load : loads)
    {
        const BeamData beam = resolve(load.elementRow);
        const std::vector<BeamLineLoad> resolved =
            resolveLineLoadsOrThrow(view, std::span<const ElementLineLoad>(&load, 1), beam.geometry);
        const std::array<double, 12> values = beamGridLoads(beam, resolved, view.areLoadsOnOffsets());
        const std::array<ElementDof, 12> dofs = beam.geometry.dofs();
        block.append(load.elementRow, dofs, values);
    }
}

// Calls visit(elementRow, loadsOfThatElement) for each run of loads of one element (the loads of an element are
// contiguous: precondition of ElementFamily::recoverUnderLineLoads).
template <typename Visit>
void forEachLoadedElement(std::span<const ElementLineLoad> loads, Visit&& visit)
{
    std::size_t first = 0;
    while (first < loads.size())
    {
        std::size_t last = first + 1;
        while (last < loads.size() && loads[last].elementRow == loads[first].elementRow)
            ++last;
        visit(loads[first].elementRow, loads.subspan(first, last - first));
        first = last;
    }
}

} // namespace oos::elements
