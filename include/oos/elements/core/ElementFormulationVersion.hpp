#pragma once

#include <cstdint>

namespace oos::elements
{

// The version of the element formulations: what the facets of every ElementFamily compute from a
// given input (stiffness, mass, differential stiffness, pressure, line and thermal loads). What is kept beyond a run
// and was computed from them carries it, so that a result of another formulation is never reused: the key of the
// condensation cache (reduction::CacheKey::elementFormulation: K*, the reduced mass, the condensed load patterns and
// the rigid-body mass properties of a unit all come from its elements' matrices and loads). O_o_Reader's revision
// stamps and the solver's version text see the model and the release, not a change of a formulation between two
// builds of one release: without this number, a cache filled before such a change keeps the old K*.
//
// Increase it in the commit that changes, for some input, a matrix or a load any element computes (a formulation, a
// node order, an integration rule, a default, a tolerance that decides a branch), and record it in the history below;
// ElementFormulationVersionTest's fingerprint of a few element matrices is the tripwire that fails when they change
// and this number does not.
//
// History:
// - 1: the first formulations (never written in a cache file, whose format 3 had no such field);
// - 2: the warped ShellQuad4 on its corrected in-plane field; a ShellTria3 with tied shortest edges takes the median
//   by the basic position of the edge's middle;
// - 3: the mass of offset shells, bars and beams at their grid points by default, under ModelParams OffsetDefinition
//   ElemOffsetsOnly (on the offsets under LoadRecoveryOffsets and ElemOffsetZ, as every formulation before);
// - 4: the thermal, pressure and BeamDistributedLoad loads of offset shells, bars and beams at their grid points under
//   ElemOffsetsOnly, the default (through the offsets under LoadRecoveryOffsets and ElemOffsetZ);
// - 5: a Beam2 on a BeamShapeProp has its axis at the shear centre of its section and its neutral axis offset from
//   it; a BeamProp's N1, N2 and M1, M2 taken (refused before);
// - 6: the ShellQuad4 and ShellTria3 of NASTRAN-95 (their former formulations kept for the ShellQuad4Drilling and
//   ShellTria3Drilling only); the ShellQuad8 and ShellTria6 (MindlinShell); the defaults of the SolidHexa of 8 and
//   20 grid points and of the SolidPenta of 6; the fasteners; the warped ShellQuad4Drilling formed flat in its mean
//   plane and joined to its grid points by rigid offsets;
// - 7: the ShellQuad4 of docs/shell-quad4-formulation.md and the ShellTria3 of docs/shell-tria3-formulation.md
//   (enhanced strains, a stabilised rotation about the normal, the rotary inertia in the coupled mass, no transverse
//   shear flexibility without a shear material);
// - 8: the ShellTria3 on the director kinematics of the ShellQuad4; the normals of the grid points shared between
//   the ShellQuad4 and ShellTria3 (PARAM ShellNormalAngle); their drilling factor from PARAM ShellDrillingStiffness
//   (1e-4 by default, 4e-4 before); the thermal loads of a dilatation as the loads of the homothety of the element;
// - 9: the ShellQuad8 and ShellTria6 of docs/shell-quad8-tria6-formulation.md: the normals of the grid points and the
//   drilling factor of the ShellQuad4 and ShellTria3 (the normals of the four cards grouped in direction), the tied
//   transverse shears and membrane strains of the ShellTria6 with the seven-point rule, the thermal loads of a
//   dilatation as the loads of the homothety; the x axis of a ShellQuad8 along its edge G1-G2;
// - 10: a Fastener or a Weld on a grid point or on a side of the shell it pierces reads the rotation of the sheet
//   from the shell elements all around the point (FastenerPiercingTable::patchGrids), not from the pierced element
//   alone; the default y-axis of a fastener along a basic axis no longer depends on round-off (FASTENER_AXIS_TIE);
//   the diagonal of the mass of a shell carried to the displacement systems is not negative;
// - 11: a Fastener, and a Weld of the forms PartPatch and ElemPatch, is tied to the shells over the square of the
//   area of its cross-section (input/tables/PiercingFootprint.hpp): in the ClassicFormulation the shell is read at
//   the four corners of the square, each on the element it falls on, in place of the grid points of the pierced
//   element; in the MixedHybridFormulation (every fastener and weld that pierces a shell) by the mean of the motion
//   of the shells over the square, not at the point alone. In the ClassicFormulation the piercing point of a
//   Fastener or a Weld takes the rotation of the shell about the axes of its plane (the rotations of the grid
//   points, by the shape functions), not the slope of the sheet between its grid points; its rotation about the
//   normal is still that of the translations around it.
inline constexpr std::uint32_t ELEMENT_FORMULATION_VERSION = 11;

} // namespace oos::elements
