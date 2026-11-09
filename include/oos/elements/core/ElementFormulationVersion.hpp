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
// - 2: the warped ShellQuad4 on its corrected in-plane field; a ShellTria3 with tied shortest edges takes the median by
// the
//   basic position of the edge's middle;
// - 3: the mass of offset shells, bars and beams at their grid points by default, under ModelParams OFFDEF ELMOFF (on
// the
//   offsets under LROFF and ELMZ, as every formulation before).
// - 4: the thermal, pressure and BeamDistributedLoad loads of offset shells, bars and beams at their grid points under
// ELMOFF, the
//   default (through the offsets under LROFF and ELMZ).
// - 5: a Beam2 on a BeamShapeProp has its axis at the shear centre of its section and its neutral axis offset from
// it; a
//   BeamProp's N1, N2 and M1, M2 taken (refused before);
// - 6: the ShellQuad4 and ShellTria3 of NASTRAN-95 (their former formulations kept for the ShellQuad4Drilling and
// ShellTria3Drilling only); the ShellQuad8
//   and ShellTria6 (MindlinShell); the defaults of the CHEXA8, CHEXA20 and CPENTA6; the fasteners; the warped
//   ShellQuad4Drilling formed flat in its mean plane and joined to its grid points by rigid offsets.
inline constexpr std::uint32_t ELEMENT_FORMULATION_VERSION = 6;

} // namespace oos::elements
