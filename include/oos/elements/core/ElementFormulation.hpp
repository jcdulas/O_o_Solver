#pragma once

#include <string>
#include <string_view>

namespace oos::elements
{

// The formulations of the element types.
//
// An element card may be computed by several formulations, each an ElementFamily of its own with the same card and its
// own formulation() name. A model chooses one formulation per card, for all the elements of that card (the ShellQuad4 of
// a run all take the same one, whatever the ShellTria3 or the SolidHexa take); a card the model says nothing about takes
// its default formulation, the first one registered for it (ElementRegistry::standard).
//
// Where the choice comes from, the first one given winning:
//   1. the program that builds the snapshot (input::SnapshotBuilder::setElementFormulation, api::ModelInput, the
//      WebAssembly module's setElementFormulation);
//   2. the model parameter <card>Formulation ("Parameter;ShellQuad4Formulation;ClassicFormulation"), in the model
//      section or global case section (a choice applies to the whole model; a per-subcase choice is refused);
//   3. the default formulation of the card.
// A name the card has no formulation for is refused before anything is computed (ElementRegistry::selectionProblems,
// reported by the support check of the run).

// The formulations of the first release of O_o_Solver, the default of every card.
inline constexpr std::string_view CLASSIC_FORMULATION = "ClassicFormulation";

// The formulation parameter is the neutral card type followed by "Formulation", such as ShellQuad4Formulation.

} // namespace oos::elements
