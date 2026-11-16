#pragma once

#include <string>
#include <string_view>

namespace oos::elements
{

// The formulations of the element types.
//
// An element card may be computed by several formulations, each an ElementFamily of its own with the same card and its
// own formulation() name. A model chooses one formulation per card, for all the elements of that card (the ShellQuad4
// of a run all take the same one, whatever the ShellTria3 or the SolidHexa take); a card the model says nothing about
// takes its default formulation, the first one registered for it (ElementRegistry::standard).
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

// The mixed hybrid formulation of the shells, line elements and solids (docs/mixed-hybrid-formulation-spec.md): the
// shell of Neunteufel and Schöberl, the line elements on its edges and alone, the hybrid stress solids. Its elements
// share unknowns on edges and faces, so it is chosen for the whole model, never for one card alone:
//   1. the program that builds the snapshot (input::SnapshotBuilder::setModelFormulation, api::ModelInput);
//   2. the model parameter ElementFormulation ("Parameter;ElementFormulation;MixedHybridFormulation"), in the model
//      section or global case section;
//   3. ClassicFormulation.
// Under it every card of MIXED_HYBRID_CARDS takes the formulation; a per-card choice of it is refused
// (formulation-per-card), and so is a per-card choice of another formulation for one of those cards
// (formulation-mix). The cards of MIXED_HYBRID_REFUSED_CARDS have no counterpart and are refused.
inline constexpr std::string_view MIXED_HYBRID_FORMULATION = "MixedHybridFormulation";
inline constexpr std::string_view MODEL_FORMULATION_PARAMETER = "ElementFormulation";

inline constexpr std::string_view MIXED_HYBRID_CARDS[] = {"ShellQuad4",
                                                          "ShellTria3",
                                                          "ShellQuad4Drilling",
                                                          "ShellTria3Drilling",
                                                          "ShellQuad8",
                                                          "ShellTria6",
                                                          "Rod2",
                                                          "Rod2InlineSection",
                                                          "Tube2",
                                                          "Bar2",
                                                          "Beam2",
                                                          "SolidTetra",
                                                          "SolidPyramid",
                                                          "SolidPenta",
                                                          "SolidHexa"};
inline constexpr std::string_view MIXED_HYBRID_REFUSED_CARDS[] = {"ShearPanel4"};

// True when `card` is one of MIXED_HYBRID_CARDS.
[[nodiscard]] constexpr bool isMixedHybridCard(std::string_view card) noexcept
{
    for (const std::string_view listed : MIXED_HYBRID_CARDS)
    {
        if (listed == card)
            return true;
    }
    return false;
}

} // namespace oos::elements
