#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::solutions
{

// The stabilisation terms of the elements against the strain energy of a solution.
//
// Some elements carry a term that is not a stiffness of the structure: the drilling term of the ShellQuad4,
// ShellTria3, ShellQuad8 and ShellTria6 (elements/shells/ShellDirectors.hpp, PARAM ShellDrillingStiffness) only keeps
// the rotation of a grid point about the normal of its shells from being singular. It holds next to nothing of the
// strain energy of a sound model, whose results do not depend on its factor. Where a moment about the normal of a
// shell reaches a single grid point (a bar or a beam attached to the shell at one grid point, a rigid element with
// its rotations on one grid point, a moment load, a support of that rotation), the stabilisation alone carries it:
// the result there depends on the factor and on the size of the elements. A solution is told so when the
// stabilisation holds more than STABILISATION_ENERGY_SHARE of its strain energy.

// The share of the strain energy above which the stabilisation is pointed out.
inline constexpr double STABILISATION_ENERGY_SHARE = 0.01;
// The elements named in the warning: those with the largest energies.
inline constexpr std::size_t STABILISATION_NAMED_ELEMENTS = 5;

// The energy the stabilisation terms hold under one displacement vector.
struct StabilisationEnergy
{
    struct Element
    {
        std::string card;
        std::int32_t id = 0;
        double energy = 0.0;
    };

    double total = 0.0;
    std::vector<Element> largest; // at most STABILISATION_NAMED_ELEMENTS, by decreasing energy
};

// The stabilisation energy of the elements of `set` (ElementFamily::computeStabilisationEnergies) under the g-set
// vector `displacements` (6 values per grid point, then the scalar points: a model without entities).
[[nodiscard]] StabilisationEnergy stabilisationEnergyOf(const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                                                        std::span<const double> displacements);

// Reports the warning `stabilisation-carries-load` when `energy` is more than STABILISATION_ENERGY_SHARE of
// `strainEnergy` (½ uᵀ K u of the same vector; nothing when it is not positive). `where` names the solution in the
// message ("subcase 3", "mode 2 of subcase 1").
void reportStabilisationEnergy(const StabilisationEnergy& energy, double strainEnergy, std::string_view where,
                               base::DiagnosticSink& sink);

} // namespace oos::solutions
