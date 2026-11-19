#include "oos/solutions/common/StabilisationEnergy.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementFamily.hpp"

#include <algorithm>
#include <cstdio>

namespace oos::solutions
{

namespace
{

// The elements of a batch are taken by ranges of this many rows (the size of the buffer of their energies).
constexpr std::int64_t RANGE_ELEMENTS = 4096;

// `share` as a percentage: one decimal below 10 %, none above.
[[nodiscard]] std::string percentOf(double share)
{
    char text[32];
    std::snprintf(text, sizeof text, share < 0.1 ? "%.1f" : "%.0f", 100.0 * share);
    return text;
}

} // namespace

StabilisationEnergy stabilisationEnergyOf(const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                                          std::span<const double> displacements)
{
    StabilisationEnergy result;
    const elements::NodalDisplacements nodal =
        elements::NodalDisplacements::ofGSet(displacements, static_cast<std::size_t>(dofMap.gridCount()),
                                             static_cast<std::size_t>(dofMap.scalarCount()), nullptr);
    std::vector<double> energies(static_cast<std::size_t>(RANGE_ELEMENTS)); // one range of elements
    for (const assembly::AssemblyBatch& batch : set.batches())
    {
        const std::int64_t rows = batch.view.rowCount();
        for (std::int64_t first = 0; first < rows; first += RANGE_ELEMENTS)
        {
            const std::int64_t last = std::min(rows, first + RANGE_ELEMENTS);
            const std::span<double> ofRange(energies.data(), static_cast<std::size_t>(last - first));
            if (!batch.family->computeStabilisationEnergies(batch.view, {first, last}, nodal, ofRange))
                break; // the family has no stabilisation term
            for (std::int64_t row = first; row < last; ++row)
            {
                const double energy = ofRange[static_cast<std::size_t>(row - first)];
                result.total += energy;
                // Kept among the largest: inserted in order, the smallest dropped.
                if (result.largest.size() == STABILISATION_NAMED_ELEMENTS && energy <= result.largest.back().energy)
                    continue;
                StabilisationEnergy::Element element{std::string(batch.view.card()), batch.view.elementId(row), energy};
                const auto place =
                    std::find_if(result.largest.begin(), result.largest.end(),
                                 [&](const StabilisationEnergy::Element& other) { return other.energy < energy; });
                result.largest.insert(place, std::move(element));
                if (result.largest.size() > STABILISATION_NAMED_ELEMENTS)
                    result.largest.pop_back();
            }
        }
    }
    return result;
}

void reportStabilisationEnergy(const StabilisationEnergy& energy, double strainEnergy, std::string_view where,
                               base::DiagnosticSink& sink)
{
    if (!(strainEnergy > 0.0) || !(energy.total > STABILISATION_ENERGY_SHARE * strainEnergy))
        return;
    std::string named;
    for (const StabilisationEnergy::Element& element : energy.largest)
    {
        if (!(element.energy > 0.0))
            break;
        named += (named.empty() ? "" : ", ") + element.card + " " + std::to_string(element.id);
    }
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Warning;
    diagnostic.code = "stabilisation-carries-load";
    diagnostic.card = "Parameter ShellDrillingStiffness";
    diagnostic.field = "ShellDrillingStiffness";
    diagnostic.message = base::concatenate(
        {std::string(where), ": the drilling stabilisation of the shells holds ",
         percentOf(energy.total / strainEnergy), " % of the strain energy (the most in ", named,
         "): a moment about the normal of a shell reaches a single grid point there (a bar or a beam attached to the "
         "shell at one grid point, a rigid element with its rotations on one grid point, a moment load, a support of "
         "that rotation), where the shells have only that stabilisation to hold it; the result depends there on "
         "ShellDrillingStiffness and on the size of the elements. Spread the moment over the neighbouring grid points "
         "(a rigid or an interpolation element on several grid points, a line element continued along an edge of "
         "the shell)"});
    sink.report(std::move(diagnostic));
}

} // namespace oos::solutions
