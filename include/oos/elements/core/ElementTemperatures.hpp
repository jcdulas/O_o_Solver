#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

namespace oos::elements
{

// The temperature an element-wise card gives to one element (TEMPP1: the mean temperature TBAR of a shell and its
// gradient TPRIME through the thickness; LineElemTemperatureLoad: the temperature at end A of a bar and the change to
// end B).
struct ElementTemperature
{
    std::int32_t elementId = 0;
    double mean = 0.0;
    double gradient = 0.0;
};

// The temperatures of one temperature set as the element facets read them: one value per grid point row (TEMP, the
// DefaultTemperatureLoad default in place of a grid point the TEMP cards leave out; NaN for a grid point with neither)
// and the element-wise temperatures (TEMPP1, LineElemTemperatureLoad), sorted by element ID.
struct TemperatureSet
{
    std::span<const double> atGrid; // empty: the set gives no temperature
    std::span<const ElementTemperature> ofElements;

    [[nodiscard]] bool isEmpty() const noexcept { return atGrid.empty() && ofElements.empty(); }
    // The temperature of a grid point row, or nothing.
    [[nodiscard]] std::optional<double> atGridRow(std::int32_t row) const
    {
        if (atGrid.empty() || !std::isfinite(atGrid[static_cast<std::size_t>(row)]))
            return std::nullopt;
        return atGrid[static_cast<std::size_t>(row)];
    }
    // The element-wise temperature of an element, or nothing.
    [[nodiscard]] const ElementTemperature* ofElement(std::int32_t elementId) const
    {
        const auto found =
            std::lower_bound(ofElements.begin(), ofElements.end(), elementId,
                             [](const ElementTemperature& t, std::int32_t id) { return t.elementId < id; });
        return found != ofElements.end() && found->elementId == elementId ? &*found : nullptr;
    }
};

// An enforced axial deformation of one element (DEFORM: D, the change of length of a Rod2, Rod2InlineSection,
// Tube2, Bar2 or Beam2, positive an elongation).
struct ElementDeformation
{
    std::int32_t elementId = 0;
    double elongation = 0.0;
};

// The element types DEFORM applies to (the one-dimensional elements), sorted.
inline constexpr std::array<std::string_view, 5> DEFORMABLE_LINE_CARDS{
    "Bar2", "Beam2", "Rod2", "Rod2InlineSection", "Tube2"};
static_assert(std::is_sorted(DEFORMABLE_LINE_CARDS.begin(), DEFORMABLE_LINE_CARDS.end()));
[[nodiscard]] inline bool takesElementDeformations(std::string_view card) noexcept
{
    return std::binary_search(DEFORMABLE_LINE_CARDS.begin(), DEFORMABLE_LINE_CARDS.end(), card);
}

// The initial strains of a static subcase: its thermal load and its element deformations, which load the structure
// the same way (an imposed strain held by the stiffness; the total load is the sum of the external, thermal, element
// deformation and enforced displacement loads).
// - The temperatures (linear statics): the thermal strain is α (T − T0), T the
//   temperatures of the TEMPERATURE(LOAD) set and T0 those of the TEMPERATURE(INITIAL) set when the Case Control gives
//   one, else the reference temperature TREF of each element's material (`initial` empty). `hasTemperatureLoad` false:
//   the subcase has no TEMPERATURE(LOAD) (a DEFORM alone), no element takes a thermal strain.
// - The element deformations of the DEFORM set (Case Control DEFORM, Bulk Data DEFORM), sorted by element ID, one
//   per element: the free axial strain D/L of a rod, bar or beam, added to its thermal strain α ΔT (the equivalent
//   loads are those of the free elongation, the recovery subtracts it: the axial force is EA (ε − α ΔT − D/L)).
//
// Thread-safety: a view (spans); immutable. Ownership: borrows the buffers of its builder (assembly::TemperatureField).
struct ElementTemperatures
{
    TemperatureSet load;
    TemperatureSet initial;
    bool hasTemperatureLoad = true;
    std::span<const ElementDeformation> deformations{};
    std::int32_t deformationSet = 0; // the DEFORM set ID, for the messages (0: none)

    // The enforced elongation D of an element, 0 when the DEFORM set gives it none.
    [[nodiscard]] double elongationOf(std::int32_t elementId) const
    {
        const auto found =
            std::lower_bound(deformations.begin(), deformations.end(), elementId,
                             [](const ElementDeformation& d, std::int32_t id) { return d.elementId < id; });
        return found != deformations.end() && found->elementId == elementId ? found->elongation : 0.0;
    }
};

} // namespace oos::elements
