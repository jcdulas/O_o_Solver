#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oos::output
{

// The components of a grid point as a mask: bit c − 1 for component c (1 to 3 the translations, 4 to 6 the rotations).
using ComponentMask = std::uint8_t;
inline constexpr ComponentMask TRANSLATIONS = 0x07;
inline constexpr ComponentMask ROTATIONS = 0x38;
inline constexpr ComponentMask ALL_COMPONENTS = 0x3F;

// The names of the components of contour data, in the system of the zone: t1 t2 t3 r1 r2 r3.
inline constexpr std::array<std::string_view, 6> ZONE_COMPONENTS{"t1", "t2", "t3", "r1", "r2", "r3"};

// The mask of a text of component digits ("123456", "12"); throws std::invalid_argument for another character.
[[nodiscard]] ComponentMask componentMaskOf(std::string_view digits);
// The digits of a mask, increasing ("" for none).
[[nodiscard]] std::string componentDigitsOf(ComponentMask mask);
[[nodiscard]] constexpr bool hasComponent(ComponentMask mask, int component) noexcept
{
    return component >= 1 && component <= 6 && (mask & (1U << (component - 1))) != 0;
}

// One thing outside a zone that acts on a contour point (the contour report gives, per contour point, its components
// and their causes). The causes O_o_Reader gives (zoneBoundaries: `kind` "element", "rigid", "mpc" or "connection",
// `card` and `id`, an MPC equation by its set ID), and those O_o_Solver adds:
// "enclosed" (an enclosed element kept outside, enclosed="warn" or "exclude") and "constraint" (a grid point
// constrained in some subcases only: `card` "SinglePointConstr", `id` the SPC set of a subcase that constrains it).
// `components` is what the cause itself uses there when O_o_Solver knows it (0: O_o_Reader gives the union of a point's
// causes only).
struct ContourCause
{
    std::string kind;
    std::string card;
    std::int64_t id = 0;
    ComponentMask components = 0;
};

// One boundary point of a zone as O_o_Reader gives it (zoneBoundaries): its grid point, the components through which
// the outside acts on it, whether it is a grid point of the element a CFAST or CWELD pierces when its other side is
// outside, and the outside things that act on it.
struct ZoneBoundaryPoint
{
    std::int32_t grid = 0;
    ComponentMask components = 0;
    bool isPiercedPoint = false;
    std::vector<ContourCause> causes;
};

// An element not in the list of a zone whose grid points are all grid points of the zone (O_o_Reader
// enclosedElements): a stringer forgotten in the list, a rigid element or a connector inside, a mass on one point.
struct EnclosedElement
{
    std::int32_t elementId = 0;
    std::string card;
    bool isPointElement = false;
};

// What a zone does with its enclosed elements that are not rigid elements and not on one grid point: "warn" (the
// default) keeps such an element outside and lists it, "include" adds it to the zone, "exclude" keeps it outside
// without a warning. A rigid element whose grid points are all in the zone belongs to it, and a 0D element on one grid
// point of the zone is added to it, whatever the policy (the report says so).
enum class EnclosedPolicy : std::uint8_t
{
    Warn,
    Include,
    Exclude
};

// The `frameId` of a zone whose contour data are in the displacement system of each grid point (CD).
inline constexpr std::int32_t DISPLACEMENT_SYSTEMS = -1;

// The subcases whose constraints decide the contour of a zone (a grid point constrained in some of them only is a
// contour point): the label and the SPC set it selects (nothing: none), and its load set (the SPCD of an enforced
// displacement).
struct ZoneSubcaseConstraint
{
    std::int32_t subcase = 0;
    std::optional<std::int32_t> spcSet;
    std::optional<std::int32_t> loadSet;
};

// A zone of an external local model: the set of elements a local model represents, as the caller gives it, with what
// O_o_Reader says of it as a group of elements (zoneBoundaries, enclosedElements): its boundary points with their
// components and causes and its enclosed elements. O_o_Reader's group holds its enclosed elements; ZoneContour applies
// the enclosed policy, adds the grid points constrained in some subcases only, and resolves the rest against the
// snapshot.
//
// `frameId`: the system of the contour data: 0 the basic system, a rectangular CORDij, or DISPLACEMENT_SYSTEMS.
struct ZoneDefinition
{
    std::int32_t id = 0;
    std::string name;
    std::int32_t group = 0;             // a tag of the caller carried to the labels (a DesignProblem: the zone's ID)
    std::vector<std::int32_t> elements; // the members the caller gave: element and rigid element IDs
    std::vector<ZoneBoundaryPoint> boundary; // O_o_Reader zoneBoundaries
    std::vector<EnclosedElement> enclosed;   // O_o_Reader enclosedElements
    EnclosedPolicy enclosedPolicy = EnclosedPolicy::Warn;
    std::int32_t frameId = 0;
    // The unit a zone was made from, 0 for a zone of elements: the report names it.
    std::int64_t unit = 0;
};

} // namespace oos::output
