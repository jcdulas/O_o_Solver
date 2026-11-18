#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace oos::elements
{

// What the ShellQuad4, ShellTria3, ShellQuad8 and ShellTria6 of a model share: the normal of each grid point, and the
// drilling factor.
//
// The normals. Each element has a normal at each of its grid points: that of its bilinear surface there for a
// ShellQuad4, its own for a ShellTria3, that of its quadratic surface there for a ShellQuad8 or a ShellTria6 (the
// normal of the plane of its corners when the element is flat). On a curved surface the elements that meet at a grid
// point have normals that differ by the angle of their facets (by the interpolation error of their surfaces for the
// quadratic elements), and the rotation about the normal of one is a bending rotation for the others: they hold it
// by their bending, the less the finer the mesh against the thickness, and the surface softens with the refinement.
// So, as the directors of a shell are defined at the nodes of its mesh (T. J. R. Hughes, W. K. Liu, Nonlinear finite
// element analysis of shells: part I, Comput. Methods Appl. Mech. Eng. 26 (1981); K.-J. Bathe, Finite Element
// Procedures (1996), section 5.4), at each grid point, over every element of these four cards in the snapshot:
// - the elements are grouped into smooth groups: two elements whose normals there make, in direction, an angle of at
//   most PARAM ShellNormalAngle (20 degrees by default) are in one group, and the groups are the connected sets of
//   that relation; a fold, a T-junction or a stiffener meeting the skin keeps one normal per smooth side. The
//   orientation of the elements plays no part: two neighbours numbered in opposite senses are in one group;
// - a group of one element, a group whose normals agree to round-off (a flat mesh: it gives the bits it gave before)
//   or a group one of whose normals is farther than that angle from their mean keeps each element's own normal;
// - any other group shares the normalised sum of its elements' unit normals (each turned to the side of the first),
//   with equal weights; each element takes it on the side of its own normal.
// The rotation of a grid point about its shared normal then moves nothing in any of the elements of the group: it is
// held by their drilling term alone. A negative or zero ShellNormalAngle shares no normal.
//
// A ShellNormal card gives a grid point its normal (in the basic system, or in a coordinate system taken at the grid
// point): the elements whose own normal there is within the angle of it, on either side, take it in place of the
// normal of their group (within 20 degrees when ShellNormalAngle is not positive); a zero vector leaves every element
// of the grid point its own normal. The other shell elements do not read these cards.
//
// The drilling factor c_d of the elements (docs/shell-quad4-formulation.md, section 8): 1e-6 times ModelParams
// ShellDrillingStiffness, 100 by default; zero or negative: no drilling term (the automatic constraint of singular
// degrees of freedom then removes the rotation about the normal where nothing else holds it).
//
// Thread-safety: immutable once built. Built once per snapshot index (shellDirectors).
inline constexpr double DEFAULT_SHELL_NORMAL_ANGLE_DEGREES = 20.0;
inline constexpr double DEFAULT_SHELL_DRILLING_STIFFNESS = 100.0;
inline constexpr double SHELL_DRILLING_STIFFNESS_SCALE = 1.0e-6;
// Normals whose cross product is below this sine are the same normal.
inline constexpr double SAME_SHELL_NORMAL_SINE = 1.0e-12;

// The cards of the elements that take their normals and their drilling factor here, and their number of grid points.
struct DirectorShellCard
{
    std::string_view card;
    std::size_t nodes = 0;
};
inline constexpr std::array<DirectorShellCard, 4> DIRECTOR_SHELL_CARDS{
    DirectorShellCard{"ShellQuad4", 4}, DirectorShellCard{"ShellTria3", 3}, DirectorShellCard{"ShellQuad8", 8},
    DirectorShellCard{"ShellTria6", 6}};

// Whether `card` is one of DIRECTOR_SHELL_CARDS.
[[nodiscard]] constexpr bool isDirectorShellCard(std::string_view card) noexcept
{
    for (const DirectorShellCard& shell : DIRECTOR_SHELL_CARDS)
        if (shell.card == card)
            return true;
    return false;
}

class ShellDirectors
{
public:
    [[nodiscard]] static ShellDirectors build(const SnapshotIndex& index);

    // The normal (basic, on the side of the element's own) that grid point `node` (its place in the card) of the
    // element of batch row `row` of `card` (one of DIRECTOR_SHELL_CARDS) shares with its smooth group, or nothing:
    // the element keeps its own.
    [[nodiscard]] std::optional<base::Vec3> sharedNormal(std::string_view card, std::int64_t row,
                                                         std::size_t node) const;
    // c_d.
    [[nodiscard]] double drillingFactor() const noexcept { return m_drillingFactor; }
    // Whether the elements of these cards at grid point row `gridRow` are on one smooth side (their normals there in
    // one smooth group, whatever ShellNormalAngle; false for a grid point of none of them): the rotation of the grid
    // point about their normal is then held by nothing else in these elements than their drilling term.
    [[nodiscard]] bool hasOneSmoothSide(std::int32_t gridRow) const noexcept
    {
        const auto row = static_cast<std::size_t>(gridRow);
        return row < m_hasOneSmoothSide.size() && m_hasOneSmoothSide[row] != 0;
    }

private:
    // Per card of DIRECTOR_SHELL_CARDS: row · nodes + node; the zero vector: not shared.
    std::array<std::vector<base::Vec3>, DIRECTOR_SHELL_CARDS.size()> m_normals;
    std::vector<std::uint8_t> m_hasOneSmoothSide; // by grid point row
    double m_drillingFactor = DEFAULT_SHELL_DRILLING_STIFFNESS * SHELL_DRILLING_STIFFNESS_SCALE;
};

// The directors of the snapshot of `index`, built on first use and kept by the index (SnapshotIndex::computedOnce).
[[nodiscard]] const ShellDirectors& shellDirectors(const SnapshotIndex& index);

} // namespace oos::elements
