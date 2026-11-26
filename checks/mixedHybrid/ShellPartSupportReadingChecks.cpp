// Checks of what an edge of shells reads of the supports of its two grids (model/ShellPartSupports.cpp, specification
// section 5.6): a plane of symmetry turned from the basic system whose grids have coordinates of seven digits, a
// clamped grid beside such a plane (one grid holding by basic axes, the other by directions off them), a rotation
// both grids hold that the edge does not take, and the index of the shells of a grid.

#include "CheckRunner.hpp"
#include "ShellPartCheckSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace oos::checks
{

namespace
{

using namespace oos::checks::shellpart;

// The plane of symmetry of the checks holds the side s = 0 of a plate turned by this angle about z.
constexpr double TURN = std::numbers::pi / 6.0;

// The plate [0, 20] × [0, 10] of 2 × 1 elements in the plane z = 0, turned by TURN about z and moved away from the
// origin; `decimals`: the decimals its coordinates are written with (negative: as computed). Its side s = 0 is the edge
// of grids 0 and 1.
MappedMesh turnedPlate(int decimals)
{
    const double scale = std::pow(10.0, decimals);
    const auto written = [&](double v) { return decimals < 0 ? v : std::round(v * scale) / scale; };
    return mappedMesh(2, 1,
                      [&](double s, double t)
                      {
                          const double x = 20.0 * s;
                          const double y = 10.0 * t;
                          return mh::SpaceVector{written(100.0 + std::cos(TURN) * x - std::sin(TURN) * y),
                                                 written(200.0 + std::sin(TURN) * x + std::cos(TURN) * y), 0.0};
                      });
}

// The normal of the plane of symmetry (the side s = 0 lies in it), and the direction of that side.
mh::SpaceVector planeNormal()
{
    return {std::cos(TURN), std::sin(TURN), 0.0};
}

mh::SpaceVector sideDirection()
{
    return {-std::sin(TURN), std::cos(TURN), 0.0};
}

// The support of a grid on the plane of symmetry, as the model states it: the translation along the normal (component
// 0 of its displacement system) and the rotation about the side (component 4) off the basic axes, the rotation about
// z by its basic flag.
mh::ShellGridSupport symmetrySupport(std::size_t grid)
{
    mh::ShellGridSupport support;
    support.grid = grid;
    support.held[5] = true;
    support.skewAxes = {{planeNormal(), 0}, {sideDirection(), 4}};
    return support;
}

// The DOFs of an edge that the derived supports hold: displacement moments, shear, rotation.
struct HeldOfEdge
{
    std::size_t moments = 0;
    std::size_t shears = 0;
    std::size_t rotations = 0;
};

HeldOfEdge heldOfEdge(const mh::ShellPart& part, const mh::ShellSupports& supports, std::size_t edge)
{
    HeldOfEdge out;
    const std::size_t first = part.entities()[part.edgeEntity(edge)].firstUnknown;
    for (const mh::ShellDerivedSupport& d : supports.held)
    {
        if (d.unknown < first || d.unknown >= first + part.edgeDofCount(edge))
            continue;
        const std::size_t dof = d.unknown - first;
        if (dof < part.edgeShearDof(edge, 0))
            ++out.moments;
        else if (dof < part.edgeRotationDof(edge, 0))
            ++out.shears;
        else
            ++out.rotations;
    }
    return out;
}

mh::ShellPart plateOf(const MappedMesh& mesh)
{
    mh::ShellPartInput input = shellInput(mesh, isotropicSection(7e10, 0.3, 0.1), false);
    input.checkElements = false;
    return mh::ShellPart(input);
}

} // namespace

CHECK_CASE("shellpart supports: a turned plane of symmetry with coordinates of seven digits holds its edge")
{
    // coordinates with four decimals at 100 to 220: seven digits, the side s = 0 some 1e-6 rad off the plane
    const mh::ShellPart part = plateOf(turnedPlate(4));
    runner.expect(part.usable(), "the plate is read");
    const std::size_t edge = part.topology().indexOf(0, 1);
    const mh::ShellSupports supports = part.supportsOf(std::vector<mh::ShellGridSupport>{symmetrySupport(0),
                                                                                           symmetrySupport(1)});
    const HeldOfEdge held = heldOfEdge(part, supports, edge);
    // the rotation about the side is held along the edge: the tangent is in the plane to the precision of the
    // coordinates (it was left free, a hinge along the plane, when the test was at the round-off)
    runner.expect(held.rotations == part.edgeDegree(edge), "the rotation of the edge is held");
    // the translation along the normal is a relation on each displacement moment, no support of one component
    runner.expect(held.moments == 0 && supports.rows.size() == part.edgeDegree(edge) - 1,
                  "the translation along the normal is held by relations");
    // the conormal is the normal of the plane: the rotation about it is free, and so is the shear
    runner.expect(held.shears == 0, "the shear of the edge is free");
    runner.expect(supports.uncarriedEdges == 0 && supports.notes.empty(), "nothing is left out");
}

CHECK_CASE("shellpart supports: a clamped grid beside a turned plane of symmetry shares its directions")
{
    const mh::ShellPart part = plateOf(turnedPlate(-1));
    const std::size_t edge = part.topology().indexOf(0, 1);
    // grid 0 is clamped by its basic flags, grid 1 is on the plane of symmetry: they share the translation along the
    // normal of the plane and the rotations in the plane
    std::vector<mh::ShellGridSupport> grids = gridSupports({0}, {0, 1, 2, 3, 4, 5});
    grids.push_back(symmetrySupport(1));
    const mh::ShellSupports supports = part.supportsOf(grids);
    const HeldOfEdge held = heldOfEdge(part, supports, edge);
    runner.expect(held.rotations == part.edgeDegree(edge), "the rotation of the edge is held");
    runner.expect(supports.rows.size() == part.edgeDegree(edge) - 1, "one relation per displacement moment");
    bool carried = !supports.rows.empty();
    for (const mh::ShellSupportRow& row : supports.rows)
    {
        // the grid of the directions by its component 0, the clamped grid by its basic components x and y
        carried = carried && row.grids.size() == 1 && row.grids.front().grid == 1 && row.grids.front().component == 0;
        carried = carried && row.basicGrids.size() == 2 && row.basicGrids.front().grid == 0;
        // Σ_c d_c q_c along the normal of the plane
        carried = carried && row.unknowns.size() == 2;
    }
    runner.expect(carried, "each relation names both grids");
    runner.expect(supports.uncarriedEdges == 0, "the two grids hold no direction that differs");
    // two grids that state the same supports the same way are read as before: the flags alone
    const mh::ShellSupports clamped = part.supportsOf(gridSupports({0, 1}, {0, 1, 2, 3, 4, 5}));
    const HeldOfEdge all = heldOfEdge(part, clamped, edge);
    runner.expect(all.moments == 3 * (part.edgeDegree(edge) - 1) && all.rotations == part.edgeDegree(edge) &&
                      all.shears == part.edgeDegree(edge) && clamped.rows.empty(),
                  "a clamped edge holds every DOF, without relations");
}

CHECK_CASE("shellpart supports: a rotation the edge does not take is noted, the rotation about the normal is not")
{
    const mh::ShellPart part = plateOf(turnedPlate(-1));
    const std::size_t edge = part.topology().indexOf(0, 1);
    // both grids hold the rotation about the basic axis x, oblique to the side (30 degrees from its conormal): it is
    // neither about the edge nor about its conormal
    const mh::ShellSupports oblique = part.supportsOf(gridSupports({0, 1}, {3}));
    const HeldOfEdge none = heldOfEdge(part, oblique, edge);
    runner.expect(none.rotations == 0 && none.shears == 0, "the edge takes nothing of an oblique rotation");
    runner.expect(oblique.notes.size() == 1 && oblique.notes.front().find("oblique") != std::string::npos,
                  "one note says so");
    // the rotation about the normal of the plate turns its membrane field: the edge has nothing to take
    const mh::ShellSupports drilling = part.supportsOf(gridSupports({0, 1}, {5}));
    runner.expect(drilling.notes.empty(), "no note for the rotation about the normal");
}

CHECK_CASE("shellpart supports: the shells of a grid, with their corner there")
{
    const MappedMesh mesh = mappedMesh(2, 2, [](double s, double t) { return mh::SpaceVector{s, t, 0.0}; });
    const mh::ShellPart part = plateOf(mesh);
    bool same = true;
    for (std::size_t grid = 0; grid < mesh.grids.size(); ++grid)
    {
        // by a scan of every shell, in their order
        std::vector<std::pair<std::size_t, std::size_t>> expected;
        for (std::size_t shell = 0; shell < mesh.quads.size(); ++shell)
        {
            const auto found = std::find(mesh.quads[shell].begin(), mesh.quads[shell].end(), grid);
            if (found != mesh.quads[shell].end())
                expected.emplace_back(shell, static_cast<std::size_t>(found - mesh.quads[shell].begin()));
        }
        same = same && part.shellsAt(grid) == expected;
    }
    runner.expect(same, "shellsAt gives the shells of each grid and their corners");
    runner.expect(part.shellsAt(4).size() == 4 && part.shellsAt(0).size() == 1 && part.shellsAt(99).empty(),
                  "four shells at the centre, one at a corner, none at a grid that is not one");
}

} // namespace oos::checks
