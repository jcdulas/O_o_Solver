// Checks of what a glued contact reads of the solids of the mixed hybrid formulation (SolidPart::skinTraces): the
// functions of the edges and faces of the skin as projections of a displacement field known at their points.

#include "CheckRunner.hpp"
#include "mixedHybrid/SolidPartCheckSupport.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace
{

using namespace oos::checks::solidpart;

// The unknown of a tie in the global vector of the checks (three translations per grid, then the entity DOFs), its
// DOF moved by `component`.
std::size_t globalOf(const mh::SolidPart& part, const mh::TieUnknown& unknown, std::size_t component)
{
    return unknown.kind == mh::TieUnknown::Kind::Grid
               ? 3 * unknown.index + unknown.dof + component
               : 3 * part.gridPositions().size() + part.dofOffset(unknown.index) + unknown.dof + component;
}

} // namespace

CHECK_CASE("solidpart: the functions of the skin are the projections of the motion of its points")
{
    // u = b + A x with strains and a rotation: every function of the skin has the value SolidPart::affineField
    // gives it, on straight and on curved edges and faces (the skin of a sector and of a dome read from the mesh,
    // the parabolas of mid-side grids).
    const std::array<std::array<double, 3>, 3> a{{{1e-3, 2e-4, 3e-4}, {-3e-4, -5e-4, 1e-4}, {2e-4, -4e-4, 6e-4}}};
    const mh::SolidPoint b{1e-4, -2e-4, 3e-4};
    struct Case
    {
        std::string name;
        Mesh mesh;
    };
    const std::vector<Case> cases{{"hexahedra", hexaBlock(2, 4)},
                                  {"tetrahedra", tetraBlock(2, 4)},
                                  {"sector", sectorMesh(4)},
                                  {"dome", domeMesh(4)},
                                  {"curved mid-side block", curvedMidSideBlock(4)}};
    for (const Case& c : cases)
    {
        const mh::SolidPart part(c.mesh.input);
        const mh::Vector exact = affineVector(part, Kind::Static, a, b);
        const double scale = maxAbsOf(exact);
        std::vector<std::size_t> grids(part.gridPositions().size());
        std::iota(grids.begin(), grids.end(), std::size_t{0});
        const std::vector<mh::SolidSkinTrace> traces = part.skinTraces(grids);
        std::size_t edgeCount = 0;
        std::size_t faceCount = 0;
        double error = 0.0;
        for (const mh::SolidSkinTrace& trace : traces)
        {
            ++(trace.grids.size() == 2 ? edgeCount : faceCount);
            for (std::size_t m = 0; m < trace.weights.rows(); ++m)
                for (std::size_t component = 0; component < 3; ++component)
                {
                    double value = 0.0;
                    for (std::size_t p = 0; p < trace.points.size(); ++p)
                    {
                        const mh::SolidPoint& x = trace.points[p];
                        value += trace.weights(m, p) * (b[component] + a[component][0] * x[0] + a[component][1] * x[1] +
                                                        a[component][2] * x[2]);
                    }
                    for (const mh::TieTerm& term : trace.terms[m])
                        value += term.coefficient * exact[globalOf(part, term.unknown, component)];
                    const std::size_t dof =
                        3 * part.gridPositions().size() + part.dofOffset(trace.entity) + 3 * m + component;
                    error = std::max(error, std::fabs(value - exact[dof]));
                }
        }
        runner.expect(edgeCount > 0 && faceCount > 0, c.name + ": edges and faces of the skin");
        char text[64];
        std::snprintf(text, sizeof(text), "%.3e", error / scale);
        runner.expect(error <= 1e-9 * scale, c.name + ": the functions of an affine field, error " + text);
    }
}
