#include "CheckRunner.hpp"
#include "mixedHybrid/SolidPartCheckSupport.hpp"
#include "oos/elements/mixedHybrid/model/SolidPartProjections.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

// The checks of SolidPart, the solids of the model of the mixed hybrid formulation (src/tdnns/solid.rs of the
// reference without its ties): topology, entities, geometry, supports, loads and stresses, verified on small meshes of
// every shape through a dense global system assembled here (SolidPartCheckSupport.hpp).

namespace
{

using namespace oos::checks::solidpart;
using oos::checks::CheckRunner;

// The Voigt stresses D ε of the uniform displacement gradient A in an isotropic steel.
mh::VoigtVector stressOfGradient(const std::array<std::array<double, 3>, 3>& a)
{
    const mh::SolidMaterial material = steel();
    const mh::VoigtVector strain{a[0][0], a[1][1], a[2][2], a[0][1] + a[1][0], a[1][2] + a[2][1], a[2][0] + a[0][2]};
    mh::VoigtVector out{};
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            out[i] += material.stiffness[i][j] * strain[j];
    return out;
}

// The largest |K r| over the six rigid motions, over the largest diagonal term of K.
double rigidResidual(const mh::SolidPart& part, const System& system, Kind kind)
{
    double diagonal = 0.0;
    for (std::size_t i = 0; i < system.stiffness.rows(); ++i)
        diagonal = std::fmax(diagonal, system.stiffness(i, i));
    double worst = 0.0;
    for (const mh::Vector& r : rigidVectors(part, kind))
        worst = std::fmax(worst, maxAbsOf(system.stiffness * r) / diagonal);
    return worst;
}

// The held DOFs of the grids `grids` (all three components, the values of `field`) and the supports SolidPart
// derives from them.
std::map<std::size_t, double> heldWithDerived(const mh::SolidPart& part, const std::vector<std::size_t>& grids,
                                              const mh::Vector& field,
                                              std::vector<mh::SolidDerivedSupport>* derivedOut = nullptr)
{
    std::map<std::size_t, double> held;
    std::vector<mh::SolidHeldComponent> components;
    for (const std::size_t g : grids)
        for (std::size_t c = 0; c < 3; ++c)
        {
            held[3 * g + c] = field[3 * g + c];
            components.push_back({g, c, field[3 * g + c]});
        }
    const std::vector<mh::SolidDerivedSupport> derived = part.deriveSupports(components);
    for (const mh::SolidDerivedSupport& s : derived)
        held[globalDof(part, {false, s.entity, s.dof})] = s.value;
    if (derivedOut)
        *derivedOut = derived;
    return held;
}

// The grids of the solids on the skin (on a face of one solid).
std::vector<std::size_t> skinGrids(const mh::SolidPart& part)
{
    std::set<std::size_t> out;
    for (const mh::SolidPartFace& face : part.faces())
        if (face.onSkin())
            out.insert(face.view().begin(), face.view().end());
    return {out.begin(), out.end()};
}

// The grids used as corners by the solids.
std::vector<std::size_t> cornerGrids(const mh::SolidPart& part)
{
    std::set<std::size_t> out;
    for (std::size_t is = 0; is < part.solidCount(); ++is)
        out.insert(part.solid(is).corners.begin(), part.solid(is).corners.end());
    return {out.begin(), out.end()};
}

// The grids of a mesh with lattice index i = 0 (the clamped side of the checks).
std::vector<std::size_t> sideGrids(const Mesh& mesh)
{
    std::vector<std::size_t> out;
    for (std::size_t j = 0; j <= mesh.lattice.nj; ++j)
        for (std::size_t k = 0; k <= mesh.lattice.nk; ++k)
            out.push_back(mesh.lattice.id(0, j, k));
    return out;
}

// The grids of a mesh with lattice index j = 0 (the clamped end of the sector).
std::vector<std::size_t> endGrids(const Mesh& mesh)
{
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i <= mesh.lattice.ni; ++i)
        for (std::size_t k = 0; k <= mesh.lattice.nk; ++k)
            out.push_back(mesh.lattice.id(i, 0, k));
    return out;
}

// The static solution of a mesh clamped at the grids `clamped` (its side i = 0 when empty) under a force at one grid
// and the acceleration (3, −9, 6) (the load case of the reference probe).
struct ClampedSolution
{
    mh::SolidPart part;
    System system;
    mh::Vector u;
    bool ok = false;
};

const mh::SolidPoint PROBE_ACCELERATION{3.0, -9.0, 6.0};

// The loads of a uniform acceleration on every solid.
VolumeLoads accelerationOf(const mh::SolidPart& part, const mh::SolidPoint& acceleration)
{
    return [&part, acceleration](std::size_t is) { return std::vector{part.accelerationLoad(is, acceleration)}; };
}

ClampedSolution clampedSolution(const Mesh& mesh, std::size_t forceGrid, const mh::SolidPoint& force,
                                const std::vector<std::size_t>& clamped = {})
{
    ClampedSolution out{mh::SolidPart(mesh.input), {}, {}, false};
    const mh::SolidPart& part = out.part;
    out.system = assemble(part, Kind::Static, false, accelerationOf(part, PROBE_ACCELERATION));
    for (std::size_t c = 0; c < 3; ++c)
        out.system.load[3 * forceGrid + c] += force[c];
    const mh::Vector zero(out.system.load.size(), 0.0);
    const std::vector<std::size_t> grids = clamped.empty() ? sideGrids(mesh) : clamped;
    const std::optional<mh::Vector> u = solveHeld(out.system, heldWithDerived(part, grids, zero));
    out.ok = out.system.ok && u.has_value();
    if (u)
        out.u = *u;
    return out;
}

// The centre stress of solid 0 of a solution.
mh::VoigtVector centreStress(const ClampedSolution& s)
{
    const mh::VolumeKernel& kernel = *s.system.kernels[0];
    const mh::Vector q = localValues(s.part, 0, Kind::Static, s.u);
    const std::vector<double> factors(kernel.loads().size(), 1.0);
    const mh::Vector beta = mh::SolidPart::stressCoefficients(kernel, q, factors);
    return mh::SolidPart::pointStresses(kernel, beta).centre;
}

void expectVector(CheckRunner& runner, std::span<const double> value, std::span<const double> expected,
                  double tolerance, const std::string& what)
{
    const double scale = maxAbsOf(expected);
    double error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        error = std::fmax(error, std::fabs(value[i] - expected[i]));
    char relative[32];
    std::snprintf(relative, sizeof relative, "%.3e", error / scale);
    runner.expect(error <= tolerance * scale, what + ": error " + relative + " of the largest value");
}

} // namespace

// ---- topology and entities

CHECK_CASE("solidpart: edges, faces, owners and entities of a block of hexahedra")
{
    const Mesh mesh = hexaBlock(2, 2);
    const mh::SolidPart part(mesh.input);
    runner.expect(part.problems().empty(), "no problem");
    runner.expect(part.edges().size() == 54 && part.faces().size() == 36, "54 edges and 36 faces");
    std::size_t skinFaces = 0;
    std::size_t skinEdges = 0;
    for (const mh::SolidPartFace& face : part.faces())
    {
        skinFaces += face.onSkin() ? 1 : 0;
        runner.expect(face.solids.size() == 1 || face.solids.size() == 2, "one or two owners per face");
        runner.expect(face.grids[0] == *std::min_element(face.view().begin(), face.view().end()) &&
                          face.grids[1] < face.grids[face.count - 1],
                      "canonical cyclic order");
    }
    for (const mh::SolidPartEdge& edge : part.edges())
        skinEdges += edge.onSkin ? 1 : 0;
    runner.expect(skinFaces == 24, "24 faces on the skin");
    runner.expect(skinEdges == 48, "48 edges on the skin");
    // degree 2: 1 function per edge and per quadrilateral face, three components each
    runner.expect(part.entities(Kind::Static).size() == 90, "edges and faces are the static entities");
    runner.expect(part.entities(Kind::Modal).size() == 98, "and the interior fields of the 8 solids");
    const std::size_t staticDofs = part.dofOffset(part.entities(Kind::Static).size());
    runner.expect(staticDofs == 3 * 90, "three DOFs per edge and face at degree 2");
    // edges by key, then faces by key, then interiors by element ID
    for (std::size_t e = 1; e < part.edges().size(); ++e)
        runner.expect(part.edges()[e - 1].grids < part.edges()[e].grids, "edges by key");
    const std::span<const mh::SolidPartEntity> modal = part.entities(Kind::Modal);
    runner.expect(modal[89].kind == mh::SolidPartEntity::Kind::Face &&
                      modal[90].kind == mh::SolidPartEntity::Kind::Interior && modal[90].index == 0 &&
                      modal[90].dofCount == 3,
                  "the interior fields last");
    // the local unknowns: 8 corners, 12 edges, 6 faces, three each; modal: three interior fields more
    const std::vector<mh::SolidLocalUnknown> local = part.localUnknowns(0, Kind::Static);
    runner.expect(local.size() == 3 * (8 + 12 + 6), "78 local unknowns");
    runner.expect(part.localUnknowns(0, Kind::Modal).size() == local.size() + 3, "81 in a modal analysis");
    runner.expect(local[0].onGrid && local[0].index == part.solid(0).corners[0] && local[2].component == 2,
                  "the corners first");
    const std::size_t firstEdge = *part.edgeEntity(part.solidEdges(0)[0]);
    runner.expect(
        !local[24].onGrid && local[24].index == firstEdge && local[24].component == 0 && local[26].component == 2,
        "then edge 0, component by component");

    // degree 4: 9 DOFs per edge, 27 per quadrilateral face, 9 per triangle
    const mh::SolidPart quartic(pyramidTetraMesh(4).input);
    for (const mh::SolidPartEntity& entity : quartic.entities(Kind::Static))
    {
        const std::size_t expected = entity.kind == mh::SolidPartEntity::Kind::Edge ? 9
                                     : quartic.faces()[entity.index].count == 4     ? 27
                                                                                    : 9;
        runner.expect(entity.dofCount == expected, "DOFs of an entity at degree 4");
    }
    runner.expect(quartic.faceIndex(std::vector<std::size_t>{0, 2, 1}).has_value(), "a face by three grids");
}

// ---- rigid motions and affine fields

CHECK_CASE("solidpart: rigid motions have no energy, every shape and geometry")
{
    struct Case
    {
        const char* name;
        Mesh mesh;
    };
    const std::vector<Case> cases{{"hexahedra", hexaBlock(2, 2)},
                                  {"tetrahedra", tetraBlock(2, 2)},
                                  {"pyramids and tetrahedra", pyramidTetraMesh(3)},
                                  {"pentahedra", pentaLayerMesh(3)},
                                  {"curved mid-side block", curvedMidSideBlock(2)}};
    for (const Case& c : cases)
    {
        const mh::SolidPart part(c.mesh.input);
        for (const Kind kind : {Kind::Static, Kind::Modal})
        {
            const System system = assemble(part, kind, kind == Kind::Modal);
            runner.expect(system.ok, std::string(c.name) + ": kernels built");
            const double residual = rigidResidual(part, system, kind);
            runner.expect(residual < 1e-12, std::string(c.name) + ": K r " + std::to_string(residual));
            if (kind != Kind::Modal)
                continue;
            // the rigid translation carries the mass of the block
            double volume = 0.0;
            for (const auto& kernel : system.kernels)
                volume += kernel->volume();
            const mh::Vector r = rigidVectors(part, kind)[0];
            const double mass = mh::dot(r, system.mass * r);
            runner.expectNear(mass / (DENSITY * volume), 1.0, 1e-12, std::string(c.name) + ": rigid-body mass");
        }
    }
}

// The rigid motions of quartic meshes whose skin is read from the mesh.
void curvedRigidMotions(CheckRunner& runner, const Mesh& mesh, const std::string& name)
{
    const mh::SolidPart part(mesh.input);
    const System system = assemble(part, Kind::Static, false);
    runner.expect(system.ok, name + ": kernels built");
    const double residual = rigidResidual(part, system, Kind::Static);
    runner.expect(residual < 1e-12, name + ": K r " + std::to_string(residual));
    bool curved = false;
    for (const mh::SolidPartEdge& edge : part.edges())
        for (const mh::SolidPoint& g : edge.geometry)
            curved = curved || mh::norm3(g) > 1e-3;
    runner.expect(curved, name + ": some edge is curved");
}

CHECK_CASE("solidpart: rigid motions of a sector and of a box with mid-side grids, skins read from the mesh")
{
    curvedRigidMotions(runner, sectorMesh(4), "sector");
    curvedRigidMotions(runner, partialMidSideMesh(4), "partial mid-side grids");
}

CHECK_CASE("solidpart: rigid motions under a dome read from the mesh")
{
    curvedRigidMotions(runner, domeMesh(4), "dome");
}

// ---- the reference: the values below were computed by the reference crate at commit 9f55f0b (tdnns::Analysis on the
// same meshes, the side i = 0 clamped (the end j = 0 of the sector), the force at one grid and the acceleration
// (3, −9, 6); its supports hold the edges and faces of the clamped side at zero, as the derived supports do for zero
// values)

CHECK_CASE("solidpart: reference values, sector of a cylinder with its skin read from the mesh")
{
    const Mesh mesh = sectorMesh(4);
    const ClampedSolution s = clampedSolution(mesh, 11, {1.0e6, -2.0e6, 3.0e6}, endGrids(mesh));
    runner.expect(s.ok, "solved");
    if (!s.ok)
        return;
    expectVector(runner, std::span(s.u).subspan(3 * 11, 3),
                 std::vector{1.72525103930773688e-3, -2.75536521498767270e-3, 4.19000497058390122e-3}, 1e-10,
                 "displacement of grid 11");
    expectVector(runner, centreStress(s),
                 std::vector{1.21159262677371921e6, -9.40796334749752842e6, -4.92965749410826527e6,
                             1.13824307233353648e7, 3.10914189155810773e7, -1.50615467076717992e5},
                 1e-10, "stress at the centre of solid 0");
    // the conic of the outer edge between grids 7 and 9, projected on its functions
    const mh::SolidPartEdge& edge = s.part.edges()[*s.part.edgeIndex(mh::EdgeKey::of(7, 9))];
    const std::vector<std::vector<double>> edgeExpected{{4.40430946604726259e-2, 5.79838341352933995e-3, 0.0},
                                                        {-3.80046127654409815e-4, 2.88673693714658341e-3, 0.0},
                                                        {-1.26023093628191418e-4, -1.65912550300628681e-5, 0.0}};
    for (std::size_t m = 0; m < 3; ++m)
        for (std::size_t c = 0; c < 3; ++c)
            runner.expect(std::fabs(edge.geometry[m][c] - edgeExpected[m][c]) < 1e-12,
                          "edge geometry " + std::to_string(m) + "," + std::to_string(c));
    // the outer face is ruled: its patch is the blend of its edges, nothing is left for its functions
    const mh::SolidPartFace& face = s.part.faces()[*s.part.faceIndex(std::vector<std::size_t>{6, 7, 8, 9})];
    for (const mh::SolidPoint& g : face.geometry)
        runner.expect(mh::norm3(g) < 1e-12, "outer face geometry");
    // the stress at grid 4: the mean of the corner stresses of its solids
    std::vector<std::vector<mh::VoigtVector>> corners(s.part.solidCount());
    for (std::size_t is = 0; is < s.part.solidCount(); ++is)
    {
        const mh::VolumeKernel& kernel = *s.system.kernels[is];
        const std::vector<double> factors(kernel.loads().size(), 1.0);
        const mh::Vector beta =
            mh::SolidPart::stressCoefficients(kernel, localValues(s.part, is, Kind::Static, s.u), factors);
        corners[is] = mh::SolidPart::pointStresses(kernel, beta).corners;
    }
    const std::vector<std::optional<mh::VoigtVector>> grid = s.part.gridStresses(corners);
    runner.expect(grid[4].has_value(), "a grid stress at a grid of solids");
    if (grid[4])
        expectVector(runner, *grid[4],
                     std::vector{-5.01816906385595024e8, -2.43665078879502803e8, -4.46738079719208658e8,
                                 2.45155296029061973e7, -4.46250221528339088e7, -2.10631614669532299e8},
                     1e-10, "stress at grid 4");
}

CHECK_CASE("solidpart: reference values, box with a partial set of mid-side grids")
{
    const ClampedSolution s = clampedSolution(partialMidSideMesh(4), 11, {2.0e6, 1.0e6, -3.0e6});
    runner.expect(s.ok, "solved");
    for (const mh::SolidPartProblem& problem : s.part.problems())
        runner.expect(false, "a problem: " + problem.message);
    if (!s.ok)
        return;
    expectVector(runner, std::span(s.u).subspan(3 * 11, 3),
                 std::vector{2.92795312304211649e-3, 1.87856447817152396e-3, -8.87903259360424239e-3}, 1e-10,
                 "displacement of grid 11");
    expectVector(runner, centreStress(s),
                 std::vector{1.01383519910720754e7, 1.03326298591630894e6, -1.01644665798320640e7,
                             8.26439983362530358e6, 5.50391392797730397e6, -3.77769920536082387e7},
                 1e-10, "stress at the centre of solid 0");
    // five mid-side grids, each fixing its edge; the shared edge 5-7 given by element 1 alone
    runner.expect(s.part.midSideGrids().size() == 5, "five curved edges");
    const mh::SolidPartEdge& shared = s.part.edges()[*s.part.edgeIndex(mh::EdgeKey::of(5, 7))];
    runner.expect(shared.midSideGrid == 13 && std::fabs(shared.geometry[0][2] - 4.0 * 0.03) < 1e-15,
                  "the parabola of a shared edge given by one element");
}

CHECK_CASE("solidpart: reference values, pyramids and tetrahedra")
{
    const ClampedSolution s = clampedSolution(pyramidTetraMesh(3), 7, {1.0e6, 2.0e6, -1.5e6});
    runner.expect(s.ok, "solved");
    if (!s.ok)
        return;
    expectVector(runner, std::span(s.u).subspan(3 * 7, 3),
                 std::vector{1.42724875292208313e-4, 4.20651555846576308e-4, -4.45192694020886566e-4}, 1e-10,
                 "displacement of grid 7");
    expectVector(runner, centreStress(s),
                 std::vector{-3.49745512910267431e6, -4.35889926045937580e5, 2.18383168681327166e5,
                             -2.19946594640082913e6, -5.56277727472788189e5, -1.27415962727441988e6},
                 1e-10, "stress at the centre of solid 0");
}

CHECK_CASE("solidpart: reference values, a layer of pentahedra")
{
    const ClampedSolution s = clampedSolution(pentaLayerMesh(3), 17, {-1.0e6, 1.0e6, 2.0e6});
    runner.expect(s.ok, "solved");
    if (!s.ok)
        return;
    expectVector(runner, std::span(s.u).subspan(3 * 17, 3),
                 std::vector{-1.26855960775256599e-3, 3.04955903837687023e-4, 3.23610283909542620e-3}, 1e-10,
                 "displacement of grid 17");
    expectVector(runner, centreStress(s),
                 std::vector{1.75775014857193120e7, -8.99891252756784204e5, -1.45735428714210604e5,
                             3.62579017897030152e6, -4.52540766752129048e6, 1.66324811194310850e6},
                 1e-10, "stress at the centre of solid 0");
}

CHECK_CASE("solidpart: reference values, hexahedra under a dome read from the mesh")
{
    const ClampedSolution s = clampedSolution(domeMesh(4), 17, {1.0e6, 1.0e6, -2.0e6});
    runner.expect(s.ok, "solved");
    if (!s.ok)
        return;
    expectVector(runner, std::span(s.u).subspan(3 * 17, 3),
                 std::vector{7.21098784854161358e-4, 7.44429841359244438e-4, -2.06003970250274223e-3}, 1e-10,
                 "displacement of grid 17");
    expectVector(runner, centreStress(s),
                 std::vector{3.73277657814556826e6, 3.90925593904458219e5, -5.14720763122976932e5,
                             1.62018493313656398e6, 2.86586335622489965e6, -2.87762327140155854e6},
                 1e-10, "stress at the centre of solid 0");
    const mh::SolidPartEdge& edge = s.part.edges()[*s.part.edgeIndex(mh::EdgeKey::of(1, 7))];
    const std::vector<std::vector<double>> edgeExpected{{-1.11667294042114840e-2, 0.0, 7.44448626947467051e-2},
                                                        {5.55230670631767786e-3, 0.0, 8.32846005946796808e-4},
                                                        {4.13619168886381061e-5, 0.0, -2.75746112592178960e-4}};
    for (std::size_t m = 0; m < 3; ++m)
        for (std::size_t c = 0; c < 3; ++c)
            runner.expect(std::fabs(edge.geometry[m][c] - edgeExpected[m][c]) < 1e-12,
                          "edge geometry " + std::to_string(m) + "," + std::to_string(c));
    const mh::SolidPartFace& face = s.part.faces()[*s.part.faceIndex(std::vector<std::size_t>{1, 3, 7, 9})];
    expectVector(runner, face.geometry[0],
                 std::vector{6.18268795052480499e-4, 5.60200403025239873e-4, -1.43913942188758138e-3}, 1e-10,
                 "top face geometry 0");
    expectVector(runner, face.geometry[8],
                 std::vector{3.07784571111640079e-9, 2.78877468866494009e-9, -7.16428708285710634e-9}, 1e-4,
                 "top face geometry 8");
}

// ---- exact states

namespace
{

const std::array<std::array<double, 3>, 3> UNIFORM_GRADIENT{
    {{1.0e-3, 2.0e-4, -3.0e-4}, {5.0e-4, -2.0e-3, 1.0e-4}, {-1.0e-4, 3.0e-4, 1.5e-3}}};
const mh::SolidPoint UNIFORM_TRANSLATION{1.0e-3, -2.0e-3, 5.0e-4};

// The uniform strain of a block with a curved boundary under the stresses of the field on its faces, three grids
// held (the exact states of document 35 of the reference, section 5).
void tractionProblem(CheckRunner& runner, const Mesh& mesh, const std::vector<std::size_t>& heldGrids,
                     const std::string& name)
{
    const std::array<std::array<double, 3>, 3>& a = UNIFORM_GRADIENT;
    const mh::SolidPoint& b = UNIFORM_TRANSLATION;
    const mh::VoigtVector sigma = stressOfGradient(a);
    const mh::SolidPart part(mesh.input);
    // the stresses of the field on every face of the skin
    const FaceLoads onSkin = [&](std::size_t is, const mh::VolumeKernel& kernel)
    {
        mh::Vector f(kernel.parameterCount(), 0.0);
        for (std::size_t lf = 0; lf < part.solidFaces(is).size(); ++lf)
        {
            if (!part.faces()[part.solidFaces(is)[lf]].onSkin())
                continue;
            const mh::Vector onFace = part.faceStressLoad(kernel, lf, [&](const mh::SolidPoint&) { return sigma; });
            for (std::size_t i = 0; i < f.size(); ++i)
                f[i] += onFace[i];
        }
        return f;
    };
    const System system = assemble(part, Kind::Static, false, {}, onSkin);
    const mh::Vector field = affineVector(part, Kind::Static, a, b);
    std::map<std::size_t, double> held;
    for (const std::size_t g : heldGrids)
        for (std::size_t c = 0; c < 3; ++c)
            held[3 * g + c] = field[3 * g + c];
    const std::optional<mh::Vector> u = solveHeld(system, held);
    runner.expect(system.ok && u.has_value(), name + ": solved");
    if (!u)
        return;
    // the grids of the corners and the entities (the mid-side grids carry no unknown)
    mh::Vector value;
    mh::Vector expected;
    for (const std::size_t g : cornerGrids(part))
        for (std::size_t c = 0; c < 3; ++c)
        {
            value.push_back((*u)[3 * g + c]);
            expected.push_back(field[3 * g + c]);
        }
    for (std::size_t i = 3 * part.gridPositions().size(); i < u->size(); ++i)
    {
        value.push_back((*u)[i]);
        expected.push_back(field[i]);
    }
    expectVector(runner, value, expected, 1e-10, name + ": displacements and entity values");
    for (std::size_t is = 0; is < part.solidCount(); ++is)
    {
        const mh::Vector beta =
            mh::SolidPart::stressCoefficients(*system.kernels[is], localValues(part, is, Kind::Static, *u), {});
        expectVector(runner, mh::SolidPart::pointStresses(*system.kernels[is], beta).centre, sigma, 1e-10,
                     name + ": stress at a centre");
    }
    // a mid-side grid moves with its edge
    for (const auto& [grid, edge] : part.midSideGrids())
        for (std::size_t c = 0; c < 3; ++c)
        {
            double moved = 0.0;
            for (const mh::SolidTerm& term : part.midSideTerms(edge, c))
                moved += term.coefficient * (*u)[globalDof(part, term.unknown)];
            const mh::SolidPoint& x = part.gridPositions()[grid];
            const double exact = b[c] + a[c][0] * x[0] + a[c][1] * x[1] + a[c][2] * x[2];
            runner.expect(std::fabs(moved - exact) < 1e-12, name + ": displacement of a mid-side grid");
        }
}

// The same mesh with the grid IDs renumbered: grid g of `mesh` is grid perm[g] of the result.
Mesh renumbered(const Mesh& mesh, const std::vector<std::size_t>& perm)
{
    Mesh out = mesh;
    for (std::size_t g = 0; g < perm.size(); ++g)
        out.input.gridPositions[perm[g]] = mesh.input.gridPositions[g];
    for (mh::SolidPartElement& element : out.input.solids)
    {
        for (std::size_t& g : element.corners)
            g = perm[g];
        for (std::size_t& g : element.midSideGrids)
            if (g != mh::NO_MID_SIDE_GRID)
                g = perm[g];
    }
    out.input.sharpEdges.clear();
    for (const mh::EdgeKey& edge : mesh.input.sharpEdges)
        out.input.sharpEdges.insert(mh::EdgeKey::of(perm[edge.lower], perm[edge.higher]));
    return out;
}

// The stress at the centre of solid `is` of a static solution, its loads applied.
mh::VoigtVector centreStressOf(const mh::SolidPart& part, const System& system, const mh::Vector& u, std::size_t is)
{
    const mh::VolumeKernel& kernel = *system.kernels[is];
    const std::vector<double> factors(kernel.loads().size(), 1.0);
    const mh::Vector beta = mh::SolidPart::stressCoefficients(kernel, localValues(part, is, Kind::Static, u), factors);
    return mh::SolidPart::pointStresses(kernel, beta).centre;
}

} // namespace

CHECK_CASE("solidpart: a uniform strain is exact with the boundary held, every shape")
{
    const mh::VoigtVector sigma = stressOfGradient(UNIFORM_GRADIENT);
    for (const Mesh& mesh : {hexaBlock(2, 2), tetraBlock(2, 2), pyramidTetraMesh(3), pentaLayerMesh(3)})
    {
        const mh::SolidPart part(mesh.input);
        const System system = assemble(part, Kind::Static, false);
        const mh::Vector field = affineVector(part, Kind::Static, UNIFORM_GRADIENT, UNIFORM_TRANSLATION);
        std::vector<mh::SolidDerivedSupport> derived;
        const std::map<std::size_t, double> held = heldWithDerived(part, skinGrids(part), field, &derived);
        // the skin is flat and its edges straight: every derived value is zero, by an empty map
        const bool straight = std::all_of(derived.begin(), derived.end(), [](const mh::SolidDerivedSupport& s)
                                          { return s.value == 0.0 && s.carriedTo.empty(); });
        runner.expect(!derived.empty() && straight, "derived supports of a straight skin");
        const std::optional<mh::Vector> u = solveHeld(system, held);
        runner.expect(system.ok && u.has_value(), "solved");
        if (!u)
            continue;
        expectVector(runner, *u, field, 1e-10, "displacements and entity values");
        std::vector<std::vector<mh::VoigtVector>> corners(part.solidCount());
        for (std::size_t is = 0; is < part.solidCount(); ++is)
        {
            const mh::Vector beta =
                mh::SolidPart::stressCoefficients(*system.kernels[is], localValues(part, is, Kind::Static, *u), {});
            const mh::SolidPointStresses stresses = mh::SolidPart::pointStresses(*system.kernels[is], beta);
            expectVector(runner, stresses.centre, sigma, 1e-10, "stress at a centre");
            for (const mh::VoigtVector& s : stresses.corners)
                expectVector(runner, s, sigma, 1e-10, "stress at a corner");
            corners[is] = stresses.corners;
        }
        // the grid point stresses of the uniform field
        const std::vector<std::optional<mh::VoigtVector>> grid = part.gridStresses(corners);
        for (const std::size_t g : cornerGrids(part))
            expectVector(runner, *grid[g], sigma, 1e-10, "stress at a grid");
    }
}

CHECK_CASE("solidpart: a uniform strain is exact on a block with curved mid-side edges")
{
    const Mesh mesh = curvedMidSideBlock(2);
    const Lattice& l = mesh.lattice;
    tractionProblem(runner, mesh, {l.id(0, 0, 0), l.id(2, 0, 0), l.id(0, 2, 0)}, "mid-side block");
}

CHECK_CASE("solidpart: a uniform strain is exact under a dome read from the mesh")
{
    const Mesh mesh = domeMesh(4);
    const Lattice& l = mesh.lattice;
    tractionProblem(runner, mesh, {l.id(0, 0, 0), l.id(2, 0, 0), l.id(0, 2, 0)}, "dome");
}

// ---- derived supports

CHECK_CASE("solidpart: the derived supports of a clamped curved face")
{
    const Mesh mesh = domeMesh(4);
    const mh::SolidPart part(mesh.input);
    std::vector<std::size_t> top;
    for (std::size_t i = 0; i <= 2; ++i)
        for (std::size_t j = 0; j <= 2; ++j)
            top.push_back(mesh.lattice.id(i, j, 1));
    // a rigid motion given at the grids of the curved top: the derived values are those of the motion
    const mh::SolidPoint omega{3.0e-4, -2.0e-4, 5.0e-4};
    std::array<std::array<double, 3>, 3> w{};
    for (std::size_t axis = 0; axis < 3; ++axis)
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t d = 0; d < 3; ++d)
                w[c][d] += omega[axis] * rotationGradient(axis)[c][d];
    const mh::Vector rigid = affineVector(part, Kind::Static, w, {1.0e-3, 2.0e-3, -1.0e-3});
    std::vector<mh::SolidDerivedSupport> derived;
    const std::map<std::size_t, double> held = heldWithDerived(part, top, rigid, &derived);
    // the 4 faces of the top (27 DOFs each) and its 12 edges (9 each)
    runner.expect(derived.size() == 4 * 27 + 12 * 9, "the top faces and edges held");
    double worst = 0.0;
    std::size_t carried = 0;
    for (const mh::SolidDerivedSupport& s : derived)
    {
        worst = std::fmax(worst, std::fabs(s.value - rigid[globalDof(part, {false, s.entity, s.dof})]));
        carried += s.carriedTo.empty() ? 0 : 1;
        for (const mh::SolidSupportTerm& term : s.carriedTo)
            runner.expect(std::find(top.begin(), top.end(), term.grid) != top.end(),
                          "a reaction goes to the grids of the clamped face");
    }
    runner.expect(worst < 1e-12 * maxAbsOf(rigid), "derived values of a rigid motion: " + std::to_string(worst));
    runner.expect(carried > 0, "the curved top carries its reactions to the grids");
    // without load, the block follows the motion of its clamped top
    System system = assemble(part, Kind::Static, false, accelerationOf(part, PROBE_ACCELERATION));
    System unloaded = system;
    std::fill(unloaded.load.begin(), unloaded.load.end(), 0.0);
    const std::optional<mh::Vector> u = solveHeld(unloaded, held);
    runner.expect(u.has_value(), "solved");
    if (u)
        expectVector(runner, *u, rigid, 1e-10, "the rigid motion everywhere");

    // under loads with the top clamped at zero: the support forces carried to the grids balance the loads
    const std::size_t loadedGrid = mesh.lattice.id(2, 2, 0);
    const mh::SolidPoint force{1.0e6, 1.0e6, -2.0e6};
    for (std::size_t c = 0; c < 3; ++c)
        system.load[3 * loadedGrid + c] += force[c];
    std::vector<mh::SolidDerivedSupport> clamped;
    const std::optional<mh::Vector> v =
        solveHeld(system, heldWithDerived(part, top, mh::Vector(system.load.size(), 0.0), &clamped));
    runner.expect(v.has_value(), "solved under loads");
    if (!v)
        return;
    mh::Vector reaction = system.stiffness * *v;
    for (std::size_t i = 0; i < reaction.size(); ++i)
        reaction[i] -= system.load[i];
    // the support forces reported at the grids: their own, and those of the derived DOFs carried by the maps
    mh::Vector reported(3 * part.gridPositions().size(), 0.0);
    for (const std::size_t g : top)
        for (std::size_t c = 0; c < 3; ++c)
            reported[3 * g + c] = reaction[3 * g + c];
    for (const mh::SolidDerivedSupport& s : clamped)
        for (const mh::SolidSupportTerm& term : s.carriedTo)
            reported[3 * term.grid + term.component] +=
                term.coefficient * reaction[globalDof(part, {false, s.entity, s.dof})];
    const double scale = maxAbsOf(reported);
    for (const mh::Vector& r : rigidVectors(part, Kind::Static))
    {
        // the work of the reported forces on the motion of the grids against that of the loads on the whole motion
        const double applied = mh::dot(system.load, r);
        const double onGrids = mh::dot(reported, std::span(r).first(reported.size()));
        runner.expect(std::fabs(onGrids + applied) < 1e-10 * scale * maxAbsOf(r),
                      "the reported support forces balance the loads: " +
                          std::to_string(std::fabs(onGrids + applied) / (scale * maxAbsOf(r))));
    }
}

// ---- invariance

CHECK_CASE("solidpart: renumbering the grids gives the same results")
{
    for (const Mesh& mesh : {curvedMidSideBlock(2), sectorMesh(4)})
    {
        const std::size_t n = mesh.input.gridPositions.size();
        std::size_t step = 7;
        while (std::gcd(step, n) != 1)
            step += 2;
        std::vector<std::size_t> perm(n);
        for (std::size_t g = 0; g < n; ++g)
            perm[g] = (step * g + 3) % n;
        const Lattice& l = mesh.lattice;
        const std::size_t loaded = l.id(l.ni, l.nj, l.nk);
        const mh::SolidPoint force{1.0e6, -2.0e6, 3.0e6};
        const ClampedSolution base = clampedSolution(mesh, loaded, force);
        // the same problem on the renumbered grids
        const Mesh other = renumbered(mesh, perm);
        const mh::SolidPart part(other.input);
        System system = assemble(part, Kind::Static, false, accelerationOf(part, PROBE_ACCELERATION));
        for (std::size_t c = 0; c < 3; ++c)
            system.load[3 * perm[loaded] + c] += force[c];
        std::vector<std::size_t> side;
        for (const std::size_t g : sideGrids(mesh))
            side.push_back(perm[g]);
        const std::optional<mh::Vector> u =
            solveHeld(system, heldWithDerived(part, side, mh::Vector(system.load.size(), 0.0)));
        runner.expect(base.ok && u.has_value(), "solved");
        if (!u || !base.ok)
            continue;
        mh::Vector expected;
        mh::Vector value;
        for (const std::size_t g : cornerGrids(base.part))
            for (std::size_t c = 0; c < 3; ++c)
            {
                expected.push_back(base.u[3 * g + c]);
                value.push_back((*u)[3 * perm[g] + c]);
            }
        expectVector(runner, value, expected, 1e-10, "displacements of the grids");
        for (std::size_t is = 0; is < part.solidCount(); ++is)
            expectVector(runner, centreStressOf(part, system, *u, is),
                         centreStressOf(base.part, base.system, base.u, is), 1e-10, "stresses at the centres");
    }
}

CHECK_CASE("solidpart: elements given from another corner give the same results")
{
    const Mesh mesh = sectorMesh(4);
    Mesh turned = mesh;
    for (mh::SolidPartElement& element : turned.input.solids)
    {
        const std::vector<std::size_t> c = element.corners;
        element.corners = {c[1], c[2], c[3], c[0], c[5], c[6], c[7], c[4]};
    }
    const std::size_t loaded = mesh.lattice.id(1, 2, 1);
    const ClampedSolution base = clampedSolution(mesh, loaded, {1.0e6, -2.0e6, 3.0e6});
    const ClampedSolution other = clampedSolution(turned, loaded, {1.0e6, -2.0e6, 3.0e6});
    runner.expect(base.ok && other.ok, "solved");
    if (!base.ok || !other.ok)
        return;
    expectVector(runner, std::span(other.u).first(3 * mesh.lattice.gridCount()),
                 std::span(base.u).first(3 * mesh.lattice.gridCount()), 1e-10, "displacements of the grids");
    for (std::size_t is = 0; is < base.part.solidCount(); ++is)
        expectVector(runner, centreStressOf(other.part, other.system, other.u, is),
                     centreStressOf(base.part, base.system, base.u, is), 1e-10, "stresses at the centres");
}

// ---- problems

CHECK_CASE("solidpart: problems attached to their elements")
{
    using Severity = mh::SolidPartProblem::Severity;
    Mesh mesh = hexaBlock(2, 2);
    std::vector<mh::SolidPartElement>& solids = mesh.input.solids;
    const auto middleOffset = [&](std::size_t a, std::size_t b, double alongChord, const mh::SolidPoint& across)
    {
        const mh::SolidPoint chord = mh::sub3(mesh.input.gridPositions[b], mesh.input.gridPositions[a]);
        return addMidSideGrid(mesh, a, b, mh::add3(mh::scl3(chord, alongChord), across));
    };
    for (std::size_t is = 0; is < 4; ++is)
        solids[is].midSideGrids.assign(12, mh::NO_MID_SIDE_GRID);
    // solids 0 and 1 share the edge 8 of solid 0 (its edge 0): each gives its own mid-side grid
    solids[0].midSideGrids[8] = middleOffset(solids[0].corners[4], solids[0].corners[5], 0.0, {0.0, 0.0, 0.01});
    solids[1].midSideGrids[0] = middleOffset(solids[1].corners[0], solids[1].corners[1], 0.0, {0.0, 0.01, 0.0});
    // solid 2: a mid-side grid 0.4 of its edge from the middle: the edge folds
    solids[2].midSideGrids[1] = middleOffset(solids[2].corners[1], solids[2].corners[2], 0.4, {});
    // solid 3: a corner of the block as a mid-side grid
    solids[3].midSideGrids[2] = mesh.lattice.id(2, 2, 2);
    // two solids that cannot be read
    solids.push_back(elementOf(9, {0, 1, 2, 3, 4, 5, 6}));
    solids.push_back(elementOf(10, {0, 1, 1, 2}));
    const mh::SolidPart part(mesh.input);
    const auto problemOf = [&](std::int32_t elementId) -> const mh::SolidPartProblem*
    {
        for (const mh::SolidPartProblem& problem : part.problems())
            if (problem.elementId == elementId)
                return &problem;
        return nullptr;
    };
    runner.expect(part.problems().size() == 6, "six problems: " + std::to_string(part.problems().size()));
    const mh::SolidPartProblem* other = problemOf(2);
    runner.expect(other && other->severity == Severity::Warning && other->message.find("lower ID") != std::string::npos,
                  "a second mid-side grid on an edge: the first kept");
    const mh::SolidPartProblem* folds = problemOf(3);
    runner.expect(folds && folds->severity == Severity::Error && folds->message.find("folds") != std::string::npos,
                  "a mid-side grid too far from the middle");
    const mh::SolidPartProblem* corner = problemOf(4);
    runner.expect(
        corner && corner->severity == Severity::Error && corner->message.find("corner of a solid") != std::string::npos,
        "a mid-side grid that is a corner");
    runner.expect(problemOf(9) && !part.usable(8) && problemOf(10) && !part.usable(9), "solids that cannot be read");
    runner.expect(part.usable(0) && part.edges().size() == 54, "the others are read");
    const mh::SolidPartEdge& shared = part.edges()[part.solidEdges(0)[8]];
    runner.expect(shared.midSideGrid == solids[0].midSideGrids[8], "the mid-side grid of the lower element ID");
    // the kernels find the folded elements
    runner.expect(!part.validate(0).has_value(), "an element accepted");
    const std::optional<mh::SolidPartProblem> folded = part.validate(2);
    runner.expect(folded && folded->elementId == 3 && folded->severity == Severity::Error,
                  "the element of the folding edge refused by its kernel");
    Mesh twisted = hexaBlock(1, 2);
    std::vector<std::size_t>& c = twisted.input.solids[0].corners;
    std::swap(c[2], c[3]);
    std::swap(c[6], c[7]);
    const mh::SolidPart bowTie(twisted.input);
    runner.expect(bowTie.validate(0).has_value(), "a hexahedron whose faces cross is refused");
}

// ---- loads

CHECK_CASE("solidpart: volume and face loads")
{
    const Mesh mesh = curvedMidSideBlock(2);
    const mh::SolidPart part(mesh.input);
    // a uniform acceleration: the grids carry ρ a V (the corner functions are a partition of unity)
    const mh::SolidPoint acceleration{1.0, -2.0, 3.0};
    const System system = assemble(part, Kind::Static, false, accelerationOf(part, acceleration));
    double volume = 0.0;
    for (const auto& kernel : system.kernels)
        volume += kernel->volume();
    for (std::size_t c = 0; c < 3; ++c)
    {
        double total = 0.0;
        for (std::size_t g = 0; g < part.gridPositions().size(); ++g)
            total += system.load[3 * g + c];
        runner.expectNear(total / (DENSITY * volume * acceleration[c]), 1.0, 1e-12, "the force of the acceleration");
    }
    // a pressure on the curved top: by a field, by its corner values, and its resultant −p ∫ n dA
    constexpr double PRESSURE = 3.0e5;
    for (std::size_t is = 0; is < part.solidCount(); ++is)
    {
        const mh::VolumeKernel& kernel = *system.kernels[is];
        for (std::size_t lf = 0; lf < part.solidFaces(is).size(); ++lf)
        {
            const mh::SolidPartFace& face = part.faces()[part.solidFaces(is)[lf]];
            if (!face.onSkin())
                continue;
            const std::vector<std::size_t> grids(face.view().begin(), face.view().end());
            runner.expect(part.localFaceOf(is, grids) == lf, "the local face of its grids");
            const auto uniform = [](const mh::SolidPoint&) { return PRESSURE; };
            const mh::Vector byField = part.facePressureLoad(kernel, lf, uniform);
            const std::vector<double> corners(grids.size(), PRESSURE);
            const mh::Vector byCorners = part.faceCornerPressureLoad(kernel, is, lf, grids, corners);
            expectVector(runner, byCorners, byField, 1e-13, "a uniform pressure by its corners");
            mh::SolidPoint resultant{};
            for (const mh::VolumeFacePoint& point : kernel.facePointsOf(lf, part.traceDegree() + 4))
                resultant = mh::add3(resultant, mh::scl3(mh::cross3(point.tangentS, point.tangentT),
                                                         -PRESSURE * point.weight * point.outward));
            mh::SolidPoint onCorners{};
            for (std::size_t j = 0; j < part.solid(is).corners.size(); ++j)
                for (std::size_t c = 0; c < 3; ++c)
                    onCorners[c] += byField[3 * j + c];
            expectVector(runner, onCorners, resultant, 1e-12, "the resultant of a pressure");
            // a pressure linear in x on a flat face: its corner values interpolate it exactly
            if (face.view().size() == 4 && std::fabs(part.gridPositions()[grids[0]][2]) < 1e-12 &&
                std::fabs(part.gridPositions()[grids[2]][2]) < 1e-12)
            {
                const auto linear = [](const mh::SolidPoint& x) { return 1.0e5 + 4.0e5 * x[0] - 2.0e5 * x[1]; };
                std::vector<double> values;
                for (const std::size_t g : grids)
                    values.push_back(linear(part.gridPositions()[g]));
                expectVector(runner, part.faceCornerPressureLoad(kernel, is, lf, grids, values),
                             part.facePressureLoad(kernel, lf, linear), 1e-12, "a linear pressure by its corners");
            }
            // the load stiffness of a following pressure is symmetric
            const mh::Matrix follower = part.followerPressureStiffness(kernel, lf, uniform);
            runner.expect(mh::maxAbs(follower - follower.transposed()) <= 1e-12 * mh::maxAbs(follower),
                          "symmetric load stiffness");
        }
    }
    // the face of three grids, and the first solid that has it
    const std::optional<std::pair<std::size_t, std::size_t>> found = part.solidFaceOf(
        std::vector<std::size_t>{mesh.lattice.id(0, 0, 0), mesh.lattice.id(1, 0, 0), mesh.lattice.id(1, 1, 0)});
    runner.expect(found && found->first == 0 && found->second == 0, "a face by three of its grids");
}

CHECK_CASE("solidpart: the free expansion of a temperature rise is stress free")
{
    constexpr double EXPANSION = 1.2e-5;
    constexpr double RISE = 50.0;
    Mesh mesh = hexaBlock(2, 2);
    for (mh::SolidPartElement& element : mesh.input.solids)
        element.material = steel(EXPANSION);
    const mh::SolidPart part(mesh.input);
    const auto rise = [](const mh::SolidPoint&) { return RISE; };
    const System system =
        assemble(part, Kind::Static, false, [&](std::size_t is) { return std::vector{part.thermalLoad(is, rise)}; });
    // u = α ΔT x, held at three grids
    const double strain = EXPANSION * RISE;
    const mh::Vector field = affineVector(part, Kind::Static, {{{strain, 0, 0}, {0, strain, 0}, {0, 0, strain}}}, {});
    std::map<std::size_t, double> held;
    for (const std::size_t g : {mesh.lattice.id(0, 0, 0), mesh.lattice.id(2, 0, 0), mesh.lattice.id(0, 2, 0)})
        for (std::size_t c = 0; c < 3; ++c)
            held[3 * g + c] = field[3 * g + c];
    const std::optional<mh::Vector> u = solveHeld(system, held);
    runner.expect(u.has_value(), "solved");
    if (!u)
        return;
    expectVector(runner, *u, field, 1e-10, "the free expansion");
    for (std::size_t is = 0; is < part.solidCount(); ++is)
    {
        // the particular stresses of the thermal load, factor 1
        const std::vector<double> thermal{1.0};
        const mh::Vector beta =
            mh::SolidPart::stressCoefficients(*system.kernels[is], localValues(part, is, Kind::Static, *u), thermal);
        for (const mh::VoigtVector& s : mh::SolidPart::pointStresses(*system.kernels[is], beta).corners)
            runner.expect(maxAbsOf(s) < 1e-8 * YOUNG * strain, "no stress");
    }
}

// ---- material axes, geometric stiffness, projections

CHECK_CASE("solidpart: the law and the stresses in the material axes")
{
    // an orthotropic law along axes turned by 90 degrees about z is the law with x and y exchanged
    mh::SolidMaterial local = steel();
    local.stiffness[0][0] *= 1.5;
    local.stiffness[3][3] *= 0.7;
    local.stiffness[4][4] *= 1.2;
    constexpr std::array<std::size_t, 6> EXCHANGE{1, 0, 2, 3, 5, 4};
    mh::SolidMaterial exchanged = local;
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            exchanged.stiffness[i][j] = local.stiffness[EXCHANGE[i]][EXCHANGE[j]];
    oos::base::Mat3 axes;
    axes(1, 0) = 1.0;  // material x along basic y
    axes(0, 1) = -1.0; // material y along basic −x
    axes(2, 2) = 1.0;
    Mesh mesh = hexaBlock(1, 2);
    mesh.input.solids[0].material = local;
    mesh.input.solids[0].materialAxes = axes;
    Mesh reference = mesh;
    reference.input.solids[0].material = exchanged;
    reference.input.solids[0].materialAxes = oos::base::Mat3::identity();
    const mh::SolidPart turned(mesh.input);
    const mh::SolidPart plain(reference.input);
    const mh::SolidResult<mh::VolumeKernel> a = turned.buildKernel(0, Kind::Static, false);
    const mh::SolidResult<mh::VolumeKernel> b = plain.buildKernel(0, Kind::Static, false);
    runner.expect(a.ok() && b.ok(), "kernels built");
    if (a.ok() && b.ok())
        runner.expect(
            mh::maxAbs(a.value().stiffness() - b.value().stiffness()) <= 1e-12 * mh::maxAbs(b.value().stiffness()),
            "the same stiffness");
    // a stress along basic y is along material x
    const mh::VoigtVector inMaterial = turned.toMaterialAxes(0, {0.0, 100.0, 0.0, 0.0, 0.0, 0.0});
    expectVector(runner, inMaterial, std::vector{100.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1e-15, "a stress in material axes");
}

CHECK_CASE("solidpart: the geometric stiffness of a preload")
{
    const Mesh mesh = hexaBlock(2, 2);
    const mh::SolidPart part(mesh.input);
    const System system = assemble(part, Kind::Static, false);
    const mh::Vector preload =
        affineVector(part, Kind::Static, {{{-1.0e-3, 0, 0}, {0, 3.0e-4, 0}, {0, 0, 3.0e-4}}}, {});
    for (std::size_t is = 0; is < part.solidCount(); ++is)
    {
        const mh::VolumeKernel& kernel = *system.kernels[is];
        const mh::Vector beta =
            mh::SolidPart::stressCoefficients(kernel, localValues(part, is, Kind::Static, preload), {});
        const mh::SolidResult<mh::Matrix> kg = mh::SolidPart::geometricStiffness(kernel, beta);
        runner.expect(kg.ok(), "built");
        if (!kg.ok())
            continue;
        const mh::Matrix& g = kg.value();
        runner.expect(mh::maxAbs(g - g.transposed()) <= 1e-12 * mh::maxAbs(g), "symmetric");
        // a translation has no gradient
        mh::Vector translation(g.rows(), 0.0);
        for (std::size_t j = 0; j < part.solid(is).corners.size(); ++j)
            translation[3 * j + 1] = 1.0;
        runner.expect(maxAbsOf(g * translation) <= 1e-12 * mh::maxAbs(g), "no stiffness on a translation");
    }
}

CHECK_CASE("solidpart: projections on the functions of an edge and of a face")
{
    // a parabola through its middle point: 4 h on the first function, nothing on the others
    const mh::EdgeProjection edge = mh::EdgeProjection::of(4);
    const mh::SolidPoint xa{0.1, 0.2, 0.3};
    const mh::SolidPoint xb{1.1, -0.2, 0.5};
    const mh::SolidPoint h{0.02, 0.05, -0.03};
    std::vector<mh::SolidPoint> values;
    for (const double s : edge.parameters)
        values.push_back(mh::add3(mh::add3(mh::scl3(xa, 1.0 - s), mh::scl3(xb, s)), mh::scl3(h, 4.0 * s * (1.0 - s))));
    const std::vector<mh::SolidPoint> coefficients = edge.project(values, xa, xb);
    expectVector(runner, coefficients[0], mh::scl3(h, 4.0), 1e-14, "the parabola on its first function");
    runner.expect(mh::norm3(coefficients[1]) < 1e-15 && mh::norm3(coefficients[2]) < 1e-15, "and on no other");
    // a surface made of the functions of a face is projected back on them
    for (const std::vector<oos::base::EntityId>& order :
         {std::vector<oos::base::EntityId>{3, 9, 5}, std::vector<oos::base::EntityId>{4, 2, 8, 6}})
    {
        const std::size_t nf = order.size();
        const mh::FaceProjection face = mh::FaceProjection::of(order, 4, 4);
        std::vector<mh::SolidPoint> corners;
        std::vector<std::vector<mh::SolidPoint>> edges(nf);
        for (std::size_t k = 0; k < nf; ++k)
        {
            const double f = static_cast<double>(k);
            corners.push_back({std::cos(f), std::sin(f), 0.1 * f});
            for (std::size_t m = 0; m < 3; ++m)
                edges[k].push_back({0.01 * (f + 1.0), -0.02 * static_cast<double>(m), 0.005});
        }
        std::vector<mh::SolidPoint> inside;
        for (std::size_t m = 0; m < face.interiorCount; ++m)
            inside.push_back({0.003 * static_cast<double>(m + 1), 0.001, -0.002 * static_cast<double>(m)});
        std::vector<mh::SolidPoint> surface;
        for (std::size_t q = 0; q < face.points.size(); ++q)
        {
            const std::vector<double>& f = face.functionValues[q];
            mh::SolidPoint x{};
            for (std::size_t i = 0; i < nf; ++i)
                x = mh::add3(x, mh::scl3(corners[i], f[i]));
            for (std::size_t k = 0; k < nf; ++k)
                for (std::size_t m = 0; m < 3; ++m)
                    x = mh::add3(x, mh::scl3(edges[k][m], f[nf + 3 * k + m]));
            for (std::size_t m = 0; m < face.interiorCount; ++m)
                x = mh::add3(x, mh::scl3(inside[m], f[nf + 3 * nf + m]));
            surface.push_back(x);
        }
        const std::vector<mh::SolidPoint> projected = face.project(surface, corners, edges);
        for (std::size_t m = 0; m < face.interiorCount; ++m)
            expectVector(runner, projected[m], inside[m], 1e-10, "a face function projected back");
    }
}
