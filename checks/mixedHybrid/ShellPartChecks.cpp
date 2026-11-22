// Checks of the shells of a model of the mixed hybrid formulation (model/ShellPart.hpp): the numbering, the constant
// states of a plate held by its boundary, the rigid motions of curved panels with and without lines on their edges,
// the mean rotation at the grids, a cantilever against the beam theory, the invariance under a renumbering of the
// grids, the mid-side grids, the resultants of the loads, the problems, and numbers of the reference implementation.

#include "CheckRunner.hpp"
#include "ShellPartCheckSupport.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/line/LineAnalysis.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string>

namespace oos::checks
{

namespace
{

using namespace oos::checks::shellpart;

// The square plate [0, 1]² in the plane z = 0, 2 × 2 elements, and its boundary grids (all but the centre, 4).
MappedMesh squarePlate()
{
    return mappedMesh(2, 2, [](double s, double t) { return mh::SpaceVector{s, t, 0.0}; });
}

const std::vector<std::size_t> SQUARE_BOUNDARY{0, 1, 2, 3, 5, 6, 7, 8};
// The two edges of its side x = 1, each from its lower grid.
const std::array<std::pair<std::size_t, std::size_t>, 2> FAR_EDGE{{{6, 7}, {7, 8}}};

// The cylinder panel of the reference probe: axis x, radius 1, length 2, 0.8 rad, 2 × 2 elements.
MappedMesh cylinderPanel()
{
    return mappedMesh(2, 2,
                      [](double s, double t)
                      {
                          const double a = 0.8 * (t - 0.5);
                          return mh::SpaceVector{2.0 * s, std::sin(a), std::cos(a)};
                      });
}

// Every grid component of the six rigid motions about the origin: translations x, y, z, rotations about x, y, z.
std::array<mh::Vector, 6> rigidVectors(const mh::ShellPart& part)
{
    std::array<mh::Vector, 6> out;
    for (std::size_t c = 0; c < 3; ++c)
    {
        mh::SpaceVector b{};
        b[c] = 1.0;
        out[c] = affineVector(part, {}, b);
        mh::SpaceVector omega{};
        omega[c] = 1.0;
        out[3 + c] = affineVector(part, rotationGradient(omega), {});
    }
    return out;
}

// max_i |(K r)_i| / (max_i K_ii · max_i |r_i|), and |rᵀ K r| / (max_i K_ii · |r|²).
std::pair<double, double> rigidResidual(const mh::Matrix& k, const mh::Vector& r)
{
    double diagonal = 0.0;
    for (std::size_t i = 0; i < k.rows(); ++i)
        diagonal = std::max(diagonal, k(i, i));
    const mh::Vector kr = k * r;
    return {maxAbsOf(kr) / (diagonal * maxAbsOf(r)), std::fabs(mh::dot(r, kr)) / (diagonal * mh::dot(r, r))};
}

// The input of a part without the check of its elements (the dense analysis builds them).
mh::ShellPartInput quietInput(mh::ShellPartInput input)
{
    input.checkElements = false;
    return input;
}

// The resultants of the centre of every shell of an analysis for the unknowns u.
std::vector<std::array<double, 8>> centreResultants(const mh::ShellPart& part, const DenseAnalysis& analysis,
                                                    const mh::Vector& u)
{
    std::vector<std::array<double, 8>> out;
    for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
    {
        const mh::ShellElementData& el = analysis.shell(ie);
        const mh::Vector x = el.local(u);
        out.push_back(part.recover(el, x, nullptr, {}, {}).centre.state.resultants);
    }
    return out;
}

// e_I · T · e_J of a symmetric 2 × 2 tensor of the plane z = 0 in the frame (e1, e2): the components (11, 22, 12).
std::array<double, 3> inFrame(const mh::SurfaceFrame& frame, const std::array<std::array<double, 2>, 2>& t)
{
    const auto component = [&](std::size_t i, std::size_t j)
    {
        double v = 0.0;
        for (std::size_t a = 0; a < 2; ++a)
            for (std::size_t b = 0; b < 2; ++b)
                v += frame[i][a] * t[a][b] * frame[j][b];
        return v;
    };
    return {component(0, 0), component(1, 1), component(0, 1)};
}

} // namespace

CHECK_CASE("shellpart: numbering, entities and signs of a mesh")
{
    for (const bool triangles : {false, true})
    {
        const mh::ShellPart part(quietInput(shellInput(squarePlate(), isotropicSection(2.1e11, 0.3, 0.01), triangles)));
        runner.expect(part.usable() && part.problems().empty(), "the plate is read");
        const std::size_t edges = triangles ? 16 : 12;
        const std::size_t interior = triangles ? 17 : 51;
        const std::size_t shells = triangles ? 8 : 4;
        runner.expect(part.topology().edges.size() == edges, "edges of the plate");
        runner.expect(part.entities().size() == edges + shells, "entities: the edges then the interiors");
        runner.expect(part.edgeDofCount(0) == 17, "17 DOFs per edge at k = 4");
        runner.expect(part.unknownCount() == 54 + 17 * edges + interior * shells, "unknowns of the plate");
        // every unknown of an entity is reached, and describe() inverts the numbering
        std::vector<int> reached(part.unknownCount(), 0);
        for (std::size_t ie = 0; ie < shells; ++ie)
            for (const std::size_t d : part.localUnknowns(ie))
                ++reached[d];
        bool allReached = true;
        for (std::size_t u = part.firstEntityUnknown(); u < part.unknownCount(); ++u)
        {
            const mh::ShellPartUnknown what = part.describe(u);
            allReached = allReached && reached[u] > 0 && what.kind == mh::ShellPartUnknown::Kind::EntityDof &&
                         part.entities()[what.index].firstUnknown + what.component == u;
        }
        runner.expect(allReached, "every entity DOF is a local unknown of a shell and describe() inverts it");
        // the edge shear: +1 from the first shell of the edge, −1 from the second (times (−1)^l on a reversed edge)
        const mh::ShellReferenceElement& re = part.referenceElement(part.elementAtPoints(0));
        bool joints = true;
        for (std::size_t e = 0; e < part.topology().edges.size(); ++e)
        {
            const std::vector<std::size_t>& owners = part.topology().edgeElements[e];
            if (owners.size() != 2)
                continue;
            const std::size_t unknown = part.entities()[e].firstUnknown + part.edgeShearDof(e, 0);
            double sum = 0.0;
            for (const std::size_t ie : owners)
                for (std::size_t i = 0; i < re.hcurlCount; ++i)
                    if (part.localUnknowns(ie)[3 * re.h1Count + i] == unknown)
                        sum += part.localSigns(ie)[3 * re.h1Count + i];
            joints = joints && sum == 0.0;
        }
        runner.expect(joints, "the shear of an edge of two shells has opposite signs in them");
    }
}

CHECK_CASE("shellpart: constant membrane state of a plate held by its boundary")
{
    const double e = 2.1e11;
    const double nu = 0.3;
    const double t = 0.01;
    // u = A x: strains e11 = 1e-3, e22 = −5e-4, γ12 = −1e-4, and a rotation about z
    const std::array<mh::SpaceVector, 3> a{{{1e-3, 2e-4, 0.0}, {-3e-4, -5e-4, 0.0}, {0.0, 0.0, 0.0}}};
    const mh::SpaceVector b{1e-4, -2e-4, 0.0};
    for (const bool triangles : {false, true})
    {
        const MappedMesh mesh = squarePlate();
        const std::vector<mh::ShellGridSupport> supports =
            gridSupports(SQUARE_BOUNDARY, {0, 1, 2},
                         [&](std::size_t g, std::size_t c)
                         {
                             const mh::SpaceVector& x = mesh.grids[g];
                             return b[c] + a[c][0] * x[0] + a[c][1] * x[1];
                         });
        mh::ShellPartInput input = quietInput(shellInput(mesh, isotropicSection(e, nu, t), triangles));
        input.heldTranslations = heldTranslationsOf(supports);
        const mh::ShellPart part(input);
        DenseAnalysis analysis(part);
        mh::Vector u = analysis.solve(supports);
        analysis.fillRotations(u);
        // every unknown is that of the affine field: grids, edges, interiors, the mean rotation at the grids
        const mh::Vector exact = affineVector(part, a, b);
        double error = 0.0;
        for (std::size_t i = 0; i < u.size(); ++i)
            error = std::max(error, std::fabs(u[i] - exact[i]));
        runner.expect(error <= 1e-9 * maxAbsOf(exact), std::string(triangles ? "triangles" : "quadrilaterals") +
                                                           ": the solution is the affine field, error " +
                                                           sci(error / maxAbsOf(exact)));
        // N = A ε in the local frame of each element
        const double q = e * t / (1.0 - nu * nu);
        const double exx = a[0][0];
        const double eyy = a[1][1];
        const double gxy = a[0][1] + a[1][0];
        const std::array<std::array<double, 2>, 2> n{
            {{q * (exx + nu * eyy), q * (1.0 - nu) / 2.0 * gxy}, {q * (1.0 - nu) / 2.0 * gxy, q * (eyy + nu * exx)}}};
        for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
        {
            const mh::ShellElementData& el = analysis.shell(ie);
            const mh::ShellPointState state = part.recover(el, el.local(u), nullptr, {}, {}).centre.state;
            const std::array<double, 3> expected = inFrame(state.frame, n);
            for (std::size_t c = 0; c < 3; ++c)
                runner.expect(std::fabs(state.resultants[c] - expected[c]) <= 1e-9 * std::fabs(n[0][0]),
                              "membrane force " + std::to_string(c) + " of element " + std::to_string(ie));
        }
    }
}

CHECK_CASE("shellpart: constant bending state of a plate held by its boundary")
{
    const double e = 2.1e11;
    const double nu = 0.3;
    const double t = 0.01;
    // w = (kx x² + ky y²) / 2 + kxy x y, the rotation θ = (∂w/∂y, −∂w/∂x, 0)
    const double kx = 2e-3;
    const double ky = -1e-3;
    const double kxy = 5e-4;
    const auto w = [&](const mh::SpaceVector& x)
    { return 0.5 * (kx * x[0] * x[0] + ky * x[1] * x[1]) + kxy * x[0] * x[1]; };
    const auto theta = [&](const mh::SpaceVector& x)
    { return mh::SpaceVector{ky * x[1] + kxy * x[0], -(kx * x[0] + kxy * x[1]), 0.0}; };
    struct Variant
    {
        const char* name;
        bool triangles;
        bool kirchhoff;
    };
    for (const Variant& variant : {Variant{"quadrilaterals", false, false}, Variant{"triangles", true, false},
                                   Variant{"Kirchhoff quadrilaterals", false, true}})
    {
        const MappedMesh mesh = squarePlate();
        const std::vector<mh::ShellGridSupport> supports =
            gridSupports(SQUARE_BOUNDARY, {0, 1, 2, 3, 4, 5},
                         [&](std::size_t g, std::size_t c)
                         {
                             const mh::SpaceVector& x = mesh.grids[g];
                             return c < 2 ? 0.0 : c == 2 ? w(x) : theta(x)[c - 3];
                         });
        mh::Section section = isotropicSection(e, nu, t);
        if (variant.kirchhoff)
            section.h.reset();
        mh::ShellPartInput input = quietInput(shellInput(mesh, section, variant.triangles));
        input.heldTranslations = heldTranslationsOf(supports);
        const mh::ShellPart part(input);
        DenseAnalysis analysis(part);
        mh::Vector u = analysis.solve(supports);
        analysis.fillRotations(u);
        const mh::SpaceVector centre = mesh.grids[4];
        const double scale = std::fabs(kx);
        runner.expect(std::fabs(u[6 * 4 + 2] - w(centre)) <= 1e-9 * scale,
                      std::string(variant.name) + ": deflection of the centre " + std::to_string(u[6 * 4 + 2]));
        for (std::size_t c = 0; c < 3; ++c)
            runner.expect(std::fabs(u[6 * 4 + 3 + c] - theta(centre)[c]) <= 1e-9 * scale,
                          std::string(variant.name) + ": mean rotation of the centre, component " + std::to_string(c));
        // M = D κ, κ = −ν · ∂∂u = −∂∂w (ν = +z), the shear force zero
        const double d = e * t * t * t / (12.0 * (1.0 - nu * nu));
        const std::array<std::array<double, 2>, 2> curvature{{{-kx, -kxy}, {-kxy, -ky}}};
        const double trace = curvature[0][0] + curvature[1][1];
        std::array<std::array<double, 2>, 2> moment{};
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j)
                moment[i][j] = d * ((1.0 - nu) * curvature[i][j] + (i == j ? nu * trace : 0.0));
        for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
        {
            const mh::ShellElementData& el = analysis.shell(ie);
            const mh::ShellRecovery results = part.recover(el, el.local(u), nullptr, {}, {});
            for (const mh::ShellRecoveryPoint* point : {&results.centre, &results.corners[0]})
            {
                const std::array<double, 3> expected = inFrame(point->state.frame, moment);
                for (std::size_t c = 0; c < 3; ++c)
                    runner.expect(std::fabs(point->state.resultants[3 + c] - expected[c]) <= 1e-9 * d * scale,
                                  std::string(variant.name) + ": moment " + std::to_string(c) + " of element " +
                                      std::to_string(ie));
                // no shear: the strain γ = Q / H vanishes to the round-off of the rotations (κ L, L = 1); H = 5/6 G t
                // magnifies it in Q by H L² / D ≈ 10⁵ (the Kirchhoff section holds γ = 0: Q is zero)
                const double shearStiffness = 5.0 / 6.0 * e / (2.0 * (1.0 + nu)) * t;
                for (std::size_t c = 0; c < 2; ++c)
                    runner.expect(std::fabs(point->state.resultants[6 + c]) / shearStiffness <= 1e-12 * scale,
                                  std::string(variant.name) + ": no shear strain " +
                                      sci(point->state.resultants[6 + c] / shearStiffness));
            }
        }
    }
}

CHECK_CASE("shellpart: rigid motions of a cylinder panel read from the mesh have no energy")
{
    for (const bool triangles : {false, true})
    {
        const mh::ShellPart part(quietInput(shellInput(cylinderPanel(), isotropicSection(7e10, 0.3, 0.01), triangles)));
        DenseAnalysis analysis(part);
        const std::array<mh::Vector, 6> rigid = rigidVectors(part);
        for (std::size_t m = 0; m < 6; ++m)
        {
            const auto [force, energy] = rigidResidual(analysis.stiffness(), rigid[m]);
            runner.expect(force < 1e-10 && energy < 1e-12, std::string(triangles ? "triangles" : "quadrilaterals") +
                                                               ", rigid motion " + std::to_string(m) + ": K r " +
                                                               sci(force) + ", energy " + sci(energy));
        }
    }
}

CHECK_CASE("shellpart: mean corner rotation of a rigid rotation is the rotation")
{
    for (const bool triangles : {false, true})
    {
        const mh::ShellPart part(quietInput(shellInput(cylinderPanel(), isotropicSection(7e10, 0.3, 0.01), triangles)));
        const mh::SpaceVector omega{0.3, -0.5, 0.7};
        const mh::Vector r = affineVector(part, rotationGradient(omega), {0.1, 0.2, 0.3});
        double error = 0.0;
        for (std::size_t g = 0; g < part.gridCount(); ++g)
            for (std::size_t c = 0; c < 3; ++c)
            {
                double v = 0.0;
                for (const mh::ShellPartTerm& term : part.meanRotationRow(g, c))
                    v += term.coefficient * r[term.unknown];
                error = std::max(error, std::fabs(v - omega[c]));
            }
        runner.expect(error < 1e-12,
                      std::string(triangles ? "triangles" : "quadrilaterals") + ": mean rotation error " + sci(error));
    }
}

CHECK_CASE("shellpart: clamped cantilever strip against the beam theory")
{
    // a strip 1 × 0.1, thickness 0.01, ν = 0, clamped at x = 0, a force P across it at its end
    const double length = 1.0;
    const double width = 0.1;
    const double t = 0.01;
    const double e = 7e10;
    const double force = 1.0;
    const MappedMesh mesh =
        mappedMesh(4, 1, [&](double s, double y) { return mh::SpaceVector{length * s, width * (y - 0.5), 0.0}; });
    const std::vector<mh::ShellGridSupport> supports = gridSupports({0, 1}, {0, 1, 2, 3, 4, 5});
    mh::ShellPartInput input = quietInput(shellInput(mesh, isotropicSection(e, 0.0, t), false));
    input.heldTranslations = heldTranslationsOf(supports);
    const mh::ShellPart part(input);
    DenseAnalysis analysis(part);
    analysis.addGridForce(8, 2, 0.5 * force);
    analysis.addGridForce(9, 2, 0.5 * force);
    const mh::Vector u = analysis.solve(supports);
    const double tip = 0.5 * (u[6 * 8 + 2] + u[6 * 9 + 2]);
    const double inertia = width * t * t * t / 12.0;
    // w = P L³ / (3 E I) (S. P. Timoshenko, J. M. Gere, Mechanics of Materials, 1972)
    const double beam = force * length * length * length / (3.0 * e * inertia);
    runner.expect(std::fabs(tip / beam - 1.0) < 1e-2,
                  "tip deflection over the beam theory " + std::to_string(tip / beam));
}

CHECK_CASE("shellpart: rigid motions with lines on the edges of the shells")
{
    // a flat strip stiffened on its centre line, and the cylinder panel with an eccentric bar on its middle arc
    const double e = 7e10;
    const mh::LineSection bar = mh::LineSection::bar(e, e / 2.6, 2e-4, 4e-8, 1e-8, 0.0, 5e-9, 2700.0);
    struct Model
    {
        const char* name;
        MappedMesh mesh;
        std::vector<std::array<std::size_t, 2>> lines;
    };
    const Model models[] = {
        {"stiffened strip",
         mappedMesh(4, 2, [](double s, double y) { return mh::SpaceVector{2.0 * s, 0.4 * (y - 0.5), 0.0}; }),
         {{1, 4}, {4, 7}, {7, 10}, {10, 13}}},
        {"stiffened cylinder panel", cylinderPanel(), {{3, 4}, {5, 4}}},
    };
    for (const Model& model : models)
    {
        mh::ShellPartInput input = quietInput(shellInput(model.mesh, isotropicSection(e, 0.3, 0.01), false));
        std::int32_t id = 101;
        for (const auto& [a, b] : model.lines)
            input.lines.push_back({id++, mh::LineElement::between(a, b, 0).eccentric(0.05, 0.02), bar});
        const mh::ShellPart part(input);
        DenseAnalysis analysis(part);
        const std::array<mh::Vector, 6> rigid = rigidVectors(part);
        for (std::size_t m = 0; m < 6; ++m)
        {
            const auto [force, energy] = rigidResidual(analysis.stiffness(), rigid[m]);
            runner.expect(force < 1e-10 && energy < 1e-12,
                          std::string(model.name) + ", rigid motion " + std::to_string(m) + ": K r " + sci(force));
        }
    }
}

CHECK_CASE("shellpart: renumbering the grids leaves the results unchanged")
{
    // a thick panel (R / t = 20): the round-off of the dense solution grows with the conditioning of a thin one
    const MappedMesh mesh = cylinderPanel();
    const mh::Section section = isotropicSection(7e10, 0.3, 0.05);
    struct Renumbered
    {
        mh::Vector u;
        std::vector<std::array<double, 8>> resultants;
        std::vector<mh::SpaceVector> points;
    };
    const auto solveWith = [&](const std::vector<std::int32_t>& ids, std::vector<std::size_t>& indexOf)
    {
        // the grids by increasing ID
        std::vector<std::size_t> order(mesh.grids.size());
        for (std::size_t g = 0; g < order.size(); ++g)
            order[g] = g;
        std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return ids[x] < ids[y]; });
        indexOf.assign(order.size(), 0);
        for (std::size_t i = 0; i < order.size(); ++i)
            indexOf[order[i]] = i;
        MappedMesh renumbered;
        for (const std::size_t g : order)
            renumbered.grids.push_back(mesh.grids[g]);
        for (const auto& q : mesh.quads)
            renumbered.quads.push_back({indexOf[q[0]], indexOf[q[1]], indexOf[q[2]], indexOf[q[3]]});
        std::vector<std::int32_t> sortedIds;
        for (const std::size_t g : order)
            sortedIds.push_back(ids[g]);
        const std::vector<mh::ShellGridSupport> supports =
            gridSupports({indexOf[0], indexOf[1], indexOf[2]}, {0, 1, 2, 3, 4, 5});
        mh::ShellPartInput input = quietInput(shellInput(renumbered, section, false, sortedIds));
        input.heldTranslations = heldTranslationsOf(supports);
        const mh::ShellPart part(input);
        DenseAnalysis analysis(part);
        analysis.addGridForce(indexOf[8], 1, 30.0);
        analysis.addGridForce(indexOf[8], 2, -100.0);
        analysis.addGridForce(indexOf[7], 2, -50.0);
        for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
            analysis.addShellLoad(ie, part.pressureLoad(analysis.shell(ie), {500.0, 600.0, 700.0, 800.0}));
        mh::Vector u = analysis.solve(supports);
        analysis.fillRotations(u);
        // the surface of the elements at a few points of their reference domain (their corners keep their order)
        std::vector<mh::SpaceVector> points;
        for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
            for (const double xi : {-0.7, 0.1, 0.9})
                for (const double eta : {-0.4, 0.6})
                    points.push_back(mh::geometryAt(part.referenceElement(analysis.shell(ie)),
                                                    analysis.shell(ie).definition(), xi, eta)
                                         .x);
        return Renumbered{u, centreResultants(part, analysis, u), points};
    };
    std::vector<std::size_t> first;
    std::vector<std::size_t> second;
    const Renumbered one = solveWith({1, 2, 3, 4, 5, 6, 7, 8, 9}, first);
    const Renumbered two = solveWith({50, 7, 31, 2, 90, 14, 3, 66, 21}, second);
    const mh::Vector& u1 = one.u;
    const mh::Vector& u2 = two.u;
    const std::vector<std::array<double, 8>>& r1 = one.resultants;
    const std::vector<std::array<double, 8>>& r2 = two.resultants;
    // the geometry read from the mesh does not depend on the numbering
    double geometryError = 0.0;
    for (std::size_t i = 0; i < one.points.size(); ++i)
        geometryError = std::max(geometryError, mh::norm3(mh::sub3(one.points[i], two.points[i])));
    runner.expect(geometryError < 1e-14, "surface of the elements, error " + sci(geometryError));
    double scale = 0.0;
    double error = 0.0;
    for (std::size_t g = 0; g < mesh.grids.size(); ++g)
        for (std::size_t c = 0; c < 6; ++c)
        {
            scale = std::max(scale, std::fabs(u1[6 * first[g] + c]));
            error = std::max(error, std::fabs(u1[6 * first[g] + c] - u2[6 * second[g] + c]));
        }
    runner.expect(error <= 1e-10 * scale, "grid displacements and rotations, error " + sci(error / scale));
    double resultantScale = 0.0;
    double resultantError = 0.0;
    for (std::size_t ie = 0; ie < r1.size(); ++ie)
        for (std::size_t c = 0; c < 8; ++c)
        {
            resultantScale = std::max(resultantScale, std::fabs(r1[ie][c]));
            resultantError = std::max(resultantError, std::fabs(r1[ie][c] - r2[ie][c]));
        }
    runner.expect(resultantError <= 1e-10 * resultantScale,
                  "resultants, error " + sci(resultantError / resultantScale));
}

CHECK_CASE("shellpart: mid-side grids fix the parabola of their edge")
{
    // a flat square of one quadrilateral whose edge 0 bows out of its plane through its mid-side grid, and whose edge 1
    // has its mid-side grid on its chord
    mh::ShellPartInput input;
    input.grids = {{{0.0, 0.0, 0.0}, 1}, {{1.0, 0.0, 0.0}, 2}, {{1.0, 1.0, 0.0}, 3},
                   {{0.0, 1.0, 0.0}, 4}, {{0.5, 0.0, 0.1}, 5}, {{1.0, 0.5, 0.0}, 6}};
    mh::ShellPartShell shell;
    shell.elementId = 1;
    shell.corners = mh::ShellCorners::quadrilateral(0, 1, 2, 3);
    shell.midside = {4, 5, -1, -1};
    shell.section = isotropicSection(7e10, 0.3, 0.01);
    input.shells.push_back(shell);
    input.checkElements = false;
    const mh::ShellPart part(input);
    runner.expect(part.usable() && part.problems().empty(), "the element is read");
    const mh::ShellElementData el = part.element(0);
    const mh::ShellReferenceElement& re = part.referenceElement(el);
    for (std::size_t k = 0; k < 2; ++k)
    {
        const mh::ReferencePoint middle = mh::referenceEdgePointAndDirection(4, k, 0.5).first;
        const mh::PointGeometry g = mh::geometryAt(re, el.definition(), middle[0], middle[1]);
        const mh::SpaceVector& xm = input.grids[4 + k].position;
        runner.expect(mh::norm3(mh::sub3(g.x, xm)) < 1e-12,
                      "edge " + std::to_string(k) + " passes through its mid-side grid");
        // a parabola: its point at τ = 1/4 is (3/4)² X_a + 2 (3/4)(1/4) C + (1/4)² X_b, C = 2 X_m − (X_a + X_b)/2
        const mh::ReferencePoint quarter = mh::referenceEdgePointAndDirection(4, k, 0.25).first;
        const mh::PointGeometry gq = mh::geometryAt(re, el.definition(), quarter[0], quarter[1]);
        const mh::SpaceVector& xa = input.grids[k].position;
        const mh::SpaceVector& xb = input.grids[k + 1].position;
        const mh::SpaceVector control = mh::sub3(mh::scl3(xm, 2.0), mh::scl3(mh::add3(xa, xb), 0.5));
        const mh::SpaceVector expected =
            mh::add3(mh::add3(mh::scl3(xa, 9.0 / 16.0), mh::scl3(control, 6.0 / 16.0)), mh::scl3(xb, 1.0 / 16.0));
        runner.expect(mh::norm3(mh::sub3(gq.x, expected)) < 1e-12, "edge " + std::to_string(k) + " is the parabola");
    }
    // the curved element keeps its rigid motions
    DenseAnalysis analysis(part);
    const std::array<mh::Vector, 6> rigid = rigidVectors(part);
    for (std::size_t m = 0; m < 6; ++m)
        runner.expect(rigidResidual(analysis.stiffness(), rigid[m]).first < 1e-10,
                      "rigid motion " + std::to_string(m) + " of the element with mid-side grids");
}

CHECK_CASE("shellpart: the loads of a rigid motion are their resultants")
{
    // the plate, a pressure given at the corners, a traction, and forces and couples on its edge x = 1
    mh::ShellPartInput input = quietInput(shellInput(squarePlate(), isotropicSection(2.1e11, 0.3, 0.01), false));
    input.shells[3].offset = 0.004;
    const mh::ShellPart part(input);
    const std::array<mh::Vector, 6> rigid = rigidVectors(part);
    const auto resultant = [&](std::size_t ie, const mh::Vector& local, std::size_t mode)
    {
        const std::span<const std::size_t> dofs = part.localUnknowns(ie);
        const std::span<const double> signs = part.localSigns(ie);
        double v = 0.0;
        for (std::size_t i = 0; i < local.size(); ++i)
            v += signs[i] * local[i] * rigid[mode][dofs[i]];
        return v;
    };
    // a pressure linear over the plate, p = 100 + 50 x + 20 y (corner values of each element): F_z = ∫ p = 135
    double fz = 0.0;
    double mx = 0.0;
    for (std::size_t ie = 0; ie < 4; ++ie)
    {
        const mh::ShellElementData el = part.element(ie);
        std::array<double, 4> corners{};
        for (std::size_t c = 0; c < 4; ++c)
        {
            const mh::SpaceVector& x = input.grids[el.nodes[c]].position;
            corners[c] = 100.0 + 50.0 * x[0] + 20.0 * x[1];
        }
        const mh::Vector load = part.pressureLoad(el, corners);
        fz += resultant(ie, load, 2);
        mx += resultant(ie, load, 3);
    }
    runner.expectNear(fz, 135.0, 1e-12, "resultant of the pressure");
    // M_x = ∫ y p = 50 + 25 / 2 + 20 / 3
    runner.expectNear(mx, 50.0 + 12.5 + 20.0 / 3.0, 1e-12, "moment of the pressure about x");
    // a uniform traction on the offset element: per unit area of its own surface (flat: the same area)
    const mh::Vector traction =
        part.areaForceLoad(part.element(3), [](const mh::ShellLoadPoint&) { return mh::SpaceVector{3.0, -4.0, 5.0}; });
    runner.expectNear(resultant(3, traction, 0), 0.75, 1e-12, "traction x");
    runner.expectNear(resultant(3, traction, 1), -1.0, 1e-12, "traction y");
    // forces and couples on the edge x = 1 (grids 6, 7, 8), f(s) = (10, 0, 100 + 50 s) and m = (3, 7, 0)
    const auto field = [](double s) { return std::array<double, 6>{10.0, 0.0, 100.0 + 50.0 * s, 3.0, 7.0, 0.0}; };
    std::array<double, 6> total{};
    for (const auto& [a, b] : FAR_EDGE)
    {
        const std::variant<mh::ShellEdgeLoad, std::string> load = part.edgeLoad(a, b, field);
        runner.expect(std::holds_alternative<mh::ShellEdgeLoad>(load), "the edge load is accepted");
        if (const mh::ShellEdgeLoad* l = std::get_if<mh::ShellEdgeLoad>(&load))
            for (std::size_t m = 0; m < 6; ++m)
                total[m] += resultant(l->shell, l->local, m);
    }
    // per segment of length 1/2 from y0: f_z = 100 + 50 (y − y0) / (1/2); X × f + m with X = (1, y, 0)
    double fzEdge = 0.0;
    double myz = 0.0;
    for (const double y0 : {0.0, 0.5})
    {
        fzEdge += 0.5 * (100.0 + 25.0);
        // ∫ y f_z dy over the segment: f_z = 100 + 100 (y − y0)
        const double y1 = y0 + 0.5;
        const double squares = (y1 * y1 - y0 * y0) / 2.0;
        const double cubes = (y1 * y1 * y1 - y0 * y0 * y0) / 3.0;
        myz += 100.0 * squares + 100.0 * (cubes - y0 * squares);
    }
    runner.expectNear(total[0], 10.0, 1e-12, "edge force x");
    runner.expectNear(total[2], fzEdge, 1e-12, "edge force z");
    runner.expectNear(total[3], myz + 3.0, 1e-12, "edge moment about x");
    runner.expectNear(total[4], -fzEdge + 7.0, 1e-12, "edge moment about y");
    runner.expectNear(total[5], -0.5 * 10.0, 1e-12, "edge moment about z");
    // a couple about the normal is refused
    const auto twisting = [](double) { return std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}; };
    runner.expect(std::holds_alternative<std::string>(part.edgeLoad(6, 7, twisting)),
                  "a couple about the normal is refused");
    runner.expect(std::holds_alternative<std::string>(part.edgeLoad(0, 4, field)),
                  "an edge that is not one of the shells");
}

CHECK_CASE("shellpart: problems are attached to their elements")
{
    mh::ShellPartInput input = shellInput(squarePlate(), isotropicSection(2.1e11, 0.3, 0.01), false);
    // a section without bending stiffness, and a line between two grids that no shell edge joins
    input.shells[2].section.d = {};
    const mh::LineSection rod = mh::LineSection::rod(7e10, 2.7e10, 1e-4, 1e-9, 2700.0);
    input.lines.push_back({7, mh::LineElement::between(0, 4, 0), rod});
    input.lines.push_back({8, mh::LineElement::between(0, 1, 0), rod});
    const mh::ShellPart part(input);
    runner.expect(part.usable(), "the mesh is read");
    bool section = false;
    bool line = false;
    for (const mh::ShellPartProblem& problem : part.problems())
    {
        section = section || (problem.subject == mh::ShellPartProblem::Subject::Shell && problem.index == 2 &&
                              problem.message.find("singular section") != std::string::npos);
        line = line || (problem.subject == mh::ShellPartProblem::Subject::Line && problem.index == 0);
    }
    runner.expect(section, "the shell without bending stiffness is refused");
    runner.expect(line, "the line off the edges is reported");
    runner.expect(part.problems().size() == 2, "nothing else is reported");
    runner.expect(part.lineSource(1).has_value() && !part.lineSource(0).has_value(), "the line on an edge is placed");
    // coincident corners: the mesh is not read
    mh::ShellPartInput degenerate = shellInput(squarePlate(), isotropicSection(2.1e11, 0.3, 0.01), false);
    degenerate.shells[1].corners.grids[2] = degenerate.shells[1].corners.grids[1];
    const mh::ShellPart refused(degenerate);
    runner.expect(!refused.usable() && refused.problems().size() == 1 && refused.problems()[0].index == 1,
                  "a shell of coincident corners is reported and the mesh not read");
}

CHECK_CASE("shellpart: geometry report and supports of a folded plate")
{
    // two squares folded at 90 degrees along a raw edge, held along the two edges of one end
    mh::ShellPartInput input;
    input.grids = {{{0.0, 0.0, 0.0}, 1}, {{0.0, 1.0, 0.0}, 2}, {{1.0, 0.0, 0.0}, 3},
                   {{1.0, 1.0, 0.0}, 4}, {{1.0, 0.0, 1.0}, 5}, {{1.0, 1.0, 1.0}, 6}};
    for (const auto& corners :
         {mh::ShellCorners::quadrilateral(0, 2, 3, 1), mh::ShellCorners::quadrilateral(2, 4, 5, 3)})
    {
        mh::ShellPartShell shell;
        shell.elementId = static_cast<std::int32_t>(input.shells.size() + 1);
        shell.corners = corners;
        shell.section = isotropicSection(7e10, 0.3, 0.01);
        input.shells.push_back(shell);
    }
    input.rawEdges.insert(mh::EdgeKey::of(2, 3));
    input.checkElements = false;
    const mh::ShellPart part(input);
    double kink = 0.0;
    for (const mh::EdgeGeometry& edge : part.geometryReport())
        if (edge.grids == mh::EdgeKey::of(2, 3))
        {
            kink = edge.kink;
            runner.expect(edge.raw && edge.elements == 2, "the fold is raw and has two shells");
        }
    runner.expectNear(kink, 0.5 * std::numbers::pi, 1e-9, "the fold is a right angle");
    // grids 0 and 1 hold their translations and the rotation about y: the edge holds its displacement, its rotation
    // (the tangent is y) and its shear (the conormal is x: not in the span, nothing)
    const std::vector<mh::ShellGridSupport> supports =
        gridSupports({0, 1}, {0, 1, 2, 4}, [](std::size_t, std::size_t c) { return c == 4 ? 0.01 : 0.0; });
    const mh::ShellSupports held = part.supportsOf(supports);
    const std::size_t edge = part.topology().indexOf(0, 1);
    const std::size_t first = part.entities()[part.edgeEntity(edge)].firstUnknown;
    std::size_t moments = 0;
    std::size_t rotations = 0;
    std::size_t shears = 0;
    double rotationValue = 0.0;
    for (const mh::ShellDerivedSupport& d : held.held)
    {
        if (d.unknown < first || d.unknown >= first + part.edgeDofCount(edge))
            continue;
        const std::size_t dof = d.unknown - first;
        if (dof < part.edgeShearDof(edge, 0))
            ++moments;
        else if (dof < part.edgeRotationDof(edge, 0))
            ++shears;
        else
        {
            ++rotations;
            if (dof == part.edgeRotationDof(edge, 0))
                rotationValue = d.value;
        }
    }
    runner.expect(moments == 9 && rotations == 4 && shears == 0, "the held DOFs of the end edge");
    // â_0 = ∫ W² θ · X_τ in the canonical direction: θ_y = 0.01 along the edge of length 1
    runner.expectNear(rotationValue, 0.01, 1e-12, "the held rotation about the edge");
    runner.expect(held.notes.size() == 1, "the rotation held at the grids is noted");
}

// ---- numbers of the reference implementation (computed by the reference crate at commit 9f55f0b, tdnns::Analysis on
// the same meshes, solve_static and fill_rotations)

CHECK_CASE("shellpart: reference numbers, plate")
{
    const MappedMesh mesh = squarePlate();
    const std::vector<mh::ShellGridSupport> supports = gridSupports({0, 1, 2}, {0, 1, 2, 3, 4, 5});
    mh::ShellPartInput input = quietInput(shellInput(mesh, isotropicSection(2.1e11, 0.3, 0.01, 7800.0, 1.2e-5), false));
    input.heldTranslations = heldTranslationsOf(supports);
    const mh::ShellPart part(input);
    DenseAnalysis analysis(part);
    const mh::TemperatureAtPosition temperatures = [](const mh::SpaceVector& x)
    { return std::array<double, 2>{50.0 + 20.0 * x[0], 100.0 + 30.0 * x[1]}; };
    // u[8][0 … 5] per case, then the resultants at the centre of element 3
    const double expectedU[4][6] = {
        {0.0, 0.0, 6.62014317864244740e-3, -3.21750050416591082e-4, -9.01029539476818839e-3, 0.0},
        {5.30259444307728674e-9, -9.05206558547180172e-9, 2.55647936712190033e-4, 1.87766118195890837e-4,
         -3.79501300112296367e-4, -1.29526679031855969e-8},
        {7.16018798844953743e-4, 4.30058665008089484e-4, -9.05171496138429139e-4, -6.29952126783245932e-4,
         1.59303981987676813e-3, 1.48406707064832372e-4},
        {4.72589720500832624e-9, -7.39080343606410340e-10, 2.09247494173147924e-3, -1.30690117955619990e-5,
         -3.11082043499896592e-3, -6.53610633420624161e-11}};
    const double expectedResultants[4][8] = {
        {0.0, 0.0, 0.0, -3.15798118875455849e1, 6.59928230539448002e0, -1.25541888526824974e0, 2.70744025553875076e2,
         -5.53646146299925035e-1},
        {5.82283797720755025e0, 1.48808039365289968e-1, -7.01417201209724261e0, -2.60242820176901679e0,
         -1.35549923679755868e0, -3.14693869924227521e0, 2.02724129223356755e1, 1.32832009740800387e1},
        {-6.09989217321356682e3, 1.42078250681017871e4, -3.74973128672991006e4, -7.04796163869202075e-1,
         -5.40517397403143196e0, 2.40712918564443434e0, -2.81227701589961221e0, 4.78599328082378861e0},
        {1.00139335361871122e1, -4.17829665101185910e-2, 9.73872787822880753e-2, -2.44306407855983814e1,
         -6.00194103075477603e-2, -2.68213749628373410e0, 1.42447264090463705e2, 8.48830616139043204e-1}};
    const auto edgeField = [](double s) { return std::array<double, 6>{10.0, 0.0, 100.0 + 50.0 * s, 3.0, 7.0, 0.0}; };
    for (std::size_t ic = 0; ic < 4; ++ic)
    {
        analysis.clearLoads();
        std::vector<std::optional<mh::ShellThermalLoad>> thermal(4);
        if (ic == 0)
            for (std::size_t ie = 0; ie < 4; ++ie)
                analysis.addShellLoad(ie, part.pressureLoad(analysis.shell(ie), {1000.0, 1000.0, 1000.0, 1000.0}));
        if (ic == 1)
        {
            analysis.addGridForce(8, 2, 10.0);
            analysis.addGridForce(7, 0, 5.0);
            analysis.addGridForce(7, 1, -3.0);
        }
        if (ic == 2)
            for (std::size_t ie = 0; ie < 4; ++ie)
            {
                thermal[ie] = part.thermalLoad(analysis.shell(ie), temperatures);
                analysis.addShellLoad(ie, thermal[ie]->forces);
            }
        if (ic == 3)
            for (const auto& [a, b] : FAR_EDGE)
            {
                const mh::ShellEdgeLoad load = std::get<mh::ShellEdgeLoad>(part.edgeLoad(a, b, edgeField));
                analysis.addShellLoad(load.shell, load.local);
            }
        mh::Vector u = analysis.solve(supports);
        analysis.fillRotations(u);
        double scale = 0.0;
        for (const double v : expectedU[ic])
            scale = std::max(scale, std::fabs(v));
        for (std::size_t c = 0; c < 6; ++c)
            runner.expect(
                relativeDifference(u[6 * 8 + c], expectedU[ic][c], scale) < 1e-9,
                "case " + std::to_string(ic) + ": u[8][" + std::to_string(c) + "] = " + std::to_string(u[6 * 8 + c]));
        const mh::ShellElementData& el = analysis.shell(3);
        const mh::Vector* momentTerm = thermal[3] ? &thermal[3]->momentTerm : nullptr;
        const std::array<double, 8> r =
            mh::sectionResultants(part.referenceElement(el), el, 0.0, 0.0, el.local(u), momentTerm,
                                  ic == 2 ? temperatures : mh::TemperatureAtPosition{});
        double resultantScale = 0.0;
        for (const double v : expectedResultants[ic])
            resultantScale = std::max(resultantScale, std::fabs(v));
        for (std::size_t c = 0; c < 8; ++c)
            runner.expect(
                relativeDifference(r[c], expectedResultants[ic][c], resultantScale) < 1e-9,
                "case " + std::to_string(ic) + ": resultant " + std::to_string(c) + " = " + std::to_string(r[c]));
    }
    // the clamped edge turned by 0.01 about itself: the plate turns rigidly, w = −0.01 x
    const std::vector<mh::ShellGridSupport> turned =
        gridSupports({0, 1, 2}, {0, 1, 2, 3, 4, 5}, [](std::size_t, std::size_t c) { return c == 4 ? 0.01 : 0.0; });
    analysis.clearLoads();
    mh::Vector u = analysis.solve(turned);
    analysis.fillRotations(u);
    runner.expect(relativeDifference(u[6 * 8 + 2], -1.00000000000153629e-2) < 1e-9, "turned: u[8][2]");
    runner.expect(relativeDifference(u[6 * 7 + 2], -1.00000000000331993e-2) < 1e-9, "turned: u[7][2]");
    runner.expect(relativeDifference(u[6 * 4 + 2], -5.00000000001042926e-3) < 1e-9, "turned: u[4][2]");
    runner.expect(relativeDifference(u[6 * 8 + 4], 1.00000000000382162e-2) < 1e-9, "turned: u[8][4]");
}

CHECK_CASE("shellpart: reference numbers, cylinder panel")
{
    // u[8][0 … 5] per case: forces, pressure, inertia; quadrilaterals then triangles
    const double expected[2][3][6] = {{{-7.34878099716084436e-5, -8.08233457780759587e-4, -4.53262227829771824e-3,
                                        -8.93561118887640539e-3, 3.55562153421785975e-3, -6.91981459344580534e-4},
                                       {8.54049459078067778e-5, 3.28476804646046045e-5, 2.64092469133071469e-3,
                                        4.61838584905001608e-4, -1.64482270143568362e-3, 4.49858380126390287e-6},
                                       {-4.64914657368202368e-5, -1.65742977235159534e-5, -1.43302037911922290e-3,
                                        -2.32457331780120319e-4, 8.95231903221537298e-4, -2.45731207478724950e-6}},
                                      {{-7.33596638936229921e-5, -8.06579129491937765e-4, -4.52246791588128017e-3,
                                        -8.88054893249031786e-3, 3.56261010625933024e-3, -6.94729169355397377e-4},
                                       {8.52423562015507667e-5, 3.19352194818090006e-5, 2.63150213205992764e-3,
                                        4.49274050230950308e-4, -1.64270471349012986e-3, 4.86236509099459161e-6},
                                       {-4.64028547500125704e-5, -1.60810715593905448e-5, -1.42792076655935034e-3,
                                        -2.26174435141006228e-4, 8.93958188033190508e-4, -2.62269363726931413e-6}}};
    for (const bool triangles : {false, true})
    {
        const std::vector<mh::ShellGridSupport> supports = gridSupports({0, 1, 2}, {0, 1, 2, 3, 4, 5});
        mh::ShellPartInput input =
            quietInput(shellInput(cylinderPanel(), isotropicSection(7e10, 0.3, 0.01, 2700.0), triangles));
        input.heldTranslations = heldTranslationsOf(supports);
        const mh::ShellPart part(input);
        DenseAnalysis analysis(part);
        for (std::size_t ic = 0; ic < 3; ++ic)
        {
            analysis.clearLoads();
            if (ic == 0)
            {
                analysis.addGridForce(8, 1, 30.0);
                analysis.addGridForce(8, 2, -100.0);
                analysis.addGridForce(7, 2, -50.0);
            }
            for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
            {
                if (ic == 1)
                    analysis.addShellLoad(ie, part.pressureLoad(analysis.shell(ie), {500.0, 500.0, 500.0, 500.0}));
                if (ic == 2)
                    analysis.addShellLoad(ie, part.inertiaLoad(analysis.shell(ie), {0.0, 0.0, -9.81}));
            }
            mh::Vector u = analysis.solve(supports);
            analysis.fillRotations(u);
            const double* e = expected[triangles ? 1 : 0][ic];
            double scale = 0.0;
            for (std::size_t c = 0; c < 6; ++c)
                scale = std::max(scale, std::fabs(e[c]));
            for (std::size_t c = 0; c < 6; ++c)
                runner.expect(relativeDifference(u[6 * 8 + c], e[c], scale) < 1e-9,
                              std::string(triangles ? "triangles" : "quadrilaterals") + ", case " + std::to_string(ic) +
                                  ": u[8][" + std::to_string(c) + "] = " + std::to_string(u[6 * 8 + c]));
        }
    }
}

namespace
{

// The stiffened strip of the reference probe: 2 × 0.4, 4 × 2 elements, an eccentric bar on its centre line, clamped
// at x = 0.
struct StiffenedStrip
{
    mh::ShellPartInput input;
    std::vector<mh::ShellGridSupport> supports = gridSupports({0, 1, 2}, {0, 1, 2, 3, 4, 5});
    double length = 2.0;
    // E A of the bar.
    double axialStiffness = 7e10 * 2e-4;

    StiffenedStrip()
    {
        const double e = 7e10;
        input = quietInput(shellInput(
            mappedMesh(4, 2, [](double s, double y) { return mh::SpaceVector{2.0 * s, 0.4 * (y - 0.5), 0.0}; }),
            isotropicSection(e, 0.3, 0.01, 2700.0, 2.3e-5), false));
        mh::LineSection bar = mh::LineSection::bar(e, e / 2.6, 2e-4, 4e-8, 1e-8, 0.0, 5e-9, 2700.0);
        bar.thermalExpansion = 2.3e-5;
        for (std::size_t i = 0; i < 4; ++i)
            input.lines.push_back({static_cast<std::int32_t>(101 + i),
                                   mh::LineElement::between(3 * i + 1, 3 * i + 4, 0).eccentric(0.05, 0.0), bar});
        input.heldTranslations = heldTranslationsOf(supports);
    }
};

} // namespace

CHECK_CASE("shellpart: reference numbers, stiffened strip")
{
    const StiffenedStrip strip;
    const mh::ShellPart part(strip.input);
    DenseAnalysis analysis(part);
    const std::size_t tip = 13;
    const mh::TemperatureAtPosition shellTemperatures = [](const mh::SpaceVector& x)
    { return std::array<double, 2>{30.0 + 10.0 * x[0], 0.0}; };
    const mh::LineTemperatureField lineTemperature = [](const mh::Vec3Of<double>& x) { return 30.0 + 10.0 * x[0]; };
    // u[13][0 … 5] per case: tip forces z and y, thermal, inertia; then the forces of line 1 at 0.3 of its chord
    const double expected[4][6] = {{1.25863467447678373e-7, -2.99284042486498191e-20, 6.92968975782571711e-5,
                                    -1.73229349004683206e-15, -5.19931014532923835e-5, 4.26589171417996006e-22},
                                   {2.59326795053815905e-21, 7.34252811516427233e-7, -3.11399753914681484e-20,
                                    -4.26880915098626022e-21, -3.83646794384121184e-20, 5.67396777443766299e-7},
                                   {1.85362560365316285e-3, -2.20833304529561890e-15, 4.97201889105684217e-4,
                                    -1.05794848981779400e-14, -2.38190961099217892e-4, -1.36220579290480503e-15},
                                   {-9.08775220349723459e-6, 2.93988961200436352e-18, -5.77226561961171562e-3,
                                    1.44783087898672283e-13, 3.84136117074054687e-3, 6.96479165907531052e-19}};
    const double expectedLine[4][6] = {{-2.33918252192479201e1, 9.36535850325845409e-1, -1.29723009392812148e-14,
                                        -1.35614120445467209e-13, 3.35062759157846819e-16, 9.82260049145191327e-2},
                                       {-2.59132157252301887e-13, 1.18417755151038943e-14, -2.18948938800590982e-4,
                                        -5.53380381317332104e-19, 2.55091600109746985e-4, 6.36405913695702573e-16},
                                       {3.42093112704468094e0, 1.92659583986476823e0, -6.54458220981502871e-13,
                                        -2.45601575755694349e-13, -5.37629038331588694e-13, 2.60735573011866006e-1},
                                       {1.75398256120628753e3, -1.40841248248920891e2, 9.46415594906107692e-13,
                                        1.04315887145998824e-11, -2.98325843476529630e-14, -7.34875642069391954e0}};
    const mh::SpaceVector acceleration{0.0, 0.0, -9.81};
    for (std::size_t ic = 0; ic < 4; ++ic)
    {
        analysis.clearLoads();
        if (ic == 0)
            analysis.addGridForce(tip, 2, 1.0);
        if (ic == 1)
            analysis.addGridForce(tip, 1, 1.0);
        for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
        {
            if (ic == 2)
                analysis.addShellLoad(ie, part.thermalLoad(analysis.shell(ie), shellTemperatures).forces);
            if (ic == 3)
                analysis.addShellLoad(ie, part.inertiaLoad(analysis.shell(ie), acceleration));
        }
        for (std::size_t il = 0; il < part.lineCount(); ++il)
        {
            if (ic == 2)
                analysis.addLineLoad(
                    il, mh::lineThermalLoad(analysis.line(il), part.input().lines[il].section, lineTemperature));
            if (ic == 3)
                analysis.addLineLoad(il, mh::lineInertiaLoad(analysis.line(il), acceleration));
        }
        const mh::Vector u = analysis.solve(strip.supports);
        double scale = 0.0;
        for (const double v : expected[ic])
            scale = std::max(scale, std::fabs(v));
        for (std::size_t c = 0; c < 6; ++c)
            runner.expect(relativeDifference(u[6 * tip + c], expected[ic][c], scale) < 1e-9,
                          "case " + std::to_string(ic) + ": u[13][" + std::to_string(c) +
                              "] = " + sci(u[6 * tip + c] - expected[ic][c]));
        const mh::LineData& line = analysis.line(1);
        const std::variant<mh::LineState, std::string> state =
            part.lineState(1, line, 0.3, line.local(u), ic == 2 ? lineTemperature : mh::LineTemperatureField{});
        runner.expect(std::holds_alternative<mh::LineState>(state), "state of the line");
        if (const mh::LineState* s = std::get_if<mh::LineState>(&state))
        {
            // the forces are differences of strains of the order of u / L (the thermal ones less their free part):
            // their round-off is that of E A u / L, the scale of the comparison when it is the larger
            double lineScale = strip.axialStiffness * scale / strip.length;
            for (const double v : expectedLine[ic])
                lineScale = std::max(lineScale, std::fabs(v));
            for (std::size_t c = 0; c < 6; ++c)
                runner.expect(relativeDifference(s->forces[c], expectedLine[ic][c], lineScale) < 1e-9,
                              "case " + std::to_string(ic) + ": line force " + std::to_string(c) + " = " +
                                  sci(s->forces[c] - expectedLine[ic][c]));
        }
    }
}

CHECK_CASE("shellpart: reference numbers, buckling of the stiffened strip")
{
    const StiffenedStrip strip;
    const mh::ShellPart part(strip.input);
    DenseAnalysis analysis(part);
    // a unit compression at the tip, then the geometric stiffness of its solution
    analysis.addGridForce(13, 0, -1.0);
    const mh::Vector u = analysis.solve(strip.supports);
    mh::Matrix kg(analysis.size(), analysis.size());
    for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
    {
        const mh::ShellElementData& el = analysis.shell(ie);
        analysis.scatterInto(kg, el.dofs, el.signs,
                             part.geometricStiffness(el, el.local(u), nullptr, nullptr, nullptr));
    }
    for (std::size_t il = 0; il < part.lineCount(); ++il)
    {
        const mh::LineData& line = analysis.line(il);
        analysis.scatterInto(kg, line.dofs, line.signs, mh::lineGeometricStiffness(line, line.local(u), {}));
    }
    // the lowest load factor of K φ = λ (−K_g) φ on the free unknowns, by inverse iteration
    const std::vector<std::size_t>& free = analysis.freeUnknowns();
    const std::size_t n = free.size();
    mh::Matrix a(n, n);
    mh::Matrix b(n, n);
    for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q)
        {
            a(p, q) = analysis.stiffness()(free[p], free[q]);
            b(p, q) = -kg(free[p], free[q]);
        }
    const mh::LuDecomposition lu(a);
    mh::Vector x(n, 1.0);
    double lambda = 0.0;
    for (int iteration = 0; iteration < 60; ++iteration)
    {
        mh::Vector y = *lu.solve(b * x);
        const double norm = mh::euclideanNorm(y);
        for (double& v : y)
            v /= norm;
        x = std::move(y);
        lambda = mh::dot(x, a * x) / mh::dot(x, b * x);
    }
    // computed by the reference crate at commit 9f55f0b (Analysis::buckling)
    runner.expect(relativeDifference(lambda, 2.36800478163335611e4) < 1e-9,
                  "lowest load factor " + std::to_string(lambda));
}

} // namespace oos::checks
