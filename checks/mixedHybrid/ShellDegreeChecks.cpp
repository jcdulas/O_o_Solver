#include "CheckRunner.hpp"
#include "ShellPartCheckSupport.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/mixedHybrid/model/ShellTraceDefect.hpp"
#include "oos/elements/mixedHybrid/shell/ShellDegreeChoice.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mh = oos::elements::mixedHybrid;

namespace
{

using namespace oos::checks::shellpart;

// A square plate of 2 × 2 cells whose centre grid (4) is moved off the middle in its plane: no cell is an affine
// element, and its interior edges are shared by cells of different shapes.
MappedMesh distortedPlate()
{
    MappedMesh mesh = mappedMesh(2, 2, [](double s, double t) { return mh::SpaceVector{s, t, 0.0}; });
    mesh.grids[4] = {0.56, 0.43, 0.0};
    return mesh;
}

const std::vector<std::size_t> BOUNDARY{0, 1, 2, 3, 5, 6, 7, 8};

// The degrees given to the shells of a check: one per quadrilateral, or the first of its two triangles the lowest.
std::vector<std::size_t> degreePattern(std::size_t shells)
{
    const std::array<std::size_t, 4> pattern{2, 4, 3, 2};
    std::vector<std::size_t> degrees;
    for (std::size_t ie = 0; ie < shells; ++ie)
        degrees.push_back(pattern[ie % pattern.size()]);
    return degrees;
}

mh::ShellPartInput mixedInput(const MappedMesh& mesh, const mh::Section& section, bool triangles)
{
    mh::ShellPartInput input = shellInput(mesh, section, triangles);
    input.checkElements = false;
    input.shellDegrees = degreePattern(input.shells.size());
    // straight edges: the reading would otherwise round the mesh lines through the moved grid
    for (std::size_t ie = 0; ie < input.shells.size(); ++ie)
        input.rawElements.push_back(ie);
    return input;
}

std::string shapeName(bool triangles)
{
    return triangles ? "triangles" : "quadrilaterals";
}

// The constant state: u = b + A x in the plane, the resultants N = A ε of every shell.
void checkMembrane(oos::checks::CheckRunner& runner, bool triangles)
{
    const double e = 2.1e11;
    const double nu = 0.3;
    const double t = 0.01;
    const std::array<mh::SpaceVector, 3> a{{{1e-3, 2e-4, 0.0}, {-3e-4, -5e-4, 0.0}, {0.0, 0.0, 0.0}}};
    const mh::SpaceVector b{1e-4, -2e-4, 0.0};
    const MappedMesh mesh = distortedPlate();
    const std::vector<mh::ShellGridSupport> supports =
        gridSupports(BOUNDARY, {0, 1, 2},
                     [&](std::size_t g, std::size_t c)
                     {
                         const mh::SpaceVector& x = mesh.grids[g];
                         return b[c] + a[c][0] * x[0] + a[c][1] * x[1];
                     });
    mh::ShellPartInput input = mixedInput(mesh, isotropicSection(e, nu, t), triangles);
    input.heldTranslations = heldTranslationsOf(supports);
    const mh::ShellPart part(input);
    DenseAnalysis analysis(part);
    mh::Vector u = analysis.solve(supports);
    analysis.fillRotations(u);
    const mh::Vector exact = affineVector(part, a, b);
    double error = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i)
        error = std::max(error, std::fabs(u[i] - exact[i]));
    runner.expect(error <= 1e-9 * maxAbsOf(exact), shapeName(triangles) + ": the solution is the affine field, error " +
                                                       sci(error / maxAbsOf(exact)));
    const double q = e * t / (1.0 - nu * nu);
    const std::array<std::array<double, 2>, 2> n{
        {{q * (a[0][0] + nu * a[1][1]), q * (1.0 - nu) / 2.0 * (a[0][1] + a[1][0])},
         {q * (1.0 - nu) / 2.0 * (a[0][1] + a[1][0]), q * (a[1][1] + nu * a[0][0])}}};
    double worst = 0.0;
    for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
    {
        const mh::ShellElementData& el = analysis.shell(ie);
        const mh::ShellPointState state = part.recover(el, el.local(u), nullptr, {}, {}).centre.state;
        const auto component = [&](std::size_t i, std::size_t j)
        {
            double v = 0.0;
            for (std::size_t x = 0; x < 2; ++x)
                for (std::size_t y = 0; y < 2; ++y)
                    v += state.frame[i][x] * n[x][y] * state.frame[j][y];
            return v;
        };
        const std::array<double, 3> expected{component(0, 0), component(1, 1), component(0, 1)};
        for (std::size_t c = 0; c < 3; ++c)
            worst = std::max(worst, std::fabs(state.resultants[c] - expected[c]) / std::fabs(n[0][0]));
    }
    runner.expect(worst <= 1e-9, shapeName(triangles) + ": membrane forces, error " + sci(worst));
}

// The constant curvature: w = (kx x² + ky y²) / 2 + kxy x y held by its boundary, the moments M = D κ.
void checkBending(oos::checks::CheckRunner& runner, bool triangles)
{
    const double e = 2.1e11;
    const double nu = 0.3;
    const double t = 0.01;
    const double kx = 2e-3;
    const double ky = -1e-3;
    const double kxy = 5e-4;
    const auto w = [&](const mh::SpaceVector& x)
    { return 0.5 * (kx * x[0] * x[0] + ky * x[1] * x[1]) + kxy * x[0] * x[1]; };
    const auto theta = [&](const mh::SpaceVector& x)
    { return mh::SpaceVector{ky * x[1] + kxy * x[0], -(kx * x[0] + kxy * x[1]), 0.0}; };
    const MappedMesh mesh = distortedPlate();
    const std::vector<mh::ShellGridSupport> supports =
        gridSupports(BOUNDARY, {0, 1, 2, 3, 4, 5},
                     [&](std::size_t g, std::size_t c)
                     {
                         const mh::SpaceVector& x = mesh.grids[g];
                         return c < 2 ? 0.0 : c == 2 ? w(x) : theta(x)[c - 3];
                     });
    mh::ShellPartInput input = mixedInput(mesh, isotropicSection(e, nu, t), triangles);
    input.heldTranslations = heldTranslationsOf(supports);
    const mh::ShellPart part(input);
    DenseAnalysis analysis(part);
    mh::Vector u = analysis.solve(supports);
    analysis.fillRotations(u);
    const mh::SpaceVector centre = mesh.grids[4];
    runner.expect(std::fabs(u[6 * 4 + 2] - w(centre)) <= 1e-9 * kx,
                  shapeName(triangles) + ": deflection of the centre " + sci(u[6 * 4 + 2]));
    const double d = e * t * t * t / (12.0 * (1.0 - nu * nu));
    const std::array<std::array<double, 2>, 2> curvature{{{-kx, -kxy}, {-kxy, -ky}}};
    const double trace = curvature[0][0] + curvature[1][1];
    std::array<std::array<double, 2>, 2> moment{};
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            moment[i][j] = d * ((1.0 - nu) * curvature[i][j] + (i == j ? nu * trace : 0.0));
    double worst = 0.0;
    for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
    {
        const mh::ShellElementData& el = analysis.shell(ie);
        const mh::ShellRecovery results = part.recover(el, el.local(u), nullptr, {}, {});
        for (const mh::ShellRecoveryPoint* point : {&results.centre, &results.corners[0]})
        {
            const auto component = [&](std::size_t i, std::size_t j)
            {
                double v = 0.0;
                for (std::size_t x = 0; x < 2; ++x)
                    for (std::size_t y = 0; y < 2; ++y)
                        v += point->state.frame[i][x] * moment[x][y] * point->state.frame[j][y];
                return v;
            };
            const std::array<double, 3> expected{component(0, 0), component(1, 1), component(0, 1)};
            for (std::size_t c = 0; c < 3; ++c)
                worst = std::max(worst, std::fabs(point->state.resultants[3 + c] - expected[c]) / (d * kx));
        }
    }
    runner.expect(worst <= 1e-9, shapeName(triangles) + ": moments, error " + sci(worst));
}

// The departure of a surface given by its points.
double departureOf(std::size_t cornerCount, const std::function<mh::SpaceVector(double, double)>& surface)
{
    return mh::shellShapeDeparture(cornerCount, surface).relative;
}

std::function<mh::SpaceVector(double, double)> bilinear(const std::array<mh::SpaceVector, 4>& p)
{
    return [p](double xi, double eta)
    {
        const double n[4] = {0.25 * (1 - xi) * (1 - eta), 0.25 * (1 + xi) * (1 - eta), 0.25 * (1 + xi) * (1 + eta),
                             0.25 * (1 - xi) * (1 + eta)};
        mh::SpaceVector x{};
        for (std::size_t k = 0; k < 4; ++k)
            for (std::size_t c = 0; c < 3; ++c)
                x[c] += n[k] * p[k][c];
        return x;
    };
}

} // namespace

// The degrees of the shells of a part: the numbering of its edges (the degree of an edge is the lowest of its
// shells), and a grid component of each unknown reached through describe().
CHECK_CASE("shellpart: the numbering of shells of different degrees")
{
    for (const bool triangles : {false, true})
    {
        const mh::ShellPart part(mixedInput(distortedPlate(), isotropicSection(2.1e11, 0.3, 0.01), triangles));
        runner.expect(part.usable() && part.problems().empty(), shapeName(triangles) + ": the part is read");
        bool degrees = true;
        for (std::size_t e = 0; e < part.topology().edges.size(); ++e)
        {
            std::size_t lowest = part.degree();
            for (const std::size_t ie : part.topology().edgeElements[e])
                lowest = std::min(lowest, part.shellDegree(ie));
            degrees = degrees && part.edgeDegree(e) == lowest && part.edgeDofCount(e) == 5 * lowest - 3 &&
                      part.entities()[part.edgeEntity(e)].dofCount == part.edgeDofCount(e);
        }
        runner.expect(degrees, shapeName(triangles) + ": the degree of an edge is the lowest of its shells");
        std::vector<int> reached(part.unknownCount(), 0);
        for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
            for (const std::size_t d : part.localUnknowns(ie))
                ++reached[d];
        bool allReached = true;
        for (std::size_t u = part.firstEntityUnknown(); u < part.unknownCount(); ++u)
        {
            const mh::ShellPartUnknown what = part.describe(u);
            allReached = allReached && reached[u] > 0 &&
                         part.entities()[what.index].firstUnknown + what.component == u;
        }
        runner.expect(allReached, shapeName(triangles) + ": every entity DOF is an unknown of a shell");
    }
}

// Rigid motions and constant states of a plate whose shells have different degrees: the constant stress and
// curvature are exact, across the edges between the degrees as within each.
CHECK_CASE("shellpart: constant states of shells of different degrees")
{
    for (const bool triangles : {false, true})
    {
        checkMembrane(runner, triangles);
        checkBending(runner, triangles);
    }
}

CHECK_CASE("shellpart: rigid motions of shells of different degrees have no energy")
{
    // The flat plate, and a cylinder panel of radius 1 that spans 0.8 rad. The edges along its arcs are conics, whose
    // rotation about the edge is a polynomial of degree 2 of the edge in a rigid motion: an edge of degree 2 (a
    // rotation of degree 1) holds it up to 1e-9 of the largest term of the stiffness for this arc, whatever the degree
    // of the shells beside it.
    constexpr double ON_CONICS = 1e-8;
    constexpr double ON_STRAIGHT_EDGES = 1e-10;
    const MappedMesh panel = mappedMesh(2, 2,
                                        [](double s, double t)
                                        {
                                            const double angle = 0.8 * (t - 0.5);
                                            return mh::SpaceVector{2.0 * s, std::sin(angle), std::cos(angle)};
                                        });
    for (const bool triangles : {false, true})
        for (const bool curved : {false, true})
        {
            mh::ShellPartInput input =
                mixedInput(curved ? panel : distortedPlate(), isotropicSection(7e10, 0.3, 0.01), triangles);
            if (curved)
                input.rawElements.clear();
            const mh::ShellPart part(input);
            DenseAnalysis analysis(part);
            for (std::size_t m = 0; m < 6; ++m)
            {
                mh::SpaceVector b{};
                mh::SpaceVector omega{};
                if (m < 3)
                    b[m] = 1.0;
                else
                    omega[m - 3] = 1.0;
                const std::array<mh::SpaceVector, 3> gradient =
                    m < 3 ? std::array<mh::SpaceVector, 3>{} : rotationGradient(omega);
                const mh::Vector r = affineVector(part, gradient, b);
                double diagonal = 0.0;
                for (std::size_t i = 0; i < analysis.stiffness().rows(); ++i)
                    diagonal = std::max(diagonal, analysis.stiffness()(i, i));
                const mh::Vector kr = analysis.stiffness() * r;
                const double force = maxAbsOf(kr) / (diagonal * maxAbsOf(r));
                runner.expect(force < (curved ? ON_CONICS : ON_STRAIGHT_EDGES),
                              shapeName(triangles) + (curved ? " on a curved panel" : "") + ", rigid motion " +
                                  std::to_string(m) + ": " + sci(force));
            }
        }
}

// The departure of known shapes from an affine element, and its invariance by a change of scale and a rotation.
CHECK_CASE("shell degree: departure of the shapes")
{
    const auto parallelogram = bilinear({mh::SpaceVector{0, 0, 0}, mh::SpaceVector{2, 0.3, 0},
                                         mh::SpaceVector{2.5, 1.3, 0}, mh::SpaceVector{0.5, 1.0, 0}});
    runner.expect(departureOf(4, parallelogram) < 1e-14, "a flat parallelogram");
    // a trapezoid of parallel sides a = 1.2 and b = 0.8 as high as it is wide: |a − b| / (2 (a + b))
    const double a = 1.2;
    const double b = 0.8;
    const auto trapezoid = bilinear({mh::SpaceVector{-a / 2, 0, 0}, mh::SpaceVector{a / 2, 0, 0},
                                     mh::SpaceVector{b / 2, (a + b) / 2, 0}, mh::SpaceVector{-b / 2, (a + b) / 2, 0}});
    runner.expectNear(departureOf(4, trapezoid), (a - b) / (2.0 * (a + b)), 1e-12, "a trapezoid");
    // a square with a corner lifted by h: the bilinear term is h / 4 times the side
    const double h = 0.1;
    const auto warped = bilinear({mh::SpaceVector{0, 0, 0}, mh::SpaceVector{1, 0, 0}, mh::SpaceVector{1, 1, h},
                                  mh::SpaceVector{0, 1, 0}});
    runner.expect(departureOf(4, warped) > 0.9 * h / 4.0 && departureOf(4, warped) < 1.1 * h / 4.0, "a lifted corner");
    // a flat triangle has none
    const auto triangle = [](double xi, double eta)
    { return mh::SpaceVector{0.5 * (1 + xi) * 1.3 + 0.1 * (1 + eta), 0.5 * (1 + eta) * 0.9, 0.2 * (1 + xi)}; };
    runner.expect(departureOf(3, triangle) < 1e-14, "a flat triangle");
    // an arc of radius 1 that subtends 0.4 rad, of unit depth: the sagitta over the size, an eighth of the angle
    const double angle = 0.4;
    const auto arc = [angle](double xi, double eta)
    { return mh::SpaceVector{std::sin(0.5 * angle * xi), std::cos(0.5 * angle * xi), 0.5 * angle * (1.0 + eta)}; };
    runner.expect(std::fabs(departureOf(4, arc) / (angle / 8.0) - 1.0) < 0.05,
                  "an arc: " + sci(departureOf(4, arc) * 8.0 / angle));
    // scaled by 7 and turned
    const auto scaled = [&](double xi, double eta)
    {
        const mh::SpaceVector x = trapezoid(xi, eta);
        return mh::SpaceVector{7.0 * (x[0] * std::cos(0.7) - x[1] * std::sin(0.7)), 2.0,
                               7.0 * (x[0] * std::sin(0.7) + x[1] * std::cos(0.7))};
    };
    runner.expectNear(departureOf(4, scaled), departureOf(4, trapezoid), 1e-12,
                      "the departure of a moved and scaled shape");
    const mh::ShellDegreeChoice choice;
    const auto degreeOf = [&](const mh::SurfaceAt& surface, double thickness)
    { return mh::chooseShellDegree(mh::shellShapeDeparture(4, surface), thickness, choice); };
    runner.expect(degreeOf(parallelogram, 0.01) == choice.baseDegree, "degree of a parallelogram");
    // the trapezoid above: a departure of 0.1; one that tapers to a fifth of its base: 0.33
    runner.expect(degreeOf(trapezoid, 0.01) == choice.enrichedDegree, "degree of a trapezoid");
    const auto tapered = bilinear({mh::SpaceVector{0, 0, 0}, mh::SpaceVector{1, 0, 0}, mh::SpaceVector{0.6, 1, 0},
                                   mh::SpaceVector{0.4, 1, 0}});
    runner.expect(degreeOf(tapered, 0.01) == choice.enrichedDegree, "degree of a tapered element");
}

// The degrees a part chooses: the lowest for a regular mesh, the highest for the shells away from an affine one, and
// the degree of the part for those of an edge named full.
CHECK_CASE("shell degree: the degrees a part chooses from the shapes")
{
    const mh::Section section = isotropicSection(2.1e11, 0.3, 0.01);
    const MappedMesh regular =
        mappedMesh(3, 2, [](double s, double t) { return mh::SpaceVector{3.0 * s, 2.0 * t, 0.0}; });
    mh::ShellPartInput input = shellInput(regular, section, false);
    input.checkElements = false;
    input.degreeChoice = mh::ShellDegreeChoice{};
    const mh::ShellPart flat(input);
    bool lowest = true;
    for (std::size_t ie = 0; ie < 6; ++ie)
        lowest = lowest && flat.shellDegree(ie) == 3;
    runner.expect(lowest, "the cells of a regular mesh keep the lowest degree");

    MappedMesh moved = regular;
    moved.grids[4] = {1.4, 1.25, 0.0}; // a grid of the inside, away from the regular mesh
    mh::ShellPartInput distortedInput = shellInput(moved, section, false);
    distortedInput.checkElements = false;
    distortedInput.degreeChoice = mh::ShellDegreeChoice{};
    const mh::ShellPart distorted(distortedInput);
    std::size_t highest = 0;
    for (std::size_t ie = 0; ie < 6; ++ie)
        highest += distorted.shellDegree(ie) == 4 ? 1 : 0;
    runner.expect(highest == 4, "the four cells about the moved grid take the highest degree: " +
                                    std::to_string(highest));
    runner.expect(distorted.degree() == 4, "the degree of the part is the highest a shell may have");

    // an element that tapers takes the highest degree
    MappedMesh tapered = regular;
    tapered.grids[4] = {0.12, 1.0, 0.0};
    mh::ShellPartInput taperedInput = shellInput(tapered, section, false);
    taperedInput.checkElements = false;
    taperedInput.degreeChoice = mh::ShellDegreeChoice{};
    const mh::ShellPart taper(taperedInput);
    std::size_t four = 0;
    for (std::size_t ie = 0; ie < 6; ++ie)
        four += taper.shellDegree(ie) == 4 ? 1 : 0;
    runner.expect(four > 0, "an element that tapers much has the degree 4");

    input.fullDegreeEdges = {mh::EdgeKey::of(0, 1)};
    const mh::ShellPart full(input);
    runner.expect(full.shellDegree(0) == full.degree(), "a shell on an edge named full has the degree of the part");
    input.shellDegrees.assign(input.shells.size(), 3);
    const mh::ShellPart prescribed(input);
    runner.expect(prescribed.shellDegree(0) == 4, "a frozen cubic layout keeps a full solid interface");
    input.shellDegrees.clear();
    input.fullDegreeEdges.clear();
    const mh::ShellPart again(input);
    runner.expect(again.shellDegree(0) == 3, "no edge named full: the degree of its shape");
}

CHECK_CASE("shell degree: local director work on an affine element")
{
    const auto plate = mappedMesh(1, 1, [](double s, double t) { return mh::SpaceVector{s, t, 0.0}; });
    for (const bool triangles : {false, true})
    {
        auto input = shellInput(plate, isotropicSection(1000., 0.3, 0.1), triangles);
        input.degree = 3;
        for (auto& cell : input.shells)
            cell.offset = 0.05;
        const mh::ShellPart part(input);
        const auto& element = part.elementAtPoints(0);
        const auto& ref = part.referenceElement(element);
        mh::Vector x(mh::shellUnknownCount(ref), 0.0);
        mh::ShellSolutionCoefficients field;
        field.moments.resize(ref.momentCount);
        const auto work = [&] { return mh::directorEquilibrium(part, 0, element, x, field, nullptr).work; };
        // Constant M with Q=0 equilibrates every zero-boundary director.
        field.moments.back() = 1.;
        const double constant = work();
        field.moments.back() = 0.;
        double unbalanced = 0.0;
        for (std::size_t i = 0; i + 3 < ref.momentCount && unbalanced < 1.e-8; ++i)
        {
            field.moments.assign(ref.momentCount, 0.);
            field.moments[i] = 1.;
            unbalanced = work();
        }
        runner.expect(
            unbalanced > 1.e-8 && constant < 1.e-20 * unbalanced,
            shapeName(triangles) + ": distinguishes a varying unbalanced moment from a constant equilibrated moment");
        for (auto& m : field.moments)
            m *= -3.;
        runner.expectNear(work(), 9. * unbalanced, 1.e-11, shapeName(triangles) + ": quadratic correction work");

        // An offset isotropic skin has a uniform director load I1 a. Project
        // the exact constant shear H^-1 I1 a onto the covariant shear basis.
        oos::elements::BodyAcceleration acceleration;
        acceleration.constant = {1., 2., 0.};
        const auto g = mh::geometryAt(ref, element.definition(), 0., 0.);
        const auto section = mh::sectionAtGeometry(element.section, element.offset, 0., 0., g);
        std::array<double, 2> target{};
        for (std::size_t c = 0; c < 2; ++c)
            target[c] = section.mass[1] * mh::dot3(g.a[c], acceleration.constant.components) / section.h[0][0];
        mh::Matrix gram(ref.hcurlCount, ref.hcurlCount);
        mh::Vector rhs(ref.hcurlCount);
        for (std::size_t q = 0; q < ref.points.size(); ++q)
            for (std::size_t i = 0; i < ref.hcurlCount; ++i)
            {
                const auto& a = ref.hcurl[q * ref.hcurlCount + i];
                rhs[i] += ref.weights[q] * (a[0] * target[0] + a[1] * target[1]);
                for (std::size_t j = 0; j < ref.hcurlCount; ++j)
                {
                    const auto& b = ref.hcurl[q * ref.hcurlCount + j];
                    gram(i, j) += ref.weights[q] * (a[0] * b[0] + a[1] * b[1]);
                }
            }
        const auto shear = mh::EquilibratedCholesky::factor(gram).value().solve(rhs);
        std::copy(shear.begin(), shear.end(), x.begin() + 3 * ref.h1Count);
        field.moments.assign(ref.momentCount, 0.);
        const double unloaded = work();
        const auto balanced = mh::directorEquilibrium(part, 0, element, x, field, &acceleration);
        runner.expect(unloaded > 0. && balanced.work < 1.e-20 * unloaded,
                      shapeName(triangles) + ": transverse shear balances the applied director couple");
    }
}

// The same panel uses the same transition construction. Only the selected degrees respond to thickness.
CHECK_CASE("shell degree: curvature enriches a thin panel even on regular cells")
{
    const MappedMesh panel = mappedMesh(3, 2, [](double s, double t)
    {
        const double angle = 0.6 * (t - 0.5);
        return mh::SpaceVector{2.0 * s, std::sin(angle), std::cos(angle)};
    });
    const auto degreesAt = [&](double thickness, std::optional<std::array<double, 4>> corners = std::nullopt)
    {
        mh::ShellPartInput input = shellInput(panel, isotropicSection(7e10, 0.3, thickness), false);
        input.checkElements = false;
        input.degreeChoice = mh::ShellDegreeChoice{};
        for (auto& shell : input.shells)
            shell.gridThicknesses = corners;
        const mh::ShellPart part(input);
        std::vector<std::size_t> degrees;
        for (std::size_t ie = 0; ie < input.shells.size(); ++ie)
            degrees.push_back(part.shellDegree(ie));
        return degrees;
    };
    runner.expect(degreesAt(0.001) == std::vector<std::size_t>(6, 4), "sagitta exceeds the thickness");
    runner.expect(degreesAt(0.2) == std::vector<std::size_t>(6, 3), "sagitta within the thickness");
    runner.expect(degreesAt(0.2, std::array{0.2, 0.2, 0.001, 0.2}) == degreesAt(0.001),
                  "the thinnest corner controls enrichment");
    runner.expect(degreesAt(0.001, std::array{0.2, 0.2, 0.2, 0.2}) == degreesAt(0.2),
                  "corner thicknesses override the section thickness");
}
