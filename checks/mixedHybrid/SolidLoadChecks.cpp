#include "CheckRunner.hpp"
#include "SolidCheckSupport.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/solid/SolidData.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeGeometry.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeKernel.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using namespace oos::checks::solid;

namespace
{

// |value − expected| ≤ tolerance |expected|.
void expectRelative(oos::checks::CheckRunner& runner, double value, double expected, double tolerance,
                    const std::string& what)
{
    runner.expect(std::fabs(value - expected) <= tolerance * std::fabs(expected),
                  what + ": " + std::to_string(value) + " expected " + std::to_string(expected));
}

// Every term within tolerance × the largest expected term.
template <std::size_t N>
void expectVector(oos::checks::CheckRunner& runner, std::span<const double> value,
                  const std::array<double, N>& expected, double tolerance, const std::string& what)
{
    double largest = 0.0;
    for (double e : expected)
        largest = std::max(largest, std::fabs(e));
    double worst = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        worst = std::max(worst, std::fabs(value[i] - expected[i]));
    runner.expect(worst <= tolerance * largest, what + ": off by " + std::to_string(worst / largest));
}

// Numbers of one element computed by the reference crate at commit 9f55f0b (a throwaway probe: VolumeGeometry::
// straight on the distorted element of its tests, the mid-side offsets set on the first edge coefficients,
// volume_kernel with the steel of its tests, a body force (1e3, −2e3, −7800 · 9.81) and a free strain
// 1e-3 (1 + x/2), the face loads and follower stiffness of tdnns::solid on face 0 for a pressure 1e6 (1 + x)).
struct ReferenceValues
{
    mh::VolumeShape shape;
    std::array<GridId, 8> order;
    std::size_t degree; // p = pf = d
    bool midSide;
    double volume;
    double coefficientSum; // Σ |coefficients of the functions|
    double stiffnessTrace;
    std::array<double, 3> eigenvalues7to9; // sorted increasing: the first three that are not zero
    double largestEigenvalue;
    double massTrace;
    std::array<double, 3> interiorStiffnesses; // sorted increasing
    std::array<double, 3> bodyForceSums;       // Σ over the parameters of each component
    double bodyForce5;
    double thermalForceNorm;
    double thermalForce4;
    std::array<double, 6> bodyStressAtCentre;
    std::array<double, 6> thermalStressAtCentre;
    std::array<double, 6> stressAtPoint; // q_i = 1e-4 sin(0.37 i) with the body force, at r = (0.1, −0.2, −0.3)
    double geometricTrace;               // for the same stresses
    double geometric37;
    std::array<double, 3> pressureSums;
    double followerNorm; // Frobenius norm of the symmetric load stiffness
};

const std::array<ReferenceValues, 5> REFERENCE{{
    {mh::VolumeShape::Hexahedron,
     {7, 3, 9, 5, 12, 1, 4, 10},
     4,
     false,
     1.68691419886795413e0,
     2.44721995464858608e1,
     1.82768790548380566e12,
     {5.14476779484482715e5, 5.65821302202283288e5, 5.95884415614197380e5},
     3.80296826701285950e11,
     1.35486601864436270e4,
     {3.39998365675846279e8, 3.95921333237890005e8, 5.05904378134500742e8},
     {3.36704753380271541e3, -4.62889398949624774e3, -1.67832838432561053e5},
     -1.64144747002693912e4,
     9.60181575296879888e8,
     -2.15456006306221902e8,
     {3.90957385791758369e1, -4.65227524152435876e1, 9.22086603888428868e0, -8.20473134848223538e1,
      1.65171766392201619e2, 1.44773715765781901e2},
     {-5.02599566427570581e8, -5.02992341830105662e8, -5.02202174404426217e8, 2.56737990541587991e5,
      -1.66894439771927515e5, 5.30755212595019475e4},
     {8.88234847802455910e6, 8.04305395481759217e5, -4.40998656838365889e5, 3.95001746887599630e6,
      -7.48592736606848240e6, -2.31157521902724914e6},
     1.03758388392272934e8,
     3.15095866141308972e5,
     {-1.83149665969654219e5, 7.35334271978927609e3, 2.29751849782364815e6},
     1.23603439368197252e6},
    {mh::VolumeShape::Pyramid,
     {7, 3, 9, 5, 12, 0, 0, 0},
     4,
     false,
     5.77758266574381341e-1,
     1.19624645125993911e1,
     9.37442311494606445e11,
     {9.05773625200408627e5, 9.27168844965070020e5, 1.08575866411762405e6},
     2.97777753239881775e11,
     5.44099243631958689e3,
     {8.00413240575334907e8, 9.14043318286324382e8, 1.12514051239206243e9},
     {7.58699732032195811e2, -1.44072031735442124e3, -5.68863709347217955e4},
     -8.31070324526144032e3,
     4.83610294829386771e8,
     -1.41617708089081109e8,
     {3.59410062660479980e2, 7.00214736729598258e2, -6.24569035945042560e3, -3.49761956103804508e1,
      1.43067531348961069e2, 4.51203348440522348e2},
     {-4.98855823193279088e8, -4.99100941067679465e8, -4.97782551938963830e8, 1.28821217630701271e5,
      -9.16680589062416111e4, -1.97909825953674084e6},
     {5.11091526678621862e6, -1.68154043743855692e7, -2.62894427889306359e7, 8.09906711144011002e6,
      -1.42230666468241718e7, -6.31924710197580513e6},
     -1.52919154604918599e8,
     1.87288963822706544e5,
     {-1.83149665969654801e5, 7.35334271978880679e3, 2.29751849782364815e6},
     1.23603439368197531e6},
    {mh::VolumeShape::Pentahedron,
     {7, 3, 9, 5, 12, 1, 0, 0},
     4,
     true,
     7.81257552724394122e-1,
     1.52058512949696016e1,
     1.15344826949445093e12,
     {5.52880992031724425e5, 6.43912685934819747e5, 6.84375629121985519e5},
     2.89952021443234253e11,
     6.86122386415277560e3,
     {6.41172835338557363e8, 7.07671060171161175e8, 8.10351507307083368e8},
     {5.61101511094059219e2, -1.74900976391464883e3, -7.64921676026893983e4},
     -1.01997443548753308e4,
     4.99704929018039525e8,
     -3.68631803142281324e7,
     {3.99562151022200737e2, 2.35053195596809502e2, 8.02205811522420049e2, 1.15247597953514401e2, 3.03847089392000100e2,
      8.04784158385456180e2},
     {-4.31641824729299843e8, -4.29156568323437512e8, -4.32884580140652180e8, 7.38653942691430822e6,
      3.06989916225435620e5, 1.11264563388890726e6},
     {-1.57630835990974456e7, -3.35295937258720398e7, -3.95999405180171058e7, -3.32039268924689759e6,
      -1.75439529229394309e7, -1.15595516988781821e7},
     -3.02908087400539935e8,
     -5.86741947640221100e5,
     {-5.83941040377744939e4, -6.88175271797490495e3, 7.91559671961866203e5},
     6.61182625285577495e5},
    {mh::VolumeShape::Tetrahedron,
     {7, 3, 9, 5, 0, 0, 0, 0},
     4,
     true,
     2.89208036991931983e-1,
     8.08922184438886838e0,
     6.65220229524869751e11,
     {5.67020273288257653e5, 8.15089501163206762e5, 8.90143177840116085e5},
     2.73853618983596710e11,
     2.91165124171967318e3,
     {1.40949349449963355e9, 1.50676216151101327e9, 2.05135220899645090e9},
     {3.28211801400317256e2, -7.16048571181347484e2, -2.77405602183932424e4},
     -5.42947190923356629e3,
     2.54240061927149028e8,
     -2.08525567077897973e7,
     {4.10806893890657022e2, 6.67397724776078690e2, 3.60556087711304883e2, 5.60992059608118367e1, 1.19801854043337585e3,
      1.75128152202480464e3},
     {-3.99832833919327199e8, -4.02111058827674508e8, -3.99501838113900423e8, 4.52046502206473611e6,
      8.04909556010561995e5, 3.33570629874570249e6},
     {1.49251737578199320e7, 2.00945112218184881e7, -1.79596293598825228e5, 9.04661920648888312e6,
      -5.20049662552973069e6, 8.85250592250584811e6},
     -2.03633505161243938e7,
     1.77710185478225696e2,
     {-5.02409121705669037e5, -5.58762904950989876e5, -8.31398939262569416e5},
     7.45858953690847033e5},
    {mh::VolumeShape::Hexahedron,
     {21, 2, 40, 5, 13, 8, 30, 11},
     2,
     true,
     1.60509990271198499e0,
     1.31666666666665986e1,
     1.67830851752116431e12,
     {2.09168575891788304e7, 2.43100634949680455e7, 3.08363778250029758e7},
     3.92410381284939331e11,
     1.27265794073713296e4,
     {3.25678646096526325e8, 3.73250201811521709e8, 5.02871254325299561e8},
     {2.11115606337243617e3, -4.37126224776272011e3, -1.73082622587560065e5},
     -1.57416783521070429e4,
     9.40592727520803571e8,
     -2.04247880685312450e8,
     {9.35277929463907327e1, -8.90096479309909938e1, -7.79696470688855356e1, -1.22069999770505575e2,
      1.74107761850292633e2, 1.42424063715251037e2},
     {-5.02973622300284386e8, -5.02958780935138643e8, -5.02166257898718596e8, 1.20619679663531017e5,
      -4.16350110580629495e5, 2.18862684009173710e5},
     {1.06132089391277023e7, 5.07922064926003106e6, 1.07027677234475911e7, 2.51802957757303864e6,
      -8.89261231469244929e5, -2.13400561679612985e6},
     1.46184508306404054e8,
     2.19089827209928539e5,
     {-1.62135480324119912e5, 2.65433327091583715e3, 2.46885078112229146e6},
     1.23554249673470203e6},
}};

// The geometry of the probe: the distorted element, the mid-side offsets set as the first edge coefficients.
std::optional<mh::VolumeGeometry> probeGeometry(const ReferenceValues& values, std::span<const GridId> order)
{
    const std::optional<mh::VolumeGeometry> straight =
        geometryOf(values.shape, distorted(values.shape), order, values.degree, values.degree, false);
    if (!straight || !values.midSide)
        return straight;
    mh::VolumeGeometry geometry = *straight;
    for (std::size_t e = 0; e < geometry.edges.size(); ++e)
    {
        const mh::SolidPoint offset = midSideOffset(e);
        geometry.edges[e][0] = {4.0 * offset[0], 4.0 * offset[1], 4.0 * offset[2]};
    }
    return geometry;
}

double sumOfAbsoluteValues(const mh::Matrix& m)
{
    double sum = 0.0;
    for (double value : m.values())
        sum += std::fabs(value);
    return sum;
}

std::array<double, 3> componentSums(std::span<const double> f)
{
    std::array<double, 3> sums{};
    for (std::size_t i = 0; i < f.size(); ++i)
        sums[i % 3] += f[i];
    return sums;
}

void checkReferenceValues(oos::checks::CheckRunner& runner, const ReferenceValues& expected)
{
    constexpr double TOLERANCE = 1e-10;
    const std::string what =
        mh::shapeName(expected.shape) + std::string(expected.midSide ? " with mid-side nodes" : "");
    const std::span<const GridId> order(expected.order.data(), mh::shapeCornerCount(expected.shape));
    const std::optional<mh::VolumeGeometry> geometry = probeGeometry(expected, order);
    runner.expect(geometry.has_value(), what + ": geometry");
    if (!geometry)
        return;
    const std::size_t p = expected.degree;
    const std::vector<mh::SolidLoad> loads{
        mh::SolidLoad::body([](const mh::SolidPoint&) { return mh::SolidPoint{1.0e3, -2.0e3, -7800.0 * 9.81}; }),
        mh::SolidLoad::thermal([](const mh::SolidPoint& x) { return 1.0e-3 * (1.0 + 0.5 * x[0]); })};
    const mh::SolidResult<mh::VolumeKernel> built =
        mh::VolumeKernel::build(*geometry, order, steel(), optionsOf(p, p, p), true, loads);
    runner.expect(built.ok(), what + ": kernel (" + built.refusal() + ")");
    if (!built.ok())
        return;
    const mh::VolumeKernel& kernel = built.value();
    expectRelative(runner, kernel.volume(), expected.volume, TOLERANCE, what + ": volume");
    expectRelative(runner, sumOfAbsoluteValues(geometry->functionCoefficients), expected.coefficientSum, TOLERANCE,
                   what + ": coefficients of the functions");
    expectRelative(runner, mh::trace(kernel.stiffness()), expected.stiffnessTrace, TOLERANCE, what + ": trace of K");
    const std::vector<double> eigenvalues = sortedEigenvalues(kernel.stiffness());
    for (std::size_t i = 0; i < 6; ++i)
        runner.expect(std::fabs(eigenvalues[i]) < 1e-9 * eigenvalues.back(), what + ": zero eigenvalue");
    for (std::size_t i = 0; i < 3; ++i)
        expectRelative(runner, eigenvalues[6 + i], expected.eigenvalues7to9[i], TOLERANCE, what + ": eigenvalue");
    expectRelative(runner, eigenvalues.back(), expected.largestEigenvalue, TOLERANCE, what + ": largest eigenvalue");
    expectRelative(runner, mh::trace(*kernel.mass()), expected.massTrace, TOLERANCE, what + ": trace of M");
    std::vector<double> interior = kernel.interiorFields()->stiffnesses;
    std::sort(interior.begin(), interior.end());
    expectVector(runner, interior, expected.interiorStiffnesses, TOLERANCE, what + ": interior fields");
    const mh::SolidLoadResult& body = kernel.loads()[0];
    const mh::SolidLoadResult& thermal = kernel.loads()[1];
    expectVector(runner, componentSums(body.force), expected.bodyForceSums, TOLERANCE, what + ": body force");
    expectRelative(runner, body.force[5], expected.bodyForce5, TOLERANCE, what + ": body force on parameter 5");
    expectRelative(runner, mh::euclideanNorm(thermal.force), expected.thermalForceNorm, TOLERANCE,
                   what + ": thermal force");
    expectRelative(runner, thermal.force[4], expected.thermalForce4, TOLERANCE, what + ": thermal force 4");
    const mh::SolidPoint centre = mh::shapeCentre(expected.shape);
    expectVector(runner, kernel.stressAt(body.particularStresses, centre), expected.bodyStressAtCentre, TOLERANCE,
                 what + ": particular stresses of the body force");
    expectVector(runner, kernel.stressAt(thermal.particularStresses, centre), expected.thermalStressAtCentre, TOLERANCE,
                 what + ": particular stresses of the free strain");
    mh::Vector q(kernel.parameterCount());
    for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = 1e-4 * std::sin(0.37 * static_cast<double>(i));
    const std::array<std::size_t, 1> bodyOnly{0};
    const mh::Vector beta = kernel.stressCoefficients(q, bodyOnly);
    expectVector(runner, kernel.stressAt(beta, {0.1, -0.2, -0.3}), expected.stressAtPoint, TOLERANCE,
                 what + ": stresses at a point");
    const mh::SolidResult<mh::Matrix> geometric = kernel.geometric(beta);
    runner.expect(geometric.ok(), what + ": geometric stiffness");
    if (geometric.ok())
    {
        expectRelative(runner, mh::trace(geometric.value()), expected.geometricTrace, TOLERANCE,
                       what + ": trace of K_g");
        expectRelative(runner, geometric.value()(3, 7), expected.geometric37, TOLERANCE, what + ": K_g (3, 7)");
    }
    const auto pressure = [](const mh::SolidPoint& x) { return 1.0e6 * (1.0 + x[0]); };
    const mh::Vector pressureLoad = kernel.facePressureLoad(0, pressure, p + 4);
    expectVector(runner, componentSums(pressureLoad), expected.pressureSums, TOLERANCE, what + ": pressure load");
    const mh::Matrix follower = kernel.followerPressureStiffness(0, pressure, p + 4);
    expectRelative(runner, mh::frobeniusNorm(follower), expected.followerNorm, TOLERANCE,
                   what + ": load stiffness of the pressure");
}

} // namespace

// Stiffness, mass, interior fields, loads, stresses, geometric and load stiffness of distorted elements against the
// numbers of the reference (relative 1e-10).
CHECK_CASE("solid: reference values, hexahedron")
{
    checkReferenceValues(runner, REFERENCE[0]);
}
CHECK_CASE("solid: reference values, pyramid")
{
    checkReferenceValues(runner, REFERENCE[1]);
}
CHECK_CASE("solid: reference values, pentahedron with mid-side nodes")
{
    checkReferenceValues(runner, REFERENCE[2]);
}
CHECK_CASE("solid: reference values, tetrahedron with mid-side nodes")
{
    checkReferenceValues(runner, REFERENCE[3]);
}
CHECK_CASE("solid: reference values, quadratic hexahedron")
{
    checkReferenceValues(runner, REFERENCE[4]);
}

namespace
{

// The kernel of the straight hexahedron of degree 4, with the loads `loads`.
mh::SolidResult<mh::VolumeKernel> straightHexahedron(std::span<const mh::SolidLoad> loads, bool midSide)
{
    const std::vector<GridId> order = numbersOf(mh::VolumeShape::Hexahedron);
    const std::optional<mh::VolumeGeometry> geometry =
        geometryOf(mh::VolumeShape::Hexahedron, affine(mh::VolumeShape::Hexahedron), order, 4, 4, midSide);
    return mh::VolumeKernel::build(*geometry, order, steel(), optionsOf(4, 4, 4), true, loads);
}

} // namespace

// The loads of a straight hexahedron add up to their resultants: a uniform acceleration ρ a V, a pressure −p A n and a
// traction t A on a flat face (the corner functions are a partition of unity); a free expansion is stress free and K q
// is its thermal load.
CHECK_CASE("solid: loads of a straight hexahedron")
{
    const mh::SolidPoint gravity{0.5, -1.5, -9.81};
    constexpr double FREE_STRAIN = 1.0e-3;
    const std::vector<mh::SolidLoad> loads{mh::SolidLoad::uniformAcceleration(7800.0, gravity),
                                           mh::SolidLoad::thermal([&](const mh::SolidPoint&) { return FREE_STRAIN; })};
    const mh::SolidResult<mh::VolumeKernel> built = straightHexahedron(loads, false);
    runner.expect(built.ok(), "kernel");
    if (!built.ok())
        return;
    const mh::VolumeKernel& kernel = built.value();
    const std::size_t corners = kernel.geometry().corners.size();
    const auto cornerSums = [&](const mh::Vector& f)
    {
        mh::SolidPoint sums{};
        for (std::size_t j = 0; j < corners; ++j)
            for (std::size_t c = 0; c < 3; ++c)
                sums[c] += f[3 * j + c];
        return sums;
    };
    const mh::SolidPoint weight = cornerSums(kernel.loads()[0].force);
    for (std::size_t c = 0; c < 3; ++c)
        expectRelative(runner, weight[c], 7800.0 * gravity[c] * kernel.volume(), 1e-12, "weight");
    // face 0 (corners 0, 1, 2, 3 of the reference at z = −1), a parallelogram: area vector (x1 − x0) × (x3 − x0)
    const std::vector<mh::SolidPoint>& x = kernel.geometry().corners;
    mh::SolidPoint area = mh::cross3(mh::sub3(x[1], x[0]), mh::sub3(x[3], x[0]));
    const mh::SolidPoint inside = kernel.geometry().at(mh::shapeCentre(mh::VolumeShape::Hexahedron));
    if (mh::dot3(area, mh::sub3(x[0], inside)) < 0.0)
        area = mh::scl3(area, -1.0);
    constexpr double PRESSURE = 2.5e6;
    const mh::SolidPoint pressureForce = cornerSums(
        kernel.facePressureLoad(0, [&](const mh::SolidPoint&) { return PRESSURE; }, kernel.traceDegree() + 4));
    const mh::SolidPoint traction{1.0e5, -3.0e5, 2.0e5};
    const mh::SolidPoint tractionForce = cornerSums(
        kernel.faceTractionLoad(0, [&](const mh::SolidPoint&) { return traction; }, kernel.traceDegree() + 4));
    const double areaNorm = mh::norm3(area);
    for (std::size_t c = 0; c < 3; ++c)
    {
        runner.expect(std::fabs(pressureForce[c] + PRESSURE * area[c]) <= 1e-12 * PRESSURE * areaNorm, "pressure");
        expectRelative(runner, tractionForce[c], traction[c] * areaNorm, 1e-12, "traction");
    }
    // the free expansion u = e0 x
    const std::array<std::array<double, 3>, 3> expansion{
        {{FREE_STRAIN, 0.0, 0.0}, {0.0, FREE_STRAIN, 0.0}, {0.0, 0.0, FREE_STRAIN}}};
    const mh::Vector q = linearParameters(kernel.geometry(), expansion, {0.0, 0.0, 0.0});
    const std::array<std::size_t, 1> thermalOnly{1};
    const mh::Vector beta = kernel.stressCoefficients(q, thermalOnly);
    const double scaleOfStress = 3.0 * steel().stiffness[0][1] * FREE_STRAIN;
    double worst = 0.0;
    for (const mh::SolidPoint& r : {mh::shapeCentre(mh::VolumeShape::Hexahedron), mh::SolidPoint{0.7, -0.6, 0.8}})
        for (double s : kernel.stressAt(beta, r))
            worst = std::max(worst, std::fabs(s));
    runner.expect(worst < 1e-10 * scaleOfStress, "free expansion stress free: " + std::to_string(worst));
    const mh::Vector kq = kernel.stiffness() * q;
    const mh::Vector& f = kernel.loads()[1].force;
    double difference = 0.0;
    for (std::size_t i = 0; i < f.size(); ++i)
        difference = std::max(difference, std::fabs(kq[i] - f[i]));
    runner.expect(difference < 1e-10 * mh::euclideanNorm(f), "K q is the thermal load of a free expansion");
}

// The load stiffness of a uniform pressure that follows the faces: symmetric; on the closed surface of an element the
// resultant of the pressure is zero whatever its shape, so that a translation changes no force.
CHECK_CASE("solid: load stiffness of a following pressure")
{
    for (mh::VolumeShape shape : SHAPES)
    {
        const std::vector<GridId> order = numbersOf(shape);
        const std::optional<mh::VolumeGeometry> geometry = geometryOf(shape, distorted(shape), order, 4, 4, true);
        const mh::SolidResult<mh::VolumeKernel> built =
            mh::VolumeKernel::build(*geometry, order, steel(), optionsOf(4, 4, 4), false, {});
        runner.expect(built.ok(), std::string(mh::shapeName(shape)) + ": kernel");
        if (!built.ok())
            continue;
        const mh::VolumeKernel& kernel = built.value();
        mh::Matrix total(kernel.parameterCount(), kernel.parameterCount());
        for (std::size_t lf = 0; lf < mh::shapeFaces(shape).size(); ++lf)
        {
            const mh::Matrix face =
                kernel.followerPressureStiffness(lf, [](const mh::SolidPoint&) { return 1.0e6; }, 8);
            runner.expect(asymmetry(face) == 0.0, std::string(mh::shapeName(shape)) + ": symmetric");
            total += face;
        }
        for (std::size_t mode = 0; mode < 3; ++mode)
        {
            const mh::Vector t = rigidParameters(*geometry, mode);
            runner.expect(mh::euclideanNorm(total * t) <= 1e-12 * mh::frobeniusNorm(total),
                          std::string(mh::shapeName(shape)) + ": a translation changes no force");
        }
    }
}

// The same element given with its corners in the other orientation, or from another corner, with the same grid
// numbers: the same stiffness (its functions are written from the numbers), refused only when it folds.
CHECK_CASE("solid: elements given from another corner")
{
    struct Renumbering
    {
        mh::VolumeShape shape;
        std::vector<std::size_t> corners; // new corner k is old corner corners[k]
    };
    const std::vector<Renumbering> cases{
        {mh::VolumeShape::Hexahedron, {4, 5, 6, 7, 0, 1, 2, 3}},
        {mh::VolumeShape::Hexahedron, {1, 2, 3, 0, 5, 6, 7, 4}},
        {mh::VolumeShape::Hexahedron, {3, 2, 1, 0, 7, 6, 5, 4}},
        {mh::VolumeShape::Pyramid, {1, 2, 3, 0, 4}},
        {mh::VolumeShape::Pyramid, {3, 2, 1, 0, 4}},
        {mh::VolumeShape::Pentahedron, {3, 4, 5, 0, 1, 2}},
        {mh::VolumeShape::Pentahedron, {1, 2, 0, 4, 5, 3}},
        {mh::VolumeShape::Tetrahedron, {1, 2, 0, 3}},
        {mh::VolumeShape::Tetrahedron, {0, 2, 1, 3}},
    };
    for (const Renumbering& renumbering : cases)
    {
        const std::string what = std::string(mh::shapeName(renumbering.shape)) + " from another corner";
        const std::vector<GridId> order = numbersOf(renumbering.shape);
        const std::vector<mh::SolidPoint> x = distorted(renumbering.shape);
        std::vector<GridId> otherOrder;
        std::vector<mh::SolidPoint> otherX;
        for (std::size_t k : renumbering.corners)
        {
            otherOrder.push_back(order[k]);
            otherX.push_back(x[k]);
        }
        const std::optional<mh::VolumeGeometry> a = geometryOf(renumbering.shape, x, order, 3, 3, false);
        const std::optional<mh::VolumeGeometry> b = geometryOf(renumbering.shape, otherX, otherOrder, 3, 3, false);
        const mh::SolidResult<mh::VolumeKernel> ka =
            mh::VolumeKernel::build(*a, order, steel(), optionsOf(3, 3, 3), false, {});
        const mh::SolidResult<mh::VolumeKernel> kb =
            mh::VolumeKernel::build(*b, otherOrder, steel(), optionsOf(3, 3, 3), false, {});
        runner.expect(ka.ok() && kb.ok(), what + ": kernels");
        if (!ka.ok() || !kb.ok())
            continue;
        expectRelative(runner, mh::trace(kb.value().stiffness()), mh::trace(ka.value().stiffness()), 1e-12,
                       what + ": trace of K");
        expectRelative(runner, kb.value().volume(), ka.value().volume(), 1e-13, what + ": volume");
        const std::vector<double> ea = sortedEigenvalues(ka.value().stiffness());
        const std::vector<double> eb = sortedEigenvalues(kb.value().stiffness());
        double worst = 0.0;
        for (std::size_t i = 0; i < ea.size(); ++i)
            worst = std::max(worst, std::fabs(ea[i] - eb[i]));
        runner.expect(worst <= 1e-11 * ea.back(), what + ": eigenvalues off by " + std::to_string(worst / ea.back()));
    }
}

// Degrees out of range, a geometry built for other degrees, and a folded element are refused.
CHECK_CASE("solid: refusals")
{
    const mh::VolumeShape shape = mh::VolumeShape::Hexahedron;
    const std::vector<GridId> order = numbersOf(shape);
    const std::optional<mh::VolumeGeometry> geometry = geometryOf(shape, distorted(shape), order, 2, 2, false);
    const auto refused = [&](const mh::VolumeGeometry& g, const mh::SolidOptions& options)
    { return !mh::VolumeKernel::build(g, order, steel(), options, true, {}).ok(); };
    runner.expect(!refused(*geometry, optionsOf(2, 2, 2)), "accepted");
    runner.expect(refused(*geometry, optionsOf(2, 2, 1)), "stress degree below the trace degree");
    runner.expect(refused(*geometry, optionsOf(3, 3, 3)), "geometry built for other degrees");
    mh::SolidOptions fields = optionsOf(2, 2, 2);
    fields.interiorFieldCount = 5;
    runner.expect(refused(*geometry, fields), "five interior fields");
    fields.interiorFieldCount = 12;
    runner.expect(!refused(*geometry, fields), "twelve interior fields");
    const std::optional<mh::VolumeGeometry> high = geometryOf(shape, distorted(shape), order, 5, 5, false);
    runner.expect(!high || refused(*high, optionsOf(5, 5, 5)), "trace degree 5");
    // corners 0 and 1 exchanged: the element folds
    std::vector<mh::SolidPoint> folded = distorted(shape);
    std::swap(folded[0], folded[1]);
    const std::optional<mh::VolumeGeometry> foldedGeometry = geometryOf(shape, folded, order, 2, 2, false);
    const mh::SolidResult<mh::VolumeKernel> foldedKernel =
        mh::VolumeKernel::build(*foldedGeometry, order, steel(), optionsOf(2, 2, 2), false, {});
    runner.expect(!foldedKernel.ok() && foldedKernel.refusal().find("folds") != std::string::npos,
                  "a folded element: " + foldedKernel.refusal());
}
