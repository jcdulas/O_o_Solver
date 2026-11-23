#include "CheckRunner.hpp"
#include "LineCheckSupport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
using lineChecks::AloneLine;
using lineChecks::buildAloneLine;
using lineChecks::expectRelative;
using lineChecks::Vec3;
using mh::LineElement;
using mh::LineSection;

namespace
{

const Vec3 SKEW_A{0.3, -0.2, 0.5};
const Vec3 SKEW_B{1.6, 0.4, 1.1};
const Vec3 ORIENTATION{0.0, 0.3, 1.0};

// The beam of the reference probe: tapered, shear factors, neutral axis off the reference axis, expansion,
// non-structural mass.
LineSection probeSection()
{
    LineSection s = LineSection::beam(
        2.0e11, 8.0e10, 7800.0, {{{3.0e-3, 4.0e-6, 9.0e-6, 1.0e-6, 5.0e-6}, {2.0e-3, 2.5e-6, 6.0e-6, 0.5e-6, 3.0e-6}}},
        std::array<double, 2>{0.85, 0.6}, {{{0.01, -0.02}, {0.015, 0.0}}});
    s.thermalExpansion = 1.2e-5;
    s.nonStructuralMass = 0.5;
    return s;
}

// The line of the probe: eccentric, end offsets that differ, from grid 0 to grid 1 (forward) or the reverse.
LineElement probeLine(bool forward)
{
    return LineElement::between(forward ? 0 : 1, forward ? 1 : 0, 0)
        .oriented(ORIENTATION)
        .eccentric(0.02, -0.01)
        .withOffsets({0.0, 0.01, 0.02}, {0.01, 0.0, -0.01});
}

double maxAbsOf(const mh::Vector& v)
{
    double m = 0.0;
    for (const double x : v)
        m = std::max(m, std::fabs(x));
    return m;
}

bool exactlySymmetric(const mh::Matrix& a)
{
    return a == a.transposed();
}

// The 6 rigid motions (translations x, y, z, rotations x, y, z about the origin) on the local unknowns.
std::vector<mh::Vector> rigidMotions(const mh::AloneEdge& edge)
{
    std::vector<mh::Vector> out;
    for (std::size_t m = 0; m < 6; ++m)
    {
        Vec3 axis{};
        axis[m % 3] = 1.0;
        out.push_back(m < 3 ? mh::aloneLineRigidMotion(edge, axis, {}) : mh::aloneLineRigidMotion(edge, {}, axis));
    }
    return out;
}

// Coordinates and edge-traction values below are from the reference crate at 9f55f0b.
constexpr double PROBE_TOLERANCE = 1e-10;

} // namespace

CHECK_CASE("line: section work and rigid rotation inertia of a skew eccentric tapered beam")
{
    const LineSection section = probeSection();
    // The temperature rise of the probe, linear in space.
    const mh::LineTemperatureField temperature = [](const Vec3& p) { return 10.0 + 3.0 * p[0] - 2.0 * p[1] + p[2]; };
    for (const bool forward : {true, false})
    {
        const std::string tag = forward ? "forward: " : "backward: ";
        const Vec3 expectedPoint = forward ? Vec3{0.6808489339589036, -0.0015375174500409303, 0.7058648272057497}
                                           : Vec3{1.206761348945899, 0.22016215044735382, 0.9513549268365313};
        const LineElement line = probeLine(forward);
        const std::optional<AloneLine> built = buildAloneLine(runner, SKEW_A, SKEW_B, line, section, 4);
        if (!built)
            continue;
        const mh::LineData& data = built->data;
        runner.expect(built->forward == forward, tag + "direction of the line on its edge");
        runner.expect(data.dofs.size() == 25, tag + "25 local unknowns");
        // The rotations of the line's grids A and B come last.
        runner.expect(data.dofs[19] == 6 * line.grids[0] + 3 && data.dofs[22] == 6 * line.grids[1] + 3,
                      tag + "rotation unknowns of grids A and B");
        mh::Vector x(data.dofs.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = 1e-4 * std::sin(static_cast<double>(i) * 0.7 + 0.3);
        const auto sample = mh::aloneLineKinematics(built->edge, forward, 0.3);
        auto result = mh::lineStateOfCase(data, sample, line, section, x, temperature);
        const auto* state = std::get_if<mh::LineState>(&result);
        runner.expect(state != nullptr, tag + "section state");
        if (!state)
            continue;
        for (std::size_t c = 0; c < 3; ++c)
            expectRelative(runner, state->point[c], expectedPoint[c], PROBE_TOLERANCE, tag + "axis point");
        const auto chord = mh::sub3(built->edge.upperPoint, built->edge.lowerPoint);
        const double metric =
            mh::norm3(mh::add3(mh::scl3(chord, forward ? 1.0 : -1.0), mh::sub3(line.offsets[1], line.offsets[0]))) /
            built->edge.length();
        const Vec3 spin{0.4, -0.7, 0.2};
        double elastic = 0.0, thermal = 0.0, geometric = 0.0, inertia = 0.0;
        for (const auto& q : data.axial)
        {
            const auto kin = mh::aloneLineKinematics(built->edge, forward, q.fraction);
            const auto response = mh::lineStateOfCase(data, kin, line, section, x, {});
            if (const auto* error = std::get_if<std::string>(&response))
            {
                runner.expect(false, tag + *error);
                break;
            }
            const auto& st = std::get<mh::LineState>(response);
            elastic += q.weight * metric * mh::dot(st.forces, st.strains);
            thermal += q.weight * metric * st.forces[0] * section.thermalExpansion * temperature(st.point);
            const auto du = kin.displacementRate * std::span<const double>(x).first(kin.displacementRate.columns());
            const Vec3 transverse = mh::sub3({du[0], du[1], du[2]}, mh::scl3(kin.tangent, mh::dot(du, kin.tangent)));
            double edgeForce = 0.0;
            for (std::size_t c = 0; c < 3; ++c)
                edgeForce += mh::dot3(kin.tangent, st.axes[c]) * st.forces[c];
            geometric += q.weight * edgeForce * mh::dot3(transverse, transverse);
            const auto sp = section.at(q.fraction);
            const auto omega = lineChecks::toAxes(st.axes, spin);
            const auto speedSquared = [&](const std::array<double, 2>& centre)
            {
                const auto position = mh::add3(st.point, mh::add3(mh::scl3(st.axes[1], centre[0]),
                                                                 mh::scl3(st.axes[2], centre[1])));
                const auto velocity = mh::cross3(spin, position);
                return mh::dot3(velocity, velocity);
            };
            const auto [jx, jy, jz, jyz] = sp.inertia;
            inertia += q.weight * metric *
                       (sp.structuralMassPerLength * speedSquared(sp.neutralAxis) + sp.nonStructuralMassPerLength * speedSquared(sp.massCentre) + jx * omega[0] * omega[0] +
                        jy * omega[1] * omega[1] + jz * omega[2] * omega[2] - 2.0 * jyz * omega[1] * omega[2]);
        }
        const auto rigid = mh::aloneLineRigidMotion(built->edge, {}, spin);
        expectRelative(runner, mh::dot(x, data.stiffness * x), elastic, PROBE_TOLERANCE, tag + "elastic work");
        expectRelative(runner, mh::dot(x, mh::lineThermalLoad(data, section, temperature)), thermal, PROBE_TOLERANCE,
                       tag + "thermal work");
        expectRelative(runner, mh::dot(x, mh::lineGeometricStiffness(data, x, {}) * x), geometric, PROBE_TOLERANCE,
                       tag + "geometric stiffness work");
        expectRelative(runner, mh::dot(rigid, *data.mass * rigid), inertia, PROBE_TOLERANCE,
                       tag + "rigid rotation inertia");
        // Forces and couples per length given from the upper grid.
        const mh::EdgeTractionField traction = [](double s)
        { return std::array<double, 6>{100.0 + 30.0 * s, -50.0 * s * s, 20.0, 5.0, -3.0 * s, 2.0}; };
        const mh::Vector edgeLoad = mh::aloneEdgeTraction(built->edge, false, traction, mh::linePointCount(4));
        expectRelative(runner, edgeLoad[0], 9.44977770930018437e1, PROBE_TOLERANCE, tag + "edge load 0");
        expectRelative(runner, edgeLoad[4], -6.72606877400200798e0, PROBE_TOLERANCE, tag + "edge load 4");
        expectRelative(runner, edgeLoad[9], -1.49999999999999828e-1, PROBE_TOLERANCE, tag + "edge load 9");
        expectRelative(runner, edgeLoad[16], 1.93246987920249247e-1, PROBE_TOLERANCE, tag + "edge load 16");
    }
}

CHECK_CASE("line: a bar and a beam alone have exactly the six rigid modes")
{
    const LineSection bar = LineSection::bar(2.0e11, 8.0e10, 3.0e-3, 4.0e-6, 9.0e-6, 1.0e-6, 5.0e-6, 7800.0);
    LineSection shearBar = bar;
    shearBar.shearFactors = std::array<double, 2>{0.85, 0.6};
    const std::vector<std::pair<std::string, std::pair<LineSection, LineElement>>> cases{
        {"bar", {bar, LineElement::between(0, 1, 0).oriented(ORIENTATION)}},
        {"bar with shear, eccentric",
         {shearBar, LineElement::between(0, 1, 0).oriented(ORIENTATION).eccentric(0.03, 0.01)}},
        {"tapered eccentric beam", {probeSection(), probeLine(true)}},
        {"tapered eccentric beam, reversed", {probeSection(), probeLine(false)}},
    };
    for (const auto& [name, setup] : cases)
    {
        for (const std::size_t k : {2U, 3U, 4U})
        {
            const std::string tag = name + ", k = " + std::to_string(k) + ": ";
            const std::optional<AloneLine> built = buildAloneLine(runner, SKEW_A, SKEW_B, setup.second, setup.first, k);
            if (!built)
                continue;
            const mh::Matrix& kk = built->data.stiffness;
            // Every rigid motion carries no energy.
            const double norm = mh::frobeniusNorm(kk);
            for (const mh::Vector& r : rigidMotions(built->edge))
            {
                const mh::Vector kr = kk * r;
                runner.expect(mh::euclideanNorm(kr) <= 1e-12 * norm * mh::euclideanNorm(r),
                              tag + "rigid motion: " + std::to_string(mh::euclideanNorm(kr) / norm));
            }
            // And there are no others: six zero eigenvalues, the seventh well away.
            const std::vector<double> spectrum = lineChecks::scaledSpectrum(kk);
            runner.expect(spectrum[5] < 1e-12 && spectrum[6] > 1e-6, tag + "six zero eigenvalues (sixth " +
                                                                         std::to_string(spectrum[5]) + ", seventh " +
                                                                         std::to_string(spectrum[6]) + ")");
            runner.expect(lineChecks::zeroEigenvalueCount(kk, 1e-10) == 6, tag + "zero eigenvalue count");
        }
    }
}

CHECK_CASE("line: a rod alone has the rigid modes and its free bending unknowns")
{
    // A rod has no stiffness in bending: of its 4k + 9 unknowns, the axial ones carry k independent strains and the
    // twist ones k + 1 (with a torsion constant). Zero eigenvalues: 2k + 8 with a torsion constant (the 2 (k − 1)
    // functions across the line, the four translations across it at the grids, the four rotations Λ across it, the
    // axial translation and the rotation about the line as a whole: the six rigid modes are among them), 3k + 9
    // without one (the k twist coefficients and the two Λ · t have no stiffness either).
    for (const std::size_t k : {2U, 3U, 4U})
    {
        for (const double j : {5.0e-6, 0.0})
        {
            const LineSection rod = LineSection::rod(2.0e11, 8.0e10, 3.0e-3, j, 7800.0);
            const std::optional<AloneLine> built =
                buildAloneLine(runner, SKEW_A, SKEW_B, LineElement::between(0, 1, 0), rod, k);
            if (!built)
                continue;
            const std::size_t expected = j > 0.0 ? 2 * k + 8 : 3 * k + 9;
            const std::size_t zeros = lineChecks::zeroEigenvalueCount(built->data.stiffness, 1e-10);
            runner.expect(zeros == expected, "rod, k = " + std::to_string(k) + ", J = " + std::to_string(j) + ": " +
                                                 std::to_string(zeros) + " zero eigenvalues, expected " +
                                                 std::to_string(expected));
            for (const mh::Vector& r : rigidMotions(built->edge))
                runner.expect(mh::euclideanNorm(built->data.stiffness * r) <=
                                  1e-12 * mh::frobeniusNorm(built->data.stiffness) * mh::euclideanNorm(r),
                              "rod: rigid motion");
        }
    }
}

CHECK_CASE("line: the axial stiffness at a grid is E A / L exactly")
{
    const double e = 2.0e11;
    const double a = 3.0e-3;
    const Vec3 d = mh::sub3(SKEW_B, SKEW_A);
    const double len = mh::norm3(d);
    const Vec3 t = mh::scl3(d, 1.0 / len);
    const std::vector<std::pair<std::string, std::pair<LineSection, LineElement>>> cases{
        {"rod", {LineSection::rod(e, 8.0e10, a, 5.0e-6, 0.0), LineElement::between(0, 1, 0)}},
        {"bar",
         {LineSection::bar(e, 8.0e10, a, 4.0e-6, 9.0e-6, 0.0, 5.0e-6, 0.0),
          LineElement::between(1, 0, 0).oriented(ORIENTATION)}},
    };
    for (const auto& [name, setup] : cases)
    {
        const std::optional<AloneLine> built = buildAloneLine(runner, SKEW_A, SKEW_B, setup.second, setup.first, 4);
        if (!built)
            continue;
        const mh::Matrix& kk = built->data.stiffness;
        // tᵀ K_gh t over the translations of grids g, h of the edge.
        const auto axial = [&](std::size_t g, std::size_t h)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    sum += t[r] * kk(3 * g + r, 3 * h + c) * t[c];
            return sum;
        };
        expectRelative(runner, axial(0, 0), e * a / len, 1e-13, name + ": E A / L at grid 0");
        expectRelative(runner, axial(1, 1), e * a / len, 1e-13, name + ": E A / L at grid 1");
        expectRelative(runner, axial(0, 1), -e * a / len, 1e-13, name + ": -E A / L between the grids");
    }
}

CHECK_CASE("line: symmetry of the stiffness, the mass and the geometric stiffness")
{
    for (const bool forward : {true, false})
    {
        const LineSection section = probeSection();
        const std::optional<AloneLine> built = buildAloneLine(runner, SKEW_A, SKEW_B, probeLine(forward), section, 4);
        if (!built)
            continue;
        const mh::LineData& data = built->data;
        runner.expect(exactlySymmetric(data.stiffness), "K symmetric");
        runner.expect(exactlySymmetric(*data.mass), "M symmetric");
        mh::Vector x(data.dofs.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = 1e-4 * std::cos(1.3 * static_cast<double>(i));
        runner.expect(exactlySymmetric(mh::lineGeometricStiffness(data, x, {})), "K_g symmetric");
        // Positive semi-definite mass: its smallest eigenvalue is not below the round-off.
        const mh::SymmetricEigen eigen = mh::jacobiEigen(*data.mass);
        const double largest = *std::max_element(eigen.values.begin(), eigen.values.end());
        const double smallest = *std::min_element(eigen.values.begin(), eigen.values.end());
        runner.expect(smallest > -1e-12 * largest, "M positive semi-definite");
    }
}

CHECK_CASE("line: rigid-body mass and the load of an acceleration")
{
    // Translations: the mass of the line, μ ∫ (ρ A + m_ns) ds, μ = |t + do/ds| (the end offsets differ).
    for (const bool forward : {true, false})
    {
        const LineSection section = probeSection();
        const LineElement line = probeLine(forward);
        const std::optional<AloneLine> built = buildAloneLine(runner, SKEW_A, SKEW_B, line, section, 4);
        if (!built)
            continue;
        const mh::LineData& data = built->data;
        const Vec3 chord = forward ? mh::sub3(SKEW_B, SKEW_A) : mh::sub3(SKEW_A, SKEW_B);
        const double len = mh::norm3(chord);
        const double mu = mh::norm3(mh::add3(chord, mh::sub3(line.offsets[1], line.offsets[0]))) / len;
        const double meanArea = 0.5 * (section.area[0] + section.area[1]);
        const double total = mu * len * (section.density * meanArea + section.nonStructuralMass);
        const Vec3 acceleration{0.3, -1.2, 2.0};
        const mh::Vector r = mh::aloneLineRigidMotion(built->edge, acceleration, {});
        const mh::Vector mr = *data.mass * r;
        expectRelative(runner, mh::dot(r, mr), total * mh::dot3(acceleration, acceleration), 1e-13,
                       "mass of a rigid translation");
        const mh::Vector load = mh::lineInertiaLoad(data, acceleration);
        double difference = 0.0;
        for (std::size_t i = 0; i < load.size(); ++i)
            difference = std::max(difference, std::fabs(load[i] - mr[i]));
        runner.expect(difference <= 1e-13 * maxAbsOf(mr), "the inertia load is M times the rigid translation");
    }
    // Rotations of a uniform bar without offset: ∫ m |ω × X|² ds + L ωᵀ Rᵀ J R ω.
    const LineSection bar = LineSection::bar(2.0e11, 8.0e10, 3.0e-3, 4.0e-6, 9.0e-6, 1.0e-6, 5.0e-6, 7800.0);
    const std::optional<AloneLine> built =
        buildAloneLine(runner, SKEW_A, SKEW_B, LineElement::between(0, 1, 0).oriented(ORIENTATION), bar, 4);
    if (!built)
        return;
    const mh::LineSectionPoint p = bar.at(0.5);
    const mh::SectionAxes& axes = built->data.basis;
    const Vec3 d = mh::sub3(SKEW_B, SKEW_A);
    const double len = mh::norm3(d);
    for (std::size_t c = 0; c < 3; ++c)
    {
        Vec3 omega{};
        omega[c] = 1.0;
        const Vec3 a = mh::cross3(omega, SKEW_A);
        const Vec3 b = mh::cross3(omega, d);
        // X(s) = X_A + s d: ∫_0^1 |a + s b|² ds = |a|² + a · b + |b|² / 3.
        const double translation = p.massPerLength * len * (mh::dot3(a, a) + mh::dot3(a, b) + mh::dot3(b, b) / 3.0);
        const Vec3 w = lineChecks::toAxes(axes, omega);
        const double rotary = len * (p.inertia[0] * w[0] * w[0] + p.inertia[1] * w[1] * w[1] +
                                     p.inertia[2] * w[2] * w[2] - 2.0 * p.inertia[3] * w[1] * w[2]);
        const mh::Vector r = mh::aloneLineRigidMotion(built->edge, {}, omega);
        expectRelative(runner, mh::dot(r, *built->data.mass * r), translation + rotary, 1e-13,
                       "mass of a rigid rotation about axis " + std::to_string(c));
    }
}

CHECK_CASE("line: a free expansion is stress-free")
{
    // Constant offsets (μ = 1): a uniform temperature rise moves the line by the homothety u = e0 X, θ = 0.
    LineSection section = probeSection();
    const double rise = 30.0;
    const double e0 = section.thermalExpansion * rise;
    const mh::LineTemperatureField temperature = [rise](const Vec3&) { return rise; };
    for (const bool forward : {true, false})
    {
        const LineElement line = LineElement::between(forward ? 0 : 1, forward ? 1 : 0, 0)
                                     .oriented(ORIENTATION)
                                     .eccentric(0.02, -0.01)
                                     .withOffsets({0.01, 0.02, -0.01}, {0.01, 0.02, -0.01});
        const std::optional<AloneLine> built = buildAloneLine(runner, SKEW_A, SKEW_B, line, section, 4);
        if (!built)
            continue;
        const mh::LineData& data = built->data;
        mh::Vector x(data.dofs.size(), 0.0);
        for (std::size_t c = 0; c < 3; ++c)
        {
            x[c] = e0 * SKEW_A[c];
            x[3 + c] = e0 * SKEW_B[c];
        }
        const mh::Vector f = mh::lineThermalLoad(data, section, temperature);
        const mh::Vector kx = data.stiffness * x;
        double difference = 0.0;
        for (std::size_t i = 0; i < f.size(); ++i)
            difference = std::max(difference, std::fabs(kx[i] - f[i]));
        runner.expect(difference <= 1e-10 * maxAbsOf(f), "K u = f for the homothety");
        const mh::LineSectionPoint p = section.at(0.0);
        for (const double lam : {0.0, 0.3, 1.0})
        {
            const auto state = mh::lineStateOfCase(data, mh::aloneLineKinematics(built->edge, built->forward, lam),
                                                   line, section, x, temperature);
            if (const auto* s = std::get_if<mh::LineState>(&state))
            {
                const double scale = p.stiffness[0][0] * e0;
                for (std::size_t c = 0; c < 6; ++c)
                    runner.expect(std::fabs(s->forces[c]) <= 1e-9 * scale,
                                  "no force " + std::to_string(c) + ": " + std::to_string(s->forces[c] / scale));
                runner.expect(std::fabs(s->strains[0] - e0) <= 1e-12 * e0, "axial strain e0");
                expectRelative(runner, s->freeStrain, e0, 1e-14, "free strain e0");
            }
        }
    }
}

CHECK_CASE("line: geometric stiffness of a stretched bar")
{
    // A uniform stretch ε along the bar: N = E A ε at every point, and a rigid rotation ω has the second-order energy
    // N L |ω × t|².
    const double e = 2.0e11;
    const double a = 3.0e-3;
    const LineSection bar = LineSection::bar(e, 8.0e10, a, 4.0e-6, 9.0e-6, 0.0, 5.0e-6, 0.0);
    const std::optional<AloneLine> built =
        buildAloneLine(runner, SKEW_A, SKEW_B, LineElement::between(1, 0, 0).oriented(ORIENTATION), bar, 4);
    if (!built)
        return;
    const mh::LineData& data = built->data;
    const Vec3 d = mh::sub3(SKEW_B, SKEW_A);
    const double len = mh::norm3(d);
    const Vec3 t = mh::scl3(d, 1.0 / len);
    const double strain = 1e-4;
    mh::Vector x(data.dofs.size(), 0.0);
    for (std::size_t c = 0; c < 3; ++c)
        x[3 + c] = strain * d[c];
    const mh::Vector forces = mh::lineAxialForces(data, x, {});
    for (const double n : forces)
        expectRelative(runner, n, e * a * strain, 1e-12, "N = E A ε");
    const mh::Matrix kg = mh::lineGeometricStiffness(data, x, {});
    for (std::size_t c = 0; c < 3; ++c)
    {
        Vec3 omega{};
        omega[c] = 1.0;
        const mh::Vector r = mh::aloneLineRigidMotion(built->edge, {}, omega);
        const Vec3 wt = mh::cross3(omega, t);
        expectRelative(runner, mh::dot(r, kg * r), e * a * strain * len * mh::dot3(wt, wt), 1e-12,
                       "second-order energy of a rotation about axis " + std::to_string(c));
    }
}
