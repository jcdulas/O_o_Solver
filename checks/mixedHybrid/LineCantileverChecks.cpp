#include "CheckRunner.hpp"
#include "LineCheckSupport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
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

constexpr double E = 2.1e11;
constexpr double G = 8.1e10;

// A rectangle b (along z) by h (along y, the first axis): {area, I1, I2, J}, J that of a thin rectangle with its
// first correction, h b³ / 3 (1 − 0.63 b / h) for b < h (S. P. Timoshenko, J. N. Goodier, Theory of Elasticity, 1970).
std::array<double, 4> rectangle(double b, double h)
{
    const double lo = std::min(b, h);
    const double hi = std::max(b, h);
    return {b * h, b * h * h * h / 12.0, h * b * b * b / 12.0, hi * lo * lo * lo / 3.0 * (1.0 - 0.63 * lo / hi)};
}

LineSection rectangularBar(double b, double h, const std::optional<std::array<double, 2>>& shearFactors)
{
    const auto [a, i1, i2, j] = rectangle(b, h);
    LineSection s = LineSection::bar(E, G, a, i1, i2, 0.0, j, 0.0);
    s.shearFactors = shearFactors;
    return s;
}

// The local unknowns of the clamp (the translations of grid A and its rotations Λ) and of the tip (grid B), for a
// line alone that runs forward (A the lower grid of the edge) or not.
struct EndSlots
{
    std::vector<std::size_t> clamp;
    std::size_t tipTranslation = 0;
    std::size_t tipRotation = 0;
};

EndSlots endSlots(const AloneLine& built)
{
    const std::size_t nx = built.edge.unknownCount();
    const std::size_t a = built.forward ? 0 : 3;
    EndSlots slots;
    slots.clamp = {a, a + 1, a + 2, nx, nx + 1, nx + 2};
    slots.tipTranslation = built.forward ? 3 : 0;
    slots.tipRotation = nx + 3;
    return slots;
}

// The tip displacement and rotation (global) of the cantilever clamped at grid A for the force F and couple M
// (global) at grid B; empty when the clamped stiffness is singular.
std::optional<std::array<Vec3, 2>> solveTip(const AloneLine& built, const Vec3& force, const Vec3& couple)
{
    const EndSlots slots = endSlots(built);
    mh::Vector f(built.data.dofs.size(), 0.0);
    for (std::size_t c = 0; c < 3; ++c)
    {
        f[slots.tipTranslation + c] = force[c];
        f[slots.tipRotation + c] = couple[c];
    }
    const std::optional<mh::Vector> x = lineChecks::solveHeld(built.data.stiffness, f, slots.clamp);
    if (!x)
        return std::nullopt;
    std::array<Vec3, 2> tip{};
    for (std::size_t c = 0; c < 3; ++c)
    {
        tip[0][c] = (*x)[slots.tipTranslation + c];
        tip[1][c] = (*x)[slots.tipRotation + c];
    }
    return tip;
}

// max over the six components of |q − q_exact|, over max |q_exact|.
double tipError(const std::array<Vec3, 2>& tip, const std::array<Vec3, 2>& exact)
{
    double error = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            error = std::max(error, std::fabs(tip[i][c] - exact[i][c]));
            scale = std::max(scale, std::fabs(exact[i][c]));
        }
    }
    return error / scale;
}

// A line from SKEW_A in the direction DIRECTION.
const Vec3 ORIGIN{0.2, -0.4, 0.3};
const Vec3 DIRECTION = mh::unit3(Vec3{2.0, 1.0, 0.5});
const Vec3 ORIENTATION{0.0, 0.3, 1.0};
constexpr double LENGTH = 1.3;

} // namespace

// Port of the reference test src/line.rs::cantilever (written there for the trace of the other shell formulation of
// the reference crate), on the line alone of the mixed hybrid formulation.
CHECK_CASE("line: cantilever of the reference unit test")
{
    const double e = 2.0e11;
    const double g = 8.0e10;
    const double a = 3.0e-3;
    const double i1 = 4.0e-6;
    const double i2 = 9.0e-6;
    const double j = 5.0e-6;
    const double l = 1.7;
    LineSection section = LineSection::bar(e, g, a, i1, i2, 0.0, j, 7800.0);
    section.shearFactors = std::array<double, 2>{0.8, 0.6};
    const Vec3 xa{0.0, 0.0, 0.0};
    const Vec3 xb{l, 0.0, 0.0};
    const std::optional<AloneLine> built =
        buildAloneLine(runner, xa, xb, LineElement::between(0, 1, 0).oriented({0.0, 1.0, 0.0}), section, 4);
    if (!built)
        return;
    const mh::LineData& data = built->data;
    double length = 0.0;
    for (const mh::LineAxialPoint& p : data.axial)
        length += p.weight;
    runner.expect(std::fabs(length - l) < 1e-13, "length of the rule");
    // Rigid motions.
    for (std::size_t m = 0; m < 6; ++m)
    {
        Vec3 axis{};
        axis[m % 3] = 1.0;
        const mh::Vector r =
            m < 3 ? mh::aloneLineRigidMotion(built->edge, axis, {}) : mh::aloneLineRigidMotion(built->edge, {}, axis);
        runner.expect(mh::euclideanNorm(data.stiffness * r) < 1e-9 * mh::frobeniusNorm(data.stiffness),
                      "rigid motion " + std::to_string(m));
    }
    // Grid A held: the flexibility at B is the inverse of the stiffness of the other unknowns.
    const EndSlots slots = endSlots(*built);
    std::vector<std::size_t> freeUnknowns;
    for (std::size_t i = 0; i < data.dofs.size(); ++i)
        if (std::find(slots.clamp.begin(), slots.clamp.end(), i) == slots.clamp.end())
            freeUnknowns.push_back(i);
    mh::Matrix kff(freeUnknowns.size(), freeUnknowns.size());
    for (std::size_t r = 0; r < freeUnknowns.size(); ++r)
        for (std::size_t c = 0; c < freeUnknowns.size(); ++c)
            kff(r, c) = data.stiffness(freeUnknowns[r], freeUnknowns[c]);
    const std::optional<mh::Matrix> inverse = mh::inverse(kff);
    runner.expect(inverse.has_value(), "clamped stiffness regular");
    if (!inverse)
        return;
    const auto tip = [&](std::size_t slot)
    {
        const std::size_t at =
            static_cast<std::size_t>(std::find(freeUnknowns.begin(), freeUnknowns.end(), slot) - freeUnknowns.begin());
        return (*inverse)(at, at);
    };
    const std::size_t u = slots.tipTranslation;
    const std::size_t th = slots.tipRotation;
    const double tolerance = 1e-9;
    expectRelative(runner, tip(u), l / (e * a), tolerance, "axial");
    // y is the first section axis: bending in the plane (x, y) with I1 and k1.
    expectRelative(runner, tip(u + 1), l * l * l / (3.0 * e * i1) + l / (0.8 * g * a), tolerance, "bending y");
    expectRelative(runner, tip(u + 2), l * l * l / (3.0 * e * i2) + l / (0.6 * g * a), tolerance, "bending z");
    expectRelative(runner, tip(th), l / (g * j), tolerance, "torsion");
    expectRelative(runner, tip(th + 1), l / (e * i2), tolerance, "moment about y");
    expectRelative(runner, tip(th + 2), l / (e * i1), tolerance, "moment about z");
    // Mass of a translation along x.
    mh::Vector tx(data.dofs.size(), 0.0);
    tx[0] = 1.0;
    tx[3] = 1.0;
    expectRelative(runner, mh::dot(tx, *data.mass * tx), 7800.0 * a * l, 1e-9, "mass");
}

CHECK_CASE("line: skew cantilever of one element under end loads, against the Timoshenko beam")
{
    // Cantilevers of one element in a skew direction, from either grid, with and without shear flexibility: the six
    // end loads, written out (S. P. Timoshenko, J. M. Gere, Mechanics of Materials, 1972: the cantilever with shear).
    const double b = 0.02;
    const double h = 0.05;
    const auto [a, i1, i2, j] = rectangle(b, h);
    const std::array<double, 2> k{0.85, 0.6};
    for (const bool withShear : {true, false})
    {
        const LineSection section = rectangularBar(b, h, withShear ? std::optional(k) : std::nullopt);
        // Without shear factors the shear stiffness is 1e4 E A.
        const double ga1 = withShear ? k[0] * G * a : mh::RIGID_SHEAR_STIFFNESS_FACTOR * E * a;
        const double ga2 = withShear ? k[1] * G * a : mh::RIGID_SHEAR_STIFFNESS_FACTOR * E * a;
        for (const bool forward : {true, false})
        {
            const Vec3 tipPoint = mh::add3(ORIGIN, mh::scl3(DIRECTION, LENGTH));
            // forward: grid 0 at the origin is A; else grid 1 at the origin is A (the edge then runs from the tip).
            const Vec3 x0 = forward ? ORIGIN : tipPoint;
            const Vec3 x1 = forward ? tipPoint : ORIGIN;
            const LineElement line = LineElement::between(forward ? 0 : 1, forward ? 1 : 0, 0).oriented(ORIENTATION);
            const std::optional<AloneLine> built = buildAloneLine(runner, x0, x1, line, section, 4);
            if (!built)
                continue;
            const mh::SectionAxes& axes = built->data.basis;
            const std::string tag = std::string(withShear ? "with shear, " : "rigid shear, ") +
                                    (forward ? "from the lower grid: " : "from the upper grid: ");
            const double l = LENGTH;
            struct Case
            {
                std::string name;
                Vec3 force;
                Vec3 couple;
                std::array<Vec3, 2> exact;
            };
            const Vec3 zero{};
            const std::vector<Case> cases{
                {"axial force", axes[0], zero, {mh::scl3(axes[0], l / (E * a)), zero}},
                {"force along y",
                 axes[1],
                 zero,
                 {mh::scl3(axes[1], l * l * l / (3.0 * E * i1) + l / ga1), mh::scl3(axes[2], l * l / (2.0 * E * i1))}},
                {"force along z",
                 axes[2],
                 zero,
                 {mh::scl3(axes[2], l * l * l / (3.0 * E * i2) + l / ga2), mh::scl3(axes[1], -l * l / (2.0 * E * i2))}},
                {"torque", zero, axes[0], {zero, mh::scl3(axes[0], l / (G * j))}},
                {"moment about y",
                 zero,
                 axes[1],
                 {mh::scl3(axes[2], -l * l / (2.0 * E * i2)), mh::scl3(axes[1], l / (E * i2))}},
                {"moment about z",
                 zero,
                 axes[2],
                 {mh::scl3(axes[1], l * l / (2.0 * E * i1)), mh::scl3(axes[2], l / (E * i1))}},
            };
            for (const Case& c : cases)
            {
                const std::optional<std::array<Vec3, 2>> tip = solveTip(*built, c.force, c.couple);
                runner.expect(tip.has_value(), tag + c.name + ": solved");
                if (tip)
                {
                    const double error = tipError(*tip, c.exact);
                    runner.expect(error <= 1e-10, tag + c.name + ": " + std::to_string(error));
                }
            }
        }
    }
}

CHECK_CASE("line: skew cantilever with an offset axis and a neutral axis, against the beam on the edge")
{
    // The axis offset from the grids (end offsets and eccentricity, constant) and the neutral axis off the reference
    // axis: the line is the beam on its edge with the section law TᵀDT (document 34 of the reference crate, 8.1:
    // Γ_axis = Γ_edge + (K × o) · x; the shear and the torsion stay those of the edge). Every end load, against the
    // unit load method on that law (lineChecks::cantileverFlexibility).
    const double b = 0.02;
    const double h = 0.05;
    LineSection section = rectangularBar(b, h, std::array<double, 2>{0.85, 0.6});
    section.neutralAxis = {{{0.004, -0.007}, {0.004, -0.007}}};
    const Vec3 tipPoint = mh::add3(ORIGIN, mh::scl3(DIRECTION, LENGTH));
    for (const bool forward : {true, false})
    {
        const LineElement line = LineElement::between(forward ? 0 : 1, forward ? 1 : 0, 0)
                                     .oriented(ORIENTATION)
                                     .eccentric(0.03, -0.02)
                                     .withOffsets({0.01, -0.02, 0.015}, {0.01, -0.02, 0.015});
        const std::optional<AloneLine> built =
            buildAloneLine(runner, forward ? ORIGIN : tipPoint, forward ? tipPoint : ORIGIN, line, section, 4);
        if (!built)
            continue;
        const mh::SectionAxes& axes = built->data.basis;
        const Vec3 eccentricity =
            mh::add3(mh::scl3(axes[1], line.eccentricity[0]), mh::scl3(axes[2], line.eccentricity[1]));
        const Vec3 offset = mh::add3(line.offsets[0], eccentricity);
        const mh::Matrix law = lineChecks::lawAtEdge(section.at(0.5), lineChecks::toAxes(axes, offset));
        const std::optional<mh::Matrix> flexibility = lineChecks::cantileverFlexibility(law, LENGTH);
        runner.expect(flexibility.has_value(), "closed form");
        if (!flexibility)
            continue;
        for (std::size_t load = 0; load < 6; ++load)
        {
            // A unit load along a section axis, in global components.
            const Vec3 force = load < 3 ? axes[load] : Vec3{};
            const Vec3 couple = load < 3 ? Vec3{} : axes[load - 3];
            std::array<Vec3, 2> exact{};
            const mh::Matrix& flex = *flexibility;
            exact[0] = lineChecks::fromAxes(axes, {flex(0, load), flex(1, load), flex(2, load)});
            exact[1] = lineChecks::fromAxes(axes, {flex(3, load), flex(4, load), flex(5, load)});
            const std::optional<std::array<Vec3, 2>> tip = solveTip(*built, force, couple);
            runner.expect(tip.has_value(), "solved");
            if (tip)
            {
                const double error = tipError(*tip, exact);
                runner.expect(error <= 1e-10, std::string(forward ? "forward" : "backward") + ", load " +
                                                  std::to_string(load) + ": " + std::to_string(error));
            }
        }
    }
    // The case of the reference suite: a force along the line at the grid of an axis offset in the plane (x, y)
    // bends the bar, u = P (L / E A + e² L / E I1), deflection P e L² / (2 E I1) (rigid shear).
    const double ecc = 0.04;
    const LineSection bar = rectangularBar(b, h, std::nullopt);
    const auto [a, i1, i2, j] = rectangle(b, h);
    const LineElement offsetLine =
        LineElement::between(0, 1, 0).oriented({0.0, 0.0, 1.0}).withOffsets({0.0, 0.0, ecc}, {0.0, 0.0, ecc});
    const std::optional<AloneLine> built =
        buildAloneLine(runner, {0.0, 0.0, 0.0}, {LENGTH, 0.0, 0.0}, offsetLine, bar, 4);
    if (!built)
        return;
    const double p = 1000.0;
    const std::optional<std::array<Vec3, 2>> tip = solveTip(*built, {p, 0.0, 0.0}, {});
    if (!tip)
        return;
    expectRelative(runner, (*tip)[0][0], p * (LENGTH / (E * a) + ecc * ecc * LENGTH / (E * i1)), 1e-10,
                   "offset axis: displacement along the line");
    expectRelative(runner, std::fabs((*tip)[0][2]), p * ecc * LENGTH * LENGTH / (2.0 * E * i1), 1e-10,
                   "offset axis: deflection");
    mh::Vector x(built->data.dofs.size(), 0.0);
    {
        const EndSlots slots = endSlots(*built);
        mh::Vector f(built->data.dofs.size(), 0.0);
        f[slots.tipTranslation] = p;
        x = *lineChecks::solveHeld(built->data.stiffness, f, slots.clamp);
    }
    const auto state =
        mh::lineStateOfCase(built->data, mh::aloneLineKinematics(built->edge, true, 0.5), offsetLine, bar, x, {});
    if (const auto* s = std::get_if<mh::LineState>(&state))
    {
        expectRelative(runner, s->forces[0], p, 1e-10, "offset axis: axial force");
        expectRelative(runner, std::fabs(s->forces[5]), p * ecc, 1e-8, "offset axis: moment");
    }
}

CHECK_CASE("line: skew cantilever under a uniform load")
{
    // q per length along a section axis: deflection q L⁴ / (8 E I) + q L² / (2 k G A), rotation q L³ / (6 E I); at the
    // clamp the moment q L² / 2, the shear force q L and the stress of the outer fibre.
    const double b = 0.02;
    const double h = 0.05;
    const auto [a, i1, i2, j] = rectangle(b, h);
    const std::array<double, 2> k{0.85, 0.6};
    const LineSection section = rectangularBar(b, h, k);
    const double q = 3.0e3;
    const double l = LENGTH;
    for (const bool forward : {true, false})
    {
        const Vec3 tipPoint = mh::add3(ORIGIN, mh::scl3(DIRECTION, LENGTH));
        const LineElement line = LineElement::between(forward ? 0 : 1, forward ? 1 : 0, 0).oriented(ORIENTATION);
        const std::optional<AloneLine> built =
            buildAloneLine(runner, forward ? ORIGIN : tipPoint, forward ? tipPoint : ORIGIN, line, section, 4);
        if (!built)
            continue;
        const mh::SectionAxes& axes = built->data.basis;
        for (const std::size_t axis : {1U, 2U})
        {
            const std::string tag =
                std::string(forward ? "forward" : "backward") + (axis == 1 ? ", along y: " : ", along z: ");
            const double inertia = axis == 1 ? i1 : i2;
            const double shear = axis == 1 ? k[0] * G * a : k[1] * G * a;
            const Vec3 direction = axes[axis];
            const mh::EdgeTractionField traction = [&](double)
            { return std::array<double, 6>{q * direction[0], q * direction[1], q * direction[2], 0.0, 0.0, 0.0}; };
            const mh::Vector edgeLoad =
                mh::aloneEdgeTraction(built->edge, built->forward, traction, mh::linePointCount(built->edge.degree));
            mh::Vector f(built->data.dofs.size(), 0.0);
            for (std::size_t i = 0; i < edgeLoad.size(); ++i)
                f[i] = edgeLoad[i];
            const EndSlots slots = endSlots(*built);
            const std::optional<mh::Vector> x = lineChecks::solveHeld(built->data.stiffness, f, slots.clamp);
            if (!x)
                continue;
            Vec3 u{};
            Vec3 theta{};
            for (std::size_t c = 0; c < 3; ++c)
            {
                u[c] = (*x)[slots.tipTranslation + c];
                theta[c] = (*x)[slots.tipRotation + c];
            }
            const double l4 = l * l * l * l;
            expectRelative(runner, mh::dot3(u, direction), q * l4 / (8.0 * E * inertia) + q * l * l / (2.0 * shear),
                           1e-10, tag + "deflection");
            // The rotation about z of a deflection along y is positive, about y of one along z negative.
            const double rotation = mh::dot3(theta, axes[axis == 1 ? 2 : 1]) * (axis == 1 ? 1.0 : -1.0);
            expectRelative(runner, rotation, q * l * l * l / (6.0 * E * inertia), 1e-10, tag + "rotation");
            const mh::LineKinematics clamp = mh::aloneLineKinematics(built->edge, built->forward, 0.0);
            const auto state = mh::lineStateOfCase(built->data, clamp, line, section, *x, {});
            if (const auto* s = std::get_if<mh::LineState>(&state))
            {
                expectRelative(runner, std::fabs(s->forces[axis == 1 ? 5 : 4]), 0.5 * q * l * l, 1e-10,
                               tag + "moment at the clamp");
                expectRelative(runner, std::fabs(s->forces[axis]), q * l, 1e-10, tag + "shear force at the clamp");
                const double y = axis == 1 ? 0.5 * h : 0.0;
                const double z = axis == 1 ? 0.0 : 0.5 * b;
                const double fibre = std::fabs(mh::lineStress(section, *s, y, z));
                const double distance = axis == 1 ? 0.5 * h : 0.5 * b;
                expectRelative(runner, fibre, 0.5 * q * l * l * distance / inertia, 1e-10,
                               tag + "stress of the outer fibre at the clamp");
            }
        }
    }
}
