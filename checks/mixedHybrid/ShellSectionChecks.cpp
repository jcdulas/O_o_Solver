#include "CheckRunner.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>

namespace mh = oos::elements::mixedHybrid;

namespace
{

double maxDifference(const mh::Matrix33& a, const mh::Matrix33& b)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            worst = std::fmax(worst, std::fabs(a[i][j] - b[i][j]));
    return worst;
}

} // namespace

// A homogeneous section: A = Q t, D = Q t³/12, and the equilibrium shear stiffness of one layer is 5/6 G t (the
// parabolic distribution, 3/2 Q/t at the middle, zero on the faces).
CHECK_CASE("shell: isotropic section and its shear distribution")
{
    const double e = 70000.0;
    const double nu = 0.3;
    const double t = 0.05;
    const double g = e / (2.0 * (1.0 + nu));
    const mh::Section s = mh::Section::isotropic(e, nu, t, 5.0 / 6.0, 2.7e-9, 2.3e-5);
    const mh::Matrix33 q = mh::isotropicReducedStiffness(e, nu);
    runner.expectNear(s.a[0][0], q[0][0] * t, 1e-14, "A11");
    runner.expectNear(s.d[2][2], q[2][2] * t * t * t / 12.0, 1e-14, "D66");
    runner.expect(s.shear.size() == 1, "one layer of shear distribution");
    const std::optional<mh::ShearDistribution> distribution = s.shearDistribution();
    runner.expect(distribution && distribution->stiffness.has_value(), "shear stiffness of the distribution");
    if (distribution && distribution->stiffness)
    {
        runner.expectNear((*distribution->stiffness)[0][0] / (5.0 / 6.0 * g * t), 1.0, 1e-12, "5/6 G t (xz)");
        runner.expectNear((*distribution->stiffness)[1][1] / (5.0 / 6.0 * g * t), 1.0, 1e-12, "5/6 G t (yz)");
        runner.expectNear((*distribution->stiffness)[0][1] / (g * t), 0.0, 1e-12, "no coupling");
    }
    const mh::PlyShear& ply = s.shear[0];
    const double z0 = s.plies[0].z0;
    runner.expectNear(ply.at(z0, 0.0)[0][0] * t, 1.5, 1e-12, "3/2 Q/t at the middle");
    runner.expectNear(ply.at(z0, s.plies[0].z1)[0][0] * t, 0.0, 1e-12, "zero on the top face");
    runner.expectNear(ply.at(z0, z0)[0][0], 0.0, 1e-12, "zero on the bottom face");
}

// A laminate of one layer of an isotropic material is the homogeneous section.
CHECK_CASE("shell: laminate of one layer")
{
    const double e = 70000.0;
    const double nu = 0.3;
    const double t = 0.05;
    const double g = e / (2.0 * (1.0 + nu));
    const mh::LaminateLayer layer{e, e, nu, g, g, g, 2.7e-9, 2.3e-5, 2.3e-5, t, 0.4};
    const mh::Section l = mh::Section::laminate({&layer, 1});
    const mh::Section s = mh::Section::isotropic(e, nu, t, 5.0 / 6.0, 2.7e-9, 2.3e-5);
    runner.expect(maxDifference(l.a, s.a) < 1e-9 * s.a[0][0], "A");
    runner.expect(maxDifference(l.b, mh::Matrix33{}) < 1e-9 * s.a[0][0] * t, "B = 0");
    runner.expect(maxDifference(l.d, s.d) < 1e-9 * s.d[0][0], "D");
    runner.expect(l.h.has_value(), "transverse shear stiffness");
    if (l.h)
        runner.expectNear((*l.h)[0][0] / (*s.h)[0][0], 1.0, 1e-12, "H");
    runner.expectNear(l.thermalA[0] / s.thermalA[0], 1.0, 1e-12, "thermal membrane");
    runner.expectNear(l.thermalA[2] / s.thermalA[0], 0.0, 1e-12, "no thermal shear");
    runner.expectNear(l.i0 / s.i0, 1.0, 1e-14, "I0");
    runner.expectNear(l.i2 / s.i2, 1.0, 1e-12, "I2");
    // a layer rigid in transverse shear: no transverse shear flexibility
    const mh::LaminateLayer rigid{e, e, nu, g, 0.0, 0.0, 2.7e-9, 0.0, 0.0, t, 0.0};
    runner.expect(!mh::Section::laminate({&rigid, 1}).h.has_value(), "Kirchhoff laminate");
}

// The transformations of the axes: T(θ) T(−θ) = I, an isotropic stiffness does not turn, and the section turned by
// an angle has the stiffness turned.
CHECK_CASE("shell: transformations of the section axes")
{
    const double angle = 0.7;
    const mh::Matrix33 t = mh::strainTransformation(angle);
    const mh::Matrix33 back = mh::strainTransformation(-angle);
    mh::Matrix33 product{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k)
                product[i][j] += t[i][k] * back[k][j];
    mh::Matrix33 identity{};
    for (std::size_t i = 0; i < 3; ++i)
        identity[i][i] = 1.0;
    runner.expect(maxDifference(product, identity) < 1e-15, "T(θ) T(−θ) = I");
    const mh::Matrix33 q = mh::isotropicReducedStiffness(70000.0, 0.3);
    runner.expect(maxDifference(mh::rotateStiffness(q, angle), q) < 1e-10, "isotropic stiffness invariant");
    const mh::Matrix33 q0 = mh::orthotropicReducedStiffness(140000.0, 10000.0, 0.3, 5000.0, 0.0);
    const mh::Matrix33 q90 = mh::orthotropicReducedStiffness(140000.0, 10000.0, 0.3, 5000.0, std::numbers::pi / 2.0);
    runner.expectNear(q90[0][0], q0[1][1], 1e-10, "a ply at 90°");
    runner.expectNear(q90[1][1], q0[0][0], 1e-10, "a ply at 90° (2)");
    const mh::Section s = mh::Section::isotropic(70000.0, 0.3, 0.05, 5.0 / 6.0, 2.7e-9, 2.3e-5);
    const mh::SectionPoint thick = s.at(0.1, angle);
    runner.expectNear(thick.a[0][0] / s.a[0][0], 2.0, 1e-13, "A scales with t");
    runner.expectNear(thick.d[0][0] / s.d[0][0], 8.0, 1e-13, "D scales with t³");
    runner.expectNear(thick.mass[2] / s.i2, 8.0, 1e-13, "I2 scales with t³");
    const mh::ShellSection field(s, {0.05, 0.07, 0.09, 0.03}, {1.0, 0.0, 0.0});
    runner.expectNear(field.thicknessAt(0.0, 0.0), 0.06, 1e-15, "mean thickness at the centre");
    runner.expectNear(field.thicknessAt(1.0, -1.0), 0.07, 1e-15, "corner thickness");
    const mh::SurfaceFrame frame{mh::SpaceVector{0.0, 1.0, 0.0}, mh::SpaceVector{-1.0, 0.0, 0.0}};
    runner.expectNear(field.angle(frame), -std::numbers::pi / 2.0, 1e-15, "material angle");
}
