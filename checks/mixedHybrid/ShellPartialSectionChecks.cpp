#include "CheckRunner.hpp"
#include "ShellCheckElements.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
using namespace oos::checks::shell;

// The membrane-only and bending-only sections of the shells (docs/mixed-hybrid-formulation-spec.md section 4.1).

namespace
{

constexpr std::size_t DEGREE = 4;
constexpr double YOUNG = 70000.0;
constexpr double POISSON = 0.3;
constexpr double THICKNESS = 0.05;
constexpr double EXPANSION = 2.3e-5;

// The isotropic section of the checks with both parts, the membrane alone or the bending alone.
mh::Section sectionWith(mh::SectionParts parts)
{
    mh::Section s = mh::Section::isotropic(YOUNG, POISSON, THICKNESS, 5.0 / 6.0, 2.7e-9, EXPANSION);
    s.parts = parts;
    if (parts == mh::SectionParts::MembraneOnly)
    {
        s.b = {};
        s.d = {};
        s.h.reset();
        s.thermalB = {};
        s.thermalD = {};
        s.i1 = 0.0;
        s.i2 = 0.0;
    }
    else if (parts == mh::SectionParts::BendingOnly)
    {
        s.a = {};
        s.b = {};
        s.thermalA = {};
        s.thermalB = {};
    }
    return s;
}

// The flat test elements in the plane z = 0 with a section of the given parts.
std::vector<std::pair<std::string, TestElement>> flatElements(mh::SectionParts parts)
{
    const mh::ShellSection section = mh::ShellSection::uniform(sectionWith(parts), {1.0, 0.0, 0.0});
    const TestSurface quadrilateral =
        bilinearQuadrilateral({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{2.2, 0.1, 0.0},
                               mh::SpaceVector{1.9, 1.7, 0.0}, mh::SpaceVector{-0.2, 1.2, 0.0}});
    const TestSurface triangle = affineTriangle(
        {mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{1.7, 0.3, 0.0}, mh::SpaceVector{0.4, 1.4, 0.0}});
    return {{"flat quadrilateral", makeElement(quadrilateral, DEGREE, section, 0.0, {1.0, -1.0, 1.0, -1.0}, false)},
            {"flat triangle", makeElement(triangle, DEGREE, section, 0.0, {1.0, -1.0, 1.0}, false)}};
}

// A pseudo-random vector of the local unknowns kept by `keep` (the others zero).
mh::Vector randomUnknowns(std::size_t count, const std::function<bool(std::size_t)>& keep)
{
    Xorshift random;
    mh::Vector x(count, 0.0);
    for (std::size_t i = 0; i < count; ++i)
    {
        const double value = random.next();
        if (keep(i))
            x[i] = value;
    }
    return x;
}

double maxAbsOf(const mh::Vector& v)
{
    double top = 0.0;
    for (const double x : v)
        top = std::fmax(top, std::fabs(x));
    return top;
}

// The 6 × 6 matrix [A B; Bᵀ D] of a section point.
mh::Matrix membraneBending(const mh::SectionPoint& point)
{
    mh::Matrix m(6, 6);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            m(i, j) = point.a[i][j];
            m(i, 3 + j) = point.b[i][j];
            m(3 + i, j) = point.b[j][i];
            m(3 + i, 3 + j) = point.d[i][j];
        }
    return m;
}

} // namespace

// On a flat element without coupling the membrane and the bending of the section with both parts do not mix: the
// membrane-only element gives its stiffness on the in-plane displacements and nothing on the others (the normal
// displacement, the shear, the rotations about the edges), the bending-only element the converse.
CHECK_CASE("shell: membrane-only and bending-only sections split a flat element")
{
    const auto full = flatElements(mh::SectionParts::MembraneAndBending);
    const auto membrane = flatElements(mh::SectionParts::MembraneOnly);
    const auto bending = flatElements(mh::SectionParts::BendingOnly);
    for (std::size_t e = 0; e < full.size(); ++e)
    {
        const TestElement& reference = full[e].second;
        const std::size_t n1 = reference.re->h1Count;
        const std::size_t count = reference.data.matrices.unknownCount;
        // the in-plane unknowns: the x and y components of the displacement functions
        const auto inPlane = [&](std::size_t i) { return i < 3 * n1 && i % 3 != 2; };
        const mh::Vector xIn = randomUnknowns(count, inPlane);
        const mh::Vector xOut = randomUnknowns(count, [&](std::size_t i) { return !inPlane(i); });
        const mh::Matrix& k = reference.data.matrices.stiffness;
        const mh::Vector kIn = k * xIn;
        const mh::Vector kOut = k * xOut;
        const double scale = std::fmax(maxAbsOf(kIn), maxAbsOf(kOut));
        const mh::Matrix& km = membrane[e].second.data.matrices.stiffness;
        const mh::Matrix& kb = bending[e].second.data.matrices.stiffness;
        const auto difference = [](const mh::Vector& a, const mh::Vector& b)
        {
            double top = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i)
                top = std::fmax(top, std::fabs(a[i] - b[i]));
            return top;
        };
        const std::string name = full[e].first;
        runner.expect(difference(km * xIn, kIn) < 1e-10 * scale, name + ": membrane-only on the in-plane unknowns");
        runner.expect(maxAbsOf(km * xOut) < 1e-10 * scale, name + ": membrane-only on the other unknowns");
        runner.expect(difference(kb * xOut, kOut) < 1e-10 * scale, name + ": bending-only on the other unknowns");
        runner.expect(maxAbsOf(kb * xIn) < 1e-10 * scale, name + ": bending-only on the in-plane unknowns");
        runner.expect(mh::maxAbs(membrane[e].second.data.matrices.pairing) == 0.0,
                      name + ": membrane-only has no moments");
        runner.expect(asymmetry(km) < 1e-14 && asymmetry(kb) < 1e-14, name + ": symmetric");
    }
}

// The compliance of a section that lacks a part inverts the part it has (the strains of the other are zero), and is a
// generalised inverse of an offset membrane (B = z0 A, D = z0² A); the free strains of a temperature are those of
// the part the section has.
CHECK_CASE("shell: compliance and free strains of membrane-only and bending-only sections")
{
    const mh::SurfaceFrame frame{mh::SpaceVector{1.0, 0.0, 0.0}, mh::SpaceVector{0.0, 1.0, 0.0}};
    const mh::Matrix22 flat{};
    const double rise = 40.0;
    const double gradient = 300.0;
    for (const mh::SectionParts parts : {mh::SectionParts::MembraneOnly, mh::SectionParts::BendingOnly})
    {
        const std::string name = parts == mh::SectionParts::MembraneOnly ? "membrane-only" : "bending-only";
        const mh::ShellSection section = mh::ShellSection::uniform(sectionWith(parts), {1.0, 0.0, 0.0});
        for (const double offset : {0.0, 0.02})
        {
            const mh::SectionPoint point = mh::sectionAt(section, offset, 0.0, 0.0, frame);
            const mh::Matrix m = membraneBending(point);
            const mh::Matrix c = mh::membraneBendingCompliance(point);
            const mh::Matrix mcm = m * c * m;
            runner.expect(mh::maxAbs(mcm - m) < 1e-10 * mh::maxAbs(m),
                          name + " offset " + std::to_string(offset) + ": M C M = M");
        }
        const mh::SectionPoint point = mh::sectionAt(section, 0.0, 0.0, 0.0, frame);
        const auto [e0, k0] = mh::freeStrains(point, flat, {rise, gradient});
        const bool membraneOnly = parts == mh::SectionParts::MembraneOnly;
        for (std::size_t i = 0; i < 2; ++i)
        {
            runner.expectNear(e0[i] / EXPANSION, membraneOnly ? rise : 0.0, 1e-10, name + ": free membrane strain");
            runner.expectNear(k0[i] / EXPANSION, membraneOnly ? 0.0 : gradient, 1e-10, name + ": free curvature");
        }
        runner.expect(std::fabs(e0[2]) + std::fabs(k0[2]) < 1e-18, name + ": no free shear strain");
    }
}
