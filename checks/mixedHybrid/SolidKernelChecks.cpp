#include "CheckRunner.hpp"
#include "SolidCheckSupport.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
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
#include <utility>
#include <vector>

using namespace oos::checks::solid;

namespace
{

std::string name(mh::VolumeShape shape, std::size_t p, std::size_t pf, std::size_t d)
{
    return std::string(mh::shapeName(shape)) + " p " + std::to_string(p) + " pf " + std::to_string(pf) + " d " +
           std::to_string(d);
}

// The rigid motions carry no energy, nothing else does, the mass of a translation is that of the volume.
void checkRigidAndRank(oos::checks::CheckRunner& runner, const mh::VolumeGeometry& geometry,
                       const mh::VolumeKernel& kernel, const std::string& what)
{
    const mh::Matrix& k = kernel.stiffness();
    double worst = 0.0;
    for (std::size_t mode = 0; mode < 6; ++mode)
    {
        const mh::Vector r = rigidParameters(geometry, mode);
        worst = std::max(worst, mh::euclideanNorm(k * r) / mh::frobeniusNorm(k));
    }
    const std::vector<double> eigenvalues = sortedEigenvalues(k);
    const double largest = eigenvalues.back();
    const auto zero =
        std::count_if(eigenvalues.begin(), eigenvalues.end(), [&](double v) { return std::fabs(v) < 1e-9 * largest; });
    runner.expect(worst < 1e-10, what + ": rigid residual " + std::to_string(worst));
    runner.expect(zero == 6, what + ": " + std::to_string(zero) + " zero eigenvalues");
    if (kernel.mass())
    {
        const mh::Vector t = rigidParameters(geometry, 0);
        const double total = mh::dot(t, *kernel.mass() * t);
        runner.expect(std::fabs(total / (7800.0 * kernel.volume()) - 1.0) < 1e-11, what + ": mass of a translation");
    }
}

// Every shape, distorted, several degrees: the rigid motions carry no energy, nothing else does, the mass is that of
// the volume. One check per shape.
// Reference: volume::tests::rigid_motions_and_rank
void rigidMotionsAndRank(oos::checks::CheckRunner& runner, mh::VolumeShape shape)
{
    const mh::SolidMaterial material = steel();
    const std::vector<GridId> order = numbersOf(shape);
    constexpr std::array<std::pair<std::size_t, std::size_t>, 5> DEGREES{{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {4, 2}}};
    for (const auto& [p, pf] : DEGREES)
    {
        const std::size_t d = std::max<std::size_t>(p, 1);
        const std::optional<mh::VolumeGeometry> geometry = geometryOf(shape, distorted(shape), order, p, pf, false);
        runner.expect(geometry.has_value(), name(shape, p, pf, d) + ": geometry");
        if (!geometry)
            continue;
        const mh::SolidResult<mh::VolumeKernel> kernel =
            mh::VolumeKernel::build(*geometry, order, material, optionsOf(p, pf, d), true, {});
        runner.expect(kernel.ok(), name(shape, p, pf, d) + ": kernel (" + kernel.refusal() + ")");
        if (kernel.ok())
            checkRigidAndRank(runner, *geometry, kernel.value(), name(shape, p, pf, d));
    }
}

} // namespace

CHECK_CASE("solid: rigid motions and rank, tetrahedron")
{
    rigidMotionsAndRank(runner, mh::VolumeShape::Tetrahedron);
}
CHECK_CASE("solid: rigid motions and rank, pyramid")
{
    rigidMotionsAndRank(runner, mh::VolumeShape::Pyramid);
}
CHECK_CASE("solid: rigid motions and rank, pentahedron")
{
    rigidMotionsAndRank(runner, mh::VolumeShape::Pentahedron);
}
CHECK_CASE("solid: rigid motions and rank, hexahedron")
{
    rigidMotionsAndRank(runner, mh::VolumeShape::Hexahedron);
}

// A distorted tetrahedron, every trace degree, every face degree up to it, stress degrees p to p + 2: the rigid
// motions carry no energy, nothing else does, the mass is that of the volume. The test of the tetrahedron of document
// 28, on the kernel that replaces it.
// Reference: solid::tests::rigid_motions_and_rank
CHECK_CASE("solid: rigid motions and rank of the tetrahedron, every degree")
{
    const std::vector<mh::SolidPoint> x{{0.1, 0.0, -0.2}, {1.3, 0.2, 0.1}, {0.3, 1.1, 0.0}, {0.2, 0.4, 0.9}};
    const std::vector<GridId> order{7, 3, 9, 5};
    const mh::SolidMaterial material = steel();
    constexpr std::array<std::pair<std::size_t, std::size_t>, 7> DEGREES{
        {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {3, 2}, {4, 3}, {4, 2}}};
    for (const auto& [p, pf] : DEGREES)
    {
        const std::optional<mh::VolumeGeometry> geometry =
            geometryOf(mh::VolumeShape::Tetrahedron, x, order, p, pf, false);
        runner.expect(geometry.has_value(), "tetrahedron geometry");
        if (!geometry)
            continue;
        for (std::size_t d = p; d <= p + 2; ++d)
        {
            const mh::SolidResult<mh::VolumeKernel> kernel =
                mh::VolumeKernel::build(*geometry, order, material, optionsOf(p, pf, d), true, {});
            runner.expect(kernel.ok(), name(mh::VolumeShape::Tetrahedron, p, pf, d) + ": " + kernel.refusal());
            if (kernel.ok())
                checkRigidAndRank(runner, *geometry, kernel.value(), name(mh::VolumeShape::Tetrahedron, p, pf, d));
        }
    }
}

namespace
{

using Gradient = std::array<std::array<double, 3>, 3>;

// The displacement gradient and translation of the uniform-strain field of the checks.
constexpr Gradient GRADIENT{{{1.0e-3, 2.0e-4, -3.0e-4}, {5.0e-4, -2.0e-3, 1.0e-4}, {-1.0e-4, 3.0e-4, 1.5e-3}}};
constexpr mh::SolidPoint TRANSLATION{1.0e-3, -2.0e-3, 5.0e-4};

// σ = D ε for the displacement gradient A: ε = (A00, A11, A22, A01 + A10, A12 + A21, A20 + A02).
mh::VoigtVector stressOfGradient(const mh::SolidMaterial& material, const Gradient& a)
{
    const mh::VoigtVector strain{a[0][0], a[1][1], a[2][2], a[0][1] + a[1][0], a[1][2] + a[2][1], a[2][0] + a[0][2]};
    mh::VoigtVector out{};
    for (std::size_t c = 0; c < 6; ++c)
        for (std::size_t c2 = 0; c2 < 6; ++c2)
            out[c] += material.stiffness[c][c2] * strain[c2];
    return out;
}

// The centre of the reference element and its corners pulled a tenth of the way towards it.
std::vector<mh::SolidPoint> samplePoints(mh::VolumeShape shape)
{
    const mh::SolidPoint centre = mh::shapeCentre(shape);
    std::vector<mh::SolidPoint> out{centre};
    for (const mh::SolidPoint& vertex : mh::shapeVertices(shape))
        out.push_back(
            {0.9 * vertex[0] + 0.1 * centre[0], 0.9 * vertex[1] + 0.1 * centre[1], 0.9 * vertex[2] + 0.1 * centre[2]});
    return out;
}

double largestOf(std::span<const double> values)
{
    double largest = 0.0;
    for (double value : values)
        largest = std::max(largest, std::fabs(value));
    return largest;
}

// The largest difference between the stresses of β at the sample points and σ, over the largest term of σ.
double stressError(const mh::VolumeKernel& kernel, std::span<const double> beta, const mh::VoigtVector& sigma,
                   double scaleOfStress)
{
    double worst = 0.0;
    for (const mh::SolidPoint& r : samplePoints(kernel.geometry().shape))
    {
        const mh::VoigtVector s = kernel.stressAt(beta, r);
        for (std::size_t c = 0; c < 6; ++c)
            worst = std::max(worst, std::fabs(s[c] - sigma[c]));
    }
    return worst / scaleOfStress;
}

// The consistent forces of a uniform stress on every face of the element.
mh::Vector boundaryForces(const mh::VolumeKernel& kernel, const mh::VoigtVector& sigma)
{
    mh::Vector out(kernel.parameterCount(), 0.0);
    const std::size_t n = kernel.traceDegree() + 4;
    for (std::size_t lf = 0; lf < mh::shapeFaces(kernel.geometry().shape).size(); ++lf)
    {
        const mh::Vector f = kernel.faceStressLoad(lf, [&](const mh::SolidPoint&) { return sigma; }, n);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] += f[i];
    }
    return out;
}

double relativeDifference(const mh::Vector& a, const mh::Vector& b)
{
    mh::Vector difference(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        difference[i] = a[i] - b[i];
    return mh::euclideanNorm(difference) / mh::euclideanNorm(b);
}

// V tr(A σ Bᵀ) = ∫ σ_ab u_k,a v_k,b dV for u = A x, v = B x.
double geometricWork(const Gradient& a, const mh::VoigtVector& sigma, const Gradient& b, double volume)
{
    constexpr std::array<std::array<std::size_t, 3>, 3> VOIGT{{{0, 3, 5}, {3, 1, 4}, {5, 4, 2}}};
    double sum = 0.0;
    for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                sum += a[k][i] * sigma[VOIGT[i][j]] * b[k][j];
    return volume * sum;
}

// The geometric stiffness of the uniform stress of β: symmetric, ∫ σ : ∇u ∇v on linear fields, no work in a
// translation.
void checkGeometric(oos::checks::CheckRunner& runner, const mh::VolumeGeometry& geometry,
                    const mh::VolumeKernel& kernel, std::span<const double> beta, const mh::VoigtVector& sigma,
                    const std::string& what)
{
    const mh::SolidResult<mh::Matrix> geometric = kernel.geometric(beta);
    runner.expect(geometric.ok(), what + ": geometric stiffness");
    if (!geometric.ok())
        return;
    const mh::Matrix& kg = geometric.value();
    runner.expect(asymmetry(kg) == 0.0, what + ": symmetric geometric stiffness");
    const Gradient other{{{0.3, -0.7, 0.2}, {0.5, 0.1, -0.4}, {-0.2, 0.6, 0.9}}};
    const mh::Vector u = linearParameters(geometry, GRADIENT, TRANSLATION);
    const mh::Vector v = linearParameters(geometry, other, {0.4, -0.3, 0.8});
    const double work = mh::dot(u, kg * v);
    const double expected = geometricWork(GRADIENT, sigma, other, kernel.volume());
    runner.expect(std::fabs(work - expected) <= 1e-10 * std::fabs(expected),
                  what + ": geometric work " + std::to_string(work) + " expected " + std::to_string(expected));
    const mh::Vector translation = rigidParameters(geometry, 0);
    runner.expect(mh::euclideanNorm(kg * translation) <= 1e-12 * mh::frobeniusNorm(kg),
                  what + ": no geometric work in a translation");
    // The gradient of φᵀ K_g(β(q)) φ with respect to q: K_g is linear in β = S q, so φᵀ K_g(S q) φ = g · q for every q.
    const std::size_t n = kernel.parameterCount();
    mh::Vector phi(n);
    mh::Vector q(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        phi[k] = std::sin(1.3 * static_cast<double>(k) + 0.4);
        q[k] = 1e-3 * std::cos(0.7 * static_cast<double>(k) - 0.2);
    }
    const mh::SolidResult<mh::Vector> gradient = kernel.geometricGradient(phi);
    const mh::SolidResult<mh::Matrix> atQ = kernel.geometric(kernel.stressCoefficients(q, {}));
    runner.expect(gradient.ok() && atQ.ok(), what + ": geometric gradient");
    if (!gradient.ok() || !atQ.ok())
        return;
    const double form = mh::dot(phi, atQ.value() * phi);
    runner.expect(std::fabs(mh::dot(gradient.value(), q) - form) <= 1e-10 * std::fabs(form),
                  what + ": geometric gradient " + std::to_string(mh::dot(gradient.value(), q)) + " expected " +
                      std::to_string(form));
}

// The invariants of one element of degree 4: six zero eigenvalues exactly and no energy in the rigid motions, a
// uniform strain reproduced (constant stresses, and K q the consistent forces of its stress on the faces), the mass of
// a translation, symmetric matrices, the geometric stiffness of a uniform stress, three interior fields.
void checkInvariants(oos::checks::CheckRunner& runner, mh::VolumeShape shape, std::vector<mh::SolidPoint> corners,
                     bool midSide, const std::string& what)
{
    const std::vector<GridId> order = numbersOf(shape);
    const std::optional<mh::VolumeGeometry> geometry = geometryOf(shape, std::move(corners), order, 4, 4, midSide);
    runner.expect(geometry.has_value() && geometry->curved() == midSide, what + ": geometry");
    if (!geometry)
        return;
    const mh::SolidMaterial material = steel();
    const mh::SolidResult<mh::VolumeKernel> built =
        mh::VolumeKernel::build(*geometry, order, material, optionsOf(4, 4, 4), true, {});
    runner.expect(built.ok(), what + ": kernel (" + built.refusal() + ")");
    if (!built.ok())
        return;
    const mh::VolumeKernel& kernel = built.value();
    const mh::Matrix& k = kernel.stiffness();
    checkRigidAndRank(runner, *geometry, kernel, what);
    double energy = 0.0;
    for (std::size_t mode = 0; mode < 6; ++mode)
    {
        const mh::Vector r = rigidParameters(*geometry, mode);
        energy = std::max(energy, std::fabs(mh::dot(r, k * r)) / (mh::frobeniusNorm(k) * mh::dot(r, r)));
    }
    runner.expect(energy < 1e-12, what + ": energy of the rigid motions " + std::to_string(energy));
    // a uniform strain
    const mh::Vector q = linearParameters(*geometry, GRADIENT, TRANSLATION);
    const mh::VoigtVector sigma = stressOfGradient(material, GRADIENT);
    const mh::Vector beta = kernel.stressCoefficients(q, {});
    const double stressOff = stressError(kernel, beta, sigma, largestOf(sigma));
    runner.expect(stressOff < 1e-10, what + ": uniform stress off by " + std::to_string(stressOff));
    const double forcesOff = relativeDifference(k * q, boundaryForces(kernel, sigma));
    runner.expect(forcesOff < 1e-10, what + ": K q against the forces of the faces " + std::to_string(forcesOff));
    runner.expect(asymmetry(k) == 0.0, what + ": symmetric stiffness");
    runner.expect(kernel.mass() && asymmetry(*kernel.mass()) == 0.0, what + ": symmetric mass");
    checkGeometric(runner, *geometry, kernel, beta, sigma, what);
    const std::optional<mh::SolidInteriorFields>& interior = kernel.interiorFields();
    runner.expect(interior && interior->stiffnesses.size() == 3 &&
                      interior->massCoupling.rows() == kernel.parameterCount() &&
                      std::all_of(interior->stiffnesses.begin(), interior->stiffnesses.end(),
                                  [](double lambda) { return lambda > 0.0; }),
                  what + ": three interior fields of positive stiffness");
}

// One element: straight (an affine image of the reference element) or distorted, with or without a mid-side node on
// every edge (which curves them).
void invariantsOf(oos::checks::CheckRunner& runner, mh::VolumeShape shape, bool isDistorted, bool midSide)
{
    const std::string what = std::string(mh::shapeName(shape)) + (isDistorted ? ", distorted" : ", straight") +
                             (midSide ? " with mid-side nodes" : "");
    checkInvariants(runner, shape, isDistorted ? distorted(shape) : affine(shape), midSide, what);
}

} // namespace

// One check per element, each well below two seconds (the eigenvalues of a stiffness of 294 rows by Jacobi rotations
// are most of it).
CHECK_CASE("solid: invariants, tetrahedron")
{
    invariantsOf(runner, mh::VolumeShape::Tetrahedron, false, false);
}
CHECK_CASE("solid: invariants, distorted tetrahedron")
{
    invariantsOf(runner, mh::VolumeShape::Tetrahedron, true, false);
}
CHECK_CASE("solid: invariants, curved tetrahedron")
{
    invariantsOf(runner, mh::VolumeShape::Tetrahedron, false, true);
}
CHECK_CASE("solid: invariants, distorted curved tetrahedron")
{
    invariantsOf(runner, mh::VolumeShape::Tetrahedron, true, true);
}
CHECK_CASE("solid: invariants, pyramid")
{
    invariantsOf(runner, mh::VolumeShape::Pyramid, false, false);
}
CHECK_CASE("solid: invariants, distorted pyramid")
{
    invariantsOf(runner, mh::VolumeShape::Pyramid, true, false);
}
CHECK_CASE("solid: invariants, curved pyramid")
{
    invariantsOf(runner, mh::VolumeShape::Pyramid, false, true);
}
CHECK_CASE("solid: invariants, distorted curved pyramid")
{
    invariantsOf(runner, mh::VolumeShape::Pyramid, true, true);
}
CHECK_CASE("solid: invariants, pentahedron")
{
    invariantsOf(runner, mh::VolumeShape::Pentahedron, false, false);
}
CHECK_CASE("solid: invariants, distorted pentahedron")
{
    invariantsOf(runner, mh::VolumeShape::Pentahedron, true, false);
}
CHECK_CASE("solid: invariants, curved pentahedron")
{
    invariantsOf(runner, mh::VolumeShape::Pentahedron, false, true);
}
CHECK_CASE("solid: invariants, distorted curved pentahedron")
{
    invariantsOf(runner, mh::VolumeShape::Pentahedron, true, true);
}
CHECK_CASE("solid: invariants, hexahedron")
{
    invariantsOf(runner, mh::VolumeShape::Hexahedron, false, false);
}
CHECK_CASE("solid: invariants, distorted hexahedron")
{
    invariantsOf(runner, mh::VolumeShape::Hexahedron, true, false);
}
CHECK_CASE("solid: invariants, curved hexahedron")
{
    invariantsOf(runner, mh::VolumeShape::Hexahedron, false, true);
}
CHECK_CASE("solid: invariants, distorted curved hexahedron")
{
    invariantsOf(runner, mh::VolumeShape::Hexahedron, true, true);
}
