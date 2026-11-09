#include "oos/aero/dlm/DoubletLatticeSteady.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace oos::aero
{

namespace
{

constexpr double FOUR_PI = 4.0 * std::numbers::pi;
const base::Vec3 FLOW{1.0, 0.0, 0.0};

// The velocity of a straight vortex of unit circulation from a to b at p (Biot-Savart, Katz & Plotkin, Low-Speed
// Aerodynamics, eq. 10.115): v = (r1 × r2) / (4π |r1 × r2|²) · r0 · (r1/|r1| − r2/|r2|), r0 = b − a, r1 = p − a,
// r2 = p − b.
base::Vec3 segmentVelocity(base::Vec3 p, base::Vec3 a, base::Vec3 b, double coreLength)
{
    const base::Vec3 r0 = b - a;
    const base::Vec3 r1 = p - a;
    const base::Vec3 r2 = p - b;
    const base::Vec3 normal = base::cross(r1, r2);
    const double normalSquared = base::dot(normal, normal);
    const double lengthSquared = base::dot(r0, r0);
    // |r1 × r2| = |r0| · the distance from p to the line.
    if (normalSquared <= coreLength * coreLength * lengthSquared)
        return {};
    const double r1Length = base::norm(r1);
    const double r2Length = base::norm(r2);
    if (r1Length <= coreLength || r2Length <= coreLength)
        return {};
    const double projection = base::dot(r0, r1 / r1Length - r2 / r2Length);
    return (projection / (FOUR_PI * normalSquared)) * normal;
}

// The velocity of a straight vortex of unit circulation from a to +∞ along the flow at p: the limit of segmentVelocity
// for b = a + L x as L → ∞, v = (x × r) (1 + x·r/|r|) / (4π |x × r|²), r = p − a.
base::Vec3 trailingVelocity(base::Vec3 p, base::Vec3 a, double coreLength)
{
    const base::Vec3 r = p - a;
    const base::Vec3 normal = base::cross(FLOW, r);
    const double normalSquared = base::dot(normal, normal);
    if (normalSquared <= coreLength * coreLength)
        return {};
    const double rLength = base::norm(r);
    return ((1.0 + base::dot(FLOW, r) / rLength) / (FOUR_PI * normalSquared)) * normal;
}

base::Vec3 mirroredXz(base::Vec3 p)
{
    return {p.x(), -p.y(), p.z()};
}

base::Vec3 mirroredXy(base::Vec3 p)
{
    return {p.x(), p.y(), -p.z()};
}

base::Vec3 stretched(base::Vec3 p, double beta)
{
    return {p.x() / beta, p.y(), p.z()};
}

} // namespace

base::Vec3 DoubletLatticeSteady::horseshoeVelocity(base::Vec3 point, base::Vec3 start, base::Vec3 end,
                                                   double coreLength)
{
    // From +∞ to start (the opposite of a vortex from start to +∞), start to end, end to +∞.
    return segmentVelocity(point, start, end, coreLength) - trailingVelocity(point, start, coreLength) +
           trailingVelocity(point, end, coreLength);
}

linalg::DenseMatrix DoubletLatticeSteady::normalwashFactors(const AeroBoxes& boxes, double mach,
                                                            std::int32_t symmetryXz, std::int32_t symmetryXy,
                                                            base::MemoryBudget& budget)
{
    if (!(mach >= 0.0 && mach < 1.0))
        throw std::invalid_argument(base::concatenate(
            {"DoubletLatticeSteady: expected a subsonic Mach number in [0, 1), found ", std::to_string(mach)}));
    for (const std::int32_t key : {symmetryXz, symmetryXy})
        if (key < -1 || key > 1)
            throw std::invalid_argument(base::concatenate(
                {"DoubletLatticeSteady: expected symmetry keys of -1, 0 or 1, found ", std::to_string(key)}));
    const double beta = std::sqrt(1.0 - mach * mach);
    const std::size_t n = boxes.size();
    linalg::DenseMatrix d(budget, n, n, "doublet-lattice normalwash factors");
    for (std::size_t j = 0; j < n; ++j)
    {
        const AeroBox& source = boxes[j];
        const base::Vec3 start = stretched(source.boundStart, beta);
        const base::Vec3 end = stretched(source.boundEnd, beta);
        const double coreLength = VORTEX_CORE_RATIO * base::norm(source.boundEnd - source.boundStart);
        const double halfChord = 0.5 * source.chord;
        for (std::size_t i = 0; i < n; ++i)
        {
            const AeroBox& receiver = boxes[i];
            const base::Vec3 point = stretched(receiver.controlPoint, beta);
            base::Vec3 velocity = horseshoeVelocity(point, start, end, coreLength);
            // The images: −s·v of the mirrored horseshoe in each plane, sXZ·sXY·v in both.
            if (symmetryXz != 0)
                velocity = velocity - static_cast<double>(symmetryXz) *
                                          horseshoeVelocity(point, mirroredXz(start), mirroredXz(end), coreLength);
            if (symmetryXy != 0)
                velocity = velocity - static_cast<double>(symmetryXy) *
                                          horseshoeVelocity(point, mirroredXy(start), mirroredXy(end), coreLength);
            if (symmetryXz != 0 && symmetryXy != 0)
                velocity = velocity + static_cast<double>(symmetryXz * symmetryXy) *
                                          horseshoeVelocity(point, mirroredXy(mirroredXz(start)),
                                                            mirroredXy(mirroredXz(end)), coreLength);
            d(i, j) = halfChord * base::dot(receiver.normal, velocity);
        }
    }
    return d;
}

void solvePressures(const linalg::DenseLu& factor, linalg::DenseMatrixView normalwash)
{
    if (factor.isSingular())
        throw base::SolverError(base::concatenate({"the doublet-lattice influence matrix is singular at box row ",
                                                   std::to_string(factor.singularColumn()),
                                                   ": two boxes are at the same place or a box has no area; check the "
                                                   "AeroLiftingPanel1 panels for overlaps"}));
    factor.solve(normalwash);
}

} // namespace oos::aero
