#include "oos/elements/mixedHybrid/model/ShellTraceDefect.hpp"

#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements::mixedHybrid
{
namespace
{
void addMembraneWork(const ShellElementData& element, const ShellReferenceElement& re, const ReferencePoint& at,
                     const ShellEdgeRotationFrame& frame, std::size_t k, double t, double weight,
                     std::span<const double> x, const ShellSolutionCoefficients& coefficients,
                     const TemperatureAtPosition& temperature, ShellEdgeDefect& side)
{
    const auto g = geometryAt(re, element.definition(), at[0], at[1]);
    const auto forces = gridResultants(re, element, at[0], at[1], x, coefficients, temperature);
    const auto outward = scl3(cross3(frame.tangent, g.nu), element.edgeSigns[k]);
    const double a = dot3(outward, g.e[0]);
    const double b = dot3(outward, g.e[1]);
    const auto traction =
        add3(scl3(g.e[0], a * forces[0] + b * forces[2]), scl3(g.e[1], a * forces[2] + b * forces[1]));
    // A quartic edge displacement with zero endpoints and zero cubic edge moments
    // is (1-t^2)(5t^2-1)/W. Its membrane work uses physical arc length.
    const double w = element.weight.evaluate(at[0], at[1]);
    const double dl = weight * frame.measure / (w * w);
    side.membraneWork = add3(side.membraneWork, scl3(traction, dl * (1 - t * t) * (5 * t * t - 1) / w));
    side.membraneScale += dl * norm3(traction) / w;
}
} // namespace

void appendTraceDefects(const ShellPart& part, std::size_t shell, const ShellElementData& element,
                        std::span<const double> x, const ShellSolutionCoefficients& coefficients,
                        const TemperatureAtPosition& temperature, std::vector<ShellEdgeDefect>& defects)
{
    static const auto rule = unitIntervalRule(5);
    const auto& re = part.referenceElement(element);
    const bool bends = part.carriesMoments(shell);
    std::array<SpaceVector, 4> cornerU{};
    for (std::size_t i = 0; i < re.h1Count; ++i)
        if (re.h1Kinds[i].place == FunctionKind::Place::Vertex)
            for (std::size_t c = 0; c < 3; ++c)
                cornerU[re.h1Kinds[i].entity][c] = x[3 * i + c];
    double membraneStiffness = 0.0;
    for (const auto& rowA : part.input().shells[shell].section.a)
        for (double a : rowA)
            membraneStiffness = std::max(membraneStiffness, std::abs(a));
    for (std::size_t k = 0; k < element.cornerCount; ++k)
    {
        if (part.edgeDegree(element.edges[k]) != 3)
            continue;
        ShellEdgeDefect side{element.edges[k], 0.0, 0.0};
        const bool shared =
            part.topology().edgeElements[side.edge].size() == 2 && part.topology().partners[side.edge].contains(shell);
        // The kinematic rotation and the hybrid rotation have the same orientation.
        // Project their difference on P0..P3 in the canonical edge parameter. P3
        // probes the additional quartic trace, P0..P2 measure weak compatibility.
        std::array<double, 4> moments{};
        double measure = 0.0;
        double scale = 0.0;
        const auto first = shellRotationUnknown(re, k, 0);
        for (const auto& q : rule)
        {
            const auto at = part.edgeReferencePoint(shell, k, q.point);
            const auto frame = part.edgeRotationFrame(shell, k, q.point);
            const auto theta = bends ? rotationAt(re, element.definition(), at[0], at[1], x) : SpaceVector{};
            const double t = 2.0 * q.point - 1.0;
            const std::array<double, 4> p{1.0, t, 0.5 * (3.0 * t * t - 1.0), 0.5 * t * (5.0 * t * t - 3.0)};
            const double hybrid = bends ? x[first] + p[1] * x[first + 1] + p[2] * x[first + 2] : 0.0;
            if (shared)
            {
                addMembraneWork(element, re, at, frame, k, t, q.weight, x, coefficients, temperature, side);
            }
            const double trace = frame.measure * dot3(theta, frame.tangent);
            for (std::size_t l = 0; l < p.size(); ++l)
                moments[l] += q.weight * (trace - hybrid) * p[l];
            measure += q.weight * frame.measure;
            scale = std::max({scale, norm3(theta), std::abs(hybrid) / frame.measure});
        }
        double squared = 0.0;
        for (std::size_t l = 0; l < moments.size(); ++l)
            squared += (2 * l + 1) * moments[l] * moments[l];
        side.defect = std::sqrt(squared) / measure;
        const auto next = (k + 1) % element.cornerCount;
        const double chord = norm3(
            sub3(part.input().grids[element.nodes[k]].position, part.input().grids[element.nodes[next]].position));
        const double increment = norm3(sub3(cornerU[k], cornerU[next]));
        // Keep a numerically absent field (membrane in pure bending, rotations in
        // pure stretch) from setting its own normalization at roundoff level.
        side.scale = std::max(scale, increment / chord);
        side.membraneScale =
            std::max(side.membraneScale, 1.e-4 * membraneStiffness * std::max(increment, scale * chord));
        defects.push_back(side);
    }
}
} // namespace oos::elements::mixedHybrid
