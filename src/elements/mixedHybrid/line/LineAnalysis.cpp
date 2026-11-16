#include "oos/elements/mixedHybrid/line/LineAnalysis.hpp"

#include "oos/elements/mixedHybrid/core/Quadrature.hpp"

#include <utility>

namespace oos::elements::mixedHybrid
{

std::variant<AloneEdge, std::string> aloneEdgeOfLine(const LineElement& line, const LineSection& section,
                                                     const std::array<std::size_t, 2>& edgeGrids,
                                                     const Vec3Of<double>& lowerPoint, const Vec3Of<double>& upperPoint,
                                                     std::size_t degree, std::size_t firstEdgeUnknown)
{
    if (norm3(sub3(upperPoint, lowerPoint)) == 0.0)
        return std::string("its two grids are at the same point");
    // The axes of the functions of the edge: the section axes of its first line, the tangent from the lower grid.
    auto frame = sectionAxes(line, section, unit3(sub3(upperPoint, lowerPoint)), std::nullopt);
    if (const std::string* problem = std::get_if<std::string>(&frame))
        return *problem;
    return AloneEdge::make(edgeGrids, lowerPoint, upperPoint, std::get<SectionAxes>(frame), degree, firstEdgeUnknown);
}

std::variant<LineData, std::string> aloneLineData(const AloneEdge& edge, std::size_t edgeNumber,
                                                  const LineElement& line, const LineSection& section,
                                                  const std::array<std::size_t, 2>& rotationDofs,
                                                  std::size_t pointCount, bool withMass)
{
    const bool forward = edge.grids[0] == line.grids[0];
    std::vector<WeightedLineKinematics> points;
    points.reserve(pointCount);
    for (const LineRulePoint& rulePoint : unitIntervalRule(pointCount))
    {
        LineKinematics kin = edge.kinematics(rulePoint.point, forward);
        // ds = w dλ / (dλ/ds)
        const double ds = rulePoint.weight / kin.fractionRate;
        points.push_back({std::move(kin), ds});
    }
    const std::vector<double> signs(edge.unknownCount(), 1.0);
    // Moments of the degree k of the edge, with the shear flexibility.
    return lineCore(LineSource::alone(edgeNumber, forward), points, edge.unknownCount(), edge.degree, true, line,
                    section, edge.dofs, signs, rotationDofs, withMass);
}

LineKinematics aloneLineKinematics(const AloneEdge& edge, bool forward, double fraction)
{
    return edge.kinematics(forward ? fraction : 1.0 - fraction, forward);
}

std::vector<double> lineFreeStrains(const LineData& data, const LineSection& section,
                                    const LineTemperatureField& temperature)
{
    std::vector<double> out;
    if (!temperature || section.thermalExpansion == 0.0)
        return out;
    out.reserve(data.axisPoints.size());
    for (const Vec3Of<double>& x : data.axisPoints)
        out.push_back(section.thermalExpansion * temperature(x));
    return out;
}

std::variant<LineState, std::string> lineStateOfCase(const LineData& data, const LineKinematics& kinematics,
                                                     const LineElement& line, const LineSection& section,
                                                     std::span<const double> x, const LineTemperatureField& temperature)
{
    const std::vector<double> freeStrains = lineFreeStrains(data, section, temperature);
    Vector freeMoments;
    if (!freeStrains.empty())
        freeMoments = data.thermalMoments(freeStrains);
    const double rise = temperature ? temperature(kinematics.point) : 0.0;
    return lineState(data, kinematics, line, section, x, rise, section.thermalExpansion * rise, freeMoments);
}

double lineStress(const LineSection& section, const LineState& state, double y, double z)
{
    return section.youngModulus * (state.strains[0] + z * state.strains[4] - y * state.strains[5] - state.freeStrain);
}

Vector lineInertiaLoad(const LineData& data, const Vec3Of<double>& acceleration)
{
    return data.inertia * std::span<const double>(acceleration);
}

Vector lineThermalLoad(const LineData& data, const LineSection& section, const LineTemperatureField& temperature)
{
    const std::vector<double> freeStrains = lineFreeStrains(data, section, temperature);
    if (freeStrains.empty())
        return Vector(data.dofs.size(), 0.0);
    return data.thermalLoad(freeStrains);
}

Vector aloneEdgeTraction(const AloneEdge& edge, bool fromLowerGrid, const EdgeTractionField& traction,
                         std::size_t pointCount)
{
    const std::size_t n = edge.unknownCount();
    const double len = edge.length();
    Vector load(n, 0.0);
    for (const LineRulePoint& rulePoint : unitIntervalRule(pointCount))
    {
        const double s = rulePoint.point;
        // u and θ do not depend on the direction of the line.
        const LineKinematics kin = edge.kinematics(s, true);
        const std::array<double, 6> v = traction(fromLowerGrid ? s : 1.0 - s);
        for (std::size_t col = 0; col < n; ++col)
        {
            double work = 0.0;
            for (std::size_t r = 0; r < 3; ++r)
                work += kin.displacement(r, col) * v[r] + kin.rotation(r, col) * v[3 + r];
            load[col] += rulePoint.weight * len * work;
        }
    }
    return load;
}

Vector lineAxialForces(const LineData& data, std::span<const double> x, std::span<const double> freeStrains)
{
    // The moment coefficients less their free part.
    Vector coefficients = data.moments * x;
    if (!freeStrains.empty())
    {
        const Vector thermal = data.complianceInverse * std::span<const double>(data.thermalMoments(freeStrains));
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            coefficients[i] -= thermal[i];
    }
    Vector forces(data.axial.size(), 0.0);
    for (std::size_t q = 0; q < data.axial.size(); ++q)
    {
        const LineAxialPoint& p = data.axial[q];
        const double freeStrain = freeStrains.empty() ? 0.0 : freeStrains[q];
        forces[q] = p.axialStiffness * (dot(p.strainRow, x) - freeStrain) + dot(p.momentRow, coefficients);
    }
    return forces;
}

Matrix lineGeometricStiffness(const LineData& data, std::span<const double> x, std::span<const double> freeStrains)
{
    const Vector forces = lineAxialForces(data, x, freeStrains);
    const std::size_t n = data.dofs.size();
    Matrix kg(n, n);
    for (std::size_t q = 0; q < data.axial.size(); ++q)
    {
        const LineAxialPoint& p = data.axial[q];
        kg += transposeTimes(p.transverseRate, p.transverseRate) * (forces[q] * p.weight);
    }
    return kg;
}

Vector aloneLineRigidMotion(const AloneEdge& edge, const Vec3Of<double>& translation, const Vec3Of<double>& rotation)
{
    const std::size_t nx = edge.unknownCount();
    Vector x(nx + 6, 0.0);
    const std::array<Vec3Of<double>, 2> points{edge.lowerPoint, edge.upperPoint};
    for (std::size_t g = 0; g < 2; ++g)
    {
        const Vec3Of<double> u = add3(translation, cross3(rotation, points[g]));
        for (std::size_t c = 0; c < 3; ++c)
            x[3 * g + c] = u[c];
    }
    // θ · t = a_0 P_0 / L: the twist of a rotation ω is ω · t, so a_0 = ω · (X_upper − X_lower).
    x[6 + 3 * (edge.degree - 1)] = dot3(rotation, sub3(edge.upperPoint, edge.lowerPoint));
    for (std::size_t end = 0; end < 2; ++end)
        for (std::size_t c = 0; c < 3; ++c)
            x[nx + 3 * end + c] = rotation[c];
    return x;
}

} // namespace oos::elements::mixedHybrid
