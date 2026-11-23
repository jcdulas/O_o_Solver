#include "oos/elements/mixedHybrid/line/LineSection.hpp"

namespace oos::elements::mixedHybrid
{

namespace
{

// (1 − λ) p_A + λ p_B.
double linearBetween(const std::array<double, 2>& ends, double fraction)
{
    return (1.0 - fraction) * ends[0] + fraction * ends[1];
}

} // namespace

LineSection LineSection::rod(double e, double g, double area, double j, double rho)
{
    LineSection section = bar(e, g, area, 0.0, 0.0, 0.0, j, rho);
    section.bending = false;
    return section;
}

LineSection LineSection::bar(double e, double g, double area, double i1, double i2, double i12, double j, double rho)
{
    LineSection section;
    section.youngModulus = e;
    section.shearModulus = g;
    section.density = rho;
    section.thermalExpansion = 0.0;
    section.area = {area, area};
    section.i1 = {i1, i1};
    section.i2 = {i2, i2};
    section.i12 = {i12, i12};
    section.torsionConstant = {j, j};
    section.shearFactors = std::nullopt;
    section.nonStructuralMass = 0.0;
    section.neutralAxis = {};
    section.bending = true;
    return section;
}

LineSection LineSection::beam(double e, double g, double rho, const std::array<std::array<double, 5>, 2>& ends,
                              const std::optional<std::array<double, 2>>& shearFactors,
                              const std::array<std::array<double, 2>, 2>& neutralAxis)
{
    const auto column = [&](std::size_t c) { return std::array<double, 2>{ends[0][c], ends[1][c]}; };
    LineSection section = bar(e, g, 0.0, 0.0, 0.0, 0.0, 0.0, rho);
    section.area = column(0);
    section.i1 = column(1);
    section.i2 = column(2);
    section.i12 = column(3);
    section.torsionConstant = column(4);
    section.shearFactors = shearFactors;
    section.neutralAxis = neutralAxis;
    return section;
}

LineSectionPoint LineSection::at(double fraction) const
{
    const double a = linearBetween(area, fraction);
    const double inertia1 = linearBetween(i1, fraction);
    const double inertia2 = linearBetween(i2, fraction);
    const double product = linearBetween(i12, fraction);
    const double j = linearBetween(torsionConstant, fraction);
    LineSectionPoint point;
    point.neutralAxis = {(1.0 - fraction) * neutralAxis[0][0] + fraction * neutralAxis[1][0],
                         (1.0 - fraction) * neutralAxis[0][1] + fraction * neutralAxis[1][1]};
    const double ea = youngModulus * a;
    // The axial strain at the neutral axis, Γ_x + z_n K_y − y_n K_z, carries E A.
    const std::array<double, 6> axial{1.0, 0.0, 0.0, 0.0, point.neutralAxis[1], -point.neutralAxis[0]};
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            point.stiffness[r][c] = ea * axial[r] * axial[c];
    point.stiffness[3][3] += shearModulus * j;
    if (bending)
    {
        std::array<double, 2> shear{RIGID_SHEAR_STIFFNESS_FACTOR * ea, RIGID_SHEAR_STIFFNESS_FACTOR * ea};
        if (shearFactors)
            shear = {(*shearFactors)[0] * shearModulus * a, (*shearFactors)[1] * shearModulus * a};
        point.stiffness[1][1] += shear[0];
        point.stiffness[2][2] += shear[1];
        // K_y bends in the plane (x, z) with I2, K_z in the plane (x, y) with I1.
        point.centroidalBending = {youngModulus * inertia2, youngModulus * inertia1, -youngModulus * product};
        point.stiffness[4][4] += point.centroidalBending[0];
        point.stiffness[5][5] += point.centroidalBending[1];
        point.stiffness[4][5] += point.centroidalBending[2];
        point.stiffness[5][4] += point.centroidalBending[2];
    }
    point.structuralMassPerLength = density * a;
    point.nonStructuralMassPerLength = nonStructuralMass + linearBetween(stationNonStructuralMass, fraction);
    point.massPerLength = point.structuralMassPerLength + point.nonStructuralMassPerLength;
    for (std::size_t c = 0; c < 2; ++c)
        point.massCentre[c] = (1.0 - fraction) * massCentre[0][c] + fraction * massCentre[1][c];
    point.inertia = {density * (inertia1 + inertia2), density * inertia2, density * inertia1, density * product};
    return point;
}

LineElement LineElement::between(std::size_t a, std::size_t b, std::size_t sectionIndex)
{
    LineElement line;
    line.grids = {a, b};
    line.section = sectionIndex;
    return line;
}

LineElement LineElement::oriented(const Vec3Of<double>& v) const
{
    LineElement line = *this;
    line.orientation = v;
    return line;
}

LineElement LineElement::eccentric(double y, double z) const
{
    LineElement line = *this;
    line.eccentricity = {y, z};
    return line;
}

LineElement LineElement::withOffsets(const Vec3Of<double>& atA, const Vec3Of<double>& atB) const
{
    LineElement line = *this;
    line.offsets = {atA, atB};
    return line;
}

std::array<double, 6> lineForces(const LineSectionPoint& point, const std::array<double, 6>& strains, double freeStrain,
                                 const Vec3Of<double>& freeDirection)
{
    // The free expansion acts on the strains Γ only; the curvatures have no free part.
    std::array<double, 6> elastic = strains;
    for (std::size_t c = 0; c < 3; ++c)
        elastic[c] -= freeStrain * freeDirection[c];
    std::array<double, 6> forces{};
    for (std::size_t r = 0; r < 6; ++r)
    {
        double sum = 0.0;
        for (std::size_t c = 0; c < 6; ++c)
            sum += point.stiffness[r][c] * elastic[c];
        forces[r] = sum;
    }
    return forces;
}

} // namespace oos::elements::mixedHybrid
