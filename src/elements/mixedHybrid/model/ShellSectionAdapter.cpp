#include "oos/elements/mixedHybrid/model/ShellSectionAdapter.hpp"

#include <cmath>

namespace oos::elements::mixedHybrid
{

namespace
{

Matrix33 toMatrix33(const base::Mat3& m, double factor)
{
    Matrix33 result{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            result[i][j] = factor * m(i, j);
    return result;
}

Voigt3 toVoigt3(const std::array<double, 3>& v)
{
    return {v[0], v[1], v[2]};
}

// The transverse shear stiffness H per unit thickness in the section's frame: the inverse of the compliance
// (xx, yy, xy) of elements::ShellSection; a direction rigid in transverse shear (no compliance) takes `rigid`, the
// stiffness the mixed shell gives the Kirchhoff limit. Empty when both directions are rigid.
std::optional<Matrix22> shearStiffnessPerThickness(const elements::ShellSection& section, double rigid)
{
    if (!section.hasShearFlexibility)
        return std::nullopt;
    const std::array<double, 3> compliance = section.transverseShearCompliance();
    const double determinant = compliance[0] * compliance[1] - compliance[2] * compliance[2];
    if (determinant > 0.0)
        return Matrix22{{{compliance[1] / determinant, -compliance[2] / determinant},
                         {-compliance[2] / determinant, compliance[0] / determinant}}};
    // One direction rigid: the other's stiffness along the material axes, the rigid one large.
    const std::array<double, 2> along = section.transverseShearStiffnessAlongAxes();
    return Matrix22{{{along[0] > 0.0 ? along[0] : rigid, 0.0}, {0.0, along[1] > 0.0 ? along[1] : rigid}}};
}

} // namespace

std::variant<Section, std::string> mixedSectionOf(const elements::ShellSection& section)
{
    if (section.materialIds[1] < 0)
        return std::string(
            "expected a ShellProp with a bending material in MixedHybridFormulation, found the "
            "plane-strain option (MID2 = -1); give a bending material");
    if (!section.hasMembrane && !section.hasBending)
        return std::string(
            "expected a shell section with a membrane (MID1) or a bending (MID2) material in "
            "MixedHybridFormulation, found neither; give at least one");
    const double t = section.isLaminate() ? section.laminate->thickness : section.thickness;
    if (!(t > 0.0))
        return std::string("expected a positive thickness T of the shell property in MixedHybridFormulation, found ") +
               std::to_string(t);

    Section result;
    result.parts = !section.hasBending    ? SectionParts::MembraneOnly
                   : !section.hasMembrane ? SectionParts::BendingOnly
                                          : SectionParts::MembraneAndBending;
    result.referenceThickness = t;
    if (section.hasMembrane)
        result.a = toMatrix33(section.membraneMatrix, t);
    if (section.hasMembrane && section.hasBending)
        result.b = toMatrix33(section.couplingStiffness(t), 1.0);
    if (section.hasBending)
        result.d = toMatrix33(section.bendingStiffness(t), 1.0);
    // The transverse shear of a section that bends; a membrane alone has none. The Kirchhoff limit is measured on the
    // membrane stiffness, or on 12 D / t² (that of a homogeneous plate) for a bending-only section.
    if (section.hasBending)
    {
        double reference = (result.a[0][0] + result.a[1][1] + result.a[2][2]) / 3.0;
        if (!section.hasMembrane)
            reference = 12.0 * (result.d[0][0] + result.d[1][1] + result.d[2][2]) / (3.0 * t * t);
        if (const std::optional<Matrix22> h =
                shearStiffnessPerThickness(section, result.kirchhoffRatio * reference / t))
            result.h = Matrix22{{{(*h)[0][0] * t, (*h)[0][1] * t}, {(*h)[1][0] * t, (*h)[1][1] * t}}};
    }

    // The inertia about the reference plane: from the plies of a laminate, ρ T and ρ T³ / 12 for a ShellProp.
    result.nonStructuralMass = section.nonStructuralMass;
    if (section.isLaminate())
    {
        double i0 = 0.0;
        double i1 = 0.0;
        double i2 = 0.0;
        for (const LaminatePly& ply : section.laminate->plies)
        {
            const double rho = ply.material.density;
            const double z0 = ply.zBottom;
            const double z1 = ply.zTop;
            i0 += rho * (z1 - z0);
            i1 += rho * (z1 * z1 - z0 * z0) / 2.0;
            i2 += rho * (z1 * z1 * z1 - z0 * z0 * z0) / 3.0;
        }
        result.i0 = i0 + section.nonStructuralMass;
        result.i1 = i1;
        result.i2 = i2;
    }
    else
    {
        result.i0 = section.density * t + section.nonStructuralMass;
        result.i1 = 0.0;
        result.i2 = section.density * t * t * t / 12.0;
    }
    // A membrane alone carries its mass as a surface mass: no rotary inertia of its own.
    if (result.parts == SectionParts::MembraneOnly)
    {
        result.i1 = 0.0;
        result.i2 = 0.0;
    }

    // The thermal resultants per unit mean temperature change and per unit gradient, at the reference thickness.
    const ThermalResultants perDegree = section.thermalResultants(1.0, 0.0, t);
    const ThermalResultants perGradient = section.thermalResultants(0.0, 1.0, t);
    result.thermalA = toVoigt3(perDegree.force);
    result.thermalB = toVoigt3(perDegree.moment);
    result.thermalD = toVoigt3(perGradient.moment);
    // A part the section lacks has no thermal resultant (the membrane alone: the force of a rise; the bending alone:
    // the moments).
    if (result.parts == SectionParts::MembraneOnly)
    {
        result.thermalB = {};
        result.thermalD = {};
    }
    else if (result.parts == SectionParts::BendingOnly)
        result.thermalA = {};
    return result;
}

} // namespace oos::elements::mixedHybrid
