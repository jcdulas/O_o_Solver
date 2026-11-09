#include "oos/elements/shells/ShellSectionRate.hpp"

#include "oos/elements/core/MaterialMatrices.hpp"

#include <cmath>

namespace oos::elements
{

namespace
{

// The "stiffness" items of a AnisoMat2D: G11, G12, G13, G22, G23, G33 (ShellMaterial.cpp).
constexpr std::int32_t ANISO_MAT2D_TERMS = 6;
constexpr std::int32_t G11 = 0;
constexpr std::int32_t G22 = 3;
constexpr std::int32_t G33 = 5;

// The rate of the symmetric matrix [[G11, G12, G13], [G12, G22, G23], [G13, G23, G33]] with respect to item k.
[[nodiscard]] base::Mat3 unitTerm(std::int32_t item)
{
    constexpr std::array<std::array<std::size_t, 2>, ANISO_MAT2D_TERMS> AT{
        {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};
    const auto [i, j] = AT[static_cast<std::size_t>(item)];
    base::Mat3 unit;
    unit(i, j) = 1.0;
    unit(j, i) = 1.0;
    return unit;
}

// The rate of the compliance 1/s of a principal shear stiffness s (0 for a rigid direction, as its compliance).
[[nodiscard]] double complianceRate(double stiffness, double stiffnessRate)
{
    return stiffness > 0.0 ? -stiffnessRate / (stiffness * stiffness) : 0.0;
}

} // namespace

bool isAnisoMat2DStiffnessTerm(const DesignedField& field) noexcept
{
    return field.card == "AnisoMat2D" && field.column == "stiffness" && field.id > 0 && field.item >= 0 &&
           field.item < ANISO_MAT2D_TERMS;
}

std::optional<ShellSectionRate> shellSectionRate(const ShellSection& section, const DesignedField& field,
                                                 double materialAngle, bool takesCoupling)
{
    if (!isAnisoMat2DStiffnessTerm(field) || section.isLaminate())
        return std::nullopt;
    const auto& [membraneMatId, bendingMatId, shearMatId, couplingMatId] = section.materialIds;
    const base::Mat3 unit = rotateToElementFrame(unitTerm(field.item), materialAngle);
    ShellSectionRate rate;
    bool isRead = false;
    if (membraneMatId == field.id && section.hasMembrane)
    {
        rate.membraneMatrix = unit;
        rate.membraneShearModulus = field.item == G33 ? 1.0 : 0.0;
        rate.movesMembrane = true;
        isRead = true;
    }
    if (bendingMatId == field.id && section.hasBending)
    {
        rate.bendingMatrix = unit;
        rate.movesBending = true;
        isRead = true;
    }
    // MID3 acts through G11 (x) and G22 (y) only (ShellMaterial.cpp), and only on a bending.
    if (shearMatId == field.id && section.hasBending)
    {
        isRead = true;
        for (const auto& [item, axis] : {std::pair<std::int32_t, std::size_t>{G11, 0}, {G22, 1}})
        {
            if (field.item != item)
                continue;
            if (!(section.transverseShearStiffness[axis] > 0.0))
                return std::nullopt;
            rate.transverseShearStiffness[axis] = section.shearRatio;
            rate.movesShear = true;
        }
    }
    if (couplingMatId == field.id)
    {
        // A coupling zero at the point takes its rate (see the header), but on a family that refuses a coupling.
        if (!section.hasCoupling && !takesCoupling)
            return std::nullopt;
        // B/t² = −G4 (ShellSection.cpp: the input's sign of the moments).
        rate.couplingMatrix = -1.0 * unit;
        rate.movesCoupling = true;
        isRead = true;
    }
    if (!isRead)
        return std::nullopt;
    return rate;
}

std::optional<LaminateSectionRate> laminateSectionRate(const ShellSection& section, const DesignedField& field,
                                                       double materialAngle, bool takesCoupling)
{
    if (!section.isLaminate())
        return std::nullopt;
    const Laminate& laminate = *section.laminate;
    std::optional<LaminateRate> found = laminateThicknessRate(laminate, field);
    if (!found)
        return std::nullopt;
    LaminateSectionRate out;
    out.laminate = std::move(*found);
    const LaminateRate& rate = out.laminate;
    const double t = laminate.thickness;
    const double dt = rate.thickness;
    // The section's per-unit-thickness parts in the material frame (laminateSection) and their rates at fixed T.
    const base::Mat3 membrane = (1.0 / t) * laminate.a;
    const base::Mat3 bending = (12.0 / (t * t * t)) * laminate.d;
    const base::Mat3 coupling = (1.0 / (t * t)) * laminate.b;
    const base::Mat3 membraneRate = (1.0 / t) * (rate.a - dt * membrane);
    const base::Mat3 bendingRate = (12.0 / (t * t * t)) * rate.d - (3.0 * dt / t) * bending;
    const base::Mat3 couplingRate = (1.0 / (t * t)) * rate.b - (2.0 * dt / t) * coupling;
    const auto isNonZero = [](const base::Mat3& matrix)
    {
        for (const double value : matrix.values)
            if (value != 0.0)
                return true;
        return false;
    };
    const bool movesCoupling = section.hasCoupling || isNonZero(couplingRate);
    if (movesCoupling && !section.hasCoupling && !takesCoupling)
        return std::nullopt;
    ShellSectionRate& fixed = out.atFixedThickness;
    ShellSectionRate& products = out.ofProducts;
    for (ShellSectionRate* part : {&fixed, &products})
    {
        part->movesMembrane = section.hasMembrane;
        part->movesBending = section.hasBending;
        part->movesCoupling = movesCoupling;
    }
    fixed.membraneMatrix = rotateToElementFrame(membraneRate, materialAngle);
    fixed.bendingMatrix = rotateToElementFrame(bendingRate, materialAngle);
    fixed.couplingMatrix = rotateToElementFrame(couplingRate, materialAngle);
    fixed.membraneShearModulus = (rate.a(2, 2) - dt * section.membraneShearModulus) / t;
    products.membraneMatrix = rotateToElementFrame((1.0 / t) * rate.a, materialAngle);
    products.bendingMatrix = rotateToElementFrame((12.0 / (t * t * t)) * rate.d, materialAngle);
    products.couplingMatrix = rotateToElementFrame((1.0 / (t * t)) * rate.b, materialAngle);
    products.membraneShearModulus = rate.a(2, 2) / t;
    for (std::size_t k = 0; k < 2; ++k)
    {
        // κG per unit thickness = H/T along the material axes (no rotation: the section keeps them at frameAngle).
        fixed.transverseShearStiffness[k] = (rate.transverseShear[k] - dt * section.transverseShearStiffness[k]) / t;
        products.transverseShearStiffness[k] = rate.transverseShear[k] / t;
    }
    fixed.movesShear = section.hasBending && section.hasShearFlexibility;
    products.movesShear = fixed.movesShear;
    out.thickness = dt;
    out.density = (rate.massPerArea - section.density * dt) / t;
    return out;
}

std::array<double, 3> transverseShearComplianceRate(const ShellSection& section, const ShellSectionRate& rate)
{
    // ShellSection::transverseShearCompliance differentiated: C = R diag(C₁, C₂) Rᵀ at the fixed angle frameAngle.
    std::array<double, 2> principal{};
    for (std::size_t k = 0; k < 2; ++k)
        principal[k] = complianceRate(section.transverseShearStiffness[k], rate.transverseShearStiffness[k]);
    const double c = std::cos(section.frameAngle);
    const double s = std::sin(section.frameAngle);
    return {c * c * principal[0] + s * s * principal[1], s * s * principal[0] + c * c * principal[1],
            c * s * (principal[0] - principal[1])};
}

std::array<double, 2> transverseShearStiffnessAlongAxesRate(const ShellSection& section, const ShellSectionRate& rate)
{
    const std::array<double, 3> compliance = section.transverseShearCompliance();
    const std::array<double, 3> complianceRates = transverseShearComplianceRate(section, rate);
    std::array<double, 2> rates{};
    for (std::size_t k = 0; k < 2; ++k)
        rates[k] = compliance[k] > 0.0 ? -complianceRates[k] / (compliance[k] * compliance[k]) : 0.0;
    return rates;
}

} // namespace oos::elements
