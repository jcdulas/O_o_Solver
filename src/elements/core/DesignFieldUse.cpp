#include "oos/elements/core/DesignFieldUse.hpp"

#include <algorithm>
#include <array>
#include <span>

namespace oos::elements
{

namespace
{

// The stress and strain allowables of the material cards (MAT1 ST, SC, SS; AnisoMat2D ST, SC, SS; OrthoMat2D XT, XC,
// YT, YC, S, F12, STRN): they enter the margins of safety and the failure indices only.
constexpr std::array<std::string_view, 11> ALLOWABLES{
    "tensionAllowable",     "compressionAllowable", "shearAllowable",       "tensionStrength1",
    "compressionStrength1", "tensionStrength2",     "compressionStrength2", "shearStrength",
    "interactionTerm12",    "strainAllowables",     "bondAllowable"};
// The expansion coefficients of the material cards (MAT1 A; AnisoMat2D A1-A3; OrthoMat2D A1, A2; AnisoMat3D A1-A6).
constexpr std::array<std::string_view, 7> EXPANSIONS{
    "thermalExpansion",  "thermalExpansion1", "thermalExpansion2", "thermalExpansion3", "a",
    "thermalExpansionX", "thermalExpansionZ"};

[[nodiscard]] bool contains(std::span<const std::string_view> names, std::string_view name) noexcept
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

// A material card: its neutral name ends with "Mat", "Mat2D" or "Mat3D" (IsoMat, OrthoMat2D, AnisoMat3D).
[[nodiscard]] bool isMaterialCard(std::string_view card) noexcept
{
    return card.ends_with("Mat") || card.ends_with("Mat2D") || card.ends_with("Mat3D");
}

} // namespace

FieldUse commonFieldUse(const DesignedField& field) noexcept
{
    if (field.column == "structuralDamping")
        return FieldUse::none();
    if (isMaterialCard(field.card))
    {
        if (field.column == "density")
            return FieldUse::massOnly();
        if (field.column == "referenceTemperature" || contains(EXPANSIONS, field.column))
            return FieldUse::loadsAndRecovery();
        if (contains(ALLOWABLES, field.column))
            return FieldUse::recoveryOnly();
        return FieldUse::all().withoutDistributedLoads();
    }
    if (field.column == "nonStructMass")
        return FieldUse::massOnly();
    return FieldUse::all();
}

FieldUse findFieldUse(const DesignedField& field, std::initializer_list<FieldUseRule> rules) noexcept
{
    for (const FieldUseRule& rule : rules)
        if (rule.card == field.card && rule.column == field.column)
            return rule.use;
    return commonFieldUse(field);
}

} // namespace oos::elements
