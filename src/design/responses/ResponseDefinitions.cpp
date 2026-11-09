#include "oos/design/responses/ResponseDefinitions.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace oos::design
{

namespace
{

// The name of each type, as the responseKind of a DesignResp1 gives it; the extension types of the optimisation aids
// and of the sizing tools get names of O_o_Solver's own, which the screening (DSCREEN types) and the messages use.
constexpr std::array<std::pair<std::string_view, ResponseType>, 26> RESPONSE_TYPES{{
    {"Weight", ResponseType::Weight},
    {"Volume", ResponseType::Volume},
    {"FractionalMass", ResponseType::FractionalMass},
    {"Displacement", ResponseType::Displacement},
    {"Stress", ResponseType::Stress},
    {"Strain", ResponseType::Strain},
    {"Force", ResponseType::Force},
    {"ConstraintForce", ResponseType::SpcForce},
    {"StrainEnergy", ResponseType::StrainEnergy},
    {"CompositeStress", ResponseType::CompositeStress},
    {"CompositeStrain", ResponseType::CompositeStrain},
    {"CompositeFailure", ResponseType::CompositeFailure},
    {"CompositeStrengthRatio", ResponseType::CompositeStrengthRatio},
    {"Compliance", ResponseType::Compliance},
    {"MeanDisplacement", ResponseType::MeanDisplacement},
    {"DeflectionToThickness", ResponseType::DeflectionToThickness},
    {"Rotation", ResponseType::Rotation},
    {"SectionLoad", ResponseType::SectionLoad},
    {"CutLoad", ResponseType::CutLoad},
    {"ContourDisplacement", ResponseType::ContourDisplacement},
    {"ContourForce", ResponseType::ContourForce},
    {"Eigenvalue", ResponseType::Eigenvalue},
    {"Frequency", ResponseType::Frequency},
    {"BucklingFactor", ResponseType::BucklingFactor},
    {"TrimVariable", ResponseType::TrimVariable},
    {"StabilityDerivative", ResponseType::StabilityDerivative},
}};

// The argument kinds of DesignResp2 and DesignResp3 (the card types of their arguments), in the order the entries
// list them.
constexpr std::array<std::pair<std::string_view, ArgumentKind>, 11> ARGUMENT_KINDS{{
    {"DesignVar", ArgumentKind::DesignVariable},
    {"DesignConstants", ArgumentKind::TableConstant},
    {"DesignResp1", ArgumentKind::DesignResp1},
    {"DesignNodes", ArgumentKind::Node},
    {"DesignPropRelation1", ArgumentKind::DesignPropRelation1},
    {"DesignElemRelation1", ArgumentKind::DesignElemRelation1},
    {"DesignMatRelation1", ArgumentKind::DesignMatRelation1},
    {"DesignPropRelation2", ArgumentKind::DesignPropRelation2},
    {"DesignElemRelation2", ArgumentKind::DesignElemRelation2},
    {"DesignMatRelation2", ArgumentKind::DesignMatRelation2},
    {"DesignResp2", ArgumentKind::DesignResp2},
}};

} // namespace

std::optional<ResponseType> findResponseType(std::string_view rtype) noexcept
{
    for (const auto& [name, type] : RESPONSE_TYPES)
        if (name == rtype)
            return type;
    return std::nullopt;
}

std::string_view responseTypeName(ResponseType type) noexcept
{
    for (const auto& [name, candidate] : RESPONSE_TYPES)
        if (candidate == type)
            return name;
    return "";
}

bool isGlobalType(ResponseType type) noexcept
{
    return type == ResponseType::Weight || type == ResponseType::Volume || type == ResponseType::FractionalMass;
}

bool isExtensionType(ResponseType type) noexcept
{
    return type == ResponseType::Compliance || type == ResponseType::MeanDisplacement ||
           type == ResponseType::DeflectionToThickness || type == ResponseType::Rotation ||
           type == ResponseType::SectionLoad || type == ResponseType::CutLoad ||
           type == ResponseType::ContourDisplacement || type == ResponseType::ContourForce;
}

bool isSelfAdjointType(ResponseType type) noexcept
{
    return type == ResponseType::Compliance || type == ResponseType::MeanDisplacement;
}

bool isModalType(ResponseType type) noexcept
{
    return type == ResponseType::Eigenvalue || type == ResponseType::Frequency;
}

bool isEigenType(ResponseType type) noexcept
{
    return isModalType(type) || type == ResponseType::BucklingFactor;
}

bool isAeroelasticType(ResponseType type) noexcept
{
    return type == ResponseType::TrimVariable || type == ResponseType::StabilityDerivative;
}

SubcaseAnalysis analysisOfType(ResponseType type) noexcept
{
    if (isModalType(type))
        return SubcaseAnalysis::Modes;
    if (type == ResponseType::BucklingFactor)
        return SubcaseAnalysis::Buckling;
    if (isAeroelasticType(type))
        return SubcaseAnalysis::Aeroelastic;
    return SubcaseAnalysis::Statics;
}

bool isElementType(ResponseType type) noexcept
{
    switch (type)
    {
    case ResponseType::Stress:
    case ResponseType::Strain:
    case ResponseType::Force:
    case ResponseType::StrainEnergy:
    case ResponseType::CompositeStress:
    case ResponseType::CompositeStrain:
    case ResponseType::CompositeFailure:
    case ResponseType::CompositeStrengthRatio:
        return true;
    default:
        return false;
    }
}

std::optional<ArgumentKind> findArgumentKind(std::string_view flag) noexcept
{
    for (const auto& [name, kind] : ARGUMENT_KINDS)
        if (name == flag)
            return kind;
    return std::nullopt;
}

std::string_view argumentKindName(ArgumentKind kind) noexcept
{
    for (const auto& [name, candidate] : ARGUMENT_KINDS)
        if (candidate == kind)
            return name;
    return "";
}

} // namespace oos::design
