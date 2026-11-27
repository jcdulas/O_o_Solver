#include "oos/design/responses/composite/StrengthRatio.hpp"

#include "oos/elements/shells/ShellRecovery.hpp"

namespace oos::design
{

double StrengthRatio::ofDirectStresses(elements::FailureTheory theory, const elements::ShellMaterial& material,
                                       const std::array<double, 3>& stress, double failureIndex) noexcept
{
    return elements::plyStrengthRatio(theory, material, stress, failureIndex);
}

double StrengthRatio::ofBond(double bondFailureIndex) noexcept
{
    return elements::bondStrengthRatio(bondFailureIndex);
}

} // namespace oos::design
