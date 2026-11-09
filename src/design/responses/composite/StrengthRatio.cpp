#include "oos/design/responses/composite/StrengthRatio.hpp"

#include <cmath>

namespace oos::design
{

double StrengthRatio::ofDirectStresses(elements::FailureTheory theory, const elements::ShellMaterial& material,
                                       const std::array<double, 3>& stress, double failureIndex) noexcept
{
    const elements::PlyAllowables& allowables = material.allowables;
    switch (theory)
    {
    case elements::FailureTheory::Hill:
        return failureIndex > 0.0 ? 1.0 / std::sqrt(failureIndex) : UNLOADED_STRENGTH_RATIO;
    case elements::FailureTheory::MaxStrain:
        return failureIndex > 0.0 ? 1.0 / failureIndex : UNLOADED_STRENGTH_RATIO;
    case elements::FailureTheory::Hoffman:
    case elements::FailureTheory::TsaiWu:
    {
        // The quadratic and linear parts of the criteria of ShellRecovery's plyFailureIndex.
        const double s1 = stress[0];
        const double s2 = stress[1];
        const double t12 = stress[2];
        const double xt = allowables.xt;
        const double xc = allowables.xc;
        const double yt = allowables.yt;
        const double yc = allowables.yc;
        const double s = allowables.s;
        const double linear = (1.0 / xt - 1.0 / xc) * s1 + (1.0 / yt - 1.0 / yc) * s2;
        double quadratic = s1 * s1 / (xt * xc) + s2 * s2 / (yt * yc) + t12 * t12 / (s * s);
        quadratic += theory == elements::FailureTheory::Hoffman ? -s1 * s2 / (xt * xc) : 2.0 * allowables.f12 * s1 * s2;
        if (quadratic <= 0.0)
            return linear > 0.0 ? 1.0 / linear : UNLOADED_STRENGTH_RATIO;
        return (-linear + std::sqrt(linear * linear + 4.0 * quadratic)) / (2.0 * quadratic);
    }
    case elements::FailureTheory::None:
        break;
    }
    return UNLOADED_STRENGTH_RATIO;
}

double StrengthRatio::ofBond(double bondFailureIndex) noexcept
{
    return bondFailureIndex > 0.0 ? 1.0 / bondFailureIndex : UNLOADED_STRENGTH_RATIO;
}

} // namespace oos::design
