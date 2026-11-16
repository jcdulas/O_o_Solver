#include "oos/elements/mixedHybrid/solid/SolidData.hpp"

#include "oos/base/errors/InternalError.hpp"

namespace oos::elements::mixedHybrid
{

namespace
{

constexpr std::size_t DIRECT_STRAINS = 3;
constexpr std::size_t RIGID_TRANSLATIONS = 3;
constexpr std::size_t RIGID_MODES = 6;

} // namespace

SolidMaterial SolidMaterial::isotropic(double youngModulus, double poissonRatio, double density,
                                       double thermalExpansion)
{
    const double lambda = youngModulus * poissonRatio / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));
    const double mu = youngModulus / (2.0 * (1.0 + poissonRatio));
    SolidMaterial material;
    for (std::size_t i = 0; i < DIRECT_STRAINS; ++i)
    {
        for (std::size_t j = 0; j < DIRECT_STRAINS; ++j)
            material.stiffness[i][j] = lambda + (i == j ? 2.0 * mu : 0.0);
        material.stiffness[DIRECT_STRAINS + i][DIRECT_STRAINS + i] = mu;
    }
    material.density = density;
    material.thermalExpansion = thermalExpansion;
    return material;
}

SolidLoad SolidLoad::body(std::function<SolidPoint(const SolidPoint&)> forcePerVolume)
{
    SolidLoad load;
    load.kind = Kind::Body;
    load.force = std::move(forcePerVolume);
    return load;
}

SolidLoad SolidLoad::thermal(std::function<double(const SolidPoint&)> freeStrainAt)
{
    SolidLoad load;
    load.kind = Kind::Thermal;
    load.freeStrain = std::move(freeStrainAt);
    return load;
}

SolidLoad SolidLoad::uniformAcceleration(double density, const SolidPoint& acceleration)
{
    const SolidPoint force{density * acceleration[0], density * acceleration[1], density * acceleration[2]};
    return body([force](const SolidPoint&) { return force; });
}

SolidLoad SolidLoad::rigidInertia(double density, std::size_t mode)
{
    OOS_ASSERT(mode < RIGID_MODES);
    SolidPoint axis{};
    axis[mode % RIGID_TRANSLATIONS] = density;
    if (mode < RIGID_TRANSLATIONS)
        return body([axis](const SolidPoint&) { return axis; });
    // ρ (e × x): the force of a unit angular acceleration about the axis e through the origin
    return body(
        [axis](const SolidPoint& x)
        {
            return SolidPoint{axis[1] * x[2] - axis[2] * x[1], axis[2] * x[0] - axis[0] * x[2],
                              axis[0] * x[1] - axis[1] * x[0]};
        });
}

} // namespace oos::elements::mixedHybrid
