#include "oos/aero/trim/ReferenceCoefficients.hpp"

#include <vector>

namespace oos::aero
{

namespace
{

// One copy of the modelled part: itself, or its image in the xz plane (y → −y), the xy plane (z → −z) or both, times
// `factor` (the product of the symmetry keys of its planes).
struct Image
{
    bool mirrorsY = false;
    bool mirrorsZ = false;
    double factor = 1.0;

    [[nodiscard]] base::Vec3 of(base::Vec3 v) const
    {
        if (mirrorsY)
            v[1] = -v[1];
        if (mirrorsZ)
            v[2] = -v[2];
        return v;
    }
    // The reflection of an axial vector (a moment): −mirror for one plane, the rotation by π (mirror) for two.
    [[nodiscard]] base::Vec3 ofMoment(base::Vec3 v) const
    {
        const base::Vec3 mirrored = of(v);
        return mirrorsY != mirrorsZ ? -1.0 * mirrored : mirrored;
    }
};

std::vector<Image> imagesOf(const AeroReference& reference)
{
    std::vector<Image> images{Image{}};
    if (reference.symmetryXz != 0)
        images.push_back({true, false, static_cast<double>(reference.symmetryXz)});
    if (reference.symmetryXy != 0)
    {
        const std::size_t count = images.size();
        for (std::size_t k = 0; k < count; ++k)
            images.push_back({images[k].mirrorsY, true, images[k].factor * static_cast<double>(reference.symmetryXy)});
    }
    return images;
}

} // namespace

std::array<double, 6> ReferenceCoefficients::of(const AeroReference& reference, const AeroResultant& resultant)
{
    const base::Vec3 origin = reference.referenceOriginInAeroFrame();
    base::Vec3 force;
    base::Vec3 moment;
    const std::vector<Image> images = imagesOf(reference);
    for (const Image& image : images)
    {
        // The moment of the modelled part about the mirrored origin, M(o') = M(o) + (o − o')·F.
        const base::Vec3 aboutImageOrigin = resultant.moment + base::cross(origin - image.of(origin), resultant.force);
        force = force + image.factor * image.of(resultant.force);
        moment = moment + image.factor * image.ofMoment(aboutImageOrigin);
    }
    const double share = 1.0 / static_cast<double>(images.size());
    const base::Vec3 forceInReference = reference.directionInReferenceFrame(share * force);
    const base::Vec3 momentInReference = reference.directionInReferenceFrame(share * moment);
    return {forceInReference[0] / reference.area,
            forceInReference[1] / reference.area,
            forceInReference[2] / reference.area,
            momentInReference[0] / (reference.area * reference.span),
            momentInReference[1] / (reference.area * reference.chord),
            momentInReference[2] / (reference.area * reference.span)};
}

} // namespace oos::aero
