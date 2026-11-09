#include "oos/elements/shells/LaminateRate.hpp"

#include <cmath>
#include <cstddef>

namespace oos::elements
{

namespace
{

// A value and its rate along the ply thickness: the forward derivative of setTransverseShear's operations.
struct Rated
{
    double value = 0.0;
    double rate = 0.0;
};

[[nodiscard]] Rated operator+(Rated a, Rated b) noexcept
{
    return {a.value + b.value, a.rate + b.rate};
}

[[nodiscard]] Rated operator-(Rated a, Rated b) noexcept
{
    return {a.value - b.value, a.rate - b.rate};
}

[[nodiscard]] Rated operator*(Rated a, Rated b) noexcept
{
    return {a.value * b.value, a.rate * b.value + a.value * b.rate};
}

[[nodiscard]] Rated operator*(double s, Rated a) noexcept
{
    return {s * a.value, s * a.rate};
}

[[nodiscard]] Rated operator/(Rated a, Rated b) noexcept
{
    return {a.value / b.value, (a.rate * b.value - a.value * b.rate) / (b.value * b.value)};
}

// The transverse shear modulus of a ply in direction d of the material frame (Laminate.cpp's plyShearModulus).
[[nodiscard]] double plyShearModulus(const LaminatePly& ply, std::size_t d)
{
    const double c2 = std::cos(ply.angle) * std::cos(ply.angle);
    const double s2 = 1.0 - c2;
    const std::array<double, 2>& g = ply.material.transverseShear;
    return d == 0 ? g[0] * c2 + g[1] * s2 : g[0] * s2 + g[1] * c2;
}

// The rate of H in direction d and of each ply's shear at its top per unit force: Laminate.cpp's setTransverseShear
// (τ(z) = −S(z)/D*, 1/H = ∫ τ²/G dz by three Gauss points per ply) on rated values, the same operations.
void setTransverseShearRate(const Laminate& laminate, LaminateRate& rate, std::size_t d)
{
    Rated a;
    Rated b;
    Rated dd;
    for (std::size_t k = 0; k < laminate.plies.size(); ++k)
    {
        const LaminatePly& ply = laminate.plies[k];
        const double e = ply.stiffness(d, d);
        const Rated top{ply.zTop, rate.zTop[k]};
        const Rated bottom{ply.zBottom, rate.zBottom[k]};
        a = a + e * (top - bottom);
        b = b + (e * 0.5) * (top * top - bottom * bottom);
        dd = dd + (e / 3.0) * (top * top * top - bottom * bottom * bottom);
    }
    const Rated neutralAxis = b / a;
    const Rated reducedRigidity = dd - b * b / a;
    const auto firstMoment = [&neutralAxis](double e, Rated z0, Rated z)
    { return (0.5 * e) * ((z - neutralAxis) * (z - neutralAxis) - (z0 - neutralAxis) * (z0 - neutralAxis)); };
    constexpr std::array<double, 3> GAUSS_POINTS{-0.7745966692414834, 0.0, 0.7745966692414834};
    constexpr std::array<double, 3> GAUSS_WEIGHTS{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    Rated below;
    Rated compliance;
    for (std::size_t k = 0; k < laminate.plies.size(); ++k)
    {
        const LaminatePly& ply = laminate.plies[k];
        const double e = ply.stiffness(d, d);
        const double g = plyShearModulus(ply, d);
        const Rated top{ply.zTop, rate.zTop[k]};
        const Rated bottom{ply.zBottom, rate.zBottom[k]};
        const Rated half = 0.5 * (top - bottom);
        const Rated middle = 0.5 * (top + bottom);
        for (std::size_t p = 0; p < GAUSS_POINTS.size(); ++p)
        {
            const Rated tau = (below + firstMoment(e, bottom, middle + GAUSS_POINTS[p] * half)) / reducedRigidity;
            compliance = compliance + (GAUSS_WEIGHTS[p] / g) * (half * tau * tau);
        }
        below = below + firstMoment(e, bottom, top);
        rate.shearAtTopPerForce[k][d] = -(below / reducedRigidity).rate;
    }
    // H = 1/C: dH = −dC/C².
    rate.transverseShear[d] = -compliance.rate / (compliance.value * compliance.value);
}

} // namespace

bool isPlyThickness(const DesignedField& field) noexcept
{
    return (field.card == "CompProp" || field.card == "CompPropGlobal") && field.column == "plyThickness" &&
           field.id > 0 && field.item >= 0;
}

std::optional<LaminateRate> laminateThicknessRate(const Laminate& laminate, const DesignedField& field)
{
    if (!isPlyThickness(field) || field.id != laminate.propId ||
        (field.card == "CompPropGlobal") != laminate.isCompPropGlobal)
        return std::nullopt;
    const std::size_t count = laminate.plies.size();
    LaminateRate rate;
    rate.plyThickness.assign(count, 0.0);
    rate.zBottom.assign(count, 0.0);
    rate.zTop.assign(count, 0.0);
    rate.shearAtTopPerForce.assign(count, {});
    for (std::size_t k = 0; k < count; ++k)
        if (laminate.plies[k].cardPly == field.item)
        {
            rate.plyThickness[k] = 1.0;
            rate.thickness += 1.0;
        }
    if (rate.thickness == 0.0)
        return std::nullopt; // an item out of the card's plies
    // The stack's z: Z0 (−T/2 when blank) at the bottom, each ply on the one below (DerivedProperties::addLaminate).
    double z = laminate.isZ0Blank ? -0.5 * rate.thickness : 0.0;
    for (std::size_t k = 0; k < count; ++k)
    {
        const LaminatePly& ply = laminate.plies[k];
        rate.zBottom[k] = z;
        z += rate.plyThickness[k];
        rate.zTop[k] = z;
        // Jones §4.2 differentiated (see the header).
        rate.a += rate.plyThickness[k] * ply.stiffness;
        rate.b += (ply.zTop * rate.zTop[k] - ply.zBottom * rate.zBottom[k]) * ply.stiffness;
        rate.d += (ply.zTop * ply.zTop * rate.zTop[k] - ply.zBottom * ply.zBottom * rate.zBottom[k]) * ply.stiffness;
        rate.massPerArea += ply.material.density * rate.plyThickness[k];
    }
    // A direction in which a ply is rigid stays rigid (H = 0, no rate).
    for (std::size_t d = 0; d < 2; ++d)
        if (laminate.transverseShear[d] > 0.0)
            setTransverseShearRate(laminate, rate, d);
    switch (laminate.option)
    {
    case LaminateOption::Full:
        break;
    case LaminateOption::Membrane:
        rate.b = {};
        rate.d = {};
        break;
    case LaminateOption::Bending:
        rate.a = {};
        rate.b = {};
        break;
    case LaminateOption::Smeared:
    {
        // D = A T²/12 (Laminate.cpp), the stack's A unsmeared: dD = dA T²/12 + A T dT/6.
        const double t = laminate.thickness;
        rate.d = (t * t / 12.0) * rate.a + (t * rate.thickness / 6.0) * laminate.a;
        rate.b = {};
        rate.transverseShear = {};
        for (std::array<double, 2>& shear : rate.shearAtTopPerForce)
            shear = {};
        break;
    }
    }
    return rate;
}

} // namespace oos::elements
