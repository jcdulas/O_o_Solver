#include "oos/elements/shells/ShellRecovery.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Arctangent.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/input/tables/ParamTable.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>

namespace oos::elements
{

namespace
{

constexpr double DEGREES_PER_RADIAN = 180.0 / std::numbers::pi;
// The first value of the curvature row of a STRCUR strain table (the membrane row has 0): -1.0 by the convention of
// the result tables (fiber_curvature is 0 on layer 1 and -1 on layer 2).
constexpr double CURVATURE_ROW_FLAG = -1.0;
// The value written where a failure index has no meaning (the bond above the last ply, the element index on the
// plies but the last): NaN.
constexpr double NO_INDEX = std::numeric_limits<double>::quiet_NaN();
// A shear below this fraction of the normal values is round-off (a few units in the last place of a double).
constexpr double PRINCIPAL_ROUND_OFF = 1.0e-13;
// The sum of squares h² + τ² of a plane radius below which std::hypot takes it (a square there may have lost bits to
// the subnormal range): 2⁻⁹⁶⁸, the larger square then normal and the smaller one below its last bit.
constexpr double RADIUS_SQUARES_MIN = 0x1p-968;
// The equivalent strain of a plane strain state: ⅔·√(e1² − e1·e2 + e2²) of the principal strains.
constexpr double STRAIN_VON_MISES_SCALE = 2.0 / 3.0;

// The principal values of a plane tensor (xx, yy, xy as a tensor component) and the angle of the major one.
struct Principal
{
    double major = 0.0;
    double minor = 0.0;
    double angleInDegrees = 0.0;
};

[[nodiscard]] Principal principalOf(double xx, double yy, double xy)
{
    // A shear at the round-off of the normal values has no sign: it is taken as +0, so that the angle of a state with
    // σx < σy is +90° (a stable output) and not ±90° by the sign of the round-off.
    if (std::abs(xy) <= PRINCIPAL_ROUND_OFF * std::max(std::abs(xx), std::abs(yy)))
        xy = 0.0;
    const double centre = 0.5 * (xx + yy);
    // PERFORMANCE: the radius √(h² + τ²) and the angle by base::arctangent2, without the library's hypot and atan2
    // (26 of the 30 ns of a row's principal values, each row of each case of a block); std::hypot only where a square
    // leaves the range of normal doubles. The same bits on every target; within 2 units in the last place of the
    // library's values. Measured: the recovery of ShellQuad4 and CompProp plates.
    const double half = 0.5 * (xx - yy);
    const double squares = half * half + xy * xy;
    const double radius = squares >= RADIUS_SQUARES_MIN && squares <= std::numeric_limits<double>::max()
                              ? std::sqrt(squares)
                              : std::hypot(half, xy);
    return {centre + radius, centre - radius, 0.5 * base::arctangent2(2.0 * xy, xx - yy) * DEGREES_PER_RADIAN};
}

[[nodiscard]] double planeVonMises(const Principal& principal)
{
    return std::sqrt(principal.major * principal.major - principal.major * principal.minor +
                     principal.minor * principal.minor);
}

// The transverse shear stresses (τxz, τyz) of the plies of a laminate in the material frame, from the element's shear
// forces: in a direction with a shear distribution (Laminate), the equilibrium stress at the top of each ply, τ_k =
// Q·shearAtTopPerForce, zero at the top surface; otherwise the mean stress Q / t in every ply (a rigid direction, or a
// homogeneous stack).
[[nodiscard]] std::vector<std::array<double, 2>> plyShears(const Laminate& laminate,
                                                           const std::array<double, 2>& shearForce)
{
    std::vector<std::array<double, 2>> shears(laminate.plies.size());
    for (std::size_t direction = 0; direction < 2; ++direction)
        for (std::size_t k = 0; k < laminate.plies.size(); ++k)
            shears[k][direction] = laminate.hasShearDistribution[direction]
                                       ? shearForce[direction] * laminate.plies[k].shearAtTopPerForce[direction]
                                       : shearForce[direction] / laminate.thickness;
    return shears;
}

// The ply values at the centre of a laminated shell: stresses and strains in the ply frame, the interlaminar shear in
// the material frame.
struct PlyState
{
    std::array<double, 3> stress{};           // σ1, σ2, τ12
    std::array<double, 3> strain{};           // ε1, ε2, γ12 (total)
    std::array<double, 3> mechanicalStrain{}; // ε − α ΔT
    std::array<double, 2> shear{};            // τ1z, τ2z: the interlaminar shears, ply frame
    std::array<double, 2> shearStrain{};      // τ / G of the ply (0 for a rigid direction)
};

[[nodiscard]] std::vector<PlyState> plyStates(const ShellPlaceState& centre, const ShellSection& section)
{
    const Laminate& laminate = *section.laminate;
    // The element's shear forces in the material frame (the frame of the plies' interlaminar output).
    const double c = std::cos(section.frameAngle);
    const double s = std::sin(section.frameAngle);
    const std::array<double, 2> shearForce{c * centre.shearForce[0] + s * centre.shearForce[1],
                                           -s * centre.shearForce[0] + c * centre.shearForce[1]};
    const std::vector<std::array<double, 2>> shears = plyShears(laminate, shearForce);
    std::vector<PlyState> states(laminate.plies.size());
    for (std::size_t k = 0; k < laminate.plies.size(); ++k)
    {
        const LaminatePly& ply = laminate.plies[k];
        const double z = 0.5 * (ply.zBottom + ply.zTop);
        std::array<double, 3> strain{};
        for (std::size_t i = 0; i < 3; ++i)
            strain[i] = centre.membraneStrain[i] - z * centre.curvature[i];
        PlyState& state = states[k];
        state.strain = strainRotation(section.frameAngle + ply.angle) * strain;
        const double deltaT = centre.temperature.deltaT + z * centre.temperature.gradient;
        for (std::size_t i = 0; i < 3; ++i)
            state.mechanicalStrain[i] = state.strain[i] - ply.material.expansion[i] * deltaT;
        state.stress = ply.material.planeStress * state.mechanicalStrain;
        state.shear = shears[k];
        for (std::size_t d = 0; d < 2; ++d)
        {
            const double g = ply.material.transverseShear[d];
            state.shearStrain[d] = g > 0.0 ? state.shear[d] / g : 0.0;
        }
    }
    return states;
}

} // namespace

CompositeOutput compositeOutputOf(const input::ModelSnapshot& snapshot)
{
    if (!snapshot.hasTable(input::TableRole::Params))
        return CompositeOutput::PliesOnly;
    const std::optional<input::ParamValue> value = snapshot.params().find("NoCompositeStress");
    if (!value || value->kind != input::ParamValue::Kind::Integer)
        return CompositeOutput::PliesOnly;
    if (value->integer == 0)
        return CompositeOutput::PliesAndHomogeneous;
    if (value->integer < 0)
        return CompositeOutput::HomogeneousOnly;
    return CompositeOutput::PliesOnly;
}

std::vector<std::string> shellOutputColumns(ElementOutput output)
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"nx", "ny", "nxy", "mx", "my", "mxy", "qx", "qy"};
    case ElementOutput::Stresses:
        return {"fiberDistance", "oxx", "oyy", "txy", "angle", "omax", "omin", "vonMises"};
    case ElementOutput::Strains:
        return {"fiberDistance", "exx", "eyy", "exy", "angle", "emax", "emin", "vonMises"};
    case ElementOutput::StrainCurvatures:
        return {"fiberCurvature", "exx", "eyy", "exy", "angle", "emax", "emin", "vonMises"};
    }
    return {};
}

std::vector<ColumnForm> shellOutputColumnForms(ElementOutput output)
{
    const ColumnForm linear = ColumnForm::linear();
    const ColumnForm other = ColumnForm::other();
    switch (output)
    {
    case ElementOutput::Forces:
        return std::vector<ColumnForm>(8, linear);
    case ElementOutput::Stresses:
        return {linear, linear, linear, linear, other, other, other, ColumnForm::planeVonMises({1, 2, 3}, 1.0, 1.0)};
    case ElementOutput::Strains:
    case ElementOutput::StrainCurvatures:
        return {linear, linear, linear, linear,
                other,  other,  other,  ColumnForm::planeVonMises({1, 2, 3}, 0.5, STRAIN_VON_MISES_SCALE)};
    }
    return {};
}

std::vector<ColumnForm> laminateOutputColumnForms(LayerOutput output)
{
    const ColumnForm linear = ColumnForm::linear();
    const ColumnForm other = ColumnForm::other();
    switch (output)
    {
    case LayerOutput::Stresses:
        return {linear, linear, linear,
                linear, linear, other,
                other,  other,  ColumnForm::planeVonMises({0, 1, 2}, 1.0, 1.0)};
    case LayerOutput::Strains:
        return {linear, linear, linear,
                linear, linear, other,
                other,  other,  ColumnForm::planeVonMises({0, 1, 2}, 0.5, STRAIN_VON_MISES_SCALE)};
    case LayerOutput::FailureIndices:
        return {other, other, other};
    case LayerOutput::PlyStates:
        return std::vector<ColumnForm>(PLY_STATE_COLUMNS, linear);
    }
    return {};
}

std::array<double, 8> shellStressRow(double fibre, const std::array<double, 3>& stress)
{
    const Principal principal = principalOf(stress[0], stress[1], stress[2]);
    return {fibre,
            stress[0],
            stress[1],
            stress[2],
            principal.angleInDegrees,
            principal.major,
            principal.minor,
            planeVonMises(principal)};
}

std::array<double, 8> shellStrainRow(double first, const std::array<double, 3>& strain)
{
    const Principal principal = principalOf(strain[0], strain[1], 0.5 * strain[2]);
    return {first,
            strain[0],
            strain[1],
            strain[2],
            principal.angleInDegrees,
            principal.major,
            principal.minor,
            STRAIN_VON_MISES_SCALE * planeVonMises(principal)};
}

void completeShellRows(ElementOutput output, std::span<ElementResultBlock> blocks)
{
    if (output == ElementOutput::Forces)
        return;
    for (ElementResultBlock& block : blocks)
        for (std::size_t i = 0; i < block.size(); ++i)
        {
            const std::span<double> values = block.mutableValues(i);
            const std::array<double, 3> components{values[1], values[2], values[3]};
            const std::array<double, 8> row = output == ElementOutput::Stresses ? shellStressRow(values[0], components)
                                                                                : shellStrainRow(values[0], components);
            std::copy(row.begin(), row.end(), values.begin());
        }
}

namespace
{

// A stress or strain row of its first column and its components, the other columns left at zero (LinearOnly).
std::array<double, 8> linearShellRow(double first, const std::array<double, 3>& components)
{
    return {first, components[0], components[1], components[2], 0.0, 0.0, 0.0, 0.0};
}

} // namespace

void appendShellRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId, const ShellPlaceState& state,
                     const ShellSection& section, ElementOutput output, CompositeOutput composite,
                     ShellRowValues rowValues)
{
    const bool isLinearOnly = rowValues == ShellRowValues::LinearOnly;
    if (output == ElementOutput::Forces)
    {
        // The force table holds the plate moments with the opposite sign of the resultants ∫σ·z dz (the convention of
        // the result tables; the fibre stresses, membrane forces and shear forces keep their sign): −M, which is D·κ
        // with the shell curvature κ (the curvature row of STRCUR below).
        const std::array<double, 8> values{state.membraneForce[0], state.membraneForce[1], state.membraneForce[2],
                                           -state.moment[0],       -state.moment[1],       -state.moment[2],
                                           state.shearForce[0],    state.shearForce[1]};
        block.append(row, elementId, {state.nodeId, 1}, values);
        return;
    }
    // PARAM NOCOMPS: a laminate writes the stresses and strains of its plies by default, those of the equivalent
    // homogeneous element only on request.
    if (section.isLaminate() && composite == CompositeOutput::PliesOnly)
        return;
    if (output == ElementOutput::StrainCurvatures)
    {
        // The shell curvature, conjugate to the moments of the force table (M = D·κ, the signs of the result tables)
        // and with ε(z) = ε_m − z·κ at the fibres.
        const std::array<double, 3>& curvature = state.curvature;
        block.append(
            row, elementId, {state.nodeId, 1},
            isLinearOnly ? linearShellRow(0.0, state.membraneStrain) : shellStrainRow(0.0, state.membraneStrain));
        block.append(row, elementId, {state.nodeId, 2},
                     isLinearOnly ? linearShellRow(CURVATURE_ROW_FLAG, curvature)
                                  : shellStrainRow(CURVATURE_ROW_FLAG, curvature));
        return;
    }
    const double t = state.thickness;
    const double inertia = section.bendingRatio * t * t * t / 12.0;
    const std::array<double, 2> fibres = section.fibres(t);
    for (std::size_t layer = 0; layer < 2; ++layer)
    {
        const double z = fibres[layer];
        std::array<double, 3> values{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            if (output == ElementOutput::Strains)
                values[k] = state.membraneStrain[k] - z * state.curvature[k];
            else
                values[k] = state.membraneForce[k] / t + (section.hasBending ? z * state.moment[k] / inertia : 0.0);
        }
        const std::array<double, 8> rowValues = isLinearOnly                       ? linearShellRow(z, values)
                                                : output == ElementOutput::Strains ? shellStrainRow(z, values)
                                                                                   : shellStressRow(z, values);
        block.append(row, elementId, {state.nodeId, static_cast<std::int32_t>(layer + 1)}, rowValues);
    }
}

ShellPlaceState shellResultantSecondRates(const ShellPlaceState& state, const ShellSection& section, bool isCoupled,
                                          const std::array<double, 2>& shearForceSecondRates, double thickness)
{
    ShellPlaceState rates;
    rates.nodeId = state.nodeId;
    const base::Mat3 coupling = isCoupled ? section.couplingStiffness(state.thickness) : base::Mat3{};
    const std::array<double, 3> forceCoupling = coupling * state.curvature;
    const std::array<double, 3> momentCoupling = coupling * state.membraneStrain;
    const double squared = thickness * thickness;
    for (std::size_t c = 0; c < 3; ++c)
    {
        rates.membraneForce[c] = 2.0 * SHELL_COUPLING_SIGN * forceCoupling[c] / squared;
        rates.moment[c] = (6.0 * state.moment[c] + 4.0 * SHELL_COUPLING_SIGN * momentCoupling[c]) / squared;
    }
    rates.shearForce = shearForceSecondRates;
    return rates;
}

void appendShellRowSecondRates(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                               const ShellPlaceState& state, const ShellPlaceState& rates,
                               const ShellPlaceState& secondRates, const ShellSection& section, ElementOutput output,
                               CompositeOutput composite)
{
    constexpr double NONE = std::numeric_limits<double>::quiet_NaN();
    if (output == ElementOutput::Forces)
    {
        const std::array<double, 8> values{secondRates.membraneForce[0], secondRates.membraneForce[1],
                                           secondRates.membraneForce[2], -secondRates.moment[0],
                                           -secondRates.moment[1],       -secondRates.moment[2],
                                           secondRates.shearForce[0],    secondRates.shearForce[1]};
        block.append(row, elementId, {state.nodeId, 1}, values);
        return;
    }
    if (section.isLaminate() && composite == CompositeOutput::PliesOnly)
        return;
    const auto rowOf = [&](const std::array<double, 3>& values)
    { return std::array<double, 8>{0.0, values[0], values[1], values[2], NONE, NONE, NONE, NONE}; };
    if (output == ElementOutput::StrainCurvatures)
    {
        block.append(row, elementId, {state.nodeId, 1}, rowOf({}));
        block.append(row, elementId, {state.nodeId, 2}, rowOf({}));
        return;
    }
    const double t = state.thickness;
    const double dt = rates.thickness;
    const double inertia = section.bendingRatio * t * t * t / 12.0;
    const double dInertia = 3.0 * inertia * dt / t;
    const double d2Inertia = 6.0 * inertia * dt * dt / (t * t);
    const std::array<double, 2> fibres = section.fibres(t);
    for (std::size_t layer = 0; layer < 2; ++layer)
    {
        const double z = fibres[layer];
        const bool isGiven = layer == 0 ? section.hasFiberDistance1 : section.hasFiberDistance2;
        const double dz = isGiven ? 0.0 : z / t * dt;
        std::array<double, 3> values{};
        for (std::size_t k = 0; k < 3 && output == ElementOutput::Stresses; ++k)
        {
            // σ = N/t + z·M/I twice differentiated, t'' = z'' = 0.
            const double n = state.membraneForce[k];
            const double dn = rates.membraneForce[k];
            const double d2n = secondRates.membraneForce[k];
            values[k] = d2n / t - 2.0 * dn * dt / (t * t) + 2.0 * n * dt * dt / (t * t * t);
            if (!section.hasBending)
                continue;
            const double m = state.moment[k];
            const double dm = rates.moment[k];
            const double d2m = secondRates.moment[k];
            values[k] += 2.0 * dz * dm / inertia - 2.0 * dz * m * dInertia / (inertia * inertia) + z * d2m / inertia -
                         2.0 * z * dm * dInertia / (inertia * inertia) - z * m * d2Inertia / (inertia * inertia) +
                         2.0 * z * m * dInertia * dInertia / (inertia * inertia * inertia);
        }
        block.append(row, elementId, {state.nodeId, static_cast<std::int32_t>(layer + 1)}, rowOf(values));
    }
}

ShellPlaceState shellResultantRates(const ShellPlaceState& state, const ShellSection& section, bool isCoupled,
                                    const std::array<double, 2>& shearForceRates, double thickness)
{
    ShellPlaceState rates;
    rates.nodeId = state.nodeId;
    rates.thickness = state.thickness / thickness;
    const base::Mat3 coupling = isCoupled ? section.couplingStiffness(state.thickness) : base::Mat3{};
    const std::array<double, 3> forceCoupling = coupling * state.curvature;
    const std::array<double, 3> momentCoupling = coupling * state.membraneStrain;
    for (std::size_t c = 0; c < 3; ++c)
    {
        rates.membraneForce[c] = (state.membraneForce[c] + SHELL_COUPLING_SIGN * forceCoupling[c]) / thickness;
        rates.moment[c] = (3.0 * state.moment[c] + SHELL_COUPLING_SIGN * momentCoupling[c]) / thickness;
    }
    rates.shearForce = shearForceRates;
    return rates;
}

void appendShellRowRates(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                         const ShellPlaceState& state, const ShellPlaceState& rates, const ShellSection& section,
                         ElementOutput output, CompositeOutput composite)
{
    constexpr double NONE = std::numeric_limits<double>::quiet_NaN();
    if (output == ElementOutput::Forces)
    {
        const std::array<double, 8> values{rates.membraneForce[0], rates.membraneForce[1], rates.membraneForce[2],
                                           -rates.moment[0],       -rates.moment[1],       -rates.moment[2],
                                           rates.shearForce[0],    rates.shearForce[1]};
        block.append(row, elementId, {state.nodeId, 1}, values);
        return;
    }
    if (section.isLaminate() && composite == CompositeOutput::PliesOnly)
        return;
    const auto withVonMises = [&](double first, const std::array<double, 3>& values,
                                  const std::array<double, 3>& valueRates, double shearFactor, double scale)
    {
        const ColumnForm form = ColumnForm::planeVonMises({1, 2, 3}, shearFactor, scale);
        return std::array<double, 8>{first, valueRates[0], valueRates[1], valueRates[2],
                                     NONE,  NONE,          NONE,          planeVonMisesRate(form, values, valueRates)};
    };
    if (output == ElementOutput::StrainCurvatures)
    {
        // The rates of the membrane strains and the curvatures (zero for T at fixed displacements; those of a section
        // term).
        block.append(row, elementId, {state.nodeId, 1},
                     withVonMises(0.0, state.membraneStrain, rates.membraneStrain, 0.5, STRAIN_VON_MISES_SCALE));
        block.append(row, elementId, {state.nodeId, 2},
                     withVonMises(0.0, state.curvature, rates.curvature, 0.5, STRAIN_VON_MISES_SCALE));
        return;
    }
    const double t = state.thickness;
    const double dt = rates.thickness;
    const double inertia = section.bendingRatio * t * t * t / 12.0;
    const std::array<double, 2> fibres = section.fibres(t);
    // A blank Z1, Z2 is ∓t/2 of the place's thickness: z/t·dt/dT; a given one is fixed.
    const std::array<double, 2> fibreRates{section.hasFiberDistance1 ? 0.0 : fibres[0] / t * dt,
                                           section.hasFiberDistance2 ? 0.0 : fibres[1] / t * dt};
    for (std::size_t layer = 0; layer < 2; ++layer)
    {
        const double z = fibres[layer];
        const double dz = fibreRates[layer];
        std::array<double, 3> values{};
        std::array<double, 3> valueRates{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            if (output == ElementOutput::Strains)
            {
                values[k] = state.membraneStrain[k] - z * state.curvature[k];
                valueRates[k] = rates.membraneStrain[k] - z * rates.curvature[k] - dz * state.curvature[k];
                continue;
            }
            // σ = N/t + z·M/I: dσ = dN/t − N·dt/t² + (dz·M + z·dM)/I − z·M·dI/I², dI = 3·I·dt/t.
            values[k] = state.membraneForce[k] / t + (section.hasBending ? z * state.moment[k] / inertia : 0.0);
            valueRates[k] = rates.membraneForce[k] / t - state.membraneForce[k] * dt / (t * t);
            if (section.hasBending)
                valueRates[k] += (dz * state.moment[k] + z * rates.moment[k]) / inertia -
                                 z * state.moment[k] * 3.0 * dt / (t * inertia);
        }
        const bool isStrain = output == ElementOutput::Strains;
        block.append(
            row, elementId, {state.nodeId, static_cast<std::int32_t>(layer + 1)},
            withVonMises(dz, values, valueRates, isStrain ? 0.5 : 1.0, isStrain ? STRAIN_VON_MISES_SCALE : 1.0));
    }
}

std::vector<std::string> laminateOutputColumns(LayerOutput output)
{
    switch (output)
    {
    case LayerOutput::Stresses:
        return {"o11", "o22", "t12", "t1z", "t2z", "angle", "major", "minor", "vonMises"};
    case LayerOutput::Strains:
        return {"e11", "e22", "e12", "e1z", "e2z", "angle", "major", "minor", "vonMises"};
    case LayerOutput::FailureIndices:
        return {"failureIndex", "bondFailureIndex", "elementFailureIndex"};
    case LayerOutput::PlyStates:
        return {"o11", "o22", "t12", "mechanicalE11", "mechanicalE22", "mechanicalE12", "t1z", "t2z"};
    }
    return {};
}

FailureIndexGradient plyFailureIndexGradient(FailureTheory theory, const PlyAllowables& allowables,
                                             const ShellMaterial& material, const std::array<double, 3>& stress,
                                             const std::array<double, 3>& strain)
{
    // plyFailureIndex differentiated, term by term.
    const double s1 = stress[0];
    const double s2 = stress[1];
    const double t12 = stress[2];
    const double xt = allowables.xt;
    const double xc = allowables.xc;
    const double yt = allowables.yt;
    const double yc = allowables.yc;
    const double s = allowables.s;
    FailureIndexGradient gradient;
    switch (theory)
    {
    case FailureTheory::None:
        return gradient;
    case FailureTheory::Hill:
    {
        const double x = s1 >= 0.0 ? xt : xc;
        const double y = s2 >= 0.0 ? yt : yc;
        const double xProduct = s1 * s2 >= 0.0 ? xt : xc;
        gradient.stress = {2.0 * s1 / (x * x) - s2 / (xProduct * xProduct),
                           -s1 / (xProduct * xProduct) + 2.0 * s2 / (y * y), 2.0 * t12 / (s * s)};
        return gradient;
    }
    case FailureTheory::Hoffman:
        gradient.stress = {(1.0 / xt - 1.0 / xc) + 2.0 * s1 / (xt * xc) - s2 / (xt * xc),
                           (1.0 / yt - 1.0 / yc) + 2.0 * s2 / (yt * yc) - s1 / (xt * xc), 2.0 * t12 / (s * s)};
        return gradient;
    case FailureTheory::TsaiWu:
        gradient.stress = {(1.0 / xt - 1.0 / xc) + 2.0 * s1 / (xt * xc) + 2.0 * allowables.f12 * s2,
                           (1.0 / yt - 1.0 / yc) + 2.0 * s2 / (yt * yc) + 2.0 * allowables.f12 * s1,
                           2.0 * t12 / (s * s)};
        return gradient;
    case FailureTheory::MaxStrain:
    {
        const bool isStrain = allowables.areStrainAllowables;
        const double e1t = isStrain ? xt : xt / material.e1;
        const double e1c = isStrain ? xc : xc / material.e1;
        const double e2t = isStrain ? yt : yt / material.e2;
        const double e2c = isStrain ? yc : yc / material.e2;
        const double gs = isStrain ? s : s / material.g12;
        const std::array<double, 3> ratios{strain[0] >= 0.0 ? strain[0] / e1t : -strain[0] / e1c,
                                           strain[1] >= 0.0 ? strain[1] / e2t : -strain[1] / e2c,
                                           std::abs(strain[2]) / gs};
        const std::array<double, 3> slopes{strain[0] >= 0.0 ? 1.0 / e1t : -1.0 / e1c,
                                           strain[1] >= 0.0 ? 1.0 / e2t : -1.0 / e2c,
                                           (strain[2] >= 0.0 ? 1.0 : -1.0) / gs};
        const std::size_t largest =
            static_cast<std::size_t>(std::max_element(ratios.begin(), ratios.end()) - ratios.begin());
        for (std::size_t k = 0; k < 3; ++k)
            if (k != largest && ratios[k] == ratios[largest])
                gradient.isDefined = false;
        gradient.strain[largest] = slopes[largest];
        return gradient;
    }
    }
    return gradient;
}

double plyFailureIndex(FailureTheory theory, const PlyAllowables& allowables, const ShellMaterial& material,
                       const std::array<double, 3>& stress, const std::array<double, 3>& strain)
{
    const double s1 = stress[0];
    const double s2 = stress[1];
    const double t12 = stress[2];
    const double xt = allowables.xt;
    const double xc = allowables.xc;
    const double yt = allowables.yt;
    const double yc = allowables.yc;
    const double s = allowables.s;
    switch (theory)
    {
    case FailureTheory::None:
        return 0.0;
    case FailureTheory::Hill:
    {
        const double x = s1 >= 0.0 ? xt : xc;
        const double y = s2 >= 0.0 ? yt : yc;
        const double xProduct = s1 * s2 >= 0.0 ? xt : xc;
        return s1 * s1 / (x * x) - s1 * s2 / (xProduct * xProduct) + s2 * s2 / (y * y) + t12 * t12 / (s * s);
    }
    case FailureTheory::Hoffman:
        return (1.0 / xt - 1.0 / xc) * s1 + (1.0 / yt - 1.0 / yc) * s2 + s1 * s1 / (xt * xc) + s2 * s2 / (yt * yc) -
               s1 * s2 / (xt * xc) + t12 * t12 / (s * s);
    case FailureTheory::TsaiWu:
        return (1.0 / xt - 1.0 / xc) * s1 + (1.0 / yt - 1.0 / yc) * s2 + s1 * s1 / (xt * xc) + s2 * s2 / (yt * yc) +
               t12 * t12 / (s * s) + 2.0 * allowables.f12 * s1 * s2;
    case FailureTheory::MaxStrain:
    {
        // The strain allowables: as given (STRN = 1.0), or the stress allowables over the moduli.
        const bool isStrain = allowables.areStrainAllowables;
        const double e1t = isStrain ? xt : xt / material.e1;
        const double e1c = isStrain ? xc : xc / material.e1;
        const double e2t = isStrain ? yt : yt / material.e2;
        const double e2c = isStrain ? yc : yc / material.e2;
        const double gs = isStrain ? s : s / material.g12;
        const double f1 = strain[0] >= 0.0 ? strain[0] / e1t : -strain[0] / e1c;
        const double f2 = strain[1] >= 0.0 ? strain[1] / e2t : -strain[1] / e2c;
        return std::max({f1, f2, std::abs(strain[2]) / gs});
    }
    }
    return 0.0;
}

namespace
{

// Whether a laminate writes failure indices (PCOMP field FT): a failure theory, and the allowables of every ply
// written.
[[nodiscard]] bool writesFailureIndices(const Laminate& laminate)
{
    if (laminate.failureTheory == FailureTheory::None)
        return false;
    for (const LaminatePly& ply : laminate.plies)
        if (ply.hasOutput && !ply.material.allowables.hasStrengths)
            return false;
    return true;
}

// The plies with an output, in their order.
[[nodiscard]] std::vector<std::size_t> writtenPlies(const Laminate& laminate)
{
    std::vector<std::size_t> written; // bounded by the plies
    for (std::size_t k = 0; k < laminate.plies.size(); ++k)
        if (laminate.plies[k].hasOutput)
            written.push_back(k);
    return written;
}

// The failure index rows of the written plies from their states (`stateOf(w)` the state of the w-th written ply): the
// index of each ply by the laminate's theory, the bond index above it (none above the last ply) and the element index,
// the largest of them, on the last row (appendLaminateRows' comment).
template <typename StateOf>
void appendFailureRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId, const Laminate& laminate,
                       const std::vector<std::size_t>& written, StateOf&& stateOf)
{
    double largest = -std::numeric_limits<double>::infinity();
    std::vector<std::array<double, 2>> indices(written.size()); // bounded by the plies
    for (std::size_t w = 0; w < written.size(); ++w)
    {
        const std::size_t k = written[w];
        const LaminatePly& ply = laminate.plies[k];
        const PlyState state = stateOf(w);
        const double index = plyFailureIndex(laminate.failureTheory, ply.material.allowables, ply.material,
                                             state.stress, state.mechanicalStrain);
        const bool isLast = k + 1 == laminate.plies.size();
        double bond = NO_INDEX;
        if (!isLast && laminate.bondAllowable > 0.0)
            bond = std::max(std::abs(state.shear[0]), std::abs(state.shear[1])) / laminate.bondAllowable;
        indices[w] = {index, bond};
        largest = std::max(largest, index);
        if (std::isfinite(bond))
            largest = std::max(largest, bond);
    }
    for (std::size_t w = 0; w < written.size(); ++w)
    {
        const bool isLastRow = w + 1 == written.size();
        const std::array<double, 3> values{indices[w][0], indices[w][1], isLastRow ? largest : NO_INDEX};
        block.append(row, elementId, {0, laminate.plies[written[w]].number}, values);
    }
}

// A ply row of stresses or strains (appendLaminateRows): the in-plane and transverse components, the angle and the
// principal values of the in-plane ones and their von Mises value.
[[nodiscard]] std::array<double, 9> laminateRow(bool isStress, const std::array<double, 3>& plane,
                                                const std::array<double, 2>& shear)
{
    const Principal principal =
        isStress ? principalOf(plane[0], plane[1], plane[2]) : principalOf(plane[0], plane[1], 0.5 * plane[2]);
    const double vonMises = isStress ? planeVonMises(principal) : STRAIN_VON_MISES_SCALE * planeVonMises(principal);
    return {plane[0],        plane[1],        plane[2], shear[0], shear[1], principal.angleInDegrees,
            principal.major, principal.minor, vonMises};
}

} // namespace

void appendLaminateRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                        const ShellPlaceState& centre, const ShellSection& section, LayerOutput output,
                        CompositeOutput composite)
{
    if (!section.isLaminate() || composite == CompositeOutput::HomogeneousOnly)
        return;
    const Laminate& laminate = *section.laminate;
    const std::vector<PlyState> states = plyStates(centre, section);
    if (output == LayerOutput::FailureIndices || output == LayerOutput::PlyStates)
    {
        if (!writesFailureIndices(laminate))
            return;
        const std::vector<std::size_t> written = writtenPlies(laminate);
        if (output == LayerOutput::FailureIndices)
        {
            appendFailureRows(block, row, elementId, laminate, written,
                              [&](std::size_t w) { return states[written[w]]; });
            return;
        }
        for (const std::size_t k : written)
        {
            const PlyState& state = states[k];
            const std::array<double, PLY_STATE_COLUMNS> values{
                state.stress[0],           state.stress[1],           state.stress[2], state.mechanicalStrain[0],
                state.mechanicalStrain[1], state.mechanicalStrain[2], state.shear[0],  state.shear[1]};
            block.append(row, elementId, {0, laminate.plies[k].number}, values);
        }
        return;
    }
    for (std::size_t k = 0; k < laminate.plies.size(); ++k)
    {
        const LaminatePly& ply = laminate.plies[k];
        if (!ply.hasOutput)
            continue;
        const PlyState& state = states[k];
        const bool isStress = output == LayerOutput::Stresses;
        block.append(
            row, elementId, {0, ply.number},
            laminateRow(isStress, isStress ? state.stress : state.strain, isStress ? state.shear : state.shearStrain));
    }
}

ShellPlaceState combinedStateRates(const ShellPlaceState& section, const ShellPlaceState& thickness,
                                   double thicknessRate)
{
    ShellPlaceState out = section;
    out.thickness = section.thickness + thicknessRate * thickness.thickness;
    for (std::size_t c = 0; c < 3; ++c)
    {
        out.membraneStrain[c] = section.membraneStrain[c] + thicknessRate * thickness.membraneStrain[c];
        out.curvature[c] = section.curvature[c] + thicknessRate * thickness.curvature[c];
        out.membraneForce[c] = section.membraneForce[c] + thicknessRate * thickness.membraneForce[c];
        out.moment[c] = section.moment[c] + thicknessRate * thickness.moment[c];
    }
    for (std::size_t c = 0; c < 2; ++c)
        out.shearForce[c] = section.shearForce[c] + thicknessRate * thickness.shearForce[c];
    return out;
}

void appendLaminateRowRates(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                            const ShellPlaceState& centre, const ShellPlaceState& centreRates,
                            const ShellSection& section, const LaminateRate& rate, LayerOutput output,
                            CompositeOutput composite)
{
    constexpr double NONE = std::numeric_limits<double>::quiet_NaN();
    if (!section.isLaminate() || composite == CompositeOutput::HomogeneousOnly)
        return;
    const Laminate& laminate = *section.laminate;
    const std::vector<PlyState> states = plyStates(centre, section);
    // plyStates differentiated: the element's shear forces in the material frame and their rates.
    const double c = std::cos(section.frameAngle);
    const double s = std::sin(section.frameAngle);
    const std::array<double, 2> shearForce{c * centre.shearForce[0] + s * centre.shearForce[1],
                                           -s * centre.shearForce[0] + c * centre.shearForce[1]};
    const std::array<double, 2> shearForceRate{c * centreRates.shearForce[0] + s * centreRates.shearForce[1],
                                               -s * centreRates.shearForce[0] + c * centreRates.shearForce[1]};
    std::vector<PlyState> rates(laminate.plies.size());
    for (std::size_t k = 0; k < laminate.plies.size(); ++k)
    {
        const LaminatePly& ply = laminate.plies[k];
        const double z = 0.5 * (ply.zBottom + ply.zTop);
        const double zRate = 0.5 * (rate.zBottom[k] + rate.zTop[k]);
        std::array<double, 3> strainRate{};
        for (std::size_t i = 0; i < 3; ++i)
            strainRate[i] = centreRates.membraneStrain[i] - z * centreRates.curvature[i] - zRate * centre.curvature[i];
        PlyState& out = rates[k];
        out.strain = strainRotation(section.frameAngle + ply.angle) * strainRate;
        out.mechanicalStrain = out.strain;
        out.stress = ply.material.planeStress * out.mechanicalStrain;
        for (std::size_t d = 0; d < 2; ++d)
        {
            out.shear[d] =
                laminate.hasShearDistribution[d]
                    ? shearForceRate[d] * ply.shearAtTopPerForce[d] + shearForce[d] * rate.shearAtTopPerForce[k][d]
                    : shearForceRate[d] / laminate.thickness -
                          shearForce[d] * rate.thickness / (laminate.thickness * laminate.thickness);
            const double g = ply.material.transverseShear[d];
            out.shearStrain[d] = g > 0.0 ? out.shear[d] / g : 0.0;
        }
    }
    if (output == LayerOutput::FailureIndices || output == LayerOutput::PlyStates)
    {
        if (!writesFailureIndices(laminate))
            return;
        const std::vector<std::size_t> written = writtenPlies(laminate);
        for (std::size_t w = 0; w < written.size(); ++w)
        {
            const std::size_t k = written[w];
            const PlyState& state = states[k];
            const PlyState& d = rates[k];
            if (output == LayerOutput::PlyStates)
            {
                const std::array<double, PLY_STATE_COLUMNS> values{
                    d.stress[0],           d.stress[1],           d.stress[2], d.mechanicalStrain[0],
                    d.mechanicalStrain[1], d.mechanicalStrain[2], d.shear[0],  d.shear[1]};
                block.append(row, elementId, {0, laminate.plies[k].number}, values);
                continue;
            }
            const LaminatePly& ply = laminate.plies[k];
            const FailureIndexGradient gradient = plyFailureIndexGradient(
                laminate.failureTheory, ply.material.allowables, ply.material, state.stress, state.mechanicalStrain);
            double index = NONE;
            if (gradient.isDefined)
            {
                index = 0.0;
                for (std::size_t i = 0; i < 3; ++i)
                    index += gradient.stress[i] * d.stress[i] + gradient.strain[i] * d.mechanicalStrain[i];
            }
            // The bond index max(|τ1z|, |τ2z|)/SB above the ply (appendFailureRows), along its larger shear; none
            // above the last ply, no derivative where the two are equal.
            const bool isLast = k + 1 == laminate.plies.size();
            double bond = NONE;
            if (!isLast && laminate.bondAllowable > 0.0)
            {
                const double first = std::abs(state.shear[0]);
                const double second = std::abs(state.shear[1]);
                if (first != second)
                {
                    const std::size_t larger = first > second ? 0 : 1;
                    const double sign = state.shear[larger] >= 0.0 ? 1.0 : -1.0;
                    bond = sign * d.shear[larger] / laminate.bondAllowable;
                }
            }
            const std::array<double, 3> values{index, bond, NONE};
            block.append(row, elementId, {0, laminate.plies[k].number}, values);
        }
        return;
    }
    const bool isStress = output == LayerOutput::Stresses;
    const ColumnForm form =
        ColumnForm::planeVonMises({0, 1, 2}, isStress ? 1.0 : 0.5, isStress ? 1.0 : STRAIN_VON_MISES_SCALE);
    for (std::size_t k = 0; k < laminate.plies.size(); ++k)
    {
        const LaminatePly& ply = laminate.plies[k];
        if (!ply.hasOutput)
            continue;
        const PlyState& state = states[k];
        const PlyState& d = rates[k];
        const std::array<double, 3>& plane = isStress ? state.stress : state.strain;
        const std::array<double, 3>& planeRate = isStress ? d.stress : d.strain;
        const std::array<double, 2>& shearRate = isStress ? d.shear : d.shearStrain;
        const std::array<double, 9> values{planeRate[0], planeRate[1], planeRate[2],
                                           shearRate[0], shearRate[1], NONE,
                                           NONE,         NONE,         planeVonMisesRate(form, plane, planeRate)};
        block.append(row, elementId, {0, ply.number}, values);
    }
}

void completeLaminateRows(LayerOutput output, std::span<ElementResultBlock> blocks)
{
    if (output != LayerOutput::Stresses && output != LayerOutput::Strains)
        throw std::invalid_argument(
            "completeLaminateRows: expected the ply stresses or strains (the failure indices are "
            "made from the ply states: appendLaminateFailureRows)");
    const bool isStress = output == LayerOutput::Stresses;
    for (ElementResultBlock& block : blocks)
        for (std::size_t i = 0; i < block.size(); ++i)
        {
            const std::span<double> values = block.mutableValues(i);
            const std::array<double, 9> row =
                laminateRow(isStress, {values[0], values[1], values[2]}, {values[3], values[4]});
            std::copy(row.begin(), row.end(), values.begin());
        }
}

void appendLaminateFailureRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                               const ShellSection& section, const ElementResultBlock& plyStateRows, std::size_t first,
                               std::size_t last)
{
    if (!section.isLaminate() || !writesFailureIndices(*section.laminate))
        throw std::invalid_argument("appendLaminateFailureRows: expected a laminate that writes failure indices");
    const Laminate& laminate = *section.laminate;
    const std::vector<std::size_t> written = writtenPlies(laminate);
    if (last - first != written.size())
        throw std::invalid_argument(base::concatenate(
            {"appendLaminateFailureRows: expected the states of the ", std::to_string(written.size()),
             " plies written, found ", std::to_string(last - first), " rows (a bug of O_o_Solver: report it)"}));
    appendFailureRows(block, row, elementId, laminate, written,
                      [&](std::size_t w)
                      {
                          const std::span<const double> values = plyStateRows.values(first + w);
                          PlyState state;
                          state.stress = {values[0], values[1], values[2]};
                          state.mechanicalStrain = {values[3], values[4], values[5]};
                          state.shear = {values[6], values[7]};
                          return state;
                      });
}

std::vector<ColumnForm> shellRecoveryOutputForms(const RecoveryOutput& output)
{
    return output.isLayer ? laminateOutputColumnForms(output.layerOutput) : shellOutputColumnForms(output.output);
}

std::vector<std::string> shellRecoveryOutputColumns(const RecoveryOutput& output)
{
    return output.isLayer ? laminateOutputColumns(output.layerOutput) : shellOutputColumns(output.output);
}

void appendShellOutputRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                           std::span<const ShellPlaceState> states, const ShellSection& section,
                           const RecoveryOutput& output, CompositeOutput composite, ShellRowValues values)
{
    if (output.isLayer)
    {
        appendLaminateRows(block, row, elementId, states.front(), section, output.layerOutput, composite);
        return;
    }
    for (const ShellPlaceState& state : states)
        appendShellRows(block, row, elementId, state, section, output.output, composite, values);
}

bool resetShellRecoveryBlocks(std::span<const RecoveryOutput> outputs, std::span<ElementResultBlock> blocks)
{
    bool hasElementOutput = false;
    for (std::size_t k = 0; k < outputs.size(); ++k)
    {
        blocks[k].reset(shellRecoveryOutputColumns(outputs[k]));
        hasElementOutput = hasElementOutput || !outputs[k].isLayer;
    }
    return hasElementOutput;
}

std::size_t laminateVectorisedRecoveryMinCases(LayerOutput output)
{
    return output == LayerOutput::PlyStates ? 0 : LAMINATE_VECTORISED_RECOVERY_MIN_CASES;
}

LayerOutput laminateVectorisedSource(LayerOutput output)
{
    return output == LayerOutput::FailureIndices ? LayerOutput::PlyStates : output;
}

} // namespace oos::elements
