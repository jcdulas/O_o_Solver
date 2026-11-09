#include "oos/elements/shells/ShellSection.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/input/tables/ParamTable.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace oos::elements
{

namespace
{

// ShellProp MID2: −1 asks for the plane-strain formulation.
constexpr std::int32_t PLANE_STRAIN_BENDING_MAT_ID = -1;

[[nodiscard]] std::span<const std::int32_t> intColumn(const base::ContractReader& reader, const char* name)
{
    return reader.values<std::int32_t>(name, base::NullPolicy::Allow);
}

[[nodiscard]] std::span<const double> realColumn(const base::ContractReader& reader, const char* name)
{
    return reader.values<double>(name, base::NullPolicy::Allow);
}

// A material ID field read as O_o_Reader gives it: null or 0 is blank.
[[nodiscard]] std::int32_t matIdAt(std::span<const std::int32_t> column, const base::ValidityView& isGiven,
                                   std::size_t p)
{
    return isGiven.isValid(static_cast<std::int64_t>(p)) ? column[p] : 0;
}

[[nodiscard]] double realOr(std::span<const double> column, const base::ValidityView& isGiven, std::size_t p,
                            double fallback)
{
    return isGiven.isValid(static_cast<std::int64_t>(p)) ? column[p] : fallback;
}

// The plane-strain matrix of an isotropic material (Zienkiewicz & Taylor, The Finite Element Method, vol. 1, eq. 6.26):
// E / ((1 + ν)(1 − 2ν)) [[1 − ν, ν, 0], [ν, 1 − ν, 0], [0, 0, (1 − 2ν)/2]].
[[nodiscard]] base::Mat3 planeStrainMatrix(double youngModulus, double poissonRatio)
{
    const double scale = youngModulus / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));
    return base::matrixFromRows({scale * (1.0 - poissonRatio), scale * poissonRatio, 0.0},
                                {scale * poissonRatio, scale * (1.0 - poissonRatio), 0.0},
                                {0.0, 0.0, scale * (1.0 - 2.0 * poissonRatio) / 2.0});
}

[[nodiscard]] bool isZero(const base::Mat3& matrix)
{
    for (const double value : matrix.values)
        if (value != 0.0)
            return false;
    return true;
}

// The equivalent ShellProp of a laminate (see ShellSection.hpp).
[[nodiscard]] ShellSection laminateSection(Laminate laminate)
{
    ShellSection section;
    section.propId = laminate.propId;
    const double t = laminate.thickness;
    section.hasMembrane = laminate.hasMembrane();
    section.hasBending = laminate.hasBending();
    section.hasCoupling = !isZero(laminate.b);
    section.membraneMatrix = (1.0 / t) * laminate.a;
    section.bendingMatrix = (12.0 / (t * t * t)) * laminate.d;
    section.couplingMatrix = (1.0 / (t * t)) * laminate.b;
    section.bendingRatio = 1.0;
    section.shearRatio = LAMINATE_SHEAR_FACTOR;
    // The rotation penalty of the drilling membranes takes the in-plane shear modulus of the membrane: A33/T for a
    // laminate (a design choice of this project).
    section.membraneShearModulus = laminate.a(2, 2) / t;
    section.transverseShearStiffness = {laminate.transverseShear[0] / t, laminate.transverseShear[1] / t};
    section.hasShearFlexibility =
        section.hasBending && (section.transverseShearStiffness[0] > 0.0 || section.transverseShearStiffness[1] > 0.0);
    section.density = laminate.massPerArea / t;
    section.nonStructuralMass = laminate.nonStructuralMass;
    section.hasThickness = true;
    section.thickness = t;
    section.hasFiberDistance1 = true;
    section.hasFiberDistance2 = true;
    section.fiberDistance1 = laminate.plies.front().zBottom;
    section.fiberDistance2 = laminate.plies.back().zTop;
    section.referenceTemperature = laminate.referenceTemperature;
    section.laminate = std::make_shared<const Laminate>(std::move(laminate));
    return section;
}

} // namespace

ShellSection ShellSection::rotatedToElementFrame(double materialAngle) const
{
    ShellSection rotated = *this;
    rotated.membraneMatrix = rotateToElementFrame(membraneMatrix, materialAngle);
    rotated.bendingMatrix = rotateToElementFrame(bendingMatrix, materialAngle);
    rotated.couplingMatrix = rotateToElementFrame(couplingMatrix, materialAngle);
    // A strain-like vector of the material frame in the element frame: v_e = T(φ)⁻¹ v_m = T(−φ) v_m.
    const base::Mat3 back = strainRotation(-materialAngle);
    rotated.membraneExpansion = back * membraneExpansion;
    rotated.bendingExpansion = back * bendingExpansion;
    rotated.frameAngle = frameAngle + materialAngle;
    return rotated;
}

std::array<double, 3> ShellSection::transverseShearCompliance() const
{
    std::array<double, 2> principal{};
    for (std::size_t k = 0; k < 2; ++k)
        principal[k] = transverseShearStiffness[k] > 0.0 ? 1.0 / transverseShearStiffness[k] : 0.0;
    const double c = std::cos(frameAngle);
    const double s = std::sin(frameAngle);
    // C = R diag(C₁, C₂) Rᵀ, R = [c −s; s c]: the material axis 1 is (c, s) in the section's frame.
    return {c * c * principal[0] + s * s * principal[1], s * s * principal[0] + c * c * principal[1],
            c * s * (principal[0] - principal[1])};
}

std::array<double, 2> ShellSection::transverseShearStiffnessAlongAxes() const
{
    const std::array<double, 3> compliance = transverseShearCompliance();
    return {compliance[0] > 0.0 ? 1.0 / compliance[0] : 0.0, compliance[1] > 0.0 ? 1.0 / compliance[1] : 0.0};
}

ThermalResultants ShellSection::thermalResultants(double deltaT, double gradient, double t) const
{
    ThermalResultants resultants;
    if (laminate)
    {
        // A stress-like vector of the material frame in the section's frame: σ_e = T(φ)ᵀ σ_m.
        const base::Mat3 toFrame = strainRotation(frameAngle).transposed();
        std::array<double, 3> force{};
        std::array<double, 3> moment{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            force[k] = laminate->forcePerDegree[k] * deltaT + laminate->forcePerGradient[k] * gradient;
            moment[k] = laminate->momentPerDegree[k] * deltaT + laminate->momentPerGradient[k] * gradient;
        }
        resultants.force = toFrame * force;
        resultants.moment = toFrame * moment;
        return resultants;
    }
    if (hasMembrane)
    {
        const std::array<double, 3> stress = membraneMatrix * membraneExpansion;
        for (std::size_t k = 0; k < 3; ++k)
            resultants.force[k] = t * stress[k] * deltaT;
    }
    if (hasBending)
    {
        const std::array<double, 3> stress = bendingStiffness(t) * bendingExpansion;
        for (std::size_t k = 0; k < 3; ++k)
            resultants.moment[k] = stress[k] * gradient;
    }
    return resultants;
}

ShellSectionColumns::ShellSectionColumns(const ElementBatchView& view)
    : shellProps(view, "ShellProp"),
      compProps(view, "CompProp"),
      compPropGlobals(view, "CompPropGlobal"),
      materials(view),
      laminates(view)
{
    if (shellProps.table == nullptr)
        return;
    const base::ContractReader& shellProp = shellProps.table->reader();
    membraneMatId = intColumn(shellProp, "membraneMatId");
    bendingMatId = intColumn(shellProp, "bendingMatId");
    shearMatId = intColumn(shellProp, "shearMatId");
    couplingMatId = intColumn(shellProp, "couplingMatId");
    isMatIdGiven = {shellProp.validity("membraneMatId"), shellProp.validity("bendingMatId"),
                    shellProp.validity("shearMatId"), shellProp.validity("couplingMatId")};
    t = realColumn(shellProp, "thickness");
    bendingRatio = realColumn(shellProp, "bendingStiffnessRatio");
    shearRatio = realColumn(shellProp, "shearThicknessRatio");
    nonStructMass = realColumn(shellProp, "nonStructMass");
    fiberDistance1 = realColumn(shellProp, "fiberDistance1");
    fiberDistance2 = realColumn(shellProp, "fiberDistance2");
    isTGiven = shellProp.validity("thickness");
    isBendingRatioGiven = shellProp.validity("bendingStiffnessRatio");
    isShearRatioGiven = shellProp.validity("shearThicknessRatio");
    isNonStructMassGiven = shellProp.validity("nonStructMass");
    isFiberDistance1Given = shellProp.validity("fiberDistance1");
    isFiberDistance2Given = shellProp.validity("fiberDistance2");
}

Resolution<ShellSection> resolveShellSection(const ElementBatchView& view, const ShellSectionColumns& columns,
                                             std::int64_t row)
{
    using Result = Resolution<ShellSection>;
    // The property type O_o_Reader gives for the PID decides the table (propType); a PID it does not find is read as
    // a missing ShellProp.
    const std::string_view type = columns.shellProps.valid.isValid(row) ? columns.shellProps.types[row] : "ShellProp";
    if (type == "CompProp" || type == "CompPropGlobal")
    {
        const PropertyRows& rows = type == "CompProp" ? columns.compProps : columns.compPropGlobals;
        Resolution<PropertyRef> property = resolveProperty(view, rows, row);
        if (!property)
            return Result::failure(std::move(property.problem));
        // PERFORMANCE: the elements of one laminate follow one another in a batch; each resolved its stack again (its
        // plies' materials, Q̄, the lamination sums and the equilibrium shear: 16 plies, some 10 µs), three to five
        // times an element per design point (stiffness, mass, recovery). The section depends on the property only
        // (the element row names the diagnostics of a failure, never kept), so the last one is taken again: the same
        // section, so the same bits.
        // Measured: the evaluation of a CompProp plate.
        const std::optional<ShellSectionColumns::LastLaminate>& last = columns.lastLaminate;
        if (last && last->table == property.value->table && last->row == property.value->row)
            return Result::success(last->section);
        Resolution<Laminate> laminate =
            resolveLaminate(view, columns.laminates, row, *property.value->table, property.value->row);
        if (!laminate)
            return Result::failure(std::move(laminate.problem));
        ShellSection section = laminateSection(std::move(*laminate.value));
        columns.lastLaminate = ShellSectionColumns::LastLaminate{property.value->table, property.value->row, section};
        return Result::success(std::move(section));
    }
    Resolution<PropertyRef> property = resolveProperty(view, columns.shellProps, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const auto p = static_cast<std::size_t>(property.value->row);

    ShellSection section;
    section.propId = columns.shellProps.propIds[static_cast<std::size_t>(row)];
    const std::int32_t membraneMatId = matIdAt(columns.membraneMatId, columns.isMatIdGiven[0], p);
    const std::int32_t bendingMatId = matIdAt(columns.bendingMatId, columns.isMatIdGiven[1], p);
    const std::int32_t shearMatId = matIdAt(columns.shearMatId, columns.isMatIdGiven[2], p);
    const std::int32_t couplingMatId = matIdAt(columns.couplingMatId, columns.isMatIdGiven[3], p);
    section.materialIds = {membraneMatId, bendingMatId, shearMatId, couplingMatId};
    const std::string propId = std::to_string(section.propId);
    const bool isPlaneStrain = bendingMatId == PLANE_STRAIN_BENDING_MAT_ID;
    if (bendingMatId < 0 && !isPlaneStrain)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "ShellProp.bendingMatId",
            base::concatenate({"expected a material ID, -1 or blank for ShellProp.bendingMatId of "
                               "ShellProp ",
                               propId, ", found ", std::to_string(bendingMatId), "; correct the property"})));
    if (membraneMatId <= 0 && bendingMatId <= 0)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "ShellProp.membraneMatId",
            base::concatenate({"expected ShellProp.membraneMatId or ShellProp.bendingMatId on ShellProp ", propId,
                               ", found both blank; a shell needs a membrane or a bending material"})));
    if (couplingMatId > 0 && (membraneMatId <= 0 || bendingMatId <= 0))
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, "ShellProp.couplingMatId",
                           base::concatenate({"expected ShellProp.membraneMatId and "
                                              "ShellProp.bendingMatId with ShellProp.couplingMatId ",
                                              std::to_string(couplingMatId), " on ShellProp ", propId,
                                              ", found one of them blank; the coupling needs both"})));

    if (membraneMatId > 0)
    {
        Resolution<ShellMaterial> material =
            resolveShellMaterial(view, columns.materials, row, membraneMatId, "ShellProp.membraneMatId");
        if (!material)
            return Result::failure(std::move(material.problem));
        section.hasMembrane = true;
        section.membraneMatrix = material.value->planeStress;
        section.membraneShearModulus = material.value->inPlaneShearModulus;
        section.density = material.value->density;
        section.membraneExpansion = material.value->expansion;
        section.referenceTemperature = material.value->referenceTemperature;
        if (isPlaneStrain)
        {
            if (material.value->type != ShellMaterialType::IsoMat)
                return Result::failure(elementProblem(
                    view, row, problem_code::UNSUPPORTED_MATERIAL, "ShellProp.membraneMatId",
                    base::concatenate({"expected a IsoMat for ShellProp.membraneMatId of the plane-strain ShellProp ",
                                       propId, " (ShellProp.bendingMatId = -1), found another material type"})));
            // The plane-strain matrix of the IsoMat's E and ν; the in-plane thermal strain of a plane-strain state is
            // (1 + ν) α (Zienkiewicz & Taylor vol. 1, eq. 6.27).
            const Resolution<IsotropicMaterial> isotropic = resolveIsotropicMaterial(
                view, columns.materials.isotropic, row, membraneMatId, "ShellProp.membraneMatId");
            const double nu = isotropic.value ? isotropic.value->poissonRatio : 0.0;
            const double e = isotropic.value ? isotropic.value->youngModulus : 0.0;
            section.membraneMatrix = planeStrainMatrix(e, nu);
            for (std::size_t k = 0; k < 2; ++k)
                section.membraneExpansion[k] *= 1.0 + nu;
        }
    }
    if (bendingMatId > 0)
    {
        Resolution<ShellMaterial> material =
            resolveShellMaterial(view, columns.materials, row, bendingMatId, "ShellProp.bendingMatId");
        if (!material)
            return Result::failure(std::move(material.problem));
        section.hasBending = true;
        section.bendingMatrix = material.value->planeStress;
        section.bendingExpansion = material.value->expansion;
        if (!section.hasMembrane)
        {
            section.density = material.value->density;
            section.referenceTemperature = material.value->referenceTemperature;
        }
    }
    section.bendingRatio =
        realOr(columns.bendingRatio, columns.isBendingRatioGiven, p, SHELL_PROP_DEFAULT_BENDING_RATIO);
    section.shearRatio = realOr(columns.shearRatio, columns.isShearRatioGiven, p, SHELL_PROP_DEFAULT_SHEAR_RATIO);
    // MID3 acts on the transverse shear of the bending; without MID2 there is no bending for it to act on.
    if (shearMatId > 0 && section.hasBending)
    {
        Resolution<ShellMaterial> material =
            resolveShellMaterial(view, columns.materials, row, shearMatId, "ShellProp.shearMatId");
        if (!material)
            return Result::failure(std::move(material.problem));
        // G3 (G1Z for x, G2Z for y) blank or zero, no saturation in that direction.
        for (std::size_t k = 0; k < 2; ++k)
            section.transverseShearStiffness[k] = material.value->transverseShear[k] > 0.0
                                                      ? section.shearRatio * material.value->transverseShear[k]
                                                      : 0.0;
        section.hasShearFlexibility =
            section.transverseShearStiffness[0] > 0.0 || section.transverseShearStiffness[1] > 0.0;
    }
    if (couplingMatId > 0)
    {
        Resolution<ShellMaterial> material =
            resolveShellMaterial(view, columns.materials, row, couplingMatId, "ShellProp.couplingMatId");
        if (!material)
            return Result::failure(std::move(material.problem));
        // The input format gives MID4 in the sign of the plate moments of the result tables, M = −∫σ·z dz (the force
        // table's sign, ShellRecovery.hpp), where the coupling of a plate whose stiff side lies on +n is negative: its
        // [B] = t² G4 is −B of the lamination theory. The section keeps B in the lamination theory's sign (that of a
        // laminate), hence −G4. The enriched plates write K_mb with +[B] = +t² G4, the input's sign.
        section.couplingMatrix = -1.0 * material.value->planeStress;
        section.hasCoupling = !isZero(section.couplingMatrix);
    }
    section.nonStructuralMass = realOr(columns.nonStructMass, columns.isNonStructMassGiven, p, 0.0);
    section.hasThickness = columns.isTGiven.isValid(static_cast<std::int64_t>(p));
    section.thickness = section.hasThickness ? columns.t[p] : 0.0;
    // A blank Z1 or Z2 is null, or NaN since O_o_Reader's contract 45 (before it, O_o_Reader resolved it from the
    // property's T, so that it was always given).
    section.hasFiberDistance1 =
        columns.isFiberDistance1Given.isValid(static_cast<std::int64_t>(p)) && !std::isnan(columns.fiberDistance1[p]);
    section.hasFiberDistance2 =
        columns.isFiberDistance2Given.isValid(static_cast<std::int64_t>(p)) && !std::isnan(columns.fiberDistance2[p]);
    section.fiberDistance1 = section.hasFiberDistance1 ? columns.fiberDistance1[p] : 0.0;
    section.fiberDistance2 = section.hasFiberDistance2 ? columns.fiberDistance2[p] : 0.0;
    if (section.hasBending && section.bendingRatio <= 0.0)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "ShellProp.bendingStiffnessRatio",
            base::concatenate({"expected a positive ShellProp.bendingStiffnessRatio on ShellProp ", propId, ", found ",
                               std::to_string(section.bendingRatio), "; correct the property"})));
    if (section.hasShearFlexibility && section.shearRatio <= 0.0)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "ShellProp.shearThicknessRatio",
            base::concatenate({"expected a positive ShellProp.shearThicknessRatio on ShellProp ", propId, ", found ",
                               std::to_string(section.shearRatio), "; correct the property"})));
    return Result::success(std::move(section));
}

template <std::size_t N>
Resolution<std::array<double, N>> resolveCornerThicknesses(const ElementBatchView& view, std::int64_t row,
                                                           const ShellSection& section, std::int32_t thicknessFlag,
                                                           std::span<const double> cornerThickness)
{
    using Result = Resolution<std::array<double, N>>;
    std::array<double, N> thickness{};
    if (section.isLaminate())
    {
        // The plies give the thickness of a laminate; the corner thicknesses of the element do not apply.
        thickness.fill(section.thickness);
        return Result::success(thickness);
    }
    if (thicknessFlag != 0 && thicknessFlag != 1)
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, "thicknessFlag",
                           base::concatenate({"expected ShellQuad4.thicknessFlag 0 or 1, found ",
                                              std::to_string(thicknessFlag), "; correct the card"})));
    for (std::size_t i = 0; i < N; ++i)
    {
        // O_o_Reader writes a blank corner thickness as -1 (ShellQuad4Card::NO_THICKNESS).
        const bool isBlank = i >= cornerThickness.size() || cornerThickness[i] < 0.0;
        const std::string field = std::string("cornerThickness").append(std::to_string(i + 1));
        if (isBlank && !section.hasThickness)
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD, field,
                base::concatenate(
                    {"expected a corner thickness or a ShellProp.thickness on ShellProp ",
                     std::to_string(section.propId),
                     ", found neither; give ShellProp.thickness on the property or the corner thicknesses"})));
        if (thicknessFlag == 0)
            thickness[i] = isBlank ? section.thickness : cornerThickness[i];
        else
            thickness[i] = (isBlank ? 1.0 : cornerThickness[i]) * section.thickness;
        if (!(thickness[i] > 0.0))
            return Result::failure(
                elementProblem(view, row, problem_code::INVALID_FIELD, field,
                               base::concatenate({"expected a positive thickness at corner ", std::to_string(i + 1),
                                                  ", found ", std::to_string(thickness[i]),
                                                  "; correct ShellProp.thickness on the property or the card"})));
    }
    return Result::success(thickness);
}

template Resolution<std::array<double, 3>> resolveCornerThicknesses<3>(const ElementBatchView&, std::int64_t,
                                                                       const ShellSection&, std::int32_t,
                                                                       std::span<const double>);
template Resolution<std::array<double, 4>> resolveCornerThicknesses<4>(const ElementBatchView&, std::int64_t,
                                                                       const ShellSection&, std::int32_t,
                                                                       std::span<const double>);
template Resolution<std::array<double, 6>> resolveCornerThicknesses<6>(const ElementBatchView&, std::int64_t,
                                                                       const ShellSection&, std::int32_t,
                                                                       std::span<const double>);
template Resolution<std::array<double, 8>> resolveCornerThicknesses<8>(const ElementBatchView&, std::int64_t,
                                                                       const ShellSection&, std::int32_t,
                                                                       std::span<const double>);

bool isShellThickness(const DesignedField& field) noexcept
{
    return field.card == "ShellProp" && field.column == "thickness";
}

bool cornersFollowThickness(const ShellSection& section, std::int32_t thicknessFlag,
                            std::span<const double> cornerThickness) noexcept
{
    if (section.isLaminate() || !section.hasThickness || !(section.thickness > 0.0))
        return false;
    if (thicknessFlag == 1)
        return true;
    // O_o_Reader writes a blank corner thickness as -1 (resolveCornerThicknesses).
    return std::all_of(cornerThickness.begin(), cornerThickness.end(), [](double value) { return value < 0.0; });
}

std::optional<std::string> offsetSectionProblem(const ShellSection& section, double offset)
{
    if (offset == 0.0 || (section.hasMembrane && section.hasBending))
        return std::nullopt;
    std::string message(
        "expected ShellProp.membraneMatId and ShellProp.bendingMatId on the property of a shell offset "
        "by ShellQuad4.offset, "
        "found ShellQuad4.offset = ");
    message.append(std::to_string(offset))
        .append(" with ")
        .append(section.hasMembrane ? "ShellProp.bendingMatId blank (no bending)"
                                    : "ShellProp.membraneMatId blank (no membrane)")
        .append(
            "; the offset would leave the rotations of the grid points without stiffness: give both "
            "ShellProp.membraneMatId and ShellProp.bendingMatId, or set ShellQuad4.offset "
            "blank");
    return message;
}

std::optional<std::string> offsetDefinitionProblem(const input::ModelSnapshot& snapshot, double offset)
{
    if (offset == 0.0 || snapshot.offsetDefinition() != input::OffsetDefinition::LargeRotationOffsets)
        return std::nullopt;
    std::string message("expected a shell offset along its element normal, found ShellQuad4.offset = ");
    message.append(std::to_string(offset))
        .append(
            " under ModelParams OffsetDefinition LoadRecoveryOffsets, which offsets a shell along the shell normals at "
            "its grid points, "
            "not computed by O_o_Solver; use ModelParams OffsetDefinition ElemOffsetZ (LoadRecoveryOffsets with the "
            "element z direction: the mass "
            "on the offset) or ElemOffsetsOnly (the default: the mass at the grid points)");
    return message;
}

} // namespace oos::elements
