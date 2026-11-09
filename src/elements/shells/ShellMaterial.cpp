#include "oos/elements/shells/ShellMaterial.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/input/contract/ContractVersion.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace oos::elements
{

namespace
{

// The row of a material in its card table, by ID. O_o_Reader sorts its card tables by ID, so a binary search finds it;
// the IDs are checked (a table out of order falls back to a scan, a defence against an unsorted input).
[[nodiscard]] std::optional<std::int64_t> cardRowOf(const input::CardTable& table, std::int32_t matId)
{
    const std::span<const std::int32_t> ids = table.ids();
    const auto found = std::lower_bound(ids.begin(), ids.end(), matId);
    if (found != ids.end() && *found == matId)
        return found - ids.begin();
    const auto scanned = std::find(ids.begin(), ids.end(), matId);
    if (scanned != ids.end())
        return scanned - ids.begin();
    return std::nullopt;
}

// The value O_o_Reader kept for a blank G1Z or G2Z of a OrthoMat2D before its contract 45 (its Mat8Card::
// DEFAULT_TRANSVERSE_SHEAR): the field was not null but 1.0E8, and O_o_Reader wrote 1.0E8 back as a blank field, so in
// its data model the value meant "blank". A blank G1Z or G2Z of a OrthoMat2D is an infinite transverse shear modulus
// (zero or blank: no transverse shear flexibility). Read as a modulus, 1.0E8 is a soft transverse shear in SI units
// (Pa), which on a laminated wing model lowers the buckling roots by 1 % to 13 % against the rigid reading. Since
// contract 45 a blank is NaN and an explicit 1.0E8 a modulus.
constexpr double READER_BLANK_TRANSVERSE_SHEAR = 1.0e8;

// A transverse shear modulus of a OrthoMat2D: 0 (rigid) when blank (null or NaN), zero, or, in the tables
// of a contract before 45, O_o_Reader's blank value.
[[nodiscard]] double orthoMat2DTransverseShear(const base::ContractReader& card, const char* column, std::int64_t row,
                                               std::int64_t contractVersion)
{
    if (!card.hasColumn(column) || !card.validity(column).isValid(row))
        return 0.0;
    const double value = card.values<double>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
    if (std::isnan(value))
        return 0.0;
    const bool isFilledBlank =
        contractVersion < input::FIRST_CONTRACT_WITH_BLANK_FIELDS && value == READER_BLANK_TRANSVERSE_SHEAR;
    return isFilledBlank ? 0.0 : value;
}

// A real field of a card table, 0 when the column is absent or the value null (a blank field whose default is 0).
[[nodiscard]] double realOrZero(const base::ContractReader& reader, const char* column, std::int64_t row)
{
    if (!reader.hasColumn(column) || !reader.validity(column).isValid(row))
        return 0.0;
    return reader.values<double>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

// A real field, or nothing when it is absent or null.
[[nodiscard]] std::optional<double> realIfGiven(const base::ContractReader& reader, const char* column,
                                                std::int64_t row)
{
    if (!reader.hasColumn(column) || !reader.validity(column).isValid(row))
        return std::nullopt;
    return reader.values<double>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

// The value k of a fixed-list column (AnisoMat2D stiffness, thermalExpansion), 0 when absent or null.
[[nodiscard]] double listValue(const base::ContractReader& reader, const char* column, std::int64_t row, std::size_t k)
{
    if (!reader.hasColumn(column) || !reader.validity(column).isValid(row))
        return 0.0;
    const std::span<const double> values = reader.fixedLists<double>(column, base::NullPolicy::Allow)[row];
    return k < values.size() ? values[k] : 0.0;
}

[[nodiscard]] base::Diagnostic missingCardTable(const ElementBatchView& view, std::int64_t row, std::int32_t matId,
                                                std::string_view type, const std::string& field)
{
    return elementProblem(
        view, row, problem_code::MISSING_MATERIAL, field,
        base::concatenate({"expected the ", type, " card table for material ", std::to_string(matId),
                           ", found a snapshot taken without it; take the snapshot with "
                           "O_o_Solver.snapshot.take_snapshot, which gathers the AnisoMat2D and OrthoMat2D "
                           "tables"}));
}

[[nodiscard]] ShellMaterial isoMatMaterial(const ShellMaterialColumns& columns, std::int32_t matId, std::int32_t row,
                                           const IsotropicMaterial& isotropic)
{
    ShellMaterial material;
    material.matId = matId;
    material.type = ShellMaterialType::IsoMat;
    material.planeStress = planeStressMatrix(isotropic.youngModulus, isotropic.poissonRatio);
    // ShellProp: a IsoMat as MID3 gives the transverse shear modulus G in both directions; G = 0 leaves the shear rigid
    // (G3 blank or zero, no saturation).
    material.transverseShear = {isotropic.shearModulus, isotropic.shearModulus};
    material.inPlaneShearModulus = isotropic.shearModulus;
    material.density = isotropic.density;
    const double alpha = columns.isExpansionGiven.isValid(row) ? columns.expansion[static_cast<std::size_t>(row)] : 0.0;
    material.expansion = {alpha, alpha, 0.0};
    material.referenceTemperature = columns.isReferenceTemperatureGiven.isValid(row)
                                        ? columns.referenceTemperature[static_cast<std::size_t>(row)]
                                        : 0.0;
    material.e1 = isotropic.youngModulus;
    material.e2 = isotropic.youngModulus;
    material.g12 = isotropic.shearModulus;
    return material;
}

[[nodiscard]] ShellMaterial anisoMat2DMaterial(const base::ContractReader& card, std::int32_t matId, std::int64_t row)
{
    ShellMaterial material;
    material.matId = matId;
    material.type = ShellMaterialType::AnisoMat2D;
    // stiffness: G11, G12, G13, G22, G23, G33 (the upper triangle of the AnisoMat2D matrix, row by row).
    const double g11 = listValue(card, "stiffness", row, 0);
    const double g12 = listValue(card, "stiffness", row, 1);
    const double g13 = listValue(card, "stiffness", row, 2);
    const double g22 = listValue(card, "stiffness", row, 3);
    const double g23 = listValue(card, "stiffness", row, 4);
    const double g33 = listValue(card, "stiffness", row, 5);
    material.planeStress = base::matrixFromRows({g11, g12, g13}, {g12, g22, g23}, {g13, g23, g33});
    // AnisoMat2D as MID3: [[G11, G12], [G12, G22]] relates the transverse shears to their strains; the edge shear of
    // the plates takes one modulus per direction: G11 for x, G22 for y (a design choice of this project).
    material.transverseShear = {g11, g22};
    material.inPlaneShearModulus = g33;
    material.density = realOrZero(card, "density", row);
    material.expansion = {listValue(card, "thermalExpansion", row, 0), listValue(card, "thermalExpansion", row, 1),
                          listValue(card, "thermalExpansion", row, 2)};
    material.referenceTemperature = realOrZero(card, "referenceTemperature", row);
    material.e1 = g11;
    material.e2 = g22;
    material.g12 = g33;
    return material;
}

[[nodiscard]] Resolution<ShellMaterial> orthoMat2DMaterial(const ElementBatchView& view, std::int64_t elementRow,
                                                           const base::ContractReader& card, std::int32_t matId,
                                                           std::int64_t row, const std::string& field)
{
    using Result = Resolution<ShellMaterial>;
    ShellMaterial material;
    material.matId = matId;
    material.type = ShellMaterialType::OrthoMat2D;
    const double e1 = realOrZero(card, "youngsModulus1", row);
    const double e2 = realOrZero(card, "youngsModulus2", row);
    const double nu12 = realOrZero(card, "poissonRatio12", row);
    const double g12 = realOrZero(card, "shearModulus12", row);
    if (!(e1 > 0.0) || !(e2 > 0.0))
        return Result::failure(
            elementProblem(view, elementRow, problem_code::INVALID_FIELD, field,
                           base::concatenate({"expected OrthoMat2D.youngsModulus1 and "
                                              "OrthoMat2D.youngsModulus2 positive on OrthoMat2D ",
                                              std::to_string(matId), ", found ", std::to_string(e1), " and ",
                                              std::to_string(e2), "; correct the material"})));
    // Plane stress of an orthotropic lamina (Jones, Mechanics of Composite Materials, eq. 2.61): ν21 = ν12 E2 / E1,
    // Q11 = E1 / (1 − ν12 ν21), Q22 = E2 / (1 − ν12 ν21), Q12 = ν12 E2 / (1 − ν12 ν21), Q66 = G12.
    const double nu21 = nu12 * e2 / e1;
    const double denominator = 1.0 - nu12 * nu21;
    material.planeStress = base::matrixFromRows({e1 / denominator, nu12 * e2 / denominator, 0.0},
                                                {nu12 * e2 / denominator, e2 / denominator, 0.0}, {0.0, 0.0, g12});
    // OrthoMat2D: G1Z and G2Z zero or blank, no transverse shear flexibility.
    const std::int64_t version = view.snapshot().versions().dataContractVersion;
    material.transverseShear = {orthoMat2DTransverseShear(card, "shearModulus1Z", row, version),
                                orthoMat2DTransverseShear(card, "shearModulus2Z", row, version)};
    material.inPlaneShearModulus = g12;
    material.density = realOrZero(card, "density", row);
    material.expansion = {realOrZero(card, "thermalExpansion1", row), realOrZero(card, "thermalExpansion2", row), 0.0};
    material.referenceTemperature = realOrZero(card, "referenceTemperature", row);
    material.e1 = e1;
    material.e2 = e2;
    material.g12 = g12;

    PlyAllowables& allowables = material.allowables;
    const std::optional<double> xt = realIfGiven(card, "tensionStrength1", row);
    const std::optional<double> yt = realIfGiven(card, "tensionStrength2", row);
    const std::optional<double> s = realIfGiven(card, "shearStrength", row);
    allowables.hasStrengths = xt && yt && s && *xt > 0.0 && *yt > 0.0 && *s > 0.0;
    allowables.xt = xt.value_or(0.0);
    allowables.yt = yt.value_or(0.0);
    allowables.s = s.value_or(0.0);
    // OrthoMat2D: Xc defaults to Xt, Yc to Yt (O_o_Reader gives a blank as null or 0).
    const std::optional<double> xc = realIfGiven(card, "compressionStrength1", row);
    const std::optional<double> yc = realIfGiven(card, "compressionStrength2", row);
    allowables.xc = xc && *xc > 0.0 ? *xc : allowables.xt;
    allowables.yc = yc && *yc > 0.0 ? *yc : allowables.yt;
    allowables.f12 = realOrZero(card, "interactionTerm12", row);
    allowables.areStrainAllowables = realOrZero(card, "strainAllowables", row) == 1.0;
    return Result::success(material);
}

} // namespace

ShellMaterialColumns::ShellMaterialColumns(const ElementBatchView& view) : isotropic(view)
{
    const input::ModelSnapshot& snapshot = view.snapshot();
    if (snapshot.hasTable(input::TableRole::Materials))
    {
        const base::ContractReader& materials = snapshot.materials().reader();
        expansion = materials.values<double>("a", base::NullPolicy::Allow);
        referenceTemperature = materials.values<double>("referenceTemperature", base::NullPolicy::Allow);
        isExpansionGiven = materials.validity("a");
        isReferenceTemperatureGiven = materials.validity("referenceTemperature");
    }
    anisoMat2D = snapshot.findMaterialCardTable("AnisoMat2D");
    orthoMat2D = snapshot.findMaterialCardTable("OrthoMat2D");
}

Resolution<ShellMaterial> resolveShellMaterial(const ElementBatchView& view, const ShellMaterialColumns& columns,
                                               std::int64_t row, std::int32_t matId, const std::string& matIdField)
{
    using Result = Resolution<ShellMaterial>;
    const std::optional<std::int32_t> materialRow = view.index().findMaterialRow(matId);
    if (!materialRow)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_MATERIAL, matIdField,
            base::concatenate({"expected the ID of a material card, found ", std::to_string(matId),
                               ", which the model does not define; define the material or correct the property"})));
    const std::string_view type = columns.isotropic.types[*materialRow];
    if (type == "IsoMat")
    {
        Resolution<IsotropicMaterial> isotropic =
            resolveIsotropicMaterial(view, columns.isotropic, row, matId, matIdField.c_str());
        if (!isotropic)
            return Result::failure(std::move(isotropic.problem));
        return Result::success(isoMatMaterial(columns, matId, *materialRow, *isotropic.value));
    }
    if (type == "AnisoMat2D" || type == "OrthoMat2D")
    {
        const input::CardTable* table = type == "AnisoMat2D" ? columns.anisoMat2D : columns.orthoMat2D;
        const std::optional<std::int64_t> cardRow = table ? cardRowOf(*table, matId) : std::nullopt;
        if (!cardRow)
            return Result::failure(missingCardTable(view, row, matId, type, matIdField));
        if (type == "AnisoMat2D")
            return Result::success(anisoMat2DMaterial(table->reader(), matId, *cardRow));
        return orthoMat2DMaterial(view, row, table->reader(), matId, *cardRow, matIdField);
    }
    return Result::failure(
        elementProblem(view, row, problem_code::UNSUPPORTED_MATERIAL, matIdField,
                       base::concatenate({"expected a IsoMat, AnisoMat2D or OrthoMat2D for material ",
                                          std::to_string(matId), ", found a ", type, "; ", view.card(),
                                          " takes the shell materials (IsoMat, AnisoMat2D, OrthoMat2D)"})));
}

} // namespace oos::elements
