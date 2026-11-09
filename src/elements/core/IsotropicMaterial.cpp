#include "oos/elements/core/IsotropicMaterial.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <string>

namespace oos::elements
{

namespace
{

// The materialsTable columns of IsotropicMaterial, in the order of IsotropicMaterialColumns::constants.
constexpr std::array<const char*, 9> CONSTANT_COLUMNS{
    "e", "g", "density", "st", "sc", "ss", "nu", "a", "referenceTemperature"};

} // namespace

IsotropicMaterialColumns::IsotropicMaterialColumns(const ElementBatchView& view)
{
    if (!view.snapshot().hasTable(input::TableRole::Materials))
        return;
    const input::MaterialTable& materials = view.snapshot().materials();
    types = materials.types();
    for (std::size_t k = 0; k < CONSTANT_COLUMNS.size(); ++k)
    {
        constants[k] = materials.reader().values<double>(CONSTANT_COLUMNS[k], base::NullPolicy::Allow);
        isGiven[k] = materials.reader().validity(CONSTANT_COLUMNS[k]);
    }
}

Resolution<IsotropicMaterial> resolveIsotropicMaterial(const ElementBatchView& view,
                                                       const IsotropicMaterialColumns& columns, std::int64_t row,
                                                       std::int32_t matId, const char* matIdField)
{
    using Result = Resolution<IsotropicMaterial>;
    const std::optional<std::int32_t> materialRow = view.index().findMaterialRow(matId);
    if (!materialRow)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_MATERIAL, matIdField,
            base::concatenate({"expected the ID of a material card, found ", std::to_string(matId),
                               ", which the model does not define; define the IsoMat or correct the property"})));
    const std::string_view type = columns.types[*materialRow];
    if (type != "IsoMat")
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_MATERIAL, matIdField,
                           base::concatenate({"expected a IsoMat for material ", std::to_string(matId), ", found a ",
                                              type, "; ", view.card(), " takes an isotropic material"})));
    // A null constant is a blank field that the IsoMat rules leave blank: it counts as 0 (see the struct comment).
    std::array<double, 9> values{};
    for (std::size_t k = 0; k < values.size(); ++k)
        values[k] = columns.isGiven[k].isValid(*materialRow)
                        ? columns.constants[k][static_cast<std::size_t>(*materialRow)]
                        : 0.0;
    return Result::success(
        {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]});
}

} // namespace oos::elements
