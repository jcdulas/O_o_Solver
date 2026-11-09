#include "oos/elements/springs/bushing1D/Bushing1DElement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/springs/SpringAxes.hpp"

#include <string>

namespace oos::elements
{

Bushing1DElement::Columns::Columns(const ElementBatchView& view)
    : ga(view.columns().values<std::int32_t>("nodeA")),
      gb(view.columns().values<std::int32_t>("nodeB")),
      frameId(view.columns().values<std::int32_t>("frameId")),
      properties(view, "BushProp1D")
{
    if (properties.table == nullptr)
        return;
    const base::ContractReader& bushProp1D = properties.table->reader();
    k = bushProp1D.values<double>("stiffness");
    m = bushProp1D.values<double>("mass");
    sa = bushProp1D.values<double>("stressFactor");
    se = bushProp1D.values<double>("strainFactor");
    groups = bushProp1D.values<std::uint8_t>("groups");
}

std::span<const ElementDof> Bushing1DElement::translationDofs(const Bush& bush, std::array<ElementDof, 6>& dofs)
{
    for (int c = 0; c < 3; ++c)
    {
        dofs[static_cast<std::size_t>(c)] = ElementDof::grid(bush.rowA, c + 1);
        dofs[static_cast<std::size_t>(3 + c)] = ElementDof::grid(bush.rowB, c + 1);
    }
    return {dofs.data(), bush.isGrounded ? std::size_t{3} : std::size_t{6}};
}

Resolution<Bushing1DElement::Bush> Bushing1DElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                               std::int64_t row)
{
    using Result = Resolution<Bush>;
    const auto r = static_cast<std::size_t>(row);
    Bush bush;
    Resolution<std::int32_t> rowA = resolveGridRow(view, row, columns.ga[r], "nodeA");
    if (!rowA)
        return Result::failure(std::move(rowA.problem));
    bush.rowA = *rowA.value;
    bush.isGrounded = columns.gb[r] == 0;
    if (!bush.isGrounded)
    {
        Resolution<std::int32_t> rowB = resolveGridRow(view, row, columns.gb[r], "nodeB");
        if (!rowB)
            return Result::failure(std::move(rowB.problem));
        bush.rowB = *rowB.value;
    }
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const auto p = static_cast<std::size_t>(property.value->row);
    if (columns.groups[p] != 0)
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, "propId",
                           base::concatenate({"BushProp1D ", std::to_string(columns.properties.propIds[r]),
                                              ": expected the linear spring BushProp1D.stiffness alone, found "
                                              "nonlinear groups (SHOCKA, SPRING, DAMPER "
                                              "or GENER); they are not supported in a linear analysis: remove them"})));
    bush.stiffness = columns.k[p];
    bush.mass = columns.m[p];
    bush.stressCoefficient = columns.sa[p];
    bush.strainCoefficient = columns.se[p];

    SpringAxesInput axesInput;
    axesInput.positionA = view.positionInBasic(bush.rowA);
    axesInput.positionB = bush.isGrounded ? axesInput.positionA : view.positionInBasic(bush.rowB);
    axesInput.frameA = view.displacementFrame(bush.rowA);
    axesInput.isGrounded = bush.isGrounded;
    axesInput.frameId = columns.frameId[r];
    Resolution<base::Mat3> axes = resolveSpringAxes(view, row, axesInput);
    if (!axes)
        return Result::failure(std::move(axes.problem));
    const base::Vec3 xAxis{(*axes.value)(0, 0), (*axes.value)(1, 0), (*axes.value)(2, 0)};

    // h = [−xᵀ·R_A, xᵀ·R_B]: x·(u_B − u_A) with u_basic = R·u_CD.
    const base::Vec3 alongA = axesInput.frameA.transposed() * xAxis;
    for (std::size_t c = 0; c < 3; ++c)
        bush.axialMap[c] = -alongA[c];
    if (!bush.isGrounded)
    {
        const base::Vec3 alongB = view.displacementFrame(bush.rowB).transposed() * xAxis;
        for (std::size_t c = 0; c < 3; ++c)
            bush.axialMap[3 + c] = alongB[c];
    }
    return Result::success(bush);
}

void Bushing1DElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
