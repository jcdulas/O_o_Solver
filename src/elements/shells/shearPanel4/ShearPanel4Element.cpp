#include "oos/elements/shells/shearPanel4/ShearPanel4Element.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <string>
#include <utility>

namespace oos::elements
{

ShearPanel4Element::Columns::Columns(const ElementBatchView& view)
    : connectivity(view), properties(view, "ShearPanelProp"), materials(view)
{
    if (properties.table == nullptr)
        return;
    const base::ContractReader& shearPanelProp = properties.table->reader();
    matId = shearPanelProp.values<std::int32_t>("matId");
    t = shearPanelProp.values<double>("thickness", base::NullPolicy::Allow);
    nonStructMass = shearPanelProp.values<double>("nonStructMass", base::NullPolicy::Allow);
    f1 = shearPanelProp.values<double>("extensionalFactor1", base::NullPolicy::Allow);
    f2 = shearPanelProp.values<double>("extensionalFactor2", base::NullPolicy::Allow);
    isNonStructMassGiven = shearPanelProp.validity("nonStructMass");
    isF1Given = shearPanelProp.validity("extensionalFactor1");
    isF2Given = shearPanelProp.validity("extensionalFactor2");
}

Resolution<ShearPanel4Element::Data> ShearPanel4Element::resolve(const ElementBatchView& view,
                                                                       const Columns& columns, std::int64_t row)
{
    using Result = Resolution<Data>;
    Resolution<ShellCorners<4>> corners = resolveShellCorners<4>(view, columns.connectivity, row);
    if (!corners)
        return Result::failure(std::move(corners.problem));
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const auto p = static_cast<std::size_t>(property.value->row);
    const auto pRow = static_cast<std::int64_t>(p);
    Resolution<IsotropicMaterial> material =
        resolveIsotropicMaterial(view, columns.materials, row, columns.matId[p], "ShearPanelProp.matId");
    if (!material)
        return Result::failure(std::move(material.problem));

    Data data;
    data.gridRows = corners.value->rows;
    data.cdFrames = corners.value->cdFrames;
    data.shearLimit = material.value->shearLimit;
    ShearPanelSection& section = data.section;
    section.thickness = columns.t[p];
    section.shearModulus = material.value->shearModulus;
    section.youngModulus = material.value->youngModulus;
    section.density = material.value->density;
    section.nonStructuralMass = columns.isNonStructMassGiven.isValid(pRow) ? columns.nonStructMass[p] : 0.0;
    section.f1 = columns.isF1Given.isValid(pRow) ? columns.f1[p] : 0.0;
    section.f2 = columns.isF2Given.isValid(pRow) ? columns.f2[p] : 0.0;
    const std::string propId = std::to_string(columns.properties.propIds[static_cast<std::size_t>(row)]);
    if (!(section.thickness > 0.0))
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "ShearPanelProp.thickness",
            base::concatenate({"expected a positive ShearPanelProp.thickness on ShearPanelProp ", propId, ", found ",
                               std::to_string(section.thickness), "; correct the property"})));
    if (!(section.shearModulus > 0.0))
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "ShearPanelProp.matId",
            base::concatenate({"expected a IsoMat with a positive IsoMat.shearModulus for ShearPanelProp ", propId,
                               ", found IsoMat.shearModulus = ", std::to_string(section.shearModulus),
                               "; give IsoMat.youngsModulus and IsoMat.poissonRatio, or "
                               "IsoMat.shearModulus, on the IsoMat"})));
    ShearPanel4Panel::BuildResult panel = ShearPanel4Panel::build(corners.value->positions, section);
    switch (panel.problem)
    {
    case ShearPanelProblem::None:
        break;
    case ShearPanelProblem::Degenerate:
        return Result::failure(
            elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "node1",
                           "expected a quadrilateral, found diagonals ShearPanel4.node1-ShearPanel4.node3 and "
                           "ShearPanel4.node2-ShearPanel4.node4 parallel or of zero length; correct the grid "
                           "point positions or their order"));
    case ShearPanelProblem::Folded:
        return Result::failure(
            elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "node1",
                           "expected a convex panel with its grid points ordered around it, found an "
                           "interior angle of 180 degrees or more; reorder "
                           "ShearPanel4.node1-ShearPanel4.node4 or correct their positions"));
    case ShearPanelProblem::Warped:
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "node4",
            "expected a flat shear panel, found its four grid points off one plane; the kick forces of a warped "
            "panel are not supported: flatten the panel or split it"));
    }
    data.panel = panel.panel;
    return Result::success(std::move(data));
}

std::array<ElementDof, 12> ShearPanel4Element::Data::dofs() const
{
    std::array<ElementDof, 12> result;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            result[3 * i + k] = ElementDof::grid(gridRows[i], static_cast<int>(k + 1));
    return result;
}

base::MatN<12> ShearPanel4Element::Data::toDisplacementSystems(const base::MatN<12>& local) const
{
    const base::Mat3& axes = panel.axes();
    std::array<base::Mat3, 4> toLocal{}; // T_i = Eᵀ R_i: CD of grid i to the element frame
    for (std::size_t i = 0; i < 4; ++i)
        toLocal[i] = axes.transposed() * cdFrames[i];
    base::MatN<12> result;
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t b = 0; b < 4; ++b)
        {
            base::Mat3 block;
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    block(r, c) = local(3 * a + r, 3 * b + c);
            const base::Mat3 carried = toLocal[a].transposed() * block * toLocal[b];
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    result(3 * a + r, 3 * b + c) = carried(r, c);
        }
    return result;
}

std::array<double, 12> ShearPanel4Element::Data::localDisplacements(const NodalDisplacements& displacements) const
{
    const base::Mat3& axes = panel.axes();
    std::array<double, 12> q{};
    for (std::size_t i = 0; i < 4; ++i)
    {
        std::array<double, 3> u{};
        for (std::size_t k = 0; k < 3; ++k)
            u[k] = displacements.valueAt(ElementDof::grid(gridRows[i], static_cast<int>(k + 1)));
        const std::array<double, 3> local = (axes.transposed() * cdFrames[i]) * u;
        for (std::size_t k = 0; k < 3; ++k)
            q[3 * i + k] = local[k];
    }
    return q;
}

void ShearPanel4Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
