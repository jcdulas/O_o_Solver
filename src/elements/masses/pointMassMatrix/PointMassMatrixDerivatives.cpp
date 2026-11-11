#include "oos/elements/masses/pointMassMatrix/PointMassMatrixElement.hpp"

#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

namespace oos::elements
{

namespace
{

bool isMatrixTerm(const DesignedField& field)
{
    return field.card == "PointMassMatrix" && field.column == "massMatrix" && field.item >= 0 && field.item < 21;
}

} // namespace

// The lower-triangle terms enter the symmetric matrix linearly, before the frame rotation.
FieldUse PointMassMatrixElement::designFieldUse(const DesignedField& field) const
{
    // PointMassMatrix: the terms of the 6 x 6 mass matrix change the mass only.
    return findFieldUse(field, {{"PointMassMatrix", "massMatrix", FieldUse::massOnly()}});
}

bool PointMassMatrixElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                const DesignedField& field, const MassOptions& /*options*/,
                                                ElementMatrixBlock& block) const
{
    if (!isMatrixTerm(field))
        return false;
    const Columns columns(view.columns());
    const Mass mass = resolve(view, columns, row).required();
    const auto r = static_cast<std::size_t>(row);
    const input::CoordinateFrame frame = resolveRectangularFrame(view, row, columns.frameId[r], "frameId").required();
    const base::Mat3 rotation = frame.axes.transposed() * view.displacementFrame(mass.gridRow);
    base::MatN<6> rate;
    std::int32_t item = 0;
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j <= i; ++j, ++item)
            if (item == field.item)
            {
                rate(i, j) = 1.0;
                rate(j, i) = 1.0;
            }
    appendMatrix(block, row, gridPointDofs(mass.gridRow), toDisplacementFrames<6>(rate, {rotation, rotation}));
    return true;
}

bool PointMassMatrixElement::massFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                      const DesignedField& field, const MassOptions& /*options*/,
                                                      ElementMatrixBlock& /*block*/) const
{
    return isMatrixTerm(field);
}

bool PointMassMatrixElement::massFieldMixedDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                      const DesignedField& first, const DesignedField& second,
                                                      const MassOptions& /*options*/,
                                                      ElementMatrixBlock& /*block*/) const
{
    return isMatrixTerm(first) && isMatrixTerm(second);
}

} // namespace oos::elements
