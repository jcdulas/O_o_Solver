#include "oos/elements/masses/pointMassInertia/PointMassInertiaElement.hpp"

#include "oos/elements/core/MatrixKernels.hpp"

namespace oos::elements
{

namespace
{

enum class MassField
{
    Other,
    Mass,
    Offset,
    Inertia
};

MassField fieldOf(const DesignedField& field)
{
    if (field.card != "PointMassInertia")
        return MassField::Other;
    if (field.column == "mass" && field.item == -1)
        return MassField::Mass;
    if (field.column == "offset" && field.item >= 0 && field.item < 3)
        return MassField::Offset;
    if (field.column == "inertia" && field.item >= 0 && field.item < 6)
        return MassField::Inertia;
    return MassField::Other;
}

base::Vec3 offsetDirection(const base::Mat3& inputToBasic, std::int32_t item)
{
    const auto column = static_cast<std::size_t>(item);
    return {inputToBasic(0, column), inputToBasic(1, column), inputToBasic(2, column)};
}

base::Mat3 inertiaRate(std::int32_t wanted)
{
    base::Mat3 rate;
    std::int32_t item = 0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j <= i; ++j, ++item)
            if (item == wanted)
            {
                // Products of inertia enter the tensor with the opposite sign.
                rate(i, j) = i == j ? 1.0 : -1.0;
                rate(j, i) = rate(i, j);
            }
    return rate;
}

base::MatN<6> offsetRate(double mass, base::Vec3 offset, base::Vec3 direction)
{
    const base::Mat3 cross = crossProductMatrix(offset);
    const base::Mat3 rate = crossProductMatrix(direction);
    const base::Mat3 coupling = mass * rate;
    const base::Mat3 rotational = -mass * (rate * cross + cross * rate);
    base::MatN<6> result;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            result(i, 3 + j) = -coupling(i, j);
            result(3 + i, j) = coupling(i, j);
            result(3 + i, 3 + j) = rotational(i, j);
        }
    return result;
}

} // namespace

// Differentiate the rigid-body matrix in PointMassInertiaElement.hpp before rotating its 3-component blocks.
FieldUse PointMassInertiaElement::designFieldUse(const DesignedField& field) const
{
    // PointMassInertia: M, the offset X1-X3 and the inertias I11-I33 change the mass only.
    return findFieldUse(field, {{"PointMassInertia", "mass", FieldUse::massOnly()},
                                {"PointMassInertia", "offset", FieldUse::massOnly()},
                                {"PointMassInertia", "inertia", FieldUse::massOnly()}});
}

bool PointMassInertiaElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                 const DesignedField& field, const MassOptions& /*options*/,
                                                 ElementMatrixBlock& block) const
{
    const MassField kind = fieldOf(field);
    if (kind == MassField::Other)
        return false;
    const Columns columns(view.columns());
    const Mass mass = resolve(view, columns, row).required();
    const double value = columns.m[static_cast<std::size_t>(row)];
    base::MatN<6> rate;
    if (kind == MassField::Mass)
        rate = rigidBodyMass(1.0, mass.offsetInBasic, {});
    else if (kind == MassField::Inertia)
    {
        const base::Mat3 tensor = mass.inputToBasic * inertiaRate(field.item) * mass.inputToBasic.transposed();
        rate = rigidBodyMass(0.0, {}, tensor);
    }
    else
        rate = offsetRate(value, mass.offsetInBasic, offsetDirection(mass.inputToBasic, field.item));
    const base::Mat3 frame = view.displacementFrame(mass.gridRow);
    appendMatrix(block, row, gridPointDofs(mass.gridRow), toDisplacementFrames<6>(rate, {frame, frame}));
    return true;
}

bool PointMassInertiaElement::massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row,
                                                       const DesignedField& field, const MassOptions& /*options*/,
                                                       ElementMatrixBlock& block) const
{
    const MassField kind = fieldOf(field);
    if (kind == MassField::Other)
        return false;
    if (kind != MassField::Offset)
        return true;
    const Columns columns(view.columns());
    const Mass mass = resolve(view, columns, row).required();
    const base::Mat3 direction = crossProductMatrix(offsetDirection(mass.inputToBasic, field.item));
    // Only the parallel-axis block is quadratic in one offset: -2 m [e_i]x [e_i]x.
    const base::Mat3 tensor = (-2.0 * columns.m[static_cast<std::size_t>(row)]) * (direction * direction);
    const base::Mat3 frame = view.displacementFrame(mass.gridRow);
    const base::MatN<6> rate = rigidBodyMass(0.0, {}, tensor);
    appendMatrix(block, row, gridPointDofs(mass.gridRow), toDisplacementFrames<6>(rate, {frame, frame}));
    return true;
}

bool PointMassInertiaElement::massFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                                       const DesignedField& first, const DesignedField& second,
                                                       const MassOptions& /*options*/, ElementMatrixBlock& block) const
{
    const MassField a = fieldOf(first), b = fieldOf(second);
    if (a == MassField::Other || b == MassField::Other)
        return false;
    if (a == MassField::Inertia || b == MassField::Inertia || (a == MassField::Mass && b == MassField::Mass))
        return true;
    const Columns columns(view.columns());
    const Mass mass = resolve(view, columns, row).required();
    base::MatN<6> rate;
    if (a == MassField::Mass || b == MassField::Mass)
    {
        const std::int32_t item = a == MassField::Offset ? first.item : second.item;
        rate = offsetRate(1.0, mass.offsetInBasic, offsetDirection(mass.inputToBasic, item));
    }
    else
    {
        const base::Mat3 directionA = crossProductMatrix(offsetDirection(mass.inputToBasic, first.item));
        const base::Mat3 directionB = crossProductMatrix(offsetDirection(mass.inputToBasic, second.item));
        const double value = columns.m[static_cast<std::size_t>(row)];
        const base::Mat3 tensor = -value * (directionA * directionB + directionB * directionA);
        rate = rigidBodyMass(0.0, {}, tensor);
    }
    const base::Mat3 frame = view.displacementFrame(mass.gridRow);
    appendMatrix(block, row, gridPointDofs(mass.gridRow), toDisplacementFrames<6>(rate, {frame, frame}));
    return true;
}

} // namespace oos::elements
