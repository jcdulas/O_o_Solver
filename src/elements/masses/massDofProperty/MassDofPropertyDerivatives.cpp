#include "oos/elements/masses/massDofProperty/MassDofPropertyElement.hpp"

namespace oos::elements
{

// M = m s s^T, so the mass rate is s s^T and the same-field second rate is zero.
FieldUse MassDofPropertyElement::designFieldUse(const DesignedField& field) const
{
    // MassProp: M is a scalar mass; it changes the mass only.
    return findFieldUse(field, {{"MassProp", "mass", FieldUse::massOnly()}});
}

bool MassDofPropertyElement::massFieldDerivative(const ElementBatchView& view, std::int64_t row,
                                                  const DesignedField& field, const MassOptions& /*options*/,
                                                  ElementMatrixBlock& block) const
{
    if (field.card != "MassProp" || field.column != "mass" || field.item != -1)
        return false;
    const Columns columns(view);
    appendScalarMatrix(block, row, resolve(view, columns, row).required().connection, 1.0);
    return true;
}

bool MassDofPropertyElement::massFieldSecondDerivative(const ElementBatchView& /*view*/, std::int64_t /*row*/,
                                                        const DesignedField& field, const MassOptions& /*options*/,
                                                        ElementMatrixBlock& /*block*/) const
{
    return field.card == "MassProp" && field.column == "mass" && field.item == -1;
}

} // namespace oos::elements
