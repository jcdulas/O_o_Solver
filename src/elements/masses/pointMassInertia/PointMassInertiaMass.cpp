#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/masses/pointMassInertia/PointMassInertiaElement.hpp"

namespace oos::elements
{

// The rigid-body mass of PointMassInertia (rigidBodyMass) in the displacement system of G; lumped and coupled alike.
void PointMassInertiaElement::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& /*options*/,
                                    ElementMatrixBlock& block) const
{
    const Columns columns(view.columns());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Mass& mass = resolve(view, columns, row).required();
        appendMatrix(block, row, gridPointDofs(mass.gridRow), mass.massInDisplacementFrame);
    }
}

} // namespace oos::elements
