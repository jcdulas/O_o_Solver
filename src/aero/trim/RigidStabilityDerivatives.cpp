#include "oos/aero/trim/RigidStabilityDerivatives.hpp"

#include "oos/aero/dlm/DoubletLatticeSteady.hpp"
#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/aero/trim/ReferenceCoefficients.hpp"
#include "oos/aero/trim/TrimVariables.hpp"
#include "oos/linalg/dense/DenseLu.hpp"

#include <span>
#include <vector>

namespace oos::aero
{

std::vector<StabilityCoefficients> RigidStabilityDerivatives::compute(const input::ModelSnapshot& snapshot, double mach,
                                                                      base::MemoryBudget& budget)
{
    const AeroReference reference = AeroReference::read(snapshot);
    const AeroBoxes boxes(snapshot, reference, budget);
    const TrimVariables variables(snapshot, reference, boxes);
    const std::size_t n = boxes.size();
    const std::size_t m = variables.variables().size();
    const linalg::DenseLu factor(
        DoubletLatticeSteady::normalwashFactors(boxes, mach, reference.symmetryXz, reference.symmetryXy, budget),
        budget);
    linalg::DenseMatrix pressures(budget, n, m, "rigid pressure coefficients");
    for (std::size_t v = 0; v < m; ++v)
        variables.normalwashOf(variables.variables()[v].label, std::span<double>(&pressures(0, v), n));
    solvePressures(factor, pressures.view());

    const base::Vec3 origin = reference.referenceOriginInAeroFrame();
    std::vector<StabilityCoefficients> coefficients;
    coefficients.reserve(m);
    for (std::size_t v = 0; v < m; ++v)
    {
        // The force of box j per unit q: Δcp_j A_j n_j at its force point; summed in box order (the images of a half
        // model are added by ReferenceCoefficients).
        AeroResultant resultant;
        for (std::size_t j = 0; j < n; ++j)
        {
            const base::Vec3 boxForce = (pressures(j, v) * boxes[j].area) * boxes[j].normal;
            resultant.force = resultant.force + boxForce;
            resultant.moment = resultant.moment + base::cross(boxes[j].forcePoint - origin, boxForce);
        }
        StabilityCoefficients entry;
        entry.label = variables.variables()[v].label;
        entry.values = ReferenceCoefficients::of(reference, resultant);
        coefficients.push_back(std::move(entry));
    }
    return coefficients;
}

} // namespace oos::aero
