#include "oos/elements/solids/SolidIsoparametric.hpp"
#include "oos/elements/solids/solidHexa/SolidHexaElement.hpp"
#include "oos/elements/solids/solidHexa/SolidHexaIncompatibleModes.hpp"

namespace oos::elements
{

namespace
{

// The Gauss points per axis of the isoparametric SolidHexa: 2 for the 8-node element with ISOP = FULL, 3 for the 20-node
// element (exact on a parallelepiped), 2 for the 20-node element with the reduced rule.
constexpr std::size_t FULL_POINTS_LINEAR = 2;
constexpr std::size_t FULL_POINTS_QUADRATIC = 3;
constexpr std::size_t REDUCED_POINTS_QUADRATIC = 2;

} // namespace

void SolidHexaElement::buildResponse(const SolidElementData& data, SolidResponseParts parts, SolidResponse& response)
{
    const std::span<const base::Vec3> x = data.positionsInMaterialSystem();
    const SolidIntegration integration = data.section.integration;
    if (data.nodes.shape == SolidShape::Hexa8)
    {
        if (integration == SolidIntegration::Standard)
            buildSolidHexaIncompatibleModesResponse(x, data.section.elasticity, data.section.expansion, parts, response);
        else
            buildIsoparametricResponse(SolidShape::Hexa8, x, data.section.elasticity,
                                       hexahedronRule(FULL_POINTS_LINEAR), CornerRecovery::Direct, parts, response);
        return;
    }
    if (integration == SolidIntegration::ReducedGauss)
        buildIsoparametricResponse(SolidShape::Hexa20, x, data.section.elasticity,
                                   hexahedronRule(REDUCED_POINTS_QUADRATIC), CornerRecovery::AllFromHexaGauss, parts,
                                   response);
    else
        buildIsoparametricResponse(SolidShape::Hexa20, x, data.section.elasticity,
                                   hexahedronRule(FULL_POINTS_QUADRATIC), CornerRecovery::Direct, parts, response);
}

// The stiffness of buildResponse in the displacement systems of the grid points (SolidElementKernel).
void SolidHexaElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                       ElementMatrixBlock& block) const
{
    KERNEL.computeStiffness(view, range, block);
}

} // namespace oos::elements
