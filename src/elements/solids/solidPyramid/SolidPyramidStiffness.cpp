#include "oos/elements/solids/SolidIsoparametric.hpp"
#include "oos/elements/solids/solidPyramid/SolidPyramidElement.hpp"

namespace oos::elements
{

namespace
{

// The Gauss points per axis of the collapsed hexahedron: 2 for 5 nodes, 3 for 13 nodes.
constexpr std::size_t POINTS_LINEAR = 2;
constexpr std::size_t POINTS_QUADRATIC = 3;

} // namespace

void SolidPyramidElement::buildResponse(const SolidElementData& data, SolidResponseParts parts, SolidResponse& response)
{
    const bool isLinear = data.nodes.shape == SolidShape::Pyram5;
    buildIsoparametricResponse(data.nodes.shape, data.positionsInMaterialSystem(), data.section.elasticity,
                               hexahedronRule(isLinear ? POINTS_LINEAR : POINTS_QUADRATIC),
                               CornerRecovery::FromHexaGauss, parts, response);
}

// The stiffness of buildResponse in the displacement systems of the grid points (SolidElementKernel).
void SolidPyramidElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                          ElementMatrixBlock& block) const
{
    KERNEL.computeStiffness(view, range, block);
}

} // namespace oos::elements
