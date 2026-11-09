#include "oos/elements/solids/SolidIsoparametric.hpp"
#include "oos/elements/solids/solidPenta/SolidPentaElement.hpp"

namespace oos::elements
{

namespace
{

// The isoparametric SolidPenta: 3 triangle points × 2 Gauss points for 6 nodes, a degree-4 triangle rule × 3 points for
// 15 nodes (each exact for the stiffness of a straight prism).
constexpr std::size_t TRIANGLE_DEGREE_LINEAR = 2;
constexpr std::size_t ZETA_POINTS_LINEAR = 2;
constexpr std::size_t TRIANGLE_DEGREE_QUADRATIC = 4;
constexpr std::size_t ZETA_POINTS_QUADRATIC = 3;

} // namespace

void SolidPentaElement::buildResponse(const SolidElementData& data, SolidResponseParts parts, SolidResponse& response)
{
    const bool isLinear = data.nodes.shape == SolidShape::Penta6;
    const SolidRule& rule = isLinear ? pentahedronRule(TRIANGLE_DEGREE_LINEAR, ZETA_POINTS_LINEAR)
                                     : pentahedronRule(TRIANGLE_DEGREE_QUADRATIC, ZETA_POINTS_QUADRATIC);
    buildIsoparametricResponse(data.nodes.shape, data.positionsInMaterialSystem(), data.section.elasticity, rule,
                               CornerRecovery::Direct, parts, response);
}

// The stiffness of buildResponse in the displacement systems of the grid points (SolidElementKernel).
void SolidPentaElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                        ElementMatrixBlock& block) const
{
    KERNEL.computeStiffness(view, range, block);
}

} // namespace oos::elements
