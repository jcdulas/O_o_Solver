#include "oos/elements/solids/SolidIsoparametric.hpp"
#include "oos/elements/solids/solidTetra/SolidTetraElement.hpp"

namespace oos::elements
{

namespace
{

// The degree of the tetrahedron rules: the 4-node element has constant strains (one point); the 10-node element
// linear strains, whose stiffness the four-point rule (degree 2) integrates exactly with straight edges, but the patch
// test of curved edges needs ∫∇N dV = ∫ adj(J)ᵀ∇_ξN exactly, a cubic: the five-point rule of degree 3.
constexpr std::size_t RULE_DEGREE_LINEAR = 1;
constexpr std::size_t RULE_DEGREE_QUADRATIC = 3;

} // namespace

void SolidTetraElement::buildResponse(const SolidElementData& data, SolidResponseParts parts, SolidResponse& response)
{
    const bool isLinear = data.nodes.shape == SolidShape::Tetra4;
    buildIsoparametricResponse(data.nodes.shape, data.positionsInMaterialSystem(), data.section.elasticity,
                               tetrahedronRule(isLinear ? RULE_DEGREE_LINEAR : RULE_DEGREE_QUADRATIC),
                               CornerRecovery::Direct, parts, response);
}

// The stiffness of buildResponse in the displacement systems of the grid points (SolidElementKernel).
void SolidTetraElement::computeStiffness(const ElementBatchView& view, ElementRange range,
                                        ElementMatrixBlock& block) const
{
    KERNEL.computeStiffness(view, range, block);
}

} // namespace oos::elements
