#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/elements/core/SolidIntegrationRules.hpp"
#include "oos/elements/solids/SolidConnectivity.hpp"
#include "oos/elements/solids/SolidResponse.hpp"
#include "oos/elements/solids/SolidSection.hpp"

#include <array>
#include <span>

namespace oos::elements
{

// Everything a facet needs about one solid element: its grid points, its section, and its node positions in its
// material system (x_m = axesᵀ·(x − origin), SolidProp CORDM), where its response is formed.
struct SolidElementData
{
    SolidNodes nodes;
    SolidSection section;
    std::array<base::Vec3, MAX_SOLID_NODES> materialPositions{};

    [[nodiscard]] std::span<const base::Vec3> positionsInMaterialSystem() const
    {
        return {materialPositions.data(), nodes.count};
    }
};

// The formulation a card chooses for one element (from its shape and its integration): fills the response.
using SolidFormulation = void (*)(const SolidElementData&, SolidResponseParts, SolidResponse&);

// The facets shared by the four solid families (CHEXA, SolidPenta, SolidTetra, SolidPyramid): the families differ only by
// their shapes and by the formulation they choose; each family's facet files call these. The response of an element is
// formed in its material system and carried to the displacement systems of its grid points through T_a = axesᵀ·R_a
// (u_material = T_a·u_CD at node a): K_CD(a, b) = T_aᵀ·K(a, b)·T_b, loads F_CD = R_aᵀ·F_basic.
//
// Mass: the consistent mass ∫ ρ NᵀN dV, or (PARAM COUPMASS ≤ 0, the default) the lumped mass of the HRZ method
// (Hinton, Rock & Zienkiewicz, Earthquake Eng. Struct. Dyn. 4, 1976: the diagonal of the consistent mass scaled to
// the element's mass ρV; for the linear tetrahedron and pentahedron and for a parallelepiped SolidHexa this is ρV shared
// equally between the nodes). The NSM cards do not apply to solids (an NSM entry names no solid property):
// MassOptions' added mass is ignored.
//
// Thermal loads (linear statics: the thermal strain ε_T = a·(T − T0), a the expansion of
// SolidSection): the temperature change of each node from the grid point temperatures of the load set (TEMP,
// DefaultTemperatureLoad) minus those of the TEMPERATURE(INITIAL) set, or the material's TREF without one, interpolated
// inside the element by its shape functions; the equivalent nodal loads (Zienkiewicz & Taylor vol. 1 §6.2, initial
// strains) in the material system, carried to the displacement systems: for the element's mean change ΔT̄, K·u_T with
// the element's own stiffness and u_T the free expansion of the constant strain a·ΔT̄ (exact for every formulation,
// each reproducing a constant strain: a uniform change leaves a free body without stress); for the rest ΔT − ΔT̄, ∫
// Bᵀ·D·a·(ΔT − ΔT̄) dV with the isoparametric B on the mass rule of the shape (the isoparametric solids: the whole is
// then ∫ Bᵀ·D·ε_T dV), and for a formulation with condensed internal modes (the incompatible-modes SolidHexa) their
// share of the load (SolidResponse::thermalLoad). The recovery and the differential stiffness subtract the thermal
// stress D·ε_T at each place (the temperature change there) and add the strain the internal modes take from the thermal
// load (SolidResponse::thermalStrain); the strains are the total strains.
//
// Thread-safety: immutable; the facets may run concurrently on distinct ranges and blocks.
class SolidElementKernel
{
public:
    constexpr SolidElementKernel(SolidCardShapes shapes, SolidFormulation formulation) noexcept
        : m_shapes(shapes), m_formulation(formulation)
    {
    }

    // The batch's columns, read once per facet call.
    struct Columns
    {
        SolidConnectivity connectivity;
        SolidSectionColumns sections;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] Resolution<SolidElementData> resolve(const ElementBatchView& view, const Columns& columns,
                                                       std::int64_t row) const;

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const;
    // The stresses and strains of the range at once (ElementFamily::recoverOutputs): each element resolved and
    // formulated once for every output, each block the rows recover() gives its output. Throws
    // std::invalid_argument for a layer output or an output the solids do not give.
    void recoverOutputs(const ElementBatchView& view, ElementRange range, std::span<const RecoveryOutput> outputs,
                        const NodalDisplacements& displacements, std::span<ElementResultBlock> blocks) const;
    // The recovery matrices of the stresses and strains of the element of batch row `row` (RecoverySlopes): the
    // operators of its formulation (SolidResponse, built once for every output) carried
    // to the displacement systems of its nodes, S(p, c, 3a + j) = Σ_i O_p(c, 3a + i)·T_a(i, j) with O_p the stress or
    // strain operator of place p and T_a the rotation of node a to the material system (u_material = T_a·u_CD), exact
    // as the recovery's chain σ = O_p·(T·u) to round-off. Throws std::invalid_argument for a layer output or an output
    // the solids do not give.
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs, std::span<RecoverySlopes> slopes) const;
    // The initial-stress stiffness of the preload's stress at the centre of each element (the recovery operator of
    // the formulation, in the material system; buildInitialStressStiffness with the mass rule of the shape), carried
    // to the displacement systems.
    void computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                      const NodalDisplacements& preload, ElementMatrixBlock& block) const;
    void checkPressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                            base::DiagnosticSink& sink) const;
    void computePressureLoads(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                              ElementVectorBlock& block) const;
    // A grid point of an element without a temperature in the load set is a problem (field TEMPERATURE(LOAD)).
    void checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                           base::DiagnosticSink& sink) const;
    void computeThermalLoads(const ElementBatchView& view, ElementRange range, const ElementTemperatures& temperatures,
                             ElementVectorBlock& block) const;

private:
    SolidCardShapes m_shapes;
    SolidFormulation m_formulation;
};

// The rule of the consistent mass of a shape (exact for ρ N_a N_b on a parallelepiped or a straight-edged element:
// degree 2 for the linear shapes, 4 for the quadratic ones; the collapsed pyramids one Gauss point more along each
// axis for the factor (1 − ζ)² of their Jacobian).
[[nodiscard]] const SolidRule& solidMassRule(SolidShape shape);

} // namespace oos::elements
