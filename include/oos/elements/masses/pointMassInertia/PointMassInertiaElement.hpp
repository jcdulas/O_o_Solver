#pragma once

// PointMassInertia — concentrated mass of a rigid body:
//
//   | PointMassInertia | EID | G   | CID | M   | X1  | X2 | X3 |     |
//   |       | I11 | I21 | I22 | I31 | I32 | I33 |    |     |
//
// Formulation (CONM2): a rigid body of mass M whose centre of gravity is offset from the grid point G by
// d = (X1, X2, X3), with the inertia matrix at the centre of gravity
//
//   J = [ I11 −I21 −I31 ; −I21 I22 −I32 ; −I31 −I32 I33 ]
//
// (I11 = ∫(x2² + x3²) dm ..., and the products of inertia I21 = ∫ x1·x2 dm ... enter the matrix
// with a minus sign), in the system CID; CID −1 gives X1-X3 as the coordinates of the centre of gravity in the basic
// system (d = X − x_G) and J in the basic system. The mass matrix at the grid point is that of the rigid body carried
// from the centre of gravity by the rigid link u_cg = u_G + θ × d:
//
//   M = [ m·I      −m·[d]× ;
//         m·[d]×    J − m·[d]×·[d]× ]       ([d]× the cross-product matrix of d, all in the basic system)
//
// (−[d]×[d]× = |d|²I − d·dᵀ: the parallel-axis terms). M is then carried to the displacement system of G. The same
// matrix for the lumped and the coupled formulations.
//
// Facets: mass and its exact first and same-field second derivatives in mass, offset and inertia.
// The assembler applies gravity and rotation loads through the mass matrix.
//
// Not supported: a cylindrical or spherical CID (refused by check()). Problems reported by check(): G not a grid
// point, an unknown CID.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The 6 × 6 mass matrix of PointMassInertia at the grid point, in the basic system: mass `mass`, centre of gravity at
// `offsetInBasic` from the grid point, inertia `inertiaAtCentre` at the centre of gravity (basic system).
[[nodiscard]] base::MatN<6> rigidBodyMass(double mass, base::Vec3 offsetInBasic, const base::Mat3& inertiaAtCentre);

// The element family of PointMassInertia. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class PointMassInertiaElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "PointMassInertia"; }
    [[nodiscard]] FacetSet facets() const noexcept override { return {Facet::Mass, Facet::Derivatives}; }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    bool massFieldMixedDerivative(const ElementBatchView& view, std::int64_t row,
                                  const DesignedField& first, const DesignedField& second,
                                  const MassOptions& options, ElementMatrixBlock& block) const override;
    bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const MassOptions& options, ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   const MassOptions& options, ElementMatrixBlock& block) const override;

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;

private:
    struct Columns
    {
        std::span<const std::int32_t> g;
        std::span<const std::int32_t> frameId;
        std::span<const double> m;
        base::FixedListView<double> x;
        base::FixedListView<double> i;

        explicit Columns(const base::ContractReader& reader);
    };

    struct Mass
    {
        std::int32_t gridRow = 0;
        base::MatN<6> massInDisplacementFrame;
        base::Vec3 offsetInBasic;
        base::Mat3 inputToBasic;
    };

    [[nodiscard]] static Resolution<Mass> resolve(const ElementBatchView& view, const Columns& columns,
                                                  std::int64_t row);
};

} // namespace oos::elements
