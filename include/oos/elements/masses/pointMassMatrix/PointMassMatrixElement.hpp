#pragma once

// PointMassMatrix — concentrated mass given as a 6 × 6 matrix:
//
//   | PointMassMatrix | EID | G   | CID | M11 | M21 | M22 | M31 | M32 |
//   |       | M33 | M41 | M42 | M43 | M44 | M51 | M52 | M53 |
//   |       | M54 | M55 | M61 | M62 | M63 | M64 | M65 | M66 |
//
// Formulation (CONM1): the symmetric mass matrix M of the grid point G, its lower triangle given row by row (Mij
// with j ≤ i; O_o_Reader's column m holds the 21 terms in that order), in the coordinate system CID (0 the basic
// system). Carried to the displacement system of G by the rotation of each 3-component block:
// M_CD = Fᵀ·M·F block by block, F = Aᵀ·R (A the CID-to-basic rotation, R the CD-to-basic rotation of G). The same
// matrix for the lumped and the coupled formulations.
//
// Facets: mass and its exact first and same-field second derivatives in the 21 matrix terms.
//
// Not supported: a cylindrical or spherical CID (refused by check()). Problems reported by check(): G not a grid
// point, an unknown CID.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of PointMassMatrix. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class PointMassMatrixElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "PointMassMatrix"; }
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
        base::FixedListView<double> m;

        explicit Columns(const base::ContractReader& reader);
    };

    struct Mass
    {
        std::int32_t gridRow = 0;
        base::MatN<6> massInDisplacementFrame;
    };

    [[nodiscard]] static Resolution<Mass> resolve(const ElementBatchView& view, const Columns& columns,
                                                  std::int64_t row);
};

} // namespace oos::elements
