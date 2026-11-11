#pragma once

// MassDofProperty — scalar mass with its value on a MassProp (CMASS1, MassProp):
//
//   | MassDofProperty | EID | PID | G1 | C1 | G2 | C2 |
//   | MassProp  | PID | M   |
//
// Formulation: a scalar mass M acting on the relative motion of its two DOFs, as the scalar springs do (the mass
// lies between the two connection points): M·[1 −1; −1 1] on (u1, u2), [M] when one end is blank (grounded) — the
// usual use, a mass on one DOF. The same matrix for the lumped and the coupled formulations.
//
// Facets: mass and its exact design derivatives. A scalar mass has no stiffness or recovery output.
//
// Not supported: nothing on this card. Problems reported by check(): an unknown point, a component outside 1-6 on a
// grid point or other than 0 on a scalar point, both ends blank or equal, a PID that is not a MassProp.

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/core/ScalarConnection.hpp"

#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of MassDofProperty. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class MassDofPropertyElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "MassDofProperty"; }
    [[nodiscard]] FacetSet facets() const noexcept override { return {Facet::Mass, Facet::Derivatives}; }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    bool massFieldDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                             const MassOptions& options, ElementMatrixBlock& block) const override;
    bool massFieldSecondDerivative(const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                   const MassOptions& options, ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"MassProp"}; }

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;

private:
    struct Columns
    {
        std::span<const std::int32_t> g1;
        std::span<const std::int32_t> c1;
        std::span<const std::int32_t> g2;
        std::span<const std::int32_t> c2;
        PropertyRows properties;

        explicit Columns(const ElementBatchView& view);
    };

    struct Mass
    {
        ScalarConnection connection;
        double mass = 0.0;
    };

    [[nodiscard]] static Resolution<Mass> resolve(const ElementBatchView& view, const Columns& columns,
                                                  std::int64_t row);
};

} // namespace oos::elements
