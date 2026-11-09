#pragma once

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The DOFs of a scalar element (CELAS1-4, MassDofProperty-4): one or two (point, component) pairs, each with the sign it has
// in the relative motion the element acts on, δ = u1 − u2 (the spring force is k (u1 − u2); an end left blank is
// grounded, its displacement zero). The element matrix of a scalar value v (stiffness or mass) is then v · s·sᵀ over
// the present DOFs, s their signs: v·[1 −1; −1 1] for two ends, [v] for one (CELAS1, MassDofProperty).
struct ScalarConnection
{
    std::array<ElementDof, 2> dofs{};
    std::array<double, 2> signs{};
    std::size_t count = 0;

    [[nodiscard]] std::span<const ElementDof> presentDofs() const noexcept { return {dofs.data(), count}; }
    // δ = u1 − u2 from the displacements of the model.
    [[nodiscard]] double relativeMotion(const NodalDisplacements& displacements) const;
};

// Resolves the two ends (G1, C1) and (G2, C2) of a SpringDofProperty, SpringDofInline, MassDofProperty or MassDofInline (Gi a grid point or
// a scalar point ID, 0 or blank for a grounded end; Ci 1-6 for a grid point, 0 or blank for a scalar point). Fields are
// named `g1Field`... in the diagnostics ("G1", or "S1" for SpringScalarProperty, SpringScalarInline, MassScalarProperty, MassScalarInline, which
// connect scalar points: pass component 0 for them).
[[nodiscard]] Resolution<ScalarConnection> resolveScalarConnection(const ElementBatchView& view, std::int64_t row,
                                                                   std::int32_t g1, std::int32_t c1, std::int32_t g2,
                                                                   std::int32_t c2, bool isScalarPointCard);

// Appends v · s·sᵀ over the connection's DOFs to the block.
void appendScalarMatrix(ElementMatrixBlock& block, std::int64_t row, const ScalarConnection& connection, double value);

} // namespace oos::elements
