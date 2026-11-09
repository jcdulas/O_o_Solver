#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/output/diagnostics/SubcaseChecks.hpp"

#include <array>
#include <cstdint>
#include <set>
#include <span>
#include <vector>

namespace oos::design
{

// The applied force of one grid point (the translations of the OLOAD table, in its displacement system CD).
struct GridForce
{
    std::int32_t grid = 0;
    std::array<double, 3> force{};
};

// The self-adjoint responses of the optimisation aids (a response whose adjoint vector is known needs no solve: the
// compliance fᵀu is self-adjoint, its adjoint is u):
//
// - compliance (ResponseType::Compliance), one item per subcase:  c = Pᵀu,
//   P every load of the subcase as the solve takes it (the thermal loads included), u the displacements. With no
//   enforced displacement, K_ll·u_l = (Tᵀ·P)_l, so c = u_lᵀ·K_ll·u_l = uᵀ·K_gg·u, twice the strain energy and twice
//   the external work of the static solution (StaticResults::checks), which gives its value;
// - mean displacement (ResponseType::MeanDisplacement), one item per subcase:  m = c / W,  W = Σ_g |F_g|,
//   F_g the applied force of grid point g (OLOAD, its translations; the thermal loads are not applied loads). Under
//   parallel forces m = Σ_g |F_g|·(u_g·e) / Σ_g |F_g|, the force-weighted mean of the displacement along the forces.
//
// Their displacement weights are ∂c/∂u = P (P/W): in K·λ = P, λ = u (u/W), with no solve. The gradient is then
//     dc/dx_j = uᵀ·∂P/∂x_j + uᵀ·b_j = 2·uᵀ·∂P/∂x_j − uᵀ·∂K/∂x_j·u,
// the element strain energies differenced (b_j = ∂P/∂x_j − ∂K/∂x_j·u, PseudoLoads), and for m the quotient rule with
// ∂W/∂x_j = Σ_g (F_g·∂F_g/∂x_j)/|F_g|. StaticSensitivity folds λᵀ·b into the explicit part of the item and drops its
// weights, so that no solve is counted for it (GradientOptions::usesSelfAdjoint; false forces the adjoint solve, the
// check of the tests). The sensitivity map takes λ = u the same way (SensitivityMap).
//
// With an enforced displacement (SPC D, EnforcedMotionConstr) K_ll·u_l = (Tᵀ·(P − K·u_e))_l, u is no longer the adjoint
// vector and Pᵀu no longer the work the solve does: both kinds are refused in such a subcase (OptimisationError naming
// it), as in a SUBCOM (no work of its own) and under inertia relief (the sensitivities refuse it).
//
// Thread-safety: stateless functions. Ownership: none kept.
class SelfAdjointResponses
{
public:
    // c of a subcase: twice the external work of its check. Throws OptimisationError when the results have no check
    // of the subcase or it is a SUBCOM.
    [[nodiscard]] static double compliance(std::span<const output::SubcaseCheck> checks, std::int32_t subcase);

    // The applied forces of the grid points of the OLOAD table ("appliedLoads") of a subcase whose force is not
    // zero, in the order of the table. Throws OptimisationError when the results have no such table.
    [[nodiscard]] static std::vector<GridForce> appliedForces(const output::ResultArrays& arrays, std::int32_t subcase);
    // W = Σ_g |F_g|.
    [[nodiscard]] static double forceMagnitude(std::span<const GridForce> forces) noexcept;

    // The subcases of `subcases` whose constraints enforce a displacement (SPC D or EnforcedMotionConstr not zero), on
    // the snapshot of `index`. Throws as SpcSet and StaticSubcases (problems of the model, as the analysis meets them).
    [[nodiscard]] static std::set<std::int32_t> subcasesWithEnforcedDisplacements(
        const elements::SnapshotIndex& index, std::span<const std::int32_t> subcases, base::MemoryBudget& budget);
};

} // namespace oos::design
