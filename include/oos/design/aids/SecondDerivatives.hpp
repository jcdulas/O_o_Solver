#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/responses/BatchCallable.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::design
{

// The relative step of the differences in x of the second derivatives: as the field step of
// the semi-analytic sensitivities (FIELD_STEP_RELATIVE, 1e-2 of |x_j|, FIELD_STEP_AT_ZERO at zero), extrapolated
// (Richardson, truncation O(h⁴), none for the polynomial dependences of a thickness): the functions differenced are
// the element-level Lagrangian at fixed displacements, whose rounding ε·|λ||K||u| is far below its curvature at that
// step (measured on the clamped plate: within 1e-7 of the central differences of the gradients).
//
// The step is scaled per moved variable and kept within the bounds of the design variables (the rule of
// design/common/DifferenceStencil.hpp): along a direction v it is t = 1e-2·min_j(s_j/|v_j|) over the moved variables,
// s_j = |x_j| (at zero the width of the bounds of x_j, FIELD_STEP_AT_ZERO / 1e-2 without bounds), so that no variable
// moves by more than 1e-2 of its own scale; for one variable (v = e_j) that is 1e-2·|x_j|. The states stay within
// [lower, upper] of every moved variable: a central difference clipped to the room, one-sided (three states toward
// the side with more room) at a bound; the half step of the extrapolation takes the same form at half the offsets.
// At a bound the one-sided differences are of lower order: O(h³) after the extrapolation for the first differences,
// O(h²) for the second difference along one variable.
inline constexpr double SECOND_DERIVATIVE_STEP_RELATIVE = 1.0e-2;

// What the second derivatives are asked: the responses (empty: every evaluated one) and the independent variables
// (columns of x; empty: every one).
struct SecondDerivativeRequest
{
    std::vector<std::int32_t> responseIds;
    std::vector<std::size_t> variables;
};

// d²r/dx_j dx_k of every item of one response: items × variables × variables, row-major.
struct ResponseHessian
{
    std::int32_t id = 0;
    std::size_t itemCount = 0;
    std::vector<double> values;
};

struct SecondDerivativesResult
{
    std::vector<std::size_t> variables; // the columns of x
    std::vector<ResponseHessian> responses;
    std::int64_t solves = 0;       // the adjoint solves (one per item, none self-adjoint) and the direct ones
    std::int64_t designPoints = 0; // the perturbed design points whose element computations were made
};

// The product of the Hessians of the items of one response with a direction v of x: items × variables.
struct HessianVectorResult
{
    std::int32_t id = 0;
    std::size_t itemCount = 0;
    std::vector<double> values;
    std::int64_t solves = 0;
    std::int64_t designPoints = 0;
};

// Second derivatives and Hessian-vector products of static responses (an optimisation aid beyond gradients):
// Newton steps, trust regions.
//
// The method (the adjoint second-order sensitivities: Haftka and Gürdal, Elements of Structural Optimization, 3rd ed.,
// §7.3; Nocedal and Wright, Numerical Optimization, 2nd ed., §8.4 for the Hessian-vector form). With K·u = P, the
// adjoint λ of an item (K·λ = c = ∂r/∂u) and du_j = K⁻¹·b_j (b_j = ∂P/∂x_j − ∂K/∂x_j·u: the direct method, one solve
// per variable), the Lagrangian L(x, w, λ) = r(w, x) − λᵀ·(K(x)·w − P(x)) gives the reduced Hessian
//     H_jk = ∂²L/∂x_j∂x_k + ∂_j(∇_w L·du_k) + ∂_k(∇_w L·du_j) + du_jᵀ·∇²_w r·du_k,
// the last term zero for the items linear in u, the only ones taken. Every term is a derivative in x, at fixed u, λ
// and du, of a scalar formed element by element on the design points x ± h·e_j (± h·e_k): the stiffness and thermal
// loads of the elements the variables reach, the distributed and inertial loads of the subcases that have them (with
// the non-structural mass set the subcase selects), and the item's own function (an element item recovered on the
// point, the mass properties, the deflection over thickness), differenced by the stencils of
// SECOND_DERIVATIVE_STEP_RELATIVE extrapolated (central within the bounds of the variables, one-sided at a bound);
// ∇_w L·du_k is the difference quotient of L along w = u + ε·du_k, exact for an item linear in u. The design
// relations enter through the design points themselves (a DesignPropRelation2 is evaluated at x ± h: its curvature
// is in L).
//
// The Hessian-vector product: (H·v)_j = Σ_k ∂²L/∂x_j∂x_k·v_k + ∂_j(∇_w L·du_v) + μᵀ·b_j, with du_v = K⁻¹·Σ_k v_k·b_k
// and μ = K⁻¹·∂_v(∇_w L) (the derivative along v of the item's weights less ∂K/∂v·λ): two solves per product besides
// the adjoint, whatever the number of variables. The first term is formed as v_j·∂²L/∂x_j² + ∂_j(∂_r L), r being v
// without its component j, so that no state moves a variable twice (beyond its bounds).
//
// Cost: one adjoint solve per item (none for a compliance: λ = u), one direct solve per variable (the full Hessian);
// the element computations of 1 + 4n + 8·n(n−1)/2 design points (n variables), or 1 + 12n + 4 for a product, away
// from the bounds (a one-sided stencil has three states where a central one has two).
//
// Refused (OptimisationError naming them): an item that is not linear in u (von Mises, principal stresses, strain
// energies, a rotation magnitude: their ∇²_w r is not formed), an equation response (DesignResp2, DesignResp3: the
// second derivatives of the equation are not formed), a field of a vibration mode (a response with a mode selector:
// the Hessian-vector product with weights of design/external/VectorJacobianProducts takes them), a mean displacement
// whose applied forces change with the design,
// the relations given by functions of the caller (the design points need their values), and what StaticSensitivity
// refuses.
//
// Thread-safety: run on the calling thread. Ownership: none kept.
class SecondDerivatives
{
public:
    // The Hessians of a kept evaluation (DesignProblem::evaluateStatics with needsDisplacements).
    [[nodiscard]] static SecondDerivativesResult compute(const DesignProblem& problem,
                                                         const DesignEvaluation& evaluation,
                                                         const SecondDerivativeRequest& request,
                                                         const CallableRegistry& callables, base::MemoryBudget& budget,
                                                         const solutions::StaticOptions& analysisOptions = {},
                                                         std::size_t blockItems = EQUATION_BATCH_ITEMS);

    // H·v for the items of one response; `direction` has one value per independent variable.
    [[nodiscard]] static HessianVectorResult hessianVector(
        const DesignProblem& problem, const DesignEvaluation& evaluation, std::int32_t responseId,
        std::span<const double> direction, const CallableRegistry& callables, base::MemoryBudget& budget,
        const solutions::StaticOptions& analysisOptions = {}, std::size_t blockItems = EQUATION_BATCH_ITEMS);
};

} // namespace oos::design
