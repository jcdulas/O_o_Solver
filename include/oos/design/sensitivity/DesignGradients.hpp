#pragma once

#include "oos/design/responses/ResponseValues.hpp"
#include "oos/design/sensitivity/BoundaryReducedStatistics.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace oos::design
{

// The order in which the gradients of the items of one subcase are formed, all giving the same G to round-off:
// - Direct: du/dx_j = K⁻¹·b_j, one solve per design variable with a non-zero pseudo-load; every item then reads
//   dr/dx_j = ∂r/∂x_j + c·du/dx_j;
// - Adjoint: λ_i = K⁻¹·c_iᵀ, one solve per item read with the displacements (an item an equation reads counted once),
//   dr_i/dx_j = ∂r_i/∂x_j + λ_iᵀ·b_j; the equations combine the gradients of their arguments;
// - Chained: the adjoint of each equation item formed at once from the weights of its arguments,
//   c_e = Σ_k ∂h/∂r_k·c_k, one solve per equation item (and per plain item asked for itself).
// Automatic takes the order with the fewest solves (MethodChoice), the direct one on a tie.
enum class SensitivityMethod : std::uint8_t
{
    Automatic,
    Direct,
    Adjoint,
    Chained
};

[[nodiscard]] std::string_view sensitivityMethodName(SensitivityMethod method) noexcept;

// How the derivatives of the element computations are formed:
// - Analytic: the items' derivatives with respect to the displacements from the slopes of the element recoveries
//   (exact: the recovery matrices of the families, ElementFamily::recoverySlopes, or the recoveries at unit
//   displacements), and the stiffness and mass derivatives with respect to the fields the families give in closed form
//   (ElementFamily::stiffnessFieldDerivative, massFieldDerivative); every other derivative differenced;
// - Differences: the central differences of every element computation (the items' recoveries by DOF, the facets on
//   the perturbed snapshots), the reference implementation the analytic derivatives are tested against.
enum class ElementDerivativeMethod : std::uint8_t
{
    Analytic,
    Differences
};

// The items of one response whose gradients are wanted: isRetained[i] for its item i (in the item order of its
// ResponseValues). The retained items of a screening: the screened gradient mode differentiates
// the retained responses, counted in items.
struct ItemMask
{
    std::int32_t responseId = 0;
    std::vector<std::uint8_t> isRetained;
};

// What a gradient computation does besides the analysis.
struct GradientOptions
{
    SensitivityMethod method = SensitivityMethod::Automatic; // forced for tests and comparisons
    // The responses whose gradients are wanted (their IDs); empty: every evaluated response. The responses an
    // equation reads are linearised for it, and get no gradient row of their own unless they are asked for.
    std::vector<std::int32_t> responseIds;
    // The items wanted of some of those responses (a response without a mask: every item). The items left out are not
    // linearised, not solved for and not counted by MethodChoice, and their gradient rows are NaN; the items an
    // equation of a wanted item reads are linearised for it whatever the mask.
    std::vector<ItemMask> itemMasks;
    // Whether the self-adjoint responses (compliance, mean displacement: design/aids/SelfAdjointResponses) take their
    // adjoint vector from the displacements, with no solve (true), or are solved for as any other item (false: the
    // check that both give the same gradient).
    bool usesSelfAdjoint = true;
    // The element derivatives: analytic where the families give them, or the differences everywhere (the
    // reference the tests compare them with).
    ElementDerivativeMethod elementDerivatives = ElementDerivativeMethod::Analytic;
    // The independent variables whose pseudo-loads and explicit parts are wanted, one flag per variable (a
    // restricted Jacobian-vector product along directions on a few variables); empty: every variable. A designed field
    // whose gradient reaches no flagged variable is not differenced, so that the pseudo-loads and the explicit parts of
    // the other variables may be missing: only products along directions on the flagged variables read such a
    // linearisation.
    std::vector<std::uint8_t> isVariableWanted;
};

// Whether the gradient of item `item` of response `responseId` is wanted by `options`' masks (true without a mask).
[[nodiscard]] bool isItemDifferentiated(const GradientOptions& options, std::int32_t responseId,
                                        std::size_t item) noexcept;

// The solves of one subcase as MethodChoice counted them, and the order taken.
struct SubcaseSensitivity
{
    std::int32_t subcase = 0;
    std::int64_t variables = 0;      // the design variables with a non-zero pseudo-load in the subcase
    std::int64_t argumentItems = 0;  // the items read with the displacements, an argument of several equations once
    std::int64_t requestedItems = 0; // the plain items asked for themselves plus the equation items asked for
    std::int64_t directSolves = 0;
    std::int64_t adjointSolves = 0;
    std::int64_t chainedSolves = 0;
    SensitivityMethod method = SensitivityMethod::Direct;
    std::int64_t solves = 0; // the solves made
};

// What a gradient computation did: the method chosen per subcase, the number of solves, and what was differenced.
struct SensitivityReport
{
    std::vector<SubcaseSensitivity> subcases;
    std::int32_t factorisations = 0;          // the factorisations made for the sensitivities (one per constraint set)
    std::int64_t solves = 0;                  // the right-hand sides solved for the sensitivities
    std::int64_t fieldPerturbations = 0;      // the designed fields differenced
    std::int64_t shapePerturbations = 0;      // the shape variables differenced (their bases moved)
    std::int64_t elementDerivatives = 0;      // the element computations differenced (an element and a quantity)
    std::int64_t displacementDifferences = 0; // the element recoveries differenced with respect to the displacements
    // The element outputs whose slopes a family gave (one call each), the recoveries at unit
    // displacements of the outputs a family gives no slopes for, and the element matrices differentiated in closed
    // form (an element, a quantity and a matrix: stiffness or mass).
    std::int64_t recoverySlopes = 0;
    std::int64_t unitRecoveries = 0;
    std::int64_t analyticDerivatives = 0;
    // The explicit parts of an element job's items (a job and a quantity) the family gave in closed form, its
    // recoveries on the perturbed sides not made.
    std::int64_t analyticExplicitParts = 0;
    std::int64_t selfAdjointItems = 0; // the items whose adjoint vector was the displacement (no solve)
    std::int64_t eigenvalueItems = 0;  // the EIGN, FREQ and LAMA items
    std::int64_t repeatedRoots = 0;    // their repeated roots, each differentiated by its subspace problem
    // The wall time of the two largest parts of the linearisation:
    // the element items' values and derivatives with respect to the displacements, and the designed fields and shapes
    // (their pseudo-loads and the explicit parts of the items).
    double itemSeconds = 0.0;
    double fieldSeconds = 0.0;
    // On the condensed path, per constraint set: the solves through the boundaries, unit by unit.
    std::vector<BoundaryReducedStatistics> condensed;
};

// The gradient of one response with respect to x (the independent variables, DesignModel::independentIndices order):
// items × variables, row-major, in the item order of its ResponseValues; the rows of the items an item mask left out
// are NaN.
struct ResponseGradient
{
    std::int32_t id = 0;
    std::size_t itemCount = 0;
    std::vector<double> values;
};

// The Rayleigh quotient parts of an EIGN or FREQ item of a single root: with φ the M-normalised mode, the
// gradients with respect to x of its strain energy φᵀ·K·φ (the numerator) and of its kinetic energy φᵀ·M·φ (the
// denominator) at fixed φ, λ = φᵀKφ / φᵀMφ; dλ/dx = numerator − λ·denominator. The approximate problem builds the
// Rayleigh quotient approximation of the eigenvalue responses from them (Canfield, "High-quality approximation of
// eigenvalues in structural optimization", AIAA J. 28(6), 1990).
struct RayleighGradient
{
    std::int32_t responseId = 0;
    std::size_t item = 0;
    double root = 0.0; // λ
    std::vector<double> numerator;
    std::vector<double> denominator;
};

// The gradients of a design point: per response, an (items × variables) array.
struct DesignGradients
{
    std::size_t variableCount = 0;
    std::vector<ResponseGradient> responses; // the responses asked for, in the order of the evaluation's responses
    // Per constraint value of the evaluation (same order): d(normalised)/dx, empty for a constraint on a response
    // whose gradient was not asked for. The normalised value is (L − r)/max(|L|, GSCAL) or (r − U)/max(|U|, GSCAL),
    // so its gradient is ∓ dr/dx / max(|bound|, GSCAL).
    std::vector<std::vector<double>> constraints;
    std::optional<std::vector<double>> objective; // d(objective)/dx, when the objective's response was asked for
    SensitivityReport report;
    // The Rayleigh quotient parts of the EIGN and FREQ items of single roots that were differentiated.
    std::vector<RayleighGradient> rayleigh;
};

} // namespace oos::design
