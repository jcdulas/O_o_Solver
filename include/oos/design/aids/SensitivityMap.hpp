#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/responses/BatchCallable.hpp"
#include "oos/design/responses/ResponseValues.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oos::design
{

// How the derivatives of a sensitivity map are summed (by="element" | "property" | "unit").
enum class MapGrouping : std::uint8_t
{
    Element,  // one column per element reached
    Property, // the elements of one property summed (the derivative with respect to the property's own value)
    Unit      // the elements of one unit (component) summed
};

[[nodiscard]] std::string_view mapGroupingName(MapGrouping grouping) noexcept;
[[nodiscard]] std::optional<MapGrouping> findMapGrouping(std::string_view name) noexcept;

// What a sensitivity map is asked.
struct SensitivityMapRequest
{
    std::int32_t responseId = 0;
    std::string card;  // the card whose field is mapped: "ShellProp", "IsoMat", "SpringDofInline"
    std::string field; // the column of the data contract: "t", "e", "stiffness"
    MapGrouping grouping = MapGrouping::Element;
    bool usesSelfAdjoint = true; // false: the adjoint of a compliance is solved for (the check of the tests)
};

// A sensitivity map: the derivative of every item of a response with respect to the field, element by element as if
// each element had its own value, then summed by group. The rows are the items (in the order of the evaluation), the
// columns the groups: an element (its card and ID), a property (its card and PID: an element field is grouped by the
// element's property), a unit (its component ID, the card empty).
struct SensitivityMapResult
{
    std::int32_t responseId = 0;
    std::vector<ResponseItem> items;
    std::vector<std::string> groupCards;
    std::vector<std::int64_t> groupIds;
    std::vector<double> values; // items × groups, row-major
    std::int64_t solves = 0;    // the adjoint solves made (none for a self-adjoint response)
    std::int64_t selfAdjointItems = 0;
    std::int64_t elementDerivatives = 0; // the elements whose computations were differenced
    std::int32_t factorisations = 0;
};

// The eigenvalue responses are mapped too: per element φᵀ·(∂K_e − λ·∂M_e)·φ for EIGN and FREQ (no solve),
// λ·φᵀ·(∂K_e + λ·∂K_d,e)·φ for LAMA plus its preload part λ_cᵀ·q_e from the adjoint of its weights (one solve per
// buckling factor); a repeated vibration root takes the eigenvalues of its subspace problem along each element, a
// repeated buckling root is refused.
//
// The sensitivity map of a response (an optimisation aid beyond gradients): the
// derivative of the response with respect to a field "as if each element had its own value", for every element at
// once: where adding or removing material matters most.
//
// With the adjoint vector λ_i of item i (K·λ_i = c_iᵀ, one solve per item; λ = u for a compliance, scale·u for a mean
// displacement: SelfAdjointResponses, no solve), the derivative with respect to the value v_e of element e is
//     dr_i/dv_e = ∂r_i/∂v_e + λ_iᵀ·(∂P_e/∂v_e − ∂K_e/∂v_e·u_e),
// the stiffness and thermal loads of element e alone differenced on the snapshots where every value of the column is
// moved at once (ColumnPerturbation: an element reads its own value only), its explicit part from the item's own
// dependence on that element (an element item of e itself, the SPC forces through −wᵀ·q_e, the mass of e for WEIGHT,
// VOLUME and FRMASS, the thickness of e in the mean thickness of a deflection over thickness, the share of e in the
// total energy of ESE item 3; an equation chains those of its arguments and solves its chained adjoint). The sum over
// the elements of a property is the gradient with respect to that property's value, which the tests check, and the
// element values equal the gradients of a model where each element has its own property.
//
// Refused (OptimisationError naming them): a response with no item; a field whose change moves the inertial or the
// distributed loads of a subcase the response reads (a density under GravityLoad: the inertial load of each element
// would be needed apart); an element that reads two values of the field (a ShellProp whose MID1 and MID2 are two IsoMat
// of the map); the grouping by unit on a model without unit tables.
//
// Thread-safety: run on the calling thread (the callables of the equations are called from it). Ownership: none kept.
class SensitivityMap
{
public:
    // The map of a kept evaluation (DesignProblem::evaluateStatics with needsDisplacements). Throws OptimisationError
    // (above), and as StaticSensitivity.
    [[nodiscard]] static SensitivityMapResult compute(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                      const SensitivityMapRequest& request,
                                                      const CallableRegistry& callables, base::MemoryBudget& budget,
                                                      const solutions::StaticOptions& analysisOptions = {},
                                                      std::size_t blockItems = EQUATION_BATCH_ITEMS);
};

} // namespace oos::design
