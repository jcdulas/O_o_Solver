#pragma once

// The items of the SAERO subcases in the sensitivities: TRIM, STABDER and the static items of those
// subcases, a part of the internal computation of SensitivityComputation (not a public header; included by the
// sources of the design module only).

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/ResponseValues.hpp"
#include "oos/design/sensitivity/ElementDerivatives.hpp"
#include "oos/design/sensitivity/ResponseLinearisation.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/input/tables/ParamTable.hpp"
#include "oos/solutions/sol144/TrimSolution.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace oos::design::detail
{

// The derivatives of the responses of the SAERO subcases with respect to the design: the trim and stability
// derivatives, by the derivative of the trim system, on the trim the analysis kept as a function of the structure
// (solutions::TrimLinearisation, whose class comment derives the equations).
//
// For every designed quantity v (a field, or a shape basis) and every direction d of the trim — the trimmed state of
// each SAERO subcase (u, w = D_r·ü_r), and the elastic coefficients of each variable a STABDER item reads (u_x, w_x)
// — the pseudo-loads −∂K/∂v·u_d and −∂M/∂v·w_d are formed element by element on the elements v reaches (the
// semi-analytic differences of their stiffness and mass facets, ElementDerivatives; the mass with the
// formulation and WTMASS the trim assembled it with), then the linearisation gives
//     the trimmed state: dx (TRIM items), du (the static items of the subcase: dr = cᵀ·du, c their displacement
//     weights, the explicit part at fixed u being the static linearisation's);
//     the elastic coefficients of a variable (STABDER items).
// Each quantity costs one structural solve per SAERO subcase (two with static items) and one per coefficient read, on
// the factor of the analysis. The derivatives are added to the explicit parts of the items through d(v)/dx (the
// static items' weights are then cleared: their derivative is complete), so that the solves of the static subcases
// and the equations that read the items see complete rows.
//
// Determinism: the directions, the quantities and the elements in a fixed order, every sum serial.
//
// Thread-safety: none (filled while the sensitivities are linearised). Ownership: borrows the trim results (and solves
// on their linearisation), the DOF map and the budget, which must outlive it.
class TrimItems
{
public:
    struct Inputs
    {
        const solutions::TrimResults* trim = nullptr; // with its linearisation
        const assembly::DofMap* dofMap = nullptr;
        const input::ParamTable* params = nullptr; // the design point's PARAMs (COUPMASS, WTMASS)
        base::MemoryBudget* budget = nullptr;
    };

    // Throws OptimisationError when the trim kept no linearisation (an analysis made without the gradients' option).
    explicit TrimItems(const Inputs& inputs);

    // Whether `subcase` is a SAERO subcase of the analysis, and its trimmed displacements (g set).
    [[nodiscard]] bool isTrimSubcase(std::int32_t subcase) const;
    [[nodiscard]] std::span<const double> displacementsOf(std::int32_t subcase) const;
    [[nodiscard]] std::vector<std::int32_t> subcases() const;

    // Registers a TRIM or STABDER item (`label` as DesignResp1Response gives it: the subcase, the variable's ID, the
    // component, the restraint flag). Throws OptimisationError for a variable the trim does not have.
    void add(const ItemKey& key, ResponseType type, const ResponseItem& label);
    // Registers a static item of a SAERO subcase whose displacement weights the static linearisation gave.
    void addStaticItem(const ItemKey& key, std::int32_t subcase);
    [[nodiscard]] bool isEmpty() const noexcept { return m_items.empty() && m_staticItems.empty(); }
    [[nodiscard]] std::size_t itemCount() const noexcept { return m_items.size() + m_staticItems.size(); }

    // The vectors of the directions; call once every item is added.
    void prepare();
    // The contribution of element `elementId` (row `row` of `family`) a designed quantity reaches (`stiffness` its
    // dK/dv as the caller differenced it, empty when the quantity does not change the stiffness). Throws
    // OptimisationError for an element of a condensed unit: the derivatives of the trim need the units the design
    // drives kept full.
    void addElement(std::size_t quantity, const elements::ElementFamily& family, std::int64_t row,
                    base::EntityId elementId, const ElementDerivatives& derivatives, const elements::FieldUse& use,
                    const ElementMatrixDerivative& stiffness);
    // Once every element of the quantity is added: the derivatives of every item along it.
    void finishQuantity(std::size_t quantity, const LinearisedItems& linearised);
    // The derivatives into the explicit parts of the items, the static items' weights cleared.
    void addExplicitParts(const std::vector<FieldValue>& quantities, LinearisedItems& linearised) const;
    // The structural solves the derivatives made.
    [[nodiscard]] std::int64_t solveCount() const noexcept { return m_solves; }

private:
    // A direction of the trim: the trimmed state of a subcase, or the elastic coefficients of one of its variables.
    struct Direction
    {
        std::size_t subcase = 0; // index into the linearisation's subcases
        bool isState = true;
        std::size_t variable = 0;
        bool isUnrestrained = false;
        std::span<const double> displacements;      // u
        std::span<const double> motions;            // w
        base::BudgetedBuffer<double> stiffnessLoad; // −∂K/∂v·u of the current quantity (g set)
        base::BudgetedBuffer<double> inertialLoad;  // −∂M/∂v·w
        bool isTouched = false;
    };
    struct Item
    {
        ItemKey key;
        ResponseType type = ResponseType::TrimVariable;
        std::size_t subcase = 0;
        std::size_t variable = 0;  // index into the trim case's variables
        std::size_t component = 0; // STABDER: 0 ... 5
        bool isUnrestrained = false;
        std::size_t direction = 0;
    };
    struct StaticItem
    {
        ItemKey key;
        std::size_t subcase = 0;
    };

    [[nodiscard]] std::size_t subcaseIndex(std::int32_t label) const;
    [[nodiscard]] std::size_t directionOf(std::size_t subcase, bool isState, std::size_t variable, bool isUnrestrained);

    Inputs m_in;
    solutions::TrimLinearisation* m_linearisation = nullptr;
    std::vector<base::EntityId> m_subcases; // the linearisation's, in order
    std::vector<Direction> m_directions;
    std::vector<Item> m_items;
    std::vector<StaticItem> m_staticItems;
    std::vector<elements::MassOptions> m_massOptions; // per subcase
    std::vector<double> m_weightToMass;               // per subcase
    // Per item (m_items, then m_staticItems at m_items.size() + k) and quantity: d(item)/dv.
    std::map<std::pair<std::size_t, std::size_t>, double> m_byQuantity;
    std::int64_t m_solves = 0;
};

} // namespace oos::design::detail
