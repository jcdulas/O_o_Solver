#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/expressions/ExpressionEvaluator.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/responses/BatchCallable.hpp"
#include "oos/design/responses/ResponseContext.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/ResponseValues.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// The callables an evaluation may call, by name: "GROUP/TYPE" for a DesignResp3, the registered name of a caller's
// declaration. Borrowed; called on the calling thread.
using CallableRegistry = std::map<std::string, BatchCallable*>;

// The values of the responses evaluated so far, by response ID.
using KnownResponses = std::map<std::int32_t, const ResponseValues*>;

// Where one argument value of an equation item comes from, for the chain rule of its sensitivities
// (de/dx = ∂h/∂x + Σ_k ∂h/∂r_k · dr_k/dx): an item of a response (its ID and the index of the item in its values), a
// design variable (its index in the model), a designed field (its relation index), a grid point coordinate (a DNODE:
// `index` is the argument's position; it moves with the shape variables that move its grid point:
// EquationResponse::coordinateShapes), or a constant (a DesignConstants value: no derivative; `index` is the argument's
// position).
struct ArgumentSource
{
    enum class Kind : std::uint8_t
    {
        Response,
        Variable,
        Field,
        Constant,
        Coordinate
    };

    Kind kind = Kind::Constant;
    std::int32_t responseId = 0;
    std::size_t index = 0;
};

// The values of an equation response and, for each item (in the order of the values), the partial derivatives of its
// equation with respect to the argument values it was computed from: (source, ∂h/∂argument), the zero ones left out.
struct EquationPartials
{
    ResponseValues values;
    std::vector<std::vector<std::pair<ArgumentSource, double>>> items;
};

// One equation response (DRESP2, DesignResp3, or a caller's add_equation_response) resolved against the design model
// and the snapshot.
//
// By the card rules (byLabel false), the response has, in each subcase it is evaluated in (or once, subcase 0, when it
// is global), the values of its equation at the point made of its arguments in their order (the arguments fill the
// variables of the equation's header x1, x2, x3... in that order): a DESVAR its value, a DesignConstants its constant,
// a DesignResp1 or a DesignResp2 every item of that response in the subcase (the items of a global response when it is
// global), a DNODE the basic coordinate, a DVxRELi the value of the designed field. A DesignResp1 of several items
// gives one argument per item, in its item order (it contributes them all; src/design/README.md). An Expression
// response checks that the arguments fill the header of its DesignEquation; a Callable response gives its callable one
// argument per argument kind of a deck DesignResp3 ("desvar", "dtable", "designResp1", "dnode", "dvprel1" ... "dresp2",
// each (n, c) with c the values of that kind) or one per named argument of a caller's declaration.
//
// By label (add_equation_response(by="label")), the response has one item per label (an entity and a subcase) of its
// first response argument; every response argument gives, for each label, its items of that label as the components
// of the argument (the same count for every label); a variable gives its value, or the value of its variable of each
// label (variableByLabel).
//
// The values of a subcase or a label form one item of a batch: a Callable is called through BatchCaller (blocks of
// EQUATION_BATCH_ITEMS in a fixed order, shape and finiteness checked); an Expression is evaluated at all the points
// at once by ExpressionEvaluator (exact derivatives for the sensitivities).
//
// Thread-safety: immutable once built; evaluate() calls the callables on the calling thread. Ownership: shares its
// immutable evaluator with its copies; borrows nothing past the constructor.
class EquationResponse
{
public:
    // The problems (an unknown DESVAR, DesignConstants label, relation, grid point, DesignEquation; a DNODE component
    // outside 1-3) are appended to `problems` as diagnostics of the card with its ID.
    EquationResponse(EquationResponseDefinition definition, const DesignModel& model,
                     const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& problems);

    [[nodiscard]] const EquationResponseDefinition& definition() const noexcept { return m_definition; }
    // The IDs of the responses it reads (DRESP1 and DesignResp2 arguments), in argument order.
    [[nodiscard]] std::vector<std::int32_t> responseArguments() const;

    // The values in `subcases` (card rules: {0} for a global response) or for the labels of its arguments (by label).
    // `known` holds every response it reads. Throws OptimisationError for arguments that do not fill the equation, a
    // value the equation refuses (outside its domain), a result of a callable of the wrong shape or not finite, a
    // missing callable.
    [[nodiscard]] ResponseValues evaluate(std::span<const std::int32_t> subcases, const KnownResponses& known,
                                          ResponseContext& context, const CallableRegistry& callables,
                                          std::size_t blockItems) const;

    // The values as evaluate() gives them, with the partial derivatives of every item: exact for an
    // Expression (ExpressionEvaluator's forward mode), the callable's gradient= for a Callable (one call per block,
    // with partials; a callable without gradient= is refused, naming the response). Throws as evaluate().
    [[nodiscard]] EquationPartials evaluateWithPartials(std::span<const std::int32_t> subcases,
                                                        const KnownResponses& known, ResponseContext& context,
                                                        const CallableRegistry& callables,
                                                        std::size_t blockItems) const;

    // The second-order terms of its items (an equation response of the second order): with the
    // weights y of its items (`itemWeights`, one per item in the order of its values) and the direction of every
    // argument value (`directionOf(source)`: δL of a response item, the move of a variable or a field), the callable's
    // hessian_vector= gives, per argument value of the items with a weight, w = Σ_o y_o·∇²f_o·(v_x, v_L); returned
    // with their sources (an entry per item and value, the zeros left out), in the order of the items and arguments.
    // The items without a weight are not passed to the callable (none called when no item has one). Throws
    // OptimisationError for an Expression or a callable without hessian_vector= (naming the response), a result of the
    // wrong shape or not finite, and as evaluate().
    [[nodiscard]] std::vector<std::pair<ArgumentSource, double>> secondOrderTerms(
        std::span<const std::int32_t> subcases, const KnownResponses& known, ResponseContext& context,
        const CallableRegistry& callables, std::size_t blockItems, std::span<const double> itemWeights,
        const std::function<double(const ArgumentSource&)>& directionOf) const;

    // The shape variables that move the coordinate of a DNODE argument (its position `argument`), as (shape basis,
    // ∂coordinate/∂s): the component of the basis vector at its grid point (DVGRID); empty for another
    // argument or a grid point no shape moves. The coordinate at a design point is the base one plus Σ s·∂c/∂s.
    [[nodiscard]] const std::vector<std::pair<std::size_t, double>>& coordinateShapes(std::size_t argument) const
    {
        return m_coordinateShapes[argument];
    }

private:
    // The batch arguments of the items and the source of every value: per slot, items × components sources.
    struct Gathered
    {
        std::vector<BatchArgument> arguments;
        std::vector<std::vector<ArgumentSource>> sources;
        std::vector<std::pair<std::int32_t, std::int32_t>> labels; // per item: (subcase, entity); entity 0 by card
    };

    [[nodiscard]] Gathered gather(std::span<const std::int32_t> subcases, const KnownResponses& known,
                                  ResponseContext& context) const;
    [[nodiscard]] Gathered gatherBySubcase(std::span<const std::int32_t> subcases, const KnownResponses& known,
                                           ResponseContext& context) const;
    [[nodiscard]] Gathered gatherByLabel(const KnownResponses& known, ResponseContext& context) const;
    [[nodiscard]] ArgumentSource scalarSource(const ResponseArgument& argument, std::int32_t labelEntity) const;
    [[nodiscard]] double scalarValue(const ArgumentSource& source, ResponseContext& context) const;
    [[nodiscard]] BatchResult compute(const Gathered& gathered, const CallableRegistry& callables,
                                      std::size_t blockItems, bool withPartials) const;
    [[nodiscard]] ResponseValues valuesOf(const Gathered& gathered, const BatchResult& result) const;
    [[nodiscard]] std::string labelOf(const Gathered& gathered, std::size_t item) const;
    // The status of each item: the worst status of the response items it reads (an item that reads a failed
    // or not evaluated item of an external response takes that status and is not computed).
    [[nodiscard]] static std::vector<ItemStatus> statusesOf(const Gathered& gathered, const KnownResponses& known);
    // The items `items` of a gathering, in their order.
    [[nodiscard]] static Gathered subset(const Gathered& gathered, std::span<const std::size_t> items);
    // The values of every item of `gathered` from the result of its ok items (`result`, of `okItems`); the others NaN
    // with their status.
    [[nodiscard]] ResponseValues mergedValues(const Gathered& gathered, std::span<const ItemStatus> statuses,
                                              std::span<const std::size_t> okItems, const BatchResult& result) const;
    [[nodiscard]] std::string name() const;

    EquationResponseDefinition m_definition;
    std::shared_ptr<const ExpressionEvaluator> m_evaluator; // an Expression (immutable, shared by copies)
    std::vector<std::size_t> m_variableIndices;             // per argument: the variable index of a DESVAR argument
    std::vector<double> m_constants;                        // per argument: the value of a DesignConstants argument
    std::vector<std::size_t> m_relationIndices;             // per argument: the relation index of a DVxRELi argument
    std::vector<double> m_coordinates;                      // per argument: the coordinate of a DNODE argument
    std::vector<std::vector<std::pair<std::size_t, double>>> m_coordinateShapes;     // per argument: coordinateShapes()
    std::vector<std::vector<std::pair<std::int32_t, std::size_t>>> m_labelVariables; // per argument: by-label indices
};

} // namespace oos::design
