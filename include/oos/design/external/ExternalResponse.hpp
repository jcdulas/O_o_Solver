#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/external/ExternalEvaluator.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/responses/ResponseContext.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/ResponseValues.hpp"
#include "oos/design/responses/equation/EquationResponse.hpp"
#include "oos/output/sections/ZoneTable.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// One term of the partials of an external item: the derivative of the item with respect to component `component` of
// argument `argument` of its own case (a row of the zone data of that argument, or 0 for a design variable).
struct ExternalTerm
{
    std::size_t argument = 0;
    std::size_t component = 0;
    double weight = 0.0;
};

// The values of an external response at a design point, with the subcases evaluated per zone (the selection, for the
// report and the history) and, when asked, the partials of each item (in the order of the values).
struct ExternalValues
{
    ResponseValues values;
    ItemSelection evaluated;
    std::vector<std::vector<ExternalTerm>> partials;
};

// The arguments of an external response for the program of a two-phase coupling (problem.arguments(x, name): the
// arguments of the response per label, for a linear response on the load basis with α): per zone, the rows of each
// argument, on the cases or per constraint set on the basis vectors with α.
struct ExternalArgumentBlock
{
    std::int32_t zone = 0;
    ExternalRows kind = ExternalRows::Cases;
    std::int32_t constraintSet = 0;
    std::vector<std::int32_t> cases; // the selected cases (Cases), the cases of the set (Basis)
    std::size_t rows = 0;
    std::vector<BatchArgument> arguments;
    std::vector<double> coefficients; // Basis: α, rows (basis vectors) × cases, row-major
    // The labels of the rows of the zone data: (grid, component) per contour DOF or interior DOF.
    std::vector<std::pair<std::int32_t, std::int32_t>> contourDofs;
    std::vector<std::pair<std::int32_t, std::int32_t>> interiorDofs;
};

// An external response resolved against the design model and the zones (external local models given as plugins of
// the caller, a two-phase coupling, expensive outputs for selected cases only, statuses per item):
//
// - its labels are its zones; per zone and subcase it has one item per output of the plugin (entity the zone ID,
//   component the output number from 1), the zones in order, then the subcases, then the outputs;
// - select=: the evaluator gives the subcases of each zone to evaluate (the responses evaluated so far at hand), the
//   others are not_evaluated (NaN), as the subcases of a zone whose data the analysis did not give;
// - a callback that is not linear is called per zone and per batch of at most `blockItems` of its selected cases, with
//   the zone data of each case (rows: the cases; columns: the contour or interior DOFs of the zone, unpadded);
// - a linear callback is called per zone, constraint set and batch of basis vectors with their zone data; its outputs
//   are combined for the selected cases of the set (Y_c = Σ_j α(j, c)·Y_j, the basis vectors in order), then given to
//   post= per batch of cases when it has one; its partials are composed: ∂h/∂d = ∂post/∂Y·L (L its linear operator),
//   ∂h/∂t = ∂post/∂Y·Σ_j α(j, c)·∂Y_j/∂t for a variable;
// - supplied values (the two-phase coupling) are asked of the evaluator per zone, for the selected cases;
// - a non-finite value, or a status the evaluator gives, makes the item failed (NaN), the others kept.
// The partials of each item are with respect to the zone data of its own case (and its variables), which the
// sensitivities chain with the rows of the zone (output/sections/ZoneContour).
//
// Thread-safety: immutable once built; evaluate() calls the evaluator on the calling thread. Ownership: owns its
// resolution; borrows nothing past the constructor.
class ExternalResponse
{
public:
    // The problems (a zone not among `zones`, an unknown design variable, a variable per label missing a zone, no zone)
    // are appended to `problems` as diagnostics of the response.
    ExternalResponse(ExternalResponseDefinition definition, const DesignModel& model,
                     std::span<const output::ZoneDefinition> zones, std::vector<base::Diagnostic>& problems);

    [[nodiscard]] const ExternalResponseDefinition& definition() const noexcept { return m_definition; }
    // The design variable (index in the model) of argument a for zone `zone` (a DesignVariable argument).
    [[nodiscard]] std::size_t variableOf(std::size_t argument, std::int32_t zone) const;

    // The values in `subcases`, the zone data read from context.zones. Throws OptimisationError for a missing
    // evaluator, a zone the analysis did not give, results of the wrong size; propagates the evaluator's exceptions.
    [[nodiscard]] ExternalValues evaluate(std::span<const std::int32_t> subcases, const KnownResponses& known,
                                          ResponseContext& context, const ExternalRegistry& evaluators,
                                          std::size_t blockItems, bool withPartials) const;

    // The arguments of the selected cases of every zone (two-phase coupling), on the basis for a linear response.
    [[nodiscard]] std::vector<ExternalArgumentBlock> arguments(std::span<const std::int32_t> subcases,
                                                               const ItemSelection& selection,
                                                               ResponseContext& context) const;

private:
    [[nodiscard]] const output::ZoneTable& tableOf(std::int32_t zone, const ResponseContext& context) const;
    [[nodiscard]] std::string name() const;
    // The rows `rows` of the arguments of a zone: cases (indices into the table's cases) or basis vectors of a set.
    [[nodiscard]] std::vector<BatchArgument> gather(const output::ZoneTable& table, std::int32_t zone,
                                                    ExternalRows kind, std::size_t set,
                                                    std::span<const std::size_t> rows, ResponseContext& context) const;

    ExternalResponseDefinition m_definition;
    // Per argument: the variable index for every zone (DesignVariable), or per zone.
    std::vector<std::map<std::int32_t, std::size_t>> m_variables;
};

} // namespace oos::design
