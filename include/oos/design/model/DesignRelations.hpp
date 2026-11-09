#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/expressions/ExpressionEvaluator.hpp"
#include "oos/design/expressions/ExpressionProgram.hpp"
#include "oos/design/model/DesignLinks.hpp"
#include "oos/design/model/DesignVariables.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::design
{

// What a relation designs: a property field (DVPREL1, DesignPropRelation2), a material field (DVMREL1,
// DesignMatRelation2), or a connection or element field (DVCREL1, DesignElemRelation2).
enum class TargetKind
{
    Property,
    Material,
    Element
};

// How a relation gives its field value.
enum class RelationForm
{
    Linear,   // DVxREL1: field = C0 + Σ COEFi · DESVARi
    Equation, // DVxREL2: field = DEQATN(arguments), the DESVARs then the DesignConstants constants
    Supplied  // add_relation(function=, gradient=): the value and gradient are supplied per design point by the caller
};

// "property", "material", "element" (the target_type of designRelationsTable); nothing for another text.
[[nodiscard]] std::string_view targetKindName(TargetKind kind) noexcept;
[[nodiscard]] std::optional<TargetKind> findTargetKind(std::string_view name) noexcept;

// One term of a linear relation.
struct RelationTerm
{
    std::int32_t variableId = 0; // DVIDi
    double coefficient = 0.0;    // COEFi
};

// One argument of an equation relation, in the order of the DesignEquation header: a design variable, or a
// DesignConstants constant named by its label (DVPREL2: the DESVAR values, then the DesignConstants values).
struct RelationArgument
{
    std::int32_t variableId = 0; // when `constantLabel` is empty
    std::string constantLabel;   // DesignConstants LABLi

    [[nodiscard]] bool isConstant() const noexcept { return !constantLabel.empty(); }
};

// One relation between design variables and one designed field. The field is named as the column of the data
// contract that holds it ("t" of ShellProp, "e" of IsoMat, "offset" of ShellQuad4); `fieldItem` is the item of a
// fixed-list column ("stiffness" of BushProp: K1 is item 0), -1 for a scalar column.
struct DesignRelation
{
    std::string card; // DesignPropRelation1, DesignPropRelation2, DesignMatRelation1, DesignMatRelation2,
                      // DesignElemRelation1, DesignElemRelation2 (the caller's declarations name theirs)
    std::int32_t id = 0;
    TargetKind targetKind = TargetKind::Property;
    std::string targetCard; // TYPE: ShellProp, IsoMat, ShellQuad4
    std::int32_t targetId = 0;
    std::string field;
    std::int32_t fieldItem = -1;
    RelationForm form = RelationForm::Linear;
    double constant = 0.0;                   // C0 (Linear)
    std::vector<RelationTerm> terms;         // Linear
    std::int32_t equationId = 0;             // EQID (Equation)
    std::vector<RelationArgument> arguments; // Equation; Supplied: its variables (no constants)
    std::optional<double> lowerBound;        // PMIN, MPMIN, CPMIN: kept for the optimiser
    std::optional<double> upperBound;        // PMAX, MPMAX, CPMAX

    // "DesignPropRelation1 12 (PSHELL 3 t)", as messages name the relation.
    [[nodiscard]] std::string label() const;
};

// The value a caller supplies for a relation of the Supplied form at one design point: the value and its derivatives
// with respect to the relation's arguments, in their order.
struct SuppliedField
{
    std::size_t relationIndex = 0;
    double value = 0.0;
    std::vector<double> gradient;
};

// The value of one designed field at a design point and its non-zero derivatives with respect to x (the independent
// variables), as (column, derivative) in increasing column.
struct FieldValue
{
    double value = 0.0;
    std::vector<std::pair<std::size_t, double>> gradient;
};

// The relations of a design model, and their evaluation with derivatives.
//
// Values: Linear, field = C0 + Σ COEFi·x_i (DVPREL1, DesignMatRelation1, DesignElemRelation1); Equation, field =
// f(arguments) with f the DesignEquation program evaluated with exact forward-mode derivatives (ExpressionEvaluator),
// the DesignConstants constants bound by position (DVPREL2: the DESVAR IDs then the DesignConstants labels, matched to
// the DesignEquation arguments in order). Derivatives with respect to the independent variables by the chain rule
// through the links:
//     d(field)/d(x_k) = Σ_v ∂(field)/∂(v) · d(v)/d(x_k),
// with ∂(field)/∂(v) = COEF (Linear), the evaluator's exact partials (Equation), or the supplied gradient, and
// d(v)/d(x_k) from DesignLinks::expand (1 for v = x_k, CMULT·Ci for a dependent variable).
//
// Checks (check()): relation IDs unique per card; every DESVAR, DesignEquation and DesignConstants named exists; an
// equation has as many arguments as its header; a linear relation has at least one term; a field driven by two
// relations is refused (a field driven by several variables is one relation with several terms), the field identified
// by its target kind, card, ID, column and item.
//
// Thread-safety: immutable once built; evaluate() may run concurrently. Ownership: owns its relations and the
// evaluators of its equations.
class DesignRelations
{
public:
    DesignRelations() = default;
    // `equations`: the DesignEquation programs the relations may name; `constants`: the DesignConstants constants
    // (label, value). The evaluators of the equation relations are built here when their references resolve (check()
    // reports the others).
    DesignRelations(std::vector<DesignRelation> relations, const std::vector<ExpressionProgram>& equations,
                    const std::vector<std::pair<std::string, double>>& constants);

    [[nodiscard]] std::size_t size() const noexcept { return m_relations.size(); }
    [[nodiscard]] const std::vector<DesignRelation>& all() const noexcept { return m_relations; }
    [[nodiscard]] const DesignRelation& operator[](std::size_t index) const { return m_relations[index]; }

    void check(const DesignVariables& variables, const std::vector<ExpressionProgram>& equations,
               const std::vector<std::pair<std::string, double>>& constants,
               std::vector<base::Diagnostic>& diagnostics) const;

    // The field value of every relation, in order, with its derivatives with respect to x. `supplied` gives the
    // values of the Supplied relations. Precondition: check() gave no error. OptimisationError when an equation
    // cannot be evaluated (a point outside a function's domain: the message names the relation and the equation),
    // std::invalid_argument when a Supplied relation has no supplied value or a gradient of the wrong size.
    [[nodiscard]] std::vector<FieldValue> evaluate(const VariableValues& values, const DesignVariables& variables,
                                                   std::span<const SuppliedField> supplied) const;

private:
    // The evaluator of an Equation relation and, for each of its free arguments, the variable index it reads.
    struct EquationBinding
    {
        std::shared_ptr<const ExpressionEvaluator> evaluator;
        std::vector<std::int32_t> freeVariableIds;
    };

    std::vector<DesignRelation> m_relations;
    std::vector<EquationBinding> m_bindings; // one per relation (empty for the other forms)
};

} // namespace oos::design
