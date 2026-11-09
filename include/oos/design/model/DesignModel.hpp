#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/expressions/ExpressionProgram.hpp"
#include "oos/design/model/DesignLinks.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/model/DesignVariables.hpp"
#include "oos/design/model/ShapeRelations.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

// The pieces of a design model as they are read (DesignTables, from the design tables of O_o_Reader) or declared (by
// the caller): plain data, checked when a DesignModel is made of them.
struct DesignModelParts
{
    std::vector<DesignVariable> variables;
    std::vector<DesignLink> links;
    std::vector<DesignRelation> relations;
    std::vector<ExpressionProgram> equations;              // DesignEquation programs (design/expressions)
    std::vector<std::pair<std::string, double>> constants; // DesignConstants (label, value)
    std::vector<ShapeTerm> shapes;                         // DesignNodeMove
};

// The variable and field values of a design point.
struct DesignValues
{
    VariableValues variables;
    std::vector<FieldValue> fields; // one per relation, in order
    // One per shaped variable (ShapeRelations::bases() order): its shape value X − XINIT and d/dx.
    std::vector<FieldValue> shapes;
};

// A design model: the design variables (DESVAR), the links (DLINK), the relations to the property,
// material and element fields (DVPREL1/2, DesignMatRelation1/2, DesignElemRelation1/2), the DesignEquation equations
// and DesignConstants constants they use, and the shape variables (DVGRID: ShapeRelations). It maps a vector x of the
// independent variables to the value of every designed field and every shape value with their exact derivatives with
// respect to x (chain rule through the links and the equations). It knows nothing of the snapshot: FieldTargets binds
// its fields and its moved grid points to the snapshot's columns, DesignPoint applies them.
//
// Invariant: checked at construction; every reference resolves, so evaluate() needs no check.
//
// Thread-safety: immutable; evaluate() may run concurrently. Ownership: owns its parts.
class DesignModel
{
public:
    // Throws UnsupportedCardError when the model uses a card or field that is not supported (DDVAL), OptimisationError
    // for any other problem; the message lists every diagnostic of checkDesignModel().
    explicit DesignModel(DesignModelParts parts);

    [[nodiscard]] const DesignVariables& variables() const noexcept { return m_variables; }
    [[nodiscard]] const DesignLinks& links() const noexcept { return m_links; }
    [[nodiscard]] const DesignRelations& relations() const noexcept { return m_relations; }
    [[nodiscard]] const ShapeRelations& shapes() const noexcept { return m_shapes; }
    [[nodiscard]] const std::vector<ExpressionProgram>& equations() const noexcept { return m_equations; }
    [[nodiscard]] const std::vector<std::pair<std::string, double>>& constants() const noexcept { return m_constants; }

    // The number of independent variables: the size of x.
    [[nodiscard]] std::size_t independentCount() const noexcept { return m_links.independentIndices().size(); }
    // The variable of each column of x (indices into variables()).
    [[nodiscard]] const std::vector<std::size_t>& independentIndices() const noexcept
    {
        return m_links.independentIndices();
    }
    // The starting point: XINIT of each independent variable.
    [[nodiscard]] std::vector<double> initialValues() const;

    // The values at x. std::invalid_argument for an x of the wrong size or a missing supplied value; OptimisationError
    // when an equation leaves its domain.
    [[nodiscard]] DesignValues evaluate(std::span<const double> x, std::span<const SuppliedField> supplied = {}) const;

private:
    DesignVariables m_variables;
    DesignLinks m_links;
    DesignRelations m_relations;
    ShapeRelations m_shapes;
    std::vector<ExpressionProgram> m_equations;
    std::vector<std::pair<std::string, double>> m_constants;
};

// Every problem of a set of design model parts (none for a valid model), as diagnostics naming the card, its ID and the
// field.
[[nodiscard]] std::vector<base::Diagnostic> checkDesignModel(const DesignModelParts& parts);

// Throws the error of a list of diagnostics (UnsupportedCardError when one is an unsupported card or target,
// OptimisationError otherwise), its message listing them all after `what`. Does nothing for an empty list.
void throwDesignDiagnostics(const std::string& what, const std::vector<base::Diagnostic>& diagnostics);

} // namespace oos::design
