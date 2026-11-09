#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/model/DesignLinks.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/model/DesignVariables.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::design
{

// One DesignNodeMove entry (DVID, GID, CID, COEFF, N1-N3), as O_o_Reader's shapeVariablesTable gives it: the
// vector N already taken along the directions of CID at the grid point and expressed in the basic system, not made a
// unit vector. CID is kept for the messages only.
struct ShapeTerm
{
    std::int32_t variableId = 0;       // DVID
    std::int32_t gridId = 0;           // GID
    std::int32_t frameId = 0;          // CID
    double coefficient = 0.0;          // COEFF (default 0.0)
    std::array<double, 3> direction{}; // N in the basic system (NaN when O_o_Reader could not resolve the vector)
};

// The move of one grid point per unit change of its design variable, in the basic system.
struct ShapeMove
{
    std::int32_t gridId = 0;
    std::array<double, 3> move{};
};

// The basis vector of one shaped design variable (the participation coefficients of the variable for each grid point
// coordinate it moves): its grid points, in increasing ID, each with Σ COEFF·N over the DesignNodeMove entries of that
// variable and grid point (several entries add as vectors).
struct ShapeBasis
{
    std::size_t variable = 0;    // the index of the design variable in DesignVariables
    std::int32_t variableId = 0; // its DVID
    std::vector<ShapeMove> moves;
};

// The shape variables of a design model (DVGRID: grid point positions along basis vectors).
//
// A grid point moves by g − g0 = Σ_j COEFF_j·(X_DVIDj − X⁰_DVIDj)·N_j, the sum over the DesignNodeMove
// entries that name it, X⁰ the initial value of the variable (XINIT of its DESVAR). So each shaped variable v gives one
// designed quantity, its shape value s_v = X_v − XINIT_v (the value of the "field" that moves its basis vector), and
// the position of a grid point at a design point is g = g0 + Σ_v s_v·T_v(g), T_v the basis vector of v. The derivative
// of s_v with respect to x is that of X_v: 1 for an independent variable, CMULT·Ci through a DesignVarLink.
//
// N is taken as written: the format determines N from CID and Ni, nothing more (N is not made a unit vector), and
// O_o_Reader's model.design.apply moves the grid points by COEFF·(X − XINIT)·N with N as written (src/design/README.md
// gives the rules), so that the overlay of a design point and the model `apply` writes are the same geometry.
//
// Checks (check()): every DVID names a DESVAR; N was resolved (finite: O_o_Reader gives NaN for a grid point or a
// system CID it does not know) and is not zero (at least one Ni ≠ 0.0); COEFF is finite. The grid points are
// bound to a snapshot by FieldTargets, which refuses the moves the overlay cannot make.
//
// Thread-safety: immutable once built; evaluate() may run concurrently. Ownership: owns its terms and bases.
class ShapeRelations
{
public:
    ShapeRelations() = default;
    // The bases are built from the terms whose DVID names a variable of `variables` (check() reports the others), in
    // the order of the variables.
    ShapeRelations(std::vector<ShapeTerm> terms, const DesignVariables& variables);

    [[nodiscard]] const std::vector<ShapeTerm>& terms() const noexcept { return m_terms; }
    [[nodiscard]] const std::vector<ShapeBasis>& bases() const noexcept { return m_bases; }
    // The number of shaped variables (bases).
    [[nodiscard]] std::size_t size() const noexcept { return m_bases.size(); }
    [[nodiscard]] bool isEmpty() const noexcept { return m_bases.empty(); }

    void check(const DesignVariables& variables, std::vector<base::Diagnostic>& diagnostics) const;

    // The shape value of every basis at a design point, s_v = X_v − XINIT_v, with its derivatives with respect to x
    // (those of X_v). Precondition: check() gave no error and `values` holds every variable of `variables`.
    [[nodiscard]] std::vector<FieldValue> evaluate(const VariableValues& values,
                                                   const DesignVariables& variables) const;

private:
    std::vector<ShapeTerm> m_terms;
    std::vector<ShapeBasis> m_bases;
};

} // namespace oos::design
