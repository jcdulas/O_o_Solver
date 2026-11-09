#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/design/model/DesignModel.hpp"

#include <optional>

namespace oos::design
{

// The tables of O_o_Reader that give the design model of a deck, as O_o_Solver reads them. The columns marked
// "proposed" are not delivered by O_o_Reader yet: O_o_Solver asks for them (src/design/README.md gives the full
// contract).
//
// designVariablesTable: id int32 · label utf8 · value float64 (XINIT) · lower float64 · upper float64 · delta float64
//   (DELXV, null when blank) · ddval int32 (proposed, optional; null when blank)
// designLinksTable, one row per (DLINK, independent variable): dependent_variable int32 · constant float64 ·
//   multiplier float64 · independent_variable int32 · coefficient float64
// designRelationsTable, one row per (relation, argument): relation_card text · relation_id int32 · target_type text
//   ("property", "material", "element") · target_card text · target_id int32 · field text (the contract column) ·
//   design_variable int32 (null for a DesignConstants argument) · coefficient float64 (DVxREL1; null for DVxREL2) ·
//   constant float64 (C0; null when blank) · equation_id int32 (DVxREL2; null for DVxREL1) · field_item int32
//   (proposed, optional: the item of a fixed-list column, null for a scalar) · argument_index int32 (proposed,
//   optional: the position of the argument in the DesignEquation header, from 0) · constant_label utf8 (proposed,
//   optional: the DesignConstants label of a constant argument) · field_min float64, field_max float64 (proposed,
//   optional: PMIN, PMAX)
// designConstantsTable (proposed: DesignConstants): label utf8 · value float64
// A "text" column is "utf8" or "dictionary<utf8>".
[[nodiscard]] base::TableContract designVariablesContract();
[[nodiscard]] base::TableContract designLinksContract();
[[nodiscard]] base::TableContract designRelationsContract();
[[nodiscard]] base::TableContract designConstantsContract();
// shapeVariablesTable (O_o_Reader data contract 32), one row per DesignNodeMove card: designVariable int32 · nodeId
// int32 · frameId int32 · coefficient float64 · directionX, directionY, directionZ float64 (the vector N along the
// directions of the frame at the grid point, in the basic system, not made a unit vector; NaN when the grid point or
// the system does not exist).
[[nodiscard]] base::TableContract designShapesContract();

// The design model parts of the tables (each optional: a deck without DesignVarLink has no links table); `equations`
// are the DesignEquation programs (ProgramTables). The parts are checked when a DesignModel is made of them.
// ContractError when a table breaks its contract: an unknown target_type, a relation card other than DV[PMC]REL[12], a
// DVxREL2 row with neither a design variable nor a DesignConstants label, argument indices that are not 0..n-1, a
// DVxREL1 row without its design variable or coefficient, rows of one DesignVarLink with different constants or
// multipliers.
[[nodiscard]] DesignModelParts readDesignTables(const base::ContractReader* variables,
                                                const base::ContractReader* links,
                                                const base::ContractReader* relations,
                                                const base::ContractReader* constants,
                                                std::vector<ExpressionProgram> equations);

// The DesignNodeMove entries of shapeVariablesTable, in its row order (ShapeRelations checks them with the variables).
[[nodiscard]] std::vector<ShapeTerm> readShapeTable(const base::ContractReader& shapes);

} // namespace oos::design
