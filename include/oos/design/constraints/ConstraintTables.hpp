#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/design/constraints/ConstraintDefinitions.hpp"

namespace oos::design
{

// The tables of O_o_Reader that give the constraints of a deck and the design commands of its Case Control,
// as O_o_Solver reads them; "proposed" marks what O_o_Solver asks for (src/design/README.md).
//
// designConstraintsTable, one row per DesignConstr: constraint_set int32 (DCID) · response_id int32 (RID) · lower
//   float64 (LALLOW, null when blank) · upper float64 (UALLOW, null when blank) · subcases list<int32> (optional: the
//   subcases the row is restricted to; empty: as DESSUB selects) · proposed: low_frequency, high_frequency float64
//   (LOWFQ, HIGHFQ; null when blank, refused otherwise on a static response).
// designConstraintSetsTable (proposed, DesignConstrUnion), one row per member: set_id int32 (DCID of the
// DesignConstrUnion) ·
//   member_set int32 (DCi).
// designSubcasesTable (proposed: DESSUB, DESGLB, DESOBJ effective per subcase; columns that could also
//   join subcasesTable), one row per subcase that has one of them, subcase 0 for the global section: subcase int32 ·
//   dessub int32 · desglb int32 · desobj int32 · objective_sense text ("min", "max"); nulls when absent.
// designScreeningTable (proposed, DesignConstrScreen): response_type text (RTYPE, or "Equation" for the equation
// responses) ·
//   threshold float64 (TRS) · max_retained int32 (NSTR).
[[nodiscard]] base::TableContract designConstraintsContract();
[[nodiscard]] base::TableContract designConstraintSetsContract();
[[nodiscard]] base::TableContract designSubcasesContract();
[[nodiscard]] base::TableContract designScreeningContract();

// The constraint parts of the tables (each optional). ContractError when a table breaks its contract;
// UnsupportedCardError for LOWFQ or HIGHFQ on a constraint (dynamic responses: outside the static analysis).
[[nodiscard]] ConstraintParts readConstraintTables(const base::ContractReader* constraints,
                                                   const base::ContractReader* sets,
                                                   const base::ContractReader* subcases,
                                                   const base::ContractReader* screening);

} // namespace oos::design
