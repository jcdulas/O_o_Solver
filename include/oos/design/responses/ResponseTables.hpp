#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"

namespace oos::design
{

// The tables of O_o_Reader that give the responses of a deck, as O_o_Solver reads them; the columns marked "proposed"
// are asked of O_o_Reader (src/design/README.md gives the full contract).
//
// designResponsesTable, one row per DesignResp1, DesignResp2, DesignResp3: id int32 · label text · response_type text
// (the RTYPE of
//   a DesignResp1; for a DesignResp2 or DesignResp3 the card name) · property_type text (PTYPE, null when blank) ·
//   region int32 (null when blank) · attribute_a int32 (ATTA, null when blank) · attribute_b float64 (ATTB, null when
//   blank) · targets list<int32> (ATTi; empty for blank or ALL) · is_objective bool · objective_sense text ("min",
//   "max") · proposed: response_card text ("DesignResp1", "DesignResp2", "DesignResp3"; when absent, DesignResp2 and
//   DesignResp3 are told by response_type) · equation_id int32 (the EQID of a DesignResp2; null for a DesignResp1, a
//   DesignResp3 or a FUNC) · function text (the FUNC of a DesignResp2: SUM, AVG, SSQ, RSS, MAX, MIN, BETA, MATCH; null
//   otherwise) · group text, user_type text (GROUP and TYPE of a DesignResp3).
// designResponseArgumentsTable (proposed), one row per argument of a DesignResp2 or DesignResp3, in the order of the
// card:
//   response_id int32 · kind text (DESVAR, DesignConstants, DesignResp1, DNODE, DesignPropRelation1,
//   DesignElemRelation1, DesignMatRelation1, DesignPropRelation2, DesignElemRelation2, DesignMatRelation2, DesignResp2)
//   · position int32 (from 0, the order of the arguments on the card) · id int32 (the DESVAR, DesignResp1, relation,
//   DesignResp2 ID or the grid point of a DNODE; null for a DesignConstants) · component int32 (the Cm of a DNODE, null
//   otherwise) · label utf8 (the DesignConstants label, null otherwise).
// A "text" column is "utf8" or "dictionary<utf8>".
[[nodiscard]] base::TableContract designResponsesContract();
[[nodiscard]] base::TableContract designResponseArgumentsContract();

// The responses of the tables (the arguments table optional: a deck without DesignResp2 or DesignResp3 has none).
// ContractError when a table breaks its contract: arguments of an unknown response or kind, positions that are not
// 0..n-1. UnsupportedCardError for the response types and options O_o_Solver does not support yet, named: an RTYPE that
// is not a static DesignResp1 type (EIGN, FREQ ... come with their analyses), a DesignResp2 with FUNC (the built-in
// functions of DesignResp2), the DesignLoadRelation1 arguments.
[[nodiscard]] ResponseParts readResponseTables(const base::ContractReader& responses,
                                               const base::ContractReader* arguments);

// The objective of the responses table (is_objective, objective_sense): its response ID and whether it is maximised.
struct TableObjective
{
    std::int32_t responseId = 0;
    bool isMaximised = false;
};
[[nodiscard]] std::optional<TableObjective> readTableObjective(const base::ContractReader& responses);

} // namespace oos::design
