#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace oos::assembly
{

// One pair of contact bodies glued in a subcase, with the glue parameters in force (BCONPRG, ContactTable; O_o_Reader's
// contactPairsTable).
struct GluedPair
{
    std::int64_t pairRow = 0;   // the row of contactPairsTable
    std::string table;          // "ContactTable 5", "ContactPair 7"
    std::string card;           // "ContactTable", "ContactPair"
    base::EntityId tableId = 0; // the ID of that card
    std::int32_t secondaryBody = 0;
    std::int32_t primaryBody = 0;
    std::int32_t glueOption = 0; // 1 to 4 or −1 to −4
    std::int32_t glueSeparation = 0;
    std::int32_t coordinateCheck = 0;
    std::int32_t searchOption = 0;
    std::optional<double> errorTolerance; // the position tolerance of the tie when given (GlueConstraints)
    std::optional<double> toleranceBias;
    std::string glueOptionSource; // the card that sets IGLUE, as O_o_Reader names it; may be empty
    std::int64_t nodePairCount = 0;

    // IGLUE 3 and 4 (and −3, −4) ask for a glue of the rotations of shells too, which the tie of GlueConstraints does
    // not make (translations only): reported.
    [[nodiscard]] bool asksForRotations() const noexcept
    {
        return std::abs(glueOption) == 3 || std::abs(glueOption) == 4;
    }
};

// Refuses a model whose contact the solver cannot read: when the snapshot holds pairs of contact bodies named by a
// contact table and the tables lack a column the glue needs (the face of each node pair and the side it was searched
// from, IGLUE, the BCONTACT of each subcase), throws ContractError naming the missing columns. A model without contact
// pairs, or whose pairs come from no table (the pairing of a Bulk Data without table, which only BCONTACT = ALLBODY
// would select), passes. Never guesses a glue from the columns of contract 38.
void requireGlueContract(const input::ModelSnapshot& snapshot);

// The pairs of contact bodies the BCONTACT of a subcase selects (BCONTACT, ContactTable, ContactTable1, ContactPair),
// in the order of contactPairsTable, those with an IGLUE of 1 to 4 (or −1 to −4) kept:
// - IGLUE 0 or blank (touching contact) is an error diagnostic "touching-contact" naming the table and the pair;
//   another IGLUE an error "invalid-glue-option" (IGLUE is 0 to 4 in magnitude; every nonzero value is the same tie
//   here);
// - JGLUE not 0 (a glue that may separate or break) is a warning "breakable-glue-permanent": the glue is permanent in
//   a linear analysis;
// - a BCONTACT that selects no pair is a warning "contact-selects-nothing".
// `contactId` nothing: the subcase selects no contact; nothing is returned. Calls requireGlueContract first (throws
// ContractError). `subcase` names the subcase in the messages.
[[nodiscard]] std::vector<GluedPair> selectGluedPairs(const input::ModelSnapshot& snapshot,
                                                      std::optional<base::EntityId> contactId, base::EntityId subcase,
                                                      base::DiagnosticSink& sink);

} // namespace oos::assembly
