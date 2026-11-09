#include "oos/assembly/constraints/GluedContactPairs.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace oos::assembly
{

namespace
{

// ContactTable, ContactPairGeomParams: IGLUE is 0 to 4 in magnitude.
constexpr std::int32_t LARGEST_GLUE_OPTION = 4;
// The card whose keyword parameters name the glue options in messages, and the neutral names of those parameters
// (IGLUE and JGLUE in a deck), as the dictionary has them.
constexpr std::string_view CONTACT_TABLE_CARD = "ContactTable";
constexpr std::string_view GLUE_OPTION = "GlueOption";
constexpr std::string_view GLUE_SEPARATION = "GlueSeparationOption";

base::Diagnostic pairProblem(base::Severity severity, std::string code, const input::ContactTableName& name,
                             std::string field, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = name.card;
    diagnostic.id = name.id;
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

std::string pairText(const input::ContactTable& contact, std::int64_t row)
{
    return base::concatenate({"the pair of bodies ", std::to_string(contact.pairSecondaryBody(row)),
                              " (secondary) and ", std::to_string(contact.pairPrimaryBody(row)), " (primary) of ",
                              contact.pairTable(row)});
}

} // namespace

void requireGlueContract(const input::ModelSnapshot& snapshot)
{
    if (!snapshot.hasTable(input::TableRole::ContactPairs))
        return;
    const input::ContactTable& contact = snapshot.contact();
    std::string tables;
    for (std::int64_t row = 0; row < contact.pairCount(); ++row)
    {
        const std::string_view table = contact.pairTable(row);
        if (!table.empty() && tables.find(table) == std::string::npos)
            tables.append(tables.empty() ? "" : ", ").append(table);
    }
    if (tables.empty())
        return;
    std::vector<std::string> missing = contact.missingGlueColumns();
    if (snapshot.hasTable(input::TableRole::Subcases) && !snapshot.subcases().hasContactColumn())
        missing.emplace_back("subcasesTable.bcontact");
    if (missing.empty())
        return;
    std::string columns;
    for (const std::string& column : missing)
        columns.append(columns.empty() ? "" : ", ").append(column);
    throw base::ContractError(base::concatenate(
        {"contact: the model pairs contact bodies by ", tables,
         "; O_o_Solver ties them by its own projection of the secondary grid points on the primary faces, reading the "
         "two surfaces from O_o_Reader's contactNodePairs, and expected the glued contact columns, found none of: ",
         columns, "; install a O_o_Reader that gives them, or remove the contact tables to solve without contact"}));
}

std::vector<GluedPair> selectGluedPairs(const input::ModelSnapshot& snapshot, std::optional<base::EntityId> contactId,
                                        base::EntityId subcase, base::DiagnosticSink& sink)
{
    requireGlueContract(snapshot);
    std::vector<GluedPair> pairs; // bounded by the pairs of one contact table
    if (!contactId)
        return pairs;
    const std::string selection =
        base::concatenate({"ContactSelection = ", std::to_string(*contactId), " of subcase ", std::to_string(subcase)});
    std::vector<std::int64_t> rows;
    if (snapshot.hasTable(input::TableRole::ContactPairs))
        rows = snapshot.contact().selectedPairRows(*contactId);
    if (rows.empty())
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "contact-selects-nothing";
        diagnostic.card = "ContactSelection";
        diagnostic.id = *contactId;
        diagnostic.message = base::concatenate(
            {selection,
             " selects no pair of contact bodies (no contact table of that ID pairs two deformable bodies "
             "whose surfaces meet): the subcase is solved without contact; check the ID or the tables"});
        sink.report(std::move(diagnostic));
        return pairs;
    }
    const input::ContactTable& contact = snapshot.contact();
    // A message names the glue field by its keyword; the field of a contact table stands for that of every card that
    // sets it (the pairs of a contact table, its pair parameters).

    const std::string glueOptionLabel = std::string(GLUE_OPTION);
    const std::string glueSeparationLabel = std::string(GLUE_SEPARATION);
    for (const std::int64_t row : rows)
    {
        const std::string_view table = contact.pairTable(row);
        const input::ContactTableName name =
            input::parseContactTableName(table).value_or(input::ContactTableName{std::string(table), 0});
        const input::GlueParameters parameters = contact.glueParameters(row);
        const std::int32_t glueOption = parameters.glueOption.value_or(0);
        const std::string setBy = parameters.glueOptionSource.empty()
                                      ? std::string()
                                      : base::concatenate({", set by ", parameters.glueOptionSource});
        if (glueOption == 0)
        {
            sink.report(pairProblem(
                base::Severity::Error, "touching-contact", name, glueOptionLabel,
                base::concatenate({pairText(contact, row), ", selected by ", selection, ", is touching contact (",
                                   glueOptionLabel, " 0 or blank", setBy, "); O_o_Solver solves glued contact only (",
                                   glueOptionLabel, " 1 to 4): set ", glueOptionLabel, " on the ",
                                   std::string("ContactPairGeomParams"), " or the ", std::string(CONTACT_TABLE_CARD),
                                   " section of the pair, or take the pair out of the table"})));
            continue;
        }
        if (glueOption < -LARGEST_GLUE_OPTION || glueOption > LARGEST_GLUE_OPTION)
        {
            sink.report(pairProblem(base::Severity::Error, "invalid-glue-option", name, glueOptionLabel,
                                    base::concatenate({pairText(contact, row), ": expected ", glueOptionLabel,
                                                       " of 1 to 4 (or -1 to -4), found ", std::to_string(glueOption),
                                                       setBy, "; correct the field"})));
            continue;
        }
        GluedPair pair;
        pair.pairRow = row;
        pair.table = std::string(table);
        pair.card = name.card;
        pair.tableId = name.id;
        pair.secondaryBody = contact.pairSecondaryBody(row);
        pair.primaryBody = contact.pairPrimaryBody(row);
        pair.glueOption = glueOption;
        pair.glueSeparation = parameters.glueSeparation.value_or(0);
        pair.coordinateCheck = parameters.coordinateCheck.value_or(0);
        pair.searchOption = parameters.searchOption.value_or(0);
        pair.errorTolerance = parameters.errorTolerance;
        pair.toleranceBias = parameters.toleranceBias;
        pair.glueOptionSource = parameters.glueOptionSource;
        pair.nodePairCount = contact.pairNodeCount(row);
        if (pair.glueSeparation != 0)
            sink.report(pairProblem(
                base::Severity::Warning, "breakable-glue-permanent", name, glueSeparationLabel,
                base::concatenate({pairText(contact, row), " has ", glueSeparationLabel, " = ",
                                   std::to_string(pair.glueSeparation),
                                   " (a glue that may separate or break); a linear analysis keeps it glued for "
                                   "good"})));
        pairs.push_back(std::move(pair));
    }
    return pairs;
}

} // namespace oos::assembly
