#include "oos/design/constraints/ConstraintTables.hpp"

#include "../common/TableColumns.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"

#include <map>
#include <optional>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

using namespace table_columns;

} // namespace

base::TableContract designConstraintsContract()
{
    return {"designConstraintsTable",
            {required("constraintSet", INT32), required("responseId", INT32), required("lower", FLOAT64),
             required("upper", FLOAT64), optional("subcases", INT32_LIST), optional("lowFrequency", FLOAT64),
             optional("highFrequency", FLOAT64)}};
}

base::TableContract designConstraintSetsContract()
{
    return {"designConstraintSetsTable", {required("setId", INT32), required("memberSet", INT32)}};
}

base::TableContract designSubcasesContract()
{
    return {"designSubcasesTable",
            {required("subcase", INT32), optional("subcaseDesignConstrSelection", INT32),
             optional("globalDesignConstrSelection", INT32), optional("designObjectiveSelection", INT32),
             optionalText("objectiveSense")}};
}

base::TableContract designScreeningContract()
{
    return {"designScreeningTable",
            {text("responseType"), required("threshold", FLOAT64), required("maxRetained", INT32)}};
}

ConstraintParts readConstraintTables(const base::ContractReader* constraints, const base::ContractReader* sets,
                                     const base::ContractReader* subcases, const base::ContractReader* screening)
{
    ConstraintParts parts;
    if (constraints != nullptr)
    {
        const std::span<const std::int32_t> setIds = constraints->values<std::int32_t>("constraintSet");
        const std::span<const std::int32_t> responses = constraints->values<std::int32_t>("responseId");
        const NullableColumn<double> lower(*constraints, "lower");
        const NullableColumn<double> upper(*constraints, "upper");
        const IntListColumn restricted(*constraints, "subcases");
        const NullableColumn<double> lowFrequency(*constraints, "lowFrequency");
        const NullableColumn<double> highFrequency(*constraints, "highFrequency");
        for (std::int64_t row = 0; row < constraints->rowCount(); ++row)
        {
            const auto r = static_cast<std::size_t>(row);
            if (lowFrequency[row] || highFrequency[row])
                throw base::UnsupportedCardError(base::concatenate(
                    {"DesignConstr ", std::to_string(setIds[r]), " on response ", std::to_string(responses[r]),
                     ": DesignConstr.lowFrequency and DesignConstr.highFrequency select frequencies of "
                     "dynamic responses, outside the static analysis; "
                     "leave them blank"}));
            parts.constraints.push_back({setIds[r], responses[r], lower[row], upper[row], restricted[row]});
        }
    }
    if (sets != nullptr)
    {
        const std::span<const std::int32_t> setIds = sets->values<std::int32_t>("setId");
        const std::span<const std::int32_t> members = sets->values<std::int32_t>("memberSet");
        std::map<std::int32_t, std::size_t> unionOf;
        for (std::int64_t row = 0; row < sets->rowCount(); ++row)
        {
            const auto r = static_cast<std::size_t>(row);
            const auto [found, isNew] = unionOf.emplace(setIds[r], parts.unions.size());
            if (isNew)
                parts.unions.push_back({setIds[r], {}});
            parts.unions[found->second].members.push_back(members[r]);
        }
    }
    if (subcases != nullptr)
    {
        const std::span<const std::int32_t> ids = subcases->values<std::int32_t>("subcase");
        const NullableColumn<std::int32_t> dessub(*subcases, "subcaseDesignConstrSelection");
        const NullableColumn<std::int32_t> desglb(*subcases, "globalDesignConstrSelection");
        const NullableColumn<std::int32_t> desobj(*subcases, "designObjectiveSelection");
        const TextColumn senses(*subcases, "objectiveSense");
        // O_o_Reader gives each subcase row the value in force there, the one it inherits from the global section
        // included (its designSubcasesTable, data contract 32): a subcase row that repeats the DESGLB or the DESOBJ
        // of the global row (subcase 0) inherits it and declares nothing of its own.
        std::optional<std::int32_t> globalDesglb;
        std::optional<std::int32_t> globalDesobj;
        std::string globalSense;
        for (std::int64_t row = 0; row < subcases->rowCount(); ++row)
        {
            if (ids[static_cast<std::size_t>(row)] != 0)
                continue;
            if (desglb[row])
                globalDesglb = desglb[row];
            if (desobj[row])
            {
                globalDesobj = desobj[row];
                globalSense = senses[row];
            }
        }
        for (std::int64_t row = 0; row < subcases->rowCount(); ++row)
        {
            const std::int32_t subcase = ids[static_cast<std::size_t>(row)];
            if (dessub[row])
                parts.selection.subcases.push_back({subcase, dessub[row]});
            if (desglb[row] && (subcase == 0 || desglb[row] != globalDesglb))
            {
                if (subcase != 0)
                    refuseRow(*subcases, row,
                              "expected GlobalDesignConstrSelection in the global section (subcase 0), found "
                              "it in subcase " +
                                  std::to_string(subcase));
                parts.selection.desglb = desglb[row];
            }
            const bool isInheritedObjective = subcase != 0 && desobj[row] == globalDesobj && senses[row] == globalSense;
            if (desobj[row] && !isInheritedObjective)
            {
                parts.selection.objective = desobj[row];
                parts.selection.isMaximised = senses[row] == "max";
                if (subcase != 0)
                    parts.selection.objectiveSubcase = subcase;
            }
        }
    }
    if (screening != nullptr)
    {
        const TextColumn types(*screening, "responseType");
        const std::span<const double> thresholds = screening->values<double>("threshold");
        const std::span<const std::int32_t> counts = screening->values<std::int32_t>("maxRetained");
        for (std::int64_t row = 0; row < screening->rowCount(); ++row)
            parts.screening.push_back(
                {types[row], thresholds[static_cast<std::size_t>(row)], counts[static_cast<std::size_t>(row)]});
    }
    return parts;
}

} // namespace oos::design
