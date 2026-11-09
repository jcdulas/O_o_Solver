#include "oos/design/expressions/ProgramTables.hpp"

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/expressions/ExpressionProgram.hpp"
#include "oos/design/expressions/IntrinsicFunction.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

// The names of one equation, slot by slot (empty until a row gives the slot).
struct EquationSymbols
{
    std::vector<std::string> arguments;
    std::vector<std::string> variables;
};

[[noreturn]] void refuseRow(std::string_view table, std::int64_t row, std::string_view expected, std::string_view found)
{
    throw base::ContractError(
        base::concatenate({"table '", table, "', row ", std::to_string(row), ": expected ", expected, ", found ", found,
                           "; the producer breaks the contract of the DesignEquation program tables "
                           "(update O_o_Reader, or report the table)"}));
}

void placeName(std::vector<std::string>& names, std::int32_t slot, std::string_view name, std::int64_t row)
{
    if (slot < 0)
        refuseRow("designEquationSymbols", row, "a slot >= 0", std::to_string(slot));
    const auto index = static_cast<std::size_t>(slot);
    if (index >= names.size())
        names.resize(index + 1);
    if (!names[index].empty())
        refuseRow("designEquationSymbols", row, "each slot once",
                  base::concatenate({"slot ", std::to_string(slot), " twice"}));
    names[index] = std::string(name);
}

std::map<std::int32_t, EquationSymbols> readSymbols(const base::ContractReader& symbols)
{
    const std::span<const std::int32_t> equationIds = symbols.values<std::int32_t>("equationId");
    const base::DictionaryView kinds = symbols.dictionaryTexts("kind");
    const std::span<const std::int32_t> slots = symbols.values<std::int32_t>("slot");
    const base::Utf8View names = symbols.texts("name");
    std::map<std::int32_t, EquationSymbols> equations;
    for (std::int64_t row = 0; row < symbols.rowCount(); ++row)
    {
        const auto at = static_cast<std::size_t>(row);
        EquationSymbols& equation = equations[equationIds[at]];
        const std::string_view kind = kinds[row];
        if (kind == "argument")
            placeName(equation.arguments, slots[at], names[row], row);
        else if (kind == "variable")
            placeName(equation.variables, slots[at], names[row], row);
        else
            refuseRow("designEquationSymbols", row, "kind 'argument' or 'variable'",
                      base::concatenate({"'", kind, "'"}));
    }
    for (const auto& [equationId, equation] : equations)
    {
        for (const std::vector<std::string>* list : {&equation.arguments, &equation.variables})
        {
            for (std::size_t slot = 0; slot < list->size(); ++slot)
            {
                if ((*list)[slot].empty())
                    throw base::ContractError(base::concatenate(
                        {"table 'designEquationSymbols': equation ", std::to_string(equationId),
                         " has no name for its ", list == &equation.arguments ? "argument" : "variable", " slot ",
                         std::to_string(slot),
                         "; expected every slot from 0 to the last one: take the "
                         "snapshot again with a O_o_Reader that follows it"}));
            }
        }
    }
    return equations;
}

} // namespace

base::TableContract designEquationSymbolsContract()
{
    return {"designEquationSymbols",
            {{"equationId", "int32"}, {"kind", "dictionary<utf8>"}, {"slot", "int32"}, {"name", "utf8"}}};
}

base::TableContract designEquationInstructionsContract()
{
    return {"designEquationInstructions",
            {{"equationId", "int32"},
             {"statement", "int32"},
             {"opcode", "dictionary<utf8>"},
             {"operand", "int32"},
             {"value", "float64"},
             {"function", "dictionary<utf8>"},
             {"position", "int32"}}};
}

std::vector<ExpressionProgram> readExpressionPrograms(const base::ContractReader& symbols,
                                                      const base::ContractReader& instructions)
{
    std::map<std::int32_t, EquationSymbols> equations = readSymbols(symbols);

    const std::span<const std::int32_t> equationIds = instructions.values<std::int32_t>("equationId");
    const std::span<const std::int32_t> statements = instructions.values<std::int32_t>("statement");
    const base::DictionaryView opcodes = instructions.dictionaryTexts("opcode");
    const std::span<const std::int32_t> operands = instructions.values<std::int32_t>("operand");
    const std::span<const double> values = instructions.values<double>("value");
    const base::DictionaryView functions = instructions.dictionaryTexts("function", base::NullPolicy::Allow);
    const base::ValidityView hasFunction = instructions.validity("function");
    const std::span<const std::int32_t> positions = instructions.values<std::int32_t>("position");

    std::map<std::int32_t, std::vector<Instruction>> programs;
    std::set<std::int32_t> closedEquations;
    std::optional<std::int32_t> currentEquation;
    for (std::int64_t row = 0; row < instructions.rowCount(); ++row)
    {
        const auto at = static_cast<std::size_t>(row);
        const std::int32_t equationId = equationIds[at];
        if (currentEquation != equationId)
        {
            if (currentEquation)
                closedEquations.insert(*currentEquation);
            if (closedEquations.contains(equationId))
                refuseRow("designEquationInstructions", row, "the rows of one equation contiguous",
                          base::concatenate({"equation ", std::to_string(equationId), " again after other rows"}));
            if (!equations.contains(equationId))
                refuseRow("designEquationInstructions", row, "an equation listed in designEquationSymbols",
                          base::concatenate({"equation ", std::to_string(equationId)}));
            currentEquation = equationId;
        }
        Instruction instruction;
        const std::optional<Opcode> opcode = findOpcode(opcodes[row]);
        if (!opcode)
            refuseRow("designEquationInstructions", row,
                      "an opcode of the contract (realConstant, integerConstant, ...)",
                      base::concatenate({"'", opcodes[row], "'"}));
        instruction.opcode = *opcode;
        instruction.statement = statements[at];
        instruction.operand = operands[at];
        instruction.value = values[at];
        instruction.position = positions[at];
        if (*opcode == Opcode::Call)
        {
            const std::optional<IntrinsicFunction> function =
                hasFunction.isValid(row) ? findIntrinsicFunction(functions[row]) : std::nullopt;
            if (!function)
                refuseRow("designEquationInstructions", row, "a DesignEquation intrinsic function for a call",
                          hasFunction.isValid(row) ? base::concatenate({"'", functions[row], "'"}) : "null");
            instruction.function = *function;
        }
        programs[equationId].push_back(instruction);
    }

    std::vector<ExpressionProgram> result;
    result.reserve(equations.size());
    for (auto& [equationId, equation] : equations)
    {
        result.emplace_back(equationId, std::move(equation.arguments), std::move(equation.variables),
                            std::move(programs[equationId]));
    }
    return result;
}

} // namespace oos::design
