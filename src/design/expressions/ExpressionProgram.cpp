#include "oos/design/expressions/ExpressionProgram.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

constexpr std::array<std::string_view, 11> OPCODE_NAMES{
    "realConstant", "integerConstant", "argument", "variable", "negate", "add",
    "subtract",     "multiply",        "divide",   "power",    "call"};

// The integers a double holds exactly: an integer constant beyond them would not be one (integer arithmetic).
constexpr double LARGEST_EXACT_INTEGER = 9007199254740992.0; // 2^53

// How many values an instruction pops.
int popCount(const Instruction& instruction)
{
    switch (instruction.opcode)
    {
    case Opcode::RealConstant:
    case Opcode::IntegerConstant:
    case Opcode::Argument:
    case Opcode::Variable:
        return 0;
    case Opcode::Negate:
        return 1;
    case Opcode::Call:
        return instruction.operand;
    default:
        return 2;
    }
}

} // namespace

std::string_view opcodeName(Opcode opcode) noexcept
{
    return OPCODE_NAMES[static_cast<std::size_t>(opcode)];
}

std::optional<Opcode> findOpcode(std::string_view name) noexcept
{
    for (std::size_t index = 0; index < OPCODE_NAMES.size(); ++index)
    {
        if (OPCODE_NAMES[index] == name)
            return static_cast<Opcode>(index);
    }
    return std::nullopt;
}

ExpressionProgram::ExpressionProgram(std::int64_t equationId, std::vector<std::string> arguments,
                                     std::vector<std::string> variables, std::vector<Instruction> instructions)
    : m_equationId(equationId),
      m_arguments(std::move(arguments)),
      m_variables(std::move(variables)),
      m_instructions(std::move(instructions))
{
    const std::string where =
        base::concatenate({"the expression program of DesignEquation ", std::to_string(m_equationId)});
    if (m_variables.empty())
        throw base::ContractError(base::concatenate(
            {where,
             ": expected at least one statement (variable-1 = expression-1), found none; the producer of the "
             "designEquationSymbols table must list the statement names (kind 'variable'): take the snapshot again "
             "with a "
             "O_o_Reader that follows the contract of the DesignEquation program tables"}));
    std::set<std::string, std::less<>> names;
    for (const std::vector<std::string>* list : {&m_arguments, &m_variables})
    {
        for (const std::string& name : *list)
        {
            if (name.empty() || !names.insert(name).second)
                throw base::ContractError(
                    base::concatenate({where, ": expected unique non-empty argument and statement names, found '", name,
                                       "' empty or twice; check the designEquationSymbols rows of this equation"}));
        }
    }
    checkInstructions();
}

void ExpressionProgram::checkInstructions()
{
    const auto statementCount = static_cast<std::int32_t>(m_variables.size());
    const auto argumentCount = static_cast<std::int32_t>(m_arguments.size());
    std::int32_t statement = 0;
    std::size_t depth = 0;
    bool isStatementOpen = false;
    for (std::size_t index = 0; index < m_instructions.size(); ++index)
    {
        const Instruction& instruction = m_instructions[index];
        const auto fail = [&](std::string_view expected, const std::string& found)
        {
            throw base::ContractError(base::concatenate(
                {"the expression program of DesignEquation ", std::to_string(m_equationId), ", instruction ",
                 std::to_string(index), " (", opcodeName(instruction.opcode), ", statement ",
                 std::to_string(instruction.statement), "): expected ", expected, ", found ", found,
                 "; the designEquationInstructions table breaks the contract of the DesignEquation program tables: "
                 "take the "
                 "snapshot again with a O_o_Reader that follows it"}));
        };
        if (instruction.statement != statement)
        {
            if (instruction.statement != statement + 1 || !isStatementOpen)
                fail("the statements in order, each with instructions",
                     base::concatenate({"statement ", std::to_string(instruction.statement), " after statement ",
                                        std::to_string(statement)}));
            if (depth != 1)
                fail("one value left by the previous statement", base::concatenate({std::to_string(depth), " values"}));
            depth = 0;
            statement = instruction.statement;
        }
        if (statement >= statementCount)
            fail(base::concatenate({"a statement below ", std::to_string(statementCount)}),
                 std::to_string(instruction.statement));
        isStatementOpen = true;
        switch (instruction.opcode)
        {
        case Opcode::RealConstant:
            if (!std::isfinite(instruction.value))
                fail("a finite constant", std::to_string(instruction.value));
            break;
        case Opcode::IntegerConstant:
            if (!(std::fabs(instruction.value) <= LARGEST_EXACT_INTEGER) ||
                instruction.value != std::trunc(instruction.value))
                fail("an integer of at most 2^53 in magnitude", std::to_string(instruction.value));
            break;
        case Opcode::Argument:
            if (instruction.operand < 0 || instruction.operand >= argumentCount)
                fail(base::concatenate({"an argument slot below ", std::to_string(argumentCount)}),
                     std::to_string(instruction.operand));
            break;
        case Opcode::Variable:
            if (instruction.operand < 0 || instruction.operand >= statement)
                fail(base::concatenate({"an earlier statement (below ", std::to_string(statement), ")"}),
                     std::to_string(instruction.operand));
            break;
        case Opcode::Call:
        {
            const IntrinsicSignature& signature = signatureOf(instruction.function);
            if (instruction.operand < signature.minArguments || instruction.operand > signature.maxArguments)
                fail(base::concatenate({std::to_string(signature.minArguments), " to ",
                                        std::to_string(signature.maxArguments), " arguments for ", signature.name}),
                     std::to_string(instruction.operand));
            break;
        }
        default:
            break;
        }
        const auto pops = static_cast<std::size_t>(popCount(instruction));
        if (pops > depth)
            fail(base::concatenate({std::to_string(pops), " values on the stack"}), std::to_string(depth));
        depth = depth - pops + 1;
        m_maxStackDepth = std::max(m_maxStackDepth, depth);
    }
    if (!isStatementOpen || statement != statementCount - 1 || depth != 1)
        throw base::ContractError(base::concatenate(
            {"the expression program of DesignEquation ", std::to_string(m_equationId),
             ": expected instructions for its ", std::to_string(statementCount),
             " statements, the last leaving one value; found ", std::to_string(m_instructions.size()),
             " instructions ending in statement ", std::to_string(statement), " with ", std::to_string(depth),
             " values; the designEquationInstructions table breaks the contract of the DesignEquation program tables: "
             "take the "
             "snapshot again with a O_o_Reader that follows it"}));
}

std::string ExpressionProgram::label() const
{
    std::string text = "DesignEquation ";
    text.append(std::to_string(m_equationId)).append(" (").append(name()).append(")");
    return text;
}

} // namespace oos::design
