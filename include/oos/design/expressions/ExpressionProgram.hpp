#pragma once

#include "oos/design/expressions/IntrinsicFunction.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oos::design
{

// The operations of an expression program (the contract of the DesignEquation program tables, written out in
// src/design/README.md): a postfix program run on a stack. Each one pops its operands and pushes its result.
enum class Opcode : std::uint8_t
{
    RealConstant,    // pushes `value` (a real constant of the text: "2.", "1.5E3")
    IntegerConstant, // pushes `value`, an integer constant of the text ("2"): integer arithmetic
    Argument,        // pushes the argument of slot `operand` (its position in the header list, from 0)
    Variable,        // pushes the value of the earlier statement `operand` (from 0)
    Negate,          // pops a, pushes -a (the unary minus; a unary plus produces no instruction)
    Add,             // pops b then a, pushes a + b
    Subtract,        // a - b
    Multiply,        // a · b
    Divide,          // a / b (truncated toward zero when both are integer)
    Power,           // a ** b
    Call             // pops `operand` arguments (the last pushed is the last argument), pushes function(arguments)
};

// The contract name of an opcode ("realConstant", "add", "call", ...), and back (nothing for an unknown name).
[[nodiscard]] std::string_view opcodeName(Opcode opcode) noexcept;
[[nodiscard]] std::optional<Opcode> findOpcode(std::string_view name) noexcept;

// One instruction of a program, as the contract row gives it.
struct Instruction
{
    Opcode opcode = Opcode::RealConstant;
    std::int32_t statement = 0; // the statement (from 0) whose expression this instruction belongs to
    std::int32_t operand = 0;   // Argument: slot; Variable: statement; Call: argument count; 0 otherwise
    double value = 0.0;         // RealConstant, IntegerConstant; 0 otherwise
    IntrinsicFunction function = IntrinsicFunction::Abs; // Call only
    std::int32_t position = 0; // the column (from 1) of the token in the equation text, 0 when not known
};

// One DesignEquation equation as a program; the entry's form is:
//   variable-1(x1, ..., xn) = expression-1; variable-2 = expression-2; ...; variable-m = expression-m
// The arguments are x1..xn in header order; the statements are variable-1..variable-m; the instructions of statement s
// come before those of statement s + 1 and leave exactly one value on the stack, the value of variable-s; a statement
// may read the arguments and the variables of the statements before it. The value of the equation is variable-m.
//
// Invariants, checked at construction (ContractError otherwise, naming the equation and the instruction: a program
// that breaks them is a broken contract of the producer, O_o_Reader): at least one statement, every statement has
// instructions, statements in order, slots in range, a Variable reads an earlier statement, a Call has an argument
// count within its function's signature, the stack never underflows and each statement leaves one value, constants
// are finite and an integer constant is an integer, names are unique.
//
// Thread-safety: immutable. Ownership: owns its data.
class ExpressionProgram
{
public:
    ExpressionProgram(std::int64_t equationId, std::vector<std::string> arguments, std::vector<std::string> variables,
                      std::vector<Instruction> instructions);

    // The EQID of the DesignEquation entry (0 for an equation compiled from a text given by the caller).
    [[nodiscard]] std::int64_t equationId() const noexcept { return m_equationId; }
    [[nodiscard]] const std::vector<std::string>& arguments() const noexcept { return m_arguments; }
    // The statement names variable-1..variable-m; the first one is the name of the equation (F1 in "F1(A,B)=...").
    [[nodiscard]] const std::vector<std::string>& variables() const noexcept { return m_variables; }
    [[nodiscard]] const std::vector<Instruction>& instructions() const noexcept { return m_instructions; }
    [[nodiscard]] const std::string& name() const noexcept { return m_variables.front(); }
    // The deepest the stack goes while the program runs.
    [[nodiscard]] std::size_t maxStackDepth() const noexcept { return m_maxStackDepth; }

    // "DesignEquation 12 (F1)": how messages name the equation.
    [[nodiscard]] std::string label() const;

private:
    void checkInstructions();

    std::int64_t m_equationId = 0;
    std::vector<std::string> m_arguments;
    std::vector<std::string> m_variables;
    std::vector<Instruction> m_instructions;
    std::size_t m_maxStackDepth = 0;
};

} // namespace oos::design
