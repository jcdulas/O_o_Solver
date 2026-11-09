#include "oos/design/expressions/ExpressionEvaluator.hpp"

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/expressions/ExpressionProgram.hpp"
#include "oos/design/expressions/IntrinsicFunction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

constexpr double QUIET_NAN = std::numeric_limits<double>::quiet_NaN();
// The integers a double holds exactly: integer arithmetic (between integer constants) beyond them is refused, not
// rounded.
constexpr double LARGEST_EXACT_INTEGER = 9007199254740992.0; // 2^53
// The operands kept in a refusal message.
constexpr std::size_t REPORTED_OPERANDS = 3;

// The value of an operator at one point, with its partials with respect to its two operands.
struct OperatorValue
{
    double value = 0.0;
    double partialA = 0.0;
    double partialB = 0.0;
    DomainFault fault = DomainFault::None;
};

OperatorValue refused(DomainFault fault)
{
    return {QUIET_NAN, QUIET_NAN, QUIET_NAN, fault};
}

OperatorValue checkedInteger(double value)
{
    if (!(std::fabs(value) <= LARGEST_EXACT_INTEGER))
        return refused(DomainFault::IntegerOverflow);
    return {value, 0.0, 0.0, DomainFault::None};
}

// The integer arithmetic of DesignEquation, between two integer constants (exact integers in doubles). Fortran
// rules: the quotient is truncated toward zero; I**J with J < 0 is 1/(I**|J|) in integers, so 0 unless |I| = 1.
OperatorValue integerOperator(Opcode opcode, double a, double b)
{
    switch (opcode)
    {
    case Opcode::Add:
        return checkedInteger(a + b);
    case Opcode::Subtract:
        return checkedInteger(a - b);
    case Opcode::Multiply:
        return checkedInteger(a * b);
    case Opcode::Divide:
        if (b == 0.0)
            return refused(DomainFault::DivisionByZero);
        return checkedInteger(static_cast<double>(static_cast<std::int64_t>(a) / static_cast<std::int64_t>(b)));
    default: // Power
    {
        const auto exponent = static_cast<std::int64_t>(b);
        const bool isOddExponent = exponent % 2 != 0;
        if (exponent == 0)
            return {1.0, 0.0, 0.0, DomainFault::None};
        if (a == 0.0)
            return exponent < 0 ? refused(DomainFault::DivisionByZero) : OperatorValue{};
        if (std::fabs(a) == 1.0)
            return {(a < 0.0 && isOddExponent) ? -1.0 : 1.0, 0.0, 0.0, DomainFault::None};
        if (exponent < 0)
            return {}; // 1/(I**|J|) truncated, |I| >= 2
        // |a| >= 2: the loop overflows 2^53 within 53 turns.
        double power = 1.0;
        for (std::int64_t count = 0; count < exponent; ++count)
        {
            power *= a;
            if (!(std::fabs(power) <= LARGEST_EXACT_INTEGER))
                return refused(DomainFault::IntegerOverflow);
        }
        return {power, 0.0, 0.0, DomainFault::None};
    }
    }
}

// a ** b in real arithmetic. A whole exponent (an integer constant, or a real of whole value) accepts a negative base,
// as Fortran's X**N; otherwise the base must be >= 0. The partials: ∂/∂a = b·a^(b-1) (0 when b = 0), ∂/∂b = a^b·ln(a)
// (0 at a = 0 with b > 0; it does not exist for a < 0, left NaN, which matters only if b depends on a free argument).
OperatorValue realPower(double a, double b)
{
    const bool isWholeExponent = b == std::trunc(b);
    if (a < 0.0 && !isWholeExponent)
        return refused(DomainFault::OutsideDomain);
    if (a == 0.0 && b < 0.0)
        return refused(DomainFault::DivisionByZero);
    const double value = std::pow(a, b);
    const double partialA = b == 0.0 ? 0.0 : b * std::pow(a, b - 1.0);
    double partialB = QUIET_NAN;
    if (a > 0.0)
        partialB = value * std::log(a);
    else if (a == 0.0 && b > 0.0)
        partialB = 0.0;
    return {value, partialA, partialB, DomainFault::None};
}

OperatorValue realOperator(Opcode opcode, double a, double b)
{
    switch (opcode)
    {
    case Opcode::Add:
        return {a + b, 1.0, 1.0, DomainFault::None};
    case Opcode::Subtract:
        return {a - b, 1.0, -1.0, DomainFault::None};
    case Opcode::Multiply:
        return {a * b, b, a, DomainFault::None};
    case Opcode::Divide:
    {
        if (b == 0.0)
            return refused(DomainFault::DivisionByZero);
        const double value = a / b;
        return {value, 1.0 / b, -value / b, DomainFault::None};
    }
    default:
        return realPower(a, b);
    }
}

std::string formatNumber(double value)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.10g", value);
    return text.data();
}

std::string operationName(const Instruction& instruction)
{
    switch (instruction.opcode)
    {
    case Opcode::Call:
        return std::string(signatureOf(instruction.function).name);
    case Opcode::Negate:
        return "the unary '-'";
    case Opcode::Add:
        return "the operator '+'";
    case Opcode::Subtract:
        return "the operator '-'";
    case Opcode::Multiply:
        return "the operator '*'";
    case Opcode::Divide:
        return "the operator '/'";
    case Opcode::Power:
        return "the operator '**'";
    default:
        return std::string(opcodeName(instruction.opcode));
    }
}

std::string domainOf(const Instruction& instruction)
{
    switch (instruction.opcode)
    {
    case Opcode::Call:
        return std::string(signatureOf(instruction.function).domain);
    case Opcode::Divide:
        return "a non-zero divisor";
    case Opcode::Power:
        return "a base >= 0, or a whole exponent for a negative base, and a non-zero base for a negative exponent";
    default:
        return "finite values";
    }
}

const char* faultCode(DomainFault fault)
{
    switch (fault)
    {
    case DomainFault::OutsideDomain:
        return "design-equation-outside-domain";
    case DomainFault::DivisionByZero:
        return "design-equation-division-by-zero";
    case DomainFault::NotFinite:
        return "design-equation-not-finite";
    case DomainFault::DerivativeNotFinite:
        return "design-equation-derivative-not-finite";
    case DomainFault::IntegerOverflow:
        return "design-equation-integer-overflow";
    default:
        return "design-equation-fault";
    }
}

// What the message says about the fault and what to do, after "<where>: <operation> ".
std::pair<std::string, std::string> faultWords(DomainFault fault, const Instruction& instruction)
{
    switch (fault)
    {
    case DomainFault::OutsideDomain:
        return {base::concatenate({"is outside its domain (", domainOf(instruction), ")"}),
                "expected arguments inside the domain: keep the design variables within it (bounds, or a guard such as "
                "ABS or MAX in the equation) or correct the equation"};
    case DomainFault::DivisionByZero:
        return {"divides by zero",
                "expected a non-zero divisor: keep the design variables away from this point or "
                "guard the divisor in the equation"};
    case DomainFault::NotFinite:
        return {"gives a value that is not finite (overflow)",
                "expected values within the range of double precision: check the scale of the arguments"};
    case DomainFault::DerivativeNotFinite:
        return {"has no finite derivative with respect to a free argument",
                "expected a point where the equation is differentiable: move the design away from this point, bind "
                "the argument as a constant, or evaluate without derivatives"};
    default:
        return {"overflows the exact integers (2^53) in integer arithmetic (between integer constants)",
                "write the constants as reals (2. instead of 2)"};
    }
}

// The first refusal of each kind (instruction, fault) over the whole batch.
struct FaultSummary
{
    std::size_t firstPoint = 0;
    std::size_t pointCount = 0;
    std::array<double, REPORTED_OPERANDS> operands{};
    std::size_t operandCount = 0;
};

} // namespace

// The working memory of one evaluate() call, sized by the program and one block (never by the batch).
class ExpressionEvaluator::Workspace
{
public:
    Workspace(std::size_t registers, std::size_t statements, std::size_t freeCount, std::size_t maxOperands,
              bool withDerivatives)
        : m_freeCount(freeCount),
          m_values(registers * BLOCK, 0.0),
          m_variables(statements * BLOCK, 0.0),
          m_partials(maxOperands * BLOCK, 0.0),
          m_operands(maxOperands, 0.0),
          m_pointPartials(maxOperands, 0.0),
          m_isFaulted(BLOCK, 0)
    {
        if (withDerivatives)
        {
            m_tangents.assign(registers * freeCount * BLOCK, 0.0);
            m_variableTangents.assign(statements * freeCount * BLOCK, 0.0);
        }
    }

    double* value(std::size_t reg) { return &m_values[reg * BLOCK]; }
    double* tangent(std::size_t reg, std::size_t column) { return &m_tangents[(reg * m_freeCount + column) * BLOCK]; }
    double* variable(std::size_t statement) { return &m_variables[statement * BLOCK]; }
    double* variableTangent(std::size_t statement, std::size_t column)
    {
        return &m_variableTangents[(statement * m_freeCount + column) * BLOCK];
    }
    double* partial(std::size_t operand) { return &m_partials[operand * BLOCK]; }
    std::vector<double>& operands() { return m_operands; }
    std::vector<double>& pointPartials() { return m_pointPartials; }
    std::uint8_t& isFaulted(std::size_t point) { return m_isFaulted[point]; }
    void resetFaults() { std::fill(m_isFaulted.begin(), m_isFaulted.end(), std::uint8_t{0}); }
    void setFirstPoint(std::size_t firstPoint) { m_firstPoint = firstPoint; }

    // Records the first fault of a point: the point keeps NaN from here on and is reported once.
    void recordFault(std::size_t stepIndex, DomainFault fault, std::size_t point, std::span<const double> operands)
    {
        m_isFaulted[point] = 1;
        FaultSummary& summary = m_faults[{stepIndex, static_cast<std::uint8_t>(fault)}];
        if (summary.pointCount == 0)
        {
            summary.firstPoint = m_firstPoint + point;
            summary.operandCount = std::min(operands.size(), REPORTED_OPERANDS);
            std::copy_n(operands.begin(), summary.operandCount, summary.operands.begin());
        }
        ++summary.pointCount;
    }

    [[nodiscard]] const std::map<std::pair<std::size_t, std::uint8_t>, FaultSummary>& faults() const
    {
        return m_faults;
    }

    static constexpr std::size_t BLOCK = EVALUATION_BLOCK_POINTS;

private:
    std::size_t m_freeCount;
    std::size_t m_firstPoint = 0;
    std::vector<double> m_values;           // [register][point]
    std::vector<double> m_tangents;         // [register][free column][point]
    std::vector<double> m_variables;        // [statement][point]
    std::vector<double> m_variableTangents; // [statement][free column][point]
    std::vector<double> m_partials;         // [operand][point], the partials of the current operation
    std::vector<double> m_operands;         // the operands of the current point
    std::vector<double> m_pointPartials;    // their partials
    std::vector<std::uint8_t> m_isFaulted;  // [point]
    std::map<std::pair<std::size_t, std::uint8_t>, FaultSummary> m_faults;
};

ExpressionEvaluator::ExpressionEvaluator(ExpressionProgram program,
                                         const std::vector<std::pair<std::string, double>>& constants)
    : m_program(std::move(program))
{
    compileSteps(constants);
}

void ExpressionEvaluator::compileSteps(const std::vector<std::pair<std::string, double>>& constants)
{
    const std::vector<std::string>& arguments = m_program.arguments();
    std::vector<std::int32_t> freeColumns(arguments.size(), -1);
    std::vector<double> boundValues(arguments.size(), 0.0);
    std::vector<bool> isBound(arguments.size(), false);
    for (const auto& [name, value] : constants)
    {
        const auto found = std::find(arguments.begin(), arguments.end(), name);
        if (found == arguments.end())
        {
            std::string list;
            for (const std::string& argument : arguments)
                list.append(list.empty() ? "" : ", ").append(argument);
            throw std::invalid_argument(
                base::concatenate({"constants of ", m_program.label(), ": expected names among its arguments (", list,
                                   "), found '", name, "'"}));
        }
        const auto slot = static_cast<std::size_t>(found - arguments.begin());
        if (isBound[slot])
            throw std::invalid_argument(base::concatenate(
                {"constants of ", m_program.label(), ": the argument '", name, "' is bound twice; give it once"}));
        isBound[slot] = true;
        boundValues[slot] = value;
    }
    for (std::size_t slot = 0; slot < arguments.size(); ++slot)
    {
        if (!isBound[slot])
        {
            freeColumns[slot] = static_cast<std::int32_t>(m_freeArguments.size());
            m_freeArguments.push_back(arguments[slot]);
        }
    }

    // Static stack simulation: the type and the dependence of each register are known before any point.
    const std::vector<Instruction>& instructions = m_program.instructions();
    std::vector<std::uint8_t> registerInteger(m_program.maxStackDepth(), 0);
    std::vector<std::uint8_t> registerDepends(m_program.maxStackDepth(), 0);
    m_statementDepends.assign(m_program.variables().size(), 0);
    m_maxOperands = 2;
    std::size_t depth = 0;
    for (std::size_t index = 0; index < instructions.size(); ++index)
    {
        const Instruction& instruction = instructions[index];
        if (index > 0 && instruction.statement != instructions[index - 1].statement)
            depth = 0; // the value of the previous statement went to its variable
        Step step;
        step.instruction = instruction;
        switch (instruction.opcode)
        {
        case Opcode::RealConstant:
        case Opcode::IntegerConstant:
        case Opcode::Argument:
        case Opcode::Variable:
            step.operandCount = 0;
            break;
        case Opcode::Negate:
            step.operandCount = 1;
            break;
        case Opcode::Call:
            step.operandCount = static_cast<std::uint32_t>(instruction.operand);
            m_maxOperands = std::max<std::size_t>(m_maxOperands, step.operandCount);
            break;
        default:
            step.operandCount = 2;
            break;
        }
        depth -= step.operandCount;
        step.target = static_cast<std::uint32_t>(depth);
        step.firstOperandFlag = static_cast<std::uint32_t>(m_operandDepends.size());
        bool allInteger = step.operandCount > 0;
        bool anyDepends = false;
        for (std::uint32_t operand = 0; operand < step.operandCount; ++operand)
        {
            const std::size_t reg = step.target + operand;
            m_operandDepends.push_back(registerDepends[reg]);
            allInteger = allInteger && registerInteger[reg] != 0;
            anyDepends = anyDepends || registerDepends[reg] != 0;
        }
        switch (instruction.opcode)
        {
        case Opcode::IntegerConstant:
            step.isInteger = true;
            break;
        case Opcode::Argument:
        {
            const auto slot = static_cast<std::size_t>(instruction.operand);
            step.freeColumn = freeColumns[slot];
            step.boundValue = boundValues[slot];
            step.dependsOnFree = step.freeColumn >= 0;
            break;
        }
        case Opcode::Variable:
            step.dependsOnFree = m_statementDepends[static_cast<std::size_t>(instruction.operand)] != 0;
            break;
        case Opcode::Call:
            step.dependsOnFree = anyDepends;
            break;
        case Opcode::RealConstant:
            break;
        default: // Negate and the binary operators keep the integer type of integer operands
            step.isInteger = allInteger;
            step.dependsOnFree = anyDepends;
            break;
        }
        registerInteger[step.target] = step.isInteger ? 1 : 0;
        registerDepends[step.target] = step.dependsOnFree ? 1 : 0;
        ++depth;
        step.endsStatement =
            index + 1 == instructions.size() || instructions[index + 1].statement != instruction.statement;
        if (step.endsStatement)
            m_statementDepends[static_cast<std::size_t>(instruction.statement)] = registerDepends[0];
        m_steps.push_back(step);
    }
}

EvaluationResult ExpressionEvaluator::evaluate(std::span<const double> points, std::size_t pointCount,
                                               bool withDerivatives) const
{
    const std::size_t freeCount = m_freeArguments.size();
    if (points.size() != pointCount * freeCount)
        throw std::invalid_argument(
            base::concatenate({"points of ", m_program.label(), ": expected ", std::to_string(pointCount), " points × ",
                               std::to_string(freeCount), " free arguments = ", std::to_string(pointCount * freeCount),
                               " values, found ", std::to_string(points.size())}));
    EvaluationResult result;
    result.pointCount = pointCount;
    result.freeCount = freeCount;
    result.values.assign(pointCount, 0.0);
    if (withDerivatives)
        result.derivatives.assign(pointCount * freeCount, 0.0);

    Workspace workspace(std::max<std::size_t>(m_program.maxStackDepth(), 1), m_program.variables().size(), freeCount,
                        m_maxOperands, withDerivatives);
    for (std::size_t firstPoint = 0; firstPoint < pointCount; firstPoint += EVALUATION_BLOCK_POINTS)
    {
        const std::size_t blockPoints = std::min(EVALUATION_BLOCK_POINTS, pointCount - firstPoint);
        runBlock(workspace, points, firstPoint, blockPoints, withDerivatives, result);
    }

    for (const auto& [key, summary] : workspace.faults())
    {
        const Step& step = m_steps[key.first];
        const auto fault = static_cast<DomainFault>(key.second);
        const auto statement = static_cast<std::size_t>(step.instruction.statement);
        const std::string& variable = m_program.variables()[statement];
        std::string operands;
        for (std::size_t index = 0; index < summary.operandCount; ++index)
            operands.append(index == 0 ? "" : ", ").append(formatNumber(summary.operands[index]));
        const auto [what, todo] = faultWords(fault, step.instruction);
        std::string message = m_program.label();
        message.append(", statement ").append(std::to_string(statement + 1)).append(" (").append(variable).append(")");
        if (step.instruction.position > 0)
            message.append(", column ").append(std::to_string(step.instruction.position));
        message.append(": ").append(operationName(step.instruction)).append(" ").append(what);
        message.append(" with the arguments (").append(operands).append(") at point ");
        message.append(std::to_string(summary.firstPoint)).append(" (from 0)");
        if (summary.pointCount > 1)
            message.append(" and at ").append(std::to_string(summary.pointCount - 1)).append(" other points");
        message.append("; ").append(todo);
        result.diagnostics.push_back(base::Diagnostic{base::Severity::Error, faultCode(fault), std::move(message),
                                                      "DesignEquation", m_program.equationId(), variable});
    }
    return result;
}

void ExpressionEvaluator::runBlock(Workspace& workspace, std::span<const double> points, std::size_t firstPoint,
                                   std::size_t blockPoints, bool withDerivatives, EvaluationResult& result) const
{
    const std::size_t freeCount = m_freeArguments.size();
    workspace.resetFaults();
    workspace.setFirstPoint(firstPoint);
    for (std::size_t stepIndex = 0; stepIndex < m_steps.size(); ++stepIndex)
    {
        const Step& step = m_steps[stepIndex];
        const bool hasTangent = withDerivatives && step.dependsOnFree;
        double* value = workspace.value(step.target);
        switch (step.instruction.opcode)
        {
        case Opcode::RealConstant:
        case Opcode::IntegerConstant:
            std::fill_n(value, blockPoints, step.instruction.value);
            break;
        case Opcode::Argument:
            if (step.freeColumn < 0)
            {
                std::fill_n(value, blockPoints, step.boundValue);
                break;
            }
            for (std::size_t point = 0; point < blockPoints; ++point)
                value[point] = points[(firstPoint + point) * freeCount + static_cast<std::size_t>(step.freeColumn)];
            if (hasTangent)
            {
                for (std::size_t column = 0; column < freeCount; ++column)
                    std::fill_n(workspace.tangent(step.target, column), blockPoints,
                                column == static_cast<std::size_t>(step.freeColumn) ? 1.0 : 0.0);
            }
            break;
        case Opcode::Variable:
        {
            const auto statement = static_cast<std::size_t>(step.instruction.operand);
            std::copy_n(workspace.variable(statement), blockPoints, value);
            if (hasTangent)
            {
                for (std::size_t column = 0; column < freeCount; ++column)
                    std::copy_n(workspace.variableTangent(statement, column), blockPoints,
                                workspace.tangent(step.target, column));
            }
            break;
        }
        case Opcode::Negate:
            // Negation is exact and defined everywhere: no fault, the tangent changes sign.
            for (std::size_t point = 0; point < blockPoints; ++point)
                value[point] = -value[point];
            if (hasTangent)
            {
                for (std::size_t column = 0; column < freeCount; ++column)
                {
                    double* tangent = workspace.tangent(step.target, column);
                    for (std::size_t point = 0; point < blockPoints; ++point)
                        tangent[point] = -tangent[point];
                }
            }
            break;
        default:
            runOperation(workspace, stepIndex, blockPoints, withDerivatives);
            break;
        }
        if (step.endsStatement)
        {
            const auto statement = static_cast<std::size_t>(step.instruction.statement);
            std::copy_n(workspace.value(0), blockPoints, workspace.variable(statement));
            if (withDerivatives && m_statementDepends[statement] != 0)
            {
                for (std::size_t column = 0; column < freeCount; ++column)
                    std::copy_n(workspace.tangent(0, column), blockPoints,
                                workspace.variableTangent(statement, column));
            }
        }
    }

    const std::size_t last = m_program.variables().size() - 1;
    const bool isDependent = m_statementDepends[last] != 0;
    for (std::size_t point = 0; point < blockPoints; ++point)
    {
        const bool isFaulted = workspace.isFaulted(point) != 0;
        result.values[firstPoint + point] = isFaulted ? QUIET_NAN : workspace.variable(last)[point];
        if (!withDerivatives)
            continue;
        for (std::size_t column = 0; column < freeCount; ++column)
        {
            const double derivative = isDependent ? workspace.variableTangent(last, column)[point] : 0.0;
            result.derivatives[(firstPoint + point) * freeCount + column] = isFaulted ? QUIET_NAN : derivative;
        }
    }
}

void ExpressionEvaluator::runOperation(Workspace& workspace, std::size_t stepIndex, std::size_t blockPoints,
                                       bool withDerivatives) const
{
    const Step& step = m_steps[stepIndex];
    const Instruction& instruction = step.instruction;
    const std::size_t count = step.operandCount;
    const bool hasTangent = withDerivatives && step.dependsOnFree;
    const std::uint8_t* operandDepends = &m_operandDepends[step.firstOperandFlag];
    std::vector<double>& operands = workspace.operands();
    std::vector<double>& pointPartials = workspace.pointPartials();
    const bool isIntegerOperation = count == 2 && step.isInteger;

    // Values and partials, point by point; the operands are read before the result overwrites the first of them.
    for (std::size_t point = 0; point < blockPoints; ++point)
    {
        for (std::size_t operand = 0; operand < count; ++operand)
            operands[operand] = workspace.value(step.target + operand)[point];
        double value = 0.0;
        DomainFault fault = DomainFault::None;
        if (instruction.opcode == Opcode::Call)
        {
            const IntrinsicValue result =
                evaluateIntrinsic(instruction.function, std::span<const double>(operands.data(), count),
                                  hasTangent ? std::span<double>(pointPartials.data(), count) : std::span<double>{});
            value = result.value;
            fault = result.fault;
        }
        else
        {
            const OperatorValue result = isIntegerOperation
                                             ? integerOperator(instruction.opcode, operands[0], operands[1])
                                             : realOperator(instruction.opcode, operands[0], operands[1]);
            value = result.value;
            fault = result.fault;
            pointPartials[0] = result.partialA;
            pointPartials[1] = result.partialB;
        }
        if (workspace.isFaulted(point) == 0)
        {
            if (fault == DomainFault::None && !std::isfinite(value))
                fault = DomainFault::NotFinite;
            if (fault == DomainFault::None && hasTangent)
            {
                for (std::size_t operand = 0; operand < count; ++operand)
                {
                    if (operandDepends[operand] != 0 && !std::isfinite(pointPartials[operand]))
                        fault = DomainFault::DerivativeNotFinite;
                }
            }
            if (fault != DomainFault::None)
                workspace.recordFault(stepIndex, fault, point, std::span<const double>(operands.data(), count));
        }
        if (workspace.isFaulted(point) != 0)
            value = QUIET_NAN;
        workspace.value(step.target)[point] = value;
        if (hasTangent)
        {
            for (std::size_t operand = 0; operand < count; ++operand)
                workspace.partial(operand)[point] = pointPartials[operand];
        }
    }
    if (!hasTangent)
        return;

    // Chain rule, operand after operand in the written order: dc/dx_k = Σ_j ∂f/∂a_j · da_j/dx_k over the operands that
    // depend on a free argument. The result's tangent shares its register with the first operand, which is read
    // before it is written, element by element.
    const std::size_t freeCount = m_freeArguments.size();
    for (std::size_t column = 0; column < freeCount; ++column)
    {
        double* out = workspace.tangent(step.target, column);
        bool isFirst = true;
        for (std::size_t operand = 0; operand < count; ++operand)
        {
            if (operandDepends[operand] == 0)
                continue;
            const double* source = workspace.tangent(step.target + operand, column);
            const double* partial = workspace.partial(operand);
            if (isFirst)
            {
                for (std::size_t point = 0; point < blockPoints; ++point)
                    out[point] = partial[point] * source[point];
                isFirst = false;
            }
            else
            {
                for (std::size_t point = 0; point < blockPoints; ++point)
                    out[point] += partial[point] * source[point];
            }
        }
    }
}

} // namespace oos::design
