#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/expressions/ExpressionProgram.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

// The values of an equation at a batch of points and, when asked, its exact derivatives.
struct EvaluationResult
{
    std::size_t pointCount = 0;
    std::size_t freeCount = 0;
    std::vector<double> values;      // one per point; NaN at a refused point
    std::vector<double> derivatives; // pointCount × freeCount, point after point; empty when not asked
    // One per kind of refusal (instruction and fault), in program order, each with the number of points concerned and
    // the first of them; empty when every point gave a value (and its derivatives).
    std::vector<base::Diagnostic> diagnostics;

    [[nodiscard]] bool isValid() const noexcept { return diagnostics.empty(); }
};

// Evaluates a DesignEquation expression program at many points at once, with exact forward-mode derivatives (equations
// get automatic derivatives).
//
// Method: forward-mode automatic differentiation. Every stack value carries, beside its value v, its tangent: the
// vector of its partial derivatives dv/dx_k with respect to the free arguments x_k. An argument starts with the unit
// tangent e_k; an operation c = f(a, b, ...) gives dc/dx_k = Σ_j ∂f/∂a_j · da_j/dx_k (the chain rule), with the
// partials ∂f/∂a_j of evaluateIntrinsic for the functions and of the operators: d(a·b) = b·da + a·db,
// d(a/b) = da/b - (a/b)·db/b, d(a**b) = b·a^(b-1)·da + a^b·ln(a)·db. The derivatives are exact to rounding: no step,
// no truncation error. A value that depends on no free argument (a constant, an argument bound by `constants`) carries
// no tangent at all, decided once when the evaluator is built.
//
// Arithmetic (DEQATN follows Fortran rules): Fortran precedence is the producer's job; the evaluator runs the postfix
// program. Integer constants combine in integer arithmetic (1/2 = 0, 7/2 = 3, 2**3 = 8, 2**-1 = 0), any real operand
// makes the operation real, every function and every variable is real. A negative base to a real power is refused
// unless the exponent is a whole number (then it is computed as for an integer exponent).
//
// Refusals: a point where a function or an operator leaves its domain (SQRT of a negative, a division by zero, ACOS
// of 1.5), where a value overflows, or (when derivatives are asked) where a partial derivative with respect to a free
// argument does not exist (SQRT at 0), gets NaN and is reported by a Diagnostic naming the equation, the statement,
// the column of the token in the text, the function or operator, the arguments found and the domain expected. The
// other points of the batch are computed normally.
//
// Batches: the points are processed in blocks of EVALUATION_BLOCK_POINTS, instruction by instruction across a block,
// so a call of hundreds of points pays the program's dispatch once per block. Each point follows exactly the same
// operations in the same order whatever the batch, so a batch gives the same bits as the points evaluated one by one
// (determinism, CLAUDE.md rule 7); sums inside SUM, AVG, SSQ, RSS and the chain rule run in the written order.
//
// Thread-safety: immutable once built; evaluate() may be called from several threads at once. Ownership: owns a copy
// of the program.
class ExpressionEvaluator
{
public:
    // The points of a block. Bounded so that the working memory (stack depth × (1 + free arguments) × block) does not
    // grow with the batch.
    static constexpr std::size_t EVALUATION_BLOCK_POINTS = 256;

    // `constants`: arguments bound to fixed values (the DesignConstants constants, or the caller's `constants=`); the
    // other arguments are free, in header order. std::invalid_argument for a name that is not an argument or is bound
    // twice.
    explicit ExpressionEvaluator(ExpressionProgram program,
                                 const std::vector<std::pair<std::string, double>>& constants = {});

    [[nodiscard]] const ExpressionProgram& program() const noexcept { return m_program; }
    // The free arguments, in header order: the columns of the points and of the derivatives.
    [[nodiscard]] const std::vector<std::string>& freeArguments() const noexcept { return m_freeArguments; }

    // Evaluates at `pointCount` points; `points` holds pointCount × freeArguments().size() values, point after point.
    // std::invalid_argument when its size does not match. Complexity: O(points × instructions × (1 + free arguments)).
    [[nodiscard]] EvaluationResult evaluate(std::span<const double> points, std::size_t pointCount,
                                            bool withDerivatives) const;

private:
    // One instruction with what is decided before any point: where its operands are, and whether its value is an
    // integer (integer arithmetic) and whether it depends on a free argument.
    struct Step
    {
        Instruction instruction;
        std::uint32_t target = 0; // the stack register of the result; the operands are target .. target+count-1
        std::uint32_t operandCount = 0;
        std::uint32_t firstOperandFlag = 0; // index in m_operandDepends of the first operand's flag
        std::int32_t freeColumn = -1;       // Argument: its column among the free arguments, -1 when bound
        double boundValue = 0.0;            // Argument bound to a constant: its value
        bool isInteger = false;
        bool dependsOnFree = false;
        bool endsStatement = false;
    };

    class Workspace;

    void compileSteps(const std::vector<std::pair<std::string, double>>& constants);
    void runBlock(Workspace& workspace, std::span<const double> points, std::size_t firstPoint, std::size_t blockPoints,
                  bool withDerivatives, EvaluationResult& result) const;
    void runOperation(Workspace& workspace, std::size_t stepIndex, std::size_t blockPoints, bool withDerivatives) const;

    ExpressionProgram m_program;
    std::vector<std::string> m_freeArguments;
    std::vector<Step> m_steps;
    std::vector<std::uint8_t> m_operandDepends; // per step, per operand: depends on a free argument
    std::vector<std::uint8_t> m_statementDepends;
    std::size_t m_maxOperands = 0;
};

} // namespace oos::design
