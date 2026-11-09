#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/design/expressions/ExpressionProgram.hpp"

#include <vector>

namespace oos::design
{

// The two tables by which O_o_Reader hands over parsed DesignEquation equations (the full contract text is in
// src/design/README.md):
//
// designEquationSymbols, one row per name of an equation:
//   equation_id int32 · kind dictionary<utf8> ("argument" or "variable") · slot int32 (from 0: the position in the
//   header list for an argument, the statement index for a variable) · name utf8 (upper case)
// designEquationInstructions, one row per instruction, the rows of one equation contiguous and in execution order:
//   equation_id int32 · statement int32 · opcode dictionary<utf8> (opcodeName) · operand int32 · value float64 ·
//   function dictionary<utf8> (the function name, null unless the opcode is "call") · position int32 (column from 1,
//   0 if unknown)
[[nodiscard]] base::TableContract designEquationSymbolsContract();
[[nodiscard]] base::TableContract designEquationInstructionsContract();

// The programs of the tables, in increasing equation ID. ContractError when a row breaks the contract: an unknown
// opcode or function name (not a DesignEquation intrinsic function), a symbol slot missing or given twice, the rows of
// an equation not contiguous, instructions of an equation that has no symbols, or a program that breaks the invariants
// of ExpressionProgram.
[[nodiscard]] std::vector<ExpressionProgram> readExpressionPrograms(const base::ContractReader& symbols,
                                                                    const base::ContractReader& instructions);

} // namespace oos::design
