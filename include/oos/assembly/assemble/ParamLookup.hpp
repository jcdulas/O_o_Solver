#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/input/tables/ParamTable.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace oos::assembly
{

// The PARAM values the assembly and the solution sequences read, by their neutral names ("WeightToMass"), with the rule
// of every reader: a Case Control PARAM
// of the subcase overrides the Bulk Data one (ParamTable::find); `params` may be null (a snapshot without the table:
// the format's defaults apply); a value of the wrong kind is reported as a Warning "bad-param" naming what was
// expected, and nothing is returned, so that the caller keeps the default.

// A real PARAM (an integer value is accepted and converted, so that "PARAM,WTMASS,1" is read as 1.0 for WeightToMass).
[[nodiscard]] std::optional<double> findRealParam(const input::ParamTable* params, const std::string& name,
                                                  std::optional<std::int32_t> subcase, base::DiagnosticSink& sink);

// An integer PARAM.
[[nodiscard]] std::optional<std::int32_t> findIntegerParam(const input::ParamTable* params, const std::string& name,
                                                           std::optional<std::int32_t> subcase,
                                                           base::DiagnosticSink& sink);

// The Warning "bad-param" of a PARAM whose value is not what `expected` says; the message names the PARAM as the deck
// writes it (ParamTable::label; the neutral name when `params` is null).
[[nodiscard]] base::Diagnostic badParamDiagnostic(const std::string& name, const std::string& expected);

} // namespace oos::assembly
