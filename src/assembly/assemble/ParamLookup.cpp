#include "oos/assembly/assemble/ParamLookup.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

namespace oos::assembly
{

base::Diagnostic badParamDiagnostic(const std::string& name, const std::string& expected)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Warning;
    diagnostic.code = "bad-param";
    diagnostic.card = base::concatenate({"Parameter ", name});
    diagnostic.message = base::concatenate(
        {"Parameter ", name, " expects ", expected, "; the value given is ignored and the default used"});
    return diagnostic;
}

std::optional<double> findRealParam(const input::ParamTable* params, const std::string& name,
                                    std::optional<std::int32_t> subcase, base::DiagnosticSink& sink)
{
    if (params == nullptr)
        return std::nullopt;
    const std::optional<input::ParamValue> value = params->find(name, subcase);
    if (!value)
        return std::nullopt;
    if (value->kind == input::ParamValue::Kind::Real)
        return value->real;
    if (value->kind == input::ParamValue::Kind::Integer)
        return static_cast<double>(value->integer);
    sink.report(badParamDiagnostic(name, "a real number"));
    return std::nullopt;
}

std::optional<std::int32_t> findIntegerParam(const input::ParamTable* params, const std::string& name,
                                             std::optional<std::int32_t> subcase, base::DiagnosticSink& sink)
{
    if (params == nullptr)
        return std::nullopt;
    const std::optional<input::ParamValue> value = params->find(name, subcase);
    if (!value)
        return std::nullopt;
    if (value->kind == input::ParamValue::Kind::Integer)
        return value->integer;
    sink.report(badParamDiagnostic(name, "an integer"));
    return std::nullopt;
}

} // namespace oos::assembly
