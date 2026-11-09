#include "oos/solutions/common/EigenMethod.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace oos::solutions
{

namespace
{

// The codes of O_o_Reader's method tables (data contract of model.lanczosTable/eigr/eigb.table(): uint8 columns).
// LanczosEigenMethod normalization: blank, Mass, Maximum.
constexpr std::uint8_t LANCZOS_NORM_MASS = 1;
constexpr std::uint8_t LANCZOS_NORM_MAX = 2;
// RealEigenMethod method (in this order; the names a message gives); normalization: Mass, Maximum, Point.
constexpr std::array<std::string_view, 9> REAL_EIGEN_METHODS{
    "Lanczos",        "AutomaticHouseholder", "InverseIteration",    "SturmInverse",   "Givens",
    "ModifiedGivens", "Householder",          "ModifiedHouseholder", "AutomaticGivens"};
constexpr std::uint8_t REAL_EIGEN_LANCZOS = 0;
constexpr std::uint8_t REAL_EIGEN_NORM_MAX = 1;
constexpr std::uint8_t REAL_EIGEN_NORM_POINT = 2;
// BucklingEigenMethod method: blank, InverseIteration, SturmInverse; normalization: Maximum, Point.
constexpr std::array<std::string_view, 3> BUCKLING_EIGEN_METHODS{"blank", "InverseIteration", "SturmInverse"};
constexpr std::uint8_t BUCKLING_EIGEN_NORM_POINT = 1;
// 2π: the eigenvalue of a frequency f is (2π·f)².
constexpr double TWO_PI = 6.283185307179586;

base::Diagnostic methodProblem(base::Severity severity, std::string code, const std::string& card, base::EntityId setId,
                               std::string field, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = card;
    diagnostic.id = setId;
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The row of `setId` in a method table, or -1.
std::int64_t rowOf(const input::CardTable& table, base::EntityId setId)
{
    const std::span<const std::int32_t> ids = table.ids();
    for (std::size_t row = 0; row < ids.size(); ++row)
        if (ids[row] == setId)
            return static_cast<std::int64_t>(row);
    return -1;
}

// A real field of a method table: NaN is a blank field.
std::optional<double> realField(const input::CardTable& table, const char* column, std::int64_t row)
{
    const double value = table.reader().values<double>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
    if (std::isnan(value))
        return std::nullopt;
    return value;
}

std::int32_t integerField(const input::CardTable& table, const char* column, std::int64_t row)
{
    return table.reader().values<std::int32_t>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

std::uint8_t codeField(const input::CardTable& table, const char* column, std::int64_t row)
{
    return table.reader().values<std::uint8_t>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

std::size_t countOf(std::int32_t value)
{
    return value > 0 ? static_cast<std::size_t>(value) : 0;
}

EigenMethod readLanczosEigenMethod(const input::CardTable& table, std::int64_t row, base::EntityId setId,
                                   AnalysisKind analysis, base::DiagnosticSink& sink)
{
    EigenMethod method;
    method.card = "LanczosEigenMethod";
    method.setId = setId;
    method.lower = realField(table, "lowerFrequency", row);
    method.upper = realField(table, "upperFrequency", row);
    method.modeCount = countOf(integerField(table, "desiredModeCount", row));
    method.blockSize = countOf(integerField(table, "maximumSet", row));
    method.messageLevel = integerField(table, "messageLevel", row);
    const std::uint8_t norm = codeField(table, "norm", row);
    if (analysis == AnalysisKind::Buckling)
    {
        method.normalization = EigenNormalization::Max;
        if (norm == LANCZOS_NORM_MASS)
            sink.report(methodProblem(base::Severity::Warning, "norm-mass-buckling", "LanczosEigenMethod", setId,
                                      "norm",
                                      "LanczosEigenMethod.norm = Mass is not available for a buckling "
                                      "analysis: the buckling shapes are normalised by Maximum; write "
                                      "LanczosEigenMethod.norm = Maximum"));
    }
    else
        method.normalization = norm == LANCZOS_NORM_MAX ? EigenNormalization::Max : EigenNormalization::Mass;
    return method;
}

EigenMethod readEigr(const input::CardTable& table, std::int64_t row, base::EntityId setId, base::DiagnosticSink& sink)
{
    EigenMethod method;
    method.card = "RealEigenMethod";
    method.setId = setId;
    const std::uint8_t code = codeField(table, "method", row);
    if (code != REAL_EIGEN_LANCZOS)
    {
        const std::string_view name =
            code < REAL_EIGEN_METHODS.size() ? REAL_EIGEN_METHODS[code] : std::string_view("unknown");
        sink.report(methodProblem(
            base::Severity::Error, "unsupported-method", "RealEigenMethod", setId, "method",
            base::concatenate(
                {"expected RealEigenMethod.method = Lanczos (the block Lanczos method of O_o_Solver), "
                 "found ",
                 name,
                 ", which this version does not offer (AutomaticHouseholder, InverseIteration, SturmInverse, Givens, "
                 "ModifiedGivens, "
                 "Householder, ModifiedHouseholder and AutomaticGivens are refused); write "
                 "RealEigenMethod.method = Lanczos or use an "
                 "LanczosEigenMethod card"})));
    }
    method.lower = realField(table, "frequency1", row);
    method.upper = realField(table, "frequency2", row);
    method.modeCount = countOf(integerField(table, "desiredModeCount", row));
    const std::uint8_t norm = codeField(table, "normalization", row);
    method.normalization = norm == REAL_EIGEN_NORM_POINT ? EigenNormalization::Point
                           : norm == REAL_EIGEN_NORM_MAX ? EigenNormalization::Max
                                                         : EigenNormalization::Mass;
    method.normalizationPoint = integerField(table, "normNode", row);
    method.normalizationComponent = integerField(table, "normComponent", row);
    return method;
}

EigenMethod readBucklingEigenMethod(const input::CardTable& table, std::int64_t row, base::EntityId setId,
                                    base::DiagnosticSink& sink)
{
    EigenMethod method;
    method.card = "BucklingEigenMethod";
    method.setId = setId;
    const std::uint8_t code = codeField(table, "method", row);
    const std::string_view name =
        code < BUCKLING_EIGEN_METHODS.size() ? BUCKLING_EIGEN_METHODS[code] : std::string_view("unknown");
    sink.report(methodProblem(
        base::Severity::Information, "buckling-eigen-lanczos", "BucklingEigenMethod", setId, "method",
        base::concatenate({"the inverse power method of BucklingEigenMethod "
                           "(BucklingEigenMethod.method ",
                           name,
                           ") is replaced by the block Lanczos solver over the same range "
                           "[BucklingEigenMethod.lowerEigenvalue, BucklingEigenMethod.upperEigenvalue], with Sturm "
                           "checks: every root of the range is found, and "
                           "none outside it"})));
    method.lower =
        table.reader().values<double>("lowerEigenvalue", base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
    method.upper =
        table.reader().values<double>("upperEigenvalue", base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
    const std::size_t positive = countOf(integerField(table, "desiredPositiveModeCount", row));
    const std::size_t negative = countOf(integerField(table, "desiredNegativeModeCount", row));
    // BucklingEigenMethod: NDP roots wanted in the positive part of the range, NDN in the negative part.
    if (*method.lower >= 0.0)
        method.modeCount = positive;
    else if (*method.upper <= 0.0)
        method.modeCount = negative;
    else
        method.modeCount = positive + negative;
    method.normalization = codeField(table, "norm", row) == BUCKLING_EIGEN_NORM_POINT ? EigenNormalization::Point
                                                                                      : EigenNormalization::Max;
    method.normalizationPoint = integerField(table, "normNode", row);
    method.normalizationComponent = integerField(table, "normComponent", row);
    return method;
}

// The fields of the range of a method card, by their neutral names (V1, V2 of LanczosEigenMethod in a deck).
std::array<const char*, 2> rangeFieldsOf(const std::string& card)
{
    if (card == "RealEigenMethod")
        return {"frequency1", "frequency2"};
    if (card == "BucklingEigenMethod")
        return {"lowerEigenvalue", "upperEigenvalue"};
    return {"lowerFrequency", "upperFrequency"};
}

// A qualified neutral method field, such as "LanczosEigenMethod.lowerFrequency".
std::string qualifiedField(const std::string& card, const char* field)
{
    return base::concatenate({card, ".", field});
}

// Whether the range of a method card can be solved: the request the block Lanczos solver gets is checked here as
// linalg::ShiftStrategy checks it (where a range it cannot take is a misuse, std::invalid_argument), so that a card of
// the model with such a range is an error diagnostic naming its fields, never an exception of the solver. A bound must
// be finite (a blank one is no bound); the roots sought from the lowest (no lower bound, or one at or below 0) need an
// upper bound above 0; otherwise the lower bound must be below the upper one.
bool isRangeUsable(const EigenMethod& method, AnalysisKind analysis, base::DiagnosticSink& sink)
{
    const auto [lowerName, upperName] = rangeFieldsOf(method.card);
    const std::string lowerField = qualifiedField(method.card, lowerName);
    const std::string upperField = qualifiedField(method.card, upperName);
    const auto refuse = [&](const char* field, std::string message)
    {
        sink.report(
            methodProblem(base::Severity::Error, "method-range", method.card, method.setId, field, std::move(message)));
        return false;
    };
    if (method.lower && !std::isfinite(*method.lower))
        return refuse(lowerName, base::concatenate({"expected a finite ", lowerField, " (or a blank one), found ",
                                                    std::to_string(*method.lower), "; correct the range"}));
    if (method.upper && !std::isfinite(*method.upper))
        return refuse(upperName, base::concatenate({"expected a finite ", upperField, " (or a blank one), found ",
                                                    std::to_string(*method.upper), "; correct the range"}));
    const linalg::EigenRequest request = eigenRequestOf(method, analysis);
    if (!request.hasUpperBound())
        return true;
    if (!request.startsFromLowest() && !(request.upperBound > request.lowerBound))
        return refuse(upperName,
                      base::concatenate({"expected ", lowerField, " below ", upperField, ", found ", lowerField, " = ",
                                         std::to_string(*method.lower), " and ", upperField, " = ",
                                         std::to_string(*method.upper), "; correct the range"}));
    if (request.startsFromLowest() && !(request.upperBound > 0.0))
        return refuse(upperName,
                      base::concatenate(
                          {"expected ", upperField, " above 0 when the roots are sought from the lowest (", lowerField,
                           " blank or not above 0), found ", upperField, " = ", std::to_string(*method.upper),
                           "; give a positive ", upperField, ", or a range above 0"}));
    return true;
}

} // namespace

std::optional<EigenMethod> readEigenMethod(const input::ModelSnapshot& snapshot, base::EntityId setId,
                                           AnalysisKind analysis, base::EntityId subcase, base::DiagnosticSink& sink)
{
    const auto checked = [&](EigenMethod method) -> std::optional<EigenMethod>
    {
        if (!isRangeUsable(method, analysis, sink))
            return std::nullopt;
        return method;
    };
    if (const input::CardTable* lanczosTable = snapshot.findMethodTable("LanczosEigenMethod"))
        if (const std::int64_t row = rowOf(*lanczosTable, setId); row >= 0)
            return checked(readLanczosEigenMethod(*lanczosTable, row, setId, analysis, sink));
    const char* other = analysis == AnalysisKind::Buckling ? "BucklingEigenMethod" : "RealEigenMethod";
    if (const input::CardTable* table = snapshot.findMethodTable(other))
        if (const std::int64_t row = rowOf(*table, setId); row >= 0)
            return checked(analysis == AnalysisKind::Buckling ? readBucklingEigenMethod(*table, row, setId, sink)
                                                              : readEigr(*table, row, setId, sink));
    base::Diagnostic diagnostic;
    diagnostic.code = "missing-method";
    diagnostic.card = "SubcaseStart";
    diagnostic.id = subcase;
    diagnostic.field = "EigenMethodSelection";
    diagnostic.message =
        base::concatenate({"expected an LanczosEigenMethod or ", other, " card of set ", std::to_string(setId),
                           " for EigenMethodSelection = ", std::to_string(setId),
                           ", found none; add the card or "
                           "correct the EigenMethodSelection"});
    sink.report(std::move(diagnostic));
    return std::nullopt;
}

linalg::EigenRequest eigenRequestOf(const EigenMethod& method, AnalysisKind analysis)
{
    constexpr double INFINITE = std::numeric_limits<double>::infinity();
    const bool isBuckling = analysis == AnalysisKind::Buckling;
    const auto eigenvalueOf = [isBuckling](double value)
    {
        if (isBuckling)
            return value;
        const double omega = TWO_PI * value;
        return value < 0.0 ? -omega * omega : omega * omega;
    };
    linalg::EigenRequest request;
    request.lowerBound = method.lower ? eigenvalueOf(*method.lower) : -INFINITE;
    request.upperBound = method.upper ? eigenvalueOf(*method.upper) : INFINITE;
    request.modeCount = method.modeCount;
    // LanczosEigenMethod: without ND, all the roots of a closed range, else the lowest root (for a buckling analysis an
    // open range has infinitely many roots: the root of smallest magnitude).
    const bool isClosed = method.upper.has_value() && (method.lower.has_value() || !isBuckling);
    if (request.modeCount == 0 && !isClosed)
        request.modeCount = 1;
    return request;
}

} // namespace oos::solutions
