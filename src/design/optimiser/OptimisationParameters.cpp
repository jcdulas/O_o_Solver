#include "oos/design/optimiser/OptimisationParameters.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/optimiser/OptimiserDefaults.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

namespace oos::design
{

namespace
{

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NO_DEFAULT = std::numeric_limits<double>::quiet_NaN();

using enum ParameterKind;
using enum ParameterUse;
namespace defaults = optimiser_defaults;

// The parameters of the DesignOptimParams entry, with the default and the valid range O_o_Solver applies to each and
// what it does with it (ParameterUse). Each default is the one of the ADS or CONMIN parameter whose role the keyword
// plays (OptimiserDefaults.hpp gives the sources), or a choice of this project, as the meaning says. A NoEffect
// parameter takes any value of its type: its default and range are those of no check (NaN, ±∞).
constexpr double NO_EFFECT = NO_DEFAULT;
constexpr std::array<ParameterDefinition, DESIGN_OPTIM_PARAMS_COUNT> DEFINITIONS{{
    {DesignOptimParams::ApproximationMethod, "ApproximationMethod", Integer, Optimiser, 2.0, 1.0, 3.0, false, false,
     "approximation of the responses in a design cycle: 1 direct linearisation, 2 mixed (reciprocal variables for "
     "the structural responses, direct for the mass ones: the approximation concepts of Schmit and Farshi 1974 and "
     "Schmit and Miura, NASA CR-2552, 1976; the default), 3 convex linearisation (Fleury and Braibant 1986)"},
    {DesignOptimParams::AutomaticSuperelementCreation, "AutomaticSuperelementCreation", Integer, NoEffect, NO_EFFECT,
     -INF, INF, false, false, "request of another kind of job (superelement creation), which O_o_Solver does not run"},
    {DesignOptimParams::RelativeObjectiveConvergence, "RelativeObjectiveConvergence", Real, Optimiser,
     defaults::CYCLE_RELATIVE_OBJECTIVE_CHANGE, 0.0, INF, true, false,
     "relative change of the objective between two design cycles that indicates convergence (ADS DELSTR, 0.001)"},
    {DesignOptimParams::AbsoluteObjectiveConvergence, "AbsoluteObjectiveConvergence", Real, Optimiser, NO_DEFAULT, 0.0,
     INF, true, false,
     "absolute change of the objective between two design cycles that indicates convergence (ADS DABSTR, default "
     "|F0|/1000 with F0 the initial objective)"},
    {DesignOptimParams::DesignVarConvergence, "DesignVarConvergence", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false,
     "relative change of the design variables as a convergence test: the design cycles converge on the objective and "
     "the feasibility (ADS, CONMIN); the changes are reported by each convergence check"},
    {DesignOptimParams::PropertyConvergence, "PropertyConvergence", Real, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "relative change of the designed properties as a convergence test: as CONVDV"},
    {DesignOptimParams::ConstraintActiveThreshold, "ConstraintActiveThreshold", Real, Optimiser,
     defaults::ACTIVE_CONSTRAINT_THRESHOLD, -INF, 0.0, false, true,
     "a constraint above CT is active in the approximate optimisation (ADS CT, -0.03)"},
    {DesignOptimParams::ConstraintViolationThreshold, "ConstraintViolationThreshold", Real, Optimiser,
     defaults::VIOLATED_CONSTRAINT_THRESHOLD, 0.0, INF, true, false,
     "a constraint above CTMIN is violated in the approximate optimisation (ADS CTMIN, 0.01)"},
    {DesignOptimParams::AbsoluteObjectiveChange, "AbsoluteObjectiveChange", Real, Optimiser, NO_DEFAULT, 0.0, INF, true,
     false,
     "absolute change of the objective between two iterations of the approximate optimisation that indicates "
     "convergence (ADS DABOBJ, default |F0|/1000)"},
    {DesignOptimParams::FiniteDifferenceStep, "FiniteDifferenceStep", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false,
     "relative finite difference step of the sensitivities: O_o_Solver's sensitivities are analytic or "
     "semi-analytic with steps of their own"},
    {DesignOptimParams::RelativeObjectiveChange, "RelativeObjectiveChange", Real, Optimiser,
     defaults::ITERATION_RELATIVE_OBJECTIVE_CHANGE, 0.0, INF, true, false,
     "relative change of the objective between two iterations of the approximate optimisation that indicates "
     "convergence (ADS DELOBJ, 0.001)"},
    {DesignOptimParams::PropertyMoveLimit, "PropertyMoveLimit", Real, Optimiser, NO_DEFAULT, 0.0, INF, true, false,
     "fractional move limit of each designed property in one design cycle (ADS RMVLMZ: default 0.2 with APRCOD = 1, "
     "0.4 otherwise)"},
    {DesignOptimParams::DesignVarMoveLimit, "DesignVarMoveLimit", Real, Optimiser, NO_DEFAULT, 0.0, INF, true, false,
     "fractional move limit of each design variable in one design cycle, DesignVar "
     "DesignVar.designVarMoveLimit overriding it (ADS RMVLMZ: "
     "default 0.2 with APRCOD = 1, 0.4 otherwise)"},
    {DesignOptimParams::EquivalentLoadMoveLimit, "EquivalentLoadMoveLimit", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "move limit of the nonlinear equivalent static loads loop, which O_o_Solver does not run"},
    {DesignOptimParams::MaxDesignCycles, "MaxDesignCycles", Integer, Optimiser,
     static_cast<double>(defaults::MAXIMUM_CYCLES), 0.0, INF, true, false,
     "maximum number of design cycles (ADS JTMAX, 20)"},
    {DesignOptimParams::DiscreteMethod, "DiscreteMethod", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "discrete processing method (only with DesignDiscreteValues, which O_o_Solver refuses)"},
    {DesignOptimParams::DiscreteStartCycle, "DiscreteStartCycle", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "first design cycle of discrete processing (only with DesignDiscreteValues)"},
    {DesignOptimParams::MaxPropertyChange, "MaxPropertyChange", Real, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "trust region limit of a property (O_o_Solver has no trust region method)"},
    {DesignOptimParams::MinPropertyChange, "MinPropertyChange", Real, Optimiser, defaults::MINIMUM_SCALE, 0.0, INF,
     false, false,
     "smallest move limit of a designed property, absolute (O_o_Solver's choice: ADS's smallest scale factor SCLMIN, "
     "0.001)"},
    {DesignOptimParams::SuperelementStopRatio, "SuperelementStopRatio", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "stop ratio of a superelement creation job (AUTOSE)"},
    {DesignOptimParams::EquivalentLoadMaxCycles, "EquivalentLoadMaxCycles", Integer, NoEffect, NO_EFFECT, -INF, INF,
     false, false, "maximum cycles of the nonlinear equivalent static loads loop, which O_o_Solver does not run"},
    {DesignOptimParams::MaxDesignVarChange, "MaxDesignVarChange", Real, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "trust region limit of a design variable (O_o_Solver has no trust region method)"},
    {DesignOptimParams::MinDesignVarChange, "MinDesignVarChange", Real, Optimiser, defaults::MINIMUM_SCALE, 0.0, INF,
     false, false,
     "smallest move limit of a design variable, absolute (O_o_Solver's choice: ADS's smallest scale factor SCLMIN, "
     "0.001)"},
    {DesignOptimParams::TrustRegionLowRatio, "TrustRegionLowRatio", Real, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "trust region control"},
    {DesignOptimParams::TrustRegionMiddleRatio, "TrustRegionMiddleRatio", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "trust region control"},
    {DesignOptimParams::TrustRegionHighRatio, "TrustRegionHighRatio", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "trust region control"},
    {DesignOptimParams::FullyStressedRelaxation, "FullyStressedRelaxation", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "relaxation of the fully stressed design cycles (only with FSDMAX > 0, which O_o_Solver refuses)"},
    {DesignOptimParams::FullyStressedMaxCycles, "FullyStressedMaxCycles", Integer, DefaultOnly, 0.0, 0.0, INF, false,
     false,
     "number of fully stressed design cycles before the optimisation (0: none; O_o_Solver has no fully stressed "
     "design)"},
    {DesignOptimParams::MaxConstraintViolation, "MaxConstraintViolation", Real, Optimiser,
     defaults::VIOLATED_CONSTRAINT_THRESHOLD, 0.0, INF, true, false,
     "largest normalised constraint of a converged design: the design cycles converge only on a design whose exact "
     "constraints are at most GMAX (the role of ADS CTMIN on the exact analysis, 0.01)"},
    {DesignOptimParams::ConstraintScale, "ConstraintScale", Real, Optimiser, defaults::MINIMUM_SCALE, 0.0, INF, true,
     false,
     "normalisation of a constraint bound near zero (O_o_Solver's choice: ADS's smallest scale factor SCLMIN, "
     "0.001)"},
    {DesignOptimParams::ViolationTreatment, "ViolationTreatment", Integer, DefaultOnly, 0.0, 0.0, INF, false, false,
     "gradients of more than the active and violated constraints in the optimiser (0: only those, as the feasible "
     "directions methods need)"},
    {DesignOptimParams::OptimizerPrintLevel, "OptimizerPrintLevel", Integer, NoEffect, NO_EFFECT, -INF, INF, false,
     false,
     "print control of a printed listing, which O_o_Solver does not write (the iteration history is in "
     "OptimiserResult)"},
    {DesignOptimParams::DesignVarScaling, "DesignVarScaling", Integer, Optimiser, 0.0, -1.0, INF, false, false,
     "design variable scaling in the optimiser: -1 none, otherwise scaled by their value at the start (ADS ISCAL "
     "scales by default)"},
    {DesignOptimParams::OptimizationMethod, "OptimizationMethod", Integer, Optimiser, 0.0, 0.0, 3.0, false, false,
     "optimiser: 0 automatic (the modified method of feasible directions), 1 modified method of feasible directions, "
     "2 sequential linear programming, 3 sequential quadratic programming"},
    {DesignOptimParams::PrintAnalysisEveryCycle, "PrintAnalysisEveryCycle", Integer, NoEffect, NO_EFFECT, -INF, INF,
     false, false, "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::ObjectiveRelativeToStart, "ObjectiveRelativeToStart", Integer, DefaultOnly, 0.0, -INF, INF,
     false, false, "1 measures the objective from its value at the start (0: as it is)"},
    {DesignOptimParams::OptimizerName, "OptimizerName", Text, NoEffect, NO_DEFAULT, 0.0, 0.0, false, false,
     "code naming an optimiser library: O_o_Solver runs its own optimisers (METHOD) whatever the code"},
    {DesignOptimParams::PrintInterval, "PrintInterval", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintItems, "PrintItems", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintMaxRetained, "PrintMaxRetained", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintBeadConstraints, "PrintBeadConstraints", Integer, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintConstraintCount, "PrintConstraintCount", Integer, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintDiscreteVars, "PrintDiscreteVars", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintMaterialCount, "PrintMaterialCount", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintPropertyCount, "PrintPropertyCount", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintResponseCount, "PrintResponseCount", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintResponseSet, "PrintResponseSet", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of a printed listing, which O_o_Solver does not write"},
    {DesignOptimParams::PrintApproximation, "PrintApproximation", Integer, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "print control of topology variables, which O_o_Solver does not have"},
    {DesignOptimParams::PenaltyFactor, "PenaltyFactor", Real, DefaultOnly, 0.0, -INF, INF, false, false,
     "penalty that relaxes an infeasible approximate problem (0: none; the optimisers handle an infeasible start "
     "themselves)"},
    {DesignOptimParams::PropertyLimitViolation, "PropertyLimitViolation", Integer, Driver, 0.0, -INF, INF, false, false,
     "a property beyond its limits (DesignPropRelation1.minValue, DesignPropRelation1.maxValue) is a "
     "warning instead of an error when non-zero (0: an error)"},
    {DesignOptimParams::PropertyTolerance, "PropertyTolerance", Real, Driver, INF, 0.0, INF, true, false,
     "largest difference allowed between a property of the bulk data and its value from the design variables "
     "(default: no check)"},
    {DesignOptimParams::StepScale, "StepScale", Real, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "scale of the finite difference steps of shape variables: O_o_Solver's shape sensitivities are semi-analytic "
     "with steps of their own"},
    {DesignOptimParams::TopologyCheckerboard, "TopologyCheckerboard", Integer, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "topology control (O_o_Solver has no topology optimisation)"},
    {DesignOptimParams::TopologyMinMember, "TopologyMinMember", Real, NoEffect, NO_EFFECT, -INF, INF, false, false,
     "topology control (O_o_Solver has no topology optimisation)"},
    {DesignOptimParams::TopologyRegion, "TopologyRegion", Integer, DefaultOnly, 0.0, 0.0, 1.0, false, false,
     "trust region method (0: none; O_o_Solver has none)"},
    {DesignOptimParams::TrustRegionGrowthFactor, "TrustRegionGrowthFactor", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "factor that enlarges the trust region after a good approximation (O_o_Solver has no trust region)"},
    {DesignOptimParams::TrustRegionShrinkFactor, "TrustRegionShrinkFactor", Real, NoEffect, NO_EFFECT, -INF, INF, false,
     false, "factor that shrinks the trust region after a poor approximation (O_o_Solver has no trust region)"},
}};

// DesignOptimParams METHOD: the values above 3 select other strategies (penalty function methods, and the strategy,
// optimizer and one-dimensional search codes IJK of ADS), which O_o_Solver does not have.
constexpr std::int64_t LAST_METHOD = 3;

std::string upperCase(std::string_view text)
{
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return result;
}

std::string numberText(double value)
{
    std::ostringstream text;
    text << value;
    return text.str();
}

std::string knownNames()
{
    std::string names;
    for (const ParameterDefinition& row : DEFINITIONS)
        names.append(names.empty() ? "" : ", ").append(row.name);
    return names;
}

[[noreturn]] void refuse(std::string_view name, std::string_view expected, std::string_view found,
                         std::string_view todo)
{
    // The remedy given by each caller: `todo`.
    throw base::OptimisationError(
        base::concatenate({"DesignOptimParams ", name, ": expected ", expected, ", found ", found, "; ", todo}));
}

std::string rangeText(const ParameterDefinition& row)
{
    std::string text;
    if (std::isfinite(row.minimum))
        text.append(row.isMinimumExcluded ? "> " : ">= ").append(numberText(row.minimum));
    if (std::isfinite(row.maximum))
        text.append(text.empty() ? "" : " and ")
            .append(row.isMaximumExcluded ? "< " : "<= ")
            .append(numberText(row.maximum));
    return text.empty() ? std::string("any value") : text;
}

} // namespace

OptimisationParameters::OptimisationParameters()
{
    for (const ParameterDefinition& row : DEFINITIONS)
        m_values[static_cast<std::size_t>(row.parameter)] = row.defaultValue;
}

OptimisationParameters OptimisationParameters::fromEntries(
    std::span<const std::pair<std::string, ParameterValue>> entries)
{
    OptimisationParameters parameters;
    std::set<DesignOptimParams> given;
    for (const auto& [name, value] : entries)
    {
        if (const std::optional<DesignOptimParams> parameter = find(name);
            parameter && !given.insert(*parameter).second)
            refuse(definition(*parameter).name, "each parameter once (a deck has one DesignOptimParams entry)",
                   "it twice", "keep one value");
        parameters.set(name, value);
    }
    return parameters;
}

void OptimisationParameters::set(std::string_view name, const ParameterValue& value)
{
    const std::optional<DesignOptimParams> parameter = find(name);
    if (!parameter)
        refuse(name, base::concatenate({"a parameter of DesignOptimParams (", knownNames(), ")"}), "an unknown name",
               "correct the name or remove the parameter");
    const ParameterDefinition& row = definition(*parameter);
    const auto index = static_cast<std::size_t>(*parameter);
    if (row.kind == ParameterKind::Text)
    {
        const std::string* text = std::get_if<std::string>(&value);
        if (text == nullptr)
            refuse(row.name, "a character value", "a number", "give a text");
        // OPTCOD names an optimiser library: accepted with any code and without effect (ParameterUse::NoEffect), the
        // optimiser being the one of METHOD.
        const std::string code = upperCase(*text);
        m_optimizerChoice = code;
        m_isSet[index] = true;
        return;
    }
    double number = 0.0;
    if (const std::int64_t* integer = std::get_if<std::int64_t>(&value))
        number = static_cast<double>(*integer);
    else if (const double* real = std::get_if<double>(&value))
    {
        if (row.kind == ParameterKind::Integer)
            refuse(row.name, "an integer", base::concatenate({"the real ", numberText(*real)}),
                   "write the value without a decimal point");
        number = *real;
    }
    else
        refuse(row.name, row.kind == ParameterKind::Integer ? "an integer" : "a real", "a text", "give a number");
    if (!std::isfinite(number))
        refuse(row.name, "a finite number", numberText(number), "give a finite value");
    const bool isBelow = row.isMinimumExcluded ? number <= row.minimum : number < row.minimum;
    const bool isAbove = row.isMaximumExcluded ? number >= row.maximum : number > row.maximum;
    if (*parameter == DesignOptimParams::OptimizationMethod && number > static_cast<double>(LAST_METHOD))
        refuse(row.name, "0 (automatic), 1 (MFD), 2 (SLP) or 3 (SQP)", numberText(number),
               "the penalty function strategies and the three-digit strategy codes are not available in O_o_Solver; "
               "use 0 to 3");
    if (isBelow || isAbove)
        refuse(row.name, base::concatenate({"a value ", rangeText(row)}), numberText(number),
               "give a value in that range");
    if (row.use == ParameterUse::DefaultOnly && number != row.defaultValue)
        refuse(row.name, base::concatenate({"its default ", numberText(row.defaultValue)}), numberText(number),
               base::concatenate({"O_o_Solver does not support another value (", row.meaning,
                                  "); remove the parameter from DesignOptimParams"}));
    m_values[index] = number;
    m_isSet[index] = true;
}

double OptimisationParameters::value(DesignOptimParams parameter) const noexcept
{
    return m_values[static_cast<std::size_t>(parameter)];
}

std::int64_t OptimisationParameters::integer(DesignOptimParams parameter) const noexcept
{
    return static_cast<std::int64_t>(m_values[static_cast<std::size_t>(parameter)]);
}

bool OptimisationParameters::isSet(DesignOptimParams parameter) const noexcept
{
    return m_isSet[static_cast<std::size_t>(parameter)];
}

double OptimisationParameters::absoluteObjectiveChange(double initialObjective) const noexcept
{
    if (isSet(DesignOptimParams::AbsoluteObjectiveChange))
        return value(DesignOptimParams::AbsoluteObjectiveChange);
    return defaults::absoluteChangeOf(initialObjective);
}

double OptimisationParameters::absoluteObjectiveConvergence(double initialObjective) const noexcept
{
    if (isSet(DesignOptimParams::AbsoluteObjectiveConvergence))
        return value(DesignOptimParams::AbsoluteObjectiveConvergence);
    return defaults::absoluteChangeOf(initialObjective);
}

std::span<const ParameterDefinition> OptimisationParameters::definitions() noexcept
{
    return DEFINITIONS;
}

const ParameterDefinition& OptimisationParameters::definition(DesignOptimParams parameter) noexcept
{
    return DEFINITIONS[static_cast<std::size_t>(parameter)];
}

std::optional<DesignOptimParams> OptimisationParameters::find(std::string_view name) noexcept
{
    // The neutral name of the dictionary (designParametersTable of O_o_Reader): "MaxDesignCycles" for DESMAX. NASPR0,
    // a spelling with a zero found in some decks, is an alias of NASPRO in the dictionary: PrintAnalysisEveryCycle too.
    for (const ParameterDefinition& row : DEFINITIONS)
        if (row.name == name)
            return row.parameter;
    return std::nullopt;
}

std::vector<std::string_view> OptimisationParameters::givenNames() const
{
    std::vector<std::string_view> names;
    for (const ParameterDefinition& row : DEFINITIONS)
        if (isSet(row.parameter))
            names.push_back(row.name);
    return names;
}

std::vector<std::string_view> OptimisationParameters::ignoredNames() const
{
    std::vector<std::string_view> names;
    for (const ParameterDefinition& row : DEFINITIONS)
        if (isSet(row.parameter) && row.use == ParameterUse::NoEffect)
            names.push_back(row.name);
    return names;
}

std::string_view parameterUseName(ParameterUse use) noexcept
{
    switch (use)
    {
    case ParameterUse::Optimiser:
        return "optimiser";
    case ParameterUse::Driver:
        return "driver";
    case ParameterUse::NoEffect:
        return "no effect";
    case ParameterUse::DefaultOnly:
        return "default only";
    }
    return "";
}

} // namespace oos::design
