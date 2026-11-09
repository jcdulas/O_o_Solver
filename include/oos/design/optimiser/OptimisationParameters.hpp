#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace oos::design
{

// The parameters of the DesignOptimParams entry of the input format.
enum class DesignOptimParams : std::uint8_t
{
    ApproximationMethod,
    AutomaticSuperelementCreation,
    RelativeObjectiveConvergence,
    AbsoluteObjectiveConvergence,
    DesignVarConvergence,
    PropertyConvergence,
    ConstraintActiveThreshold,
    ConstraintViolationThreshold,
    AbsoluteObjectiveChange,
    FiniteDifferenceStep,
    RelativeObjectiveChange,
    PropertyMoveLimit,
    DesignVarMoveLimit,
    EquivalentLoadMoveLimit,
    MaxDesignCycles,
    DiscreteMethod,
    DiscreteStartCycle,
    MaxPropertyChange,
    MinPropertyChange,
    SuperelementStopRatio,
    EquivalentLoadMaxCycles,
    MaxDesignVarChange,
    MinDesignVarChange,
    TrustRegionLowRatio,
    TrustRegionMiddleRatio,
    TrustRegionHighRatio,
    FullyStressedRelaxation,
    FullyStressedMaxCycles,
    MaxConstraintViolation,
    ConstraintScale,
    ViolationTreatment,
    OptimizerPrintLevel,
    DesignVarScaling,
    OptimizationMethod,
    PrintAnalysisEveryCycle,
    ObjectiveRelativeToStart,
    OptimizerName,
    PrintInterval,
    PrintItems,
    PrintMaxRetained,
    PrintBeadConstraints,
    PrintConstraintCount,
    PrintDiscreteVars,
    PrintMaterialCount,
    PrintPropertyCount,
    PrintResponseCount,
    PrintResponseSet,
    PrintApproximation,
    PenaltyFactor,
    PropertyLimitViolation,
    PropertyTolerance,
    StepScale,
    TopologyCheckerboard,
    TopologyMinMember,
    TopologyRegion,
    TrustRegionGrowthFactor,
    TrustRegionShrinkFactor,
    Count
};

inline constexpr std::size_t DESIGN_OPTIM_PARAMS_COUNT = static_cast<std::size_t>(DesignOptimParams::Count);

// The type of a DesignOptimParams value: integer, real or character.
enum class ParameterKind : std::uint8_t
{
    Integer,
    Real,
    Text
};

// What O_o_Solver does with a DesignOptimParams parameter. The keywords are those of the input format; the default and
// the valid range O_o_Solver applies to each come from the public sources of OptimiserDefaults.hpp (each keyword mapped
// to the ADS or CONMIN parameter whose role it plays) or are a documented choice of this project.
enum class ParameterUse : std::uint8_t
{
    Optimiser,  // read by the approximate problem, the optimisers or the convergence tests (design/)
    Driver,     // read by the SOL 200 driver: the property checks (PTOL, PLVIOL)
    NoEffect,   // accepted with any value of its type and without effect: a print control of a printed listing
                // O_o_Solver does not write, a code naming an optimiser library, a capability a linear sizing run
                // never enters, or a convergence control the public criteria have no place for
                // (OptimisationParameters::ignoredNames lists the ones a deck gives, for the driver to report)
    DefaultOnly // a capability O_o_Solver does not have: its default (the value that turns it off) is accepted, any
                // other value is refused
};

// One DesignOptimParams parameter as O_o_Solver applies it: the name, the type, the default, the valid range and what
// O_o_Solver does with it. `defaultValue` is NaN when the default depends on the run (DABOBJ and CONV2 from the
// initial objective, DELX and DELP from APRCOD), when the parameter has no effect, or when it is text (OPTCOD). A
// NoEffect parameter takes any value of its type (its range is ±∞).
struct ParameterDefinition
{
    DesignOptimParams parameter = DesignOptimParams::ApproximationMethod;
    std::string_view name; // the neutral name of the dictionary (designParametersTable): "MaxDesignCycles" for DESMAX
    ParameterKind kind = ParameterKind::Real;
    ParameterUse use = ParameterUse::Optimiser;
    double defaultValue = 0.0;
    double minimum = 0.0; // the valid range, the bounds excluded when the flags say so
    double maximum = 0.0;
    bool isMinimumExcluded = false;
    bool isMaximumExcluded = false;
    std::string_view meaning; // what the parameter does in O_o_Solver, with the source of its default
};

// A value of a DesignOptimParams entry as the data contract gives it (designParametersTable): an integer, a real or a
// text.
using ParameterValue = std::variant<std::int64_t, double, std::string>;

// The DesignOptimParams parameters of a design problem: the default of each, overridden by the values a deck or a
// program gives. Every value given is checked when it is set (its type, its valid range, and for the DefaultOnly
// parameters that it is the default): an invalid or unsupported value throws OptimisationError naming the parameter,
// what was expected, what was found and what to do. An unknown name is refused with the list of the known ones.
//
// Thread-safety: immutable once built (set() only while it is being built). Ownership: owns its values.
class OptimisationParameters
{
public:
    // Every parameter at its default (OptimiserDefaults.hpp).
    OptimisationParameters();

    // The parameters of a DesignOptimParams entry given as (name, value) pairs; a deck has one DesignOptimParams entry,
    // so a name given twice is refused. Throws OptimisationError (above).
    [[nodiscard]] static OptimisationParameters fromEntries(
        std::span<const std::pair<std::string, ParameterValue>> entries);

    // Sets one parameter by its neutral name, exactly ("MaxDesignCycles"). Throws OptimisationError (above).
    void set(std::string_view name, const ParameterValue& value);

    [[nodiscard]] double value(DesignOptimParams parameter) const noexcept;
    [[nodiscard]] std::int64_t integer(DesignOptimParams parameter) const noexcept;
    [[nodiscard]] bool isSet(DesignOptimParams parameter) const noexcept;
    [[nodiscard]] const std::string& optimizerChoice() const noexcept { return m_optimizerChoice; }

    // DABOBJ: the value given, or the default of ADS, |F0|/1000 with F0 the initial objective (at least 1.0E-10, as
    // CONMIN's DABFUN).
    [[nodiscard]] double absoluteObjectiveChange(double initialObjective) const noexcept;
    // CONV2: the value given, or the default of ADS's DABSTR, the same |F0|/1000 (at least 1.0E-10).
    [[nodiscard]] double absoluteObjectiveConvergence(double initialObjective) const noexcept;

    // Every parameter in order; the definition of one parameter; the definition of a name (nothing when unknown).
    [[nodiscard]] static std::span<const ParameterDefinition> definitions() noexcept;
    [[nodiscard]] static const ParameterDefinition& definition(DesignOptimParams parameter) noexcept;
    [[nodiscard]] static std::optional<DesignOptimParams> find(std::string_view name) noexcept;

    // The names of the parameters given (in DesignOptimParams order).
    [[nodiscard]] std::vector<std::string_view> givenNames() const;
    // The names of the parameters given that have no effect (ParameterUse::NoEffect, in DesignOptimParams order): what
    // a driver reports so that a deck's author knows they were accepted and not used.
    [[nodiscard]] std::vector<std::string_view> ignoredNames() const;

private:
    std::array<double, DESIGN_OPTIM_PARAMS_COUNT> m_values{};
    std::array<bool, DESIGN_OPTIM_PARAMS_COUNT> m_isSet{};
    std::string m_optimizerChoice;
};

// The name of a use ("optimiser", "driver", "no effect", "default only").
[[nodiscard]] std::string_view parameterUseName(ParameterUse use) noexcept;

} // namespace oos::design
