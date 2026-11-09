#pragma once

#include "oos/design/approximation/ResponseApproximation.hpp"
#include "oos/design/optimiser/OptimisationParameters.hpp"
#include "oos/design/optimiser/OptimiserDefaults.hpp"

#include <optional>

namespace oos::design
{

// The move limits of a design cycle (DOPTPRM DELP, DPMIN, DELX, DXMIN; DESVAR DELXV); `isEnabled` false keeps only
// the bounds of the variables and the PMIN, PMAX of the properties (the caller's approximate(move_limits=False)). The
// defaults are those of ADS (OptimiserDefaults.hpp): a relative move limit RMVLMZ of 0.4 as its convex linearisation
// strategy takes (the approximations in reciprocal variables, APRCOD 2 and 3), 0.2 as its sequential linear
// programming takes (the direct linearisation, APRCOD 1), and O_o_Solver's floor of 0.001 (ADS SCLMIN).
struct MoveLimitParameters
{
    bool isEnabled = true;
    double designVarMoveLimit = optimiser_defaults::CONVEX_MOVE_LIMIT;
    double minDesignVarChange = optimiser_defaults::MINIMUM_SCALE;
    double propertyMoveLimit = optimiser_defaults::CONVEX_MOVE_LIMIT;
    double minPropertyChange = optimiser_defaults::MINIMUM_SCALE;

    // DELX, DXMIN, DELP, DPMIN of a DesignOptimParams; DELX and DELP not given take the default of its APRCOD.
    [[nodiscard]] static MoveLimitParameters fromParameters(const OptimisationParameters& parameters);
    // The default relative move limit (DELX, DELP) of an approximation: 0.2 for the direct linearisation, 0.4 for the
    // others.
    [[nodiscard]] static double defaultMoveLimit(ApproximationKind kind) noexcept;
};

// An interval of a design variable or property in the approximate problem (infinite sides allowed for properties).
struct Range
{
    double lower = 0.0;
    double upper = 0.0;
};

// The move limits of a design cycle (DOPTPRM, DESVAR), the move limits of the approximation concepts (Schmit and
// Miura, "Approximation concepts for efficient structural synthesis", NASA CR-2552, 1976: each approximate problem
// is solved within a fraction of the current design, as ADS's sequential strategies do):
//   a design variable moves by at most Δx = max(DELXV·|x0|, DXMIN) (DELXV of its DESVAR, DesignOptimParams DELX
//   otherwise) and stays within [XLB, XUB]; a designed property moves by at most Δp = max(DELP·|p0|, DPMIN) and stays
//   within [PMIN, PMAX] of its relation.
// The property limits become, in the approximate problem, a cut of the bounds of the variable that drives a property
// linearly and alone (DVPREL1 of one variable), and constraints on the linearised property otherwise
// (ApproximateProblem).
//
// Thread-safety: immutable.
class MoveLimits
{
public:
    explicit MoveLimits(MoveLimitParameters parameters) noexcept : m_parameters(parameters) {}

    [[nodiscard]] const MoveLimitParameters& parameters() const noexcept { return m_parameters; }
    // [max(XLB, x0 − Δx), min(XUB, x0 + Δx)]; [XLB, XUB] without move limits.
    [[nodiscard]] Range variableRange(double x0, double xlb, double xub, std::optional<double> delxv) const noexcept;
    // [p0 − Δp, p0 + Δp] ∩ [PMIN, PMAX]; [PMIN, PMAX] (±∞ when blank) without move limits.
    [[nodiscard]] Range propertyRange(double p0, std::optional<double> pmin, std::optional<double> pmax) const noexcept;

private:
    MoveLimitParameters m_parameters;
};

} // namespace oos::design
