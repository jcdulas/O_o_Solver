#pragma once

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>

namespace oos::design
{

// The status of a design point evaluated with its failures captured (a failure does not stop an optimisation loop):
// a failure of the analysis at that point gives NaN values and one of these, never an exception, so that an
// optimiser goes on with its next point.
// - Ok: the analysis and the responses were computed;
// - Singular: a mechanism or a singularity (SingularityError: a zero or too small pivot, MAXRATIO exceeded);
// - Nonlinear: the point leaves the linear domain (the linearity indicators; not raised yet);
// - Budget: the memory budget or an output volume was exhausted (MemoryBudgetError, OutputVolumeError, an allocation
//   the system refused);
// - Failed: any other failure the point causes (a property the design point makes invalid, an equation outside its
//   domain, a callable result of the wrong shape or not finite: SolverError, UnsupportedCardError, OptimisationError);
// - Partial: items of an external response failed or were not evaluated, the others kept.
enum class EvaluationStatus : std::uint8_t
{
    Ok,
    Singular,
    Nonlinear,
    Budget,
    Failed,
    Partial
};

// "ok", "singular", "nonlinear", "budget", "failed", "partial".
[[nodiscard]] std::string_view evaluationStatusName(EvaluationStatus status) noexcept;

// A failure of one design point, as a status.
struct PointFailure
{
    EvaluationStatus status = EvaluationStatus::Failed;
    std::string message;
};

// The status of the exception `failure` when it is a failure a design point can cause, or nothing when it must
// propagate even when failures are captured: a cancellation (OperationCancelledError), a misuse or a missing
// table (ContractError, FileError, CacheError, std::invalid_argument, std::logic_error), an internal error, an
// exception of a callable of the caller (it is the caller's code) — anything that is not a base::SolverError or
// std::bad_alloc.
//
// Thread-safety: stateless. Precondition: `failure` is not null.
[[nodiscard]] std::optional<PointFailure> classifyFailure(const std::exception_ptr& failure);

} // namespace oos::design
