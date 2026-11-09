#include "oos/design/problem/EvaluationStatus.hpp"

#include "oos/base/errors/CacheError.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/FileError.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/errors/OperationCancelledError.hpp"
#include "oos/base/errors/OutputVolumeError.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/errors/SolverError.hpp"

#include <new>

namespace oos::design
{

std::string_view evaluationStatusName(EvaluationStatus status) noexcept
{
    switch (status)
    {
    case EvaluationStatus::Ok:
        return "ok";
    case EvaluationStatus::Singular:
        return "singular";
    case EvaluationStatus::Nonlinear:
        return "nonlinear";
    case EvaluationStatus::Budget:
        return "budget";
    case EvaluationStatus::Failed:
        return "failed";
    case EvaluationStatus::Partial:
        return "partial";
    }
    return "failed";
}

std::optional<PointFailure> classifyFailure(const std::exception_ptr& failure)
{
    // The order of the handlers matters: the subclasses that must propagate come before SolverError.
    try
    {
        std::rethrow_exception(failure);
    }
    catch (const base::OperationCancelledError&)
    {
        return std::nullopt;
    }
    catch (const base::ContractError&)
    {
        return std::nullopt;
    }
    catch (const base::FileError&)
    {
        return std::nullopt;
    }
    catch (const base::CacheError&)
    {
        return std::nullopt;
    }
    catch (const base::SingularityError& error)
    {
        return PointFailure{EvaluationStatus::Singular, error.what()};
    }
    catch (const base::MemoryBudgetError& error)
    {
        return PointFailure{EvaluationStatus::Budget, error.what()};
    }
    catch (const base::OutputVolumeError& error)
    {
        return PointFailure{EvaluationStatus::Budget, error.what()};
    }
    catch (const base::SolverError& error)
    {
        // UnsupportedCardError (a field the point makes invalid), OptimisationError (an equation outside its domain, a
        // callable's result refused), and the other failures of the analysis at this point.
        return PointFailure{EvaluationStatus::Failed, error.what()};
    }
    catch (const std::bad_alloc&)
    {
        return PointFailure{EvaluationStatus::Budget,
                            "the system refused an allocation of the analysis (std::bad_alloc); lower the memory "
                            "budget of the Solver, or the parallel points"};
    }
    catch (...)
    {
        return std::nullopt;
    }
}

} // namespace oos::design
