#pragma once

#include "oos/design/external/VectorJacobianProducts.hpp"
#include "oos/design/sensitivity/GradientStream.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace oos::api
{

// Tables travel as owned packages between the independent reader and solver memories.
struct OptimizationInput
{
    std::map<std::string, std::vector<std::uint8_t>> tables;
    design::DesignModelParts model;
    design::ResponseParts responses;
    design::ConstraintParts constraints;
    std::map<std::string, std::shared_ptr<design::BatchCallable>> callables;
    std::function<std::vector<design::SuppliedField>(std::span<const double>)> suppliedAt;
};

struct OptimizationSettings
{
    std::uint64_t memoryBytes = 0;
    std::uint64_t maximumOutputBytes = 256ULL * 1024 * 1024;
    std::size_t blockItems = design::EQUATION_BATCH_ITEMS;
    design::DesignAnalyses analyses;
    solutions::StaticOptions analysisOptions;
    bool deckOutputs = false;
};

class OptimizationEvaluation;

// A fixed model and declarations. Rebuild it after changing the reader's model.
// Evaluations share this owner, so deleting the JavaScript problem cannot invalidate their factors or callbacks.
class OptimizationProblem : public std::enable_shared_from_this<OptimizationProblem>
{
public:
    OptimizationProblem(std::shared_ptr<const input::ModelSnapshot> snapshot, OptimizationInput input,
                        OptimizationSettings settings);
    [[nodiscard]] const design::DesignProblem& problem() const noexcept { return m_problem; }
    [[nodiscard]] const OptimizationSettings& settings() const noexcept { return m_settings; }
    [[nodiscard]] design::CallableRegistry callables() const;
    [[nodiscard]] std::vector<design::SuppliedField> supplied(std::span<const double> x) const;
    [[nodiscard]] std::shared_ptr<OptimizationEvaluation> evaluate(std::vector<double> x,
                                                                   const design::ModeReferences& references = {});

private:
    OptimizationInput m_input;
    OptimizationSettings m_settings;
    design::DesignProblem m_problem;
};

// One retained point. Budget precedes the evaluation and linearisation so it outlives their reservations.
// All methods run synchronously on the calling thread; one operation per evaluation at a time.
class OptimizationEvaluation
{
public:
    OptimizationEvaluation(std::shared_ptr<OptimizationProblem> owner, std::vector<double> x,
                           const design::ModeReferences& references = {});
    [[nodiscard]] const design::DesignEvaluation& evaluation() const noexcept { return m_evaluation; }
    [[nodiscard]] const std::vector<double>& x() const noexcept { return m_x; }
    [[nodiscard]] const OptimizationProblem& owner() const noexcept { return *m_owner; }
    [[nodiscard]] design::DesignGradients gradients(const design::GradientOptions& options = {});
    [[nodiscard]] design::GradientStreamReport streamGradients(design::GradientSink& sink, std::size_t block);
    [[nodiscard]] design::ProductReport jvp(std::span<const double> directions, std::size_t count, std::size_t block,
                                            design::ProductSink& sink, const design::ProductSelection& selection = {});
    [[nodiscard]] design::ProductReport vjp(const design::ProductWeights& weights, std::size_t block,
                                            design::ProductSink& sink);
    [[nodiscard]] design::ProductReport hessianVector(const design::ProductWeights& weights,
                                                      std::span<const double> directions, std::size_t count,
                                                      std::vector<double>& values);
    // Dense reduced Hessian of a weighted sum, column by column using the same second-order engine as H*v.
    [[nodiscard]] std::vector<double> hessian(const design::ProductWeights& weights);
    [[nodiscard]] design::DesignBlocks blocks(const design::ProductWeights* weights = nullptr);
    void clearLinearisation() noexcept { m_linearisation.clear(); }
    void requireOutput(std::size_t rows, std::size_t columns) const;

private:
    [[nodiscard]] design::ProductContext context(const design::CallableRegistry& callables) const;
    std::shared_ptr<OptimizationProblem> m_owner;
    std::vector<double> m_x;
    base::MemoryBudget m_budget;
    design::DesignEvaluation m_evaluation;
    design::ProductLinearisation m_linearisation;
};

} // namespace oos::api
