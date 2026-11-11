#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/design/responses/DesignResp1Response.hpp"
#include "oos/design/sensitivity/AffectedElements.hpp"
#include "oos/design/sensitivity/ElementDerivatives.hpp"
#include "oos/design/sensitivity/ElementItemFunction.hpp"
#include "oos/design/sensitivity/LazyFieldPerturbation.hpp"
#include "oos/design/sensitivity/ResponseLinearisation.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/solutions/sol103/ModesSolution.hpp"

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace oos::design::detail
{

// Vibration field responses of a simple mode. Their complete gradients become explicit rows in the response
// linearisation, so equations, gradient streaming and Jacobian products use the same modal derivatives.
// One bordered adjoint per field item replaces a perturbed eigensolution for every design variable.
class ModalItems
{
public:
    struct Inputs
    {
        std::shared_ptr<const input::ModelSnapshot> snapshot;
        const solutions::ModesResults* results;
        const assembly::ElementSet* set;
        const assembly::DofMap* dofMap;
        linalg::KernelKind kernel;
        base::ThreadPool* pool;
        base::MemoryBudget* budget;
    };

    explicit ModalItems(Inputs inputs);
    [[nodiscard]] bool isEmpty() const noexcept { return m_items.empty(); }
    void add(const ItemKey& key, const DesignResp1Response& response, const ResponseItem& label, double value);
    void prepare();

    using MassOf = std::function<ElementMatrixDerivative(const elements::MassOptions&)>;
    void addElement(std::size_t quantity, const ElementRef& element, const elements::FieldUse& use,
                    const ElementMatrixDerivative& stiffness, const MassOf& massOf,
                    const elements::DesignedField* field, const LazyFieldPerturbation& perturbation);
    void addExplicitParts(const std::vector<FieldValue>& quantities, LinearisedItems& linearised) const;

    [[nodiscard]] std::int64_t solveCount() const noexcept { return static_cast<std::int64_t>(m_items.size()); }
    [[nodiscard]] std::int64_t factorisationCount() const noexcept { return static_cast<std::int64_t>(m_modes.size()); }

private:
    struct Mode
    {
        std::size_t solve;
        std::size_t column;
        elements::MassOptions massOptions;
        double weightToMass = 1.0;
    };
    struct Item
    {
        ItemKey key;
        ResponseItem label;
        std::size_t mode;
        double value;
        std::optional<ElementRef> element;
        std::unique_ptr<ElementItemFunction> function;
        base::BudgetedBuffer<double> adjoint;
        double multiplier = 0.0;
        std::map<std::size_t, double> byQuantity;
    };

    [[nodiscard]] std::span<const double> shapeOf(const Mode& mode) const;
    [[nodiscard]] elements::NodalDisplacements nodal(std::span<const double> vector) const;
    [[nodiscard]] double product(const ElementMatrixDerivative& matrix, std::span<const double> left,
                                  std::span<const double> right) const;
    [[nodiscard]] double explicitRate(Item& item, const ElementRef& element, const elements::DesignedField* field,
                                       const LazyFieldPerturbation& perturbation) const;

    Inputs m_in;
    linalg::DenseKernels m_kernels;
    std::vector<Mode> m_modes;
    std::vector<Item> m_items;
};

} // namespace oos::design::detail
