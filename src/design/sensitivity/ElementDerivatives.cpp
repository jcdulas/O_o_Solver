#include "oos/design/sensitivity/ElementDerivatives.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace oos::design
{

namespace
{

constexpr std::array<FieldSide, FIELD_SIDE_COUNT> SIDES{FieldSide::Plus, FieldSide::Minus, FieldSide::HalfPlus,
                                                        FieldSide::HalfMinus};

[[nodiscard]] std::logic_error layoutError(const elements::ElementBatchView& view, std::int64_t row, const char* what)
{
    return std::logic_error(base::concatenate(
        {"ElementDerivatives: the ", what, " of ", view.card(), " ", std::to_string(view.elementId(row)),
         " do not have the same layout on the perturbed snapshots; its derivative cannot be differenced"}));
}

[[nodiscard]] bool sameDofs(std::span<const elements::ElementDof> a, std::span<const elements::ElementDof> b)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
}

} // namespace

ElementMatrixDerivative ElementDerivatives::stiffness(std::int64_t row) const
{
    std::array<elements::ElementMatrixBlock, FIELD_SIDE_COUNT> blocks;
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
        m_family->computeStiffness(view(SIDES[k]), {row, row + 1}, blocks[k]);
    ElementMatrixDerivative derivative;
    if (blocks[0].empty())
        return derivative;
    for (const elements::ElementMatrixBlock& block : blocks)
        if (block.size() != 1 || !sameDofs(block.dofs(0), blocks[0].dofs(0)))
            throw layoutError(view(FieldSide::Plus), row, "stiffness matrices");
    derivative.dofs.assign(blocks[0].dofs(0).begin(), blocks[0].dofs(0).end());
    derivative.values.resize(blocks[0].matrix(0).size());
    m_perturbation->derivatives(blocks[0].matrix(0), blocks[1].matrix(0), blocks[2].matrix(0), blocks[3].matrix(0),
                                derivative.values);
    return derivative;
}

// The derivative of one element matrix from its blocks on the four sides (Plus, Minus, HalfPlus, HalfMinus), which
// must have the same layout; empty when the element appended no matrix.
ElementMatrixDerivative ElementDerivatives::matrixDerivative(
    std::array<elements::ElementMatrixBlock, FIELD_SIDE_COUNT>& blocks, std::int64_t row, const char* what) const
{
    ElementMatrixDerivative derivative;
    if (blocks[0].empty())
        return derivative;
    for (const elements::ElementMatrixBlock& block : blocks)
        if (block.size() != 1 || !sameDofs(block.dofs(0), blocks[0].dofs(0)))
            throw layoutError(view(FieldSide::Plus), row, what);
    derivative.dofs.assign(blocks[0].dofs(0).begin(), blocks[0].dofs(0).end());
    derivative.values.resize(blocks[0].matrix(0).size());
    m_perturbation->derivatives(blocks[0].matrix(0), blocks[1].matrix(0), blocks[2].matrix(0), blocks[3].matrix(0),
                                derivative.values);
    return derivative;
}

ElementMatrixDerivative ElementDerivatives::mass(std::int64_t row, const elements::MassOptions& options) const
{
    std::array<elements::ElementMatrixBlock, FIELD_SIDE_COUNT> blocks;
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
        m_family->computeMass(view(SIDES[k]), {row, row + 1}, options, blocks[k]);
    return matrixDerivative(blocks, row, "mass matrices");
}

ElementMatrixDerivative ElementDerivatives::differentialStiffness(std::int64_t row,
                                                                  const elements::NodalDisplacements& preload) const
{
    std::array<elements::ElementMatrixBlock, FIELD_SIDE_COUNT> blocks;
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
        m_family->computeDifferentialStiffness(view(SIDES[k]), {row, row + 1}, preload, blocks[k]);
    return matrixDerivative(blocks, row, "differential stiffness matrices");
}

ElementVectorDerivative ElementDerivatives::thermalLoads(std::int64_t row,
                                                         const elements::ElementTemperatures& temperatures) const
{
    return vectorDerivative(row, "thermal loads",
                            [&](const elements::ElementBatchView& side, elements::ElementVectorBlock& block)
                            { m_family->computeThermalLoads(side, {row, row + 1}, temperatures, block); });
}

ElementVectorDerivative ElementDerivatives::distributedLoads(std::int64_t row,
                                                             std::span<const elements::ElementPressure> pressures,
                                                             std::span<const elements::ElementLineLoad> lineLoads) const
{
    if (pressures.empty() && lineLoads.empty())
        return {};
    return vectorDerivative(row, "distributed loads",
                            [&](const elements::ElementBatchView& side, elements::ElementVectorBlock& block)
                            {
                                if (!pressures.empty())
                                    m_family->computePressureLoads(side, pressures, block);
                                if (!lineLoads.empty())
                                    m_family->computeLineLoads(side, lineLoads, block);
                            });
}

template <typename Compute>
ElementVectorDerivative ElementDerivatives::vectorDerivative(std::int64_t row, const char* what,
                                                             Compute&& compute) const
{
    // The loads of each side summed DOF by DOF (one entry per element normally), in the order of the plus side.
    std::array<std::vector<double>, FIELD_SIDE_COUNT> sums;
    ElementVectorDerivative derivative;
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
    {
        elements::ElementVectorBlock block;
        compute(view(SIDES[k]), block);
        for (std::size_t i = 0; i < block.size(); ++i)
            for (std::size_t d = 0; d < block.dofs(i).size(); ++d)
            {
                const elements::ElementDof dof = block.dofs(i)[d];
                auto found = std::find(derivative.dofs.begin(), derivative.dofs.end(), dof);
                if (found == derivative.dofs.end())
                {
                    if (k != 0)
                        throw layoutError(view(FieldSide::Plus), row, what);
                    derivative.dofs.push_back(dof);
                    for (std::vector<double>& sum : sums)
                        sum.push_back(0.0);
                    found = derivative.dofs.end() - 1;
                }
                sums[k][static_cast<std::size_t>(found - derivative.dofs.begin())] += block.values(i)[d];
            }
    }
    derivative.values.resize(derivative.dofs.size());
    m_perturbation->derivatives(sums[0], sums[1], sums[2], sums[3], derivative.values);
    return derivative;
}

} // namespace oos::design
