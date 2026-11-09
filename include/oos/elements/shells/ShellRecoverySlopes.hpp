#pragma once

#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/core/RecoverySlopes.hpp"
#include "oos/elements/shells/ShellNodeFrames.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace oos::elements
{

// The slopes of the rows of a shell's recovery (RecoverySlopes), the
// recovery matrices of the element: `appendRows(q, block)` appends to `block` the rows of the element at its local
// displacements q (6N values, node by node) with no temperature change: its place states (ShellQuad4DrillingPlate::placeStates,
// ShellTria3DrillingPlate::centreState) written by appendShellRows or appendLaminateRows, from a plate built once. Those rows
// are linear in q in their Linear columns (`forms`) but for a value no displacement changes (a fibre distance), so the
// slope of local DOF k is
//
//   S_q(·, k) = rows(e_k) − rows(0),
//
// exact (no step and no truncation: the linear map of the formulation evaluated at the unit vectors; rows(0) holds
// only the values that do not depend on q, which the difference removes exactly), and the slopes over the DOFs of the
// grid points follow from q = T·u (ShellNodeFrames: q_i = T_i·u_i): S_u(·, 6i + j) = Σ_k S_q(·, 6i + k)·T_i(k, j). The
// columns that are not Linear get no slopes (zero). The rows are those of recover() for the element, in its order.
template <std::size_t N, typename AppendRows>
void shellRecoverySlopes(const ShellNodeFrames<N>& frames, std::span<const ColumnForm> forms,
                         std::vector<std::string> columns, AppendRows&& appendRows, RecoverySlopes& slopes)
{
    constexpr std::size_t DOFS = 6 * N;
    const std::size_t columnCount = columns.size();
    ElementResultBlock block;
    block.reset(std::move(columns));
    std::array<double, DOFS> q{};
    appendRows(q, block);
    const std::size_t rowCount = block.size();
    for (std::size_t k = 0; k < DOFS; ++k)
    {
        q[k] = 1.0;
        appendRows(q, block);
        q[k] = 0.0;
    }
    const std::array<ElementDof, DOFS> dofs = frames.dofs();
    slopes.reset(dofs, columnCount);
    for (std::size_t r = 0; r < rowCount; ++r)
        slopes.appendRow(block.place(r));
    std::array<double, DOFS> local{};
    for (std::size_t r = 0; r < rowCount; ++r)
    {
        const std::span<const double> atRest = block.values(r);
        for (std::size_t c = 0; c < columnCount && c < forms.size(); ++c)
        {
            if (forms[c].kind != ColumnForm::Kind::Linear)
                continue;
            for (std::size_t k = 0; k < DOFS; ++k)
                local[k] = block.values((k + 1) * rowCount + r)[c] - atRest[c];
            const std::span<double> slope = slopes.slopes(r, c);
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < 6; ++j)
                {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < 6; ++k)
                        sum += local[6 * i + k] * frames.toLocal[i](k, j);
                    slope[6 * i + j] = sum;
                }
        }
    }
}

// The slopes of several outputs of a shell at once (the vectorised recovery): `appendRows(q, blocks)` appends
// to blocks[k] the rows of the k-th output at the local displacements q from one evaluation of the place states, and
// slopes[k] gets the slopes of that output as shellRecoverySlopes gives them (the same rows, differences and frame
// products), the place states evaluated once per unit displacement for every output.
template <std::size_t N, typename AppendRows>
void shellRecoverySlopesOfOutputs(const ShellNodeFrames<N>& frames, std::span<const std::vector<ColumnForm>> forms,
                                  std::span<const std::vector<std::string>> columns, AppendRows&& appendRows,
                                  std::span<RecoverySlopes> slopes)
{
    constexpr std::size_t DOFS = 6 * N;
    const std::size_t outputs = slopes.size();
    std::vector<ElementResultBlock> blocks(outputs); // bounded by the outputs of one element
    for (std::size_t k = 0; k < outputs; ++k)
        blocks[k].reset(columns[k]);
    std::array<double, DOFS> q{};
    appendRows(q, std::span<ElementResultBlock>(blocks));
    std::vector<std::size_t> rowCounts(outputs); // bounded by the outputs
    for (std::size_t k = 0; k < outputs; ++k)
        rowCounts[k] = blocks[k].size();
    for (std::size_t d = 0; d < DOFS; ++d)
    {
        q[d] = 1.0;
        appendRows(q, std::span<ElementResultBlock>(blocks));
        q[d] = 0.0;
    }
    const std::array<ElementDof, DOFS> dofs = frames.dofs();
    std::array<double, DOFS> local{};
    for (std::size_t k = 0; k < outputs; ++k)
    {
        const ElementResultBlock& block = blocks[k];
        const std::size_t rowCount = rowCounts[k];
        const std::size_t columnCount = columns[k].size();
        slopes[k].reset(dofs, columnCount);
        for (std::size_t r = 0; r < rowCount; ++r)
            slopes[k].appendRow(block.place(r));
        for (std::size_t r = 0; r < rowCount; ++r)
        {
            const std::span<const double> atRest = block.values(r);
            for (std::size_t c = 0; c < columnCount && c < forms[k].size(); ++c)
            {
                if (forms[k][c].kind != ColumnForm::Kind::Linear)
                    continue;
                for (std::size_t d = 0; d < DOFS; ++d)
                    local[d] = block.values((d + 1) * rowCount + r)[c] - atRest[c];
                const std::span<double> slope = slopes[k].slopes(r, c);
                for (std::size_t i = 0; i < N; ++i)
                    for (std::size_t j = 0; j < 6; ++j)
                    {
                        double sum = 0.0;
                        for (std::size_t m = 0; m < 6; ++m)
                            sum += local[6 * i + m] * frames.toLocal[i](m, j);
                        slope[6 * i + j] = sum;
                    }
            }
        }
    }
}

} // namespace oos::elements
