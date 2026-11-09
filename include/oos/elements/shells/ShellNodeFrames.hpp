#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/core/OffsetTransform.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace oos::elements
{

// The passage between the six DOFs of each grid point of a shell in its displacement system CD and the six local DOFs
// (u, v, w, θx, θy, θn) of the element at that corner, in the element frame (x, y, n) and on the reference plane moved
// by ZOFFS (u_plane = u_i + θ_i × (ZOFFS·n), composed with the transformations):
//
//   q_i = T_i·u_i,  T_i = O_i·diag(Eᵀ·R_i, Eᵀ·R_i),
//
// E the element axes in the basic system (columns x, y, n), R_i the CD-to-basic rotation of grid point i and
// O_i = rigidOffset((0, 0, ZOFFS)) (or rigidOffset(d_i) for an offset d_i of its own at each grid point,
// buildWithNodeOffsets). Local matrices and vectors are ordered node by node, six DOFs per node.
//
// The mass takes T_i under ModelParams OFFDEF LROFF or ELMZ (the mass on the offset) and the rotation alone,
// diag(Eᵀ·R_i, Eᵀ·R_i), under ELMOFF, the default (input::OffsetDefinition: the offset leaves the mass properties
// unchanged).
template <std::size_t N>
struct ShellNodeFrames
{
    std::array<std::int32_t, N> gridRows{};
    std::array<base::MatN<6>, N> toLocal{};
    std::array<base::MatN<6>, N> rotationToLocal{}; // T_i without the offset

    // The frames of the grid points `gridRows` with CD rotations `cdFrames`, for an element with axes `elementAxes`
    // (columns x, y, n in basic) and the offset ZOFFS along n.
    [[nodiscard]] static ShellNodeFrames build(const std::array<std::int32_t, N>& gridRows,
                                               const std::array<base::Mat3, N>& cdFrames, const base::Mat3& elementAxes,
                                               double offset)
    {
        std::array<base::Vec3, N> offsets{};
        offsets.fill(base::Vec3{0.0, 0.0, offset});
        return buildWithNodeOffsets(gridRows, cdFrames, elementAxes, offsets);
    }

    // The same with an offset vector of its own at each grid point, in the element frame: a curved shell offsets each
    // grid point along its own director (ZOFFS·ν_i, HigherOrderShell), a flat one along n for all.
    [[nodiscard]] static ShellNodeFrames buildWithNodeOffsets(const std::array<std::int32_t, N>& gridRows,
                                                              const std::array<base::Mat3, N>& cdFrames,
                                                              const base::Mat3& elementAxes,
                                                              const std::array<base::Vec3, N>& offsets)
    {
        ShellNodeFrames frames;
        frames.gridRows = gridRows;
        for (std::size_t i = 0; i < N; ++i)
        {
            const base::MatN<6> offset = rigidOffset(offsets[i]);
            const base::Mat3 rotation = elementAxes.transposed() * cdFrames[i];
            base::MatN<6> block;
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                {
                    block(r, c) = rotation(r, c);
                    block(3 + r, 3 + c) = rotation(r, c);
                }
            frames.toLocal[i] = offset * block;
            frames.rotationToLocal[i] = block;
        }
        return frames;
    }

    // The 6N DOFs of the grid points, node by node (components 1 to 6).
    [[nodiscard]] std::array<ElementDof, 6 * N> dofs() const
    {
        std::array<ElementDof, 6 * N> result;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t k = 0; k < 6; ++k)
                result[6 * i + k] = ElementDof::grid(gridRows[i], static_cast<int>(k + 1));
        return result;
    }

    // K_CD = Tᵀ·K_local·T, block by block: K_CD(a, b) = T_aᵀ·K_local(a, b)·T_b.
    [[nodiscard]] base::MatN<6 * N> toDisplacementSystems(const base::MatN<6 * N>& local) const
    {
        return carried(local, toLocal);
    }

    // A mass matrix (or its rate) given on the local DOFs, carried to the grid points through the offset when the mass
    // is on it (`isMassOnOffsets`: OFFDEF LROFF, ELMZ), through the rotations alone otherwise (ELMOFF).
    [[nodiscard]] base::MatN<6 * N> massToDisplacementSystems(const base::MatN<6 * N>& local,
                                                              bool isMassOnOffsets) const
    {
        return carried(local, isMassOnOffsets ? toLocal : rotationToLocal);
    }

    // M_CD(a, b) = T_aᵀ·M_local(a, b)·T_b for the transformations `transforms`.
    [[nodiscard]] static base::MatN<6 * N> carried(const base::MatN<6 * N>& local,
                                                   const std::array<base::MatN<6>, N>& transforms)
    {
        base::MatN<6 * N> result;
        for (std::size_t a = 0; a < N; ++a)
            for (std::size_t b = 0; b < N; ++b)
            {
                base::MatN<6> block;
                for (std::size_t r = 0; r < 6; ++r)
                    for (std::size_t c = 0; c < 6; ++c)
                        block(r, c) = local(6 * a + r, 6 * b + c);
                const base::MatN<6> pair = transforms[a].transposed() * block * transforms[b];
                for (std::size_t r = 0; r < 6; ++r)
                    for (std::size_t c = 0; c < 6; ++c)
                        result(6 * a + r, 6 * b + c) = pair(r, c);
            }
        return result;
    }

    // An element load vector (pressure, thermal) given on the local DOFs, carried to the grid points: through the
    // offset when `areLoadsOnOffsets` (MDLPRM OFFDEF LROFF, ELMZ), f_CD,i = T_iᵀ·f_local,i (the work f·q is the same in
    // both); by the rotations alone otherwise (ELMOFF, the default: the offset does not enter the thermal, pressure
    // and gravity loads).
    [[nodiscard]] std::array<double, 6 * N> loadsToDisplacementSystems(const std::array<double, 6 * N>& local,
                                                                       bool areLoadsOnOffsets) const
    {
        const std::array<base::MatN<6>, N>& transforms = areLoadsOnOffsets ? toLocal : rotationToLocal;
        std::array<double, 6 * N> result{};
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t c = 0; c < 6; ++c)
            {
                double sum = 0.0;
                for (std::size_t r = 0; r < 6; ++r)
                    sum += transforms[i](r, c) * local[6 * i + r];
                result[6 * i + c] = sum;
            }
        return result;
    }

    // The local DOFs q of the element from the displacements of the model.
    [[nodiscard]] std::array<double, 6 * N> localDisplacements(const NodalDisplacements& displacements) const
    {
        return localDisplacementsOf([&](const ElementDof& dof) { return displacements.valueAt(dof); });
    }

    // The local displacements under displacements all zero but that of `unit`, one (ElementFamily::
    // recoverAtUnitDisplacements): the bits localDisplacements gives for that displacement vector.
    [[nodiscard]] std::array<double, 6 * N> localUnitDisplacements(const ElementDof& unit) const
    {
        return localDisplacementsOf([&](const ElementDof& dof) { return dof == unit ? 1.0 : 0.0; });
    }

private:
    template <typename ValueAt>
    [[nodiscard]] std::array<double, 6 * N> localDisplacementsOf(ValueAt&& valueAt) const
    {
        std::array<double, 6 * N> result{};
        for (std::size_t i = 0; i < N; ++i)
        {
            std::array<double, 6> u{};
            for (std::size_t k = 0; k < 6; ++k)
                u[k] = valueAt(ElementDof::grid(gridRows[i], static_cast<int>(k + 1)));
            const std::array<double, 6> q = toLocal[i] * u;
            for (std::size_t k = 0; k < 6; ++k)
                result[6 * i + k] = q[k];
        }
        return result;
    }
};

} // namespace oos::elements
