#pragma once

#include "oos/elements/bars_beams/BeamFieldDerivatives.hpp"
#include "oos/elements/bars_beams/BeamFlexibility.hpp"
#include "oos/elements/core/MixedScalar.hpp"

#include <optional>

namespace oos::elements::beam_sizing
{

struct Direction
{
    BeamSection section;
    IsotropicMaterial material;
};

[[nodiscard]] Direction zeroDirection(const BeamData& beam);
[[nodiscard]] std::optional<Direction> directionOf(const BeamData& beam, const ElementBatchView& view,
                                                  std::int64_t row, const DesignedField& field);

template <std::size_t R, std::size_t C = R>
struct Matrix
{
    std::array<MixedScalar, R * C> values{};
    MixedScalar& operator()(std::size_t i, std::size_t j) { return values[i * C + j]; }
    const MixedScalar& operator()(std::size_t i, std::size_t j) const { return values[i * C + j]; }
};

template <std::size_t R, std::size_t K, std::size_t C>
Matrix<R, C> product(const Matrix<R, K>& a, const Matrix<K, C>& b)
{
    Matrix<R, C> result;
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < C; ++j)
                result(i, j) += a(i, k) * b(k, j);
    return result;
}

template <std::size_t R, std::size_t C>
Matrix<C, R> transpose(const Matrix<R, C>& a)
{
    Matrix<C, R> result;
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            result(j, i) = a(i, j);
    return result;
}

template <std::size_t N>
Matrix<N> congruence(const Matrix<N>& a, const Matrix<N>& map)
{
    return product(transpose(map), product(a, map));
}

template <std::size_t R, std::size_t C>
Matrix<R, C> constant(const base::MatN<R, C>& a)
{
    Matrix<R, C> result;
    for (std::size_t k = 0; k < a.values.size(); ++k)
        result.values[k] = a.values[k];
    return result;
}

struct Station
{
    MixedScalar area, inertia1, inertia2, productInertia, torsion, nsm;
    std::array<MixedScalar, 2> neutral, centre;
    std::array<std::array<MixedScalar, 2>, 4> points;
};

[[nodiscard]] Station stationOf(const BeamStation& value, const BeamStation& first, const BeamStation& other);
[[nodiscard]] Matrix<12> massOf(const BeamData& beam, const Direction& first, const Direction& other,
                                double addedMass, MassFormulation formulation, TorsionalInertia torsion,
                                bool onOffsets);

struct RecoveredRow
{
    ResultPlace place;
    std::array<MixedScalar, 15> values{};
    std::size_t count = 0;
};

// k = F^-1: k_p = -k F_p k; k_pq = k F_q k F_p k + k F_p k F_q k - k F_pq k.
// The same chain rule differentiates the released-end map, geometric stiffness and recovered linear columns.
class Kernel
{
public:
    Kernel(const BeamData& beam, const Direction& first, const Direction& other);
    [[nodiscard]] Matrix<12> stiffness() const;
    [[nodiscard]] Matrix<12> differential(std::span<const double> grid, MixedScalar thermalStrain = {}) const;
    [[nodiscard]] std::array<MixedScalar, 12> loads(MixedScalar thermalStrain,
                                                  std::span<const BeamLineLoad> lineLoads, bool onOffsets) const;
    [[nodiscard]] std::vector<RecoveredRow> recover(const ElementBatchView& view, ElementOutput output,
                                                    std::span<const double> grid, MixedScalar thermalStrain = {},
                                                    std::span<const BeamLineLoad> lineLoads = {}) const;

private:
    void release(const Matrix<12>& raw);
    [[nodiscard]] Matrix<6> complianceAt(double position) const;
    [[nodiscard]] std::array<MixedScalar, 12> endForces(std::span<const double> grid,
                                                     MixedScalar thermalStrain = {},
                                                     std::span<const BeamLineLoad> lineLoads = {}) const;
    [[nodiscard]] std::array<MixedScalar, 12> lineEndLoads(std::span<const BeamLineLoad> loads) const;
    [[nodiscard]] Station at(double position) const;

    const BeamData& m_beam;
    const Direction& m_first;
    const Direction& m_other;
    BeamFlexibility m_flexibility;
    Matrix<6> m_natural;
    Matrix<12> m_end;
    Matrix<12> m_completion;
};

} // namespace oos::elements::beam_sizing
