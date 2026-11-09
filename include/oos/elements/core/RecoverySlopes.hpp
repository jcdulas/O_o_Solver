#pragma once

#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::elements
{

// How a recovered column depends on the displacements of its element:
// - Linear: affine in u, its constant part the thermal strains of the subcase or a value no displacement changes (a
//   fibre distance): value(u) = value(0) + Σ_d slope_d·u_d exactly, the slopes those of RecoverySlopes;
// - PlaneVonMises: the von Mises function of three Linear columns a, b (the normal components) and c (the shear) of the
//   same row, scale·√(a² + b² − a·b + 3·(k·c)²) (ShellRecovery: k = 1 and scale = 1 for the stresses; k = ½ and
//   scale = ⅔ for the strains, the equivalent strain of the result tables), whose rate follows from theirs by the
//   chain rule;
// - Other: any other column (a principal value, an angle, a margin, a failure index), which slopes do not give.
struct ColumnForm
{
    enum class Kind : std::uint8_t
    {
        Other,
        Linear,
        PlaneVonMises
    };
    Kind kind = Kind::Other;
    std::array<std::size_t, 3> components{}; // the columns a, b, c of a PlaneVonMises
    double shearFactor = 1.0;                // k
    double scale = 1.0;

    [[nodiscard]] static constexpr ColumnForm linear() noexcept { return {Kind::Linear, {}, 1.0, 1.0}; }
    [[nodiscard]] static constexpr ColumnForm other() noexcept { return {}; }
    [[nodiscard]] static constexpr ColumnForm planeVonMises(std::array<std::size_t, 3> components, double shearFactor,
                                                            double scale) noexcept
    {
        return {Kind::PlaneVonMises, components, shearFactor, scale};
    }
};

// The rate of a plane von Mises value (ColumnForm::Kind::PlaneVonMises) from the values (a, b, c) of its components
// and their rates: with q = a² + b² − a·b + 3·(k·c)², vm = scale·√q and
//   d(vm) = scale·((2a − b)·da + (2b − a)·db + 6·k²·c·dc) / (2·√q);
// zero at q = 0, where the function has no derivative (a state at rest: the von Mises value is 0 on every side).
[[nodiscard]] double planeVonMisesRate(const ColumnForm& form, const std::array<double, 3>& values,
                                       const std::array<double, 3>& rates) noexcept;

// The second derivative of a plane von Mises value along two directions A and B, from the rates of its
// components along each (`ratesA`, `ratesB`) and their second derivative along both (`second`): with ∇q and the
// constant Hessian H_q = [2 −1 0; −1 2 0; 0 0 6k²] of q,
//   d²(vm) = scale·(d²q/(2·√q) − dq_A·dq_B/(4·q^{3/2})),  d²q = r_Aᵀ·H_q·r_B + ∇q·second,  dq_X = ∇q·r_X;
// zero at q = 0, as the rate.
[[nodiscard]] double planeVonMisesSecondRate(const ColumnForm& form, const std::array<double, 3>& values,
                                             const std::array<double, 3>& ratesA, const std::array<double, 3>& ratesB,
                                             const std::array<double, 3>& second) noexcept;

// The slopes of the rows one element's recovery gives with respect to the displacements of the element:
// for each row of recover() (or recoverLayers()) of the element, in the same order and at the same places, and for
// each column, the change of the value per unit displacement of each DOF of the element,
//   S(r, c, d) = ∂value(r, c)/∂u_d,
// over dofs() (the DOFs of the element's stiffness, in their displacement systems). A Linear column is
// value(u) = value(0) + Σ_d S(r, c, d)·u_d exactly; the slopes of the other columns are zero here (not defined).
//
// They are the recovery matrices of the element (item = S_e·u_e for the items linear in u): the derivatives of its
// items with respect to the displacements without differences (design/sensitivity), and the matrix a vectorised
// recovery may apply to many displacement vectors at once.
//
// Bounded by one element (std::vector). Thread-safety: none for writes; one object per thread.
class RecoverySlopes
{
public:
    // Forgets the rows held; the DOFs of the element (the last index of the slopes) and the number of columns of the
    // rows to come.
    void reset(std::span<const ElementDof> dofs, std::size_t columnCount);
    // Appends a row at `place`, its slopes zero; returns its index.
    std::size_t appendRow(ResultPlace place);

    [[nodiscard]] const std::vector<ElementDof>& dofs() const noexcept { return m_dofs; }
    [[nodiscard]] std::size_t columnCount() const noexcept { return m_columnCount; }
    [[nodiscard]] std::size_t size() const noexcept { return m_places.size(); }
    [[nodiscard]] ResultPlace place(std::size_t row) const { return m_places[row]; }
    // S(row, column, ·) over dofs().
    [[nodiscard]] std::span<double> slopes(std::size_t row, std::size_t column);
    [[nodiscard]] std::span<const double> slopes(std::size_t row, std::size_t column) const;

private:
    std::vector<ElementDof> m_dofs;
    std::size_t m_columnCount = 0;
    std::vector<ResultPlace> m_places;
    std::vector<double> m_values; // rows × columns × DOFs
};

} // namespace oos::elements
