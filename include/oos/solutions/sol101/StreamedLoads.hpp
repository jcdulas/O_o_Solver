#pragma once

#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"
#include "oos/solutions/sol101/StreamedElements.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::solutions
{

// The vectors of the load basis as the full path streamed takes them: never a vector of the model, each
// basis vector's loads made at the fronts from what defines them:
// - concentrated loads (FORCE, MOMENT, ScalarPointLoad, a unit load of the loaded DOFs' basis): their values on their
// DOFs,
//   carried through Gm onto equations, each term placed at the supernode of its equation;
// - pressures on elements (PLOAD2, FacePressureLoad): the rows of pressureTable of each element, the basis vectors that
// take a
//   row's load set with their factors (a set as written, the set of a case, through the terms of its LOAD), the
//   element's equivalent nodal loads made at its front by its family's pressure facet, once per row, times each
//   factor;
// - inertial loads of GravityLoad (the inertial family by its generators): each vector's acceleration in the basic
//   system, the element's loads its mass (WTMASS applied) times that uniform translation, at its front;
// - inertia relief on rigid support motions (setRelief): each vector's rigid-body amplitudes w (P' = P − M·Φ·w),
//   the element's part −M_e·Φ_e·w at its front.
// What the path does not take is reported as an error naming the remedy: line loads (PLOAD1), pressures on the faces
// of grid points (PLOAD without element), ACCEL, AccelNodeLoad, RotationalForceLoad, thermal loads and enforced
// displacements.
//
// Thread-safety: immutable once built (setRelief before any use); const methods are safe concurrently. Ownership:
// owns its descriptions (budgeted); borrows the load cases, the load builder and the placement, which outlive it.
class StreamedLoads
{
public:
    struct Inputs
    {
        const StaticLoadCases* cases = nullptr;
        const assembly::LoadVectorBuilder* loads = nullptr;
        const StreamedElements* elements = nullptr;
        // The solved subcases of the constraint set, in the order of the load cases (a vector of a case's family part
        // takes its LOAD set).
        std::span<const StaticSubcase* const> subcases;
    };
    // A load term of a vector on an equation, placed at the supernode of that equation.
    struct EquationLoad
    {
        std::int32_t equation = 0;
        std::int32_t column = 0;
        double value = 0.0;
    };
    // A basis vector that takes the rows of a load set, with its factor.
    struct Consumer
    {
        std::int32_t column = 0;
        double factor = 1.0;
    };
    // A pressure row of an element and the consumers of its set (an index into consumerOffsets()).
    struct ElementPressureRow
    {
        std::int64_t row = 0;
        std::int32_t consumers = 0;
    };

    // Reports to `sink` (errors) the loads the path does not take. Throws MemoryBudgetError.
    StreamedLoads(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);

    [[nodiscard]] std::size_t columnCount() const noexcept { return m_columnCount; }
    // Whether an element's mass makes loads (an inertial vector, or inertia relief).
    [[nodiscard]] bool usesMass() const noexcept { return m_hasInertial || m_hasRelief; }
    [[nodiscard]] bool hasPressures() const noexcept { return m_pressureRows.size() > 0; }
    [[nodiscard]] std::int64_t pressureRowCount() const noexcept
    {
        return static_cast<std::int64_t>(m_pressureRows.size());
    }

    // A concentrated load of a vector on its g-set DOF, as the cards give it (the resultants of inertia relief).
    struct DofLoad
    {
        base::DofIndex dof = 0;
        std::int32_t column = 0;
        double value = 0.0;
    };
    [[nodiscard]] std::span<const DofLoad> concentratedLoads() const noexcept
    {
        return {m_dofLoads.data(), m_dofLoads.size()};
    }
    // The concentrated terms placed at supernode s, in the order of the vectors.
    [[nodiscard]] std::span<const EquationLoad> concentratedAt(std::int32_t s) const noexcept;
    // The pressure rows of an element (its index in the placement), in table order.
    [[nodiscard]] std::span<const ElementPressureRow> pressuresOf(std::int32_t element) const noexcept;
    // The consumers of a row's set.
    [[nodiscard]] std::span<const Consumer> consumers(std::int32_t index) const noexcept;
    // The acceleration of each vector in the basic system (zero for a vector of another family).
    [[nodiscard]] std::span<const std::array<double, 3>> accelerations() const noexcept
    {
        return {m_accelerations.data(), m_accelerations.size()};
    }
    // Inertia relief (on rigid support motions): each vector's amplitudes of the rigid-body motions about
    // the basic origin, P' = P − M·Φ·w (RigidBodyModes' convention).
    void setRelief(std::vector<std::array<double, 6>> amplitudes);
    [[nodiscard]] bool hasRelief() const noexcept { return m_hasRelief; }
    [[nodiscard]] std::span<const std::array<double, 6>> reliefAmplitudes() const noexcept
    {
        return {m_relief.data(), m_relief.size()};
    }
    // The bytes of its descriptions, and an upper bound of them before they are made (the pressure rows of the model).
    [[nodiscard]] std::uint64_t bytes() const noexcept;
    [[nodiscard]] static std::uint64_t estimateBytes(std::int64_t pressureRows, std::int64_t elements,
                                                     std::int64_t supernodes, std::size_t columns) noexcept;

private:
    std::size_t m_columnCount = 0;
    bool m_hasInertial = false;
    bool m_hasRelief = false;
    base::BudgetedBuffer<std::int64_t> m_concentratedOffsets; // supernodes + 1
    base::BudgetedBuffer<EquationLoad> m_concentrated;
    base::BudgetedBuffer<std::int64_t> m_pressureOffsets; // elements + 1 (empty without pressures)
    base::BudgetedBuffer<ElementPressureRow> m_pressureRows;
    std::vector<std::int32_t> m_consumerOffsets; // per consumed set + 1 (bounded by the load sets)
    std::vector<Consumer> m_consumerList;
    std::vector<DofLoad> m_dofLoads;                    // bounded by the point loads of the basis
    std::vector<std::array<double, 3>> m_accelerations; // per vector (bounded by the basis)
    std::vector<std::array<double, 6>> m_relief;        // per vector
};

} // namespace oos::solutions
