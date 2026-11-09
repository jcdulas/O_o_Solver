#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::reduction
{

class ExternalLoadSource;

// The derivatives of the matrices of an external unit with respect to one independent design variable the plugin uses
// (and their derivatives with respect to the variables the plugin uses): ∂K*/∂x_j
// and ∂M*/∂x_j on the unit's labels. The derivatives of its condensed loads come from its ExternalLoadSource: they
// depend on the loads of the zone, hence on the load basis.
struct ExternalUnitDerivative
{
    std::size_t variable = 0;                // the column of x: the independent variables of the design model, in order
    linalg::DenseMatrix stiffness;           // n × n
    std::optional<linalg::DenseMatrix> mass; // n × n; nothing: M* does not depend on the variable
};

// What a plugin gives for the zone it replaces (the constructor of ExternalUnit takes it whole).
struct ExternalUnitParts
{
    std::int32_t zoneId = 0;
    std::string zoneName;
    // The rows of every matrix, in the plugin's order: grid point and component in the displacement system of the grid
    // point (O_o_Solver's DOF labels: kind Grid, component 1 to 6).
    base::BudgetedBuffer<UnitDofLabel> labels;
    linalg::DenseMatrix stiffness;                   // K*, n × n
    std::optional<linalg::DenseMatrix> mass;         // M*, n × n; nothing: not given
    std::shared_ptr<const ExternalLoadSource> loads; // null: no condensed loads given
    std::vector<ExternalUnitDerivative> derivatives; // bounded by the variables the plugin uses
    // The identity of the plugin (name, version, settings: O_o_Solver.external.identity_text), part of the stamp.
    std::string identity;
};

// An external unit: the matrices a plugin computes on the
// contour DOFs of a zone (its three-dimensional model condensed onto the contour grid points), which replace the
// elements of the zone in the analysis: K* (dense, symmetric), the reduced mass M* (required when the analysis uses
// mass: inertial loads, inertia relief), the source of its condensed loads per basis vector (required when loads apply
// inside the zone), and the derivatives of all three with respect to the design variables the plugin uses.
//
// The constructor checks the structure only: square matrices of one row per label, labels of grid points with a
// component from 1 to 6 and no label twice, finite values, distinct variables. What the matrices mean against the zone
// (its contour DOFs, symmetry, rigid-body motions, semi-definiteness) is checked before use by ExternalUnitChecks. Its
// matrices belong to the plugin and are never cached by O_o_Solver; their hash is the unit's design
// stamp (designStamp()), which the plugin's identity joins.
//
// Thread-safety: immutable. Ownership: owns its matrices (budgeted: the budget must outlive the unit); shares its load
// source.
class ExternalUnit
{
public:
    // Throws std::invalid_argument naming the zone and what is wrong (a size, a label, a value that is not finite).
    explicit ExternalUnit(ExternalUnitParts parts);

    [[nodiscard]] std::int32_t zoneId() const noexcept { return m_parts.zoneId; }
    [[nodiscard]] const std::string& zoneName() const noexcept { return m_parts.zoneName; }
    // The name of the zone as messages print it: its name, or its ID.
    [[nodiscard]] std::string displayName() const;
    [[nodiscard]] std::size_t size() const noexcept { return m_parts.labels.size(); }
    [[nodiscard]] std::span<const UnitDofLabel> labels() const noexcept { return m_parts.labels.span(); }
    // The row of a label, or nothing.
    [[nodiscard]] std::optional<std::size_t> rowOf(const UnitDofLabel& label) const noexcept;
    [[nodiscard]] const linalg::DenseMatrix& stiffness() const noexcept { return m_parts.stiffness; }
    [[nodiscard]] bool hasMass() const noexcept { return m_parts.mass.has_value(); }
    // M*; precondition: hasMass().
    [[nodiscard]] const linalg::DenseMatrix& mass() const { return m_parts.mass.value(); }
    // The source of the condensed loads, or null when none was given.
    [[nodiscard]] const ExternalLoadSource* loads() const noexcept { return m_parts.loads.get(); }
    [[nodiscard]] std::span<const ExternalUnitDerivative> derivatives() const noexcept { return m_parts.derivatives; }
    [[nodiscard]] const std::string& identity() const noexcept { return m_parts.identity; }

    // The design stamp of the unit ("their hash enters the design stamp"): FNV-1a (CacheHasher) over the
    // zone, the plugin's identity, the labels and the bits of K*, M* and of every derivative, in that order. Never
    // EMPTY_DESIGN_STAMP (0), which stands for a unit no variable drives (a hash that falls on it is moved to 1).
    [[nodiscard]] std::uint64_t designStamp() const noexcept { return m_designStamp; }

private:
    [[nodiscard]] std::uint64_t computeDesignStamp() const noexcept;

    ExternalUnitParts m_parts;
    std::vector<std::size_t> m_rowsByLabel; // the rows sorted by label (grid ID, component), for rowOf()
    std::uint64_t m_designStamp = 0;
};

} // namespace oos::reduction
