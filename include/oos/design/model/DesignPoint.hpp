#pragma once

#include "oos/design/model/DesignModel.hpp"
#include "oos/design/model/FieldTargets.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::design
{

// One design point applied to a snapshot without editing the model (evaluations and gradients never edit the model:
// they apply the design point as an overlay of the snapshot, with a design stamp per unit):
// the values of x, of every design variable, of every designed field and of every shape value with their derivatives
// with respect to x, the overlay of the snapshot's columns that holds those values (the field cells, and the positions
// xyzGlobal of the grid points the shape variables move: FieldTargets::overlayOf), and the design stamp of every
// unit.
//
// Thread-safety: immutable once built. Ownership: owns its values; keeps no reference to the model or the targets.
class DesignPoint
{
public:
    // `x`: the independent variables (DesignModel::independentIndices order); `supplied`: the values of the relations
    // given by functions. Throws as DesignModel::evaluate.
    // `reference`: the field values of a design point whose meshes the finite element sections take (empty: those of
    // this point, the analysis'; a moved state of the Hessian-vector products takes its point's).
    DesignPoint(const DesignModel& model, const FieldTargets& targets, std::span<const double> x,
                std::span<const SuppliedField> supplied = {}, std::span<const double> reference = {});

    [[nodiscard]] const std::vector<double>& x() const noexcept { return m_x; }
    // Every design variable's value (the model's order) and every field's value and derivatives (relation order).
    [[nodiscard]] const DesignValues& values() const noexcept { return m_values; }
    [[nodiscard]] const input::ColumnOverlay& overlay() const noexcept { return m_overlay; }
    // The design stamp of each unit, in the order of FieldTargets::unitIds() (empty without unit tables).
    [[nodiscard]] const std::vector<std::uint64_t>& designStamps() const noexcept { return m_designStamps; }

    // The snapshot `base` with this point's field values (ColumnOverlay::applyTo). Precondition: `base` is the
    // snapshot the targets were bound to.
    [[nodiscard]] input::ModelSnapshot applyTo(const input::ModelSnapshot& base,
                                               const std::shared_ptr<const void>& keepAlive) const
    {
        return m_overlay.applyTo(base, keepAlive);
    }

private:
    std::vector<double> m_x;
    DesignValues m_values;
    input::ColumnOverlay m_overlay;
    std::vector<std::uint64_t> m_designStamps;
};

} // namespace oos::design
