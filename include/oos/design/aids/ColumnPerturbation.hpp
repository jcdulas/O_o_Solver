#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/sensitivity/FieldPerturbation.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>

namespace oos::design
{

// The snapshots of a design point on which every value of one column of a card's table is moved at once, each by its
// own step (fieldStep: 1e-2 of the value, extrapolated as FieldPerturbation does): the property field of every
// ShellProp, the material field of every IsoMat, the element field of every SpringDofInline. An element reads the value of
// one row of the column (its property, its material, itself), and nothing else changes for it, so that the differences
// of its computations on these snapshots are its derivatives with respect to its own value, "as if each element had its
// own value" (the sensitivity map): one perturbation for every element at once.
//
// The rows are keyed by their ID (PID, MID, EID); a material field is moved in materialsTable (the rows of its type)
// and in its card's own table, as FieldTargets binds it. Only real scalar columns ("float64") are moved; a null value
// is left as it is (its row is not reached).
//
// Thread-safety: immutable once built. Ownership: owns the snapshots (which keep the given one alive), their indices
// and element sets; borrows the registry and the budget, which must outlive it.
class ColumnPerturbation
{
public:
    // `snapshot`: the design point's snapshot; `card`, `field`: the card and the column of the data contract. Throws
    // OptimisationError when the model has no such card or column, or the column is not a real scalar one.
    ColumnPerturbation(const std::shared_ptr<const input::ModelSnapshot>& snapshot, std::string card, std::string field,
                       const elements::ElementRegistry& registry, base::MemoryBudget& budget);

    [[nodiscard]] TargetKind kind() const noexcept { return m_kind; }
    [[nodiscard]] const std::string& card() const noexcept { return m_card; }
    [[nodiscard]] const std::string& field() const noexcept { return m_field; }
    [[nodiscard]] const assembly::ElementSet& set(FieldSide side) const noexcept
    {
        return *m_sides[static_cast<std::size_t>(side)].set;
    }
    [[nodiscard]] const std::shared_ptr<const input::ModelSnapshot>& snapshot(FieldSide side) const noexcept
    {
        return m_sides[static_cast<std::size_t>(side)].snapshot;
    }

    // Whether the row of an ID was moved.
    [[nodiscard]] bool isMoved(std::int32_t id) const { return m_widths.count(id) != 0; }
    // The derivative with respect to the value of row `id` of a quantity from its values on the four sides
    // (Richardson, as FieldPerturbation::derivative), and element by element of four arrays.
    [[nodiscard]] double derivative(std::int32_t id, double plus, double minus, double halfPlus,
                                    double halfMinus) const;
    void derivatives(std::int32_t id, std::span<const double> plus, std::span<const double> minus,
                     std::span<const double> halfPlus, std::span<const double> halfMinus,
                     std::span<double> derivative) const;

private:
    struct Side
    {
        std::shared_ptr<const input::ModelSnapshot> snapshot;
        std::unique_ptr<elements::SnapshotIndex> index;
        std::unique_ptr<assembly::ElementSet> set;
    };
    struct Widths
    {
        double full = 0.0; // (v + h) − (v − h) as stored
        double half = 0.0; // (v + h/2) − (v − h/2) as stored
    };

    // The memory of the moved cells (the widths and the four copies of the column), estimated before they are made.
    base::MemoryReservation m_reservation;
    std::string m_card;
    std::string m_field;
    TargetKind m_kind = TargetKind::Property;
    std::map<std::int32_t, Widths> m_widths;
    std::array<Side, FIELD_SIDE_COUNT> m_sides;
};

} // namespace oos::design
