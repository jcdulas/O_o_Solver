#pragma once

#include "oos/input/tables/UnitTable.hpp"
#include "oos/reduction/units/CondensationUnit.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::reduction
{

// The units of condensation of a model: by default one per component of
// O_o_Reader, with its revision stamp and its boundary as connectedComponents and componentBoundaries give them;
// Regrouping makes other sets from it.
//
// Invariants: units sorted by ID, IDs distinct, each unit's lists sorted and distinct, a component in one unit at
// most, an element in one unit at most.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its units (their size is the model's element
// count at most: a std::vector, read once per plan).
class UnitSet
{
public:
    // Throws std::invalid_argument when an invariant above does not hold (the message names the unit and what breaks),
    // or when a declared connection is an element of a unit.
    explicit UnitSet(std::vector<CondensationUnit> units, std::vector<DeclaredConnection> declaredConnections = {});

    // The default units: one per row of connectedComponents (its revision; the 1D/0D-only flag from is_line_only, else
    // shell_count and solid_count when the table has them), its elements from connectedComponents(detail="elements")
    // and its boundary from componentBoundaries (every load group of the component merged; empty when the snapshot
    // was taken without the boundaries). `declaredConnections`: the elements the caller declared connections, which
    // O_o_Reader's component queries took (their `connections` option): in no component.
    [[nodiscard]] static UnitSet fromComponents(const input::UnitTable& table,
                                                std::vector<DeclaredConnection> declaredConnections = {});

    [[nodiscard]] std::span<const CondensationUnit> units() const noexcept { return m_units; }
    [[nodiscard]] std::size_t size() const noexcept { return m_units.size(); }
    // The unit of an ID, or nullptr.
    [[nodiscard]] const CondensationUnit* find(std::int64_t unitId) const noexcept;
    // The index in units() of an ID, or nothing.
    [[nodiscard]] std::optional<std::size_t> indexOf(std::int64_t unitId) const noexcept;
    // The unit a component belongs to, or nothing.
    [[nodiscard]] std::optional<std::int64_t> unitOfComponent(std::int64_t componentId) const noexcept;
    // The unit an element belongs to, or nothing (a connection, or an element of no unit).
    [[nodiscard]] std::optional<std::int64_t> unitOfElement(base::EntityId elementId) const noexcept;
    // The elements declared connections, by increasing ID: in no unit,
    // connections of the connection system (O_o_Reader lists them there).
    [[nodiscard]] std::span<const DeclaredConnection> declaredConnections() const noexcept
    {
        return m_declaredConnections;
    }

private:
    std::vector<CondensationUnit> m_units;
    std::vector<std::pair<std::int64_t, std::int64_t>> m_componentUnits;  // (component, unit), sorted
    std::vector<std::pair<base::EntityId, std::int64_t>> m_elementUnits; // (element, unit), sorted
    std::vector<DeclaredConnection> m_declaredConnections;                // bounded by the caller's list
};

} // namespace oos::reduction
