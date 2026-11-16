#pragma once

#include "oos/base/math/Vec3.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::elements
{

// What an entity of a formulation is, for the messages that name it.
enum class EntityKind : std::uint8_t
{
    Edge,     // the edge between two grid points
    Face,     // the face of three or four grid points
    Interior, // the interior of one element
    LineEnd   // the end of a line element at one grid point (its released components)
};

// How an entity is named in a message: its kind, the IDs of its grid points (sorted, unused ones 0) and, for an
// interior or a line end, the element ID.
struct EntityName
{
    EntityKind kind = EntityKind::Edge;
    std::array<std::int32_t, 4> gridIds{};
    std::int32_t elementId = 0;
};

// The unknowns of an element formulation that belong to no grid point: the entities (edges, faces and interiors of
// the elements of the mixed hybrid formulation, docs/mixed-hybrid-formulation-spec.md section 5.1), each with its DOF
// count, a position (for the geometric ordering of the factorisation) and a name.
//
// The DofMap numbers the entity DOFs after the scalar points, entity by entity, and splits each entity into nodes of
// at most ENTITY_NODE_DOFS DOFs for the node-block matrices of linalg, whose blocks hold at most six components (a
// grid point's): an entity of 13 DOFs is three nodes of 6, 6 and 1 DOFs, which the ordering keeps together since they
// have the same neighbours.
//
// Which entities a model has depends on the analysis (the kind): the modal layout adds the interior fields of the
// solids.
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: a plain value.
class EntityLayout
{
public:
    // The most DOFs of one node of the node-block pattern: the six components of a grid point.
    static constexpr int ENTITY_NODE_DOFS = 6;
    // The most DOFs of one entity (a DOF's index within its entity is a std::uint8_t of ElementDof).
    static constexpr int MAX_ENTITY_DOFS = 255;

    enum class Kind : std::uint8_t
    {
        Static, // statics, the preload and the eigenproblem of buckling, trim, inertia relief
        Modal   // natural modes: the static entities and the interior fields of the solids
    };

    EntityLayout() = default;
    explicit EntityLayout(Kind kind) : m_kind(kind) {}

    // Appends an entity of `dofCount` DOFs (1 to MAX_ENTITY_DOFS) and returns its row. Throws std::invalid_argument
    // for a count out of that range.
    std::int32_t add(int dofCount, base::Vec3 position, EntityName name);

    [[nodiscard]] Kind kind() const noexcept { return m_kind; }
    [[nodiscard]] bool empty() const noexcept { return m_dofOffsets.size() == 1; }
    [[nodiscard]] std::int32_t count() const noexcept { return static_cast<std::int32_t>(m_positions.size()); }
    [[nodiscard]] int dofCount(std::int32_t row) const noexcept
    {
        return static_cast<int>(m_dofOffsets[index(row) + 1] - m_dofOffsets[index(row)]);
    }
    // The DOFs of the entities before the entity of a row, and of all of them.
    [[nodiscard]] std::int64_t dofOffset(std::int32_t row) const noexcept { return m_dofOffsets[index(row)]; }
    [[nodiscard]] std::int64_t totalDofCount() const noexcept { return m_dofOffsets.back(); }
    // The nodes of the entities before the entity of a row, and of all of them.
    [[nodiscard]] std::int32_t firstNode(std::int32_t row) const noexcept { return m_nodeOffsets[index(row)]; }
    [[nodiscard]] std::int32_t nodeCount() const noexcept { return m_nodeOffsets.back(); }
    // The entity whose DOFs hold the entity DOF `offset` (0 to totalDofCount() − 1).
    [[nodiscard]] std::int32_t rowOfDof(std::int64_t offset) const noexcept;
    [[nodiscard]] const base::Vec3& position(std::int32_t row) const noexcept { return m_positions[index(row)]; }
    [[nodiscard]] const EntityName& name(std::int32_t row) const noexcept { return m_names[index(row)]; }
    // "the edge between grid points 12 and 15", "the interior of element 7", for the messages.
    [[nodiscard]] std::string describe(std::int32_t row) const;

    friend bool operator==(const EntityLayout& a, const EntityLayout& b) noexcept
    {
        return a.m_kind == b.m_kind && a.m_dofOffsets == b.m_dofOffsets;
    }

private:
    [[nodiscard]] static std::size_t index(std::int32_t row) noexcept { return static_cast<std::size_t>(row); }

    Kind m_kind = Kind::Static;
    std::vector<std::int64_t> m_dofOffsets{0};
    std::vector<std::int32_t> m_nodeOffsets{0};
    std::vector<base::Vec3> m_positions;
    std::vector<EntityName> m_names;
};

} // namespace oos::elements
