#pragma once

#include "oos/design/model/SectionLibrary.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// The tables O_o_Reader derives from the designed CompProp, CompPropGlobal, BarShapeProp and BeamShapeProp cards,
// overlaid at a design point (laminates and library sections as design targets). The elements read the plies of a
// laminate from laminatePliesTable and the properties of a library section from sectionsTable, not from the card: a
// design point that changes a ply thickness T or a dimension DIM overlays, beside the card's cell, what O_o_Reader's
// model.design.apply derives from it, with the same arithmetic, so that the elements read the bits of the same cards
// edited by apply:
// - a laminate (PCOMP, CompPropGlobal; its ply thicknesses are the list column ply_thickness, the item the ply of the
// card):
//   laminatePliesTable's thickness of every ply of the stack (the mirrored half of LAM = SYM follows its ply),
//   z_bottom and z_top (z_bottom of the first ply Z0, each next one the z_top of the ply below, z_top = z_bottom + t),
//   laminateSummary's thickness (the plies summed in stack order) and z0, and the card's Z0: a blank Z0 is −T/2 and
//   follows T, a given one stays (z0_given of data contract 54; before it, a Z0 equal to −T/2 to the bit is
//   taken as blank);
// - a library section (PBARL: its list column dims; BeamShapeProp: station_dims, the item station · n + dimension, n
// the
//   dimensions of its type): sectionsTable's area, i1, i2, i12, j, k1, k2 and the recovery points c1 ... f2 of its row
//   (each station's row for a BeamShapeProp, with its neutral axis from the shear centre n1, n2 when the table has
//   them) from the section library (SectionLibrary: O_o_Reader's formulas); NSM stays.
// THETA of a ply and the other fields of these cards are not derived (THETA is refused as a design target).
//
// The sensitivities: the perturbed sides of a section's dimension take the section's properties
// linearised at the design point, P(d) = P(d0) + ∂P/∂d·(d − d0), from one call of the library for every designed
// section with its derivatives (linearise()), the meshes those of d0: the differences of the elements over the sides
// then give ∂element/∂P·∂P/∂d exactly to their own order, whatever the finite element section's dependence on its
// mesh. A perturbation that rebuilds every cell (FieldPerturbation's first two constructors) asks the library with the
// design point's dimensions as reference instead (`reference`).
//
// Thread-safety: immutable once bound; addCells() is const. Ownership: owns its data (bounded by the designed cards);
// keeps no reference to the snapshot.
// The designed sections linearised at a design point (DerivedProperties::linearise): per section (in the order of the
// groups) and station, the dimensions there and the properties with their derivatives.
struct SectionLinearisation
{
    std::vector<std::vector<std::vector<double>>> dimensions; // [section][station]
    std::vector<std::vector<SectionValues>> values;           // [section][station], with their gradient
};

class DerivedProperties
{
public:
    // Binds relation `relation` (its field the list column `column` of the card table `table` at `row`, item `item`)
    // to its card's group; returns false with `problem` set when the derived table lacks the card or its field is not
    // one this class derives (THETA, a BarShapeProp's NSM...).
    bool bind(const input::ModelSnapshot& snapshot, std::size_t relation, const std::string& card, std::int32_t propId,
              const std::string& table, const std::string& column, std::int64_t row, std::int32_t item,
              std::string& problem);

    [[nodiscard]] bool isEmpty() const noexcept { return m_laminates.empty() && m_sections.empty(); }
    [[nodiscard]] bool hasSections() const noexcept { return !m_sections.empty(); }
    // The relation's group (an index of groupCount()), or nothing when its field is not a derived one.
    [[nodiscard]] std::optional<std::size_t> groupOf(std::size_t relation) const;
    [[nodiscard]] std::size_t groupCount() const noexcept { return m_laminates.size() + m_sections.size(); }
    // Whether group `group` is a laminate (its relations ply thicknesses), not a library section.
    [[nodiscard]] bool isLaminate(std::size_t group) const noexcept { return group < m_laminates.size(); }

    // Adds to `overlay` the derived cells of every group (or of `onlyGroup`) at the field values `fields` (one per
    // relation of the design model; a group's card values are its base values with those of its relations). Throws
    // std::invalid_argument without a library when a section is designed, base::SolverError for an impossible section
    // or a ply thickness that is not positive, naming the card.
    // `linear`: the sections linearised at a design point (linearise()), used instead of the library; `reference`:
    // the field values of a design point whose dimensions the library takes as reference (empty: none).
    void addCells(input::ColumnOverlay& overlay, std::span<const double> fields, std::optional<std::size_t> onlyGroup,
                  const SectionLibrary* library, const SectionLinearisation* linear = nullptr,
                  std::span<const double> reference = {}) const;

    // The designed sections at the field values `fields`, their properties and derivatives with respect to their
    // dimensions, from one call of the library (empty without a designed section). Throws as addCells().
    // `reference`: the field values of a design point whose meshes the finite element sections take (empty: those
    // of `fields`; the moved states of the Hessian-vector products take the point's). `isGroupWanted`
    // (one flag per group of groupOf's numbering; empty: every group): the sections linearised, the others' stations
    // left empty (a restricted jvp differences the fields of a few variables); each section's values are
    // those of the call on every section (the library computes each shape on its own).
    [[nodiscard]] SectionLinearisation linearise(std::span<const double> fields, const SectionLibrary* library,
                                                 std::span<const double> reference = {},
                                                 std::span<const std::uint8_t> isGroupWanted = {}) const;

private:
    struct Member
    {
        std::size_t relation = 0;
        std::int32_t item = 0;
    };
    struct Laminate
    {
        std::string card; // CompProp, CompPropGlobal
        std::int32_t propId = 0;
        std::string table;
        std::int64_t row = 0;
        std::vector<double> plies;           // the card's ply thicknesses at the base
        std::vector<std::int64_t> stackRows; // laminatePliesTable rows, in stack order
        std::vector<std::size_t> plyOfStack; // the card ply of each stack row
        bool isZ0Blank = true;
        double z0 = 0.0;              // the given Z0
        std::int64_t summaryRow = -1; // laminateSummary, -1 without the table or its row
        std::vector<Member> members;
    };
    struct Section
    {
        std::string card; // BarShapeProp, BeamShapeProp
        std::int32_t propId = 0;
        std::string table;
        std::string column;
        std::int64_t row = 0;
        std::string type;
        std::size_t dimensionCount = 0;
        std::vector<double> dimensions;        // stations × dimensionCount, at the base
        std::vector<std::int64_t> stationRows; // sectionsTable rows, one per station
        std::vector<Member> members;
    };

    // The dimensions of each station of the sections asked at `fields`, and the (section, station) of each.
    void stationShapes(std::span<const double> fields, std::optional<std::size_t> onlyGroup,
                       std::vector<SectionShape>& shapes,
                       std::vector<std::pair<std::size_t, std::size_t>>& stations) const;
    void writeSection(input::ColumnOverlay& overlay, std::size_t section, std::size_t station,
                      const SectionValues& value, std::span<const double> dimensions) const;

    bool bindLaminate(const input::ModelSnapshot& snapshot, const std::string& card, std::int32_t propId,
                      const std::string& table, std::int64_t row, std::string& problem);
    bool bindSection(const input::ModelSnapshot& snapshot, const std::string& card, std::int32_t propId,
                     const std::string& table, const std::string& column, std::int64_t row, std::string& problem);
    void addLaminate(const Laminate& laminate, input::ColumnOverlay& overlay, std::span<const double> fields) const;

    // A relation's group: a laminate (an index of m_laminates) or a section (of m_sections); its number among the
    // groups (groupOf) the laminates first.
    struct GroupRef
    {
        bool isLaminate = true;
        std::size_t index = 0;
    };

    std::vector<Laminate> m_laminates;
    std::vector<Section> m_sections;
    bool m_hasNeutralAxisColumns = false; // sectionsTable has n1, n2
    // The groups by (card, PID) and the relations' groups in increasing relation.
    // PERFORMANCE: a bind searched the groups one by one, renumbered the sections' relations after a new laminate and
    // sorted every relation again (some 480 000 ply relations of 30 000 CompProp: quadratic, 225 s extrapolated); the
    // groups are now found by their key and a relation's number made when asked: the same groups, the same numbers.
    // Measured: a ladder of CompProp declarations of increasing size.
    std::map<std::pair<std::string, std::int32_t>, std::size_t> m_laminateOf;
    std::map<std::pair<std::string, std::int32_t>, std::size_t> m_sectionOf;
    std::vector<std::pair<std::size_t, GroupRef>> m_groupOfRelation;
};

} // namespace oos::design
