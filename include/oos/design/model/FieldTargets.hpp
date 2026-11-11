#pragma once
#include "oos/design/model/DerivedMaterials.hpp"

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/model/DerivedGeometry.hpp"
#include "oos/design/model/DerivedProperties.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/model/SectionLibrary.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// One cell of a snapshot table that a designed field writes: the table (as ModelSnapshot::findTable names it), the
// column, the row and the item (-1 for a scalar column).
struct TargetCell
{
    std::string table;
    std::string column;
    std::int64_t row = 0;
    std::int32_t item = -1;
    // The cell holds scale × the field (a blank TubeProp T follows a designed OD as OD/2); 1 otherwise.
    double scale = 1.0;
};

// Where the field of one relation lives in a snapshot, and the units it changes.
struct FieldTarget
{
    // Usually one cell; a material field is written in materialsTable and in its card's own table
    // ("materials/OrthoMat2D") when both hold the column, as the elements read either.
    std::vector<TargetCell> cells;
    std::vector<std::size_t> units; // indices into FieldTargets::unitIds(), increasing
};

// The grid points one shape variable moves (a ShapeBasis), bound to the node table of a snapshot.
struct ShapeTarget
{
    // Per move of the basis (ShapeBasis::moves order): the index of its grid point in FieldTargets::movedRows().
    std::vector<std::size_t> points;
    std::vector<std::size_t> units; // indices into FieldTargets::unitIds(), increasing
};

// The designed fields of a design model bound to the columns of one snapshot (DesignPoint applies values to an overlay
// of the snapshot's property, material and element columns), and the units each changes (for the design stamps, part
// of the cache key).
//
// Binding: a property field to the column `field` of "properties/<TYPE>" at the row of the PID; an element field to
// "elements/<TYPE>" at the row of the EID; a material field to materialsTable at the row of (MID, TYPE) and to
// "materials/<TYPE>" at the row of the MID, each when it has the column. The column must be "float64" (fieldItem
// -1), "fixed_list<float64>[n]" (fieldItem below n) or "list<fixed_list<float64>[n]>" (the stations of a BeamProp:
// fieldItem station · n + component, below n times the stations of the row, as O_o_Reader's designRelationsTable
// flattens it): only real fields can be designed.
//
// The ply thicknesses of CompProp and CompPropGlobal (ply_thickness: the plies the elements read from
// laminatePliesTable) and the dimensions of BarShapeProp and BeamShapeProp (dims, station_dims: the section properties
// they read from sectionsTable) are "list<float64>" columns, the item the position in the list; a design point overlays
// beside them the tables O_o_Reader derives from these cards (DerivedProperties; the section properties from
// `sections`, O_o_Reader's section library). Any other field of these cards is refused (UnsupportedCardError,
// design_code::UNSUPPORTED_TARGET), THETA among them. A TubeProp's OD2, blank or given equal to its OD (to the bit),
// follows a designed OD (a blank OD2 is OD; a given equal one left behind would taper the tube, which this version does
// not analyse); its OD2 as a target of its own is refused.
//
// Units: the components of the snapshot's unit tables (connectedComponents). An element field changes the unit of
// its element; a property field the units of the elements of that PID (property IDs are unique across property types
// in the input format); a material field the units of the elements whose property or own card names the MID in a column
// "matId", "matId1", "membraneMatId" ... (plies of laminatePliesTable included). Without unit tables, unitIds() is
// empty and no stamp is computed.
//
// Shape variables (DVGRID): the grid points of each basis are bound to the rows of the node table, and a
// design point overlays the column xyzGlobal of "nodes" at those rows with g = g0 + Σ_v s_v·T_v(g) (ShapeRelations).
// A grid point changes the unit (component) componentNodes gives it, or every unit when the snapshot has the units but
// not componentNodes (a moved grid point must never leave a stamp unchanged). Refused
// (design_code::UNSUPPORTED_TARGET), because the overlay of xyzGlobal would leave a table that O_o_Reader derives from
// the positions as it was read, so that the analysis of the point would be wrong: a moved grid point whose displacement
// system CD is cylindrical or spherical (its rotation, displacementRotations, depends on the position); a grid point of
// a shell element a Fastener or Weld pierces, or the GA, GB or GS of a Fastener or Weld
// (fastenerPiercings: piercing points and shape functions); one of a glued contact (contactNodePairs: projections and
// local coordinates). A grid point that a DesignNodeMove names with a zero move (COEFF 0) does not move and is not
// refused. The rigid elements whose grid points move relative to each other are refused by the sensitivities only
// (their analysis is right: the rigid element equations read the overlaid positions).
//
// The directions of PointForceLoad1, PointForceLoad2, PointMomentLoad1, PointMomentLoad2 and the frames of RectFrame1,
// CylindricalFrame1, SphericalFrame1 whose grid points move (and what is given in those frames) are derived again at
// each design point (DerivedGeometry, with its refusals) and overlaid with the positions; a basis that moves a system
// changes every unit (any element may read it), one that turns a concentrated load also changes the unit of the load's
// grid point.
//
// Thread-safety: immutable once built. Ownership: owns its data (the base positions of the moved grid points, bounded
// by the designed grid points); keeps no reference to the snapshot.
class FieldTargets
{
public:
    // Throws UnsupportedCardError or OptimisationError listing every relation whose card, ID or field the snapshot
    // does not have (design_code::MISSING_TARGET) or that is refused (design_code::UNSUPPORTED_TARGET), and every
    // DesignNodeMove whose grid point the snapshot does not have or whose move is refused.
    // `sections`: the section library of a designed BarShapeProp or BeamShapeProp (null: none; a design point that
    // needs it throws).
    FieldTargets(const DesignModel& model, const input::ModelSnapshot& snapshot,
                 std::shared_ptr<const SectionLibrary> sections = nullptr);

    // One per relation of the model, in order.
    [[nodiscard]] const std::vector<FieldTarget>& targets() const noexcept { return m_targets; }
    [[nodiscard]] const FieldTarget& operator[](std::size_t relation) const { return m_targets[relation]; }
    // The component IDs of the snapshot's units, in the order of connectedComponents.
    [[nodiscard]] const std::vector<std::int64_t>& unitIds() const noexcept { return m_unitIds; }

    // One per shape basis of the model (ShapeRelations::bases() order).
    [[nodiscard]] const std::vector<ShapeTarget>& shapeTargets() const noexcept { return m_shapeTargets; }
    // The node table rows of every grid point a shape variable moves, increasing, and their positions in the basic
    // system in the snapshot the targets were bound to (3 per row).
    [[nodiscard]] const std::vector<std::int64_t>& movedRows() const noexcept { return m_movedRows; }
    [[nodiscard]] const std::vector<double>& basePositions() const noexcept { return m_basePositions; }
    // The directions and frames the shape variables change.
    [[nodiscard]] const DerivedGeometry& derived() const noexcept { return m_derived; }

    // The positions of the moved grid points (movedRows() order, 3 per row) for the shape values `shapes` (one per
    // basis): g0 + Σ_v s_v·T_v(g), the bases added in order (the same bits whatever the caller). Precondition: one
    // value per basis of the model the targets were built with.
    [[nodiscard]] std::vector<double> movedPositions(const ShapeRelations& relations,
                                                     std::span<const double> shapes) const;
    // The overlay of the snapshot for the field values `fields` (one per relation) and the shape values `shapes` (one
    // per basis): the designed cells, and xyzGlobal at the moved rows (always, even at their base positions, so that
    // every overlay of a model has the same columns).
    // `reference`: the field values of a design point whose section dimensions the library takes as the reference of
    // its meshes (a perturbation of that point; empty: none).
    [[nodiscard]] input::ColumnOverlay overlayOf(const ShapeRelations& relations, std::span<const double> fields,
                                                 std::span<const double> shapes,
                                                 std::span<const double> reference = {}) const;
    // The derived tables of the designed laminates and sections.
    [[nodiscard]] const DerivedProperties& derivedProperties() const noexcept { return m_derivedProperties; }
    [[nodiscard]] bool hasDerivedMaterial(std::size_t relation) const noexcept { return m_derivedMaterials.reaches(relation); }
    // The section library of the designed sections (null without one).
    [[nodiscard]] const SectionLibrary* sectionLibrary() const noexcept { return m_sections.get(); }
    // Adds to `overlay` the derived cells of relation `relation`'s card (its ply stack or its section properties) at
    // the field values `fields` (one per relation); nothing for a relation of another card. A perturbation of one field
    // (FieldPerturbation) overlays them with its own cells. Throws as DerivedProperties::addCells.
    // `linear`: the sections linearised at the design point (linearSections), used instead of the library.
    void addDerivedCells(input::ColumnOverlay& overlay, std::span<const double> fields, std::size_t relation,
                         const SectionLinearisation* linear = nullptr) const;
    // The designed sections at `fields` with the derivatives of their properties, from one call of the library
    // (DerivedProperties::linearise), those of the groups `isGroupWanted` flags (empty: every one).
    [[nodiscard]] SectionLinearisation linearSections(std::span<const double> fields,
                                                      std::span<const double> reference = {},
                                                      std::span<const std::uint8_t> isGroupWanted = {}) const
    {
        return m_derivedProperties.linearise(fields, m_sections.get(), reference, isGroupWanted);
    }

private:
    std::vector<FieldTarget> m_targets;
    std::vector<std::int64_t> m_unitIds;
    std::vector<ShapeTarget> m_shapeTargets;
    std::vector<std::int64_t> m_movedRows;
    std::vector<double> m_basePositions;
    DerivedGeometry m_derived;
    DerivedProperties m_derivedProperties;
    DerivedMaterials m_derivedMaterials;
    std::shared_ptr<const SectionLibrary> m_sections;
};

// The binding problems of a design model in a snapshot (what the FieldTargets constructor throws for), for reports.
[[nodiscard]] std::vector<base::Diagnostic> checkFieldTargets(const DesignModel& model,
                                                              const input::ModelSnapshot& snapshot);

} // namespace oos::design
