#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/loads/FieldPrecision.hpp"
#include "oos/assembly/loads/InertialLoads.hpp"
#include "oos/assembly/loads/LoadFamily.hpp"
#include "oos/assembly/loads/LoadSetIndex.hpp"
#include "oos/assembly/loads/PressureRows.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace oos::assembly
{

// The static load vector Pg of a load set, from the load tables of O_o_Reader. A load set
// is what a subcase's LOAD selects: its terms (input::LoadTable::termsOf) are the sets whose rows make its
// loads, each with its factor (the set alone with the factor 1 when the tables expand the LOAD combinations, the rows
// of a combination then listed under its SID with its factors applied; or, when O_o_Reader lists each set once,
// a LOAD entry the sets it reaches with their combined_factor); each value is the row's times its term's factor, and
// the terms are read in their order, the rows of each in table order:
// - loadsTable: FORCE, PointForceLoad1, PointForceLoad2, MOMENT, PointMomentLoad1, PointMomentLoad2 per grid point, the
// force and the moment in the basic
//   system, scaled; they are turned into the displacement system CD of the grid point (f_cd = Rᵀ·f_basic, R the CD
//   axes in basic). ScalarPointLoad on a scalar point (its value in fx);
// - the inertial loads GravityLoad, ACCEL, AccelNodeLoad, RotationalForceLoad (InertialLoads: Mgg times an acceleration
// field; Mgg with WTMASS
//   applied);
// - pressureTable: ShellPressureLoad and FacePressureLoad on an element, the corner pressures scaled, through the
// pressure facet of the
//   element's family;
// - lineLoadsTable: BeamDistributedLoad on a bar or beam, p1 and p2 times the table's `scale`, through the line-load
// facet;
// - the thermal loads of a subcase with TEMPERATURE(LOAD) (addThermalLoads, the temperatures of a TemperatureField),
//   through the thermal facet of every family, batch by batch.
// The equivalent nodal loads are added in a fixed order (the terms, the rows of each table, the tables in the order
// above, the batches and their elements in order), so the vector does not depend on the threads.
//
// The loads are grouped by family (LoadFamily: concentrated, distributed, inertial; the thermal loads apart), so that
// the load basis of a static solution (LoadBasis) takes the vectors of one family at a time (addFamilyLoads), and
// reads the values the cards give in a compact form for its rank test: the concentrated loads per loaded DOF
// (concentratedEntries), the pressures and line loads per element and card value (distributedValues).
//
// A load the solver cannot apply is an error Diagnostic, nothing is ignored: a card of loadsTable outside the list
// ("load-unsupported"; RotationalForceLoad, ACCEL, AccelNodeLoad without inertialLoadsTable), a ShellPressureLoad or
// FacePressureLoad row without an element ("load-unsupported"), an element ID that no supported batch holds
// ("load-unknown-element"), a point the model does not have ("load-unknown-point"), a FORCE or MOMENT on a scalar
// point, an ScalarPointLoad or a PointsPressureLoad on a grid point that is not one ("load-bad-point"), a
// PointsPressureLoad whose points do not make a triangle or a quadrilateral of non-zero area ("load-bad-face"), and the
// element-level refusals of the families' load facets.
//
// PointsPressureLoad (a pressure on the triangle or quadrilateral of three or four grid points, no element) is applied
// here, as the format defines it (forEachGridPressureValue); it belongs to the distributed family.
//
// Thread-safety: const methods are safe concurrently. Ownership: owns the element index (budgeted); borrows the set
// (and its snapshot) and the DOF map.
class LoadVectorBuilder
{
public:
    LoadVectorBuilder(const ElementSet& set, const DofMap& dofMap, base::MemoryBudget& budget);

    // Whether the set holds a load that needs the mass matrix (the inertial loads).
    [[nodiscard]] bool needsMass(base::EntityId loadSetId) const;

    // Adds the loads of the set to `loadsG` (one value per g-set DOF): the concentrated, distributed and inertial
    // families. `mass` is Mgg with WTMASS applied, needed when needsMass(); an inertial load without it is reported
    // ("load-needs-mass"). Throws std::invalid_argument when the size of `loadsG` is not the g-set size.
    void addLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix* mass, std::span<double> loadsG,
                  base::MemoryBudget& budget, base::DiagnosticSink& sink) const;

    // Adds the loads of one family of the set to `loadsG` (Concentrated, Distributed or Inertial; std::invalid_argument
    // for Thermal and Enforced, which come from a TemperatureField and an SpcSet).
    void addFamilyLoads(base::EntityId loadSetId, LoadFamily family, const linalg::BlockSparseMatrix* mass,
                        std::span<double> loadsG, base::MemoryBudget& budget, base::DiagnosticSink& sink) const;

    // Adds the acceleration field of the inertial loads of the set to `accelerationsG`, the vector a whose product by
    // Mgg the inertial loads are (InertialLoads::addAccelerations). Throws std::invalid_argument when the size of
    // `accelerationsG` is not the g-set size.
    void addInertialAccelerations(base::EntityId loadSetId, std::span<double> accelerationsG,
                                  base::MemoryBudget& budget, base::DiagnosticSink& sink) const;
    // The terms of the set's inertial loads (InertialLoads::termsOf: the generator of each row's field and its
    // coefficients), the inertial family's description of the load basis.
    [[nodiscard]] std::vector<InertialTerm> inertialTermsOf(base::EntityId loadSetId, base::DiagnosticSink& sink) const;

    // Reports every load of the set the solver cannot apply (the problems addLoads would report), without computing
    // the load vector of the distributed loads: the check of every case when the load basis makes only some of them.
    // `checksInertial` false (the full path streamed, which makes the inertial loads from the elements'
    // masses and checks their rows itself): the inertial loads left unchecked, no mass needed.
    void checkLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix* mass, base::MemoryBudget& budget,
                    base::DiagnosticSink& sink, bool checksInertial = true) const;

    // Whether the set holds loads of a family (Concentrated, Distributed, Inertial).
    [[nodiscard]] bool hasFamilyLoads(base::EntityId loadSetId, LoadFamily family) const;

    // The terms of the loads the set selects (input::ModelSnapshot::loadTermsOf): the sets whose rows make them, each
    // with its factor, the set itself with the factor 1 first.
    [[nodiscard]] std::vector<input::LoadTerm> termsOf(base::EntityId loadSetId) const;
    // Whether the load tables list cards under the SID itself (not through its terms): every set of its own; a LOAD
    // entry only when it holds cards besides its entry (which a valid deck does not have) or the tables expand it.
    [[nodiscard]] bool holdsOwnCards(base::EntityId setId) const;

    // One value of a concentrated load on a g-set DOF (in the CD system of its grid point), as the rows of
    // loadsTable give it: a DOF may appear more than once (the values add up).
    struct DofValue
    {
        base::DofIndex dof = 0;
        double value = 0.0;
        // The precision of the value as written in the deck (FieldPrecision), NaN when not asked for or unknown.
        double precision = 0.0;
    };
    // The concentrated loads of the set, in table order (the rows the solver cannot apply are left out; addLoads
    // reports them). With `precision`, each value carries the precision of its card as written: twice half a unit of
    // the last digit of its force or moment magnitude (F and the direction N are both rounded; loadsTable gives the
    // product), the LOAD factor applied; NaN when the field format of its card is not known.
    [[nodiscard]] base::BudgetedBuffer<DofValue> concentratedEntries(base::EntityId loadSetId,
                                                                     base::MemoryBudget& budget,
                                                                     const FieldPrecision* precision = nullptr) const;

    // One value written on a distributed load card, as the load tables give it (LOAD factors applied): a corner
    // pressure of a ShellPressureLoad or FacePressureLoad (slot 0-3), the pressure of a PointsPressureLoad (elementId
    // 0, its grid points in faceGrids), or P1 and P2 of a BeamDistributedLoad (slots 0 and 1). The equivalent nodal
    // loads are linear in these values for a given element (or face), slot and variant (card, direction, CID,
    // BeamDistributedLoad type, scale and positions), so that the rank of the cases can be tested on them.
    struct CardValue
    {
        std::int32_t elementId = 0;
        std::int32_t slot = 0;
        std::string_view card;
        std::int32_t frameId = 0;
        std::array<double, 3> direction{};
        std::string_view loadType;  // BeamDistributedLoad TYPE, empty for a pressure
        std::string_view scaleType; // BeamDistributedLoad SCALE, empty for a pressure
        double x1 = 0.0;
        double x2 = 0.0;
        std::array<std::int32_t, 4>
            faceGrids{};        // PointsPressureLoad: G1-G4 (G4 0 for a triangle); zeros for an element load
        double value = 0.0;     // the value with the LOAD factors applied
        double scale = 1.0;     // the product of the LOAD factors (the value as written is value / scale)
        double precision = 0.0; // half a unit of its last written digit times |scale|, NaN when not asked or unknown
    };
    // The card values of the distributed loads of the set, in table order (pressures, then line loads); the loads on
    // unknown elements are left out. With `precision`, each value carries the precision of its card as written.
    [[nodiscard]] base::BudgetedBuffer<CardValue> distributedValues(base::EntityId loadSetId,
                                                                    base::MemoryBudget& budget,
                                                                    const FieldPrecision* precision = nullptr) const;

    // Reports the elements whose thermal load cannot be computed under `temperatures` (the families' checks).
    void checkThermalLoads(const elements::ElementTemperatures& temperatures, base::DiagnosticSink& sink) const;
    // Adds the thermal loads of every element under `temperatures` to `loadsG`. Precondition: checkThermalLoads()
    // found no error. Throws std::invalid_argument when the size of `loadsG` is not the g-set size.
    void addThermalLoads(const elements::ElementTemperatures& temperatures, std::span<double> loadsG) const;

    // The line loads of the set (PLOAD1), per batch of the set (indexed as ElementSet::batches()), the loads of one
    // element contiguous and the elements in their batch order: what the recovery facets under line loads take
    // (ElementFamily::recoverUnderLineLoads). Loads on unknown elements are left out (addLoads() reports them).
    [[nodiscard]] std::vector<std::vector<elements::ElementLineLoad>> lineLoadsOf(base::EntityId loadSetId) const;

    // The rows of loadsTable under the SID of a load set, in table order: with tables that expand the LOAD
    // combinations, a combination's rows (each gives the elementary set of its card, source_sid, and the product of
    // the LOAD factors, scale), read by the load basis of such tables (StaticLoadCases); when each set is listed once,
    // the set's own cards only (its terms make a combination's loads: termsOf).
    [[nodiscard]] std::span<const std::int64_t> loadTableRowsOf(base::EntityId loadSetId) const
    {
        return m_loadRows.rowsOf(loadSetId);
    }
    // The card column of loadsTable, read once (empty without load tables).
    [[nodiscard]] const base::DictionaryView& loadTableCards() const noexcept { return m_loadCards; }
    // Whether a load set has line loads (PLOAD1: lineLoadsTable, which the loads table does not list), its terms'.
    [[nodiscard]] bool hasLineLoads(base::EntityId loadSetId) const;
    // The family of a load card of loadsTable (FORCE: concentrated, FacePressureLoad: distributed, GravityLoad:
    // inertial); nothing for a card outside the families (refused by addLoads).
    [[nodiscard]] static std::optional<LoadFamily> familyOfCard(std::string_view card);

    // The batch and row of an element of the set, by ID; nothing when no batch of the set holds it.
    struct ElementLocation
    {
        std::int32_t elementId = 0;
        std::int32_t batchIndex = 0;
        std::int64_t row = 0;
    };
    [[nodiscard]] std::optional<ElementLocation> findElement(std::int32_t elementId) const;

    // The pressures (PLOAD2, FacePressureLoad) and line loads (PLOAD1) of a load set on elements, per batch of the set,
    // in the order of the table rows, as addLoads applies them (the PLOADs on the faces of grid points left out; the
    // rows it cannot apply reported into `sink`). They borrow the load tables of the set's snapshot. The sensitivities
    // difference them element by element.
    struct ElementDistributedLoads
    {
        std::vector<std::vector<elements::ElementPressure>> pressures;
        std::vector<std::vector<elements::ElementLineLoad>> lineLoads;
    };
    [[nodiscard]] ElementDistributedLoads elementDistributedLoads(base::EntityId loadSetId,
                                                                  base::DiagnosticSink& sink) const;

    // Step 10.27 (the full path streamed, which loads each element at its front): the rows of pressureTable under a
    // set's own SID (not its terms), increasing (the rows of the default form: PressureRows numbers the fields' so);
    // and the pressure of each of `rows` with the factor 1, as addLoads makes it (its element's place in `locations`,
    // elementId 0 for a row on the faces of grid points or on an element no batch holds, which addLoads reports).
    [[nodiscard]] base::BudgetedBuffer<std::int64_t> pressureRowsOf(base::EntityId setId,
                                                                    base::MemoryBudget& budget) const;
    void pressuresOfRows(std::span<const std::int64_t> rows, std::vector<ElementLocation>& locations,
                         std::vector<elements::ElementPressure>& pressures) const;
    // The rows of pressureTable in its form (PressureRows: the rows, or O_o_Reader's fields).
    [[nodiscard]] const PressureRows& pressureRows() const noexcept { return m_pressures; }

private:
    void addConcentratedLoads(base::EntityId loadSetId, std::span<double> loadsG, base::DiagnosticSink& sink) const;
    void addInertialLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix* mass, std::span<double> loadsG,
                          base::MemoryBudget& budget, base::DiagnosticSink& sink) const;
    // Calls visit(dof, value, row, term) for each value of the concentrated loads of the set (`row` the row of
    // loadsTable, `term` the term it was reached through, its factor applied to `value`), in the order of the terms
    // and the rows; reports into `sink` the rows it cannot apply, named by the set of their card.
    template <typename Visit>
    void forEachConcentratedValue(base::EntityId loadSetId, base::DiagnosticSink& sink, Visit&& visit) const;
    void addPressures(base::EntityId loadSetId, std::span<double> loadsG, base::DiagnosticSink& sink) const;
    // Calls visit(dof, value) for each nodal load of the PointsPressureLoad rows of the set (pressures on the faces of
    // grid points), in table order; reports into `sink` the rows it cannot apply.
    template <typename Visit>
    void forEachGridPressureValue(base::EntityId loadSetId, base::DiagnosticSink& sink, Visit&& visit) const;
    // The pressures of the set per batch, in the order of the table rows (the PointsPressureLoad rows left to
    // forEachGridPressureValue).
    [[nodiscard]] std::vector<std::vector<elements::ElementPressure>> collectPressures(
        base::EntityId loadSetId, base::DiagnosticSink& sink) const;
    // The pressure of one row on its element (`location`) with the factor 1 (collectPressures, pressuresOfRows).
    [[nodiscard]] elements::ElementPressure elementPressureOf(const PressureRow& row,
                                                              const ElementLocation& location) const;
    void addLineLoads(base::EntityId loadSetId, std::span<double> loadsG, base::DiagnosticSink& sink) const;
    // The line loads of the set per batch, in the order of the table rows.
    [[nodiscard]] std::vector<std::vector<elements::ElementLineLoad>> collectLineLoads(
        base::EntityId loadSetId, base::DiagnosticSink& sink) const;

    const ElementSet* m_set;
    const DofMap* m_dofMap;
    base::BudgetedBuffer<ElementLocation> m_elements; // sorted by elementId
    // The rows of loadsTable, pressureTable (in its form: PressureRows) and lineLoadsTable by load set (empty
    // without load tables).
    LoadSetIndex m_loadRows;
    PressureRows m_pressures;
    LoadSetIndex m_lineLoadRows;
    // The sets whose rows of loadsTable hold a concentrated card, increasing (hasFamilyLoads): the family of a set is
    // asked for every case that takes it, never by a pass over its rows (the cabin pressure's set in each
    // of 6 000 LOAD entries).
    base::BudgetedBuffer<std::int32_t> m_concentratedSets;
    // The text columns, read once: a read checks every index of the column, which per load set would make the
    // assembly of thousands of load sets quadratic.
    base::DictionaryView m_loadCards;
    base::DictionaryView m_lineCards;
    base::DictionaryView m_lineLoadTypes;
    base::DictionaryView m_lineScaleTypes;
    std::optional<InertialLoads> m_inertial; // built last: it borrows m_loadRows
};

} // namespace oos::assembly
