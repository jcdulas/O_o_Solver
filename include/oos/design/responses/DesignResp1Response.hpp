#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/responses/ElementCatalog.hpp"
#include "oos/design/responses/ResponseContext.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/ResponseValues.hpp"
#include "oos/design/responses/stress_strain_force/ItemCodes.hpp"
#include "oos/design/responses/weight_volume/MassProperties.hpp"
#include "oos/output/arrays/OutputPlan.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::design
{

// One DesignResp1 resolved against a snapshot: its targets found, its item codes located in the
// result tables, its subcase rules known; evaluate() reads its items for a subcase.
//
// Rules by type (the rules of the DesignResp1 entry):
// - WEIGHT: MO(ATTA, ATTB) of the grid point weight generator about PARAM GRDPNT (the origin when it is not a grid
//   point), ATTA and ATTB from 1 to 6, default 3; ATTi blank, ALL or 0 (the residual: O_o_Solver has no
//   superelements yet). One global item.
// - VOLUME: the volume of the structural elements (MassProperties::volume); ATTi as for WEIGHT. One global item.
// - FRMASS: the mass of the elements of the properties ATTi of type PTYPE (every structural element when ATTi is
//   blank) over their mass at the initial design (FRMASS is 1.0 at the initial design; O_o_Solver applies that
//   definition to the sizing variables). One global item.
// - DISP, SPCFORCE: ATTA the components (any unique digits 1 to 6), ATTi the grid points; items grid by grid,
//   component by component, in the displacement system of the grid point (the output system of the result tables).
// - STRESS, STRAIN, FORCE, ESE: ATTA the item code (locateItem), ATTi the elements (PTYPE = ELEM) or the properties
//   of type PTYPE, whose elements are taken in increasing ID; one item per element.
// - CSTRESS, CSTRAIN, CFAILURE, CSTRAT: as above, on CompProp or CompPropGlobal elements, ATTB the lamina (default 1;
// several
//   with the caller's plies= selector); one item per element and lamina.
// A positive integer ATTB on DISP, STRESS, STRAIN or FORCE selects a vibration mode. Caller declarations can use
// DesignResp1Definition::mode directly, including for CSTRESS and CSTRAIN while keeping ATTB as their ply selector.
//
// The kinds of the optimisation aids (no ATTA, PTYPE nor ATTB):
// - COMPLIANCE, MEANDISP: no ATTi; one item per subcase (SelfAdjointResponses: twice the external work of the
//   subcase, over Σ|F_g| of its applied forces for the mean displacement); refused in a subcase that enforces a
//   displacement and in a SUBCOM.
// - DEFLTHK: ATTi the shell grid points (blank: every corner of a shell element), one item per grid point: |n·u|/t̄
//   (LinearityIndicator, on the snapshot of the design point: the designed thicknesses and positions are read).
// - ROTATION: ATTi the grid points (blank: every grid point), one item per grid point: |θ| in radians.
//
// The kinds of the sizing tools (no ATTA, PTYPE, ATTB nor ATTi):
// - SECTLOAD: its sections (DesignResp1Definition::sections, checked against the model as the analysis checks them:
//   output/sections/SectionLoads), one item per section and component of its family (componentName: that component
//   only), entity the section ID, component its number from 1 (sectionComponents). The analysis computes them on the
//   load basis (DesignResponses::sectionLoadRequest), the items read them (ResponseContext::sectionLoads).
// - CUTLOAD: its cuts (output/sections/CutLoads), one item per cut and component (fx ... mz), entity the cut ID.
//
// The kinds of the external local models (no ATTA, PTYPE, ATTB nor ATTi):
// - CONTDISP, CONTFORC: its zones (DesignResp1Definition::zones, ResponseParts::zones), per zone, subcase and contour
// DOF one
//   item (componentName: the DOFs of that component of the zone's system), entity the zone ID, component the component
//   (1 to 6) in the zone's system, layer the grid point: the contour displacements and forces the analysis computes on
//   its load basis (DesignResponses::zoneRequest), read from ResponseContext::zones.
//
// The eigenvalue types of the MODES and BUCK analyses:
// - EIGN, FREQ: ATTA the normal mode number (≥ 1), ATTB blank, 1 or 2 (the approximation code), PTYPE blank or
//   STRUC (the source of the mode; FLUID refused: no fluid in O_o_Solver), REGION and ATTi blank. One item per MODES
//   subcase: the eigenvalue λ (EIGN) or the frequency √λ / 2π (FREQ, the "cycles" of the eigenvalue table) of the
//   mode, its entity ATTA and its layer the mode taken (ATTA, or the mode tracked to: ResponseContext::trackedModes).
// - LAMA: ATTA the buckling mode number, ATTB blank, 1 or 2, PTYPE, REGION and ATTi blank; one item per BUCK subcase,
//   the load factor of the mode, labelled as EIGN.
// A mode number beyond the roots the analysis found is an OptimisationError naming the method to widen.
//
// Thread-safety: immutable once built; evaluate() uses the context of one thread. Ownership: owns its resolution.
class DesignResp1Response
{
public:
    // Resolves `definition` against the snapshot of the catalog; the problems (an unknown grid point, element or
    // property, an item code the element does not have, a field that the type does not take) are appended to
    // `problems` as diagnostics of card DesignResp1 with its ID and the field.
    DesignResp1Response(DesignResp1Definition definition, const input::ModelSnapshot& snapshot,
                        const ElementCatalog& catalog, std::vector<base::Diagnostic>& problems);

    [[nodiscard]] const DesignResp1Definition& definition() const noexcept { return m_definition; }
    [[nodiscard]] bool isGlobal() const noexcept { return isGlobalType(m_definition.type); }

    // Adds the outputs its items read to `outputs` (StaticOptions::requiredOutputs): the grid points or elements of
    // its kind.
    void addRequiredOutputs(std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT>& outputs) const;

    // The items of subcase `subcase` (0 for a global response), appended to `values`. Throws OptimisationError when
    // a result it needs is missing (the message names the DesignResp1, the item and the table).
    void evaluate(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const;

    // One element of an element response with its result table and the locations of its items.
    struct ElementTarget
    {
        std::int32_t elementId = 0;
        std::string card;
        std::size_t batch = 0; // index into ModelSnapshot::elementBatches()
        std::int64_t row = 0;
        std::string elementType; // the table's element type ("Rod2", "ShellQuad4Composite")
        std::vector<std::pair<std::int32_t, ItemLocation>> items; // item code and location
    };

    // What the sensitivities read to form the derivatives of the items from their labels: the elements of
    // an element response, the properties of FRMASS (empty: every structural element), the entry of WEIGHT, the grid
    // points of DISP, SPCFORCE, DEFLTHK and ROTATION.
    [[nodiscard]] const std::vector<ElementTarget>& elementTargets() const noexcept { return m_elements; }
    [[nodiscard]] const std::vector<std::int32_t>& gridTargets() const noexcept { return m_grids; }
    [[nodiscard]] const std::vector<std::pair<std::string, std::int32_t>>& massProperties() const noexcept
    {
        return m_properties;
    }
    [[nodiscard]] std::pair<std::int32_t, std::int32_t> weightEntry() const noexcept
    {
        return {m_weightRow, m_weightColumn};
    }
    // SECTLOAD, CUTLOAD: the items of a subcase, (section or cut ID, component number from 1, its name), in order.
    struct SectionItem
    {
        std::int32_t id = 0;
        std::int32_t component = 0;
        std::string_view name;
    };
    [[nodiscard]] const std::vector<SectionItem>& sectionItems() const noexcept { return m_sectionItems; }

    // The elements the mass of FRMASS takes (its properties, or every structural element).
    [[nodiscard]] ElementFilter massFilter() const;
    // The reference point of WEIGHT: the grid point of PARAM GRDPNT, the origin otherwise.
    [[nodiscard]] static base::Vec3 weightReference(const input::ModelSnapshot& snapshot);

private:
    void resolveGridTargets(const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& problems);
    void resolveElementTargets(const ElementCatalog& catalog, std::vector<base::Diagnostic>& problems);
    void resolveMassTargets(const ElementCatalog& catalog, std::vector<base::Diagnostic>& problems);
    void resolvePointTargets(const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& problems);
    void resolveSections(const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& problems);
    void sectionValues(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const;
    void resolveZones(std::vector<base::Diagnostic>& problems);
    void zoneValues(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const;
    void resolveEigen(std::vector<base::Diagnostic>& problems);
    void eigenValues(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const;
    void resolveAeroelastic(std::vector<base::Diagnostic>& problems);
    void aeroelasticValues(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const;
    [[nodiscard]] double massOfSelection(const input::ModelSnapshot& snapshot, ResponseContext& context) const;
    // COMPLIANCE, MEANDISP of a subcase; DEFLTHK, ROTATION of a grid point.
    [[nodiscard]] double selfAdjointValue(std::int32_t subcase, ResponseContext& context) const;
    [[nodiscard]] double pointValue(std::int32_t grid, std::int32_t subcase, ResponseContext& context) const;
    [[nodiscard]] double elementItem(const ElementTarget& target, std::int32_t code, const ItemLocation& location,
                                     std::int32_t lamina, std::int32_t subcase, ResponseContext& context,
                                     std::int32_t mode = 0) const;

    DesignResp1Definition m_definition;
    std::vector<std::int32_t> m_components;                         // DISP, SPCFORCE: the digits of ATTA
    std::vector<std::int32_t> m_grids;                              // DISP, SPCFORCE, DEFLTHK, ROTATION
    std::vector<ElementTarget> m_elements;                          // element responses
    std::vector<std::int32_t> m_plies;                              // composite responses: the laminae
    std::vector<std::pair<std::string, std::int32_t>> m_properties; // FRMASS: (property card, PID), empty for all
    std::int32_t m_weightRow = 3;
    std::int32_t m_weightColumn = 3;
    std::vector<SectionItem> m_sectionItems; // SECTLOAD, CUTLOAD
    std::int32_t m_zoneComponent = 0;        // CONTDISP, CONTFORC: the component selected, 0 for every one
    std::int32_t m_stabilityComponent = 0;   // STABDER: the coefficient, 1 to 6 (CX ... CMZ)
    bool m_isRestrained = false;             // STABDER: ATTB 1, the restrained coefficient (0: unrestrained)
};

} // namespace oos::design
