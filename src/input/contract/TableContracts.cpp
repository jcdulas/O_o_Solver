#include "oos/input/contract/TableContracts.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <array>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace oos::input
{

namespace
{

using base::ColumnContract;
using base::TableContract;

const char* const INT32 = "int32";
const char* const INT64 = "int64";
const char* const FLOAT64 = "float64";
const char* const UTF8 = "utf8";
const char* const BOOL = "bool";
const char* const TEXTS = "dictionary<utf8>";
const char* const INT32_LIST = "list<int32>";
const char* const VECTOR = "fixed_list<float64>[3]";
const char* const REAL_LIST = "list<float64>";
const char* const TEXT_LIST = "list<utf8>";

ColumnContract required(std::string name, std::string type)
{
    return {std::move(name), std::move(type), false, {}};
}

ColumnContract optional(std::string name, std::string type)
{
    return {std::move(name), std::move(type), true, {}};
}

// A column of grid points or of rows per element: a fixed list for a table of one card type, a list for a table of
// mixed types (O_o_Reader's two layouts of the same column).
ColumnContract optionalConnectivity(std::string name)
{
    return {std::move(name), INT32_LIST, true, {"fixed_list<int32>[*]"}};
}

TableContract tableOf(TableRole role, std::vector<ColumnContract> columns,
                      std::initializer_list<const char*> realColumns = {})
{
    for (const char* name : realColumns)
        columns.push_back(required(name, FLOAT64));
    return {std::string(tableName(role)), std::move(columns)};
}

} // namespace

TableContract contractOf(TableRole role)
{
    switch (role)
    {
    case TableRole::Nodes:
        // cardTable(deck, "nodes"): the fields of the grid points, and their positions in the basic system.
        return tableOf(role, {required("id", INT32), required("positionFrameId", INT32),
                              required("displacementFrameId", INT32), required("permanentConstraints", INT32),
                              required("superelementId", INT32), required("xyzGlobal", VECTOR)});
    case TableRole::ScalarPoints:
        return tableOf(role, {required("id", INT32), optional("type", TEXTS)});
    case TableRole::Materials:
        return tableOf(
            role, {required("matId", INT32), required("type", TEXTS)},
            {"e", "g", "nu", "density", "a", "referenceTemperature", "structuralDamping", "e1", "e2", "nu12", "g12"});
    case TableRole::Spc:
        return tableOf(role, {required("setId", INT32), required("sourceSetId", INT32), required("card", TEXTS),
                              required("nodeId", INT32), required("dof", UTF8), required("enforced", FLOAT64)});
    case TableRole::Mpc:
        return tableOf(role, {required("setId", INT32), required("equation", INT32), required("term", INT32),
                              required("nodeId", INT32), required("dof", INT32), required("coefficient", FLOAT64),
                              required("isDependent", BOOL)});
    case TableRole::RigidElements:
        return tableOf(role, {required("elementId", INT32), required("type", TEXTS), required("role", TEXTS),
                              required("nodeId", INT32), required("dof", UTF8), required("weight", FLOAT64)});
    case TableRole::Loads:
        // g1 to g4 (O_o_Reader contract 44): the grid points that define the direction of a FORCE1, FORCE2, MOMENT1 or
        // MOMENT2 (null for the other cards), from which a design point derives it again when the grid points move.
        return tableOf(role,
                       {required("setId", INT32), required("sourceSetId", INT32), required("card", TEXTS),
                        required("scale", FLOAT64), required("targetType", TEXTS), required("targetId", INT32),
                        required("frameId", INT32), optional("g1", INT32), optional("g2", INT32), optional("g3", INT32),
                        optional("g4", INT32)},
                       {"fx", "fy", "fz", "mx", "my", "mz", "pressure", "ax", "ay", "az"});
    case TableRole::Pressures:
        return tableOf(role,
                       {required("setId", INT32), required("card", TEXTS), required("elementId", INT32),
                        required("faceNodes", INT32_LIST), required("frameId", INT32)},
                       {"p1", "p2", "p3", "p4", "directionX", "directionY", "directionZ", "scale"});
    case TableRole::PressureFields:
        // O_o_Reader contract 55: pressureTable(detail="fields"), a row per field (the rows of consecutive cards
        // of a set that share all but their element and pressure): elementCount rows, the elements of its element
        // list as (first, last) runs given on the first field of the list (empty on the later ones; elementList null
        // for a PLOAD, whose faceNodes the field holds), the pressures `values` (cornerCount of them, one for four
        // equal corners, for the field when isUniform, else for each row; a NaN the null of p1 to p4), the other
        // columns as the rows'. Expanded in order (input::PressureRows), they are the rows of the default form to the
        // bit.
        return tableOf(role,
                       {required("setId", INT32), required("card", TEXTS), required("elementCount", INT64),
                        required("elementList", INT32), required("eidRuns", INT32_LIST),
                        required("faceNodes", INT32_LIST), required("isUniform", BOOL), required("cornerCount", INT32),
                        required("values", REAL_LIST), required("frameId", INT32)},
                       {"directionX", "directionY", "directionZ", "scale"});
    case TableRole::LineLoads:
        return tableOf(role,
                       {required("setId", INT32), required("card", TEXTS), required("elementId", INT32),
                        required("loadType", TEXTS), required("scaleType", TEXTS)},
                       {"x1", "p1", "x2", "p2", "scale"});
    case TableRole::Temperatures:
        return tableOf(role,
                       {required("setId", INT32), required("card", TEXTS), required("targetType", TEXTS),
                        required("targetId", INT32)},
                       {"temperature", "gradient"});
    case TableRole::Deformations:
        // The DEFORM cards as O_o_Reader stores them (cardTable(deck, "AxialDeformationLoad")): one row per card, its
        // set ID and its one to three (EID, D) pairs as two lists of the same length (a positive D is an elongation).
        return tableOf(role,
                       {required("id", INT32), required("element", INT32_LIST), required("deformation", REAL_LIST)});
    case TableRole::InertialLoads:
        // O_o_Reader inertialLoadsTable (contract 31): one row per GRAV, ACCEL, ACCEL1 and RFORCE a load set
        // reaches, LOAD factors applied; directionX/y/z a unit vector in the basic system (the acceleration of GRAV,
        // ACCEL, ACCEL1, the axis of RFORCE), acceleration its magnitude, scaled; RFORCE: centerNodeId (G, null when
        // blank), centerX/y/z (G or the origin), angularVelocity and angularAcceleration as given (scale
        // multiplies the load, not them), method; ACCEL1: nodeIds; ACCEL: locationDirection, locations and values. A
        // column that does not apply to a card is null.
        return tableOf(role,
                       {required("setId", INT32), required("sourceSetId", INT32), required("card", TEXTS),
                        required("frameId", INT32), required("centerNodeId", INT32), required("method", INT32),
                        required("nodeIds", INT32_LIST), required("locationDirection", TEXTS),
                        required("locations", REAL_LIST), required("values", REAL_LIST)},
                       {"scale", "directionX", "directionY", "directionZ", "acceleration", "centerX", "centerY",
                        "centerZ", "angularVelocity", "angularAcceleration"});
    case TableRole::LoadCombinations:
        // One row per LOAD entry and set it reaches, sorted by (load, loadSet); scale S and factor
        // Si (the product through nested entries) as the deck gives them, combinedFactor S x factor (summed when a set
        // is reached twice). unit_subcases and set_cards are not read.
        return tableOf(role, {required("load", INT32), required("loadSet", INT32)},
                       {"scale", "factor", "combinedFactor"});
    case TableRole::Subcases:
        return tableOf(role, {required("subcase", INT32),
                              required("kind", TEXTS),
                              required("title", UTF8),
                              required("subtitle", UTF8),
                              required("label", UTF8),
                              required("spc", INT32),
                              required("mpc", INT32),
                              required("load", INT32),
                              required("dynamicLoadSelection", INT32),
                              required("method", INT32),
                              required("temperatureLoad", INT32),
                              required("temperatureInitial", INT32),
                              required("outputs", "list<utf8>"),
                              optional("combinedSubcases", INT32_LIST),
                              optional("coefficients", REAL_LIST),
                              // The contact table the subcase's BCONTACT selects, effective (null: none).
                              optional("contactSelection", INT32),
                              // The element deformation set its DEFORM selects, effective: int32 as O_o_Reader gives
                              // it (contract 59), float64 (NaN: none) as the snapshot adds it from the Case Control
                              // for an older O_o_Reader; absent: no subcase selects one.
                              {"deform", INT32, true, {FLOAT64}}});
    case TableRole::Params:
        // O_o_Reader paramsTable (contract 31): one row per PARAM name (its neutral name: WeightToMass) of the Bulk
        // Data (subcase null) and of each Case Control section (subcase 0 the global section); kind "integer", "real",
        // "text", "complex" or "blank", the value in the column it names (a complex value: realValue and imagValue).
        return tableOf(role, {required("name", TEXTS), required("subcase", INT32), required("kind", TEXTS),
                              required("intValue", INT32), required("realValue", FLOAT64), required("textValue", UTF8),
                              optional("imagValue", FLOAT64)});
    case TableRole::Components:
        // isLineOnly (contract 34; else shellCount and solidCount) tells the components made only of 1D and 0D
        // elements (the connection system); optional, so that a table without them reads every
        // component as structural.
        return tableOf(role, {required("componentId", INT64), required("elementCount", INT64),
                              required("nodeCount", INT64), required("revision", INT64), optional("shellCount", INT64),
                              optional("solidCount", INT64), optional("isLineOnly", BOOL)});
    case TableRole::ComponentElements:
        return tableOf(role, {required("elementId", INT32), required("type", TEXTS), required("propId", INT32),
                              required("componentId", INT64)});
    case TableRole::Boundaries:
        return tableOf(role, {required("componentId", INT64), required("loadGroup", INT64), required("nodeId", INT32),
                              required("isInterface", BOOL), required("isRetained", BOOL)});
    case TableRole::LoadGroups:
        return tableOf(role, {required("componentId", INT64), required("loadGroup", INT64), required("subcase", INT32),
                              required("loadSetId", INT32), required("loadedNodeCount", INT64),
                              required("isDistributed", BOOL), required("loadRevision", INT64)});
    case TableRole::NonStructuralMass:
        return tableOf(role,
                       {required("setId", INT32), required("card", TEXTS), required("targetType", TEXTS),
                        required("targetId", INT32), required("kind", TEXTS)},
                       {"value", "mass"});
    case TableRole::OutputRequests:
        // O_o_Reader outputRequestsTable (contract 31): one row per effective request of each subcase (the
        // global requests inherited), the neutral request name (DisplacementOutput), describers (PrintListing or
        // ReferenceNode=12) in their given order; set is the selection word (All, None, Yes, No) and setId the named
        // set, both null for a request without a selection.
        return tableOf(role, {required("subcase", INT32), required("request", TEXTS), required("describers", TEXT_LIST),
                              required("set", TEXTS), required("setId", INT32)});
    case TableRole::CaseSets:
        // O_o_Reader caseSetsTable (contract 31): one row per member of each Case Control SET, in the order of
        // the set (THRU, BY, EXCEPT expanded); subcase 0 for the global section; id a member of a set of IDs (null for
        // a set of reals or of words); section "case" for the sets the requests name.
        return tableOf(role, {required("subcase", INT32), required("setId", INT32), required("position", INT32),
                              required("id", INT32), required("section", TEXTS)});
    case TableRole::CoordinateSystems:
        // The origin and the axes i, j, k of each system in the basic system, RID chains resolved by O_o_Reader;
        // definition (1 for a CORD1x, 2 for a CORD2x), referenceFrameId (a CORD2x's) and g1 to g3 (a CORD1x's grid
        // points, O_o_Reader contract 44), from which a design point derives a moved system again.
        return tableOf(
            role,
            {required("frameId", INT32), required("type", TEXTS), optional("definition", INT32),
             optional("referenceFrameId", INT32), optional("g1", INT32), optional("g2", INT32), optional("g3", INT32)},
            {"originX", "originY", "originZ", "iX", "iY", "iZ", "jX", "jY", "jZ", "kX", "kY", "kZ"});
    case TableRole::LaminatePlies:
        // One row per ply from the bottom, the plies of LAM = SYM included (isMirrored); zBottom and zTop from the
        // reference plane, Z0 applied; plyId the global ply ID of a PCOMPG (null for a PCOMP).
        return tableOf(
            role,
            {required("propId", INT32), required("type", TEXTS), required("ply", INT32), required("plyId", INT32),
             required("matId", INT32), required("sout", BOOL), required("isMirrored", BOOL)},
            {"thickness", "theta", "zBottom", "zTop"});
    case TableRole::LaminateSummary:
        // One row per laminate; lam is the lamination option as written (null when blank).
        return tableOf(role, {required("propId", INT32), required("type", TEXTS), required("laminateOption", TEXTS)},
                       {"thickness", "z0"});
    case TableRole::Sections:
        // O_o_Reader sectionsTable(types=["PBARL", "PBEAML"]) (contract 31): the section properties computed from the
        // dimensions of the section type, one row per property and station, sorted by propId then station (xxb from 0
        // at end A to 1 at end B), in the element system about the neutral axis; k1, k2 the shear factors; the recovery
        // points C, D, E, F as (y, z): c1 c2 d1 d2 e1 e2 f1 f2. n1, n2 (optional): the neutral axis from the shear
        // centre in the element system, as PBEAM N1, N2, which O_o_Solver's snapshot adds from
        // O_o_Reader's section_properties_batch (centroid − shear_center of each PBEAML station; 0 for a PBARL, whose
        // axis is its neutral axis); without them every section has its shear centre on its centroid.
        return tableOf(
            role,
            {required("propId", INT32), optional("station", INT32), optional("n1", FLOAT64), optional("n2", FLOAT64)},
            {"xxb", "area", "i1", "i2", "i12", "j", "nonStructMass", "k1", "k2", "c1", "c2", "d1", "d2", "e1", "e2",
             "f1", "f2"});
    case TableRole::FastenerPiercings:
        // O_o_Reader fastenerPiercings (contract 60): where each CFAST and CWELD pierces the shells, one row per
        // fastener and side ("A", "B"; a point-to-patch CWELD has side A only, a CWELD ALIGN no row: its ends are GA
        // and GB, which the element reads from its card); card "CFAST" or "CWELD", form its TYPE; pointGrid GA or GB
        // when the card gives it (null otherwise; the point is then that grid point); x y z the piercing point in the
        // basic system (null when not found); element and elementType the shell pierced (element null, elementType
        // GRIDID for the face of grid points of a CWELD GRIDID); xi eta its natural coordinates there; nodeIds the grid
        // points of the element and shapeValues its shape functions at the point (they sum to 1: the weights of the
        // interpolation of the point, elements/fasteners/PiercingInterpolation); field and problem, for a side not
        // found, the field of the card concerned and what the reader could not find (the element reports it). The
        // table replaces an earlier table of the reader, which the solver no longer reads.
        return tableOf(role,
                       {required("elementId", INT32), required("card", TEXTS), required("form", TEXTS),
                        required("side", TEXTS), required("pointGrid", INT32), required("element", INT32),
                        required("elementType", TEXTS), required("nodeIds", INT32_LIST),
                        required("shapeValues", REAL_LIST), required("field", TEXTS), required("problem", UTF8)},
                       {"x", "y", "z", "xi", "eta"});
    case TableRole::AeroReference:
        // O_o_Reader aeroReferenceTable (contract 25): one row per AERO and AEROS card (the AERO rows first);
        // aeroFrameId the aerodynamic system (x along the flow), rigidBodyFrameId the reference system of the rigid
        // body motions and of the stability derivatives, the reference chord, span and area, the symmetry keys (Case
        // Control AESYMXZ and AESYMXY applied).
        return tableOf(role, {required("card", TEXTS), required("aeroFrameId", INT32),
                              required("rigidBodyFrameId", INT32), required("velocity", FLOAT64),
                              required("refChord", FLOAT64), required("refSpan", FLOAT64), required("refArea", FLOAT64),
                              required("refDensity", FLOAT64), required("symXz", INT32), required("symXy", INT32)});
    case TableRole::AeroPanels:
        // O_o_Reader aeroPanelsTable: one row per CAERO1 to CAERO5 card, its type and the IDs of its boxes.
        return tableOf(role, {required("elementId", INT32), required("type", TEXTS), required("firstBox", INT32),
                              required("boxCount", INT64)});
    case TableRole::AeroBoxes:
        // O_o_Reader aeroBoxesTable: one row per box of a CAERO1, CAERO4 or CAERO5 panel in the basic system,
        // corners 1 and 2 on the inboard edge (leading, trailing), 3 and 4 on the outboard edge (trailing, leading);
        // control the point at three quarters of the chord at mid span; chord the length along the flow at mid span;
        // normal the unit normal of the box.
        return tableOf(role, {required("boxId", INT32), required("elementId", INT32)},
                       {"corner1X", "corner1Y", "corner1Z", "corner2X", "corner2Y", "corner2Z", "corner3X", "corner3Y",
                        "corner3Z", "corner4X", "corner4Y", "corner4Z", "centerX",  "centerY",  "centerZ",  "controlX",
                        "controlY", "controlZ", "area",     "chord",    "normalX",  "normalY",  "normalZ"});
    case TableRole::ControlSurfaces:
        // O_o_Reader controlSurfacesTable: one row per AESURF; frameId1 the system whose y axis is the hinge line of
        // its first component, boxes the boxes of its AELIST cards; effectiveness EFF; refChord and refArea CREFC and
        // CREFS, the reference values of its hinge moment coefficient; the position and hinge moment limits PLLIM,
        // PULIM, HMLLIM, HMULIM (null for none).
        return tableOf(
            role,
            {required("id", INT32), required("label", UTF8), required("frameId1", INT32), required("frameId2", INT32),
             required("aeroListId2", INT32), required("effectiveness", FLOAT64), optional("refChord", FLOAT64),
             optional("refArea", FLOAT64), optional("positionLower", FLOAT64), optional("positionUpper", FLOAT64),
             optional("hingeMomentLower", FLOAT64), optional("hingeMomentUpper", FLOAT64),
             optional("aeroListId1", INT32), required("boxes", INT32_LIST)});
    case TableRole::TrimCases:
        // O_o_Reader aeroTrimCasesTable: one row per variable of each TRIM (or TRIM2) set, the subcases that select
        // it, its Mach number, dynamic pressure and AEQR, the variable's label, value (when fixed), state ("fixed",
        // "free", "linked", "scheduled") and kind ("rigid_body", "control_surface", "user").
        return tableOf(role, {required("setId", INT32), required("subcases", INT32_LIST), required("mach", FLOAT64),
                              required("dynamicPressure", FLOAT64), optional("equivalentAspectRatio", FLOAT64),
                              required("variable", UTF8), required("value", FLOAT64), required("isFixed", BOOL),
                              required("state", TEXTS), required("variableKind", TEXTS)});
    case TableRole::Splines:
        // O_o_Reader splinesTable: one row per SPLINE card; caero the panel, firstBox and lastBox BOX1 and BOX2
        // (SPLINE1, SPLINE2) or the extremes of the AELIST (SPLINE4, SPLINE5), aelist its AELIST (null for SPLINE1,
        // SPLINE2), setg the structural set, dz DZ, method METH, usage FORCE, DISP or BOTH.
        return tableOf(role, {required("elementId", INT32), required("type", TEXTS), required("aeroPanelId", INT32),
                              required("firstBox", INT32), required("lastBox", INT32), required("aeroBoxListId", INT32),
                              required("nodeSetId", INT32), required("dz", FLOAT64), required("method", TEXTS),
                              required("usage", TEXTS)});
    case TableRole::SplineNodes:
        // O_o_Reader splineNodes: one row per structural point of each spline, by increasing nodeId.
        return tableOf(role, {required("spline", INT32), required("nodeId", INT32)});
    case TableRole::DirectMatrixShapes:
        // O_o_Reader directMatricesTable(detail="matrices") (contract 57): one row per DMI, DMIG, DMIJ, DMIJI
        // or DMIK matrix; formCode FORM or IFO; rowCount and columnCount its shape; role the fixed name a solution
        // reads it by (DownwashMatrix, AeroPressureMatrix, AeroWeightingMatrix, AeroWeightingFactor; null otherwise);
        // inEffect false for a DMIJ, DMIJI or DMIK of such a name that a DMI also has.
        return tableOf(role, {required("name", TEXTS), required("card", TEXTS), required("formCode", INT32),
                              required("isComplex", BOOL), required("rowCount", INT64), required("columnCount", INT64),
                              required("termCount", INT64), optional("role", TEXTS), required("inEffect", BOOL)});
    case TableRole::DirectMatrices:
        // O_o_Reader directMatricesTable (contract 57): one row per stored term, row and column from 0; a DMI
        // labelled by its row and column numbers (rowId, columnId from 1, the components 0), a matrix at points by
        // its degrees of freedom (a DMIJ: the box ID and component of the js set).
        return tableOf(role,
                       {required("name", TEXTS), required("card", TEXTS), required("row", INT32),
                        required("column", INT32), required("rowId", INT32), required("rowComponent", INT32),
                        required("columnId", INT32), required("columnComponent", INT32)},
                       {"real", "imag"});
    case TableRole::Supports:
        // O_o_Reader dofSetsTable(cards=["SUPORT", "SUPORT1"]) (contract 31): one row per (card, point); dof
        // the components as text; setId the SUPORT1 set (null for SUPORT); subcases the subcases whose Case Control
        // selects the SUPORT1 set (null for a SUPORT, which every subcase takes). The table may hold the other DOF set
        // cards (ASET, OMIT, ...): the readers keep SUPORT and SUPORT1.
        return tableOf(role, {required("card", TEXTS), required("setId", INT32), required("nodeId", INT32),
                              required("dof", UTF8), required("subcases", INT32_LIST)});
    case TableRole::ConnectionNodes:
        // O_o_Reader connectedComponents(detail="nodes") (contract 34): every grid point of the model and every
        // point an element uses, by increasing nodeId; componentId its component, null for a point no component holds;
        // isConnectionOnly a point only connections use.
        return tableOf(role,
                       {required("nodeId", INT32), required("componentId", INT64), optional("isConnectionOnly", BOOL)});
    case TableRole::ConnectionClusters:
        // O_o_Reader connectionClusters(detail="connections") (contract 40; contract 39 lists the connections
        // of the clusters only): one row per connection of the model, the rows of the clusters first, then those in no
        // cluster (the direct links between two components) by card type name and ID with a null clusterId; kind
        // "rigid", "rigidInterpolation", "mpc", "fastener" (CFAST, CWELD), "connector" (springs, dampers, bushes, gaps,
        // a 1D element or a mass between two components), "contact" (a pair of contact bodies, contact="connection"
        // only) or "mass" (a mass on a point only connections use); card and id the card and its element ID (the set ID
        // of an MPC equation, one row per equation); nodeIds the grid points it acts on in the order of the card (a
        // CFAST or CWELD the grid points of the elements it pierces and its GA, GB, never GS); revision the 64-bit
        // stamp of its card. The solver computes the clusters itself (reduction::ConnectionSystem: a regrouping or a
        // moved 1D unit joins them again), so clusterId is not read.
        return tableOf(role, {required("clusterId", INT64), required("kind", TEXTS), required("card", TEXTS),
                              required("id", INT32), required("nodeIds", INT32_LIST), required("revision", INT64)});
    case TableRole::ContactNodePairs:
        // O_o_Reader contactNodePairs (contract 38), taken with the reader's own rule (rule="components"): one row per
        // grid point of a secondary surface near the primary surface, with a face of that surface; table the card and
        // ID that pair the bodies ("ContactTable 5", null for the pairing of a Bulk Data without table); kind glued,
        // breakable or touching. The glue reads the rows as the two surfaces near each other and projects the points on
        // the faces itself (assembly::GlueConstraints), so the projection, local coordinates and gaps of the table are
        // not read. The glued-contact columns (contract 39) are optional here, ContactTable::missingGlueColumns says
        // which a glued contact needs: primaryFace the grid points of the face in the element's order (contract 40:
        // with the mid-side points of a quadratic face), searchedFrom "secondary" or "primary".
        return tableOf(role,
                       {required("table", UTF8), required("secondaryBody", INT32), required("primaryBody", INT32),
                        required("nodeId", INT32), required("primaryElementId", INT32), required("kind", TEXTS),
                        optional("primaryFace", INT32_LIST), ColumnContract{"searchedFrom", TEXTS, true, {UTF8}}});
    case TableRole::ContactPairs:
        // O_o_Reader contactPairsTable (contract 38): one row per pair of contact bodies of every contact
        // table of the Bulk Data; nodePairCount the rows of contactNodePairs of the pair. The glue columns are
        // optional here: the glue parameters in force (null when not given, an IGLUE null is blank: touching
        // contact) and the card that sets IGLUE. ERROR, when given, is the position tolerance of the glue.
        return tableOf(
            role,
            {required("table", UTF8), required("secondaryBody", INT32), required("primaryBody", INT32),
             required("kind", TEXTS), required("nodePairCount", INT64), optional("meets", BOOL),
             optional("glueOption", INT32), optional("glueSeparation", INT32), optional("coordinateCheck", INT32),
             optional("searchOption", INT32), optional("errorTolerance", FLOAT64), optional("toleranceBias", FLOAT64),
             optional("nearDistance", FLOAT64), ColumnContract{"glueOptionSource", TEXTS, true, {UTF8}}});
    case TableRole::ContactTables:
        // O_o_Reader contactTablesTable (contract 39): one row per contact table and pair of bodies; tableId
        // the ID a BCONTACT names, tableCard its card by its neutral name (ContactTable, ContactTable1, ContactPair,
        // ContactSet, ContactSetUnion, ContactSlideline), pairCard the card that pairs the bodies ("ContactTable 5",
        // "ContactPair 3": the `table` text of contactNodePairs), null for a table with no pair left.
        return tableOf(role, {required("tableId", INT32), ColumnContract{"tableCard", TEXTS, false, {UTF8}},
                              ColumnContract{"pairCard", UTF8, false, {TEXTS}}});
    case TableRole::Spiders:
        // O_o_Reader spiders(contact="merge") (contract 34): one row per RBE2 and RBE3, by elementId; componentId the
        // component of its legs (null when they are on several components or one on none); referenceNodeId GN of an
        // RBE2 or REFGRID of an RBE3, referenceDofs its components (REFC; "123456" for an RBE2); legNodeIds, legDofs
        // and legWeights (null for an RBE2) as the card gives them; isSpider false with its reason (um,
        // reference_is_leg, reference_dependent_elsewhere, legs_on_several_units) for a card that is not a spider;
        // isEligible no leg touched by another connection, blockingNodeIds and blockingConnections what touches them
        // (an MPC equation by its set ID); revision the stamp of the card and of what it refers to.
        return tableOf(role, {required("componentId", INT64), required("card", TEXTS), required("elementId", INT32),
                              required("referenceNodeId", INT32), required("referenceDofs", UTF8),
                              required("legNodeIds", INT32_LIST), required("legDofs", TEXT_LIST),
                              required("legWeights", REAL_LIST), required("isSpider", BOOL), required("reason", TEXTS),
                              required("isEligible", BOOL), required("blockingNodeIds", INT32_LIST),
                              required("blockingConnections", INT32_LIST), required("revision", INT64)});
    case TableRole::ConnectionSharing:
        // O_o_Reader connectionSharing(contact="merge") (contract 34): per grid point a connection touches, the
        // connections that touch it, aligned: kinds ("rigid", "rigidInterpolation", "mpc", "fastener", "connector",
        // "contact"), cards, ids (the element ID, the set ID of an MPC equation) and equations (the number of an MPC
        // equation in its set, 0 otherwise); by increasing nodeId.
        return tableOf(role,
                       {required("nodeId", INT32), required("connectionCount", INT64), required("kinds", TEXT_LIST),
                        required("cards", TEXT_LIST), required("ids", INT32_LIST), required("equations", INT32_LIST)});
    }
    return {};
}

base::ContractError missingTableError(TableRole role)
{
    const std::string_view prerequisite = prerequisiteOf(role);
    if (!isGivenByReader(role))
        return base::ContractError(
            base::concatenate({"the model snapshot has no table '", tableName(role), "' (", prerequisite,
                               "): O_o_Reader does not give it in this form; hand the table over to the snapshot "
                               "yourself"}));
    if (!prerequisite.empty())
        return base::ContractError(base::concatenate(
            {"the model snapshot has no table '", tableName(role), "' (", prerequisite,
             "): it was taken without it; take the snapshot again with that table from a O_o_Reader that gives it"}));
    return base::ContractError(
        base::concatenate({"the model snapshot has no table '", tableName(role),
                           "': it was taken without it; take the snapshot again with that table"}));
}

base::ContractReader readTable(TableRole role, base::ImportedTable table)
{
    return {std::move(table), contractOf(role)};
}

TableContract elementBatchContract(std::string_view card)
{
    return {std::string(card),
            {required(std::string(ELEMENT_ID_COLUMN), INT32), optional(std::string(ELEMENT_PROPERTY_COLUMN), INT32),
             optionalConnectivity("nodes"), optionalConnectivity("nodesRows"), optional("propIdRow", INT32),
             optional("propType", TEXTS)}};
}

TableContract cardTableContract(std::string_view card, std::string_view idColumn)
{
    return {std::string(card), {required(std::string(idColumn), INT32)}};
}

} // namespace oos::input
