#pragma once

#include <array>
#include <optional>
#include <string_view>

namespace oos::input
{

// The tables of a model snapshot that exist once per model (the element, property and method card tables exist once
// per card type and are not roles). The name of a role is the name of the O_o_Reader query or table it is taken from,
// and the name the caller hands it over with.
enum class TableRole
{
    Nodes,              // nodes: cardTable(deck, "nodes"), the grid points
    ScalarPoints,       // scalarPoints: cardTable(deck, "points"), the scalar and extra points
    Materials,          // materialsTable
    Spc,                // singlePointConstrTable
    Mpc,                // multiPointConstrTable
    RigidElements,      // rigidElementsTable
    Loads,              // loadsTable
    Pressures,          // pressureTable
    PressureFields,     // pressureFields: pressureTable(detail="fields"), a row per card and its runs (contract 55)
    LineLoads,          // lineLoadsTable
    Temperatures,       // temperatureTable
    Deformations,       // deformations: cardTable(deck, "AxialDeformationLoad"), the DEFORM cards
    InertialLoads,      // inertialLoadsTable: GRAV, ACCEL, ACCEL1, RFORCE
    LoadCombinations,   // loadCombinationsTable: the terms of every LOAD entry
    Subcases,           // subcasesTable
    Params,             // paramsTable: the typed PARAM table
    Components,         // connectedComponents
    ComponentElements,  // connectedComponents(detail="elements")
    Boundaries,         // componentBoundaries
    LoadGroups,         // loadCaseGroups
    NonStructuralMass,  // nonStructMassTable
    OutputRequests,     // outputRequestsTable: the Case Control output requests of each subcase
    CaseSets,           // caseSetsTable: one row per member of each Case Control SET
    CoordinateSystems,  // coordsTable: the CORDij systems resolved to the basic system
    LaminatePlies,      // laminatePliesTable: the ply stack of each PCOMP and PCOMPG, the SYM option expanded
    LaminateSummary,    // laminateSummary: one row per laminate, with its lamination option LAM
    Sections,           // sectionsTable(types=PBARL, PBEAML): section properties
    FastenerPiercings,  // fastenerPiercings: where CFAST and CWELD pierce the shells (contract 60)
    Supports,           // dofSetsTable(cards=SUPORT, SUPORT1): the DOFs of SUPORT and SUPORT1
    ConnectionNodes,    // componentNodes: connectedComponents(detail="nodes") (contract 34)
    ConnectionClusters, // connectionList: connectionClusters(detail="connections"), every connection (contract 40)
    ContactNodePairs,   // contactNodePairs: the grid points of glued contact
    ContactPairs,       // contactPairsTable: the pairs of contact bodies and their glue parameters
    ContactTables,      // contactTablesTable: the pairs each BCONTACT ID selects (optional)
    Spiders,            // spiders: the RBE2 and RBE3 as spiders of a component, their eligibility (contract 34)
    ConnectionSharing,  // connectionSharing: per grid point, the connections that touch it (contract 34)
    AeroReference,      // aeroReferenceTable: AERO and AEROS
    AeroPanels,         // aeroPanelsTable: the CAERO1 to CAERO5 panels
    AeroBoxes,          // aeroBoxesTable: the boxes of the doublet-lattice panels, in the basic system
    ControlSurfaces,    // controlSurfacesTable: the AESURF cards with their boxes
    TrimCases,          // aeroTrimCasesTable: the TRIM cards, one row per trim variable
    Splines,            // splinesTable: the SPLINE1 to SPLINE7 cards
    SplineNodes,        // splineNodes: the structural points of each spline
    DirectMatrixShapes, // directMatrixShapes: directMatricesTable(detail="matrices"), a row a matrix (contract 57)
    DirectMatrices,     // directMatricesTable(full=True): the terms of DownwashMatrix, AeroPressureMatrix,
                        // AeroWeightingMatrix, AeroWeightingFactor (contract 57)
};

inline constexpr std::array<TableRole, 45> ALL_TABLE_ROLES{TableRole::Nodes,
                                                           TableRole::ScalarPoints,
                                                           TableRole::Materials,
                                                           TableRole::Spc,
                                                           TableRole::Mpc,
                                                           TableRole::RigidElements,
                                                           TableRole::Loads,
                                                           TableRole::Pressures,
                                                           TableRole::PressureFields,
                                                           TableRole::LineLoads,
                                                           TableRole::Temperatures,
                                                           TableRole::Deformations,
                                                           TableRole::InertialLoads,
                                                           TableRole::LoadCombinations,
                                                           TableRole::Subcases,
                                                           TableRole::Params,
                                                           TableRole::Components,
                                                           TableRole::ComponentElements,
                                                           TableRole::Boundaries,
                                                           TableRole::LoadGroups,
                                                           TableRole::NonStructuralMass,
                                                           TableRole::OutputRequests,
                                                           TableRole::CaseSets,
                                                           TableRole::CoordinateSystems,
                                                           TableRole::LaminatePlies,
                                                           TableRole::LaminateSummary,
                                                           TableRole::Sections,
                                                           TableRole::FastenerPiercings,
                                                           TableRole::Supports,
                                                           TableRole::ConnectionNodes,
                                                           TableRole::ConnectionClusters,
                                                           TableRole::ContactNodePairs,
                                                           TableRole::ContactPairs,
                                                           TableRole::ContactTables,
                                                           TableRole::Spiders,
                                                           TableRole::ConnectionSharing,
                                                           TableRole::AeroReference,
                                                           TableRole::AeroPanels,
                                                           TableRole::AeroBoxes,
                                                           TableRole::ControlSurfaces,
                                                           TableRole::TrimCases,
                                                           TableRole::Splines,
                                                           TableRole::SplineNodes,
                                                           TableRole::DirectMatrixShapes,
                                                           TableRole::DirectMatrices};

// The name of a role: "singlePointConstrTable", "nodes".
[[nodiscard]] std::string_view tableName(TableRole role) noexcept;

// The role of a table name, or nothing for a name that is not a role.
[[nodiscard]] std::optional<TableRole> findTableRole(std::string_view name) noexcept;

// A brief description of an optional table for missing-table diagnostics; empty when no extra description is needed.
[[nodiscard]] std::string_view prerequisiteOf(TableRole role) noexcept;

// Whether O_o_Reader gives the table of this role (rather than the caller of the snapshot): every role since data
// contract 40, which gives connectionList (every connection with its stamp). Kept for a table O_o_Reader may not
// give yet in a later version of the solver.
[[nodiscard]] bool isGivenByReader(TableRole role) noexcept;

} // namespace oos::input
