#include "oos/input/contract/TableRole.hpp"

namespace oos::input
{

std::string_view tableName(TableRole role) noexcept
{
    switch (role)
    {
    case TableRole::Nodes:
        return "nodes";
    case TableRole::ScalarPoints:
        return "scalarPoints";
    case TableRole::Materials:
        return "materialsTable";
    case TableRole::Spc:
        return "singlePointConstrTable";
    case TableRole::Mpc:
        return "multiPointConstrTable";
    case TableRole::RigidElements:
        return "rigidElementsTable";
    case TableRole::Loads:
        return "loadsTable";
    case TableRole::Pressures:
        return "pressureTable";
    case TableRole::PressureFields:
        return "pressureFields";
    case TableRole::LineLoads:
        return "lineLoadsTable";
    case TableRole::Temperatures:
        return "temperatureTable";
    case TableRole::Deformations:
        return "deformations";
    case TableRole::InertialLoads:
        return "inertialLoadsTable";
    case TableRole::LoadCombinations:
        return "loadCombinationsTable";
    case TableRole::Subcases:
        return "subcasesTable";
    case TableRole::Params:
        return "paramsTable";
    case TableRole::Components:
        return "connectedComponents";
    case TableRole::ComponentElements:
        return "componentElements";
    case TableRole::Boundaries:
        return "componentBoundaries";
    case TableRole::LoadGroups:
        return "loadCaseGroups";
    case TableRole::NonStructuralMass:
        return "nonStructMassTable";
    case TableRole::OutputRequests:
        return "outputRequestsTable";
    case TableRole::CaseSets:
        return "caseSetsTable";
    case TableRole::CoordinateSystems:
        return "coordsTable";
    case TableRole::LaminatePlies:
        return "laminatePliesTable";
    case TableRole::LaminateSummary:
        return "laminateSummary";
    case TableRole::Sections:
        return "sectionsTable";
    case TableRole::FastenerPiercings:
        return "fastenerPiercings";
    case TableRole::Supports:
        return "dofSetsTable";
    case TableRole::ConnectionNodes:
        return "componentNodes";
    case TableRole::ConnectionClusters:
        return "connectionList";
    case TableRole::ContactNodePairs:
        return "contactNodePairs";
    case TableRole::ContactPairs:
        return "contactPairsTable";
    case TableRole::ContactTables:
        return "contactTablesTable";
    case TableRole::Spiders:
        return "spiders";
    case TableRole::ConnectionSharing:
        return "connectionSharing";
    case TableRole::AeroReference:
        return "aeroReferenceTable";
    case TableRole::AeroPanels:
        return "aeroPanelsTable";
    case TableRole::AeroBoxes:
        return "aeroBoxesTable";
    case TableRole::ControlSurfaces:
        return "controlSurfacesTable";
    case TableRole::TrimCases:
        return "aeroTrimCasesTable";
    case TableRole::Splines:
        return "splinesTable";
    case TableRole::SplineNodes:
        return "splineNodes";
    case TableRole::DirectMatrixShapes:
        return "directMatrixShapes";
    case TableRole::DirectMatrices:
        return "directMatricesTable";
    }
    return "";
}

std::optional<TableRole> findTableRole(std::string_view name) noexcept
{
    for (const TableRole role : ALL_TABLE_ROLES)
    {
        if (tableName(role) == name)
            return role;
    }
    return std::nullopt;
}

std::string_view prerequisiteOf(TableRole role) noexcept
{
    switch (role)
    {
    case TableRole::InertialLoads:
        return "the inertial loads, expanded";
    case TableRole::LoadCombinations:
        return "the LoadCombination combinations left unexpanded";
    case TableRole::PressureFields:
        return "the compact pressure fields";
    case TableRole::Params:
        return "the typed Parameter table";
    case TableRole::OutputRequests:
    case TableRole::CaseSets:
        return "the output requests and sets, per subcase";
    case TableRole::Sections:
        return "the section properties of BarShapeProp and BeamShapeProp";
    case TableRole::FastenerPiercings:
        return "the piercing points of Fastener and Weld on the shells, with the shape functions of the "
               "elements pierced";
    case TableRole::Supports:
        return "the DOFs of AnalysisDofSet, OmittedDofSet and FreeBodySupport";
    case TableRole::ConnectionNodes:
        return "the connection system and clusters";
    case TableRole::ConnectionClusters:
        return "the connection system and clusters: every connection with its stamp, data contract 40";
    case TableRole::ContactNodePairs:
    case TableRole::ContactPairs:
    case TableRole::ContactTables:
        return "glued contact";
    case TableRole::Spiders:
    case TableRole::ConnectionSharing:
        return "spiders and connection sharing";
    case TableRole::AeroReference:
    case TableRole::AeroPanels:
    case TableRole::AeroBoxes:
    case TableRole::ControlSurfaces:
    case TableRole::TrimCases:
    case TableRole::Splines:
    case TableRole::SplineNodes:
        return "the aeroelastic model";
    case TableRole::DirectMatrixShapes:
    case TableRole::DirectMatrices:
        return "the direct matrix input of SOL 144: DownwashMatrix and AeroPressureMatrix";
    default:
        return "";
    }
}

bool isGivenByReader(TableRole /*role*/) noexcept
{
    // Every role since contract 40: connectionList is connectionClusters(detail="connections").
    return true;
}

} // namespace oos::input
