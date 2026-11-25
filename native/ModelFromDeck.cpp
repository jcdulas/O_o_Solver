#include "oos/api/ModelFromDeck.hpp"

#include "oor/api/CardTables.hpp"
#include "oor/api/Design.hpp"
#include "oor/api/Queries.hpp"
#include "oor/query/table/ArrowExport.hpp"
#include "oos/input/snapshot/SnapshotBuilder.hpp"

#include <algorithm>
#include <stdexcept>

namespace oos::api
{
namespace
{
using Table = std::shared_ptr<const oor::query::QueryTable>;

base::ImportedTable importTable(const Table& table)
{
    // Release either output on partial export. takeOver clears both release pointers on transfer.
    struct Batch
    {
        ArrowSchema schema{};
        ArrowArray array{};
        ~Batch()
        {
            if (array.release) array.release(&array);
            if (schema.release) schema.release(&schema);
        }
    } batch;
    oor::query::ArrowExport::exportSchema(*table, &batch.schema);
    oor::query::ArrowExport::exportArray(*table, &batch.array);
    return base::ImportedTable::takeOver(&batch.schema, &batch.array);
}

bool contains(const std::vector<std::string>& columns, const std::string& name)
{
    return std::find(columns.begin(), columns.end(), name) != columns.end();
}
} // namespace

std::shared_ptr<const input::ModelSnapshot> modelFromDeck(const std::shared_ptr<oor::api::Deck>& deck,
                                                          const DeckOptions& options)
{
    if (!deck) throw std::invalid_argument("modelFromDeck requires a deck");
    input::SnapshotBuilder builder;
    const auto versions = oor::api::modelVersions(*deck);
    builder.setVersions(versions.structure, versions.content);
    if (!options.elementFormulation.empty()) builder.setModelFormulation(options.elementFormulation);
    const auto counts = deck->cardCounts();
    std::vector<input::SourceCard> sourceCards;
    for (const auto& [name, count] : counts)
        sourceCards.push_back({name, static_cast<std::int64_t>(count)});
    builder.setSourceCards(std::move(sourceCards));
    const auto has = [&](const char* name) {
        return std::any_of(counts.begin(), counts.end(), [&](const auto& c) { return c.first == name && c.second; });
    };
    const auto card = [&](const std::string& name, const std::vector<std::string>& columns = {}) {
        return importTable(oor::api::cardTable(*deck, name, columns));
    };
    const auto put = [&](const char* role, base::ImportedTable table) {
        builder.setTable(input::findTableRole(role).value(), std::move(table));
    };
    const auto query = [&](const char* name, const char* json = "{}", const char* role = nullptr) {
        auto args = oor::api::argumentsFromJson(name, json);
        put(role ? role : name, importTable(oor::api::runQuery(name, deck, nullptr, args)));
    };

    const auto nodes = oor::api::cardTable(*deck, "nodes", {"id", "positionFrameId", "displacementFrameId",
                                                            "permanentConstraints", "superelementId", "xyzGlobal"});
    put("nodes", importTable(nodes));
    const auto frames = nodes->column("displacementFrameId").as<std::int32_t>();
    if (std::any_of(frames.begin(), frames.end(), [](auto id) { return id != 0; }))
    {
        auto rotations = std::make_shared<std::vector<double>>(oor::api::displacementRotations(*deck));
        builder.setCdRotations({*rotations, rotations});
    }
    const auto points = oor::api::cardTypesOf(*deck, "points");
    std::vector<std::string> scalar;
    for (const auto& type : points)
        if (type == "ScalarPoint" || type == "ExtraPoint") scalar.push_back(type);
    if (!scalar.empty())
    {
        if (points.size() != scalar.size() && scalar.size() != 1)
            throw std::invalid_argument("ambiguous scalar point types beside other point cards");
        put("scalarPoints", card(points.size() == scalar.size() ? "points" : scalar.front(), {"id"}));
    }
    if (has("ModelParams"))
    {
        const auto parameters = oor::api::cardTable(*deck, "ModelParams", {"name", "text"});
        for (std::int64_t row = 0; row < parameters->rowCount(); ++row)
            if (parameters->column("name").textAt(row) == "OffsetDefinition")
                builder.setOffsetDefinitionValue(parameters->column("text").textAt(row));
    }
    const auto optional = [&](const char* type, const char* role, const std::vector<std::string>& columns) {
        if (has(type)) put(role, card(type, columns));
    };
    optional("ShellNormal", "shellNormals", {"nodeId", "frameId", "vector"});
    optional("SurfaceEdge", "surfaceEdges", {"id", "node1Id", "node2Id", "state", "elementId"});
    optional("SurfaceFacet", "surfaceFacets", {"elementId"});
    optional("SurfaceLine", "surfaceLines", {"id", "node1Id", "throughNodeId", "node2Id"});
    optional("Bar2Stations", "barStations", {"id", "stationScale", "listedStations", "stationCount", "station1", "stationSpacing"});
    optional("AxialDeformationLoad", "deformations", {"id", "element", "deformation"});

    for (const auto& name : oor::api::cardTypesOf(*deck, "elements"))
    {
        auto columns = oor::api::cardTableColumns(name);
        for (const char* index : {"nodesRows", "propIdRow", "propType"})
            if (contains(columns.computed, index)) columns.stored.emplace_back(index);
        builder.addElementBatch(name, card(name, columns.stored));
    }
    for (const auto& name : oor::api::cardTypesOf(*deck, "properties"))
        if (contains(oor::api::cardTableColumns(name).stored, "id")) builder.addPropertyTable(name, card(name));
    for (const char* name : {"LanczosEigenMethod", "RealEigenMethod", "BucklingEigenMethod"})
        if (has(name)) builder.addMethodTable(name, "id", card(name));
    for (const char* name : {"IsoMat", "OrthoMat2D", "AnisoMat2D", "AnisoMat3D"})
        if (has(name)) builder.addMaterialCardTable(name, card(name));
    for (const std::string name : {"PointForceLoad", "PointForceLoad1", "PointForceLoad2", "PointMomentLoad",
             "PointMomentLoad1", "PointMomentLoad2", "ScalarPointLoad", "PointsPressureLoad", "BeamDistributedLoad",
             "ShellPressureLoad", "FacePressureLoad", "LoadCombination"})
    {
        if (!has(name.c_str())) continue;
        const auto available = oor::api::cardTableColumns(name);
        std::vector<std::string> columns{"id", "significantDigits"};
        if (name == "LoadCombination") columns.emplace_back("termLoad");
        if (std::all_of(columns.begin(), columns.end(), [&](const auto& col) {
                return contains(available.stored, col) || contains(available.computed, col);
            }))
            builder.addLoadCardTable(name, card(name, columns));
    }

    for (const char* name : {"materialsTable", "multiPointConstrTable", "rigidElementsTable", "temperatureTable",
             "subcasesTable", "nonStructMassTable", "coordsTable", "laminatePliesTable", "laminateSummary",
             "paramsTable", "outputRequestsTable", "caseSetsTable", "fastenerPiercings"})
        query(name);
    for (const char* name : {"loadsTable", "lineLoadsTable", "singlePointConstrTable", "inertialLoadsTable"})
        query(name, R"({"expandCombinations":false})");
    query("pressureTable", R"({"expandCombinations":false,"detail":"fields"})", "pressureFields");
    query("loadCombinationsTable", R"({"columns":["load","scale","loadSet","factor","combinedFactor"]})");
    query("sectionsTable", R"({"types":["BarShapeProp","BeamShapeProp"]})");
    query("dofSetsTable", R"({"cards":["FreeBodySupport","FreeBodySupportSet"]})");
    if (has("ContactTable") || has("ContactTable1") || has("ContactPair") || has("ContactSet") ||
        has("ContactSetUnion") || has("ContactSlideline"))
    {
        query("contactNodePairs", R"({"rule":"components"})");
        query("contactPairsTable", R"({"rule":"components"})");
        query("contactTablesTable");
    }
    if (options.condensation)
    {
        query("connectedComponents");
        query("connectedComponents", R"({"detail":"elements"})", "componentElements");
        query("componentBoundaries");
        query("connectedComponents", R"({"detail":"nodes"})", "componentNodes");
        query("connectionClusters", R"({"detail":"connections"})", "connectionList");
        query("spiders");
        query("connectionSharing");
    }
    if (has("AeroLiftingPanel1") || has("AeroBody2") || has("AeroMachBoxPanel3") || has("AeroStripPanel4") ||
        has("AeroPistonPanel5"))
    {
        for (const char* name : {"aeroReferenceTable", "aeroPanelsTable", "aeroBoxesTable", "controlSurfacesTable",
                                  "aeroTrimCasesTable", "splinesTable", "splineNodes"}) query(name);
        for (const char* name : {"AeroBeamSpline2", "AeroBeamSpline5", "AeroBoxList", "AeroRigidTrimVariable",
                                 "AeroTrimLink"})
            if (has(name)) builder.addAeroCardTable(name, "id", card(name));
    }
    return std::make_shared<const input::ModelSnapshot>(builder.build());
}

std::shared_ptr<OptimizationProblem> prepareOptimization(const std::shared_ptr<oor::api::Deck>& deck,
                                                         OptimizationSettings settings, const DeckOptions& options,
                                                         OptimizationInput declarations)
{
    if (!deck)
        throw std::invalid_argument("prepareOptimization requires a deck");
    const auto design = oor::api::designModel(*deck);
    if (!design.problems().empty())
        throw std::invalid_argument(design.problems().front());
    if (!design.notApplied().empty())
    {
        const auto& entry = design.notApplied().front();
        throw std::invalid_argument(entry.card + " " + std::to_string(entry.id) + ": " + entry.reason);
    }
    std::map<std::string, base::ImportedTable> tables;
    for (const char* name : {"designVariablesTable", "designLinksTable", "designRelationsTable", "designConstantsTable",
                             "shapeVariablesTable", "designEquationSymbols", "designEquationInstructions",
                             "designResponsesTable", "designResponseArgumentsTable", "designConstraintsTable",
                             "designConstraintSetsTable", "designSubcasesTable", "designScreeningTable"})
        tables.emplace(name,
                       importTable(oor::api::runQuery(name, deck, nullptr, oor::api::argumentsFromJson(name, "{}"))));
    return std::make_shared<OptimizationProblem>(modelFromDeck(deck, options), std::move(declarations),
                                                 std::move(settings), std::move(tables));
}
} // namespace oos::api
