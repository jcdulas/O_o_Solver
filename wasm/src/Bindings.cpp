// The JavaScript interface of the solver (embind). It only converts between JavaScript values and the plain C++ of
// oos::api; every decision is taken there. JavaScript sees:
//
//   new ModelInput()                 .setVersions .setTable .addElementTable ... .setElementFormulation
//                                    .build() -> Model
//   elementFormulations()            -> [{ card, formulations }], the card its neutral type ("ShellQuad4"), the default
//                                       formulation of each card first
//   Model                            .tableNames() .checkSupport(analysis) .surfaceEdges()
//   runAnalysis(model, analysis, settings) -> { nodalTables, elementTables, gridPointForces, eigenvalues, diagnostics,
//                                               condensation }
//   condensationCache(memoryBytes)   -> CondensationCache .statistics(), for settings.condensation.cache
//   runSuperelements(residual, parts, analysis, settings) -> { superelements, diagnostics }
//
// The names are those of oos::api, the neutral names of O_o_Model (card types, table kinds, columns), passed through
// untranslated. Errors of the solver become JavaScript errors whose message is the solver's (what was expected, what
// was found, and what to do).

#include "Model.hpp"
#include "TrimResults.hpp"
#include "oos/api/AeroCoupling.hpp"
#include "oos/api/Analyses.hpp"
#include "oos/api/Capabilities.hpp"
#include "oos/api/ModelInput.hpp"
#include "oos/api/Optimization.hpp"
#include "oos/api/Threading.hpp"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using emscripten::val;

namespace
{

using namespace oos;

// Runs `work` and turns a C++ exception into a JavaScript Error with the same message.
template <typename Work>
auto guarded(Work work) -> decltype(work())
{
    try
    {
        return work();
    }
    catch (const std::exception& error)
    {
        val::global("Error").new_(std::string(error.what())).throw_();
    }
    catch (...)
    {
        val::global("Error").new_(std::string("unknown error in the solver")).throw_();
    }
    __builtin_unreachable(); // throw_() does not return
}

// A typed array that owns a copy of the values (the C++ buffer may go away after the call).
template <typename T>
val typedArray(const char* type, const T* data, std::size_t count)
{
    const val view(emscripten::typed_memory_view(count, data));
    // A number, not a size_t: in 64-bit WebAssembly a size_t reaches JavaScript as a BigInt, which the typed array
    // constructors refuse.
    val copy = val::global(type).new_(static_cast<double>(count));
    copy.call<void>("set", view);
    return copy;
}

val textArray(const std::vector<std::string>& texts)
{
    val array = val::array();
    for (const std::string& text : texts)
        array.call<void>("push", text);
    return array;
}

std::vector<std::uint8_t> bytesOf(const val& array)
{
    return emscripten::convertJSArrayToNumberVector<std::uint8_t>(array);
}

std::vector<double> numbersOf(const val& array)
{
    return emscripten::convertJSArrayToNumberVector<double>(array);
}

// ------------------------------------------------------------------------------------------------- model input

// A built model: the snapshot the analyses read (immutable, shared).
using oos::wasm::Model;

std::shared_ptr<api::AeroCoupling> aeroTransfer(Model& model)
{
    return guarded([&] { return std::make_shared<api::AeroCoupling>(model.snapshot); });
}

val boxMotion(const api::AeroCoupling& transfer, const val& nodalMotion)
{
    return guarded([&] {
        const auto motion = transfer.boxMotion(numbersOf(nodalMotion));
        auto result = val::object();
        result.set("plunge", typedArray("Float64Array", motion.plunge.data(), motion.plunge.size()));
        result.set("slope", typedArray("Float64Array", motion.slope.data(), motion.slope.size()));
        return result;
    });
}

val structuralLoads(const api::AeroCoupling& transfer, const val& boxForces)
{
    return guarded([&] {
        const auto loads = transfer.structuralLoads(numbersOf(boxForces));
        return typedArray("Float64Array", loads.data(), loads.size());
    });
}

val capabilitiesObject(const api::OptimizationCapabilities& source)
{
    auto result = val::object();
    result.set("version", source.version);
    result.set("responses", textArray(source.responses));
    auto analyses = val::array();
    for (const auto& entry : source.analyses)
    {
        auto item = val::object();
        item.set("analysis", entry.analysis);
        item.set("solution", entry.solution);
        item.set("sizingFirstOrder", entry.sizingFirstOrder);
        item.set("shapeFirstOrder", entry.shapeFirstOrder);
        item.set("secondOrder", entry.secondOrder);
        item.set("restrictions", textArray(entry.restrictions));
        analyses.call<void>("push", item);
    }
    result.set("analyses", analyses);
    auto fields = val::array();
    for (const auto& entry : source.fields)
    {
        auto item = val::object();
        item.set("targetKind", entry.targetKind);
        item.set("targetCard", entry.targetCard);
        item.set("field", entry.field);
        item.set("type", entry.type);
        item.set("requiresItem", entry.requiresItem);
        fields.call<void>("push", item);
    }
    result.set("fields", fields);
    return result;
}

val optimizationCatalogue()
{
    return guarded([] { return capabilitiesObject(api::optimizationCapabilities()); });
}

val modelCapabilities(const Model& model)
{
    return guarded([&] { return capabilitiesObject(api::optimizationCapabilities(model.snapshot.get())); });
}

void setVersions(api::ModelInput& input, double structure, double content)
{
    guarded([&] { input.setVersions(static_cast<std::int64_t>(structure), static_cast<std::int64_t>(content)); });
}

void setSourceCards(api::ModelInput& input, const val& cards)
{
    guarded(
        [&]
        {
            std::vector<api::SourceCard> source;
            const unsigned size = cards["length"].as<unsigned>();
            for (unsigned i = 0; i < size; ++i)
                source.push_back(
                    {cards[i]["card"].as<std::string>(), static_cast<std::int64_t>(cards[i]["count"].as<double>())});
            input.setSourceCards(std::move(source));
        });
}

// setTable(name, package, added): `added` is null or {columnName: Float64Array, ...}.
void setTable(api::ModelInput& input, const std::string& name, const val& package, const val& added)
{
    guarded(
        [&]
        {
            std::vector<api::AddedColumn> columns;
            if (!added.isNull() && !added.isUndefined())
            {
                const val names = val::global("Object").call<val>("keys", added);
                const auto count = names["length"].as<unsigned>();
                for (unsigned index = 0; index < count; ++index)
                {
                    const auto columnName = names[index].as<std::string>();
                    columns.push_back({columnName, numbersOf(added[columnName])});
                }
            }
            input.setTable(name, bytesOf(package), std::move(columns));
        });
}

void addElementTable(api::ModelInput& input, const std::string& card, const val& package)
{
    guarded([&] { input.addElementTable(card, bytesOf(package)); });
}

void addPropertyTable(api::ModelInput& input, const std::string& card, const val& package)
{
    guarded([&] { input.addPropertyTable(card, bytesOf(package)); });
}

void addMethodTable(api::ModelInput& input, const std::string& card, const std::string& idColumn, const val& package)
{
    guarded([&] { input.addMethodTable(card, idColumn, bytesOf(package)); });
}

void addMaterialTable(api::ModelInput& input, const std::string& card, const val& package)
{
    guarded([&] { input.addMaterialTable(card, bytesOf(package)); });
}

void addLoadCardTable(api::ModelInput& input, const std::string& card, const val& package)
{
    guarded([&] { input.addLoadCardTable(card, bytesOf(package)); });
}

void addAeroCardTable(api::ModelInput& input, const std::string& card, const std::string& idColumn, const val& package)
{
    guarded([&] { input.addAeroCardTable(card, idColumn, bytesOf(package)); });
}

void setDisplacementRotations(api::ModelInput& input, const val& rotations)
{
    guarded([&] { input.setDisplacementRotations(numbersOf(rotations)); });
}

void setOffsetDefinition(api::ModelInput& input, const std::string& value)
{
    guarded([&] { input.setOffsetDefinition(value); });
}

void setElementFormulation(api::ModelInput& input, const std::string& card, const std::string& formulation)
{
    guarded([&] { input.setElementFormulation(card, formulation); });
}

void setModelFormulation(api::ModelInput& input, const std::string& formulation)
{
    guarded([&] { input.setModelFormulation(formulation); });
}

// setFormulationGeometry(lists): an object of flat arrays of IDs, the members of
// api::ModelInput::FormulationGeometryLists by name (a missing member is an empty list).
void setFormulationGeometry(api::ModelInput& input, const val& lists)
{
    const auto idsOf = [&](const char* name)
    {
        const val member = lists[name];
        return member.isUndefined() ? std::vector<std::int32_t>{}
                                    : emscripten::convertJSArrayToNumberVector<std::int32_t>(member);
    };
    guarded(
        [&]
        {
            api::ModelInput::FormulationGeometryLists geometry;
            geometry.smoothEdges = idsOf("smoothEdges");
            geometry.sharpEdges = idsOf("sharpEdges");
            geometry.rawEdges = idsOf("rawEdges");
            geometry.rawElements = idsOf("rawElements");
            geometry.lineContinuations = idsOf("lineContinuations");
            geometry.lineEnds = idsOf("lineEnds");
            input.setFormulationGeometry(geometry);
        });
}

val elementFormulations()
{
    val array = val::array();
    for (const api::ElementFormulations& entry : api::elementFormulations())
    {
        val object = val::object();
        object.set("card", entry.card);
        object.set("formulations", textArray(entry.formulations));
        array.call<void>("push", object);
    }
    return array;
}

// The errors of build() are about the tables the program handed over (a missing table, a column of the wrong type),
// named as the program named them (table names, columns, neutral card types): there is no model yet whose deck
// keywords they could show. The errors about the model itself come from checkSupport and runAnalysis (guardedOn).
std::shared_ptr<Model> buildModel(api::ModelInput& input)
{
    return guarded([&] { return std::make_shared<Model>(Model{input.build()}); });
}

val modelTableNames(Model& model)
{
    return textArray(model.snapshot->tableNames());
}

// ------------------------------------------------------------------------------------------------- analyses

api::Analysis analysisNamed(const std::string& name)
{
    if (name == "statics")
        return api::Analysis::Statics;
    if (name == "modes")
        return api::Analysis::Modes;
    if (name == "buckling")
        return api::Analysis::Buckling;
    if (name == "aeroelastic")
        return api::Analysis::Aeroelastic;
    throw std::invalid_argument("unknown analysis '" + name + "': expected statics, modes, buckling or aeroelastic");
}

val diagnosticObject(const base::Diagnostic& diagnostic)
{
    val object = val::object();
    object.set("severity", std::string(base::severityName(diagnostic.severity)));
    object.set("code", diagnostic.code);
    object.set("message", diagnostic.message);
    object.set("card", diagnostic.card);
    object.set("id", static_cast<double>(diagnostic.id));
    object.set("field", diagnostic.field);
    object.set("text", base::formatDiagnostic(diagnostic));
    return object;
}

val diagnosticArray(const std::vector<base::Diagnostic>& diagnostics)
{
    val array = val::array();
    for (const base::Diagnostic& diagnostic : diagnostics)
        array.call<void>("push", diagnosticObject(diagnostic));
    return array;
}

val checkSupport(Model& model, const std::string& analysis)
{
    return guarded(
        [&]
        {
            const api::SupportCheck check = analysis == "trimPreloadedBuckling"
                ? api::checkTrimPreloadedBuckling(*model.snapshot) : api::checkSupport(*model.snapshot, analysisNamed(analysis));
            val result = val::object();
            result.set("supported", textArray(check.supported));
            result.set("refusals", diagnosticArray(check.refusals));
            result.set("ignored", diagnosticArray(check.ignored));
            result.set("isComplete", check.refusals.empty());
            return result;
        });
}

// surfaceEdges(): { grid1, grid2, surface, state, origin, angle, kink, rounding }, one entry per edge of the surfaces
// that the formulation read from the mesh (api::SurfaceEdgeTable); every array is empty for a model that reads none.
val surfaceEdges(Model& model)
{
    return guarded(
        [&]
        {
            const api::SurfaceEdgeTable table = api::surfaceEdges(*model.snapshot);
            val result = val::object();
            result.set("grid1", typedArray("Int32Array", table.grid1.data(), table.grid1.size()));
            result.set("grid2", typedArray("Int32Array", table.grid2.data(), table.grid2.size()));
            result.set("surface", textArray(table.surface));
            result.set("state", textArray(table.state));
            result.set("origin", textArray(table.origin));
            result.set("angle", typedArray("Float64Array", table.angle.data(), table.angle.size()));
            result.set("kink", typedArray("Float64Array", table.kink.data(), table.kink.size()));
            result.set("rounding", typedArray("Float64Array", table.rounding.data(), table.rounding.size()));
            return result;
        });
}

const std::vector<std::string> NODAL_COMPONENTS = {"t1", "t2", "t3", "r1", "r2", "r3"};

// A nodal table: { kind, subcase, analysis, step, eigenvalue, components, ids, pointTypes, values }, values row-major
// (rows × 6), in the displacement system of each grid point.
val nodalTableObject(const output::NodalResultTable& table)
{
    val object = val::object();
    object.set("kind", table.kind);
    object.set("subcase", table.subcase);
    object.set("analysis", table.analysis);
    object.set("step", table.step);
    object.set("eigenvalue", table.eigenvalue);
    object.set("components", textArray(NODAL_COMPONENTS));
    object.set("ids", typedArray("Int32Array", table.ids.data(), table.ids.size()));
    object.set("pointTypes", typedArray("Int8Array", table.pointTypes.data(), table.pointTypes.size()));
    object.set("values", typedArray("Float64Array", table.values.data(), table.values.size()));
    return object;
}

// An element table: { kind, elementType, subcase, analysis, step, eigenvalue, components, ids, nodeIds, layers,
// positions?, values }, values row-major (rows × components).
val elementTableObject(const output::ElementResultTable& table)
{
    val object = val::object();
    object.set("kind", table.kind);
    object.set("elementType", table.elementType);
    object.set("subcase", table.subcase);
    object.set("analysis", table.analysis);
    object.set("step", table.step);
    object.set("eigenvalue", table.eigenvalue);
    object.set("components", textArray(table.components));
    object.set("ids", typedArray("Int32Array", table.ids.data(), table.ids.size()));
    object.set("nodeIds", typedArray("Int32Array", table.nodeIds.data(), table.nodeIds.size()));
    object.set("layers", typedArray("Int32Array", table.layers.data(), table.layers.size()));
    if (table.hasPositions)
        object.set("positions", typedArray("Float64Array", table.positions.data(), table.positions.size()));
    object.set("values", typedArray("Float64Array", table.values.data(), table.values.size()));
    return object;
}

// A grid point force table: { subcase, analysis, names, nodeIds, elementIds, nameIndices, values } (rows × 6).
val gridPointForceObject(const output::GridPointForceTable& table)
{
    val object = val::object();
    object.set("subcase", table.subcase);
    object.set("analysis", table.analysis);
    object.set("names", textArray(table.names));
    object.set("nodeIds", typedArray("Int32Array", table.nodeIds.data(), table.nodeIds.size()));
    object.set("elementIds", typedArray("Int32Array", table.elementIds.data(), table.elementIds.size()));
    object.set("nameIndices", typedArray("Int32Array", table.nameIndices.data(), table.nameIndices.size()));
    object.set("values", typedArray("Float64Array", table.values.data(), table.values.size()));
    return object;
}

val eigenvalueTableObject(const output::EigenvalueTable& table)
{
    val rows = val::array();
    for (const output::EigenvalueRow& row : table.rows)
    {
        val entry = val::object();
        entry.set("mode", row.mode);
        entry.set("extractionOrder", row.extractionOrder);
        entry.set("eigenvalue", row.eigenvalue);
        entry.set("radians", row.radians);
        entry.set("cycles", row.cycles);
        entry.set("generalizedMass", row.generalizedMass);
        entry.set("generalizedStiffness", row.generalizedStiffness);
        rows.call<void>("push", entry);
    }
    val object = val::object();
    object.set("subcase", table.subcase);
    object.set("analysis", table.analysis);
    object.set("rows", rows);
    return object;
}

// settings.condensation: "all", "recommended", or { units: "all" | "recommended" | number[], fullUnits?: number[],
// retainedNodes?: number[], modeCount?: number, memoryMode?: "auto" | "speed" | "low_memory",
// cache?: CondensationCache, cacheModelTag?: string } (an array of units: those condensed).
api::CondensationSettings condensationSettingsOf(const val& given)
{
    api::CondensationSettings settings;
    if (given.isUndefined() || given.isNull())
        return settings;
    const auto unitsNamed = [&](const val& units)
    {
        if (units.isArray())
        {
            settings.units = api::CondensedUnits::Listed;
            const std::vector<double> ids = emscripten::convertJSArrayToNumberVector<double>(units);
            for (const double id : ids)
                settings.condensedUnits.push_back(static_cast<std::int64_t>(id));
            return;
        }
        const std::string name = units.isString() ? units.as<std::string>() : std::string();
        if (name == "all")
            settings.units = api::CondensedUnits::All;
        else if (name == "recommended")
            settings.units = api::CondensedUnits::Recommended;
        else if (name == "none")
            settings.units = api::CondensedUnits::None;
        else
            throw std::invalid_argument(
                "condensation: expected \"all\", \"recommended\", \"none\" or an array of unit IDs");
    };
    if (given.isString())
    {
        unitsNamed(given);
        return settings;
    }
    unitsNamed(given["units"]);
    if (const val full = given["fullUnits"]; !full.isUndefined() && !full.isNull())
        for (const double id : emscripten::convertJSArrayToNumberVector<double>(full))
            settings.fullUnits.push_back(static_cast<std::int64_t>(id));
    if (const val retained = given["retainedNodes"]; !retained.isUndefined() && !retained.isNull())
        settings.retainedNodes = emscripten::convertJSArrayToNumberVector<std::int32_t>(retained);
    if (const val modes = given["modeCount"]; !modes.isUndefined() && !modes.isNull())
    {
        const double count = modes.as<double>();
        if (!std::isfinite(count) || count < 0 || count > 1e6 || std::floor(count) != count)
            throw std::invalid_argument("condensation.modeCount must be an integer from 0");
        settings.modeCount = static_cast<std::int32_t>(count);
    }
    if (const val mode = given["memoryMode"]; !mode.isUndefined() && !mode.isNull())
        settings.memoryMode = reduction::memoryModeOf(mode.as<std::string>());
    if (const val cache = given["cache"]; !cache.isUndefined() && !cache.isNull())
        settings.cache = cache.as<std::shared_ptr<reduction::CondensationCache>>();
    if (const val tag = given["cacheModelTag"]; !tag.isUndefined() && !tag.isNull())
        settings.cacheModelTag = tag.as<std::string>();
    return settings;
}

// condensationCache(memoryBytes): a condensation cache in memory (0: its default size).
std::shared_ptr<reduction::CondensationCache> condensationCache(double memoryBytes)
{
    return guarded(
        [&]
        {
            if (!std::isfinite(memoryBytes) || memoryBytes < 0)
                throw std::invalid_argument("condensationCache: memoryBytes must be a number of bytes from 0");
            return api::makeCondensationCache(static_cast<std::uint64_t>(memoryBytes));
        });
}

// cache.statistics(): { memoryHits, diskHits, misses, stores, refusedStores, evictions, missCauses: { cause: count } }.
val cacheStatistics(const reduction::CondensationCache& cache)
{
    const reduction::CacheStatistics statistics = cache.statistics();
    val result = val::object();
    result.set("memoryHits", static_cast<double>(statistics.memoryHits));
    result.set("diskHits", static_cast<double>(statistics.diskHits));
    result.set("misses", static_cast<double>(statistics.misses));
    result.set("stores", static_cast<double>(statistics.stores));
    result.set("refusedStores", static_cast<double>(statistics.refusedStores));
    result.set("evictions", static_cast<double>(statistics.evictions));
    val causes = val::object();
    for (const auto& [cause, count] : statistics.missCauses)
        causes.set(cause, static_cast<double>(count));
    result.set("missCauses", causes);
    return result;
}

// settings.externalUnits: [{ zone: { id, name?, elements: number[], boundary: [{ grid, components? }] },
// labels: [[grid, component], ...], stiffness: number[] (labels × labels, column by column), mass?: number[] }].
std::vector<api::ExternalUnitSettings> externalUnitsOf(const val& given)
{
    std::vector<api::ExternalUnitSettings> units;
    if (given.isUndefined() || given.isNull())
        return units;
    const auto numbers = [](const val& array)
    {
        return array.isUndefined() || array.isNull() ? std::vector<double>()
                                                     : emscripten::convertJSArrayToNumberVector<double>(array);
    };
    const auto length = given["length"].as<unsigned>();
    for (unsigned u = 0; u < length; ++u)
    {
        const val entry = given[u];
        const val zone = entry["zone"];
        api::ExternalUnitSettings unit;
        unit.zone.id = zone["id"].as<std::int32_t>();
        if (const val name = zone["name"]; !name.isUndefined() && !name.isNull())
            unit.zone.name = name.as<std::string>();
        for (const double id : numbers(zone["elements"]))
            unit.zone.elements.push_back(static_cast<std::int32_t>(id));
        if (const val boundary = zone["boundary"]; !boundary.isUndefined() && !boundary.isNull())
            for (unsigned b = 0; b < boundary["length"].as<unsigned>(); ++b)
            {
                const val point = boundary[b];
                output::ZoneBoundaryPoint entry;
                entry.grid = point["grid"].as<std::int32_t>();
                const val components = point["components"];
                entry.components = output::componentMaskOf(
                    components.isUndefined() || components.isNull() ? "123456" : components.as<std::string>());
                unit.zone.boundary.push_back(std::move(entry));
            }
        const val labels = entry["labels"];
        for (unsigned k = 0; k < labels["length"].as<unsigned>(); ++k)
        {
            const std::vector<double> pair = numbers(labels[k]);
            if (pair.size() != 2)
                throw std::invalid_argument("externalUnits: each label is [grid, component]");
            unit.labels.push_back({reduction::UnitPointKind::Grid, static_cast<base::EntityId>(pair[0]),
                                   reduction::FastenerSide::None, static_cast<std::uint8_t>(pair[1])});
        }
        unit.stiffness = numbers(entry["stiffness"]);
        unit.mass = numbers(entry["mass"]);
        units.push_back(std::move(unit));
    }
    return units;
}

template <typename Integer>
val numberArray(const std::vector<Integer>& values)
{
    val array = val::array();
    for (const std::int64_t value : values)
        array.call<void>("push", static_cast<double>(value));
    return array;
}

// settings.envelopes: [{ kind, quantity, mode }] (output::EnvelopeSpec).
std::vector<output::EnvelopeSpec> envelopesOf(const val& given)
{
    std::vector<output::EnvelopeSpec> envelopes;
    if (given.isUndefined() || given.isNull())
        return envelopes;
    for (unsigned i = 0; i < given["length"].as<unsigned>(); ++i)
    {
        const val entry = given[i];
        envelopes.push_back({entry["kind"].as<std::string>(), entry["quantity"].as<std::string>(),
                             entry["mode"].as<std::string>()});
    }
    return envelopes;
}

// An envelope table: { kind, elementType, quantity, mode, ids, nodeIds, layers, positions, values, cases }.
val envelopeTableObject(const output::EnvelopeTable& table)
{
    val object = val::object();
    object.set("kind", table.kind);
    object.set("elementType", table.elementType);
    object.set("quantity", table.quantity);
    object.set("mode", table.mode);
    object.set("ids", typedArray("Int32Array", table.ids.data(), table.ids.size()));
    object.set("nodeIds", typedArray("Int32Array", table.nodeIds.data(), table.nodeIds.size()));
    object.set("layers", typedArray("Int32Array", table.layers.data(), table.layers.size()));
    object.set("positions", typedArray("Float64Array", table.positions.data(), table.positions.size()));
    object.set("values", typedArray("Float64Array", table.values.data(), table.values.size()));
    object.set("cases", typedArray("Int32Array", table.cases.data(), table.cases.size()));
    return object;
}

val analysisObject(const api::AnalysisOutput& output)
{
    val nodal = val::array();
    for (const output::NodalResultTable& table : output.tables.nodalTables)
        nodal.call<void>("push", nodalTableObject(table));
    val elements = val::array();
    for (const output::ElementResultTable& table : output.tables.elementTables)
        elements.call<void>("push", elementTableObject(table));
    val forces = val::array();
    for (const output::GridPointForceTable& table : output.tables.gridPointForceTables)
        forces.call<void>("push", gridPointForceObject(table));
    val eigenvalues = val::array();
    for (const output::EigenvalueTable& table : output.eigenvalues)
        eigenvalues.call<void>("push", eigenvalueTableObject(table));
    val envelopes = val::array();
    for (const output::EnvelopeTable& table : output.envelopes)
        envelopes.call<void>("push", envelopeTableObject(table));

    val result = val::object();
    if (output.selectedQuarticShells)
        result.set("selectedQuarticShells", numberArray(*output.selectedQuarticShells));
    result.set("nodalTables", nodal);
    result.set("elementTables", elements);
    result.set("gridPointForces", forces);
    result.set("eigenvalues", eigenvalues);
    result.set("envelopes", envelopes);
    result.set("aeroelastic", output.aeroelastic ? wasm::trimResultsObject(*output.aeroelastic) : val::null());
    result.set("diagnostics", diagnosticArray(output.diagnostics));
    if (output.condensation)
    {
        val condensation = val::object();
        condensation.set("condensedUnits", numberArray(output.condensation->condensedUnits));
        condensation.set("fullUnits", numberArray(output.condensation->fullUnits));
        condensation.set("interfaceDofs", static_cast<double>(output.condensation->interfaceDofs));
        condensation.set("note", output.condensation->note);
        condensation.set("unitByUnit", output.condensation->isUnitByUnit);
        result.set("condensation", condensation);
    }
    else
    {
        result.set("condensation", val::null());
    }
    return result;
}

val coupledObject(const api::AeroCoupledResult& run)
{
    auto result = analysisObject(run.analysis);
    auto motions = val::array();
    for (const auto& motion : run.boxMotions)
    {
        auto field = val::object();
        field.set("plunge", typedArray("Float64Array", motion.plunge.data(), motion.plunge.size()));
        field.set("slope", typedArray("Float64Array", motion.slope.data(), motion.slope.size()));
        motions.call<void>("push", field);
    }
    result.set("boxMotions", motions);
    if (run.preload) result.set("preload", val(run.preload));
    if (run.modalBasis) result.set("modalBasis", val(run.modalBasis));
    return result;
}

// runAnalysis(model, analysis, settings): settings is { subcases?: number[], memoryBytes?: number, threads?: number,
// condensation?: see condensationSettingsOf, streamed?: boolean, externalUnits?: see externalUnitsOf, envelopes?: see
// envelopesOf, keepsCaseTables?: boolean }.
val runAnalysis(Model& model, const std::string& analysis, const val& settings)
{
    return guarded(
        [&]
        {
            api::AnalysisSettings chosen;
            if (!settings.isNull() && !settings.isUndefined())
            {
                if (const val subcases = settings["subcases"]; !subcases.isUndefined() && !subcases.isNull())
                    chosen.subcases = emscripten::convertJSArrayToNumberVector<std::int32_t>(subcases);
                if (const val bytes = settings["memoryBytes"]; !bytes.isUndefined() && !bytes.isNull())
                    chosen.memoryBytes = static_cast<std::uint64_t>(bytes.as<double>());
                chosen.condensation = condensationSettingsOf(settings["condensation"]);
                if (const val streamed = settings["streamed"]; !streamed.isUndefined() && !streamed.isNull())
                    chosen.isStreamed = streamed.as<bool>();
                chosen.externalUnits = externalUnitsOf(settings["externalUnits"]);
                chosen.envelopes = envelopesOf(settings["envelopes"]);
                if (const val keeps = settings["keepsCaseTables"]; !keeps.isUndefined() && !keeps.isNull())
                    chosen.keepsCaseTables = keeps.as<bool>();
            }
            if (const val threads =
                    settings.isUndefined() || settings.isNull() ? val::undefined() : settings["threads"];
                !threads.isUndefined() && !threads.isNull())
            {
                const double count = threads.as<double>();
                if (!std::isfinite(count) || count < 1 || count > 64 || std::floor(count) != count)
                    throw std::invalid_argument("threads must be an integer from 1 to 64");
                chosen.threads = static_cast<std::size_t>(count);
            }
            const api::AnalysisOutput output = api::runAnalysis(*model.snapshot, analysisNamed(analysis), chosen);

            return analysisObject(output);
        });
}

// runSuperelements(residual, parts, analysis, settings): parts is [{ id, model, boundary: [[partPoint, residualPoint],
// ...] }], settings as runAnalysis' (subcases, memoryBytes, threads). Gives { superelements: [{ id, nodalTables,
// elementTables, eigenvalues }], diagnostics }, the residual structure (0) first.
val runSuperelements(Model& residual, const val& parts, const std::string& analysis, const val& settings)
{
    return guarded(
        [&]
        {
            api::AnalysisSettings chosen;
            if (!settings.isNull() && !settings.isUndefined())
            {
                if (const val subcases = settings["subcases"]; !subcases.isUndefined() && !subcases.isNull())
                    chosen.subcases = emscripten::convertJSArrayToNumberVector<std::int32_t>(subcases);
                if (const val bytes = settings["memoryBytes"]; !bytes.isUndefined() && !bytes.isNull())
                    chosen.memoryBytes = static_cast<std::uint64_t>(bytes.as<double>());
                if (const val threads = settings["threads"]; !threads.isUndefined() && !threads.isNull())
                    chosen.threads = static_cast<std::size_t>(threads.as<double>());
            }
            std::vector<api::SuperelementPartSettings> given;
            const auto count = parts["length"].as<unsigned>();
            for (unsigned p = 0; p < count; ++p)
            {
                const val part = parts[p];
                api::SuperelementPartSettings entry;
                entry.superelementId = part["id"].as<std::int32_t>();
                entry.snapshot = part["model"].as<Model&>().snapshot;
                const val boundary = part["boundary"];
                for (unsigned b = 0; b < boundary["length"].as<unsigned>(); ++b)
                {
                    const std::vector<double> pair = emscripten::convertJSArrayToNumberVector<double>(boundary[b]);
                    if (pair.size() != 2)
                        throw std::invalid_argument(
                            "runSuperelements: each boundary entry is [partPoint, residualPoint]");
                    entry.boundary.emplace_back(static_cast<base::EntityId>(pair[0]),
                                                static_cast<base::EntityId>(pair[1]));
                }
                given.push_back(std::move(entry));
            }
            const api::SuperelementAnalysisOutput output =
                api::runSuperelements(*residual.snapshot, std::move(given), analysisNamed(analysis), chosen);
            val superelements = val::array();
            for (const api::SuperelementOutput& entry : output.superelements)
            {
                val nodal = val::array();
                for (const output::NodalResultTable& table : entry.tables.nodalTables)
                    nodal.call<void>("push", nodalTableObject(table));
                val elements = val::array();
                for (const output::ElementResultTable& table : entry.tables.elementTables)
                    elements.call<void>("push", elementTableObject(table));
                val eigenvalues = val::array();
                for (const output::EigenvalueTable& table : entry.eigenvalues)
                    eigenvalues.call<void>("push", eigenvalueTableObject(table));
                val object = val::object();
                object.set("id", entry.superelementId);
                object.set("nodalTables", nodal);
                object.set("elementTables", elements);
                object.set("eigenvalues", eigenvalues);
                superelements.call<void>("push", object);
            }
            val result = val::object();
            result.set("superelements", superelements);
            result.set("diagnostics", diagnosticArray(output.diagnostics));
            return result;
        });
}

val threadingInfo()
{
    return guarded(
        []
        {
            const auto info = api::threadingInfo();
            auto result = val::object();
            result.set("enabled", info.enabled);
            result.set("maximumThreads", double(info.maximumThreads));
            result.set("activeWorkers", double(info.activeWorkers));
            return result;
        });
}

std::string version()
{
    return api::programName();
}

} // namespace

namespace oos::wasm
{
val optimizationResults(api::OptimizationEvaluation& point)
{
    return guarded(
        [&]
        {
            const auto& analysis = point.evaluation().analysis;
            val result = val::object();
            val nodal = val::array();
            val elements = val::array();
            val forces = val::array();
            const auto appendArrays = [&](const output::ResultArrays& tables)
            {
                for (const auto& table : tables.nodalTables)
                    nodal.call<void>("push", nodalTableObject(table));
                for (const auto& table : tables.elementTables)
                    elements.call<void>("push", elementTableObject(table));
                for (const auto& table : tables.gridPointForceTables)
                    forces.call<void>("push", gridPointForceObject(table));
            };
            appendArrays(analysis.results.arrays);
            val eigenvalues = val::array();
            auto diagnostics = analysis.results.diagnostics;
            if (analysis.modes)
            {
                appendArrays(analysis.modes->arrays);
                for (const auto& subcase : analysis.modes->subcases)
                    eigenvalues.call<void>("push", eigenvalueTableObject(subcase.eigenvalues));
            }
            if (analysis.buckling)
            {
                for (const auto& solve : analysis.buckling->solves)
                    eigenvalues.call<void>("push", eigenvalueTableObject(solve.eigenvalues));
            }
            diagnostics.insert(diagnostics.end(), point.evaluation().trackingWarnings.begin(),
                               point.evaluation().trackingWarnings.end());
            result.set("nodalTables", nodal);
            result.set("elementTables", elements);
            result.set("gridPointForces", forces);
            result.set("eigenvalues", eigenvalues);
            result.set("aeroelastic", analysis.trim ? trimResultsObject(*analysis.trim) : val::null());
            result.set("diagnostics", diagnosticArray(diagnostics));
            return result;
        });
}
} // namespace oos::wasm

EMSCRIPTEN_BINDINGS(oos_solver)
{
    using namespace emscripten;

    function("version", &version);
    function("optimizationCatalogue", &optimizationCatalogue);
    function("modelCapabilities", &modelCapabilities);
    function("threadingInfo", &threadingInfo);
    function("elementFormulations", &elementFormulations);

    class_<api::ModelInput>("ModelInput")
        .constructor<>()
        .function("setVersions", &setVersions)
        .function("setSourceCards", &setSourceCards)
        .function("setTable", &setTable)
        .function("setTable", optional_override([](api::ModelInput& input, const std::string& name, const val& bytes)
                                                { setTable(input, name, bytes, val::null()); }))
        .function("addElementTable", &addElementTable)
        .function("addPropertyTable", &addPropertyTable)
        .function("addMethodTable", &addMethodTable)
        .function("addMaterialTable", &addMaterialTable)
        .function("addLoadCardTable", &addLoadCardTable)
        .function("addAeroCardTable", &addAeroCardTable)
        .function("setDisplacementRotations", &setDisplacementRotations)
        .function("setOffsetDefinition", &setOffsetDefinition)
        .function("setElementFormulation", &setElementFormulation)
        .function("setModelFormulation", &setModelFormulation)
        .function("setFormulationGeometry", &setFormulationGeometry)
        .function("build", &buildModel);

    class_<api::AeroStaticPreload>("AeroStaticPreload")
        .smart_ptr<std::shared_ptr<api::AeroStaticPreload>>("AeroStaticPreloadPointer");
    class_<api::AeroModalBasis>("AeroModalBasis")
        .smart_ptr<std::shared_ptr<api::AeroModalBasis>>("AeroModalBasisPointer")
        .function("generalizedLoads", optional_override([](const api::AeroModalBasis& basis, const val& forces) {
            return guarded([&] {
                const auto q = basis.generalizedLoads(numbersOf(forces));
                return typedArray("Float64Array", q.data(), q.size());
            });
        }));
    class_<api::AeroCoupling>("AeroTransfer")
        .smart_ptr<std::shared_ptr<api::AeroCoupling>>("AeroTransferPointer")
        .property("nodeIds", optional_override([](const api::AeroCoupling& transfer) {
            const auto ids = transfer.nodeIds();
            return typedArray("Int32Array", ids.data(), ids.size());
        }))
        .property("boxIds", optional_override([](const api::AeroCoupling& transfer) {
            const auto ids = transfer.boxIds();
            return typedArray("Int32Array", ids.data(), ids.size());
        }))
        .function("solveStatic", optional_override([](const api::AeroCoupling& t, int subcase, const val& forces) {
            return guarded([&] { return coupledObject(t.solveStatic(subcase, numbersOf(forces))); });
        }))
        .function("solveModes", optional_override([](const api::AeroCoupling& t, int subcase) {
            return guarded([&] { return coupledObject(t.solveModes(subcase)); });
        }))
        .function("solveBuckling", optional_override([](const api::AeroCoupling& t, int subcase, const api::AeroStaticPreload& preload) {
            return guarded([&] { return coupledObject(t.solveBuckling(subcase, preload)); });
        }))
        .function("boxMotion", &boxMotion)
        .function("structuralLoads", &structuralLoads);

    class_<Model>("Model")
        .smart_ptr<std::shared_ptr<Model>>("ModelPointer")
        .function("aeroTransfer", &aeroTransfer)
        .function("tableNames", &modelTableNames)
        .function("checkSupport", &checkSupport)
        .function("surfaceEdges", &surfaceEdges);

    function("runAnalysis", &runAnalysis);
    function("runSuperelements", &runSuperelements);

    class_<reduction::CondensationCache>("CondensationCache")
        .smart_ptr<std::shared_ptr<reduction::CondensationCache>>("CondensationCachePointer")
        .function("statistics", &cacheStatistics);
    function("condensationCache", &condensationCache);
}
