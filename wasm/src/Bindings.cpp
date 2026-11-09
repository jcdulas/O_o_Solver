// The JavaScript interface of the solver (embind). It only converts between JavaScript values and the plain C++ of
// oos::api; every decision is taken there. JavaScript sees:
//
//   new ModelInput()                 .setVersions .setTable .addElementTable ... .setElementFormulation
//                                    .build() -> Model
//   elementFormulations()            -> [{ card, formulations }], the card its neutral type ("ShellQuad4"), the default
//                                       formulation of each card first
//   Model                            .tableNames() .checkSupport(analysis)
//   runAnalysis(model, analysis, settings) -> { nodalTables, elementTables, gridPointForces, eigenvalues, diagnostics }
//
// The names are those of oos::api, the neutral names of O_o_Reader (card types, table kinds, columns), passed through
// untranslated. Errors of the solver become JavaScript errors whose message is the solver's (what was expected, what
// was found, and what to do).

#include "Model.hpp"
#include "oos/api/Analyses.hpp"
#include "oos/api/ModelInput.hpp"
#include "oos/api/Optimization.hpp"

#include <emscripten/bind.h>
#include <emscripten/val.h>

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

void setVersions(api::ModelInput& input, double structure, double content)
{
    guarded([&] { input.setVersions(static_cast<std::int64_t>(structure), static_cast<std::int64_t>(content)); });
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
    throw std::invalid_argument("unknown analysis '" + name + "': expected \"statics\", \"modes\" or \"buckling\"");
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
            const api::SupportCheck check = api::checkSupport(*model.snapshot, analysisNamed(analysis));
            val result = val::object();
            result.set("supported", textArray(check.supported));
            result.set("refusals", diagnosticArray(check.refusals));
            result.set("ignored", diagnosticArray(check.ignored));
            result.set("isComplete", check.refusals.empty());
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

// runAnalysis(model, analysis, settings): settings is { subcases?: number[], memoryBytes?: number }.
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
            }
            const api::AnalysisOutput output = api::runAnalysis(*model.snapshot, analysisNamed(analysis), chosen);

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

            val result = val::object();
            result.set("nodalTables", nodal);
            result.set("elementTables", elements);
            result.set("gridPointForces", forces);
            result.set("eigenvalues", eigenvalues);
            result.set("diagnostics", diagnosticArray(output.diagnostics));
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
                diagnostics.insert(diagnostics.end(), analysis.modes->diagnostics.begin(),
                                   analysis.modes->diagnostics.end());
            }
            if (analysis.buckling)
            {
                for (const auto& solve : analysis.buckling->solves)
                    eigenvalues.call<void>("push", eigenvalueTableObject(solve.eigenvalues));
                diagnostics.insert(diagnostics.end(), analysis.buckling->diagnostics.begin(),
                                   analysis.buckling->diagnostics.end());
            }
            diagnostics.insert(diagnostics.end(), point.evaluation().trackingWarnings.begin(),
                               point.evaluation().trackingWarnings.end());
            result.set("nodalTables", nodal);
            result.set("elementTables", elements);
            result.set("gridPointForces", forces);
            result.set("eigenvalues", eigenvalues);
            result.set("diagnostics", diagnosticArray(diagnostics));
            return result;
        });
}
} // namespace oos::wasm

EMSCRIPTEN_BINDINGS(oos_solver)
{
    using namespace emscripten;

    function("version", &version);
    function("elementFormulations", &elementFormulations);

    class_<api::ModelInput>("ModelInput")
        .constructor<>()
        .function("setVersions", &setVersions)
        .function("setTable", &setTable)
        .function("addElementTable", &addElementTable)
        .function("addPropertyTable", &addPropertyTable)
        .function("addMethodTable", &addMethodTable)
        .function("addMaterialTable", &addMaterialTable)
        .function("addLoadCardTable", &addLoadCardTable)
        .function("addAeroCardTable", &addAeroCardTable)
        .function("setDisplacementRotations", &setDisplacementRotations)
        .function("setOffsetDefinition", &setOffsetDefinition)
        .function("setElementFormulation", &setElementFormulation)
        .function("build", &buildModel);

    class_<Model>("Model")
        .smart_ptr<std::shared_ptr<Model>>("ModelPointer")
        .function("tableNames", &modelTableNames)
        .function("checkSupport", &checkSupport);

    function("runAnalysis", &runAnalysis);
}
