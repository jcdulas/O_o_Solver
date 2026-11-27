#include "oos/api/AeroCoupling.hpp"
#include "oos/api/Analyses.hpp"
#include "oos/api/Capabilities.hpp"
#include "oos/api/ConstraintJacobian.hpp"
#include "oos/api/Optimization.hpp"
#include "oos/input/snapshot/SnapshotBuilder.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <mutex>
#include <optional>

namespace py = pybind11;
using namespace oos;
using namespace pybind11::literals;

namespace
{
using Snapshot = std::shared_ptr<const input::ModelSnapshot>;
struct Model
{
    Snapshot snapshot;
};
struct Point
{
    std::shared_ptr<api::OptimizationEvaluation> evaluation;
    std::mutex mutex;
};

using Numbers = py::array_t<double, py::array::c_style | py::array::forcecast | py::detail::npy_api::NPY_ARRAY_ALIGNED_>;

std::span<const double> vectorOf(const Numbers& values)
{
    if (values.ndim() != 1)
        throw py::value_error("Expected a one-dimensional vector");
    return {values.data(), static_cast<std::size_t>(values.size())};
}

// JVP blocks contain adjacent columns, VJP blocks adjacent rows.
struct ArraySink final : design::ProductSink
{
    ArraySink(double* output, std::size_t columns, bool byColumn)
        : data(output), stride(columns), columnBlocks(byColumn)
    {
    }
    void consume(const design::ProductBlock& block) override
    {
        if (block.values.empty())
            return;
        if (!columnBlocks || block.columns == stride)
        {
            std::copy(block.values.begin(), block.values.end(), data + (columnBlocks ? 0 : block.first * stride));
            return;
        }
        for (std::size_t row = 0; row < block.rows; ++row)
            std::copy_n(block.values.data() + row * block.columns, block.columns, data + row * stride + block.first);
    }
    double* data;
    std::size_t stride;
    bool columnBlocks;
};

py::array_t<double> productOutput(const py::object& out, const std::vector<py::ssize_t>& shape)
{
    if (out.is_none())
        return py::array_t<double>(shape);
    if (!py::isinstance<py::array_t<double>>(out))
        throw py::value_error("out must be a float64 NumPy array");
    auto array = py::reinterpret_borrow<py::array_t<double>>(out);
    if (!array.writeable() || !(array.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0 ||
        array.ndim() != py::ssize_t(shape.size()) || !std::equal(shape.begin(), shape.end(), array.shape()))
        throw py::value_error("out must be writable, aligned, C-contiguous and have the product shape");
    return array;
}

py::array_t<double> jvp(Point& point, const Numbers& directions, const py::object& out)
{
    const auto& evaluation = *point.evaluation;
    if ((directions.ndim() != 1 && directions.ndim() != 2) || directions.shape(0) != py::ssize_t(evaluation.x().size()))
        throw py::value_error("directions must have shape (variables,) or (variables, directions)");
    const bool batch = directions.ndim() == 2;
    const py::ssize_t count = batch ? directions.shape(1) : 1;
    if (!count)
        throw py::value_error("at least one direction is required");
    py::ssize_t rows = 0;
    for (const auto& response : evaluation.evaluation().responses)
        rows += static_cast<py::ssize_t>(response.itemCount());
    evaluation.requireOutput(rows, count);
    std::vector<py::ssize_t> shape{rows};
    if (batch)
        shape.push_back(count);
    auto result = productOutput(out, shape);
    if (!out.is_none() && result.size() && directions.size())
    {
        const auto inputStart = reinterpret_cast<std::uintptr_t>(directions.data());
        const auto outputStart = reinterpret_cast<std::uintptr_t>(result.data());
        if (inputStart < outputStart + result.nbytes() && outputStart < inputStart + directions.nbytes())
            throw py::value_error("out must not overlap directions");
    }
    ArraySink sink(result.mutable_data(), count, true);
    const std::span<const double> values(directions.data(), static_cast<std::size_t>(directions.size()));
    {
        py::gil_scoped_release unlock;
        std::lock_guard guard(point.mutex);
        (void)point.evaluation->jvp(values, count, std::min<py::ssize_t>(32, count), sink);
    }
    return result;
}

py::array_t<double> vjp(Point& point, const py::dict& weights, const py::object& out)
{
    design::ProductWeights native;
    native.rows = 1;
    bool batch = false;
    for (auto [key, value] : weights)
    {
        const auto array = py::cast<Numbers>(value);
        if (array.ndim() != 1 && array.ndim() != 2)
            throw py::value_error("weights must have shape (items,) or (sums, items)");
        const bool matrix = array.ndim() == 2;
        const auto rows = matrix ? static_cast<std::size_t>(array.shape(0)) : 1;
        if (!rows || (!native.weights.empty() && (rows != native.rows || matrix != batch)))
            throw py::value_error("all response weights must have the same rank and positive number of sums");
        batch = matrix;
        native.rows = rows;
        native.weights[py::cast<std::int32_t>(key)].assign(array.data(), array.data() + array.size());
    }
    const auto columns = point.evaluation->x().size();
    point.evaluation->requireOutput(native.rows, columns);
    std::vector<py::ssize_t> shape;
    if (batch)
        shape.push_back(static_cast<py::ssize_t>(native.rows));
    shape.push_back(static_cast<py::ssize_t>(columns));
    auto result = productOutput(out, shape);
    ArraySink sink(result.mutable_data(), columns, false);
    {
        py::gil_scoped_release unlock;
        std::lock_guard guard(point.mutex);
        (void)point.evaluation->vjp(native, std::min<std::size_t>(32, native.rows), sink);
    }
    return result;
}

py::dict hessianVector(Point& point, const py::dict& weights, const Numbers& directions)
{
    const auto n = static_cast<py::ssize_t>(point.evaluation->x().size());
    if ((directions.ndim() != 1 && directions.ndim() != 2) || directions.shape(0) != n)
        throw py::value_error("directions must have shape (variables,) or (variables, directions)");
    const auto count = directions.ndim() == 1 ? 1 : directions.shape(1);
    if (count == 0)
        throw py::value_error("at least one direction is required");
    design::ProductWeights native;
    native.rows = 1;
    for (auto [key, entry] : weights)
    {
        const auto array = py::cast<Numbers>(entry);
        const auto row = vectorOf(array);
        native.weights[py::cast<std::int32_t>(key)].assign(row.begin(), row.end());
    }
    auto values = std::make_unique<std::vector<double>>();
    const std::span<const double> input(directions.data(), static_cast<std::size_t>(directions.size()));
    design::ProductReport report;
    {
        py::gil_scoped_release release;
        std::lock_guard lock(point.mutex);
        report = point.evaluation->hessianVector(native, input, count, *values);
    }
    const auto* data = values->data();
    py::capsule owner(values.get(), [](void* pointer) { delete static_cast<std::vector<double>*>(pointer); });
    values.release();
    const std::vector<py::ssize_t> shape(directions.shape(), directions.shape() + directions.ndim());
    return py::dict("values"_a = py::array_t<double>(shape, data, owner),
                    "report"_a = py::dict("uses_buckling_tangents"_a = report.usesBucklingTangents,
                                          "uses_trim_tangents"_a = report.usesTrimTangents,
                                          "uses_modal_tangents"_a = report.usesModalTangents,
                                          "perturbed_linearisations"_a = report.perturbedLinearisations,
                                          "reason"_a = report.hessianBlocksReason));
}

base::ImportedTable importTable(const py::object& table)
{
    auto pair = table.attr("__arrow_c_array__")().cast<py::tuple>();
    if (pair.size() != 2)
        throw py::value_error("Expected an Arrow schema and array");
    auto* schema = static_cast<ArrowSchema*>(PyCapsule_GetPointer(pair[0].ptr(), "arrow_schema"));
    if (!schema)
        throw py::error_already_set();
    auto* array = static_cast<ArrowArray*>(PyCapsule_GetPointer(pair[1].ptr(), "arrow_array"));
    if (!array)
        throw py::error_already_set();
    return base::ImportedTable::takeOver(schema, array);
}

api::Analysis analysis(const std::string& name)
{
    if (name == "statics")
        return api::Analysis::Statics;
    if (name == "modes")
        return api::Analysis::Modes;
    if (name == "buckling")
        return api::Analysis::Buckling;
    if (name == "aeroelastic")
        return api::Analysis::Aeroelastic;
    throw py::value_error("Expected statics, modes, buckling or aeroelastic");
}

template <class Buffer>
py::array view(const Buffer& data, const py::object& owner, std::vector<py::ssize_t> shape = {})
{
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(data.data())>>;
    if (shape.empty())
        shape.push_back(static_cast<py::ssize_t>(data.size()));
    py::array_t<T> result(shape, data.data(), owner);
    result.attr("setflags")(false);
    return result;
}

// Hand the computed vector to NumPy without copying its data across the coupling interface.
py::array_t<double> ownedArray(std::vector<double> values, std::vector<py::ssize_t> shape)
{
    auto storage = std::make_unique<std::vector<double>>(std::move(values));
    const double* data = storage->data();
    py::capsule owner(storage.get(), [](void* p) { delete static_cast<std::vector<double>*>(p); });
    storage.release();
    return py::array_t<double>(shape, data, owner);
}

py::dict results(api::AnalysisOutput&& run)
{
    auto owner = py::cast(std::make_shared<api::AnalysisOutput>(std::move(run)));
    const auto& r = owner.cast<const api::AnalysisOutput&>();
    py::list nodal, elements, forces, eigen, diagnostics;
    for (const auto& t : r.tables.nodalTables)
        nodal.append(py::dict("kind"_a = t.kind, "subcase"_a = t.subcase, "step"_a = t.step,
                              "eigenvalue"_a = t.eigenvalue, "ids"_a = view(t.ids, owner),
                              "point_types"_a = view(t.pointTypes, owner),
                              "values"_a = view(t.values, owner, {py::ssize_t(t.rowCount()), 6})));
    for (const auto& t : r.tables.elementTables)
        elements.append(
            py::dict("kind"_a = t.kind, "element_type"_a = t.elementType, "subcase"_a = t.subcase, "step"_a = t.step,
                     "eigenvalue"_a = t.eigenvalue, "components"_a = t.components, "ids"_a = view(t.ids, owner),
                     "node_ids"_a = view(t.nodeIds, owner), "layers"_a = view(t.layers, owner),
                     "positions"_a = view(t.positions, owner),
                     "values"_a = view(t.values, owner, {py::ssize_t(t.rowCount()), py::ssize_t(t.componentCount())})));
    for (const auto& t : r.tables.gridPointForceTables)
        forces.append(py::dict("subcase"_a = t.subcase, "names"_a = t.names, "node_ids"_a = view(t.nodeIds, owner),
                               "element_ids"_a = view(t.elementIds, owner),
                               "name_indices"_a = view(t.nameIndices, owner),
                               "values"_a = view(t.values, owner, {py::ssize_t(t.rowCount()), 6})));
    for (const auto& t : r.eigenvalues)
        eigen.append(py::dict("subcase"_a = t.subcase, "rows"_a = view(t.rows, owner)));
    for (const auto& d : r.diagnostics)
        diagnostics.append(py::dict("severity"_a = base::severityName(d.severity), "code"_a = d.code,
                                    "message"_a = d.message, "card"_a = d.card, "id"_a = d.id, "field"_a = d.field));
    py::dict stages;
    for (const auto& stage : r.report.stages)
        stages[py::str(stage.name)] = stage.seconds;
    const py::dict report("threads"_a = r.report.threads, "kernel"_a = r.report.kernel, "dofs"_a = r.report.dofs,
                          "solved_dofs"_a = r.report.solvedDofs, "factorisations"_a = r.report.factorisations,
                          "seconds"_a = r.report.seconds, "stages"_a = stages,
                          "peak_budget_bytes"_a = r.report.peakBudgetBytes);
    // Normal modes and buckling: "lanczos", "substructuring", "out_of_core" or "condensed"; "" otherwise.
    report["eigen_solver"] = r.report.eigenSolver == "outOfCore" ? std::string("out_of_core") : r.report.eigenSolver;
    py::dict out("nodal_tables"_a = nodal, "element_tables"_a = elements, "grid_point_forces"_a = forces,
                 "eigenvalues"_a = eigen, "diagnostics"_a = diagnostics, "report"_a = report);
    if (r.selectedQuarticShells)
        out["selected_quartic_shells"] = view(*r.selectedQuarticShells, owner);
    if (r.aeroelastic)
    {
        py::list subcases;
        for (const auto& s : r.aeroelastic->subcases)
        {
            py::dict variables;
            for (const auto& v : s.variables)
                variables[py::str(v.label)] = v.value;
            subcases.append(
                py::dict("subcase"_a = s.subcase, "mach"_a = s.mach, "dynamic_pressure"_a = s.dynamicPressure,
                         "variables"_a = variables, "pressures"_a = view(s.pressures, owner),
                         "box_forces"_a = view(s.boxForces, owner), "box_moments"_a = view(s.boxMoments, owner)));
        }
        out["aeroelastic"] = py::dict("box_ids"_a = view(r.aeroelastic->boxIds, owner), "subcases"_a = subcases);
    }
    return out;
}
py::dict coupledResults(api::AeroCoupledResult run)
{
    auto out = results(std::move(run.analysis));
    py::list motions;
    for (auto& motion : run.boxMotions)
    {
        const auto n = static_cast<py::ssize_t>(motion.plunge.size());
        motions.append(py::dict("plunge"_a = ownedArray(std::move(motion.plunge), {n}),
                                "slope"_a = ownedArray(std::move(motion.slope), {n})));
    }
    out["box_motions"] = motions;
    if (run.preload) out["preload"] = py::cast(run.preload);
    if (run.modalBasis) out["modal_basis"] = py::cast(run.modalBasis);
    return out;
}

} // namespace

PYBIND11_MODULE(_native, m)
{
    PYBIND11_NUMPY_DTYPE(output::EigenvalueRow, mode, extractionOrder, eigenvalue, radians, cycles, generalizedMass,
                         generalizedStiffness);
    py::class_<api::AnalysisOutput, std::shared_ptr<api::AnalysisOutput>>(m, "_ResultStorage");
    py::class_<design::DesignGradients, std::shared_ptr<design::DesignGradients>>(m, "_GradientStorage");
    py::class_<input::SnapshotBuilder>(m, "_Builder")
        .def(py::init<>())
        .def("formulation", &input::SnapshotBuilder::setModelFormulation)
        .def("offset", &input::SnapshotBuilder::setOffsetDefinitionValue)
        .def("versions", &input::SnapshotBuilder::setVersions)
        .def("cards",
             [](input::SnapshotBuilder& b, const std::map<std::string, std::int64_t>& counts)
             {
                 std::vector<input::SourceCard> cards;
                 for (const auto& [name, n] : counts)
                     cards.push_back({name, n});
                 b.setSourceCards(std::move(cards));
             })
        .def("rotations",
             [](input::SnapshotBuilder& b, const py::array_t<double, py::array::c_style>& values)
             {
                 if (values.ndim() != 3 || values.shape(1) != 3 || values.shape(2) != 3)
                     throw py::value_error("Expected displacement rotations with shape (nodes, 3, 3)");
                 // The Reader creates a fresh array for this snapshot. Retain its buffer across native solves.
                 auto owner = std::shared_ptr<const void>(values.inc_ref().ptr(),
                                                          [](const void* p)
                                                          {
                                                              py::gil_scoped_acquire lock;
                                                              Py_DECREF(static_cast<PyObject*>(const_cast<void*>(p)));
                                                          });
                 b.setCdRotations({{values.data(), static_cast<std::size_t>(values.size())}, std::move(owner)});
             })
        .def("table",
             [](input::SnapshotBuilder& b, const std::string& role, const py::object& t)
             {
                 auto id = input::findTableRole(role);
                 if (!id)
                     throw py::value_error("Unknown table role: " + role);
                 b.setTable(*id, importTable(t));
             })
        .def("card",
             [](input::SnapshotBuilder& b, const std::string& group, const std::string& name, const py::object& t)
             {
                 auto data = importTable(t);
                 if (group == "elements")
                     b.addElementBatch(name, std::move(data));
                 else if (group == "properties")
                     b.addPropertyTable(name, std::move(data));
                 else if (group == "methods")
                     b.addMethodTable(name, "id", std::move(data));
                 else if (group == "materials")
                     b.addMaterialCardTable(name, std::move(data));
                 else if (group == "loads")
                     b.addLoadCardTable(name, std::move(data));
                 else if (group == "aero")
                     b.addAeroCardTable(name, "id", std::move(data));
                 else
                     throw py::value_error("Unknown card category: " + group);
             })
        .def("build",
             [](input::SnapshotBuilder& b) { return Model{std::make_shared<input::ModelSnapshot>(b.build())}; });
    py::class_<api::AeroStaticPreload, std::shared_ptr<api::AeroStaticPreload>>(m, "AeroStaticPreload");
    py::class_<api::AeroModalBasis, std::shared_ptr<api::AeroModalBasis>>(m, "AeroModalBasis")
        .def("generalized_loads", [](const api::AeroModalBasis& modes, const Numbers& forces) {
            const auto f = vectorOf(forces);
            auto q = [&] { py::gil_scoped_release release; return modes.generalizedLoads(f); }();
            const auto n = static_cast<py::ssize_t>(q.size());
            return ownedArray(std::move(q), {n});
        }, py::arg("box_forces"));
    py::class_<api::AeroCoupling, std::shared_ptr<api::AeroCoupling>>(m, "AeroTransfer")
        .def("solve_static", [](const api::AeroCoupling& t, int subcase, const Numbers& forces) {
            const auto f = vectorOf(forces);
            return coupledResults([&] { py::gil_scoped_release release; return t.solveStatic(subcase, f); }());
        }, py::arg("subcase"), py::arg("box_forces"))
        .def("solve_modes", [](const api::AeroCoupling& t, int subcase) {
            return coupledResults([&] { py::gil_scoped_release release; return t.solveModes(subcase); }());
        }, py::arg("subcase"))
        .def("solve_buckling", [](const api::AeroCoupling& t, int subcase, const api::AeroStaticPreload& preload) {
            return coupledResults([&] { py::gil_scoped_release release; return t.solveBuckling(subcase, preload); }());
        }, py::arg("subcase"), py::arg("preload"))
        .def_property_readonly("node_ids", [](const api::AeroCoupling& t) {
            return py::array_t<std::int32_t>(t.nodeIds().size(), t.nodeIds().data());
        })
        .def_property_readonly("box_ids", [](const api::AeroCoupling& t) {
            return py::array_t<std::int32_t>(t.boxIds().size(), t.boxIds().data());
        })
        .def("box_motion", [](const api::AeroCoupling& t, const Numbers& u) {
            if (u.ndim() != 2 || u.shape(0) != static_cast<py::ssize_t>(t.nodeIds().size()) || u.shape(1) != 6)
                throw py::value_error("nodal motion must have shape (number of spline nodes, 6)");
            auto motion = [&] {
                py::gil_scoped_release release;
                return t.boxMotion({u.data(), static_cast<std::size_t>(u.size())});
            }();
            const auto n = static_cast<py::ssize_t>(t.boxIds().size());
            return py::dict("plunge"_a = ownedArray(std::move(motion.plunge), {n}),
                            "slope"_a = ownedArray(std::move(motion.slope), {n}));
        }, py::arg("nodal_motion"))
        .def("structural_loads", [](const api::AeroCoupling& t, const Numbers& forces) {
            const auto input = vectorOf(forces);
            auto loads = [&] {
                py::gil_scoped_release release;
                return t.structuralLoads(input);
            }();
            return ownedArray(std::move(loads), {static_cast<py::ssize_t>(t.nodeIds().size()), 6});
        }, py::arg("box_forces"));
    py::class_<Model>(m, "Model")
        .def("aero_transfer", [](const Model& model) {
            py::gil_scoped_release release;
            return std::make_shared<api::AeroCoupling>(model.snapshot);
        })
        .def(
            "run",
            [](const Model& model, const std::string& name, std::optional<std::size_t> threads,
               const std::vector<std::int32_t>& subcases, std::uint64_t memoryBytes, bool streamed,
               const std::string& eigenSolver, const std::optional<std::string>& scratchDir, double cutoffFactor)
            {
                auto kind = analysis(name);
                api::AnalysisSettings settings;
                settings.threads = threads.value_or(api::AUTOMATIC_THREADS);
                settings.subcases = subcases;
                // 0: the memory the machine has left (api::AnalysisSettings::memoryBytes).
                settings.memoryBytes = memoryBytes;
                // Statics only: the equations solved without a factor of the whole model (a factor too large).
                settings.isStreamed = streamed;
                // Normal modes and buckling: the eigenvalue solver (api::EigenSolverSettings).
                if (eigenSolver == "substructuring")
                    settings.eigenSolver.solver = api::EigenSolver::Substructuring;
                else if (eigenSolver == "out_of_core")
                    settings.eigenSolver.solver = api::EigenSolver::OutOfCore;
                else if (eigenSolver != "lanczos")
                    throw py::value_error("Unknown eigen_solver: " + eigenSolver +
                                          " (expected lanczos, substructuring or out_of_core)");
                settings.eigenSolver.scratchFolder = scratchDir.value_or(std::string());
                settings.eigenSolver.cutoffFactor = cutoffFactor;
                auto run = [&]
                {
                    py::gil_scoped_release unlock;
                    return api::runAnalysis(*model.snapshot, kind, settings);
                }();
                return results(std::move(run));
            },
            py::arg("analysis") = "statics", py::kw_only(), py::arg("threads") = py::none(),
            py::arg("subcases") = std::vector<std::int32_t>{}, py::arg("memory_bytes") = 0,
            py::arg("streamed") = false, py::arg("eigen_solver") = "lanczos", py::arg("scratch_dir") = py::none(),
            py::arg("cutoff_factor") = 0.0)
        .def("_optimization",
             [](const Model& model, const py::dict& tables, const py::object& analyses,
                std::optional<std::size_t> threads)
             {
                 api::OptimizationSettings settings;
                 settings.threads = threads.value_or(api::AUTOMATIC_THREADS);
                 settings.analyses.statics.reset();
                 auto select = [&](const std::string& name, std::vector<std::int32_t> subcases)
                 {
                     if (name == "statics") settings.analyses.statics = std::move(subcases);
                     else if (name == "modes") settings.analyses.modes = std::move(subcases);
                     else if (name == "buckling") settings.analyses.buckling = std::move(subcases);
                     else if (name == "aeroelastic") settings.analyses.aeroelastic = std::move(subcases);
                     else throw py::value_error("Unknown design analysis: " + name);
                 };
                 if (py::isinstance<py::str>(analyses))
                     select(analyses.cast<std::string>(), {});
                 else
                 {
                     const auto selected = analyses.cast<std::map<std::string, std::vector<std::int32_t>>>();
                     if (selected.empty()) throw py::value_error("Expected at least one design analysis");
                     for (const auto& [name, cases] : selected) select(name, cases);
                 }
                 std::map<std::string, base::ImportedTable> imported;
                 for (auto item : tables)
                     imported.emplace(py::cast<std::string>(item.first),
                                      importTable(py::reinterpret_borrow<py::object>(item.second)));
                 return std::make_shared<api::OptimizationProblem>(model.snapshot, api::OptimizationInput{}, settings,
                                                                   std::move(imported));
             });
    py::class_<api::OptimizationProblem, std::shared_ptr<api::OptimizationProblem>>(m, "OptimizationProblem")
        .def_property_readonly(
            "capabilities",
            [](const api::OptimizationProblem& p)
            {
                const auto catalog = api::optimizationCapabilities(&p.problem().snapshot());
                py::list analyses, fields;
                for (const auto& a : catalog.analyses)
                    analyses.append(py::dict("analysis"_a = a.analysis, "solution"_a = a.solution,
                                             "sizing_first_order"_a = a.sizingFirstOrder,
                                             "shape_first_order"_a = a.shapeFirstOrder,
                                             "second_order"_a = a.secondOrder, "restrictions"_a = a.restrictions));
                for (const auto& f : catalog.fields)
                    fields.append(py::dict("target_kind"_a = f.targetKind, "target_card"_a = f.targetCard,
                                           "field"_a = f.field, "type"_a = f.type, "requires_item"_a = f.requiresItem));
                return py::dict("version"_a = catalog.version, "analyses"_a = analyses,
                                "responses"_a = catalog.responses, "fields"_a = fields);
            })
        .def_property_readonly("objective_definition", [](const api::OptimizationProblem& p) {
            const auto& selected = p.problem().constraints().parts().selection;
            return py::dict("response_id"_a = selected.objective, "sense"_a = (selected.isMaximised ? "max" : "min"),
                            "subcase"_a = selected.objectiveSubcase.value_or(0));
        })
        .def_property_readonly("initial_values",
                               [](const api::OptimizationProblem& p) { return p.problem().initialValues(); })
        .def_property_readonly(
            "variables",
            [](const api::OptimizationProblem& p)
            {
                py::list variables;
                const auto& model = p.problem().model();
                for (auto index : model.independentIndices())
                {
                    const auto& v = model.variables().all().at(index);
                    variables.append(py::dict("id"_a = v.id, "label"_a = v.label, "lower"_a = v.lower,
                                              "upper"_a = v.upper, "initial_value"_a = v.initialValue,
                                              "scale"_a = api::variableReferenceScale(v)));
                }
                return variables;
            })
        .def("evaluate",
             [](api::OptimizationProblem& p, const Numbers& x)
             {
                 auto v = vectorOf(x);
                 std::vector<double> owned(v.begin(), v.end());
                 auto point = std::make_unique<Point>();
                 py::gil_scoped_release unlock;
                 point->evaluation = p.evaluate(std::move(owned));
                 return point;
             });
    py::class_<api::ConstraintJacobian, std::shared_ptr<api::ConstraintJacobian>>(m, "ConstraintJacobian")
        .def_property_readonly("row_offsets", [](py::object self)
                               { return view(self.cast<api::ConstraintJacobian&>().rowOffsets(), self); })
        .def_property_readonly("column_indices", [](py::object self)
                               { return view(self.cast<api::ConstraintJacobian&>().columnIndices(), self); })
        .def_property_readonly("columns", &api::ConstraintJacobian::variableCount)
        .def(
            "values",
            [](const api::ConstraintJacobian& layout, Point& point, const py::object& out)
            {
                auto array = productOutput(out, {static_cast<py::ssize_t>(layout.columnIndices().size())});
                auto data = std::span<double>(array.mutable_data(), layout.columnIndices().size());
                {
                    py::gil_scoped_release unlock;
                    std::lock_guard guard(point.mutex);
                    layout.values(*point.evaluation, data);
                }
                return array;
            },
            py::arg("point"), py::kw_only(), py::arg("out") = py::none())
        .def(
            "constraints",
            [](const api::ConstraintJacobian& layout, const Point& point, const py::object& out)
            {
                const auto size = layout.rowOffsets().size() - 1;
                auto array = productOutput(out, {static_cast<py::ssize_t>(size)});
                layout.constraints(*point.evaluation, {array.mutable_data(), size});
                return array;
            },
            py::arg("point"), py::kw_only(), py::arg("out") = py::none());
    py::class_<Point>(m, "DesignPoint")
        .def("optimizer_values",
             [](const Point& p)
             {
                 const auto values = p.evaluation->optimizerValues();
                 py::array_t<double> constraints(values.constraints.size());
                 std::copy(values.constraints.begin(), values.constraints.end(), constraints.mutable_data());
                 return py::dict("objective"_a = values.objective, "constraints"_a = constraints,
                                 "partial"_a = values.partial, "uncertain_mode"_a = values.uncertainMode);
             })
        .def_property_readonly("objective", [](const Point& p) { return p.evaluation->evaluation().objective; })
        .def_property_readonly("responses",
                               [](py::object self)
                               {
                                   py::dict out;
                                   for (const auto& r : self.cast<Point&>().evaluation->evaluation().responses)
                                       out[py::int_(r.id)] = view(r.values, self);
                                   return out;
                               })
        .def_property_readonly(
            "constraints",
            [](const Point& p)
            {
                py::list out;
                for (const auto& c : p.evaluation->evaluation().constraints)
                    out.append(py::dict("set_id"_a = c.setId, "response_id"_a = c.responseId, "item"_a = c.item,
                                        "side"_a = c.side == design::BoundSide::Lower ? "lower" : "upper",
                                        "bound"_a = c.bound, "value"_a = c.value, "normalised"_a = c.normalised,
                                        "derivative_scale"_a = api::constraintDerivativeScale(c),
                                        "subcase"_a = c.label.subcase, "entity"_a = c.label.entity,
                                        "component"_a = c.label.component, "layer"_a = c.label.layer, "mode"_a = c.label.mode));
                return out;
            })
        .def(
            "constraint_jacobian",
            [](const Point& point, const std::optional<std::vector<std::vector<std::size_t>>>& rowColumns)
            { return std::make_shared<api::ConstraintJacobian>(*point.evaluation, rowColumns); },
            py::arg("row_columns") = py::none())
        .def("jvp", &jvp, py::arg("direction"), py::kw_only(), py::arg("out") = py::none())
        .def("hessian_vector", &hessianVector, py::arg("weights"), py::arg("direction"))
        .def("vjp", &vjp, py::arg("weights"), py::kw_only(), py::arg("out") = py::none())
        .def(
            "nodal_coordinate_gradient",
            [](Point& point, std::int32_t response, const std::vector<std::int32_t>& nodes, std::size_t item)
            {
                const auto result = [&]
                {
                    py::gil_scoped_release unlock;
                    std::lock_guard guard(point.mutex);
                    return point.evaluation->nodalCoordinateGradient(response, nodes, item);
                }();
                const auto n = static_cast<py::ssize_t>(result.nodeIds.size());
                return py::dict("node_ids"_a = py::array_t<std::int32_t>(n, result.nodeIds.data()),
                                "values"_a = py::array_t<double>(std::vector<py::ssize_t>{n, 3}, result.values.data()),
                                "report"_a = py::dict("analysis_perturbations"_a = result.report.analysisPerturbations,
                                                      "shape_perturbations"_a = result.report.shapePerturbations,
                                                      "factorisations"_a = result.report.factorisations,
                                                      "solves"_a = result.report.solves));
            },
            py::arg("response_id"), py::arg("node_ids") = std::vector<std::int32_t>{}, py::arg("item") = 0)
        .def("gradients",
             [](Point& p)
             {
                 auto storage = [&]
                 {
                     py::gil_scoped_release unlock;
                     std::lock_guard guard(p.mutex);
                     return std::make_shared<design::DesignGradients>(p.evaluation->gradients());
                 }();
                 auto owner = py::cast(storage);
                 py::dict responses;
                 for (const auto& g : storage->responses)
                     responses[py::int_(g.id)] =
                         view(g.values, owner, {py::ssize_t(g.itemCount), py::ssize_t(storage->variableCount)});
                 py::list constraints;
                 for (const auto& row : storage->constraints)
                     constraints.append(view(row, owner));
                 py::dict out("responses"_a = responses, "constraints"_a = constraints);
                 out["report"] =
                     py::dict("factorisations"_a = storage->report.factorisations, "solves"_a = storage->report.solves,
                              "analysis_perturbations"_a = storage->report.analysisPerturbations,
                              "field_perturbations"_a = storage->report.fieldPerturbations,
                              "shape_perturbations"_a = storage->report.shapePerturbations);
                 out["objective"] = storage->objective ? py::object(view(*storage->objective, owner)) : py::none();
                 return out;
             });
}
