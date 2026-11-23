#include "oos/api/Analyses.hpp"
#include "oos/api/Optimization.hpp"
#include "oos/input/snapshot/SnapshotBuilder.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mutex>

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

using Numbers = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::span<const double> vectorOf(const Numbers& values)
{
    if (values.ndim() != 1)
        throw py::value_error("Expected a one-dimensional vector");
    return {values.data(), static_cast<std::size_t>(values.size())};
}

// A single direction (or a single weighted sum) is one contiguous product block.
struct VectorSink final : design::ProductSink
{
    explicit VectorSink(std::span<double> destination) : output(destination) {}
    void consume(const design::ProductBlock& block) override
    {
        if (block.first != 0 || block.values.size() != output.size())
            throw std::logic_error("Unexpected shape of a native derivative product");
        std::copy(block.values.begin(), block.values.end(), output.begin());
    }
    std::span<double> output;
};

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
    py::dict out("nodal_tables"_a = nodal, "element_tables"_a = elements, "grid_point_forces"_a = forces,
                 "eigenvalues"_a = eigen, "diagnostics"_a = diagnostics);
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
    py::class_<Model>(m, "Model")
        .def(
            "run",
            [](const Model& model, const std::string& name, std::size_t threads,
               const std::vector<std::int32_t>& subcases)
            {
                auto kind = analysis(name);
                api::AnalysisSettings settings;
                settings.threads = threads;
                settings.subcases = subcases;
                auto run = [&]
                {
                    py::gil_scoped_release unlock;
                    return api::runAnalysis(*model.snapshot, kind, settings);
                }();
                return results(std::move(run));
            },
            py::arg("analysis") = "statics", py::kw_only(), py::arg("threads") = 1,
            py::arg("subcases") = std::vector<std::int32_t>{})
        .def("_optimization",
             [](const Model& model, const py::dict& tables, const std::string& name, std::size_t threads)
             {
                 api::OptimizationSettings settings;
                 settings.threads = threads;
                 if (name != "statics")
                     settings.analyses.statics.reset();
                 if (name == "modes")
                     settings.analyses.modes.emplace();
                 else if (name == "buckling")
                     settings.analyses.buckling.emplace();
                 else if (name == "aeroelastic")
                     settings.analyses.aeroelastic.emplace();
                 else if (name != "statics")
                     throw py::value_error("Unknown design analysis: " + name);
                 std::map<std::string, base::ImportedTable> imported;
                 for (auto item : tables)
                     imported.emplace(py::cast<std::string>(item.first),
                                      importTable(py::reinterpret_borrow<py::object>(item.second)));
                 return std::make_shared<api::OptimizationProblem>(model.snapshot, api::OptimizationInput{}, settings,
                                                                   std::move(imported));
             });
    py::class_<api::OptimizationProblem, std::shared_ptr<api::OptimizationProblem>>(m, "OptimizationProblem")
        .def_property_readonly("initial_values",
                               [](const api::OptimizationProblem& p) { return p.problem().initialValues(); })
        .def_property_readonly("variables",
                               [](const api::OptimizationProblem& p)
                               {
                                   py::list variables;
                                   const auto& model = p.problem().model();
                                   for (auto index : model.independentIndices())
                                   {
                                       const auto& v = model.variables().all().at(index);
                                       variables.append(py::dict("id"_a = v.id, "label"_a = v.label,
                                                                 "lower"_a = v.lower, "upper"_a = v.upper));
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
    py::class_<Point>(m, "DesignPoint")
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
                                        "bound"_a = c.bound, "value"_a = c.value, "normalised"_a = c.normalised));
                return out;
            })
        .def("jvp",
             [](Point& p, const Numbers& direction)
             {
                 auto v = vectorOf(direction);
                 std::size_t rows = 0;
                 for (const auto& r : p.evaluation->evaluation().responses)
                     rows += r.itemCount();
                 p.evaluation->requireOutput(rows, 1);
                 py::array_t<double> out(rows);
                 VectorSink sink({out.mutable_data(), rows});
                 {
                     py::gil_scoped_release unlock;
                     std::lock_guard guard(p.mutex);
                     (void)p.evaluation->jvp(v, 1, 1, sink);
                 }
                 return out;
             })
        .def("vjp",
             [](Point& p, const py::dict& weights)
             {
                 design::ProductWeights w;
                 w.rows = 1;
                 for (auto [key, value] : weights)
                 {
                     const auto array = py::cast<Numbers>(value);
                     const auto data = vectorOf(array);
                     w.weights[py::cast<std::int32_t>(key)].assign(data.begin(), data.end());
                 }
                 const auto columns = p.evaluation->x().size();
                 p.evaluation->requireOutput(1, columns);
                 py::array_t<double> out(columns);
                 VectorSink sink({out.mutable_data(), columns});
                 {
                     py::gil_scoped_release unlock;
                     std::lock_guard guard(p.mutex);
                     (void)p.evaluation->vjp(w, 1, sink);
                 }
                 return out;
             })
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
                 out["report"] = py::dict("analysis_perturbations"_a = storage->report.analysisPerturbations,
                                          "field_perturbations"_a = storage->report.fieldPerturbations);
                 out["objective"] = storage->objective ? py::object(view(*storage->objective, owner)) : py::none();
                 return out;
             });
}
