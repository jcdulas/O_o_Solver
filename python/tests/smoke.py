"""Exercise the installed Reader/Solver wheels, including native gradients and buffer lifetimes."""
import gc
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import oo_model
import oo_solver

with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    archive = Path(__file__).resolve().parents[2] / "docs/benchmarks/structural-comparison-inputs.zip"
    with zipfile.ZipFile(archive) as inputs:
        vm50 = directory / "vm50.oos"
        name = next(n for n in inputs.namelist() if n.endswith("frequency-8.oos"))
        vm50.write_bytes(inputs.read(name))
    deck = oo_model.read_model(vm50)
    for formulation, expected in [("ClassicFormulation", 28.758920687), ("MixedHybridFormulation", 28.758260894)]:
        model = oo_solver.model_from_reader(deck, formulation=formulation)
        run = model.run("modes")
        modes = run["eigenvalues"][0]["rows"]
        assert not modes.flags.writeable
        # Every core by default, one on request: the same bits, and a report of what each run cost.
        single = model.run("modes", threads=1)
        assert single["report"]["threads"] == 1 and run["report"]["threads"] >= 1
        assert list(run["report"]["stages"]) == ["setup", "assembly", "eigenvalues", "recovery"]
        assert run["report"]["factorisations"] >= 1 and run["report"]["peak_budget_bytes"] > 0
        # The transverse shear of the laminates: this model has no laminate property.
        assert run["report"]["laminate_shear"] == []
        # A model of bars has no shell to put in the other formulation: that note is not given.
        assert all(d["code"] != "mixed-formulation-available" for d in run["diagnostics"])
        np.testing.assert_array_equal(single["eigenvalues"][0]["rows"]["eigenvalue"], modes["eigenvalue"])
        assert run["report"]["eigen_solver"] == "lanczos"
        # The solvers of large models: the factor in scratch files gives the same roots, the substructuring
        # approximate ones.
        on_disk = model.run("modes", eigen_solver="out_of_core", scratch_dir=str(directory))
        assert on_disk["report"]["eigen_solver"] == "out_of_core"
        np.testing.assert_allclose(on_disk["eigenvalues"][0]["rows"]["eigenvalue"], modes["eigenvalue"], rtol=1e-8)
        reduced = model.run("modes", eigen_solver="substructuring")
        assert reduced["report"]["eigen_solver"] == "substructuring"
        np.testing.assert_allclose(reduced["eigenvalues"][0]["rows"]["eigenvalue"], modes["eigenvalue"], rtol=5e-3)
        del model, single, run, on_disk, reduced
        gc.collect()
        np.testing.assert_allclose(modes["cycles"][0], expected, rtol=2e-9)

    # A laminated strip in the classic formulation: the run shows its information through the warnings module (they
    # stay in the result), and its report gives the transverse shear of the laminate under both conventions.
    strip = directory / "laminate.oos"
    laminate_text = """[executive]
SolutionSequence;101
[case]
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
DisplacementOutput;All
[model]
Node;1;;0.;0.;0.
Node;2;;200.;0.;0.
Node;3;;200.;50.;0.
Node;4;;0.;50.;0.
ShellQuad4;1;1;1;2;3;4
CompProp;1;;;;;;;;10;0.25;0.;Yes;10;0.25;90.;Yes;10;0.25;90.;Yes;10;0.25;0.;Yes
OrthoMat2D;10;140000.;10000.;0.3;5000.;5000.;3500.;1.6e-9
SinglePointConstrList;1;123456;1;4
PointForceLoad;2;2;;0.5;0.;0.;1.
PointForceLoad;2;3;;0.5;0.;0.;1.
"""
    strip.write_text(laminate_text, encoding="utf-8")
    model = oo_solver.model_from_reader(oo_model.read_model(strip))
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        run = model.run("statics")
    told = [d for d in run["diagnostics"] if d["severity"] in ("warning", "information")]
    assert {"mixed-formulation-available", "laminate-shear-factor"} <= {d["code"] for d in told}
    assert len(shown) == len(told)
    for entry, diagnostic in zip(shown, told):
        assert entry.filename == __file__
        assert str(entry.message) == f"[{diagnostic['code']}] {diagnostic['message']}"
        expected = oo_solver.SolverInformation if diagnostic["severity"] == "information" else oo_solver.SolverWarning
        assert entry.category is expected
    (row,) = run["report"]["laminate_shear"]
    assert (row["card"], row["prop_id"]) == ("CompProp", 1)
    np.testing.assert_allclose(row["factor"], (0.778051, 0.660405), rtol=1e-5)
    # Silenced by category, as the documentation says.
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", oo_solver.SolverInformation)
        model.run("statics")
    assert not any(issubclass(w.category, oo_solver.SolverInformation) for w in shown)
    strip.write_text(laminate_text + "Parameter;LaminateTransverseShear;FiveSixths\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", oo_solver.SolverWarning)
        five_sixths = oo_solver.model_from_reader(oo_model.read_model(strip)).run("statics")
    np.testing.assert_allclose(five_sixths["report"]["laminate_shear"][0]["factor"], (5 / 6, 5 / 6), rtol=1e-12)
    assert all(d["code"] != "laminate-shear-factor" for d in five_sixths["diagnostics"])
    del model, run, five_sixths

    bar = directory / "axial.oos"
    bar.write_text("""[executive]
SolutionSequence;101
[case]
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
DisplacementOutput;All
[model]
Node;1;;0.;0.;0.
Node;2;;3.;0.;0.
Rod2;1;1;1;2
RodProp;1;1;2.
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1
SinglePointConstrList;1;23456;2
PointForceLoad;2;2;;10.;1.;0.;0.
DesignVar;1;AREA;2.;0.5;5.
DesignPropRelation1;1;RodProp;1;area;;;;1;1.
DesignResp1;10;TIP;Displacement;;;1;;2
""", encoding="utf-8")
    deck = oo_model.read_model(bar)
    problem = oo_solver.optimization_from_reader(deck)
    model = oo_solver.model_from_reader(deck)
    del deck
    x = problem.initial_values
    point = problem.evaluate(x)
    derivative = point.gradients()["responses"][10]
    assert problem.variables[0]["id"] == 1
    np.testing.assert_allclose(point.jvp([3.]), derivative[:, 0] * 3)
    np.testing.assert_allclose(point.vjp({10: [4.]}), derivative[0] * 4)
    # More directions than the native block size. Rank-two input keeps its rank even for one direction.
    directions = np.arange(41., dtype=np.float64).reshape(1, -1)
    out = np.empty_like(directions)
    assert point.jvp(directions, out=out) is out
    np.testing.assert_allclose(out, derivative @ directions)
    weights = directions.T.copy()
    sums = np.empty_like(weights)
    assert point.vjp({10: weights}, out=sums) is sums
    np.testing.assert_allclose(sums, weights @ derivative)
    with np.testing.assert_raises(ValueError):
        point.jvp(directions, out=directions)
    for invalid in (out[:, ::-1], out.astype(np.float32), out.ravel()):
        with np.testing.assert_raises(ValueError):
            point.jvp(directions, out=invalid)
    expected = 10 * 3 / (1000 * 2)
    np.testing.assert_allclose(point.responses[10], [expected])
    np.testing.assert_allclose(problem.evaluate([4.]).responses[10], [expected / 2])
    del problem, point
    gc.collect()
    np.testing.assert_allclose(derivative, [[-expected / 2]])
    table = next(t for t in model.run("statics")["nodal_tables"] if t["kind"] == "displacements")
    displacement = table["values"][table["ids"].tolist().index(2)]
    # The streamed path within a memory bound gives the same displacement.
    streamed = model.run("statics", streamed=True, memory_bytes=1 << 30)
    assert streamed["report"]["dofs"] == 12
    moved = next(t for t in streamed["nodal_tables"] if t["kind"] == "displacements")
    np.testing.assert_allclose(moved["values"][moved["ids"].tolist().index(2), 0], expected)
    del streamed, moved
    del model, table
    gc.collect()
    assert not displacement.flags.writeable
    np.testing.assert_allclose(displacement[0], expected)

    tetra = directory / "tetra.oos"
    tetra.write_text("""[executive]
SolutionSequence;101
[case]
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
[model]
Parameter;GeometryDeclarations;Given
Parameter;CoupledMass;1
Node;1;;0.;0.;0.
Node;2;;1.;0.;0.
Node;3;;0.;1.;0.
Node;4;;0.;0.;1.
SolidTetra;1;1;1;2;3;4
SolidProp;1;1
IsoMat;1;1000.;;0.23;2.
SinglePointConstrList;1;123456;1;3;4
SinglePointConstrList;1;23456;2
PointForceLoad;2;2;;1.;1.;0.;0.
DesignVar;1;RHO;2.;0.;4.
DesignMatRelation1;1;IsoMat;1;density;;;;1;1.
DesignResp1;10;MASS;Weight
DesignResp1;11;JYY;Weight;;;5;5
DesignResp1;12;VOL;Volume
""", encoding="utf-8")
    deck = oo_model.read_model(tetra)
    for formulation in ("ClassicFormulation", "MixedHybridFormulation"):
        problem = oo_solver.prepare_optimization(deck, formulation=formulation)
        for rho in (0., 2.):
            point = problem.evaluate([rho])
            # Integrals of 1 and x² + z² on the unit tetrahedron.
            np.testing.assert_allclose(point.jvp([1.]), [1 / 6, 1 / 30, 0.], atol=1e-12)

# A strut restrains the wing tip. The trim sets the strut compression for the buckling subcase.
wing = Path(__file__).resolve().parents[2] / "wasm/test/fixtures/trim-preloaded-wing.oos"
trimmed = oo_solver.prepare_optimization(oo_model.read_model(wing), analysis={"aeroelastic": [1], "buckling": [2]})
base = trimmed.evaluate([0.1])
rates = base.gradients()["responses"]
plus, minus = trimmed.evaluate([0.10001]), trimmed.evaluate([0.09999])
for response in range(1, 8):
    expected = (plus.responses[response] - minus.responses[response]) / 0.00002
    np.testing.assert_allclose(rates[response][:, 0], expected, rtol=2e-4, atol=1e-10)

np.testing.assert_allclose(base.jvp([1.]), [rates[i][0, 0] for i in range(1, 8)])
np.testing.assert_allclose(base.vjp({6: [1.]}), rates[6][0])
curvature = base.hessian_vector({4: [1.]}, [[1.]])
assert curvature["report"]["uses_trim_tangents"]
np.testing.assert_allclose(curvature["values"], (plus.gradients()["responses"][4] -
                           minus.gradients()["responses"][4]) / 0.00002, rtol=2e-5)

# Physical coordinate gradients are available even when the model has no design variables.
wing_point = oo_solver.prepare_optimization(oo_model.read_model(wing.with_name("trim-rbe3-wing.oos")),
    analysis={"aeroelastic": [1], "buckling": [2]}).evaluate([])
for response, expected in [(4, -0.05032213), (5, -35.537126)]:
    xyz = wing_point.nodal_coordinate_gradient(response, [6])
    np.testing.assert_array_equal(xyz["node_ids"], [6])
    assert xyz["values"].shape == (1, 3)
    np.testing.assert_allclose(xyz["values"][0, 0], expected, rtol=1e-5)
    assert xyz["report"]["analysis_perturbations"] == 0

# Standalone coupling, with both the source model and the transfer released before reading the outputs.
transfer = oo_solver.model_from_reader(oo_model.read_model(wing.with_name("trim-rbe3-wing.oos"))).aero_transfer()
np.testing.assert_array_equal(transfer.node_ids, [1, 2, 3, 6])
np.testing.assert_array_equal(transfer.box_ids, [100, 101, 102, 103])
motion = np.sin(np.arange(1, 25, dtype=np.float64)).reshape(4, 6)
forces = np.array([1., -2., 3., 5.])
aero = transfer.box_motion(motion)
loads = transfer.structural_loads(forces)
with np.testing.assert_raises(ValueError):
    transfer.box_motion(motion.T)
with np.testing.assert_raises(ValueError):
    transfer.structural_loads([0., np.inf, 0., 0.])
del transfer
gc.collect()
assert loads.shape == (4, 6)
np.testing.assert_allclose(loads[:, 2].sum(), 7., atol=1e-12)
np.testing.assert_allclose((motion * loads).sum(), forces @ (aero["plunge"] - 0.125*aero["slope"]), atol=1e-12)

coupling = oo_solver.model_from_reader(oo_model.read_model(wing.with_name("trim-rbe3-wing.oos"))).aero_transfer()
state = coupling.solve_static(1, forces)
modes = coupling.solve_modes(2)
buckling = coupling.solve_buckling(2, state["preload"])
assert np.any(np.abs(state["box_motions"][0]["plunge"]) > 1e-8)
assert len(buckling["eigenvalues"][0]["rows"]) > 0
del coupling
q = modes["modal_basis"].generalized_loads(forces)
assert len(q) == len(modes["box_motions"]) == 2
np.testing.assert_allclose(modes["modal_basis"].generalized_loads(2 * forces), 2 * q)
