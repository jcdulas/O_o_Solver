"""Exercise the installed Reader/Solver wheels, including native gradients and buffer lifetimes."""
import gc
import tempfile
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
        modes = model.run("modes")["eigenvalues"][0]["rows"]
        assert not modes.flags.writeable
        del model
        gc.collect()
        np.testing.assert_allclose(modes["cycles"][0], expected, rtol=2e-9)

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
    expected = 10 * 3 / (1000 * 2)
    np.testing.assert_allclose(point.responses[10], [expected])
    np.testing.assert_allclose(problem.evaluate([4.]).responses[10], [expected / 2])
    del problem, point
    gc.collect()
    np.testing.assert_allclose(derivative, [[-expected / 2]])
    table = next(t for t in model.run("statics")["nodal_tables"] if t["kind"] == "displacements")
    displacement = table["values"][table["ids"].tolist().index(2)]
    del model, table
    gc.collect()
    assert not displacement.flags.writeable
    np.testing.assert_allclose(displacement[0], expected)
