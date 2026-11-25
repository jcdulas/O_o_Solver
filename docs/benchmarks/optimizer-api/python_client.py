"""Run with the installed Model and the Solver under test."""
import json
import sys
from time import perf_counter

import numpy as np
import oo_model
import oo_solver


def make_deck(path):
    deck = oo_model.create_model()
    deck.set_solution("LinearStatics")
    deck.set_case_control([{"name": name, "fields": [value]} for name, value in [
        ("SinglePointConstrSelection", "1"), ("DisplacementOutput", "All"), ("SubcaseStart", "1"),
        ("StaticLoadSelection", "2"), ("SubcaseDesignConstrSelection", "7"), ("DesignObjectiveSelection", "11")]])
    with open(path) as source:
        for kind, *fields in json.load(source):
            deck.add_card(kind, fields)
    deck.apply_defaults()
    return deck


def run(path):
    deck = make_deck(path)
    start = perf_counter()
    problem = oo_solver.prepare_optimization(deck)
    prepared = perf_counter()
    point = problem.evaluate(problem.initial_values)
    evaluated = perf_counter()
    gradient = point.gradients()
    differentiated = perf_counter()
    directions = np.eye(2)
    out = np.empty((2, 2))
    point.jvp(directions, out=out)
    warm = perf_counter()
    for _ in range(20):
        point.jvp(directions, out=out)
    end = perf_counter()
    print(f"{path}: prepare_ms={(prepared-start)*1000:.3f} evaluate_ms={(evaluated-prepared)*1000:.3f} "
          f"gradient_ms={(differentiated-evaluated)*1000:.3f} warm_jvp_ms={(end-warm)*50:.3f} "
          f"gradient_factorisations={gradient['report']['factorisations']} output_bytes={out.nbytes}")
    # Compare eight native calls with one batched call at the same retained point.
    vectors = np.arange(16., dtype=float).reshape(8, 2)
    batch = vectors.T.copy()
    separate = np.empty((8, 2))
    together = np.empty((2, 8))
    pairs = list(zip(vectors, separate))
    point.jvp(batch, out=together)
    before = perf_counter()
    for _ in range(20):
        for direction, result in pairs:
            point.jvp(direction, out=result)
    between = perf_counter()
    for _ in range(20):
        point.jvp(batch, out=together)
    after = perf_counter()
    np.testing.assert_allclose(together, separate.T, atol=1.e-10)
    print(f"eight_separate_ms={(between-before)*50:.4f} eight_batched_ms={(after-between)*50:.4f}")
    wide = np.arange(82., dtype=float).reshape(2, 41)
    np.testing.assert_allclose(point.jvp(wide), out @ wide, atol=1.e-9)
    weights = wide.T.copy()
    np.testing.assert_allclose(point.vjp({10: weights[:, :1], 11: weights[:, 1:]}), weights @ out, atol=1.e-9)
    summary = point.optimizer_values()
    assert not summary["partial"] and not summary["uncertain_mode"]
    np.testing.assert_allclose(summary["constraints"], [c["normalised"] for c in point.constraints])
    assert problem.capabilities["analyses"] and [v["scale"] for v in problem.variables] == [2., 1000.]
    np.testing.assert_allclose(summary["objective"], 6.)
    np.testing.assert_allclose(out, [[-.0075, -.000015], [3., 0.]], atol=1.e-9)
    csr = point.constraint_jacobian([[0, 1], [0, 1], [0]])
    np.testing.assert_array_equal(csr.row_offsets, [0, 2, 4, 5])
    second = problem.evaluate([3., 1000.])
    values = np.empty(5)
    assert csr.values(second, out=values) is values
    np.testing.assert_allclose(values, np.concatenate([r[cols] for r, cols in zip(
        second.gradients()["constraints"], [[0, 1], [0, 1], [0]])]), atol=1.e-9)
    np.testing.assert_allclose(csr.constraints(second), [c["normalised"] for c in second.constraints])
    for bad in ([[], [], []], [[0, 0], [0, 1], [0]], [[0, 2], [0, 1], [0]]):
        with np.testing.assert_raises(ValueError):
            point.constraint_jacobian(bad).values(point)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(arg)
