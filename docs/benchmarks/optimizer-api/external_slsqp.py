"""Optional consumer example. SciPy is a dependency of this script only."""
import sys
import numpy as np
from scipy.optimize import minimize
from oo_solver import prepare_optimization
from python_client import make_deck


def optimize(path):
    problem = prepare_optimization(make_deck(path))
    objective = problem.objective_definition
    assert objective["sense"] == "min"
    scales = np.asarray([v["scale"] for v in problem.variables])
    bounds = [(v["lower"] / s, v["upper"] / s) for v, s in zip(problem.variables, scales)]
    last_z, last_point = None, None
    evaluations = 0

    def point(z):
        nonlocal last_z, last_point, evaluations
        if last_z is None or not np.array_equal(last_z, z):
            last_point = problem.evaluate(z * scales)
            last_z = z.copy()
            evaluations += 1
        return last_point

    z0 = problem.initial_values / scales
    layout = point(z0).constraint_jacobian()
    rows = len(layout.row_offsets) - 1
    # The default conservative pattern includes every variable in every row.
    # SLSQP expects dense gradients with feasible inequalities >= 0.
    result = minimize(lambda z: point(z).objective, z0, method="SLSQP", bounds=bounds,
                      jac=lambda z: point(z).vjp({objective["response_id"]: [1.]}) * scales,  # scalar volume response
                      constraints=[dict(type="ineq", fun=lambda z: -layout.constraints(point(z)),
                                        jac=lambda z: -layout.values(point(z)).reshape(rows, len(scales)) * scales)],
                      options={"ftol": 1.e-11, "maxiter": 75})
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x * scales
    final = problem.evaluate(x).optimizer_values()
    np.testing.assert_allclose(x, [.75, 2000.], rtol=1.e-6)
    np.testing.assert_allclose(final["objective"], 2.25, rtol=1.e-6)
    assert not final["partial"] and np.max(final["constraints"]) < 1.e-8
    print(f"{path}: optimum={x}, volume={final['objective']:.8f}, evaluations={evaluations}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        optimize(arg)
