"""A square membrane under unit total edge force. Return axial displacement by node ID."""
import numpy as np

from oo_model import create_model
from oo_solver import model_from_reader


def solve_plate():
    deck = create_model()
    deck.set_solution("LinearStatics")
    deck.set_case_control([
        {"name": "SinglePointConstrSelection", "fields": ["7"]},
        {"name": "DisplacementOutput", "fields": ["All"]},
        {"name": "SubcaseStart", "fields": ["1"]},
        {"name": "StaticLoadSelection", "fields": ["10"]},
    ])
    deck.add_card_batch("Node", {
        "id": np.array([1, 2, 3, 4], dtype=np.int32),
        "position1": np.array([0., 1., 1., 0.]),
        "position2": np.array([0., 0., 1., 1.]),
        "position3": 0.,
    })
    deck.add_card("ShellQuad4", [1, 1, 1, 2, 3, 4])
    deck.add_card("ShellProp", [1, 1, 0.01])
    deck.add_card("IsoMat", [1, 100000., None, 0.])
    deck.add_card("SinglePointConstrList", [7, 123456, 1, 4])
    for node in (2, 3):
        deck.add_card("PointForceLoad", [10, node, 0, 0.5, 1., 0., 0.])
    deck.apply_defaults()
    problems = deck.check_references()
    if problems:
        raise ValueError(problems[0].message)
    run = model_from_reader(deck).run("statics")
    u = next(table for table in run["nodal_tables"] if table["kind"] == "displacements")
    return dict(zip(u["ids"].tolist(), u["values"][:, 0].tolist()))


if __name__ == "__main__":
    print(solve_plate())
