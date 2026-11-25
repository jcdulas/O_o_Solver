"""Generate identical in-memory card inputs for the three independent consumers."""
import json
import sys
from pathlib import Path


def fixture(nx, ny):
    cards = []
    def add(kind, *fields):
        cards.append([kind, *fields])

    for j in range(ny + 1):
        for i in range(nx + 1):
            node = 1 + j * (nx + 1) + i
            add("Node", node, None, 3 * i / nx, j / max(1, ny), 0.)
            if i == 0 or ny == 0:
                add("SinglePointConstrList", 1, 123456 if i == 0 else 23456, node)
            if i == nx:
                force = 10. if not ny else 10. / ny * (0.5 if j in (0, ny) else 1.)
                add("PointForceLoad", 2, node, 0, force, 1., 0., 0.)
    add("IsoMat", 1, 1000., None, 0., 1.)
    if ny:
        for j in range(ny):
            for i in range(nx):
                a = 1 + j * (nx + 1) + i
                add("ShellQuad4", 1 + j * nx + i, 1, a, a + 1, a + nx + 2, a + nx + 1)
        add("ShellProp", 1, 1, 2., 1, None, 1)
        prop, field = "ShellProp", "thickness"
    else:
        add("Rod2", 1, 1, 1, 2)
        add("RodProp", 1, 1, 2.)
        prop, field = "RodProp", "area"
    add("DesignVar", 1, "SIZE", 2., 0.5, 5.)
    add("DesignVar", 2, "E", 1000., 500., 2000.)
    add("DesignPropRelation1", 1, prop, 1, field, None, None, None, 1, 1.)
    add("DesignMatRelation1", 2, "IsoMat", 1, "youngsModulus", None, None, None, 2, 1.)
    add("DesignResp1", 10, "TIP", "Displacement", None, None, 1, None, nx + 1)
    add("DesignResp1", 11, "VOLUME", "Volume")
    add("DesignConstr", 7, 10, 0.005, None, 0.02)
    add("DesignConstr", 7, 11, None, None, 8.)
    return cards


if __name__ == "__main__":
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    for name, nx, ny in [("rod", 1, 0), ("panel", 16, 8)]:
        (dest / (name + ".json")).write_text(json.dumps(fixture(nx, ny)))
