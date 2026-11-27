"""Parametric models of the scaling benchmark, written in the .oos text format of O_o_Model.

Four models, each refined by one size parameter n:

- ``plate``: a flat square plate of ShellQuad4, simply supported on its four sides;
- ``cylinder``: a closed circular cylinder of ShellQuad4 between two end diaphragms;
- ``wingbox``: a box beam of ShellQuad4 (two skins, two spars and ribs) clamped at the root;
- ``block``: an elongated block of SolidHexa (8 nodes) clamped on one end face.

Each model writes one deck per analysis (statics, modes, buckling) and knows the value its results are compared
with: a closed form of plate or shell theory for the plate and the cylinder, a beam model for the box and the block.

Run ``python models.py --ladder`` to print the sizes chosen for the benchmark's DOF counts, ``python models.py
plate 40 statics plate.oos`` to write one deck, and ``python models.py --check`` to read every deck of a small size
with oo_model and list its diagnostics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ANALYSES = ("statics", "modes", "buckling")
SOLUTION_SEQUENCES = {"statics": "LinearStatics", "modes": "NormalModes", "buckling": "LinearBuckling"}

# The identifiers shared by every deck.
CONSTRAINT_SET = 1
LOAD_SET = 2
MODES_METHOD = 10
BUCKLING_METHOD = 20
MODE_COUNT = 10

# The DOF counts of the benchmark (about), from small models to large ones.
BENCHMARK_DOFS = (1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 2.0e6)

# Grid points per SinglePointConstrList record.
NODES_PER_CONSTRAINT_RECORD = 64

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------------------------------------------------
# Writing records


def real(value: float) -> str:
    """A real field: 12 significant digits, always with a decimal point or an exponent (else it reads as an integer)."""
    text = format(value, ".12g")
    if "." in text or "e" in text:
        return text
    return text + "."


def field(value) -> str:
    """A field of a record: blank for None, a real for a float, the text of anything else."""
    if value is None:
        return ""
    if isinstance(value, float):
        return real(value)
    return str(value)


class DeckWriter:
    """Writes the records of a deck, one per line: the record name, then its fields separated by ';'."""

    def __init__(self, stream):
        self.stream = stream

    def line(self, text: str) -> None:
        self.stream.write(text + "\n")

    def record(self, name: str, *fields) -> None:
        self.stream.write(";".join([name, *(field(value) for value in fields)]) + "\n")


def components(*digits: int) -> str:
    """The component digits of a constraint (1-3 translations, 4-6 rotations), sorted as the format keeps them."""
    return "".join(str(digit) for digit in sorted(set(digits)))


def write_header(deck: DeckWriter, analysis: str) -> None:
    """The executive and case sections of an analysis, and the cards of the model section that depend on it.

    statics: one subcase with the load. modes: one subcase with the eigenvalue method (the first MODE_COUNT modes).
    buckling: the static preload subcase, then the buckling subcase (the first buckling factor above 0).
    """
    deck.line("[executive]")
    deck.record("SolutionSequence", SOLUTION_SEQUENCES[analysis])
    deck.line("[case]")
    deck.record("DisplacementOutput", "All")
    deck.record("SubcaseStart", 1)
    deck.record("SinglePointConstrSelection", CONSTRAINT_SET)
    if analysis == "modes":
        deck.record("EigenMethodSelection", MODES_METHOD)
    else:
        deck.record("StaticLoadSelection", LOAD_SET)
    if analysis == "buckling":
        deck.record("SubcaseStart", 2)
        deck.record("SinglePointConstrSelection", CONSTRAINT_SET)
        deck.record("EigenMethodSelection", BUCKLING_METHOD)
        deck.record("StaticSubcaseSelection", 1)
    deck.line("[model]")
    # No edge or facet declaration: the fold angle decides every edge (read by the mixed hybrid formulation only).
    deck.record("Parameter", "GeometryDeclarations", "Given")
    if analysis == "modes":
        # LanczosEigenMethod: id, lowerFrequency, upperFrequency, desiredModeCount.
        deck.record("LanczosEigenMethod", MODES_METHOD, None, None, MODE_COUNT)
    elif analysis == "buckling":
        # The first buckling factor above 0.
        deck.record("LanczosEigenMethod", BUCKLING_METHOD, 0.0, None, 1)


def write_constraints(deck: DeckWriter, held: dict[int, str]) -> None:
    """SinglePointConstrList records: the grid points that hold the same components, a few dozen per record."""
    by_components: dict[str, list[int]] = {}
    for node, digits in held.items():
        by_components.setdefault(digits, []).append(node)
    for digits, nodes in sorted(by_components.items()):
        for start in range(0, len(nodes), NODES_PER_CONSTRAINT_RECORD):
            chunk = nodes[start:start + NODES_PER_CONSTRAINT_RECORD]
            deck.record("SinglePointConstrList", CONSTRAINT_SET, digits, *chunk)


def edge_weights(divisions: int) -> list[float]:
    """The share of each grid point of a uniformly divided edge in a uniform line load (half at the two ends)."""
    weights = [1.0 / divisions] * (divisions + 1)
    weights[0] = weights[-1] = 0.5 / divisions
    return weights


def held_dof_count(held: dict[int, str]) -> int:
    return sum(len(digits) for digits in held.values())


# ---------------------------------------------------------------------------------------------------------------------
# Reading results


def nodal_values(result: dict, node_ids, kind: str = "displacements") -> np.ndarray:
    """The rows (6 components) of the given grid points in the first nodal table of a kind."""
    table = next(t for t in result["nodal_tables"] if t["kind"] == kind)
    ids = np.asarray(table["ids"])
    wanted = np.asarray(node_ids)
    if np.all(ids[1:] > ids[:-1]):
        rows = np.minimum(np.searchsorted(ids, wanted), len(ids) - 1)
    else:
        position = {int(node): row for row, node in enumerate(ids)}
        rows = np.array([position[int(node)] for node in wanted])
    if not np.array_equal(ids[rows], wanted):
        raise ValueError("a grid point of the check is missing from the displacement table")
    return np.asarray(table["values"])[rows]


def eigenvalue_rows(result: dict):
    """The eigenvalue rows of the first eigenvalue table (modes: frequencies; buckling: load factors)."""
    return result["eigenvalues"][0]["rows"]


def first_frequency(result: dict) -> float:
    return float(eigenvalue_rows(result)["cycles"][0])


def first_buckling_factor(result: dict) -> float:
    return float(eigenvalue_rows(result)["eigenvalue"][0])


# ---------------------------------------------------------------------------------------------------------------------
# Reference values


def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """D = E t³ / (12 (1 − ν²))."""
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def navier_centre_deflection(pressure: float, side: float, rigidity: float, terms: int = 201) -> float:
    """Centre deflection of a simply supported square thin plate under a uniform pressure (Navier's double series).

    w(a/2, a/2) = 16 q / (π⁶ D) Σ_{m,n odd} sin(mπ/2) sin(nπ/2) / (m n (m²/a² + n²/a²)²)
    (S. Timoshenko, S. Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., 1959, section 30).
    """
    total = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            sign = (-1) ** ((m - 1) // 2 + (n - 1) // 2)
            total += sign / (m * n * ((m / side) ** 2 + (n / side) ** 2) ** 2)
    return 16.0 * pressure / (math.pi**6 * rigidity) * total


def sanders_cylinder_stiffness(axial_wave: float, waves: int, radius: float, thickness: float, young: float,
                               poisson: float) -> np.ndarray:
    """The 3 × 3 stiffness of one mode of a cylinder between end diaphragms, in Sanders' thin shell theory.

    The mode u = A cos(αx) cos(nθ), v = B sin(αx) sin(nθ), w = C sin(αx) cos(nθ) (u axial, v tangential, w outward,
    α = mπ/L) gives every strain as a coefficient row times one product of a sine and a cosine; each product squared
    integrates to the same πRL/2 over the shell (n ≥ 1), so that the energy is (πRL/4) qᵀ K q with q = (A, B, C).
    Strains (J. L. Sanders Jr., An improved first-approximation theory for thin shells, NASA TR R-24, 1959):
        εx = u,x    εθ = (v,θ + w)/R    γ = v,x + u,θ/R
        κx = −w,xx  κθ = (v,θ − w,θθ)/R²    τ = (−2 w,xθ + 3/2 v,x − u,θ/(2R))/R
    and K = Et/(1 − ν²) [εx² + εθ² + 2ν εx εθ + (1 − ν)/2 γ²] + D [κx² + κθ² + 2ν κx κθ + (1 − ν)/2 τ²].
    """
    alpha, n, r = axial_wave, float(waves), radius
    strain_x = np.array([-alpha, 0.0, 0.0])
    strain_hoop = np.array([0.0, n / r, 1.0 / r])
    shear = np.array([-n / r, alpha, 0.0])
    curvature_x = np.array([0.0, 0.0, alpha**2])
    curvature_hoop = np.array([0.0, n / r**2, n**2 / r**2])
    twist = np.array([n / (2.0 * r**2), 1.5 * alpha / r, 2.0 * n * alpha / r])

    def plane_stress(first, second, third):
        return (np.outer(first, first) + np.outer(second, second)
                + poisson * (np.outer(first, second) + np.outer(second, first))
                + 0.5 * (1.0 - poisson) * np.outer(third, third))

    membrane = young * thickness / (1.0 - poisson**2)
    bending = plate_rigidity(young, poisson, thickness)
    return (membrane * plane_stress(strain_x, strain_hoop, shear)
            + bending * plane_stress(curvature_x, curvature_hoop, twist))


def sanders_cylinder_minimum(length: float, radius: float, thickness: float, young: float, poisson: float,
                             value_of, axial_modes: int = 30, waves: int = 80) -> float:
    """The smallest value_of(alpha, lowest eigenvalue of K) over the modes m = 1.., n = 1.. of the cylinder."""
    lowest = math.inf
    for m in range(1, axial_modes + 1):
        alpha = m * math.pi / length
        for n in range(1, waves + 1):
            stiffness = sanders_cylinder_stiffness(alpha, n, radius, thickness, young, poisson)
            lowest = min(lowest, value_of(alpha, np.linalg.eigvalsh(stiffness)[0]))
    return lowest


def cowper_shear_coefficient(poisson: float) -> float:
    """Shear coefficient of a rectangular section, κ = 10 (1 + ν) / (12 + 11 ν).

    G. R. Cowper, The shear coefficient in Timoshenko's beam theory, J. Appl. Mech. 33 (1966) 335-340.
    """
    return 10.0 * (1.0 + poisson) / (12.0 + 11.0 * poisson)


def timoshenko_tip_deflection(force: float, length: float, bending_stiffness: float, shear_stiffness: float) -> float:
    """Tip deflection of a cantilever under a tip force: P L³ / (3 EI) + P L / (κ G A)."""
    return force * length**3 / (3.0 * bending_stiffness) + force * length / shear_stiffness


def engesser_buckling_load(length: float, bending_stiffness: float, shear_stiffness: float) -> float:
    """Critical load of a clamped-free column with shear: P = P_E / (1 + P_E / (κ G A)), P_E = π² EI / (4 L²).

    S. P. Timoshenko, J. M. Gere, Theory of Elastic Stability, 2nd ed., 1961, section 2.17.
    """
    euler = math.pi**2 * bending_stiffness / (4.0 * length**2)
    return euler / (1.0 + euler / shear_stiffness)


def timoshenko_cantilever_frequency(length: float, bending_stiffness: float, shear_stiffness: float,
                                    mass: float, rotary_inertia: float, elements: int = 400) -> float:
    """The first natural frequency, in Hz, of a clamped-free Timoshenko beam.

    Computed on a fine model of two-node elements (deflection w and section rotation ψ at each node, linear
    interpolation, the shear strain w' − ψ taken at the element centre, consistent mass of the deflection and of the
    rotation); with 400 elements its error is far below the differences between the beam and the model it checks.
    mass is the mass per unit length, rotary_inertia the mass moment of inertia of the section per unit length.
    """
    step = length / elements
    bending = bending_stiffness / step * np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
    shear_row = np.array([-1.0 / step, -0.5, 1.0 / step, -0.5])  # (w1, ψ1, w2, ψ2) -> w' − ψ at the centre
    element_stiffness = bending + shear_stiffness * step * np.outer(shear_row, shear_row)
    consistent = step / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    element_mass = np.zeros((4, 4))
    element_mass[np.ix_([0, 2], [0, 2])] = mass * consistent
    element_mass[np.ix_([1, 3], [1, 3])] = rotary_inertia * consistent

    size = 2 * (elements + 1)
    stiffness = np.zeros((size, size))
    masses = np.zeros((size, size))
    for element in range(elements):
        dofs = slice(2 * element, 2 * element + 4)
        stiffness[dofs, dofs] += element_stiffness
        masses[dofs, dofs] += element_mass
    stiffness, masses = stiffness[2:, 2:], masses[2:, 2:]  # the first node is clamped

    # K x = ω² M x as a symmetric standard problem: with M = L Lᵀ, (L⁻¹ K L⁻ᵀ) y = ω² y.
    inverse = np.linalg.inv(np.linalg.cholesky(masses))
    omega_squared = np.linalg.eigvalsh(inverse @ stiffness @ inverse.T)[0]
    return math.sqrt(omega_squared) / TWO_PI


# ---------------------------------------------------------------------------------------------------------------------
# The models


class BenchmarkModel:
    """A model refined by one size parameter n. Subclasses give the mesh, the loads and the reference values."""

    name = ""
    title = ""
    # What each analysis compares with its reference.
    quantities: dict[str, str] = {}

    def sizes(self) -> range:
        """The valid values of the size parameter, ascending."""
        raise NotImplementedError

    def grid_points(self, n: int) -> int:
        raise NotImplementedError

    def dofs(self, n: int) -> int:
        """The free degrees of freedom: the components the elements use, less the held ones."""
        raise NotImplementedError

    def write_model(self, deck: DeckWriter, n: int, analysis: str) -> None:
        raise NotImplementedError

    def measured(self, analysis: str, result: dict, n: int) -> float | None:
        """The value of a result that is compared with the reference."""
        raise NotImplementedError

    def reference(self, analysis: str) -> float | None:
        """The reference value of an analysis, or None when the model has none."""
        raise NotImplementedError

    def write(self, path: Path, n: int, analysis: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            deck = DeckWriter(stream)
            write_header(deck, analysis)
            self.write_model(deck, n, analysis)

    def size_for_dofs(self, target: float) -> int:
        """The size whose DOF count is nearest to the target, on a logarithmic scale."""
        best, best_distance = None, math.inf
        for n in self.sizes():
            count = self.dofs(n)
            distance = abs(math.log(count / target))
            if distance < best_distance:
                best, best_distance = n, distance
            if count > 2.0 * target:
                break
        return best


class SquarePlate(BenchmarkModel):
    """A flat square plate of n × n ShellQuad4 elements, simply supported on its four sides.

    Each side holds the deflection and the rotation that would tilt the side out of its line (the support of the
    thin-plate solutions, without the edge layer of a support that holds the deflection alone). The side x = 0 also
    holds the in-plane displacement normal to it, and its corner at the origin the other in-plane displacement: the
    in-plane rigid motions are held without restraining the plate.
    statics: a uniform pressure. modes: the first 10 modes. buckling: a uniform compression applied on the side x = a,
    reacted by the side x = 0.
    """

    name = "plate"
    title = "Square plate, ShellQuad4"
    quantities = {
        "statics": "centre deflection against Navier's thin-plate series",
        "modes": "first frequency against the thin-plate value π/a² √(D/ρt)",
        "buckling": "first factor against the thin-plate value 4π²D/(a² N)",
    }

    SIDE = 1.0
    THICKNESS = 0.01
    YOUNG = 7.0e10
    POISSON = 0.3
    DENSITY = 2700.0
    PRESSURE = 1.0e3
    EDGE_LOAD = 1.0e5  # compression per unit length of the loaded side (buckling)

    def sizes(self) -> range:
        return range(4, 4001, 2)  # even: a grid point at the centre

    def grid_points(self, n: int) -> int:
        return (n + 1) ** 2

    def dofs(self, n: int) -> int:
        return 6 * self.grid_points(n) - held_dof_count(self.held(n))

    def node(self, n: int, i: int, j: int) -> int:
        return j * (n + 1) + i + 1

    def held(self, n: int) -> dict[int, str]:
        held = {}
        for j in range(n + 1):
            sides = range(n + 1) if j in (0, n) else (0, n)  # the grid points on the four sides, each once
            for i in sides:
                digits = set()
                if i in (0, n):
                    digits |= {3, 4}  # deflection, rotation about x
                if j in (0, n):
                    digits |= {3, 5}  # deflection, rotation about y
                if i == 0:
                    digits.add(1)  # reaction of the compression
                if i == 0 and j == 0:
                    digits.add(2)
                held[self.node(n, i, j)] = components(*digits)
        return held

    def write_model(self, deck: DeckWriter, n: int, analysis: str) -> None:
        step = self.SIDE / n
        for j in range(n + 1):
            for i in range(n + 1):
                deck.record("Node", self.node(n, i, j), None, i * step, j * step, 0.0)
        element = 1
        for j in range(n):
            for i in range(n):
                deck.record("ShellQuad4", element, 1, self.node(n, i, j), self.node(n, i + 1, j),
                            self.node(n, i + 1, j + 1), self.node(n, i, j + 1))
                element += 1
        # ShellProp: id, membrane material, thickness, bending material, bending ratio, shear material.
        deck.record("ShellProp", 1, 1, self.THICKNESS, 1, None, 1)
        # IsoMat: id, Young's modulus, shear modulus (from ν), Poisson's ratio, density.
        deck.record("IsoMat", 1, self.YOUNG, None, self.POISSON, self.DENSITY)
        write_constraints(deck, self.held(n))
        if analysis == "statics":
            deck.record("ShellPressureLoad", LOAD_SET, self.PRESSURE, 1, "Through", n * n)
        elif analysis == "buckling":
            for j, weight in enumerate(edge_weights(n)):
                deck.record("PointForceLoad", LOAD_SET, self.node(n, n, j), None,
                            self.EDGE_LOAD * self.SIDE * weight, -1.0, 0.0, 0.0)

    def measured(self, analysis: str, result: dict, n: int) -> float | None:
        if analysis == "statics":
            centre = self.node(n, n // 2, n // 2)
            return float(nodal_values(result, [centre])[0, 2])
        if analysis == "modes":
            return first_frequency(result)
        return first_buckling_factor(result)

    def reference(self, analysis: str) -> float | None:
        rigidity = plate_rigidity(self.YOUNG, self.POISSON, self.THICKNESS)
        if analysis == "statics":
            return navier_centre_deflection(self.PRESSURE, self.SIDE, rigidity)
        if analysis == "modes":
            return math.pi / self.SIDE**2 * math.sqrt(rigidity / (self.DENSITY * self.THICKNESS))
        return 4.0 * math.pi**2 * rigidity / (self.SIDE**2 * self.EDGE_LOAD)


class Cylinder(BenchmarkModel):
    """A closed circular cylinder of ShellQuad4 elements between two end diaphragms.

    The axis is x; n elements around, about as many per unit length along (an even number, for a ring at mid-length).
    Both end rings hold their displacements in the plane of the ring (the diaphragms). The ring at mid-length, a plane
    of symmetry of the loads, holds the axial displacement: the cylinder stays free to lengthen and the modes with an
    odd number of axial half-waves, which have no axial displacement there, are those of the end diaphragms alone. (A
    single grid point holding the axial displacement would not do: one mode of each pair would carry the axial motion
    of the whole cylinder and lose about 2 % of its frequency.)
    statics: an internal pressure. modes: the first 10 modes. buckling: a uniform axial compression, applied as equal
    and opposite forces on the two end rings.
    """

    name = "cylinder"
    title = "Cylinder, ShellQuad4"
    quantities = {
        "statics": "mean radial displacement at mid-length against the membrane value p R²/(E t)",
        "modes": "first frequency against Sanders' shell theory (end diaphragms)",
        "buckling": "first factor against Sanders' shell theory (end diaphragms)",
    }

    RADIUS = 1.0
    LENGTH = 2.0
    THICKNESS = 0.01
    YOUNG = 7.0e10
    POISSON = 0.3
    DENSITY = 2700.0
    PRESSURE = 1.0e5
    AXIAL_LOAD = 1.0e6  # compression per unit length of the circumference (buckling)

    def sizes(self) -> range:
        return range(16, 8001, 4)

    def axial_elements(self, n: int) -> int:
        """An even number of elements along the axis, for elements about as long as they are wide."""
        return max(2, 2 * round(n * self.LENGTH / (2.0 * TWO_PI * self.RADIUS)))

    def grid_points(self, n: int) -> int:
        return n * (self.axial_elements(n) + 1)

    def dofs(self, n: int) -> int:
        return 6 * self.grid_points(n) - held_dof_count(self.held(n))

    def node(self, n: int, i: int, j: int) -> int:
        """Grid point j (around, from the y axis toward the z axis) of ring i (along the axis)."""
        return i * n + (j % n) + 1

    def held(self, n: int) -> dict[int, str]:
        last = self.axial_elements(n)
        held = {}
        for j in range(n):
            held[self.node(n, 0, j)] = components(2, 3)  # end diaphragms
            held[self.node(n, last, j)] = components(2, 3)
            held[self.node(n, last // 2, j)] = components(1)  # plane of symmetry
        return held

    def write_model(self, deck: DeckWriter, n: int, analysis: str) -> None:
        rings = self.axial_elements(n)
        for i in range(rings + 1):
            x = self.LENGTH * i / rings
            for j in range(n):
                angle = TWO_PI * j / n
                deck.record("Node", self.node(n, i, j), None, x, self.RADIUS * math.cos(angle),
                            self.RADIUS * math.sin(angle))
        # Corners in this order give the outward normal, along which a positive pressure acts.
        element = 1
        for i in range(rings):
            for j in range(n):
                deck.record("ShellQuad4", element, 1, self.node(n, i, j), self.node(n, i, j + 1),
                            self.node(n, i + 1, j + 1), self.node(n, i + 1, j))
                element += 1
        deck.record("ShellProp", 1, 1, self.THICKNESS, 1, None, 1)
        deck.record("IsoMat", 1, self.YOUNG, None, self.POISSON, self.DENSITY)
        write_constraints(deck, self.held(n))
        if analysis == "statics":
            deck.record("ShellPressureLoad", LOAD_SET, self.PRESSURE, 1, "Through", rings * n)
        elif analysis == "buckling":
            force = self.AXIAL_LOAD * TWO_PI * self.RADIUS / n
            for j in range(n):
                deck.record("PointForceLoad", LOAD_SET, self.node(n, 0, j), None, force, 1.0, 0.0, 0.0)
                deck.record("PointForceLoad", LOAD_SET, self.node(n, rings, j), None, force, -1.0, 0.0, 0.0)

    def measured(self, analysis: str, result: dict, n: int) -> float | None:
        if analysis == "statics":
            middle = self.axial_elements(n) // 2
            rows = nodal_values(result, [self.node(n, middle, j) for j in range(n)])
            angles = TWO_PI * np.arange(n) / n
            radial = rows[:, 1] * np.cos(angles) + rows[:, 2] * np.sin(angles)
            return float(np.mean(radial))
        if analysis == "modes":
            return first_frequency(result)
        return first_buckling_factor(result)

    def reference(self, analysis: str) -> float | None:
        if analysis == "statics":
            # Membrane hoop strain of an open-ended cylinder: w = p R² / (E t).
            return self.PRESSURE * self.RADIUS**2 / (self.YOUNG * self.THICKNESS)
        geometry = (self.LENGTH, self.RADIUS, self.THICKNESS, self.YOUNG, self.POISSON)
        if analysis == "modes":
            # Kinetic energy (πRL/4) ρ t ω² (A² + B² + C²): ω² = eigenvalue / (ρ t).
            surface_mass = self.DENSITY * self.THICKNESS
            return sanders_cylinder_minimum(
                *geometry, lambda alpha, eigenvalue: math.sqrt(eigenvalue / surface_mass) / TWO_PI)
        # The work of the axial force −N on the axial gradient of the three translations, (πRL/4) N α² (A² + B² + C²),
        # as in the differential stiffness of the shell: factor = eigenvalue / (N α²).
        return sanders_cylinder_minimum(
            *geometry, lambda alpha, eigenvalue: eigenvalue / (self.AXIAL_LOAD * alpha**2))


class WingBox(BenchmarkModel):
    """A box beam of ShellQuad4 elements: two skins, two spars and ribs, clamped at the root, a tip load.

    Span along x, the skins at z = ±depth/2 between the spars at y = 0 and y = width. A rib closes the end of each of
    the 20 bays (the root is clamped and needs none). The size n is the number of elements across the depth of a spar;
    the elements are squares (4 n across a skin, 2 n along a bay).
    statics: a vertical tip force shared by the two spar webs. modes: the first 10 modes. buckling: the same tip force
    (the compressed skin buckles between the ribs and the spars; no closed form).
    """

    name = "wingbox"
    title = "Wing box, ShellQuad4"
    quantities = {
        "statics": "mean tip deflection against Timoshenko beam theory (bending + shear of the webs)",
        "modes": "first frequency against a Timoshenko beam with the ribs' mass spread along the span",
        "buckling": "no reference (skin panel buckling between ribs and spars)",
    }

    SPAN = 10.0
    WIDTH = 1.0
    DEPTH = 0.25
    BAYS = 20
    SKIN_THICKNESS = 0.004
    SPAR_THICKNESS = 0.006
    RIB_THICKNESS = 0.003
    YOUNG = 7.0e10
    POISSON = 0.3
    DENSITY = 2700.0
    TIP_FORCE = 1.0e3
    SKIN, SPAR, RIB = 1, 2, 3  # property identifiers

    def sizes(self) -> range:
        return range(1, 201)

    # The mesh: the box section is a closed loop of grid points (lower skin, rear spar, upper skin, front spar),
    # repeated at every station along the span; the ribs add the grid points inside the loop.

    def chord_elements(self, n: int) -> int:
        return 4 * n

    def stations(self, n: int) -> int:
        """The number of elements along the span."""
        return self.BAYS * 2 * n

    def loop_size(self, n: int) -> int:
        return 2 * self.chord_elements(n) + 2 * n

    def grid_points(self, n: int) -> int:
        inside_rib = (self.chord_elements(n) - 1) * (n - 1)
        return (self.stations(n) + 1) * self.loop_size(n) + self.BAYS * inside_rib

    def dofs(self, n: int) -> int:
        return 6 * (self.grid_points(n) - self.loop_size(n))  # the root section is clamped

    def loop_node(self, n: int, station: int, k: int) -> int:
        return station * self.loop_size(n) + (k % self.loop_size(n)) + 1

    def loop_position(self, n: int, k: int) -> tuple[float, float]:
        """(y, z) of grid point k of the loop."""
        c, half = self.chord_elements(n), 0.5 * self.DEPTH
        if k < c:  # lower skin, toward the rear spar
            return self.WIDTH * k / c, -half
        if k < c + n:  # rear spar, upward
            return self.WIDTH, -half + self.DEPTH * (k - c) / n
        if k < 2 * c + n:  # upper skin, toward the front spar
            return self.WIDTH * (1.0 - (k - c - n) / c), half
        return 0.0, half - self.DEPTH * (k - 2 * c - n) / n  # front spar, downward

    def is_skin(self, n: int, k: int) -> bool:
        """Whether the loop element from grid point k to k + 1 is a skin element."""
        c = self.chord_elements(n)
        return k < c or c + n <= k < 2 * c + n

    def rib_node(self, n: int, rib: int, jy: int, jz: int) -> int:
        """Grid point (jy across the chord from the front spar, jz up from the lower skin) of rib 1..BAYS."""
        c = self.chord_elements(n)
        station = rib * 2 * n
        if jz == 0:
            return self.loop_node(n, station, jy)
        if jy == c:
            return self.loop_node(n, station, c + jz)
        if jz == n:
            return self.loop_node(n, station, c + n + (c - jy))
        if jy == 0:
            return self.loop_node(n, station, 2 * c + n + (n - jz))
        first = (self.stations(n) + 1) * self.loop_size(n)
        return first + (rib - 1) * (c - 1) * (n - 1) + (jz - 1) * (c - 1) + jy

    def tip_spar_forces(self, n: int) -> list[tuple[int, float]]:
        """(grid point, force) of the tip load: each spar web takes half, spread evenly over its depth."""
        c = self.chord_elements(n)
        forces = []
        for jz, weight in enumerate(edge_weights(n)):
            share = 0.5 * self.TIP_FORCE * weight
            forces.append((self.rib_node(n, self.BAYS, 0, jz), share))  # front spar of the tip rib
            forces.append((self.rib_node(n, self.BAYS, c, jz), share))  # rear spar
        return forces

    def write_model(self, deck: DeckWriter, n: int, analysis: str) -> None:
        c, stations, loop = self.chord_elements(n), self.stations(n), self.loop_size(n)
        for station in range(stations + 1):
            x = self.SPAN * station / stations
            for k in range(loop):
                y, z = self.loop_position(n, k)
                deck.record("Node", self.loop_node(n, station, k), None, x, y, z)
        for rib in range(1, self.BAYS + 1):
            x = self.SPAN * rib / self.BAYS
            for jz in range(1, n):
                for jy in range(1, c):
                    deck.record("Node", self.rib_node(n, rib, jy, jz), None, x, self.WIDTH * jy / c,
                                -0.5 * self.DEPTH + self.DEPTH * jz / n)
        element = 1
        for station in range(stations):
            for k in range(loop):
                prop = self.SKIN if self.is_skin(n, k) else self.SPAR
                deck.record("ShellQuad4", element, prop, self.loop_node(n, station, k),
                            self.loop_node(n, station + 1, k), self.loop_node(n, station + 1, k + 1),
                            self.loop_node(n, station, k + 1))
                element += 1
        for rib in range(1, self.BAYS + 1):
            for jz in range(n):
                for jy in range(c):
                    deck.record("ShellQuad4", element, self.RIB, self.rib_node(n, rib, jy, jz),
                                self.rib_node(n, rib, jy + 1, jz), self.rib_node(n, rib, jy + 1, jz + 1),
                                self.rib_node(n, rib, jy, jz + 1))
                    element += 1
        for prop, thickness in ((self.SKIN, self.SKIN_THICKNESS), (self.SPAR, self.SPAR_THICKNESS),
                                (self.RIB, self.RIB_THICKNESS)):
            deck.record("ShellProp", prop, 1, thickness, 1, None, 1)
        deck.record("IsoMat", 1, self.YOUNG, None, self.POISSON, self.DENSITY)
        write_constraints(deck, {self.loop_node(n, 0, k): components(1, 2, 3, 4, 5, 6) for k in range(loop)})
        if analysis in ("statics", "buckling"):
            for node, force in self.tip_spar_forces(n):
                deck.record("PointForceLoad", LOAD_SET, node, None, force, 0.0, 0.0, 1.0)

    def measured(self, analysis: str, result: dict, n: int) -> float | None:
        if analysis == "statics":
            tip = self.stations(n)
            rows = nodal_values(result, [self.loop_node(n, tip, k) for k in range(self.loop_size(n))])
            return float(np.mean(rows[:, 2]))
        if analysis == "modes":
            return first_frequency(result)
        return first_buckling_factor(result)

    def beam(self) -> dict[str, float]:
        """The section of the equivalent beam for bending in the x-z plane (midplane dimensions)."""
        b, h = self.WIDTH, self.DEPTH
        ts, tw, tr = self.SKIN_THICKNESS, self.SPAR_THICKNESS, self.RIB_THICKNESS
        second_moment = 2.0 * (b * ts * (h / 2.0) ** 2 + b * ts**3 / 12.0) + 2.0 * tw * h**3 / 12.0
        shear_modulus = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        rib_pitch = self.SPAN / self.BAYS
        return {
            "bending_stiffness": self.YOUNG * second_moment,
            # The vertical shear is carried by the two spar webs.
            "shear_stiffness": shear_modulus * 2.0 * h * tw,
            # Skins and spars, and the ribs spread along the span.
            "mass": self.DENSITY * (2.0 * b * ts + 2.0 * h * tw + tr * b * h / rib_pitch),
            "rotary_inertia": self.DENSITY * (second_moment + tr * b * h**3 / 12.0 / rib_pitch),
        }

    def reference(self, analysis: str) -> float | None:
        beam = self.beam()
        if analysis == "statics":
            return timoshenko_tip_deflection(self.TIP_FORCE, self.SPAN, beam["bending_stiffness"],
                                             beam["shear_stiffness"])
        if analysis == "modes":
            return timoshenko_cantilever_frequency(self.SPAN, **beam)
        return None


class SolidBlock(BenchmarkModel):
    """An elongated block of SolidHexa (8-node) elements, clamped on its end face x = 0.

    Length 8 along x, a unit square section; n elements across each side of the section and 8 n along the length,
    so that the elements are cubes and the graph of the mesh is three-dimensional at every size.
    statics: a transverse force on the free end face. modes: the first 10 modes. buckling: an axial compression on
    the free end face. The forces are spread over the face as a uniform traction (bilinear face weights).
    """

    name = "block"
    title = "Block, SolidHexa"
    quantities = {
        "statics": "mean end deflection against Timoshenko beam theory (Cowper's shear coefficient)",
        "modes": "first frequency against a Timoshenko beam",
        "buckling": "first factor against the column with shear (Engesser)",
    }

    LENGTH = 8.0
    SECTION = 1.0
    YOUNG = 2.1e11
    POISSON = 0.3
    DENSITY = 7800.0
    TIP_FORCE = 1.0e6
    AXIAL_FORCE = 1.0e8

    def sizes(self) -> range:
        return range(1, 151)

    def length_elements(self, n: int) -> int:
        return round(self.LENGTH / self.SECTION) * n

    def grid_points(self, n: int) -> int:
        return (self.length_elements(n) + 1) * (n + 1) ** 2

    def dofs(self, n: int) -> int:
        return 3 * (self.grid_points(n) - (n + 1) ** 2)  # translations only; the end face x = 0 is clamped

    def node(self, n: int, i: int, j: int, k: int) -> int:
        return (i * (n + 1) + k) * (n + 1) + j + 1

    def end_face(self, n: int) -> list[tuple[int, float]]:
        """(grid point, share of the face area) on the free end face."""
        last, weights = self.length_elements(n), edge_weights(n)
        return [(self.node(n, last, j, k), weights[j] * weights[k]) for k in range(n + 1) for j in range(n + 1)]

    def write_model(self, deck: DeckWriter, n: int, analysis: str) -> None:
        length_elements = self.length_elements(n)
        step = self.SECTION / n
        for i in range(length_elements + 1):
            x = self.LENGTH * i / length_elements
            for k in range(n + 1):
                for j in range(n + 1):
                    deck.record("Node", self.node(n, i, j, k), None, x, j * step, k * step)
        element = 1
        for i in range(length_elements):
            for k in range(n):
                for j in range(n):
                    # Corners 1-4 on the face z = k, counter-clockwise seen from +z, then 5-8 above them.
                    bottom = [self.node(n, i, j, k), self.node(n, i + 1, j, k), self.node(n, i + 1, j + 1, k),
                              self.node(n, i, j + 1, k)]
                    top = [node + (n + 1) for node in bottom]
                    deck.record("SolidHexa", element, 1, *bottom, *top)
                    element += 1
        # SolidProp: id, material.
        deck.record("SolidProp", 1, 1)
        deck.record("IsoMat", 1, self.YOUNG, None, self.POISSON, self.DENSITY)
        clamped = components(1, 2, 3)
        write_constraints(deck, {self.node(n, 0, j, k): clamped for k in range(n + 1) for j in range(n + 1)})
        if analysis == "statics":
            for node, share in self.end_face(n):
                deck.record("PointForceLoad", LOAD_SET, node, None, self.TIP_FORCE * share, 0.0, 0.0, 1.0)
        elif analysis == "buckling":
            for node, share in self.end_face(n):
                deck.record("PointForceLoad", LOAD_SET, node, None, self.AXIAL_FORCE * share, -1.0, 0.0, 0.0)

    def measured(self, analysis: str, result: dict, n: int) -> float | None:
        if analysis == "statics":
            face = self.end_face(n)
            rows = nodal_values(result, [node for node, _ in face])
            return float(np.dot([share for _, share in face], rows[:, 2]))
        if analysis == "modes":
            return first_frequency(result)
        return first_buckling_factor(result)

    def beam(self) -> dict[str, float]:
        area = self.SECTION**2
        second_moment = self.SECTION**4 / 12.0
        shear_modulus = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        return {
            "bending_stiffness": self.YOUNG * second_moment,
            "shear_stiffness": cowper_shear_coefficient(self.POISSON) * shear_modulus * area,
            "mass": self.DENSITY * area,
            "rotary_inertia": self.DENSITY * second_moment,
        }

    def reference(self, analysis: str) -> float | None:
        beam = self.beam()
        if analysis == "statics":
            return timoshenko_tip_deflection(self.TIP_FORCE, self.LENGTH, beam["bending_stiffness"],
                                             beam["shear_stiffness"])
        if analysis == "modes":
            return timoshenko_cantilever_frequency(self.LENGTH, **beam)
        critical = engesser_buckling_load(self.LENGTH, beam["bending_stiffness"], beam["shear_stiffness"])
        return critical / self.AXIAL_FORCE


MODELS: dict[str, BenchmarkModel] = {
    model.name: model for model in (SquarePlate(), Cylinder(), WingBox(), SolidBlock())
}


# ---------------------------------------------------------------------------------------------------------------------
# Command line


def print_ladder() -> None:
    print(f"{'model':10} {'target DOF':>10} {'size':>6} {'grid points':>12} {'DOF':>10}")
    for model in MODELS.values():
        for target in BENCHMARK_DOFS:
            n = model.size_for_dofs(target)
            print(f"{model.name:10} {target:10.0e} {n:6d} {model.grid_points(n):12d} {model.dofs(n):10d}")


def check_decks(directory: Path, target: float) -> int:
    """Write every model and analysis near a DOF count, read each deck with oo_model, and list its diagnostics."""
    import oo_model  # only for this check: writing a deck needs no package

    problems = 0
    for model in MODELS.values():
        n = model.size_for_dofs(target)
        for analysis in ANALYSES:
            path = directory / f"{model.name}-{n}-{analysis}.oos"
            model.write(path, n, analysis)
            deck = oo_model.read_model(path, strict=False)
            diagnostics = list(deck.diagnostics)
            unknown = deck.card_counts.get("(unknown cards)", 0)
            status = "ok" if not diagnostics and not unknown else "PROBLEM"
            problems += status != "ok"
            print(f"{model.name:10} n={n:<4} {analysis:9} {status}  cards={dict(deck.card_counts)}")
            for diagnostic in diagnostics:
                print(f"    {diagnostic.severity} {diagnostic.code}: {diagnostic.message} ({diagnostic.location})")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", choices=sorted(MODELS), help="the model to write")
    parser.add_argument("size", nargs="?", type=int, help="its size parameter n")
    parser.add_argument("analysis", nargs="?", choices=ANALYSES, help="the analysis of the deck")
    parser.add_argument("path", nargs="?", type=Path, help="the .oos file to write")
    parser.add_argument("--ladder", action="store_true", help="print the sizes of the benchmark's DOF counts")
    parser.add_argument("--check", type=Path, metavar="DIRECTORY",
                        help="write and read every deck near 1e4 DOF in DIRECTORY and list their diagnostics")
    parser.add_argument("--references", action="store_true", help="print the reference values")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # a console without Greek letters
    if args.ladder:
        print_ladder()
    elif args.check:
        args.check.mkdir(parents=True, exist_ok=True)
        raise SystemExit(1 if check_decks(args.check, BENCHMARK_DOFS[0]) else 0)
    elif args.references:
        for model in MODELS.values():
            for analysis in ANALYSES:
                print(f"{model.name:10} {analysis:9} {model.reference(analysis)!s:24} {model.quantities[analysis]}")
    elif args.path:
        MODELS[args.model].write(args.path, args.size, args.analysis)
        model = MODELS[args.model]
        print(f"{args.path}: {model.grid_points(args.size)} grid points, {model.dofs(args.size)} DOF")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
