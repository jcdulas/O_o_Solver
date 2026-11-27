"""Native structural analyses and design evaluations using oo_model decks."""
import functools
import warnings

from oo_model import card_columns

from ._native import AeroModalBasis, AeroStaticPreload, AeroTransfer, DesignPoint, Model, OptimizationProblem, _Builder



class SolverWarning(UserWarning):
    """A warning of an analysis: the run went through, and something in it deserves a look."""


class SolverInformation(SolverWarning):
    """An information of an analysis: how the solver read the model where another program may read it otherwise."""


def _announce(result):
    """Shows the warnings and information of a run through the ``warnings`` module, and gives the result back.

    They stay in ``result["diagnostics"]``. Python shows a given message once per calling line by default; silence
    them with ``warnings.simplefilter("ignore", oo_solver.SolverWarning)`` (``SolverInformation`` for the information
    alone), or turn them into errors with ``"error"``.
    """
    categories = {"warning": SolverWarning, "information": SolverInformation}
    for diagnostic in result.get("diagnostics", ()):
        category = categories.get(diagnostic["severity"])
        if category is not None:
            # stacklevel 3: the line of the caller of Model.run.
            warnings.warn(f"[{diagnostic['code']}] {diagnostic['message']}", category, stacklevel=3)
    return result


_native_run = Model.run


@functools.wraps(_native_run)
def _run(self, *args, **kwargs):
    return _announce(_native_run(self, *args, **kwargs))


Model.run = _run

_DESIGN_TABLES = (
    "designVariablesTable", "designLinksTable", "designRelationsTable", "designConstantsTable",
    "shapeVariablesTable", "designEquationSymbols", "designEquationInstructions", "designResponsesTable",
    "designResponseArgumentsTable", "designConstraintsTable", "designConstraintSetsTable",
    "designSubcasesTable", "designScreeningTable",
)


def model_from_reader(deck, *, formulation=None):
    """Build an immutable snapshot. Table buffers transfer through the Arrow C Data Interface."""
    builder = _Builder()
    if formulation is not None:
        builder.formulation(formulation)
    counts = deck.card_counts
    builder.cards(counts)
    builder.versions(*deck.versions[:2])
    builder.table("nodes", deck.card_table("nodes", ["id", "positionFrameId", "displacementFrameId",
                                                    "permanentConstraints", "superelementId", "xyzGlobal"]))
    if getattr(deck, "has_displacement_frames", True):
        builder.rotations(deck.displacement_rotations())
    points = deck.card_types("points")
    scalar = [p for p in points if p in ("ScalarPoint", "ExtraPoint")]
    if scalar:
        if len(points) != len(scalar) and len(scalar) != 1:
            raise ValueError(f"Ambiguous scalar point types: {points}")
        builder.table("scalarPoints", deck.card_table("points" if len(points) == len(scalar) else scalar[0], ["id"]))
    for card, role, columns in (
        ("ShellNormal", "shellNormals", ["nodeId", "frameId", "vector"]),
        ("SurfaceEdge", "surfaceEdges", ["id", "node1Id", "node2Id", "state", "elementId"]),
        ("SurfaceFacet", "surfaceFacets", ["elementId"]),
        ("SurfaceLine", "surfaceLines", ["id", "node1Id", "throughNodeId", "node2Id"]),
        ("Bar2Stations", "barStations", ["id", "stationScale", "listedStations", "stationCount", "station1", "stationSpacing"]),
        ("AxialDeformationLoad", "deformations", ["id", "element", "deformation"]),
    ):
        if counts.get(card):
            builder.table(role, deck.card_table(card, columns))
    if counts.get("ModelParams"):
        builder.offset(deck.offset_definition)
    for card in deck.card_types("elements"):
        stored, computed = card_columns(card)
        columns = stored + [c for c in ("nodesRows", "propIdRow", "propType") if c in computed]
        builder.card("elements", card, deck.card_table(card, columns))
    for card in deck.card_types("properties"):
        if "id" in card_columns(card)[0]:
            builder.card("properties", card, deck.card_table(card))
    for group, cards in (
        ("materials", ("IsoMat", "OrthoMat2D", "AnisoMat2D", "AnisoMat3D")),
        ("methods", ("LanczosEigenMethod", "RealEigenMethod", "BucklingEigenMethod")),
    ):
        for card in cards:
            if counts.get(card):
                builder.card(group, card, deck.card_table(card))
    for card in ("PointForceLoad", "PointForceLoad1", "PointForceLoad2", "PointMomentLoad", "PointMomentLoad1",
                 "PointMomentLoad2", "ScalarPointLoad", "PointsPressureLoad", "BeamDistributedLoad",
                 "ShellPressureLoad", "FacePressureLoad", "LoadCombination"):
        if counts.get(card):
            cols = ["id", "significantDigits"]
            if card == "LoadCombination":
                cols.append("termLoad")
            stored, computed = card_columns(card)
            if set(cols).issubset(stored + computed):
                builder.card("loads", card, deck.card_table(card, cols))
    for name in ("materialsTable", "multiPointConstrTable", "rigidElementsTable", "temperatureTable",
                 "subcasesTable", "nonStructMassTable", "coordsTable", "laminatePliesTable", "laminateSummary",
                 "paramsTable", "outputRequestsTable", "caseSetsTable", "fastenerPiercings"):
        builder.table(name, deck.query(name))
    for name in ("loadsTable", "lineLoadsTable", "singlePointConstrTable", "inertialLoadsTable"):
        builder.table(name, deck.query(name, expandCombinations=False))
    builder.table("pressureFields", deck.query("pressureTable", expandCombinations=False, detail="fields"))
    builder.table("loadCombinationsTable", deck.query("loadCombinationsTable",
                  columns=["load", "scale", "loadSet", "factor", "combinedFactor"]))
    builder.table("sectionsTable", deck.query("sectionsTable", types=["BarShapeProp", "BeamShapeProp"]))
    builder.table("dofSetsTable", deck.query("dofSetsTable", cards=["FreeBodySupport", "FreeBodySupportSet"]))
    if any(counts.get(c) for c in ("ContactTable", "ContactTable1", "ContactPair", "ContactSet",
                                   "ContactSetUnion", "ContactSlideline")):
        builder.table("contactNodePairs", deck.query("contactNodePairs", rule="components"))
        builder.table("contactPairsTable", deck.query("contactPairsTable", rule="components"))
        builder.table("contactTablesTable", deck.query("contactTablesTable"))
    if any(counts.get(c) for c in ("AeroLiftingPanel1", "AeroBody2", "AeroMachBoxPanel3", "AeroStripPanel4", "AeroPistonPanel5")):
        for name in ("aeroReferenceTable", "aeroPanelsTable", "aeroBoxesTable", "controlSurfacesTable",
                     "aeroTrimCasesTable", "splinesTable", "splineNodes"):
            builder.table(name, deck.query(name))
        for card in ("AeroBeamSpline2", "AeroBeamSpline5", "AeroBoxList", "AeroRigidTrimVariable", "AeroTrimLink"):
            if counts.get(card):
                builder.card("aero", card, deck.card_table(card))
    return builder.build()


def prepare_optimization(deck, *, analysis="statics", formulation=None, threads=None):
    """Prepare a C++ design problem. Select an analysis name or a map of names to subcase IDs.

    ``threads=None`` runs the evaluations on every core of the machine; the results do not depend on it."""
    deck.validate_design()
    model = model_from_reader(deck, formulation=formulation)
    return model._optimization({name: deck.query(name) for name in _DESIGN_TABLES}, analysis, threads)


optimization_from_reader = prepare_optimization


__all__ = ["AeroModalBasis", "AeroStaticPreload", "AeroTransfer", "DesignPoint", "Model", "OptimizationProblem",
           "SolverInformation", "SolverWarning", "model_from_reader", "optimization_from_reader",
           "prepare_optimization"]
