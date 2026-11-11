// oos-solver: the linear finite element solver of O_o (statics, normal modes, buckling), in the browser or in Node.js.
// It reads its model from the O_o_Reader module: the reader reads the model file, the solver takes the reader's tables.
//
//   import { loadOorReader } from "oor-reader";
//   import { loadOosSolver } from "oos-solver";
//   const oor = await loadOorReader();
//   const solver = await loadOosSolver();
//   oor.writeFile("/work/plate.oos", modelText);
//   const deck = oor.readModel("/work/plate.oos");
//   const model = solver.modelFromReader(oor, deck);
//   const run = solver.run(model, "statics");
//   run.nodalTables[0];          // { kind: "displacements", subcase: 1, ids, values, ... }

import createOosSolverCore from "./oos_solver_core.js";
import { THREADED_BUILD, MAXIMUM_THREADS } from "./build-info.js";
import { OptimizationProblem } from "./optimization.js";
import { OptimizationStudy, restoreStudy } from "./study.js";
import { shapeBasisFromReader } from "./shape-basis.js";
export { OptimizationProblem, OptimizationEvaluation } from "./optimization.js";
export { OptimizationStudy, StudyEvaluation, restoreStudy, STUDY_DOCUMENT } from "./study.js";
export { createOptimizationWorkerPool, OptimizationWorkerPool } from "./study-workers.js";

// True when the JavaScript runtime runs 64-bit WebAssembly (Memory64), which the module needs.
export function supportsMemory64() {
  const module = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x03, 0x01, 0x04, 0x01]);
  try {
    return WebAssembly.validate(module);
  } catch {
    return false;
  }
}

// Memory64 shared memory must be available in the runtime and permitted by the browser's isolation policy.
export function supportsWasmThreads() {
  if (typeof WebAssembly === "undefined" || typeof SharedArrayBuffer === "undefined") return false;
  const browser = typeof window !== "undefined" || typeof WorkerGlobalScope !== "undefined";
  if (browser && globalThis.crossOriginIsolated !== true) return false;
  const module = new Uint8Array([0,97,115,109,1,0,0,0,5,4,1,7,1,1]);
  return WebAssembly.validate(module);
}

// Loads one build variant. A threaded browser module runs inside a dedicated Worker.
export async function loadOosSolver(options = {}) {
  if (!supportsMemory64())
    throw new Error("oos-solver needs a runtime supporting 64-bit WebAssembly (Memory64)");
  if (THREADED_BUILD && !supportsWasmThreads())
    throw new Error("the threaded solver needs shared Memory64 and cross-origin isolation (COOP/COEP)");
  if (THREADED_BUILD && typeof window !== "undefined" && typeof document !== "undefined")
    throw new Error("load the threaded solver inside a dedicated Worker, outside the display thread");
  const suggested = Math.min(4, globalThis.navigator?.hardwareConcurrency ?? 4, MAXIMUM_THREADS);
  const { threads = THREADED_BUILD ? suggested : 1, ...moduleOptions } = options;
  if (!Number.isInteger(threads) || threads < 1 || threads > MAXIMUM_THREADS)
    throw new RangeError(`threads must be an integer from 1 to ${MAXIMUM_THREADS} for this build`);
  const core = await createOosSolverCore(moduleOptions);
  return new OosSolver(core, threads);
}

// ---- what the solver reads from the reader (the same tables, under the same names, as the model snapshot)
//
// Every name is the reader's neutral name: card types (ShellQuad4), columns (positionFrameId) and queries
// (singlePointConstrTable). The deck keywords (CQUAD4, PID) appear only in the deck and in the messages.

// The columns of the grid point table.
const NODE_COLUMNS = ["id", "positionFrameId", "displacementFrameId", "permanentConstraints", "superelementId",
  "xyzGlobal"];
// The computed columns of an element table the solver reads: the rows of its grid points in the node table, and the
// row and card type of its property.
const ROW_INDEX_COLUMNS = ["nodesRows", "propIdRow", "propType"];
// The scalar points among the point cards.
const SCALAR_POINT_CARDS = ["ScalarPoint", "ExtraPoint"];
// The material cards whose own fields the shells and solids read (beyond materialsTable).
const MATERIAL_CARDS = ["IsoMat", "AnisoMat2D", "OrthoMat2D", "AnisoMat3D"];
// The eigenvalue method cards and the column of their ID.
const METHOD_CARDS = { LanczosEigenMethod: "id", RealEigenMethod: "id", BucklingEigenMethod: "id" };
// The named queries read as they are.
const QUERY_TABLES = [
  "materialsTable", "singlePointConstrTable", "multiPointConstrTable", "rigidElementsTable", "loadsTable",
  "pressureTable", "lineLoadsTable", "temperatureTable", "subcasesTable", "nonStructMassTable", "coordsTable",
  "laminatePliesTable", "laminateSummary",
];
// The load tables taken with each LOAD combination left unexpanded (its terms in loadCombinationsTable).
const UNEXPANDED_LOAD_TABLES = ["loadsTable", "pressureTable", "lineLoadsTable", "singlePointConstrTable",
  "inertialLoadsTable"];
const LOAD_COMBINATION_COLUMNS = ["load", "scale", "loadSet", "factor", "combinedFactor"];
// The tables the reader gives for the needs of the solver, with the arguments the solver calls them with.
const PREREQUISITE_QUERIES = {
  inertialLoadsTable: ["inertialLoadsTable", {}],
  paramsTable: ["paramsTable", {}],
  outputRequestsTable: ["outputRequestsTable", {}],
  caseSetsTable: ["caseSetsTable", {}],
  sectionsTable: ["sectionsTable", { types: ["BarShapeProp", "BeamShapeProp"] }],
  fastenerPiercings: ["fastenerPiercings", {}],
  dofSetsTable: ["dofSetsTable", { cards: ["FreeBodySupport", "FreeBodySupportSet"] }],
};
// The load cards whose field formats the deck tolerance of the load basis reads.
const LOAD_FORMAT_CARDS = ["PointForceLoad", "PointForceLoad1", "PointForceLoad2", "PointMomentLoad",
  "PointMomentLoad1", "PointMomentLoad2", "ScalarPointLoad", "PointsPressureLoad", "BeamDistributedLoad",
  "ShellPressureLoad", "FacePressureLoad", "LoadCombination"];
const LOAD_PRECISION_COLUMNS = ["id", "significantDigits"];
// The aeroelastic model of a static aeroelastic trim, taken when the model has aerodynamic panels.
const AERO_TABLES = ["aeroReferenceTable", "aeroPanelsTable", "aeroBoxesTable", "controlSurfacesTable",
  "aeroTrimCasesTable", "splinesTable", "splineNodes"];
const AERO_CARDS = { AeroBeamSpline2: "id", AeroBeamSpline5: "id", AeroBoxList: "id",
  AeroRigidTrimVariable: "id", AeroTrimLink: "id" };
const AERO_PANEL_CARDS = ["AeroLiftingPanel1", "AeroBody2", "AeroMachBoxPanel3", "AeroStripPanel4",
  "AeroPistonPanel5"];
// The element deformations: the axial deformation cards, one row per card.
const DEFORMATION_CARD = "AxialDeformationLoad";
const DEFORMATION_COLUMNS = ["id", "element", "deformation"];
// Glued contact, taken when the Bulk Data has a contact table: the node pairs (the grid points of each secondary
// surface near its primary surface, with a face of that surface; the solver projects the points on the faces itself)
// and the pairs of bodies, both with the reader's own contact rule, and the pairs each contact selection selects.
const CONTACT_TABLE_CARDS = ["ContactTable", "ContactTable1", "ContactPair", "ContactSet", "ContactSetUnion",
  "ContactSlideline"];
const CONTACT_PAIR_QUERIES = ["contactNodePairs", "contactPairsTable"];
const CONTACT_RULE = "components";

export class OosSolver {
  constructor(core, threads = 1) {
    this.core = core;
    this.defaultThreads = threads;
  }

  threading() { return { ...this.core.threadingInfo(), defaultThreads: this.defaultThreads }; }

  // The name and version of the solver.
  version() {
    return this.core.version();
  }

  // Every element card type the solver computes with its formulations, the default first: [{ card, formulations }],
  // `card` the neutral card type ("ShellQuad4").
  elementFormulations() {
    return this.core.elementFormulations();
  }

  capabilities(model = null) {
    return model ? this.core.modelCapabilities(model) : this.core.optimizationCatalogue();
  }

  createStudy(reader, deck, definition, settings = {}, hooks = {}) {
    return new OptimizationStudy(this, reader, deck, definition, { threads: this.defaultThreads, ...settings }, hooks);
  }

  restoreStudy(reader, deck, hooks = {}) { return restoreStudy(this, reader, deck, hooks); }
  shapeBasisFromReader(reader, auxiliary, options) { return shapeBasisFromReader(this, reader, auxiliary, options); }

  // Preparation validates declarations and analysis support without solving or evaluating callbacks. Restrictions
  // that require the actual state remain explicit in the catalogue; a successful preparation is not a gradient check.
  checkOptimization(model, declarations = {}, settings = {}) {
    const catalogue = this.capabilities(model);
    const kinds = Object.entries(settings.analyses ?? { statics: [] }).filter(([, cases]) => cases != null).map(([kind]) => kind);
    const analyses = kinds.map(analysis => ({ analysis, ...this.checkSupport(model, analysis) }));
    let problem, metadata = null, error = null;
    try { problem = this.createOptimization(model, declarations, settings); metadata = problem.metadata(); }
    catch (failure) { error = failure.message; }
    finally { problem?.delete(); }
    return { stage: "preparation", canPrepare: !error && analyses.every(analysis => analysis.isComplete), error, metadata,
      analyses, capabilities: catalogue.analyses.filter(entry => kinds.includes(entry.analysis)) };
  }

  // The model read by O_o_Reader (`oor` the loaded reader, `deck` its Deck), ready for the analyses.
  // `options.elementFormulations` chooses the formulation of the elements of some card types
  // ({ ShellQuad4: "ClassicFormulation" }), before the model's ShellQuad4Formulation parameter. The keys are the neutral
  // card types that elementFormulations() lists, exactly: an unknown name throws an Error. A
  // formulation the card does not have is refused by checkSupport and run.
  // Throws an Error naming the table and the column when a table of the reader is not what the solver reads.
  modelFromReader(oor, deck, options = {}) {
    const reader = oor.core;
    const input = new this.core.ModelInput();
    const putTable = (name, bytes, added = null) => input.setTable(name, bytes, added);
    try {
      for (const [card, formulation] of Object.entries(options.elementFormulations ?? {})) {
        input.setElementFormulation(card, formulation);
      }
      const versions = reader.modelVersions(deck);
      input.setVersions(versions.structure, versions.content);
      input.setSourceCards(Object.entries(deck.cardCounts()).map(([card, count]) => ({ card, count })));
      const catalogue = new Map(oor.queryCatalogue().map((query) => [query.name, query]));
      const hasQuery = (name) => catalogue.has(name);
      const takesParameter = (name, parameter) =>
        hasQuery(name) && catalogue.get(name).parameters.some((entry) => entry.name === parameter);
      const query = (name, args = {}) => oor.runQuery(name, { deck, args });
      const packageOf = (table) => {
        try {
          return table.toPackage();
        } finally {
          table.delete();
        }
      };
      const cardTypes = (category) => reader.cardTypesOf(deck, category);

      // The grid points, their displacement systems, and the scalar points.
      putTable("nodes", packageOf(reader.cardTable(deck, "nodes", NODE_COLUMNS)));
      input.setDisplacementRotations(reader.displacementRotations(deck));
      const points = cardTypes("points");
      const scalarPoints = points.filter((card) => SCALAR_POINT_CARDS.includes(card));
      if (scalarPoints.length > 0) {
        if (scalarPoints.length !== points.length && scalarPoints.length !== 1) {
          throw new Error("the model mixes SPOINT or EPOINT with other point cards: give the scalar points one type");
        }
        const source = scalarPoints.length === points.length ? "points" : scalarPoints[0];
        putTable("scalarPoints", packageOf(reader.cardTable(deck, source, ["id"])));
      }

      // The cards the solver reads field by field.
      for (const card of cardTypes("elements")) {
        const columns = reader.cardTableColumns(card);
        const wanted = [...columns.stored, ...ROW_INDEX_COLUMNS.filter((name) => columns.computed.includes(name))];
        input.addElementTable(card, packageOf(reader.cardTable(deck, card, wanted)));
      }
      // Every property card is keyed by its "id" column; the few that have no ID of their own (a list of properties,
      // such as CompPropIntegration) are not property tables to the solver.
      for (const card of cardTypes("properties")) {
        if (!reader.cardTableColumns(card).stored.includes("id")) continue;
        input.addPropertyTable(card, packageOf(reader.cardTable(deck, card, null)));
      }
      const methods = cardTypes("methods");
      for (const [card, idColumn] of Object.entries(METHOD_CARDS)) {
        if (methods.includes(card)) input.addMethodTable(card, idColumn, packageOf(reader.cardTable(deck, card, null)));
      }
      const materials = cardTypes("materials");
      for (const card of MATERIAL_CARDS) {
        if (materials.includes(card)) input.addMaterialTable(card, packageOf(reader.cardTable(deck, card, null)));
      }

      // The query tables. LOAD combinations are left unexpanded when the reader gives their terms, so that a load set
      // shared by many subcases is read once.
      const unexpanded = hasQuery("loadCombinationsTable");
      for (const name of QUERY_TABLES) {
        const args = unexpanded && UNEXPANDED_LOAD_TABLES.includes(name) ? { expandCombinations: false } : {};
        if (name === "pressureTable" && unexpanded && takesParameter("pressureTable", "detail")) {
          putTable("pressureFields", packageOf(query("pressureTable", { ...args, detail: "fields" })));
          continue;
        }
        putTable(name, packageOf(query(name, args)));
      }
      if (unexpanded) {
        putTable("loadCombinationsTable",
          packageOf(query("loadCombinationsTable", { columns: LOAD_COMBINATION_COLUMNS })));
      }
      for (const [name, [queryName, args]] of Object.entries(PREREQUISITE_QUERIES)) {
        if (!hasQuery(queryName)) continue;
        const queryArgs = unexpanded && UNEXPANDED_LOAD_TABLES.includes(name)
          ? { ...args, expandCombinations: false } : args;
        putTable(name, packageOf(query(queryName, queryArgs)));
      }

      // The field formats of the load cards (the deck tolerance of the load basis).
      const loads = cardTypes("loads");
      for (const card of LOAD_FORMAT_CARDS) {
        if (!loads.includes(card)) continue;
        const columns = card === "LoadCombination" ? [...LOAD_PRECISION_COLUMNS, "termLoad"] : LOAD_PRECISION_COLUMNS;
        const available = reader.cardTableColumns(card);
        const all = [...available.stored, ...available.computed];
        if (columns.every((name) => all.includes(name))) {
          input.addLoadCardTable(card, packageOf(reader.cardTable(deck, card, columns)));
        }
      }
      if (loads.includes(DEFORMATION_CARD)) {
        putTable("deformations", packageOf(reader.cardTable(deck, DEFORMATION_CARD, DEFORMATION_COLUMNS)));
      }

      // The glued contact, when the Bulk Data has a contact table.
      const cardCounts = deck.cardCounts();
      const hasContactTable = CONTACT_TABLE_CARDS.some((card) => (cardCounts[card] ?? 0) > 0);
      if (hasContactTable && CONTACT_PAIR_QUERIES.every(hasQuery)) {
        for (const name of CONTACT_PAIR_QUERIES) {
          const args = takesParameter(name, "rule") ? { rule: CONTACT_RULE } : {};
          putTable(name, packageOf(query(name, args)));
        }
        if (hasQuery("contactTablesTable")) {
          putTable("contactTablesTable", packageOf(query("contactTablesTable")));
        }
      }

      // The aeroelastic model, when the model has aerodynamic panels.
      const hasAeroPanels = AERO_PANEL_CARDS.some((card) => (cardCounts[card] ?? 0) > 0);
      if (hasAeroPanels) {
        for (const name of AERO_TABLES) {
          if (hasQuery(name)) putTable(name, packageOf(query(name)));
        }
        for (const [card, idColumn] of Object.entries(AERO_CARDS)) {
          if ((cardCounts[card] ?? 0) > 0)
            input.addAeroCardTable(card, idColumn, packageOf(reader.cardTable(deck, card, null)));
        }
      }
      return input.build();
    } finally {
      input.delete();
    }
  }

  createOptimization(model, declarations = {}, settings = {}) {
    return new OptimizationProblem(this.core.createOptimization(model, declarations, { threads: this.defaultThreads, ...settings }));
  }

  optimizationFromReader(oor, deck, declarations = {}, settings = {}) {
    const design = oor.designVariables(deck);
    if (design.problems.length || design.notApplied.length)
      throw new Error("the reader cannot export this design: " +
        [...design.problems, ...design.notApplied.map(p => `${p.card} ${p.id}: ${p.reason}`)].join("; "));
    const model = this.modelFromReader(oor, deck);
    try {
      return this.createOptimization(model, {
        ...declarations, tables: { ...oor.optimizationTables(deck), ...declarations.tables },
      }, settings);
    } finally { model.delete(); }
  }

  // The cards and fields of the model the analysis cannot take (refusals) or will not use (ignored), found in one pass.
  // analysis: "statics", "modes" or "buckling".
  checkSupport(model, analysis = "statics") {
    return model.checkSupport(analysis);
  }

  // Runs an analysis ("statics", "modes" or "buckling") on a model.
  // settings: { subcases?: number[], memoryBytes?: number, threads?: number }.
  // Gives { nodalTables, elementTables, gridPointForces, eigenvalues, diagnostics }.
  run(model, analysis = "statics", settings = {}) {
    return this.core.runAnalysis(model, analysis, { threads: this.defaultThreads, ...settings });
  }
}
