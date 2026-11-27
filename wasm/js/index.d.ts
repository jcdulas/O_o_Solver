import type { OptimizationProblem, OptimizationDeclarations, OptimizationSettings } from "./optimization.js";
export * from "./optimization.js";
export * from "./study.js";
export * from "./study-workers.js";

// Types of the oos-solver package (index.js).
//
// Every name the interface gives or takes is an exact neutral name of O_o_Model: card and result element types use
// UpperCamelCase ("ShellQuad4"); table kinds ("singlePointConstrForces") and columns ("bendingMomentA1") use lowerCamelCase.
// Messages (Diagnostic.message, Diagnostic.card, Diagnostic.field and errors) use the same neutral names.

export declare function supportsMemory64(): boolean;
export declare function supportsWasmThreads(): boolean;
// diagnostics: where the warnings and information of a run are shown beside the `diagnostics` of its result:
// "console" (the default: console.warn and console.info, each message once) or "silent".
export type DiagnosticOutput = "console" | "silent";
export declare function loadOosSolver(
  options?: Record<string, unknown> & { threads?: number; diagnostics?: DiagnosticOutput }): Promise<OosSolver>;

export type Analysis = "statics" | "modes" | "buckling" | "aeroelastic";

// An edge by its two grid point IDs, or the side of one shell element of it (which matters at an edge of three shells
// or more).
export type EdgeSide = [number, number] | [number, number, number];

// The geometry declarations of the mixed hybrid formulation: where the surface of the shells and the skin of the
// solids are smooth, where they fold, and which mesh line goes on into which (grid point and element IDs). Where
// nothing is declared, the model's ShellNormalAngle decides the state of an edge.
export interface FormulationGeometry {
  // The surface is smooth across the edge.
  smoothEdges?: EdgeSide[];
  // The surface folds along the edge, which keeps its own line.
  sharpEdges?: EdgeSide[];
  // The surface of the shells folds along the edge, which is straight.
  rawEdges?: [number, number][];
  // Shell elements whose edges are all raw: facets.
  rawElements?: number[];
  // [g1, g, g2]: at grid point g, the mesh line of edge g1 - g goes on into edge g - g2.
  lineContinuations?: [number, number, number][];
  // [g1, g]: the mesh line of edge g1 - g ends at grid point g.
  lineEnds?: [number, number][];
}

// The edges of the surfaces that MixedHybridFormulation read from the mesh, column by column, one entry per edge.
export interface SurfaceEdges {
  // The IDs of the two grid points of the edge, the lower first.
  grid1: Int32Array;
  grid2: Int32Array;
  surface: ("Shells" | "SolidSkin")[];
  // At a junction of three shells or more, "Sharp" when one side at least ends there.
  state: ("Smooth" | "Sharp" | "Raw")[];
  // Where the state comes from: a declaration, a raw element, the model's ShellNormalAngle, a shell of a junction that
  // no other faces, or an edge of one element.
  origin: ("Declared" | "RawElement" | "Angle" | "Junction" | "Outline")[];
  // Degrees: the angle measured between the normals of the elements of the edge at its grid points; the angle left
  // between the normals of the surface read on both sides; how far the surface read turns from the plane of the
  // corners of an element (the last two on the shells only).
  angle: Float64Array;
  kink: Float64Array;
  rounding: Float64Array;
}

// The options of modelFromReader. Another key throws a TypeError that lists these (a misspelt option would
// otherwise leave its default in force), as does a key of formulationGeometry that is not one of its lists.
export interface ModelOptions {
  // The formulation of the elements of some card types ({ ShellQuad4: "ClassicFormulation" }).
  elementFormulations?: Record<string, string>;
  // The formulation of the whole model ("ClassicFormulation", "MixedHybridFormulation").
  elementFormulation?: string;
  // Mandatory for a model with shells or solids in MixedHybridFormulation.
  formulationGeometry?: FormulationGeometry;
  // Also take the component tables of the reader, which a condensed static analysis reads. A boolean: the units to
  // condense are given to run (settings.condensation).
  condensation?: boolean;
}

// The units a condensed analysis condenses: the components of the reader. "recommended": every unit but those
// whose condensed stiffness would cost more than their interior; an array: these unit IDs.
export type CondensedUnits = "all" | "recommended" | "none" | number[];
export type CondensationSettings = CondensedUnits | {
  units: CondensedUnits;
  fullUnits?: number[];     // kept full whatever `units` says
  retainedNodes?: number[]; // grid points kept on the boundary of their unit
  modeCount?: number;       // modes and buckling: fixed-interface modes per unit; 0: those below a cut-off
  memoryMode?: "auto" | "speed" | "low_memory"; // statics: what the analysis keeps in memory (same results)
  cache?: CondensationCache; // units found in it are not condensed again
  cacheModelTag?: string;    // the name of the model in the entries stored
};

// An external unit of a static analysis given by its matrices: the zone whose elements it replaces, and K* (and M*) of
// the zone on its contour, one row and column per label [grid, component], column by column.
export interface ExternalUnitSettings {
  zone: { id: number; name?: string; elements: number[]; boundary: { grid: number; components?: string }[] };
  labels: [number, number][];
  stiffness: number[] | Float64Array;
  mass?: number[] | Float64Array;
}

// A condensation cache in memory (OosSolver.condensationCache).
export interface CondensationCache {
  statistics(): { memoryHits: number; diskHits: number; misses: number; stores: number; refusedStores: number;
                  evictions: number; missCauses: Record<string, number> };
  delete(): void;
}

// What the condensed path did.
export interface CondensationSummary {
  condensedUnits: number[];
  fullUnits: number[];
  interfaceDofs: number; // the DOFs of the interface problem, as the plan estimates them
  note: string;          // why the plan differs from the request (a model of one unit and no connection), or ""
  // Statics: whether a constraint set recovered its units unit by unit (each unit's recovery made once for every case;
  // with memoryMode "low_memory", envelopes and keepsCaseTables false).
  unitByUnit: boolean;
}

// An envelope asked of a static analysis: a table kind ("displacements", "singlePointConstrForces",
// "multiPointConstrForces", "appliedLoads", "stress", "strain", "forces", "strainEnergy"), a quantity (a component of
// its tables, or "magnitude" for a nodal one) and the extreme kept.
export interface EnvelopeSpec {
  kind: string;
  quantity: string;
  mode: "max" | "min" | "absMax";
}

// The envelope of one table: per row the extreme over the load cases and the subcase that gives it.
export interface EnvelopeTable {
  kind: string;
  elementType: string; // "" for a nodal table
  quantity: string;
  mode: string;
  ids: Int32Array;
  nodeIds: Int32Array;
  layers: Int32Array;
  positions: Float64Array; // empty unless the tables have stations
  values: Float64Array;
  cases: Int32Array;
}

export interface Diagnostic {
  severity: "error" | "warning" | "info";
  code: string;
  message: string;
  card: string;  // the neutral card type, for example "ShellQuad4"
  id: number;
  field: string; // the neutral field name, for example "offsetZ"
  text: string; // one line: severity, code, card and ID, field, message
}

export interface SupportCheck {
  supported: string[];
  refusals: Diagnostic[];
  ignored: Diagnostic[];
  isComplete: boolean;
}

// One nodal table: one step, one row per point, six components in the displacement system of the grid point.
export interface NodalTable {
  // "displacements", "singlePointConstrForces", "multiPointConstrForces", "appliedLoads", "eigenvectors"
  kind: string;
  subcase: number;
  analysis: "static" | "modes" | "buckling" | "aeroelastic";
  step: number;       // the mode number of a modal or buckling table, 0 for statics
  eigenvalue: number; // the eigenvalue of the mode (omega squared, or the load factor), 0 for statics
  components: string[]; // ["t1", "t2", "t3", "r1", "r2", "r3"]
  ids: Int32Array;
  pointTypes: Int8Array; // 1 a grid point, 2 a scalar point
  values: Float64Array;  // rows x 6, row-major
}

// One element table: one element type, one step; values row-major (rows x components).
export interface ElementTable {
  kind: string; // "stress", "strain", "forces", "strainEnergy", "failureIndices", "strengthRatios"
  // The result element type of O_o_Model: the exact neutral card type ("ShellQuad4",
  // "ShellTria3"), or a layout of its own ("ShellQuad4Composite": the plies of a laminate).
  elementType: string;
  subcase: number;
  analysis: "static" | "modes" | "buckling" | "aeroelastic";
  step: number;
  eigenvalue: number;
  components: string[];
  ids: Int32Array;
  nodeIds: Int32Array; // the grid point of the row, 0 for the centre
  layers: Int32Array;
  positions?: Float64Array;
  values: Float64Array;
}

// The grid point forces of a subcase: per grid point, the applied load, the constraint forces, the force of each
// element connected and their total (zero to round-off).
export interface GridPointForceTable {
  subcase: number;
  analysis: string;
  // The row names: "appliedLoad", "singlePointConstrForce", "multiPointConstrForce", "total", then the result element
  // types of the elements ("ShellQuad4"), "condensedUnit" (a condensed component on its boundary) and
  // "elementFormulation" (MixedHybridFormulation: the forces of the formulation's own relations at the grid point).
  // In MixedHybridFormulation the forces on the unknowns of the edges, faces and interiors are carried to the grid
  // points of their entities with their resultant: the rows of a grid point sum to zero as in ClassicFormulation.
  names: string[];
  nodeIds: Int32Array;
  elementIds: Int32Array;
  nameIndices: Int32Array; // into names
  values: Float64Array;    // rows x 6
}

export interface EigenvalueTable {
  subcase: number;
  analysis: "modes" | "buckling";
  rows: {
    mode: number; extractionOrder: number; eigenvalue: number; radians: number; cycles: number;
    generalizedMass: number; generalizedStiffness: number;
  }[];
}

export interface AnalysisResult {
  selectedQuarticShells?: number[]; // IDs selected by MixedHybridDegree Auto in full statics
  nodalTables: NodalTable[];
  elementTables: ElementTable[];
  gridPointForces: GridPointForceTable[];
  eigenvalues: EigenvalueTable[];
  envelopes: EnvelopeTable[]; // statics: those settings.envelopes asks for
  diagnostics: Diagnostic[];
  aeroelastic: AeroelasticResult | null;
  // A static analysis asked to condense: what its plan decided; null otherwise.
  condensation: CondensationSummary | null;
  report: RunReport;
}

// The eigenvalue solver of normal modes and buckling on the full path:
// - "lanczos" (the default): block shift-invert Lanczos on a factor in memory, the roots confirmed by Sturm counts;
// - "substructuring": multilevel substructuring on the elimination tree. Approximate roots (the lowest ones best), no
//   factor of the whole model: far less memory and time on a large model;
// - "outOfCore": the Lanczos solver with its factor in scratch files (in a browser the module's file system is
//   memory: useful where a real folder is mounted).
// "substructuring" and "outOfCore" assemble no matrix of the whole model: the constraint forces, the modal effective
// mass and the grid point weight are not produced (an information diagnostic says so).
export type EigenSolverName = "lanczos" | "substructuring" | "outOfCore";
export interface EigenSolverSettings {
  solver?: EigenSolverName;
  scratchFolder?: string; // of the scratch files; default: a folder of the temporary one ("outOfCore"), memory
  cutoffFactor?: number;  // "substructuring": more modes kept in each subtree above 1; 0: the default
}

// The transverse shear of one laminate property (CompProp, CompPropGlobal) that a shell of the model uses, along the
// material axes x and y of the element, as the model's convention forms it (Parameter;LaminateTransverseShear:
// Equilibrium, the default, or FiveSixths).
export interface LaminateShear {
  card: "CompProp" | "CompPropGlobal";
  propId: number;
  stiffness: [number, number]; // H along x and y, force per length per unit shear strain; 0: rigid along that axis
  // H / sum(G_k t_k): the shear correction factor the stiffness amounts to; 5/6 for one material and under
  // FiveSixths; 0 where it is not defined (the laminate, or one of its plies, rigid along the axis)
  factor: [number, number];
}

// What a run cost. The stages are those the analysis measures, in the order it runs them (statics: setup, assembly,
// loadBasis, factorisation, solve, recovery; modes: setup, assembly, eigenvalues, recovery; buckling: preload,
// assembly, eigenvalues, recovery; aeroelastic: setup, assembly, factorisation, flexibility, aerodynamics, trim).
export interface RunReport {
  threads: number;         // the threads of the run, the calling thread included
  kernel: string;          // the dense kernel family: "reference", "avx2", "avx512" or "wasm128"
  dofs: number;            // the degrees of freedom of the model
  solvedDofs: number;      // the equations of the largest system factorised
  factorisations: number;  // the sparse factorisations made (0 when the analysis does not count them)
  eigenSolver: string;     // modes and buckling: "lanczos", "substructuring", "outOfCore" or "condensed"; "" otherwise
  seconds: number;         // the wall time of the run
  stages: Record<string, number>; // seconds per stage
  peakBudgetBytes: number; // the largest memory the solver's budget held at once (not the process memory)
  laminateShear: LaminateShear[]; // the laminates of the shells, by card then by ID (empty without one)
}

export interface StabilityDerivative {
  label: string;
  // Six coefficients in order: CX, CY, CZ, CMX, CMY, CMZ.
  rigidUnsplined: Float64Array; rigidSplined: Float64Array;
  elasticRestrained: Float64Array; elasticUnrestrained: Float64Array;
  inertialRestrained: Float64Array; inertialUnrestrained: Float64Array;
}
export interface TrimSubcaseResult {
  subcase: number; trimSet: number; mach: number; dynamicPressure: number; hasUnrestrained: boolean;
  variables: { id: number; label: string; kind: "rigidBody" | "controlSurface" | "user" | "undefined";
    state: "fixed" | "free" | "linked" | "scheduled"; value: number }[];
  derivatives: StabilityDerivative[]; reference: StabilityDerivative;
  hingeMomentDerivatives: { surface: string; atReference: Float64Array; perVariable: Float64Array }[];
  pressures: Float64Array; boxForces: Float64Array; boxMoments: Float64Array;
  hingeMoments: Float64Array; supportAccelerations: Float64Array;
  monitorLoads: { name: string; loads: Float64Array }[];
  monitor3Loads: { name: string; loads: Float64Array }[];
}
export interface AeroelasticResult {
  boxIds: Int32Array; surfaces: string[]; referenceChord: number; referenceSpan: number; referenceArea: number;
  symmetryXz: number; symmetryXy: number; configuration: string; subcases: TrimSubcaseResult[];
  basicToReference: Float64Array; referenceOffset: Float64Array;
  surfaceDefinitions: { id: number; label: string; referenceChord: number; referenceArea: number;
    positionLower: number; positionUpper: number; hingeMomentLower: number | null; hingeMomentUpper: number | null }[];
  monitorPoints: { name: string; label: string; positionFrameId: number; displacementFrameId: number;
    point: Float64Array; component: string; pointClass: string }[];
  monitorPoints3: { name: string; label: string; positionFrameId: number; displacementFrameId: number; point: Float64Array }[];
}

export interface BoxMotion { plunge: Float64Array; slope: Float64Array }
export interface AeroStaticPreload { readonly __aeroStaticPreload: unique symbol; delete(): void }
export interface AeroModalBasis {
  generalizedLoads(boxForces: Float64Array | readonly number[]): Float64Array;
  delete(): void;
}
export interface AeroTransfer {
  solveStatic(subcase: number, boxForces: Float64Array | readonly number[]): AnalysisResult & { boxMotions: BoxMotion[]; preload: AeroStaticPreload };
  solveModes(subcase: number): AnalysisResult & { boxMotions: BoxMotion[]; modalBasis: AeroModalBasis };
  solveBuckling(subcase: number, preload: AeroStaticPreload): AnalysisResult & { boxMotions: BoxMotion[] };
  readonly nodeIds: Int32Array;
  readonly boxIds: Int32Array;
  boxMotion(nodalMotion: Float64Array | readonly number[]): { plunge: Float64Array; slope: Float64Array };
  structuralLoads(boxForces: Float64Array | readonly number[]): Float64Array;
  delete(): void;
}

export interface Model {
  aeroTransfer(): AeroTransfer;
  tableNames(): string[];
  checkSupport(analysis: Analysis): SupportCheck;
  // What MixedHybridFormulation read at every edge of the surface of the shells and of the skin of the solids; every
  // array is empty for a model in another formulation.
  surfaceEdges(): SurfaceEdges;
  delete(): void;
}

export interface OptimizationCapabilities {
  version: number;
  analyses: { analysis: Analysis; solution: 101 | 103 | 105 | 144; sizingFirstOrder: boolean;
    shapeFirstOrder: boolean; secondOrder: boolean; restrictions: string[] }[];
  responses: string[];
  fields: { targetKind: "property" | "material" | "element" | "trim"; targetCard: string; field: string;
    type: string; requiresItem: boolean }[];
}
export interface OptimizationPreparation {
  stage: "preparation"; canPrepare: boolean; error: string | null;
  metadata: import("./optimization.js").OptimizationMetadata | null;
  analyses: (SupportCheck & { analysis: Analysis })[];
  capabilities: OptimizationCapabilities["analyses"];
}

export declare class OosSolver {
  // Where run and runSuperelements show the warnings and information of a run; may be set at any time.
  diagnosticOutput: DiagnosticOutput;
  // Shows the warnings and information of `result.diagnostics` as run does, and gives `result` back.
  announceDiagnostics<T extends { diagnostics?: Diagnostic[] }>(result: T): T;
  createStudy(reader: import("oo-model").OoModel, deck: import("oo-model").Deck,
    definition: import("./study.js").StudyDefinition, settings?: import("./study.js").StudySettings,
    hooks?: import("./study.js").StudyHooks): import("./study.js").OptimizationStudy;
  restoreStudy(reader: import("oo-model").OoModel, deck: import("oo-model").Deck,
    hooks?: import("./study.js").StudyHooks): ReturnType<typeof import("./study.js").restoreStudy>;
  shapeBasisFromReader(reader: import("oo-model").OoModel, auxiliary: import("oo-model").Deck,
    options: { variableId: number; cases?: { subcase: number; coefficient?: number }[]; gridIds?: number[] }):
    { variableId: number; gridIds: Int32Array; directions: Float64Array };
  version(): string;
  threading(): { enabled: boolean; maximumThreads: number; activeWorkers: number; defaultThreads: number };
  // `oor` is a loaded OoModel (package oo-model), `deck` one of its decks.
  // Every element card type the solver computes ("ShellQuad4") with its formulations, the default first.
  elementFormulations(): { card: string; formulations: string[] }[];
  capabilities(model?: Model | null): OptimizationCapabilities;
  checkOptimization(model: Model, declarations?: OptimizationDeclarations,
    settings?: OptimizationSettings): OptimizationPreparation;
  // `options.elementFormulations` chooses the formulation of the elements of some card types
  // ({ ShellQuad4: "ClassicFormulation" }), before the model's ShellQuad4Formulation parameter. The keys are the neutral card types
  // of elementFormulations(), exactly: an unknown card type throws an Error.
  // `options.elementFormulation` chooses the formulation of the whole model ("MixedHybridFormulation"), and
  // `options.formulationGeometry` gives its geometry declarations (grid point and element IDs).
  // The model's ModelParams OffsetDefinition is read (where the mass of an offset element is).
  modelFromReader(oor: unknown, deck: unknown, options?: ModelOptions): Model;
  createOptimization(model: Model, declarations?: OptimizationDeclarations, settings?: OptimizationSettings): OptimizationProblem;
  // `modelOptions`: those of modelFromReader (a model in MixedHybridFormulation: elementFormulation and
  // formulationGeometry).
  prepareOptimization(oor: unknown, deck: unknown, declarations?: OptimizationDeclarations,
    settings?: OptimizationSettings, modelOptions?: ModelOptions): OptimizationProblem;
  optimizationFromReader(oor: unknown, deck: unknown, declarations?: OptimizationDeclarations,
    settings?: OptimizationSettings, modelOptions?: ModelOptions): OptimizationProblem;

  // What the solver refuses of the model for the analysis, without running it: the cards, fields and selections it
  // does not take, then what the element formulation of the model refuses (a geometry declaration that names no
  // edge, an element it cannot compute).
  checkSupport(model: Model, analysis?: Analysis): SupportCheck;
  // A key of `settings`, or of a condensation given as an object, that is not listed here throws a TypeError.
  run(model: Model, analysis?: Analysis,
      settings?: { subcases?: number[]; memoryBytes?: number; threads?: number;
                   condensation?: CondensationSettings; streamed?: boolean;
                   // modes and buckling on the full path: the eigenvalue solver
                   eigenSolver?: EigenSolverName | EigenSolverSettings;
                   externalUnits?: ExternalUnitSettings[];
                   // statics: envelopes over the load cases, and whether the per-case tables of their kinds are kept
                   envelopes?: EnvelopeSpec[]; keepsCaseTables?: boolean }): AnalysisResult;
  condensationCache(options?: { memoryBytes?: number }): CondensationCache;
  // The residual structure with superelements upstream of it (statics or modes).
  runSuperelements(residual: Model, parts: { id: number; model: Model; boundary: [number, number][] }[],
                   analysis?: "statics" | "modes",
                   settings?: { subcases?: number[]; memoryBytes?: number; threads?: number }):
    { superelements: { id: number; nodalTables: unknown[]; elementTables: unknown[]; eigenvalues: unknown[] }[];
      diagnostics: Diagnostic[] };
}
