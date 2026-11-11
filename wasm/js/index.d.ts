import type { OptimizationProblem, OptimizationDeclarations, OptimizationSettings } from "./optimization.js";
export * from "./optimization.js";
export * from "./study.js";
export * from "./study-workers.js";

// Types of the oos-solver package (index.js).
//
// Every name the interface gives or takes is an exact neutral name of O_o_Reader: card and result element types use
// UpperCamelCase ("ShellQuad4"); table kinds ("singlePointConstrForces") and columns ("bendingMomentA1") use lowerCamelCase.
// Messages (Diagnostic.message, Diagnostic.card, Diagnostic.field and errors) use the same neutral names.

export declare function supportsMemory64(): boolean;
export declare function supportsWasmThreads(): boolean;
export declare function loadOosSolver(options?: Record<string, unknown> & { threads?: number }): Promise<OosSolver>;

export type Analysis = "statics" | "modes" | "buckling" | "aeroelastic";

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
  kind: string; // "stress", "strain", "forces", "strainEnergy", "failureIndices"
  // The result element type of O_o_Reader: the exact neutral card type ("ShellQuad4",
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
  // types of the elements ("ShellQuad4") and "condensedUnit" (a condensed component on its boundary).
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
  nodalTables: NodalTable[];
  elementTables: ElementTable[];
  gridPointForces: GridPointForceTable[];
  eigenvalues: EigenvalueTable[];
  diagnostics: Diagnostic[];
  aeroelastic: AeroelasticResult | null;
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

export interface Model {
  tableNames(): string[];
  checkSupport(analysis: Analysis): SupportCheck;
  delete(): void;
}

export interface OptimizationCapabilities {
  version: number;
  analyses: { analysis: Analysis; solution: 101 | 103 | 105 | 144; sizingFirstOrder: boolean;
    shapeFirstOrder: boolean; secondOrder: boolean; restrictions: string[] }[];
  responses: string[];
  fields: { targetKind: "property" | "material" | "element"; targetCard: string; field: string;
    type: string; requiresItem: boolean }[];
}
export interface OptimizationPreparation {
  stage: "preparation"; canPrepare: boolean; error: string | null;
  metadata: import("./optimization.js").OptimizationMetadata | null;
  analyses: (SupportCheck & { analysis: Analysis })[];
  capabilities: OptimizationCapabilities["analyses"];
}

export declare class OosSolver {
  createStudy(reader: import("oor-reader").OorReader, deck: import("oor-reader").Deck,
    definition: import("./study.js").StudyDefinition, settings?: import("./study.js").StudySettings,
    hooks?: import("./study.js").StudyHooks): import("./study.js").OptimizationStudy;
  restoreStudy(reader: import("oor-reader").OorReader, deck: import("oor-reader").Deck,
    hooks?: import("./study.js").StudyHooks): ReturnType<typeof import("./study.js").restoreStudy>;
  shapeBasisFromReader(reader: import("oor-reader").OorReader, auxiliary: import("oor-reader").Deck,
    options: { variableId: number; cases?: { subcase: number; coefficient?: number }[]; gridIds?: number[] }):
    { variableId: number; gridIds: Int32Array; directions: Float64Array };
  version(): string;
  threading(): { enabled: boolean; maximumThreads: number; activeWorkers: number; defaultThreads: number };
  // `oor` is a loaded OorReader (package oor-reader), `deck` one of its decks.
  // Every element card type the solver computes ("ShellQuad4") with its formulations, the default first.
  elementFormulations(): { card: string; formulations: string[] }[];
  capabilities(model?: Model | null): OptimizationCapabilities;
  checkOptimization(model: Model, declarations?: OptimizationDeclarations,
    settings?: OptimizationSettings): OptimizationPreparation;
  // `options.elementFormulations` chooses the formulation of the elements of some card types
  // ({ ShellQuad4: "ClassicFormulation" }), before the model's ShellQuad4Formulation parameter. The keys are the neutral card types
  // of elementFormulations(), exactly: an unknown card type throws an Error.
  modelFromReader(oor: unknown, deck: unknown,
                  options?: { elementFormulations?: Record<string, string> }): Model;
  createOptimization(model: Model, declarations?: OptimizationDeclarations, settings?: OptimizationSettings): OptimizationProblem;
  optimizationFromReader(oor: unknown, deck: unknown, declarations?: OptimizationDeclarations,
    settings?: OptimizationSettings): OptimizationProblem;

  checkSupport(model: Model, analysis?: Analysis): SupportCheck;
  run(model: Model, analysis?: Analysis,
      settings?: { subcases?: number[]; memoryBytes?: number; threads?: number }): AnalysisResult;
}
