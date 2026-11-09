// Types of the oos-solver package (index.js).
//
// Every name the interface gives or takes is an exact neutral name of O_o_Reader: card and result element types use
// UpperCamelCase ("ShellQuad4"); table kinds ("singlePointConstrForces") and columns ("bendingMomentA1") use lowerCamelCase.
// Messages (Diagnostic.message, Diagnostic.card, Diagnostic.field and errors) use the same neutral names.

export declare function supportsMemory64(): boolean;
export declare function loadOosSolver(options?: Record<string, unknown>): Promise<OosSolver>;

export type Analysis = "statics" | "modes" | "buckling";

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
  analysis: "static" | "modes" | "buckling";
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
  analysis: "static" | "modes" | "buckling";
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
}

export interface Model {
  tableNames(): string[];
  checkSupport(analysis: Analysis): SupportCheck;
  delete(): void;
}

export declare class OosSolver {
  version(): string;
  // `oor` is a loaded OorReader (package oor-reader), `deck` one of its decks.
  // Every element card type the solver computes ("ShellQuad4") with its formulations, the default first.
  elementFormulations(): { card: string; formulations: string[] }[];
  // `options.elementFormulations` chooses the formulation of the elements of some card types
  // ({ ShellQuad4: "ClassicFormulation" }), before the model's ShellQuad4Formulation parameter. The keys are the neutral card types
  // of elementFormulations(), exactly: an unknown card type throws an Error.
  modelFromReader(oor: unknown, deck: unknown,
                  options?: { elementFormulations?: Record<string, string> }): Model;
  checkSupport(model: Model, analysis?: Analysis): SupportCheck;
  run(model: Model, analysis?: Analysis,
      settings?: { subcases?: number[]; memoryBytes?: number }): AnalysisResult;
}
