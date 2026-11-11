import type { Deck, OorReader, PartDefinition, CardScalar, WriteReport } from "oor-reader";
import type { Analysis, AnalysisResult, OosSolver } from "./index.js";
import type { OptimizationValues, ResponseItem } from "./optimization.js";

export interface StudyVariable {
  id: number; label?: string; initialValue: number; lower: number; upper: number; scale?: number;
  kind?: "continuous" | "integer" | "choice"; choices?: number[];
}
export type ParameterTarget =
  { kind: "card"; type: string; id?: number; key?: number; field?: string; slot?: number; forceReal?: boolean; part?: PartDefinition } |
  { kind: "case"; command: string; subcase?: number; sectionKey?: number; variant?: string; options?: string[] } |
  { kind: "solution" };
export interface StudyParameter {
  target: ParameterTarget; variableId?: number; coefficient?: number; constant?: number;
  terms?: { variableId: number; coefficient?: number }[];
  callback?: string;
}
export interface StudyResponse {
  id: number; label?: string; analysis?: number;
  source: "nodal" | "element" | "gridPointForce" | "eigenvalue" | "trimVariable" |
    "stabilityDerivative" | "pressure" | "boxForce" | "boxMoment" | "hingeMoment" | "callback" | "query";
  query?: string; column?: string; idColumn?: string; args?: Record<string, unknown>;
  kind?: string; subcase?: number; elementType?: string; ids?: number[]; nodeIds?: number[]; layers?: number[];
  components?: string | number | (string | number)[]; rowNames?: string[];
  step?: number; mode?: number; tracksMode?: boolean; minimumMac?: number;
  variable?: string; derivativeColumn?: "rigidUnsplined" | "rigidSplined" | "elasticRestrained" |
    "elasticUnrestrained" | "inertialRestrained" | "inertialUnrestrained";
  reduce?: "none" | "sum" | "mean" | "max" | "min" | "maxAbs" | "norm";
  scale?: number; callback?: string;
}
export interface StudyDefinition {
  version: 1; variables: StudyVariable[]; parameters?: StudyParameter[];
  analyses: { analysis: Analysis; subcases?: number[] }[];
  responses: StudyResponse[];
  objective?: { responseId: number; item?: number; sense?: "min" | "max" };
  constraints?: { responseId: number; items?: number[]; lower?: number; upper?: number; scale?: number }[];
  update?: string;
}
export interface StudyContext {
  deck: Deck; reader: OorReader; solver: OosSolver; x: Float64Array; variables: Record<number, number>;
}
export interface StudyHooks {
  parameters?: Record<string, (context: StudyContext) => CardScalar>;
  updates?: Record<string, (context: StudyContext) => void>;
  responses?: Record<string, (context: StudyContext & { runs: AnalysisResult[]; response: StudyResponse;
    mode?: number; modeSign: number }) => number | ArrayLike<number> | { values: ArrayLike<number>; items?: ResponseItem[] }>;
}
export interface StudySettings {
  threads?: number; memoryBytes?: number; maximumOutputBytes?: number; maximumCachedPoints?: number;
  applyDefaults?: boolean; checkReferences?: boolean;
  model?: { elementFormulations?: Record<string, string> };
}
export interface DifferenceSettings {
  relativeStep?: number; absoluteStep?: number; side?: "central" | "forward" | "backward";
  allowNonsmooth?: boolean; signal?: AbortSignal;
}
export interface StudyMatrix {
  rows: number; columns: number; values?: Float64Array; keys?: string[];
  report: { method: "finiteDifference"; evaluations: number; order?: number };
}
export interface StudyVector { values: Float64Array; keys: string[] }
export interface StudyPoint { values: OptimizationValues; analyses?: AnalysisResult[] }

export declare class OptimizationStudy {
  constructor(solver: OosSolver, reader: OorReader, deck: Deck, definition: StudyDefinition,
    settings?: StudySettings, hooks?: StudyHooks);
  metadata(): { version: number; revision: number; variables: StudyVariable[]; initialValues: Float64Array;
    maximumOutputBytes: number; derivativeMethod: "finiteDifference"; analyses: StudyDefinition["analyses"] };
  evaluate(x?: ArrayLike<number>, references?: StudyEvaluation | null): StudyEvaluation;
  evaluateMany(points: ArrayLike<ArrayLike<number>>, options?: { signal?: AbortSignal; includeResults?: boolean;
    onPoint?(point: StudyPoint, index: number): void | Promise<void> }): Promise<StudyPoint[]>;
  saveDefinition(deck: Deck, x?: ArrayLike<number>, state?: unknown): void;
  rebuild(deck: Deck, definition?: StudyDefinition, settings?: StudySettings, hooks?: StudyHooks): OptimizationStudy;
  clearCache(): void;
  delete(): void;
}
export declare class StudyEvaluation {
  values(): OptimizationValues;
  vector(): StudyVector;
  analysisResults(): AnalysisResult[];
  model(): Deck;
  gradients(settings?: DifferenceSettings & { onBlock?(block: { rowBegin: number; rowCount: number;
    columnBegin: number; columnCount: number; values: Float64Array }): void }): StudyMatrix;
  jvp(directions: ArrayLike<number>, settings?: DifferenceSettings & { count?: number }): StudyMatrix;
  vjp(weights: ArrayLike<number>, settings?: DifferenceSettings & { rows?: number }): StudyMatrix;
  lagrangianWeights(objectiveFactor?: number, multipliers?: ArrayLike<number> | null): Float64Array;
  hessianVector(weights: ArrayLike<number>, directions: ArrayLike<number>, settings?: DifferenceSettings & {
    count?: number; outerRelativeStep?: number }): StudyMatrix;
  hessian(weights: ArrayLike<number>, settings?: DifferenceSettings & { outerRelativeStep?: number }): StudyMatrix;
  save(path: string, options?: { state?: unknown; compressionLevel?: number; shuffle?: boolean }): WriteReport;
  delete(): void;
}
export declare function restoreStudy(solver: OosSolver, reader: OorReader, deck: Deck,
  hooks?: StudyHooks): { study: OptimizationStudy; state: unknown; diagnostics: AnalysisResult["diagnostics"][];
    trackedModes: Omit<OptimizationValues["trackedModes"][number], "shape">[] };
export declare const STUDY_DOCUMENT: string;
