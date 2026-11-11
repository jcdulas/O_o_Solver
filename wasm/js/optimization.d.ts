import type { AnalysisResult } from "./index.js";
export type Numbers = number[] | Float64Array;
export interface DesignVariable {
  id: number; label?: string; initialValue: number; lower: number; upper: number; moveLimit?: number;
}
export interface DesignRelation {
  id: number; card?: string; targetKind: "property" | "material" | "element";
  targetCard: string; targetId: number; field: string; fieldItem?: number;
  form?: "linear" | "equation" | "supplied"; constant?: number;
  terms?: { variableId: number; coefficient: number }[];
  equationId?: number; arguments?: { variableId?: number; constantLabel?: string }[];
  lower?: number; upper?: number;
}
export interface DesignResponse {
  id: number; label?: string; type: string; propertyType?: string; region?: number;
  attributeA?: number; attributeB?: number; componentName?: string;
  targets?: number[]; plies?: number[]; subcases?: number[]; tracksMode?: boolean;
  sections?: SectionDefinition[]; cuts?: CutDefinition[]; zones?: number[];
}
export interface SectionDefinition {
  id: number; name?: string; members: { elementId: number; weight?: number }[];
  axis?: [number, number, number]; frameId?: number; reduce?: "average" | "sum";
}
export interface CutDefinition {
  id: number; name?: string; grids: number[]; elements: number[];
  point?: [number, number, number]; positionFrameId?: number; displacementFrameId?: number;
}
export interface ZoneDefinition {
  id: number; name?: string; group?: number; frameId?: number; elements: number[];
  enclosedPolicy?: "warn" | "include" | "exclude";
  boundary?: { grid: number; components?: string; isPiercedPoint?: boolean;
    causes?: { kind: string; card: string; id: number; components?: string }[] }[];
  enclosed?: { elementId: number; card: string; isPointElement?: boolean }[];
}
export interface EquationArgument { kind: string; id?: number; component?: number; label?: string; name?: string }
export interface BatchArgument { name: string; components: number; values: Float64Array; isResponse: boolean }
export interface BatchCall { arguments: BatchArgument[]; items: number }
export interface BatchResult { outputs?: number; values: Numbers; partials?: Numbers[] }
export interface BatchCallable {
  evaluate(request: BatchCall): BatchResult;
  evaluateWithPartials?(request: BatchCall): BatchResult;
  hessianVector?(request: BatchCall & { outputs: number; weights: Float64Array; directions: BatchArgument[] }): Numbers[];
}
export interface SuppliedField { relationIndex: number; value: number; gradient: Numbers }
export interface OptimizationDeclarations {
  tables?: Record<string, Uint8Array>;
  variables?: DesignVariable[];
  links?: { dependentId: number; constant?: number; multiplier?: number;
    terms: { variableId: number; coefficient: number }[] }[];
  relations?: DesignRelation[];
  constants?: Record<string, number>;
  shapes?: { variableId: number; gridId: number; coefficient?: number; direction: [number, number, number] }[];
  shapeBatches?: { variableId: number; gridIds: ArrayLike<number>; directions: Numbers }[];
  responses?: DesignResponse[];
  zones?: ZoneDefinition[];
  zoneSubcases?: { subcase: number; spcSet?: number; loadSet?: number }[];
  equationResponses?: { id: number; label?: string; callable?: string; equationId?: number;
    byLabel?: boolean; subcases?: number[]; arguments: EquationArgument[] }[];
  constraints?: { responseId: number; lower?: number; upper?: number; subcases?: number[] }[];
  objective?: { responseId: number; sense?: "min" | "max"; subcase?: number };
  callables?: Record<string, BatchCallable>;
  suppliedAt?(x: Float64Array): SuppliedField[];
}
export interface OptimizationSettings {
  derivatives?: "native" | "numerical";
  threads?: number; memoryBytes?: number; maximumOutputBytes?: number; blockItems?: number; deckOutputs?: boolean;
  // Missing kind: disabled. Empty array: all subcases of that kind. Default: statics only.
  analyses?: { statics?: number[]; modes?: number[]; buckling?: number[]; aeroelastic?: number[] };
}
export interface OptimizationMetadata {
  variables: (DesignVariable & { label: string; moveLimit: number | null })[];
  independentIndices: Float64Array; initialValues: Float64Array; maximumOutputBytes: number;
  relations: { id: number; targetCard: string; targetId: number; field: string; fieldItem: number;
    lower: number | null; upper: number | null }[];
  objectiveResponseId: number | null; objectiveSense: "min" | "max"; objectiveSubcase: number;
}
export interface SparseGradient { columns: Float64Array; values: Float64Array }
export interface DesignPoint {
  variables: Float64Array; variableGradients: SparseGradient[];
  fields: (SparseGradient & { value: number })[];
}
export interface ResponseItem { subcase: number; entity: number; component: number; layer: number }
export interface ResponseValues {
  id: number; label: string; values: Float64Array; items: ResponseItem[];
  statuses: ("ok" | "failed" | "not_evaluated")[];
}
export interface ConstraintValue {
  setId: number; responseId: number; item: number; label: ResponseItem; side: "lower" | "upper";
  bound: number; value: number; normalised: number; derivativeScale: number;
}
export interface OptimizationValues {
  x: Float64Array; subcases: Int32Array; objective: number | null;
  responses: ResponseValues[]; constraints: ConstraintValue[];
  trackedModes: { responseId: number; subcase: number; mode: number; mac: number; isUncertain: boolean;
    shape: Float64Array }[];
}
export interface GradientSettings {
  method?: "automatic" | "direct" | "adjoint" | "chained" | "finiteDifference";
  relativeStep?: number; absoluteStep?: number;
  responseIds?: number[]; itemMasks?: { responseId: number; isRetained: number[] | Uint8Array }[];
}
export interface DesignGradients {
  variableCount: number; responses: { id: number; itemCount: number; values: Float64Array }[];
  objective: Float64Array | null; constraints: Float64Array[];
  report: { solves: number; factorisations: number; analyticDerivatives: number;
    fieldPerturbations: number; shapePerturbations: number; analysisPerturbations: number;
    method?: "finiteDifference"; evaluations?: number; order?: number };
}
export interface SparseWeights { rowStart: number[] | Float64Array; items: number[] | Float64Array; values: Numbers }
export interface ProductWeights { rows?: number; responses: Record<number, Numbers | SparseWeights> }
export interface ProductBlock { first: number; count: number; rows: number; columns: number; values: Float64Array }
export interface ProductReport {
  solves: number; blocks: number; factorisations: number; lineariseSeconds: number; productSeconds: number;
  isLinearisationReused: boolean; linearisationBytes: number; adjointSolves: number; isAdjointReused: boolean;
  perturbedLinearisations: number;
  hasHessianBlocks: boolean; areHessianBlocksReused: boolean; hessianBlocksReason: string;
}
export interface ProductResult { rows: number; columns: number; values?: Float64Array; report: ProductReport }
export interface ProductSettings {
  count?: number; block?: number; onBlock?(block: ProductBlock): void;
  responseIds?: number[]; subcases?: number[];
}
export interface GradientBlock {
  rowBegin: number; rowCount: number; columnBegin: number; columnCount: number; values: Float64Array;
}
export interface SparseSupport { rows: number; columns: number; row: Float64Array; column: Float64Array }
export interface SparseTriplets extends SparseSupport { values: Float64Array }
export interface DesignBlocks {
  sets: { stiffness: SparseTriplets; stiffnessSupport: SparseSupport; transform: SparseTriplets;
    equationDofs: Float64Array }[];
  subcases: { subcase: number; set: number; pseudoLoads: SparseTriplets; pseudoLoadSupport: SparseSupport;
    hessianDisplacement: Float64Array; hessianMixed: Float64Array; adjoint: Float64Array }[];
  itemWeights: SparseTriplets; itemWeightSupport: SparseSupport;
  explicitParts: SparseTriplets; explicitPartSupport: SparseSupport; itemSubcases: Int32Array;
  hasHessian: boolean; designSupport: SparseSupport; hessianDesign: Float64Array;
  hessianReport: ProductReport; hessianAsymmetry: number; dofPoints: Int32Array; dofComponents: Int8Array;
}
export declare class OptimizationProblem {
  metadata(): OptimizationMetadata;
  designPoint(x: Numbers): DesignPoint;
  evaluate(x?: Numbers, modeReferences?: { responseId: number; subcase: number; shape: Numbers }[]): OptimizationEvaluation;
  delete(): void;
}
export declare class OptimizationEvaluation {
  values(): OptimizationValues;
  analysisResults(): AnalysisResult;
  gradients(settings?: GradientSettings): DesignGradients;
  streamGradients(onBlock: (block: GradientBlock) => void, block?: number): {
    solves: number; blocks: number; rows: number; columns: number; orientation: string;
  };
  jvp(directions: Numbers, settings?: ProductSettings): ProductResult;
  vjp(weights: ProductWeights, settings?: Pick<ProductSettings, "block" | "onBlock">): ProductResult;
  hessianVector(weights: ProductWeights, directions: Numbers, count?: number): { values: Float64Array; report: ProductReport };
  hessian(weights: ProductWeights): Float64Array;
  blocks(weights?: ProductWeights | null): DesignBlocks;
  lagrangianWeights(objectiveFactor?: number, multipliers?: Numbers | null): ProductWeights;
  clearLinearisation(): void;
  delete(): void;
}
