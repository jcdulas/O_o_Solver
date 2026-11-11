import type { AnalysisResult } from "./index.js";
import type { OptimizationDeclarations, OptimizationSettings, OptimizationValues, OptimizationMetadata,
  GradientSettings, DesignGradients } from "./optimization.js";
import type { StudyDefinition, StudySettings, StudyMatrix, DifferenceSettings, StudyPoint } from "./study.js";

export interface WorkerPoolConfig {
  readerURL: string | URL; solverURL?: string | URL; hooksURL?: string | URL;
  model: Uint8Array; mode?: "study" | "native";
  definition?: StudyDefinition | OptimizationDeclarations;
  settings?: StudySettings | OptimizationSettings;
  workers?: number; initializationTimeoutMs?: number; maximumQueuedPoints?: number;
  solverOptions?: { threads?: number };
}
export interface WorkerEvaluationOptions {
  signal?: AbortSignal; timeoutMs?: number; includeResults?: boolean;
  gradients?: boolean | GradientSettings | DifferenceSettings;
  modeReferences?: { responseId: number; subcase: number; shape: ArrayLike<number> }[];
  referencePoint?: StudyPoint & { analyses: AnalysisResult[] };
}
export interface WorkerEvaluation {
  values: OptimizationValues; analyses?: AnalysisResult[]; gradients?: DesignGradients | StudyMatrix;
}
export declare class OptimizationWorkerPool {
  metadata(): OptimizationMetadata | ReturnType<import("./study.js").OptimizationStudy["metadata"]>;
  evaluate(x?: ArrayLike<number>, options?: WorkerEvaluationOptions): Promise<WorkerEvaluation>;
  evaluateMany(points: ArrayLike<ArrayLike<number>>, options?: WorkerEvaluationOptions & {
    onPoint?(point: WorkerEvaluation, index: number): void | Promise<void> }): Promise<WorkerEvaluation[]>;
  close(reason?: Error): void;
}
export declare function createOptimizationWorkerPool(config: WorkerPoolConfig): Promise<OptimizationWorkerPool>;
