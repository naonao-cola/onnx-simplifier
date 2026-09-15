export interface SimplifyOptions {
  /** onnx-optimizer pass names to skip. */
  skipOptimizers?: string[];
  /** Fold constant subgraphs. Default true. */
  constantFolding?: boolean;
  /** Run ONNX shape inference. Default true. */
  shapeInference?: boolean;
  /**
   * Skip folding a constant whose output would be larger than this many
   * bytes. Default 1.5 GiB (matches the Python CLI's default).
   */
  tensorSizeThreshold?: number;
  /** Target opset version. <= 0 (the default) keeps the model's opset. */
  targetOpsetVersion?: number;
  /** Also return a Chrome trace JSON of the simplification profile. */
  profile?: boolean;
  /** Bake MAC/FLOP counts into the model's metadata_props. */
  annotateModelInfo?: boolean;
  /**
   * Print a detailed node/value-level before/after diff (which nodes/values
   * were removed, added, or changed) via console.log, in addition to the
   * op-count summary always printed. Default false.
   */
  graphDiff?: boolean;
}

export interface SimplifyResult {
  /** Serialized `onnx.ModelProto` bytes of the simplified model. */
  model: Uint8Array;
  /** Chrome trace JSON when `options.profile` was true, otherwise "". */
  trace: string;
}

export interface Versions {
  onnxsim: string;
  onnx_optimizer: string;
  [key: string]: string;
}

/**
 * Simplify a serialized ONNX model.
 *
 * @param model - serialized `onnx.ModelProto` bytes.
 */
export function simplify(
  model: Uint8Array | ArrayBuffer,
  options?: SimplifyOptions,
): Promise<SimplifyResult>;

/** onnxsim / onnx-optimizer version strings baked into this build. */
export function versions(): Promise<Versions>;

/** Serialized `onnx.ModelProto` bytes (Uint8Array or ArrayBuffer). */
export type OnnxModelBytes = Uint8Array | ArrayBuffer;

/** A single calibration tensor: an onnxruntime-web Tensor or canonical form. */
export type CalibrationTensor =
  | { type: string; dims: number[] | readonly number[]; data: ArrayBufferView }
  | { dtype: number; dims: number[] | readonly number[]; data: ArrayBufferView };

/** One calibration batch: graph input name -> tensor. */
export type CalibrationBatch =
  | Record<string, CalibrationTensor>
  | Map<string, CalibrationTensor>;

/** One batch or an array of batches. */
export type CalibrationData = CalibrationBatch | CalibrationBatch[];

export interface SparsityOptions {
  sparsity?: number;
}

export interface NormOptions extends SparsityOptions {
  importanceNorm?: string;
  globalSparsity?: boolean;
}

export interface PatternOptions extends SparsityOptions {
  n?: number;
  m?: number;
}

export interface EmbeddingVocabResult {
  model: Uint8Array;
  matched: boolean;
  keptTokenIds: number[];
  lmHeadPruned: boolean;
}

/** Data-free quantization / pruning passes (model bytes in, model bytes out). */
export function crossLayerEqualize(model: OnnxModelBytes): Promise<Uint8Array>;
export function quantizeDynamicMatMulIntegerToFloat(model: OnnxModelBytes): Promise<Uint8Array>;
export function quantizeAttentionDynamic(model: OnnxModelBytes): Promise<Uint8Array>;
export function quantizeWeightOnlyInt16(model: OnnxModelBytes): Promise<Uint8Array>;
export function quantizeWeightOnlyInt8Block(model: OnnxModelBytes): Promise<Uint8Array>;
export function quantizeWeightOnlyMxfp4(model: OnnxModelBytes): Promise<Uint8Array>;
export function quantizeWeightOnlyMatMulNbits(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyDoubleQuantization(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyAnyPrecisionLlm(
  model: OnnxModelBytes,
  options?: { bits?: number; maxBits?: number; blockSize?: number },
): Promise<Uint8Array>;
export function applyQuarot(
  model: OnnxModelBytes,
  options?: { seed?: number; blockSize?: number; epsilon?: number },
): Promise<Uint8Array>;
export function applyIq4Nl(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ4_0(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ4_1(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ5_0(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ5_1(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ8_0(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ2K(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ3K(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufTernary(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyFp6Llm(model: OnnxModelBytes): Promise<Uint8Array>;
export function applyGgufQ6K(model: OnnxModelBytes): Promise<Uint8Array>;
export function pruneMagnitude(
  model: OnnxModelBytes,
  options?: PatternOptions & { globalSparsity?: boolean },
): Promise<Uint8Array>;
export function applyStructuredPruning(
  model: OnnxModelBytes,
  options?: NormOptions,
): Promise<Uint8Array>;
export function applyAttentionHeadPruning(
  model: OnnxModelBytes,
  options?: SparsityOptions & { importanceNorm?: string },
): Promise<Uint8Array>;
export function applyMoeExpertChannelPruning(
  model: OnnxModelBytes,
  options?: SparsityOptions,
): Promise<Uint8Array>;
export function applyQmoeExpertChannelPruning(
  model: OnnxModelBytes,
  options?: SparsityOptions,
): Promise<Uint8Array>;
export function applyEmbeddingVocabPruning(
  model: OnnxModelBytes,
  options?: { keepTokenIds?: number[]; dropTokenIds?: number[]; inputName?: string },
): Promise<EmbeddingVocabResult>;
export function applyEmbeddingVocabMagnitudePruning(
  model: OnnxModelBytes,
  options?: SparsityOptions & { protectTokenIds?: number[]; inputName?: string },
): Promise<EmbeddingVocabResult>;

/** Calibration-driven passes (need onnxruntime-web, like constant folding). */
export function applyStructuredWandaPruning(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: NormOptions & { epsilon?: number },
): Promise<Uint8Array>;
export function applyAttentionHeadWandaPruning(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: SparsityOptions & { epsilon?: number; importanceNorm?: string },
): Promise<Uint8Array>;
export function applySparsegptPruning(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: PatternOptions & { percdamp?: number; procBlockSize?: number },
): Promise<Uint8Array>;
export function applyWandaPruning(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: PatternOptions & { epsilon?: number; globalSparsity?: boolean },
): Promise<Uint8Array>;
export function applyMoeWholeExpertPruning(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: SparsityOptions,
): Promise<Uint8Array>;
export function applyQmoeWholeExpertPruning(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: SparsityOptions,
): Promise<Uint8Array>;
export function applyTransformerBlockPruning(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: SparsityOptions & { numBlocksToDrop?: number },
): Promise<Uint8Array>;
export function applyImatrixQuantization(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: {
    blockSize?: number;
    numScaleCandidates?: number;
    scaleLo?: number;
    scaleHi?: number;
    skipNames?: string[];
  },
): Promise<Uint8Array>;
export function applyOutlierSuppression(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: { alpha?: number; epsilon?: number },
): Promise<Uint8Array>;
export function applyLlmInt8(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: { outlierThreshold?: number; epsilon?: number },
): Promise<Uint8Array>;
export function applyGptq(
  floatModel: OnnxModelBytes,
  quantizedModel: OnnxModelBytes,
  calibration: CalibrationData,
  options?: { percdamp?: number; procBlockSize?: number },
): Promise<Uint8Array>;
export function applyAdaround(
  floatModel: OnnxModelBytes,
  quantizedModel: OnnxModelBytes,
  calibration: CalibrationData,
  options?: {
    numIterations?: number;
    learningRate?: number;
    regParam?: number;
    warmStart?: number;
    betaStart?: number;
    betaEnd?: number;
  },
): Promise<Uint8Array>;
export function applyQronos(
  floatModel: OnnxModelBytes,
  quantizedModel: OnnxModelBytes,
  calibration: CalibrationData,
  options?: { percdamp?: number; procBlockSize?: number },
): Promise<Uint8Array>;
export function applyTesseraq(
  floatModel: OnnxModelBytes,
  quantizedModel: OnnxModelBytes,
  calibration: CalibrationData,
  options?: {
    numBits?: number;
    numIterations?: number;
    parRounds?: number;
    learningRate?: number;
    scaleLearningRate?: number;
    regParam?: number;
    warmStart?: number;
    betaStart?: number;
    betaEnd?: number;
  },
): Promise<Uint8Array>;
export function applyAwq(
  floatModel: OnnxModelBytes,
  quantizedModel: OnnxModelBytes,
  calibration: CalibrationData,
  options?: { numAlphaSteps?: number },
): Promise<Uint8Array>;
export function applyQuarotGptq(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: {
    seed?: number;
    blockSize?: number;
    percdamp?: number;
    procBlockSize?: number;
    epsilon?: number;
  },
): Promise<Uint8Array>;
export function applyGptvq(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: {
    seed?: number;
    vectorDim?: number;
    numCentroids?: number;
    numIterations?: number;
    percdamp?: number;
    skipNames?: string[];
  },
): Promise<Uint8Array>;
export function applySmoothQuant(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: { alpha?: number; epsilon?: number },
): Promise<Uint8Array>;
export function applyOutlierSuppressionPlus(
  model: OnnxModelBytes,
  calibration: CalibrationData,
  options?: { alpha?: number; epsilon?: number },
): Promise<Uint8Array>;

declare const _default: {
  simplify: typeof simplify;
  versions: typeof versions;
  crossLayerEqualize: typeof crossLayerEqualize;
  quantizeDynamicMatMulIntegerToFloat: typeof quantizeDynamicMatMulIntegerToFloat;
  quantizeAttentionDynamic: typeof quantizeAttentionDynamic;
  quantizeWeightOnlyInt16: typeof quantizeWeightOnlyInt16;
  quantizeWeightOnlyInt8Block: typeof quantizeWeightOnlyInt8Block;
  quantizeWeightOnlyMxfp4: typeof quantizeWeightOnlyMxfp4;
  quantizeWeightOnlyMatMulNbits: typeof quantizeWeightOnlyMatMulNbits;
  applyDoubleQuantization: typeof applyDoubleQuantization;
  applyAnyPrecisionLlm: typeof applyAnyPrecisionLlm;
  applyQuarot: typeof applyQuarot;
  applyIq4Nl: typeof applyIq4Nl;
  applyGgufQ4_0: typeof applyGgufQ4_0;
  applyGgufQ4_1: typeof applyGgufQ4_1;
  applyGgufQ5_0: typeof applyGgufQ5_0;
  applyGgufQ5_1: typeof applyGgufQ5_1;
  applyGgufQ8_0: typeof applyGgufQ8_0;
  applyGgufQ2K: typeof applyGgufQ2K;
  applyGgufQ3K: typeof applyGgufQ3K;
  applyGgufTernary: typeof applyGgufTernary;
  applyFp6Llm: typeof applyFp6Llm;
  applyGgufQ6K: typeof applyGgufQ6K;
  pruneMagnitude: typeof pruneMagnitude;
  applyStructuredPruning: typeof applyStructuredPruning;
  applyAttentionHeadPruning: typeof applyAttentionHeadPruning;
  applyMoeExpertChannelPruning: typeof applyMoeExpertChannelPruning;
  applyQmoeExpertChannelPruning: typeof applyQmoeExpertChannelPruning;
  applyEmbeddingVocabPruning: typeof applyEmbeddingVocabPruning;
  applyEmbeddingVocabMagnitudePruning: typeof applyEmbeddingVocabMagnitudePruning;
  applyStructuredWandaPruning: typeof applyStructuredWandaPruning;
  applyAttentionHeadWandaPruning: typeof applyAttentionHeadWandaPruning;
  applySparsegptPruning: typeof applySparsegptPruning;
  applyWandaPruning: typeof applyWandaPruning;
  applyMoeWholeExpertPruning: typeof applyMoeWholeExpertPruning;
  applyQmoeWholeExpertPruning: typeof applyQmoeWholeExpertPruning;
  applyTransformerBlockPruning: typeof applyTransformerBlockPruning;
  applyImatrixQuantization: typeof applyImatrixQuantization;
  applyOutlierSuppression: typeof applyOutlierSuppression;
  applyOutlierSuppressionPlus: typeof applyOutlierSuppressionPlus;
  applyLlmInt8: typeof applyLlmInt8;
  applyGptq: typeof applyGptq;
  applyAdaround: typeof applyAdaround;
  applyQronos: typeof applyQronos;
  applyTesseraq: typeof applyTesseraq;
  applyAwq: typeof applyAwq;
  applyQuarotGptq: typeof applyQuarotGptq;
  applyGptvq: typeof applyGptvq;
  applySmoothQuant: typeof applySmoothQuant;
};
export default _default;
