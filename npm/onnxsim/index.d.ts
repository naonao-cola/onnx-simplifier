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

declare const _default: { simplify: typeof simplify; versions: typeof versions };
export default _default;
