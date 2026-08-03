// Fill modes for the dummy inputs the "Run inference" panel auto-generates.
//
// Mirrors the Python CLI's `--input-fill` option (see
// onnxsim.model_checking.INPUT_FILL_CHOICES): the same names produce analogous
// values so the in-browser check and the command-line check agree on what they
// feed a model. The before/after comparison uses this to build ONE shared input
// and run it through both models, so the fill choice decides how hard the graph
// change is exercised — all-zero inputs can mask a divergence (many ops map
// 0 -> 0), whereas varied inputs (random/ones/arange) actually stress it.
//
// Only float inputs receive the requested values; integer/bool inputs are left
// at zero, the safe default for index-like tensors (Gather, etc.) where an
// arbitrary value could index out of bounds. "random" is a seeded xorshift32,
// so every call yields the identical sequence: runInference() re-feeds the same
// tensor each iteration and asserts the output never changes.

// The selectable fill modes, in the order shown in the UI dropdown. Kept in
// sync with the Python side's INPUT_FILL_CHOICES.
export const INPUT_FILL_MODES = ["random", "ones", "zeros", "arange"];

// Fill `arr` (a typed array) in place according to `fill`, for an input of the
// given onnxruntime element `type` ("float32", "int64", ...). Returns `arr`.
// "zero"/"zeros" and any integer/bool input are left untouched (already zeroed
// by the typed-array constructor). Throws on an unknown mode.
export function fillInput(arr, type, fill) {
  const isFloat = type === "float32" || type === "float64";
  // Integer/bool tensors always stay zero-filled (see file header); "zero" and
  // "zeros" are an explicit request for an all-zero float tensor.
  if (!isFloat || fill === "zero" || fill === "zeros") return arr;
  if (fill === "ones") {
    arr.fill(1);
  } else if (fill === "arange") {
    // 0, 1, 2, ... in row-major order, matching numpy's np.arange fill.
    for (let i = 0; i < arr.length; i++) arr[i] = i;
  } else if (fill === "random") {
    let s = 0x2545f491; // fixed non-zero seed
    for (let i = 0; i < arr.length; i++) {
      // xorshift32 (bitwise ops keep `s` a 32-bit signed int throughout).
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      arr[i] = s / 0x80000000; // 32-bit signed / 2^31 -> [-1, 1)
    }
  } else {
    throw new Error(`unknown input fill '${fill}'`);
  }
  return arr;
}
