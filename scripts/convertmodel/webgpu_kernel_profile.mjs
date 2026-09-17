// Pure, GPU-free helper: given a per-node latency profile (as produced by
// webgpu_kernel_tuner.mjs's profileNodeLatency), decide which nodes are the
// "latency-dominant" ones worth spending a full tuning search on -- the
// second half of the "profile first, tune only the dominant ops" workflow.
// Kept in its own module (no Pyodide/WebGPU import at all) so it's testable
// with plain synthetic numbers, no browser or device involved.
//
// Sorts nodes by measured latency descending and keeps taking the biggest
// ones until their running total covers `thresholdFraction` of the whole
// profiled total (default 80%) -- a memory-bound op that dominates a
// model's real latency naturally rises to the top here even though it's not
// a Conv, which is the whole point of profiling before tuning rather than
// only ever tuning Conv nodes (or every node, unconditionally). Always keeps
// at least `minNodes` (default 1) so a caller with any profiled node at all
// gets something to tune even if one node alone already crosses the
// threshold trivially.
//
// A node profileNodeLatency marked `skipped` (no dedicated kernel in
// isolation -- a pure view op -- or a profiling error) is never selected: it
// has no measured cost to be "dominant" with.

/**
 * @param {Array<{nodeName: string, medianMs: number, skipped?: boolean}>} profileResults
 * @param {{thresholdFraction?: number, minNodes?: number}} [options]
 * @returns {string[]} node names, ranked slowest first
 */
export function selectDominantNodes(profileResults, options = {}) {
  const { thresholdFraction = 0.8, minNodes = 1 } = options;
  const timed = profileResults.filter((r) => !r.skipped && Number.isFinite(r.medianMs) && r.medianMs > 0);
  const total = timed.reduce((sum, r) => sum + r.medianMs, 0);
  if (total <= 0) return [];

  const sorted = [...timed].sort((a, b) => b.medianMs - a.medianMs);
  const dominant = [];
  let cumulative = 0;
  for (const r of sorted) {
    dominant.push(r.nodeName);
    cumulative += r.medianMs;
    if (cumulative / total >= thresholdFraction && dominant.length >= minNodes) break;
  }
  return dominant;
}
