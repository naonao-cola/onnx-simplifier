// A tiny, dependency-free "node reduction per loop" chart for the converter
// page's onnxsim profiling section.
//
// simplify(..., profile=...)'s Chrome trace (see trace_viewer.mjs's header for
// the format background) carries, alongside its span flame graph, a
// "NodeCount" counter event (ph:"C") right after every round of each
// simplification fixed-point loop, tagged in args.loop with which loop
// produced it: "Initial" (the baseline, one sample), "Optimize" (the inner
// shape-inference + onnx-optimizer fixed point), "FoldConstant" (the outer
// pipeline), and "Rewrite" (only when a custom_rewriter was used). This
// renders that as one small SVG line chart per loop -- node count against
// round index within that loop -- so how quickly (and whether) each loop
// converges to a fixed point is visible directly, without needing
// chrome://tracing or Perfetto.
//
// extractNodeCounts(trace) is pure and DOM-free (exercised by
// test/node_reduction_view.test.mjs under Node); renderNodeReduction(container,
// trace, opts) is the DOM-touching part the page calls, mounted next to
// trace_viewer.mjs's flame graph from the same trace object.

const LOOP_ORDER = ["Initial", "Optimize", "FoldConstant", "Rewrite"];

// trace -> Map<loopName, [nodeCount, ...]>, each list in the order its rounds
// ran (the same start-timestamp order onnxsim's Profiler::Finish() writes
// traceEvents in).
export function extractNodeCounts(trace) {
  const events = (trace && trace.traceEvents) || [];
  const counts = new Map();
  for (const e of events) {
    if (e.ph !== "C" || e.name !== "NodeCount") continue;
    const args = e.args || {};
    const count = args.node_count;
    if (typeof count !== "number") continue;
    const loop = typeof args.loop === "string" ? args.loop : "?";
    if (!counts.has(loop)) counts.set(loop, []);
    counts.get(loop).push(count);
  }
  return counts;
}

function orderedLoops(counts) {
  const loops = LOOP_ORDER.filter((l) => counts.has(l));
  for (const l of counts.keys()) {
    if (!loops.includes(l)) loops.push(l);
  }
  return loops;
}

const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

// One loop's node-count-per-round curve as a small SVG line chart.
function renderOneChart(series, width = 260, height = 130) {
  const pad = { l: 34, r: 12, t: 10, b: 22 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const n = series.length;
  const maxV = Math.max(...series);
  const minV = Math.min(...series);
  const range = Math.max(1, maxV - minV);

  const xOf = (i) => pad.l + (n <= 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const yOf = (v) => pad.t + innerH - ((v - minV) / range) * innerH;

  const svg = svgEl("svg", {
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
  });

  svg.appendChild(
    svgEl("line", {
      x1: pad.l, y1: pad.t, x2: pad.l, y2: pad.t + innerH,
      stroke: "var(--muted, #999)", "stroke-width": 1,
    }),
  );
  svg.appendChild(
    svgEl("line", {
      x1: pad.l, y1: pad.t + innerH, x2: pad.l + innerW, y2: pad.t + innerH,
      stroke: "var(--muted, #999)", "stroke-width": 1,
    }),
  );

  const points = series.map((v, i) => `${xOf(i)},${yOf(v)}`).join(" ");
  svg.appendChild(
    svgEl("polyline", {
      points, fill: "none", stroke: "var(--accent, #3b82f6)", "stroke-width": 2,
    }),
  );
  series.forEach((v, i) => {
    svg.appendChild(
      svgEl("circle", { cx: xOf(i), cy: yOf(v), r: 3, fill: "var(--accent, #3b82f6)" }),
    );
  });

  const labelColor = "var(--fg, #333)";
  const maxLabel = svgEl("text", {
    x: pad.l - 5, y: pad.t + 4, "text-anchor": "end", "font-size": 10, fill: labelColor,
  });
  maxLabel.textContent = String(maxV);
  svg.appendChild(maxLabel);
  const minLabel = svgEl("text", {
    x: pad.l - 5, y: pad.t + innerH, "text-anchor": "end", "font-size": 10, fill: labelColor,
  });
  minLabel.textContent = String(minV);
  svg.appendChild(minLabel);

  const lastLabel = svgEl("text", {
    x: Math.min(xOf(n - 1) + 6, width - 4), y: yOf(series[n - 1]) + 3,
    "font-size": 10, fill: labelColor,
  });
  lastLabel.textContent = String(series[n - 1]);
  svg.appendChild(lastLabel);

  const xLabel = svgEl("text", {
    x: pad.l + innerW / 2, y: height - 4, "text-anchor": "middle",
    "font-size": 10, fill: "var(--muted, #666)",
  });
  xLabel.textContent = "round";
  svg.appendChild(xLabel);

  return svg;
}

// Render `container` with one small labeled chart per fixed-point loop found
// in `trace`, side by side (wraps on narrow pages). Clears `container` first;
// shows a one-line placeholder if the trace has no NodeCount events (e.g. it
// predates this feature, or profiling was off).
export function renderNodeReduction(container, trace) {
  container.innerHTML = "";
  const counts = extractNodeCounts(trace);
  if (counts.size === 0) {
    const p = document.createElement("p");
    p.textContent = "(no NodeCount events in this trace)";
    container.appendChild(p);
    return;
  }

  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexWrap = "wrap";
  wrap.style.gap = "12px";

  for (const loop of orderedLoops(counts)) {
    const series = counts.get(loop);
    const card = document.createElement("div");
    card.style.border = "1px solid var(--border, #ccc)";
    card.style.borderRadius = "4px";
    card.style.padding = "8px";

    const converged =
      series.length >= 2 && series[series.length - 1] === series[series.length - 2];
    const status = series.length < 2 ? "" : converged ? " (converged)" : " (hit round cap)";
    const title = document.createElement("div");
    title.textContent = `${loop}: ${series.length} round(s)${status}`;
    title.style.font = "bold 12px sans-serif";
    title.style.marginBottom = "4px";
    card.appendChild(title);

    card.appendChild(renderOneChart(series));
    wrap.appendChild(card);
  }
  container.appendChild(wrap);
}
