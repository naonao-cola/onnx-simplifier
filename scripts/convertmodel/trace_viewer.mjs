// A tiny, dependency-free Chrome Trace Event viewer for the converter page.
//
// Both traces this page shows -- onnxsim's C++ simplification profiler and the
// onnxruntime-web inference profiler -- are Chrome Trace Event Format JSON
// ({ traceEvents: [ ... ] } with "X" complete events and "M" thread_name
// metadata). This renders that format inline as a zoomable flame graph on a
// <canvas> for a fast, offline-friendly first look, and can embed the full
// Perfetto UI (https://ui.perfetto.dev) right in the page for deeper analysis.
//
// renderTrace(container, trace, opts) clears `container` and mounts:
//   * a small toolbar (download JSON + embed-in-Perfetto),
//   * the flame-graph canvas (wheel to zoom at the cursor, drag to pan,
//     double-click to reset),
//   * a hover tooltip with each span's name, duration and args,
//   * a slot below into which the Perfetto UI is embedded on demand.
//
// Perfetto is embedded following its official protocol
// (https://perfetto.dev/docs/visualization/embedding-the-ui): an iframe at
// ui.perfetto.dev/#!/?mode=embedded, a PING/PONG readiness handshake, then a
// { perfetto: { buffer, title, ... } } postMessage carrying the trace bytes.

const PERFETTO_ORIGIN = "https://ui.perfetto.dev";
// `mode=embedded` hides Perfetto's sidebar for a clean in-page view.
const PERFETTO_EMBED_SRC = "https://ui.perfetto.dev/#!/?mode=embedded";
const ROW_H = 18; // px per stacked span row
const TRACK_PAD = 6; // px above each track label
const LABEL_H = 16; // px for a track's name row

// Distinct-but-stable colors per category/name, hashed so the same op keeps its
// hue across renders.
function colorFor(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  const hue = ((h % 360) + 360) % 360;
  return `hsl(${hue} 55% 60%)`;
}

function fmtUs(us) {
  if (us >= 1000) return `${(us / 1000).toFixed(3)} ms`;
  return `${us.toFixed(1)} µs`;
}

// Group "X" events into tracks keyed by (pid, tid), assign each event a stacking
// lane within its track (via a containment stack), and read "M" thread_name
// metadata for labels. Returns { tracks, tMin, tMax }.
function layout(trace) {
  const events = (trace && trace.traceEvents) || [];
  const names = new Map(); // "pid/tid" -> label
  const byTrack = new Map(); // "pid/tid" -> [event]
  let tMin = Infinity;
  let tMax = -Infinity;

  for (const e of events) {
    if (e.ph === "M" && e.name === "thread_name") {
      names.set(`${e.pid}/${e.tid}`, (e.args && e.args.name) || `tid ${e.tid}`);
      continue;
    }
    if (e.ph !== "X") continue;
    const key = `${e.pid}/${e.tid}`;
    if (!byTrack.has(key)) byTrack.set(key, []);
    byTrack.get(key).push(e);
    tMin = Math.min(tMin, e.ts);
    tMax = Math.max(tMax, e.ts + (e.dur || 0));
  }
  if (!isFinite(tMin)) {
    tMin = 0;
    tMax = 1;
  }
  if (tMax <= tMin) tMax = tMin + 1;

  const tracks = [];
  for (const [key, evs] of byTrack) {
    // Sort by start asc, then by duration desc so parents precede children, and
    // assign lanes with a stack: an event nests under the last open one that
    // still contains it.
    evs.sort((a, b) => a.ts - b.ts || (b.dur || 0) - (a.dur || 0));
    const open = []; // lane index -> event end time
    let maxLane = 0;
    for (const e of evs) {
      const end = e.ts + (e.dur || 0);
      // Prefer the provided depth (onnxsim sets args.depth) but fall back to a
      // containment lane so any well-formed trace stacks correctly.
      let lane = e.args && Number.isInteger(e.args.depth) ? e.args.depth : -1;
      if (lane < 0) {
        lane = 0;
        while (lane < open.length && open[lane] > e.ts) lane++;
      }
      open[lane] = end;
      e.__lane = lane;
      maxLane = Math.max(maxLane, lane);
    }
    tracks.push({
      key,
      label: names.get(key) || key,
      events: evs,
      lanes: maxLane + 1,
    });
  }
  // Stable order: onnxsim's process track ids are small; keep insertion order.
  return { tracks, tMin, tMax };
}

function makeButton(label) {
  const b = document.createElement("button");
  b.textContent = label;
  b.style.marginRight = "8px";
  return b;
}

// Embed the Perfetto UI in `slot` (an iframe) and load `trace` into it. Follows
// the official embedding protocol: create the iframe, PING its contentWindow
// until it answers PONG (its message listener is then registered), then post
// the trace bytes. `keepApiOpen` lets us swap in a new trace later without
// reloading the iframe. Reuses an already-ready iframe on repeat calls.
function embedInPerfetto(slot, trace, title) {
  const buffer = new TextEncoder().encode(JSON.stringify(trace)).buffer;
  const post = (win) =>
    win.postMessage(
      {
        perfetto: {
          buffer,
          title,
          fileName: `${title}.json`,
          keepApiOpen: true,
        },
      },
      PERFETTO_ORIGIN,
    );

  let iframe = slot.querySelector("iframe.perfetto");
  if (iframe && iframe.__ready) {
    post(iframe.contentWindow);
    return;
  }
  if (!iframe) {
    iframe = document.createElement("iframe");
    iframe.className = "perfetto";
    iframe.src = PERFETTO_EMBED_SRC;
    Object.assign(iframe.style, {
      width: "100%",
      height: "600px",
      border: "1px solid #ccc",
      marginTop: "6px",
    });
    slot.innerHTML = "";
    slot.appendChild(iframe);
  }

  // Only the target iframe's PONG counts (other embeds post PONGs too).
  const onMessage = (evt) => {
    if (evt.source !== iframe.contentWindow || evt.data !== "PONG") return;
    clearInterval(timer);
    window.removeEventListener("message", onMessage);
    iframe.__ready = true;
    post(iframe.contentWindow);
  };
  window.addEventListener("message", onMessage);
  const timer = setInterval(() => {
    if (iframe.contentWindow) iframe.contentWindow.postMessage("PING", PERFETTO_ORIGIN);
  }, 250);
  // Give up pinging after ~20s so we do not leak the interval.
  setTimeout(() => {
    clearInterval(timer);
    window.removeEventListener("message", onMessage);
  }, 20000);
}

function download(trace, filename) {
  const blob = new Blob([JSON.stringify(trace)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Render `trace` into `container`. opts: { title, filename }.
export function renderTrace(container, trace, opts = {}) {
  const title = opts.title || "trace";
  const filename = opts.filename || "trace.json";
  container.innerHTML = "";

  const { tracks, tMin, tMax } = layout(trace);
  if (tracks.length === 0) {
    const p = document.createElement("p");
    p.textContent = "(no timed spans in this trace)";
    container.appendChild(p);
    return;
  }

  // Toolbar.
  const bar = document.createElement("div");
  bar.style.margin = "6px 0";
  const dlBtn = makeButton("Download JSON");
  const pfBtn = makeButton("Embed in Perfetto");
  dlBtn.addEventListener("click", () => download(trace, filename));
  bar.appendChild(dlBtn);
  bar.appendChild(pfBtn);
  const hint = document.createElement("span");
  hint.style.color = "#666";
  hint.style.fontSize = "12px";
  hint.textContent = "wheel = zoom · drag = pan · double-click = reset";
  bar.appendChild(hint);
  container.appendChild(bar);

  // Slot the Perfetto UI is embedded into on demand; the button toggles it.
  const perfettoSlot = document.createElement("div");
  pfBtn.addEventListener("click", () => {
    if (perfettoSlot.querySelector("iframe.perfetto")) {
      perfettoSlot.innerHTML = "";
      pfBtn.textContent = "Embed in Perfetto";
    } else {
      embedInPerfetto(perfettoSlot, trace, title);
      pfBtn.textContent = "Hide Perfetto";
    }
  });

  // Canvas sized to fit every track's lanes.
  const totalH = tracks.reduce(
    (h, t) => h + LABEL_H + TRACK_PAD + t.lanes * ROW_H + TRACK_PAD,
    0,
  );
  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.border = "1px solid #ccc";
  wrap.style.overflow = "hidden";
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = `${totalH}px`;
  canvas.style.display = "block";
  wrap.appendChild(canvas);
  const tip = document.createElement("div");
  Object.assign(tip.style, {
    position: "absolute",
    pointerEvents: "none",
    background: "rgba(0,0,0,0.85)",
    color: "#fff",
    font: "12px monospace",
    padding: "4px 6px",
    borderRadius: "3px",
    whiteSpace: "pre",
    display: "none",
    zIndex: "10",
  });
  wrap.appendChild(tip);
  container.appendChild(wrap);
  container.appendChild(perfettoSlot);

  // View state: [viewMin, viewMax] in trace time (µs).
  let viewMin = tMin;
  let viewMax = tMax;
  const span = () => viewMax - viewMin;

  const ctx = canvas.getContext("2d");
  let cssW = 0;
  const dpr = window.devicePixelRatio || 1;

  function resize() {
    cssW = wrap.clientWidth || 800;
    canvas.width = Math.max(1, Math.floor(cssW * dpr));
    canvas.height = Math.max(1, Math.floor(totalH * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  const xOf = (t) => ((t - viewMin) / span()) * cssW;

  function draw() {
    ctx.clearRect(0, 0, cssW, totalH);
    ctx.textBaseline = "middle";
    let y = 0;
    for (const track of tracks) {
      // Track label.
      ctx.fillStyle = "#333";
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(track.label, 4, y + LABEL_H / 2);
      y += LABEL_H + TRACK_PAD;

      ctx.font = "11px sans-serif";
      for (const e of track.events) {
        const x0 = xOf(e.ts);
        const x1 = xOf(e.ts + (e.dur || 0));
        if (x1 < 0 || x0 > cssW) continue; // off-screen
        const w = Math.max(1, x1 - x0);
        const ry = y + e.__lane * ROW_H;
        ctx.fillStyle = colorFor(e.cat ? `${e.cat}/${e.name}` : e.name);
        ctx.fillRect(x0, ry, w, ROW_H - 1);
        if (w > 24) {
          ctx.fillStyle = "#111";
          ctx.save();
          ctx.beginPath();
          ctx.rect(x0 + 2, ry, w - 4, ROW_H - 1);
          ctx.clip();
          ctx.fillText(e.name, x0 + 3, ry + (ROW_H - 1) / 2);
          ctx.restore();
        }
      }
      y += track.lanes * ROW_H + TRACK_PAD;
    }
  }

  // Hit-test the cursor against drawn spans.
  function hit(px, py) {
    let y = 0;
    for (const track of tracks) {
      y += LABEL_H + TRACK_PAD;
      for (const e of track.events) {
        const ry = y + e.__lane * ROW_H;
        if (py < ry || py > ry + ROW_H - 1) continue;
        const x0 = xOf(e.ts);
        const x1 = xOf(e.ts + (e.dur || 0));
        if (px >= x0 && px <= Math.max(x0 + 1, x1)) return e;
      }
      y += track.lanes * ROW_H + TRACK_PAD;
    }
    return null;
  }

  // Interaction: wheel zooms toward the cursor, drag pans, dbl-click resets.
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const tAt = viewMin + (px / cssW) * span();
    const factor = ev.deltaY < 0 ? 0.8 : 1.25;
    let newSpan = span() * factor;
    newSpan = Math.min(tMax - tMin, Math.max(1, newSpan));
    let lo = tAt - (px / cssW) * newSpan;
    let hi = lo + newSpan;
    if (lo < tMin) {
      lo = tMin;
      hi = lo + newSpan;
    }
    if (hi > tMax) {
      hi = tMax;
      lo = hi - newSpan;
    }
    viewMin = Math.max(tMin, lo);
    viewMax = Math.min(tMax, hi);
    draw();
  }, { passive: false });

  let dragging = false;
  let lastX = 0;
  canvas.addEventListener("mousedown", (ev) => {
    dragging = true;
    lastX = ev.clientX;
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
  canvas.addEventListener("mousemove", (ev) => {
    const rect = canvas.getBoundingClientRect();
    if (dragging) {
      const dt = ((ev.clientX - lastX) / cssW) * span();
      lastX = ev.clientX;
      let lo = viewMin - dt;
      let hi = viewMax - dt;
      if (lo < tMin) {
        lo = tMin;
        hi = lo + span();
      }
      if (hi > tMax) {
        hi = tMax;
        lo = hi - span();
      }
      viewMin = lo;
      viewMax = hi;
      draw();
      tip.style.display = "none";
      return;
    }
    const px = ev.clientX - rect.left;
    const py = ev.clientY - rect.top;
    const e = hit(px, py);
    if (!e) {
      tip.style.display = "none";
      return;
    }
    const lines = [`${e.name}`, `dur: ${fmtUs(e.dur || 0)}`];
    if (e.args) {
      for (const [k, v] of Object.entries(e.args)) {
        if (v === undefined || v === null) continue;
        lines.push(`${k}: ${v}`);
      }
    }
    tip.textContent = lines.join("\n");
    tip.style.display = "block";
    tip.style.left = `${Math.min(px + 12, wrap.clientWidth - 220)}px`;
    tip.style.top = `${py + 12}px`;
  });
  canvas.addEventListener("mouseleave", () => {
    tip.style.display = "none";
  });
  canvas.addEventListener("dblclick", () => {
    viewMin = tMin;
    viewMax = tMax;
    draw();
  });

  const ro = new ResizeObserver(() => resize());
  ro.observe(wrap);
  resize();
}
