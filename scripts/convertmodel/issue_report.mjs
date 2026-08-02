// Builds the pre-filled "Report an issue" URL for the WASM converter page.
//
// Kept as a pure, dependency-free module (no DOM access) so the length-capping
// logic can be unit-tested. index.html gathers the page/run values from the DOM
// and hands them here.
//
// GitHub rejects an issue-creation URL whose total length exceeds ~8 KB with
// "Whoa there! Your request URL is too long.". The console log — the one field
// with no natural bound — is URL-encoded into the query string, and encoding
// can several-fold inflate its length (a single newline becomes "%0A"). So we
// budget the *whole encoded URL* against a conservative cap and trim the log
// from its front (keeping the tail, which usually holds the error) until the URL
// fits, leaving a note that the full log should be attached via "Download log".

// A conservative ceiling for the whole issue URL. GitHub's hard limit is ~8192
// bytes; staying well under it leaves room for the browser and any redirect.
export const DEFAULT_MAX_URL_LENGTH = 6000;

const NO_LOG_PLACEHOLDER = "(no console output)";
const OMITTED_PLACEHOLDER =
  "(console output omitted here — click \"Download log\" above and attach the file)";
const TRUNCATED_PREFIX =
  "...(truncated — click \"Download log\" above and attach the full log)...";

// The fixed (non-log) lines of the issue body, built from the page context and
// the most recent conversion parameters (`runInfo`, may be null).
function headerLines({ pageUrl, userAgent, runInfo, versions }) {
  const lines = [];
  lines.push("**Describe the bug**");
  lines.push("<!-- A clear and concise description of what went wrong. -->");
  lines.push("");
  lines.push("**Model**");
  lines.push("<!-- Please attach the model or a download link so the problem can be reproduced,");
  lines.push("or send it to daquexian566@gmail.com -->");
  lines.push("");
  lines.push("**Environment (WASM converter)**");
  lines.push("- Page: " + (pageUrl || "(unknown)"));
  lines.push("- User agent: " + (userAgent || "(unknown)"));
  // Library versions baked into the module (from onnxsim_versions), so the
  // report says exactly which onnxsim / onnx-optimizer / onnx / protobuf ran.
  if (versions) {
    lines.push("- onnxsim: " + (versions.onnxsim || "unknown"));
    lines.push("- onnx-optimizer: " + (versions.onnx_optimizer || "unknown"));
    lines.push("- onnx: " + (versions.onnx || "unknown"));
    lines.push("- protobuf: " + (versions.protobuf || "unknown"));
  }
  if (runInfo) {
    lines.push("- Optimizer: " + runInfo.optimizer);
    lines.push("- File: " + runInfo.file_name);
    lines.push("- constant fold: " + runInfo.constant_fold);
    lines.push("- shape inference: " + runInfo.shape_inference);
    lines.push("- inline functions: " + runInfo.inline_functions);
    lines.push("- tensor size threshold: " + runInfo.tensor_size_threshold);
    lines.push("- target opset version: " + runInfo.target_opset_version + " (0 = keep)");
    if (runInfo.passes && runInfo.passes.length) {
      const label = runInfo.optimizer === "simplify" ? "Skipped passes" : "Target passes";
      lines.push("- " + label + ": " + runInfo.passes.join(", "));
    }
  } else {
    lines.push("- (no conversion has been run yet in this session)");
  }
  lines.push("");
  lines.push("**Console output**");
  lines.push("<!-- The tail is inlined below; use \"Download log\" above to attach the full log. -->");
  return lines;
}

// Assemble the full issue body for a given (already-trimmed) log body.
function bodyFor(opts, logBody) {
  return headerLines(opts)
    .concat(["```", logBody, "```"])
    .join("\n");
}

// The log block contents for a given trimmed log and whether it was trimmed.
function logBodyFor(log, truncated) {
  if (!log) return truncated ? OMITTED_PLACEHOLDER : NO_LOG_PLACEHOLDER;
  return truncated ? TRUNCATED_PREFIX + "\n" + log : log;
}

// Build the pre-filled issue URL, trimming the console log so the whole URL
// stays under `maxUrlLength`. `repo` is the repository base URL (no trailing
// slash), `title` the issue title, `log` the raw console output. Returns the URL
// string. The header/environment block is always kept in full; only the log is
// trimmed (from the front), so the error-bearing tail survives.
export function buildIssueUrl(opts) {
  const {
    repo,
    title = "",
    maxUrlLength = DEFAULT_MAX_URL_LENGTH,
  } = opts;
  const base = repo + "/issues/new?title=" + encodeURIComponent(title) + "&body=";
  const budget = maxUrlLength - base.length;

  let log = (opts.log || "").trim();
  let truncated = false;
  const encodedFor = () =>
    encodeURIComponent(bodyFor(opts, logBodyFor(log, truncated)));

  let encoded = encodedFor();
  while (encoded.length > budget && log.length > 0) {
    truncated = true;
    // Drop a chunk from the front (>=100 chars, or 10% of what's left) and keep
    // re-encoding until the tail fits. Trimming the front preserves the tail,
    // where the error message usually is.
    const drop = Math.max(100, Math.floor(log.length * 0.1));
    log = log.slice(drop).replace(/^[^\n]*\n/, ""); // align to a line boundary
    encoded = encodedFor();
  }
  return base + encoded;
}
