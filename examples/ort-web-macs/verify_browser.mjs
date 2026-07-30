// Headless verification of index.html (dev-only, not shipped).
//
// jsDelivr is blocked by egress policy in the build sandbox, so this test
// serves index.html + onnx-macs.js + the sample model from a local HTTP server
// and *routes* the CDN's onnxruntime-web requests to the local node_modules
// copy. The page itself is loaded unmodified. In a normal browser the CDN load
// works directly and no routing is needed.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

const DIR = path.resolve(".");
const DIST = path.join(DIR, "node_modules", "onnxruntime-web", "dist");
const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".onnx": "application/octet-stream", ".wasm": "application/wasm",
};
const server = http.createServer((req, res) => {
  const p = path.join(DIR, decodeURIComponent(req.url.split("?")[0]));
  if (!p.startsWith(DIR) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
    res.writeHead(404); return res.end("nf");
  }
  res.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(8099, r));

const browser = await chromium.launch({
  // Uses Playwright's own Chromium after `npx playwright install`. Set
  // CHROMIUM_PATH to point at an already-installed browser instead.
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ["--no-sandbox"],
});
const ctx = await browser.newContext();
const page = await ctx.newPage();

// Fulfil the CDN's onnxruntime-web requests from the local package.
await ctx.route("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/**", async (route) => {
  const url = new URL(route.request().url());
  const file = path.join(DIST, path.basename(url.pathname));
  if (!fs.existsSync(file)) return route.fulfill({ status: 404, body: "nf" });
  route.fulfill({
    status: 200,
    contentType: MIME[path.extname(file)] || "application/octet-stream",
    body: fs.readFileSync(file),
  });
});

page.on("console", (m) => console.log(`[console.${m.type()}]`, m.text()));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

await page.goto("http://localhost:8099/index.html", { waitUntil: "load", timeout: 60000 });
console.log("title:", await page.title());

await page.click("#loadSample");
await page.waitForFunction(() => !document.getElementById("results").classList.contains("hidden"), { timeout: 60000 });
console.log("status after load:", await page.textContent("#status"));

await page.fill("#runs", "30");
await page.click("#run");
await page.waitForFunction(() => (document.getElementById("status").textContent || "").startsWith("Done"), { timeout: 60000 });
console.log("status after run:", await page.textContent("#status"));

console.log("MACs card:", (await page.textContent("#cards .card.hl .v")).replace(/\s+/g, " ").trim());
const perf = await page.$$eval("#perfcards .card", (cs) => cs.map((c) => c.textContent.replace(/\s+/g, " ").trim()));
console.log("perf cards:", perf);
const ops = await page.$$eval("#opTable tbody tr", (rs) => rs.map((r) => r.textContent.replace(/\s+/g, " ").trim()));
console.log("op rows:", ops);

await page.screenshot({ path: "demo_screenshot.png", fullPage: true });
console.log("screenshot written");
await browser.close();
server.close();
