import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const engineRoot = path.join(projectRoot, "auto research");
const runsDir = path.join(engineRoot, "runs");

function readJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function readJsonl(filePath) {
  try {
    return fs
      .readFileSync(filePath, "utf8")
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch {
    return [];
  }
}

function readText(filePath, fallback = "") {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return fallback;
  }
}

function buildPayload() {
  const rows = readJsonl(path.join(runsDir, "research_log.jsonl"));
  const best = readJson(path.join(runsDir, "best_pipeline_config.json"), null);
  const program = readText(path.join(engineRoot, "program.md"));
  const bestReward = best?.reward ?? null;
  const observedBestReward = rows.reduce((max, row) => Math.max(max, row.reward ?? Number.NEGATIVE_INFINITY), Number.NEGATIVE_INFINITY);
  const lossReference = bestReward ?? (Number.isFinite(observedBestReward) ? observedBestReward : 0);

  let runningBest = Number.NEGATIVE_INFINITY;
  let bestIndex = null;
  let bestSignature = null;
  const lineage = [];
  const events = rows.map((row, index) => {
    const previousBest = Number.isFinite(runningBest) ? runningBest : null;
    const previousBestIndex = bestIndex;
    const previousBestSignature = bestSignature;
    const isAccepted = Number.isFinite(row.reward) && row.reward > runningBest;
    if (isAccepted) {
      runningBest = row.reward;
      bestIndex = index + 1;
      bestSignature = row.signature ?? null;
    }
    const event = {
      ...row,
      index: index + 1,
      validationLoss: lossReference - (row.reward ?? 0),
      parentLoss: previousBest === null ? null : lossReference - previousBest,
      parentBest: previousBest,
      parentIndex: previousBestIndex,
      parentSignature: previousBestSignature,
      deltaFromParent: previousBest === null ? null : (row.reward ?? 0) - previousBest,
      deltaLossFromParent: previousBest === null ? null : (lossReference - (row.reward ?? 0)) - (lossReference - previousBest),
      acceptedByLog: isAccepted,
      runningBest: Number.isFinite(runningBest) ? runningBest : row.reward ?? 0,
      runningBestLoss: Number.isFinite(runningBest) ? lossReference - runningBest : lossReference - (row.reward ?? 0),
      isCurrentBest: bestReward !== null && Math.abs((row.reward ?? 0) - bestReward) < 1e-9,
    };
    if (isAccepted) lineage.push(event);
    return event;
  });

  const sorted = [...events].sort((a, b) => (b.reward ?? 0) - (a.reward ?? 0));
  const modalCount = events.filter((row) => row.backend === "modal-cpu").length;
  const acceptedCount = events.filter((row) => row.acceptedByLog).length;
  const latest = events.at(-1) ?? null;

  return {
    generatedAt: new Date().toISOString(),
    program,
    best,
    lossReference,
    events,
    lineage,
    topCandidates: sorted.slice(0, 8),
    stats: {
      totalCandidates: events.length,
      modalCount,
      localCount: events.length - modalCount,
      acceptedCount,
      bestReward,
      bestIndex,
      nextRequiredReward: Number.isFinite(runningBest) ? runningBest : null,
      latestReward: latest?.reward ?? null,
      latestGap: latest?.generalization_gap ?? null,
      latestBackend: latest?.backend ?? "local",
    },
  };
}

function runsApiPlugin() {
  return {
    name: "runs-api",
    configureServer(server) {
      server.middlewares.use("/api/runs", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Cache-Control", "no-store");
        res.end(JSON.stringify(buildPayload()));
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), runsApiPlugin()],
  server: {
    port: 5173,
    strictPort: false,
  },
});
