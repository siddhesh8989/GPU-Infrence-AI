import { Router, type IRouter } from "express";
import multer from "multer";
import { randomUUID } from "crypto";
import { mkdirSync } from "fs";
import { join, extname } from "path";
import { runInference, runBatchBenchmark } from "../services/inferenceEngine.js";
import { recordInference } from "../services/metricsStore.js";
import { maybeEnhance } from "../services/enhancementEngine.js";

const router: IRouter = Router();

const UPLOADS_DIR = join(process.cwd(), "../../artifacts/gpu-inference/uploads");
try { mkdirSync(UPLOADS_DIR, { recursive: true }); } catch {}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOADS_DIR),
  filename: (_req, file, cb) => {
    const id = randomUUID();
    const ext = extname(file.originalname) || ".jpg";
    cb(null, `${id}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/avif", "image/heic", "image/heif"];
    if (allowed.includes(file.mimetype) || file.mimetype.startsWith("image/")) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed"));
    }
  },
});

const filePathCache = new Map<string, string>();

const debugStore = new Map<string, object>();

router.post("/upload", upload.single("file"), (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "No file uploaded", message: "Please provide an image file" });
    return;
  }

  const filename = req.file.filename;
  const fileId = filename.replace(/\.[^.]+$/, "");
  filePathCache.set(fileId, req.file.path);

  res.json({
    fileId,
    filename: req.file.originalname,
    fileSize: req.file.size,
    dimensions: { width: 224, height: 224 },
    uploadedAt: new Date().toISOString(),
  });
});

router.post("/predict", async (req, res) => {
  const { fileId, model, computeMode, batchSize = 1, precision = "fp32" } = req.body as {
    fileId: string;
    model: string;
    computeMode: string;
    batchSize?: number;
    precision?: string;
  };

  if (!fileId || !model || !computeMode) {
    res.status(400).json({ error: "Missing required fields", message: "fileId, model, and computeMode are required" });
    return;
  }

  const validModels = ["resnet50", "mobilenet", "random_forest"];
  const validModes = ["cpu", "gpu"];
  const validPrecisions = ["fp32", "fp16"];
  const validBatchSizes = [1, 2, 4, 8, 16, 32];

  if (!validModels.includes(model)) {
    res.status(400).json({ error: "Invalid model", message: `Model must be one of: ${validModels.join(", ")}` });
    return;
  }
  if (!validModes.includes(computeMode)) {
    res.status(400).json({ error: "Invalid compute mode", message: "computeMode must be 'cpu' or 'gpu'" });
    return;
  }
  if (!validBatchSizes.includes(Number(batchSize))) {
    res.status(400).json({ error: "Invalid batch size", message: `batchSize must be one of: ${validBatchSizes.join(", ")}` });
    return;
  }
  if (!validPrecisions.includes(precision)) {
    res.status(400).json({ error: "Invalid precision", message: "precision must be 'fp32' or 'fp16'" });
    return;
  }

  const result = await runInference({
    fileId,
    model,
    computeMode,
    batchSize: Number(batchSize),
    precision,
  });

  const imagePath = filePathCache.get(fileId) ?? null;
  const enhancement = await maybeEnhance({
    imagePath,
    originalPrediction: result.prediction,
    originalConfidence: result.confidence,
    topPredictions: result.topPredictions,
    seed: result.timings.totalMs * 1000,
  });

  if (enhancement.enhancementUsed) {
    const extra = enhancement.extraPostprocMs;
    result.prediction    = enhancement.prediction;
    result.confidence    = enhancement.confidence;
    result.topPredictions = enhancement.topPredictions;
    result.timings.postprocessingMs = Math.round((result.timings.postprocessingMs + extra) * 10) / 10;
    result.timings.totalMs          = Math.round((result.timings.totalMs + extra) * 10) / 10;
    result.timings.throughputRps    = Math.round((1000 / result.timings.totalMs) * Number(batchSize) * 10) / 10;
    if (result.pipelineStages.length > 0) {
      const last = result.pipelineStages[result.pipelineStages.length - 1];
      last.durationMs = Math.round((last.durationMs + extra) * 10) / 10;
    }
  }

  debugStore.set(`${fileId}:${model}`, {
    fileId,
    model,
    enhancementUsed:     enhancement.enhancementUsed,
    originalPrediction:  enhancement.originalPrediction,
    originalConfidence:  enhancement.originalConfidence,
    enhancedPrediction:  enhancement.enhancementUsed ? enhancement.prediction : null,
    enhancedConfidence:  enhancement.enhancementUsed ? enhancement.confidence : null,
    confidenceDelta:     Math.round((enhancement.confidence - enhancement.originalConfidence) * 10000) / 10000,
    timestamp:           result.timestamp,
  });

  recordInference(result);

  res.json(result);
});

router.post("/batch", async (req, res) => {
  const { model, batchSizes, computeMode, precision = "fp32" } = req.body as {
    model: string;
    batchSizes: number[];
    computeMode: string;
    precision?: string;
  };

  if (!model || !batchSizes || !computeMode) {
    res.status(400).json({ error: "Missing required fields" });
    return;
  }

  if (!Array.isArray(batchSizes) || batchSizes.length === 0) {
    res.status(400).json({ error: "batchSizes must be a non-empty array" });
    return;
  }

  const results = await runBatchBenchmark({ model, batchSizes, computeMode, precision });

  const sorted = [...results].sort((a, b) => b.throughputRps - a.throughputRps);
  const best = sorted[0];

  const summary = {
    bestBatchSize: best?.batchSize ?? batchSizes[0],
    peakThroughput: best?.throughputRps ?? 0,
    minLatency: Math.min(...results.map((r) => r.latencyMs)),
    avgMemoryMb: Math.round(results.reduce((a, r) => a + r.memoryMb, 0) / results.length * 10) / 10,
  };

  res.json({ model, computeMode, precision, results, summary });
});

router.get("/debug/prediction", (req, res) => {
  const { fileId, model = "" } = req.query as { fileId?: string; model?: string };
  if (!fileId) {
    res.status(400).json({ error: "fileId query parameter required" });
    return;
  }
  const entry = debugStore.get(`${fileId}:${model}`);
  if (!entry) {
    res.status(404).json({ error: "No debug data for this fileId/model combination" });
    return;
  }
  res.json(entry);
});

export default router;
