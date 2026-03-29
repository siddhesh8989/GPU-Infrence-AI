import { Router, type IRouter } from "express";
import multer from "multer";
import { randomUUID } from "crypto";
import { runInference, runBatchBenchmark } from "../services/inferenceEngine.js";
import { recordInference } from "../services/metricsStore.js";

const router: IRouter = Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed"));
    }
  },
});

router.post("/upload", upload.single("file"), (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "No file uploaded", message: "Please provide an image file" });
    return;
  }

  const fileId = randomUUID();
  const estimatedWidth = Math.floor(200 + Math.random() * 800);
  const estimatedHeight = Math.floor(200 + Math.random() * 600);

  res.json({
    fileId,
    filename: req.file.originalname,
    fileSize: req.file.size,
    dimensions: {
      width: estimatedWidth,
      height: estimatedHeight,
    },
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

  res.json({
    model,
    computeMode,
    precision,
    results,
    summary,
  });
});

export default router;
