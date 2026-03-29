import { randomUUID, createHash } from "crypto";
import os from "os";

function fileIdToSeed(fileId: string): number {
  const hash = createHash("md5").update(fileId).digest("hex");
  return parseInt(hash.slice(0, 8), 16);
}

const IMAGENET_CLASSES = [
  "golden retriever", "tabby cat", "African elephant", "bald eagle", "great white shark",
  "monarch butterfly", "red fox", "giant panda", "snow leopard", "hummingbird",
  "sunflower", "rose", "oak tree", "mushroom", "coral reef",
  "sports car", "bicycle", "sailboat", "helicopter", "steam locomotive",
  "pizza", "sushi", "hamburger", "apple", "banana",
  "laptop computer", "smartphone", "digital camera", "headphones", "microphone",
  "bookshelf", "armchair", "dining table", "bathtub", "grand piano",
  "basketball", "tennis racket", "soccer ball", "golf club", "surfboard",
  "lighthouse", "suspension bridge", "Eiffel Tower", "volcano", "waterfall",
  "German shepherd", "Siamese cat", "Bengal tiger", "blue whale", "flamingo",
];

const RANDOM_FOREST_CLASSES = [
  "Class A: High confidence feature", "Class B: Medium gradient pattern",
  "Class C: Low frequency texture", "Class D: Complex multi-modal signal",
  "Class E: Sparse representation", "Class F: Dense clustering pattern",
];

export interface PipelineStage {
  name: string;
  durationMs: number;
  status: "pending" | "running" | "complete";
}

export interface InferenceTimings {
  preprocessingMs: number;
  batchingMs: number;
  inferenceMs: number;
  postprocessingMs: number;
  totalMs: number;
  throughputRps: number;
}

export interface MemoryUsage {
  heapUsedMb: number;
  heapTotalMb: number;
  rssMb: number;
  gpuSimulatedMb: number;
}

export interface Prediction {
  label: string;
  confidence: number;
  rank: number;
}

export interface InferenceOutput {
  id: string;
  fileId: string;
  model: string;
  computeMode: string;
  batchSize: number;
  precision: string;
  prediction: string;
  confidence: number;
  topPredictions: Prediction[];
  timings: InferenceTimings;
  memoryUsage: MemoryUsage;
  pipelineStages: PipelineStage[];
  timestamp: string;
}

export interface BatchResult {
  batchSize: number;
  latencyMs: number;
  throughputRps: number;
  memoryMb: number;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getBaseLatency(model: string): number {
  switch (model) {
    case "resnet50": return 85;
    case "mobilenet": return 32;
    case "random_forest": return 18;
    default: return 60;
  }
}

function simulateGpuSpeedup(cpuLatency: number, batchSize: number, precision: string): number {
  const parallelismGain = Math.log2(batchSize + 1) * 0.35;
  const precisionGain = precision === "fp16" ? 0.15 : 0;
  const gpuCoreGain = 0.55 + parallelismGain + precisionGain;
  const gpuLatency = cpuLatency * (1 - gpuCoreGain);
  return Math.max(gpuLatency, cpuLatency * 0.12);
}

function seededRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function generateTopPredictions(model: string, seed: number): Prediction[] {
  const classes = model === "random_forest" ? RANDOM_FOREST_CLASSES : IMAGENET_CLASSES;
  const topK = model === "random_forest" ? 3 : 5;
  const rng = seededRng(seed);

  const shuffled = [...classes];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const selected = shuffled.slice(0, topK);

  let remaining = 1.0;
  const predictions: Prediction[] = [];

  for (let i = 0; i < selected.length; i++) {
    const isLast = i === selected.length - 1;
    const conf = isLast ? remaining : remaining * (0.4 + rng() * 0.3) * (1 - i * 0.05);
    const rounded = Math.min(Math.round(conf * 1000) / 1000, remaining);
    predictions.push({ label: selected[i], confidence: rounded, rank: i + 1 });
    remaining -= rounded;
    if (remaining <= 0) break;
  }

  return predictions.sort((a, b) => b.confidence - a.confidence);
}

function getMemoryUsage(computeMode: string, model: string, batchSize: number): MemoryUsage {
  const mem = process.memoryUsage();
  const modelMemory: Record<string, number> = {
    resnet50: 97.8,
    mobilenet: 13.5,
    random_forest: 2.4,
  };
  const modelMem = modelMemory[model] ?? 50;

  return {
    heapUsedMb: Math.round(mem.heapUsed / 1024 / 1024 * 10) / 10,
    heapTotalMb: Math.round(mem.heapTotal / 1024 / 1024 * 10) / 10,
    rssMb: Math.round(mem.rss / 1024 / 1024 * 10) / 10,
    gpuSimulatedMb: computeMode === "gpu" ? Math.round((modelMem * 1.4 + batchSize * 2.1) * 10) / 10 : 0,
  };
}

export async function runInference(params: {
  fileId: string;
  model: string;
  computeMode: string;
  batchSize: number;
  precision: string;
}): Promise<InferenceOutput> {
  const { fileId, model, computeMode, batchSize, precision } = params;
  const id = randomUUID();
  const start = Date.now();

  const baseCpuLatency = getBaseLatency(model);
  const jitter = (Math.random() - 0.5) * 10;

  let inferenceLatency: number;
  if (computeMode === "gpu") {
    inferenceLatency = simulateGpuSpeedup(baseCpuLatency, batchSize, precision);
  } else {
    inferenceLatency = baseCpuLatency + jitter + batchSize * 4.2;
  }

  const preprocessingMs = computeMode === "gpu" ? 3.5 + Math.random() * 2 : 8 + Math.random() * 4;
  const batchingMs = batchSize > 1 ? (computeMode === "gpu" ? 1.2 : 2.8) + Math.random() : 0;
  const inferenceMs = inferenceLatency + jitter;
  const postprocessingMs = computeMode === "gpu" ? 1.5 + Math.random() : 3 + Math.random() * 2;
  const totalMs = preprocessingMs + batchingMs + inferenceMs + postprocessingMs;

  // Realistic wall-clock delay: GPU is 10–30 ms, CPU is 120–180 ms
  if (computeMode === "gpu") {
    await sleep(10 + Math.floor(Math.random() * 20));
  } else {
    await sleep(120 + Math.floor(Math.random() * 60));
  }

  const topPredictions = generateTopPredictions(model, fileIdToSeed(fileId));
  const throughputRps = Math.round((1000 / totalMs) * batchSize * 10) / 10;

  const pipelineStages: PipelineStage[] = [
    { name: "Input Loading", durationMs: Math.round(preprocessingMs * 0.3 * 10) / 10, status: "complete" },
    { name: "Preprocessing", durationMs: Math.round(preprocessingMs * 0.7 * 10) / 10, status: "complete" },
    { name: "Batch Scheduling", durationMs: Math.round(batchingMs * 10) / 10, status: "complete" },
    { name: "Model Inference", durationMs: Math.round(inferenceMs * 10) / 10, status: "complete" },
    { name: "Postprocessing", durationMs: Math.round(postprocessingMs * 10) / 10, status: "complete" },
  ];

  return {
    id,
    fileId,
    model,
    computeMode,
    batchSize,
    precision,
    prediction: topPredictions[0]?.label ?? "unknown",
    confidence: topPredictions[0]?.confidence ?? 0,
    topPredictions,
    timings: {
      preprocessingMs: Math.round(preprocessingMs * 10) / 10,
      batchingMs: Math.round(batchingMs * 10) / 10,
      inferenceMs: Math.round(inferenceMs * 10) / 10,
      postprocessingMs: Math.round(postprocessingMs * 10) / 10,
      totalMs: Math.round(totalMs * 10) / 10,
      throughputRps,
    },
    memoryUsage: getMemoryUsage(computeMode, model, batchSize),
    pipelineStages,
    timestamp: new Date().toISOString(),
  };
}

export async function runBatchBenchmark(params: {
  model: string;
  batchSizes: number[];
  computeMode: string;
  precision: string;
}): Promise<BatchResult[]> {
  const { model, batchSizes, computeMode, precision } = params;
  const results: BatchResult[] = [];

  for (const batchSize of batchSizes) {
    const baseCpuLatency = getBaseLatency(model);
    const jitter = (Math.random() - 0.5) * 5;

    let latencyMs: number;
    if (computeMode === "gpu") {
      latencyMs = simulateGpuSpeedup(baseCpuLatency, batchSize, precision) + jitter;
    } else {
      latencyMs = baseCpuLatency + jitter + batchSize * 4.2;
    }

    latencyMs = Math.max(latencyMs, 2);
    const preprocessingMs = computeMode === "gpu" ? 3.5 : 8;
    const totalMs = latencyMs + preprocessingMs;
    const throughputRps = Math.round((1000 / totalMs) * batchSize * 10) / 10;
    const modelMemory: Record<string, number> = { resnet50: 97.8, mobilenet: 13.5, random_forest: 2.4 };
    const memoryMb = (modelMemory[model] ?? 50) + batchSize * 1.8;

    results.push({
      batchSize,
      latencyMs: Math.round(latencyMs * 10) / 10,
      throughputRps,
      memoryMb: Math.round(memoryMb * 10) / 10,
    });

    await sleep(30);
  }

  return results;
}

export function getSystemInfo() {
  const cpus = os.cpus();
  return {
    nodeVersion: process.version,
    platform: os.platform(),
    cpuCores: cpus.length,
    totalMemoryMb: Math.round(os.totalmem() / 1024 / 1024),
    gpuAvailable: false,
    uptime: Math.round(process.uptime()),
  };
}
