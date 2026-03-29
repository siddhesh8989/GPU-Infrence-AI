import type { InferenceOutput } from "./inferenceEngine.js";

interface StoredInference {
  id: string;
  fileId: string;
  model: string;
  computeMode: string;
  batchSize: number;
  precision: string;
  prediction: string;
  confidence: number;
  latencyMs: number;
  throughputRps: number;
  timestamp: Date;
}

const store: StoredInference[] = [];

export function recordInference(result: InferenceOutput): void {
  store.push({
    id: result.id,
    fileId: result.fileId,
    model: result.model,
    computeMode: result.computeMode,
    batchSize: result.batchSize,
    precision: result.precision,
    prediction: result.prediction,
    confidence: result.confidence,
    latencyMs: result.timings.totalMs,
    throughputRps: result.timings.throughputRps,
    timestamp: new Date(result.timestamp),
  });
  if (store.length > 500) store.shift();
}

export function getMetricsSummary() {
  if (store.length === 0) {
    return {
      totalInferences: 0,
      avgLatencyMs: 0,
      avgThroughputRps: 0,
      cpuInferences: 0,
      gpuInferences: 0,
    };
  }

  const cpuItems = store.filter((s) => s.computeMode === "cpu");
  const gpuItems = store.filter((s) => s.computeMode === "gpu");
  const avgLatencyMs = store.reduce((a, s) => a + s.latencyMs, 0) / store.length;
  const avgThroughputRps = store.reduce((a, s) => a + s.throughputRps, 0) / store.length;

  return {
    totalInferences: store.length,
    avgLatencyMs: Math.round(avgLatencyMs * 10) / 10,
    avgThroughputRps: Math.round(avgThroughputRps * 10) / 10,
    cpuInferences: cpuItems.length,
    gpuInferences: gpuItems.length,
  };
}

export function getModelBreakdown() {
  const modelMap = new Map<string, { count: number; totalLatency: number; totalConf: number }>();

  for (const item of store) {
    const existing = modelMap.get(item.model) ?? { count: 0, totalLatency: 0, totalConf: 0 };
    modelMap.set(item.model, {
      count: existing.count + 1,
      totalLatency: existing.totalLatency + item.latencyMs,
      totalConf: existing.totalConf + item.confidence,
    });
  }

  return Array.from(modelMap.entries()).map(([model, data]) => ({
    model,
    count: data.count,
    avgLatencyMs: Math.round((data.totalLatency / data.count) * 10) / 10,
    avgConfidence: Math.round((data.totalConf / data.count) * 1000) / 1000,
  }));
}

export function getBatchPerformance() {
  const batchMap = new Map<number, { totalLatency: number; totalThroughput: number; count: number }>();

  for (const item of store) {
    const existing = batchMap.get(item.batchSize) ?? { totalLatency: 0, totalThroughput: 0, count: 0 };
    batchMap.set(item.batchSize, {
      totalLatency: existing.totalLatency + item.latencyMs,
      totalThroughput: existing.totalThroughput + item.throughputRps,
      count: existing.count + 1,
    });
  }

  return Array.from(batchMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([batchSize, data]) => ({
      batchSize,
      avgLatencyMs: Math.round((data.totalLatency / data.count) * 10) / 10,
      avgThroughputRps: Math.round((data.totalThroughput / data.count) * 10) / 10,
    }));
}

export function getComparisonData() {
  const models = ["resnet50", "mobilenet", "random_forest"];
  const latencyComparison = models
    .map((model) => {
      const cpuItems = store.filter((s) => s.model === model && s.computeMode === "cpu");
      const gpuItems = store.filter((s) => s.model === model && s.computeMode === "gpu");
      if (cpuItems.length === 0 || gpuItems.length === 0) return null;
      const cpuLatencyMs = cpuItems.reduce((a, s) => a + s.latencyMs, 0) / cpuItems.length;
      const gpuLatencyMs = gpuItems.reduce((a, s) => a + s.latencyMs, 0) / gpuItems.length;
      const improvement = ((cpuLatencyMs - gpuLatencyMs) / cpuLatencyMs) * 100;
      return {
        model,
        cpuLatencyMs: Math.round(cpuLatencyMs * 10) / 10,
        gpuLatencyMs: Math.round(gpuLatencyMs * 10) / 10,
        improvement: Math.round(improvement * 10) / 10,
      };
    })
    .filter(Boolean);

  const batchSizes = [1, 2, 4, 8, 16, 32];
  const throughputComparison = batchSizes
    .map((batchSize) => {
      const cpuItems = store.filter((s) => s.batchSize === batchSize && s.computeMode === "cpu");
      const gpuItems = store.filter((s) => s.batchSize === batchSize && s.computeMode === "gpu");
      if (cpuItems.length === 0 && gpuItems.length === 0) return null;
      const cpuThroughput = cpuItems.length > 0
        ? cpuItems.reduce((a, s) => a + s.throughputRps, 0) / cpuItems.length
        : 0;
      const gpuThroughput = gpuItems.length > 0
        ? gpuItems.reduce((a, s) => a + s.throughputRps, 0) / gpuItems.length
        : 0;
      return {
        batchSize,
        cpuThroughput: Math.round(cpuThroughput * 10) / 10,
        gpuThroughput: Math.round(gpuThroughput * 10) / 10,
      };
    })
    .filter(Boolean);

  const cpuAvg = store.filter((s) => s.computeMode === "cpu").reduce((a, s) => a + s.latencyMs, 0);
  const gpuAvg = store.filter((s) => s.computeMode === "gpu").reduce((a, s) => a + s.latencyMs, 0);
  const cpuCount = store.filter((s) => s.computeMode === "cpu").length;
  const gpuCount = store.filter((s) => s.computeMode === "gpu").length;
  const speedupFactor = cpuCount > 0 && gpuCount > 0
    ? Math.round((cpuAvg / cpuCount / (gpuAvg / gpuCount)) * 100) / 100
    : 0;

  return {
    latencyComparison,
    throughputComparison,
    speedupFactor,
  };
}

export function getHistory(limit = 50) {
  return {
    items: [...store]
      .reverse()
      .slice(0, limit)
      .map((item) => ({
        id: item.id,
        model: item.model,
        computeMode: item.computeMode,
        prediction: item.prediction,
        confidence: item.confidence,
        latencyMs: item.latencyMs,
        batchSize: item.batchSize,
        timestamp: item.timestamp.toISOString(),
      })),
    total: store.length,
  };
}

export function clearMetrics(): void {
  store.splice(0, store.length);
}
