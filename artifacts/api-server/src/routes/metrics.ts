import { Router, type IRouter } from "express";
import {
  getMetricsSummary,
  getModelBreakdown,
  getBatchPerformance,
  getComparisonData,
  getHistory,
  clearMetrics,
} from "../services/metricsStore.js";
import { getSystemInfo } from "../services/inferenceEngine.js";

const router: IRouter = Router();

router.get("/", (_req, res) => {
  const summary = getMetricsSummary();
  const modelBreakdown = getModelBreakdown();
  const batchPerformance = getBatchPerformance();
  const systemInfo = getSystemInfo();

  res.json({
    ...summary,
    modelBreakdown,
    batchPerformance,
    systemInfo,
  });
});

router.get("/compare", (_req, res) => {
  const data = getComparisonData();
  res.json(data);
});

router.get("/history", (_req, res) => {
  const data = getHistory(100);
  res.json(data);
});

router.delete("/clear", (_req, res) => {
  clearMetrics();
  res.json({ success: true, message: "All metrics cleared" });
});

export default router;
