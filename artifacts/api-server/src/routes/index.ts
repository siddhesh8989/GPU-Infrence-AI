import { Router, type IRouter } from "express";
import healthRouter from "./health.js";
import inferenceRouter from "./inference.js";
import metricsRouter from "./metrics.js";

const router: IRouter = Router();

router.use(healthRouter);
router.use("/inference", inferenceRouter);
router.use("/metrics", metricsRouter);

export default router;
