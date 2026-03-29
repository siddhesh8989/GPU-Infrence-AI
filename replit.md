# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Structure

```text
artifacts-monorepo/
├── artifacts/              # Deployable applications
│   ├── api-server/         # Express API server
│   └── gpu-inference/      # GPU Inference Pipeline frontend (React + Vite)
├── lib/                    # Shared libraries
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   └── db/                 # Drizzle ORM schema + DB connection
├── scripts/                # Utility scripts (single workspace package)
│   └── src/                # Individual .ts scripts, run via `pnpm --filter @workspace/scripts run <script>`
├── pnpm-workspace.yaml     # pnpm workspace (artifacts/*, lib/*, lib/integrations/*, scripts)
├── tsconfig.base.json      # Shared TS options (composite, bundler resolution, es2022)
├── tsconfig.json           # Root TS project references
└── package.json            # Root package with hoisted devDeps
```

## App: GPU-Optimized Neural Network Inference Pipeline

A full-stack research-level AI inference pipeline dashboard.

### Features
- **Image upload** → inference with ResNet-50, MobileNet, or Random Forest
- **CPU vs GPU simulated inference** with realistic latency differences
- **Dynamic batching** (batch sizes 1, 2, 4, 8, 16, 32) with throughput tracking
- **FP32 vs FP16 precision** simulation (FP16 adds ~15% GPU speedup)
- **Animated pipeline visualization**: Input Buffer → Pre-process → Batching → Inference Engine → Output
- **Analytics dashboard** with Recharts: CPU vs GPU latency bars, throughput vs batch size line charts
- **Inference history table** with filtering
- **Real-time metrics** auto-refreshed every 5 seconds

### API Endpoints
- `POST /api/inference/upload` — Upload image
- `POST /api/inference/predict` — Run inference (CPU or GPU mode)
- `POST /api/inference/batch` — Run batch benchmark
- `GET /api/metrics` — Aggregated stats
- `GET /api/metrics/compare` — CPU vs GPU comparison
- `GET /api/metrics/history` — Inference history
- `DELETE /api/metrics/clear` — Reset metrics

### Performance Simulation
- ResNet-50 CPU baseline: ~85ms | GPU: ~10-20ms (6x speedup)
- MobileNet CPU: ~32ms | GPU: ~4-8ms
- Random Forest CPU: ~18ms | GPU: ~3-5ms
- Metrics stored in-memory (up to 500 records); DB table `inferences` available for persistence

## TypeScript & Composite Projects

Every package extends `tsconfig.base.json` which sets `composite: true`. The root `tsconfig.json` lists all packages as project references. This means:

- **Always typecheck from the root** — run `pnpm run typecheck` (which runs `tsc --build --emitDeclarationOnly`). This builds the full dependency graph so that cross-package imports resolve correctly. Running `tsc` inside a single package will fail if its dependencies haven't been built yet.
- **`emitDeclarationOnly`** — we only emit `.d.ts` files during typecheck; actual JS bundling is handled by esbuild/tsx/vite...etc, not `tsc`.
- **Project references** — when package A depends on package B, A's `tsconfig.json` must list B in its `references` array. `tsc --build` uses this to determine build order and skip up-to-date packages.

## Root Scripts

- `pnpm run build` — runs `typecheck` first, then recursively runs `build` in all packages that define it
- `pnpm run typecheck` — runs `tsc --build --emitDeclarationOnly` using project references

## Packages

### `artifacts/api-server` (`@workspace/api-server`)

Express 5 API server. Routes in `src/routes/`, services in `src/services/`.

- Routes: `/api/healthz`, `/api/inference/*`, `/api/metrics/*`
- Services: `inferenceEngine.ts` (GPU simulation), `metricsStore.ts` (in-memory metrics)
- Dependencies: multer (file upload), express, cors, pino
- `pnpm --filter @workspace/api-server run dev` — run the dev server

### `artifacts/gpu-inference` (`@workspace/gpu-inference`)

React + Vite frontend dashboard.

- Pages: `dashboard.tsx` (main), `metrics.tsx` (analytics), `history.tsx` (log)
- Components: `layout.tsx`, `pipeline-visualization.tsx`, `file-upload.tsx`
- Charts: Recharts (BarChart, LineChart)
- Animations: framer-motion

### `lib/db` (`@workspace/db`)

Database layer using Drizzle ORM. Schema: `inferences` table.

### `lib/api-spec` (`@workspace/api-spec`)

OpenAPI 3.1 spec. Run codegen: `pnpm --filter @workspace/api-spec run codegen`
