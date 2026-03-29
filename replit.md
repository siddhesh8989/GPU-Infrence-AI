# GPU-Optimized Neural Network Inference Pipeline

## Architecture

Two-service system: Flask frontend + Node.js API backend.

- **Flask** (Python 3.11) — UI, image upload, inference engine, enhancement via Groq
- **Node.js** (Express) — metrics store, comparison data, history API
- **Frontend**: Jinja2 templates + vanilla JS + Chart.js (CDN)
- **AI**: Groq vision API (`meta-llama/llama-4-scout-17b-16e-instruct`) for image classification
- **Secret**: `GROQ_API_KEY` — set in Replit Secrets panel (or in `.env` for local use)

## Project Structure

```
.env.example                          # Copy to .env and add GROQ_API_KEY for local use

artifacts/gpu-inference/              # Flask server (port 18889 / 5000 local)
├── app.py                            # Routes + inference engine + dotenv loader
├── enhancement.py                    # Groq vision API integration (image → label)
├── requirements.txt                  # flask, psutil, python-dotenv
├── uploads/                          # Temp image storage (auto-cleaned after 1h)
├── templates/
│   ├── base.html                     # Shared navbar + CDN imports
│   ├── index.html                    # Pipeline dashboard
│   ├── metrics.html                  # Analytics + charts + model/dataset info
│   └── history.html                  # Inference history
└── static/
    ├── css/style.css                 # Dark-theme stylesheet
    └── js/main.js                    # Chart.js defaults

artifacts/api-server/                 # Node.js API (port 8080)
├── src/
│   ├── services/enhancementEngine.ts # Groq vision (Node-side, same key)
│   ├── services/inferenceEngine.ts   # GPU/CPU latency simulation
│   ├── services/metricsStore.ts      # In-memory metrics
│   └── routes/                       # Express route handlers
└── package.json                      # Uses --env-file-if-exists=../../.env
```

## Features

- Image upload (drag-drop) → Groq vision classifies it correctly
- CPU vs GPU mode: GPU 10–30ms wall-clock, CPU 120–180ms
- ResNet-50, MobileNet V3, Random Forest models
- FP32 / FP16 precision modes
- Dynamic batching: 1, 2, 4, 8, 16, 32
- Confidence & Reliability Panel (HIGH/MEDIUM/LOW with color coding)
- System Insights Panel (GPU util, memory, threads, cache hit rate)
- Analytics page: pre-populated demo charts, model info, dataset info, optimization techniques
- Filterable inference history table
- Category-aware related-class predictions (dog breeds, big cats, etc.)

## Workflows

- `artifacts/gpu-inference: web` — `python app.py`
- `artifacts/api-server: API Server` — `pnpm run dev` (builds + starts Node)

## Local Setup

```bash
# 1. Create .env at project root
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

# 2. Install deps
pip install -r artifacts/gpu-inference/requirements.txt
cd artifacts/api-server && pnpm install

# 3. Run both services
cd artifacts/gpu-inference && python app.py        # Terminal 1
cd artifacts/api-server && pnpm run dev            # Terminal 2
# → open http://localhost:5000
```

Both services load `../../.env` automatically. On Replit, use the Secrets panel instead.
