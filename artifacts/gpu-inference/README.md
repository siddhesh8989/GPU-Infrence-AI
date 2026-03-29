# GPU-Optimized Neural Network Inference Pipeline

A single-file Flask application that simulates a research-level AI inference pipeline with GPU/CPU mode switching, dynamic batching, precision control, and a full analytics dashboard — all served from one port with no external database.

---

## Features

| Feature | Description |
|---|---|
| **Image Upload** | Drag-and-drop or click-to-upload (JPG, PNG, GIF, WebP, max 10 MB) |
| **Model Selection** | ResNet-50, MobileNet V3, Random Forest |
| **Compute Mode** | CPU baseline vs GPU-accelerated simulation |
| **Dynamic Batching** | Batch sizes: 1, 2, 4, 8, 16, 32 |
| **Precision** | FP32 full-precision vs FP16 half-precision |
| **Pipeline Visualization** | Animated 5-stage pipeline: Input → Preprocess → Batch → Inference → Output |
| **Batch Benchmark** | One-click benchmark across all batch sizes with Chart.js graphs |
| **Analytics Dashboard** | CPU vs GPU latency bars, throughput scaling lines, model breakdown table |
| **Inference History** | Searchable & filterable log of all inferences |
| **Temporary Storage** | Uploaded files auto-cleaned after 1 hour; no database required |

---

## Quick Start (Local)

### Prerequisites

- Python 3.9 or later
- pip

### 1. Clone / download the project

```bash
git clone <repo-url>
cd flask-app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
python app.py
```

The server starts on **http://localhost:5000**

---

## Project Structure

```
flask-app/
├── app.py               # Flask application — all routes, inference engine, metrics store
├── requirements.txt     # Python dependencies (Flask, psutil)
├── README.md            # This file
├── uploads/             # Temporary image storage (auto-created, auto-cleaned)
├── templates/
│   ├── base.html        # Shared navbar, layout, Chart.js & font imports
│   ├── index.html       # Main pipeline dashboard page
│   ├── metrics.html     # Analytics & charts page
│   └── history.html     # Inference history log page
└── static/
    ├── css/
    │   └── style.css    # Full dark-theme stylesheet
    └── js/
        └── main.js      # Shared Chart.js defaults
```

---

## API Endpoints

All endpoints are available at `http://localhost:5000`.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Main pipeline dashboard |
| `GET`  | `/metrics` | Analytics dashboard |
| `GET`  | `/history` | Inference history |
| `POST` | `/api/upload` | Upload an image file (multipart/form-data) |
| `POST` | `/api/predict` | Run inference on an uploaded image |
| `POST` | `/api/batch` | Run batch benchmark across multiple batch sizes |
| `GET`  | `/api/metrics` | Get aggregated performance statistics |
| `GET`  | `/api/metrics/compare` | CPU vs GPU comparison data |
| `GET`  | `/api/metrics/history` | Recent inference history (last 100) |
| `DELETE` | `/api/metrics/clear` | Clear all stored metrics |
| `GET`  | `/api/health` | Health check |

### Example: Run Inference

```bash
# First upload an image
curl -F "file=@photo.jpg" http://localhost:5000/api/upload
# → {"fileId": "abc-123", ...}

# Then predict
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"fileId":"abc-123","model":"resnet50","computeMode":"gpu","batchSize":8,"precision":"fp16"}'
```

### Example: Batch Benchmark

```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"model":"resnet50","computeMode":"gpu","batchSizes":[1,2,4,8,16,32],"precision":"fp32"}'
```

---

## GPU Simulation Details

Since no physical GPU is required, the system simulates GPU acceleration using mathematically modelled speedup factors:

| Model | CPU Baseline | GPU Speedup |
|-------|-------------|-------------|
| ResNet-50 | ~85 ms | ~5–7× faster |
| MobileNet V3 | ~32 ms | ~5–7× faster |
| Random Forest | ~18 ms | ~4–5× faster |

Speedup is computed as:

```
gpu_latency = cpu_latency × (1 − (0.55 + log2(batch+1)×0.35 + fp16_bonus×0.15))
```

This realistically models:
- **Parallelism gain** — larger batches fill GPU compute units better
- **FP16 bonus** — half-precision reduces memory bandwidth requirements
- **Minimum floor** — GPU latency never drops below 12% of CPU latency

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | `5000` | Port to listen on |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | ≥ 3.0 | Web framework |
| `psutil` | ≥ 5.9 | System memory info for metrics |

No database, no ORM, no heavy ML libraries required.
