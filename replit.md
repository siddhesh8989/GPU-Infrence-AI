# GPU-Optimized Neural Network Inference Pipeline

## Architecture

Single Flask-based Python web application. No Node.js, no React, no Vite, no database.

- **Backend**: Flask (Python 3.11)
- **Frontend**: Jinja2 HTML templates + vanilla JS + Chart.js (CDN)
- **Storage**: Temporary file uploads only (`uploads/`), auto-cleaned after 1 hour
- **Port**: 18889 (Replit env) / 5000 (local)

## Project Structure

```
artifacts/gpu-inference/
├── app.py               # Flask application — all routes + inference engine
├── requirements.txt     # Python deps (flask, psutil)
├── README.md            # Local setup instructions
├── uploads/             # Temporary image storage (auto-created + auto-cleaned)
├── templates/
│   ├── base.html        # Shared navbar, layout, CDN imports
│   ├── index.html       # Pipeline dashboard
│   ├── metrics.html     # Analytics & charts
│   └── history.html     # Inference history log
└── static/
    ├── css/style.css    # Dark-theme stylesheet
    └── js/main.js       # Shared Chart.js defaults
```

## Features

- Image upload (drag-drop or click) with validation
- Inference simulation: ResNet-50, MobileNet V3, Random Forest
- CPU vs GPU mode with mathematically modelled speedup (5–7×)
- FP32 / FP16 precision modes
- Dynamic batching: 1, 2, 4, 8, 16, 32
- Animated 5-stage pipeline visualization
- Batch benchmark with Chart.js throughput graphs
- Analytics dashboard: CPU vs GPU comparison, model breakdown, system info
- Filterable inference history table
- In-memory metrics store (thread-safe, up to 500 records)

## Workflows

- `artifacts/gpu-inference: web` — runs `python app.py` (Flask on $PORT)

## Local Setup

See `artifacts/gpu-inference/README.md` for full instructions. Quick start:

```bash
cd artifacts/gpu-inference
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```
