# GPU-Optimized Neural Network Inference Pipeline

A research-grade AI inference simulation with real image classification via Groq's free vision API.

---

## Local Setup

### 1. Get a free Groq API key
Sign up at **https://console.groq.com** → API Keys → Create key.

### 2. Create your `.env` file
```bash
cp .env.example .env
```
Open `.env` and replace `paste_your_groq_api_key_here` with your actual key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```
> The `.env` file lives at the **project root** (same folder as `artifacts/`). Both services load it automatically — you only need to set it once.

### 3. Install dependencies
```bash
# Python (Flask server)
cd artifacts/gpu-inference
pip install -r requirements.txt

# Node.js (API server)
cd artifacts/api-server
pnpm install
```

### 4. Start both servers
Open **two terminals** from the project root:

**Terminal 1 — Flask frontend**
```bash
cd artifacts/gpu-inference
python app.py
```

**Terminal 2 — Node.js API**
```bash
cd artifacts/api-server
pnpm run dev
```

Open **http://localhost:5000** in your browser.

---

## How it works

| Component | Role |
|-----------|------|
| `app.py` | Flask server · serves UI · upload + predict routes |
| `enhancement.py` | Calls Groq vision API to correctly identify uploaded images |
| `api-server/` | Node.js service · metrics · comparison data |
| `templates/` | Jinja2 HTML pages (Pipeline, Analytics, History) |

### Inference flow
1. Upload image → stored in `uploads/`
2. Python simulates GPU/CPU latency
3. `enhancement.py` sends image to Groq → returns the real label
4. Frontend shows prediction, confidence, top-5 classes, timing breakdown, system insights

### CPU vs GPU simulated latency
| Mode | Wall-clock delay |
|------|-----------------|
| GPU  | 10–30 ms |
| CPU  | 120–180 ms |

---

## On Replit
No `.env` file needed. Set `GROQ_API_KEY` in the **Secrets** panel — both services pick it up automatically.
