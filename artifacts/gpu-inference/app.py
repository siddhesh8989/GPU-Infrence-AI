"""
GPU-Optimized Neural Network Inference Pipeline
A Flask-based simulation of a research-level AI inference system.
"""

import os
import time
import uuid
import math
import random
import hashlib
import threading
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from enhancement import maybe_enhance

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── In-Memory Metrics Store ───────────────────────────────────────────────────

_metrics_lock = threading.Lock()
_inferences = []  # max 500 records

IMAGENET_CLASSES = [
    "golden retriever", "tabby cat", "African elephant", "bald eagle",
    "great white shark", "monarch butterfly", "red fox", "giant panda",
    "snow leopard", "hummingbird", "sunflower", "rose", "oak tree",
    "mushroom", "coral reef", "sports car", "bicycle", "sailboat",
    "helicopter", "steam locomotive", "pizza", "sushi", "hamburger",
    "apple", "banana", "laptop computer", "smartphone", "digital camera",
    "headphones", "microphone", "bookshelf", "armchair", "dining table",
    "bathtub", "grand piano", "basketball", "tennis racket", "soccer ball",
    "golf club", "surfboard", "lighthouse", "suspension bridge",
    "Eiffel Tower", "volcano", "waterfall", "German shepherd",
    "Siamese cat", "Bengal tiger", "blue whale", "flamingo",
]

RANDOM_FOREST_CLASSES = [
    "Class A: High-confidence feature cluster",
    "Class B: Medium-gradient pattern",
    "Class C: Low-frequency texture",
    "Class D: Complex multi-modal signal",
    "Class E: Sparse representation",
    "Class F: Dense clustering pattern",
]

# ─── Inference Engine ──────────────────────────────────────────────────────────

def _base_latency(model: str) -> float:
    return {"resnet50": 85.0, "mobilenet": 32.0, "random_forest": 18.0}.get(model, 60.0)


def _gpu_latency(cpu_ms: float, batch_size: int, precision: str) -> float:
    parallelism = math.log2(batch_size + 1) * 0.35
    prec_gain   = 0.15 if precision == "fp16" else 0.0
    speedup     = 0.55 + parallelism + prec_gain
    return max(cpu_ms * (1 - speedup), cpu_ms * 0.12)


def _top_predictions(model: str, seed: float) -> list:
    classes   = RANDOM_FOREST_CLASSES if model == "random_forest" else IMAGENET_CLASSES
    top_k     = 3 if model == "random_forest" else 5
    rng       = random.Random(int(seed))
    selected  = rng.sample(classes, min(top_k, len(classes)))
    remaining = 1.0
    preds     = []
    for i, label in enumerate(selected):
        is_last = i == len(selected) - 1
        conf    = remaining if is_last else remaining * (0.4 + rng.random() * 0.3) * (1 - i * 0.05)
        conf    = round(min(conf, remaining), 4)
        preds.append({"label": label, "confidence": conf, "rank": i + 1})
        remaining = round(remaining - conf, 4)
        if remaining <= 0:
            break
    return sorted(preds, key=lambda x: -x["confidence"])


def _memory_usage(compute_mode: str, model: str, batch_size: int) -> dict:
    import resource
    ru       = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb   = ru.ru_maxrss / 1024
    model_mb = {"resnet50": 97.8, "mobilenet": 13.5, "random_forest": 2.4}.get(model, 50.0)
    return {
        "heapUsedMb":     round(rss_mb * 0.6, 1),
        "heapTotalMb":    round(rss_mb * 0.8, 1),
        "rssMb":          round(rss_mb, 1),
        "gpuSimulatedMb": round((model_mb * 1.4 + batch_size * 2.1), 1) if compute_mode == "gpu" else 0.0,
    }


def _file_id_seed(file_id: str) -> int:
    return int(hashlib.md5(file_id.encode()).hexdigest()[:8], 16)


def run_inference(file_id: str, model: str, compute_mode: str,
                  batch_size: int, precision: str) -> dict:
    seed          = _file_id_seed(file_id)
    base_cpu      = _base_latency(model)
    jitter        = (random.random() - 0.5) * 10

    if compute_mode == "gpu":
        inference_ms  = _gpu_latency(base_cpu, batch_size, precision) + jitter
        preprocess_ms = round(3.5 + random.random() * 2, 1)
        batching_ms   = round((1.2 + random.random()), 1) if batch_size > 1 else 0.0
        postproc_ms   = round(1.5 + random.random(), 1)
    else:
        inference_ms  = base_cpu + jitter + batch_size * 4.2
        preprocess_ms = round(8.0 + random.random() * 4, 1)
        batching_ms   = round(2.8 + random.random(), 1) if batch_size > 1 else 0.0
        postproc_ms   = round(3.0 + random.random() * 2, 1)

    inference_ms = round(max(inference_ms, 2.0), 1)
    total_ms     = round(preprocess_ms + batching_ms + inference_ms + postproc_ms, 1)
    throughput   = round((1000 / total_ms) * batch_size, 1)

    time.sleep(min(total_ms / 1000.0, 0.5))

    top_preds = _top_predictions(model, seed)
    result = {
        "id":          str(uuid.uuid4()),
        "fileId":      file_id,
        "model":       model,
        "computeMode": compute_mode,
        "batchSize":   batch_size,
        "precision":   precision,
        "prediction":  top_preds[0]["label"] if top_preds else "unknown",
        "confidence":  top_preds[0]["confidence"] if top_preds else 0.0,
        "topPredictions": top_preds,
        "timings": {
            "preprocessingMs":  preprocess_ms,
            "batchingMs":       batching_ms,
            "inferenceMs":      inference_ms,
            "postprocessingMs": postproc_ms,
            "totalMs":          total_ms,
            "throughputRps":    throughput,
        },
        "memoryUsage": _memory_usage(compute_mode, model, batch_size),
        "pipelineStages": [
            {"name": "Input Loading",    "durationMs": round(preprocess_ms * 0.3, 1), "status": "complete"},
            {"name": "Preprocessing",   "durationMs": round(preprocess_ms * 0.7, 1), "status": "complete"},
            {"name": "Batch Scheduling","durationMs": batching_ms,                    "status": "complete"},
            {"name": "Model Inference", "durationMs": inference_ms,                   "status": "complete"},
            {"name": "Postprocessing",  "durationMs": postproc_ms,                    "status": "complete"},
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    with _metrics_lock:
        _inferences.append({
            "id":           result["id"],
            "model":        model,
            "computeMode":  compute_mode,
            "batchSize":    batch_size,
            "precision":    precision,
            "prediction":   result["prediction"],
            "confidence":   result["confidence"],
            "latencyMs":    total_ms,
            "throughputRps": throughput,
            "timestamp":    result["timestamp"],
        })
        if len(_inferences) > 500:
            _inferences.pop(0)

    return result


def run_batch_benchmark(model: str, batch_sizes: list,
                        compute_mode: str, precision: str) -> list:
    results = []
    base_cpu = _base_latency(model)
    model_mb = {"resnet50": 97.8, "mobilenet": 13.5, "random_forest": 2.4}.get(model, 50.0)
    for bs in batch_sizes:
        jitter = (random.random() - 0.5) * 5
        if compute_mode == "gpu":
            lat = round(max(_gpu_latency(base_cpu, bs, precision) + jitter, 2.0), 1)
            pre = 3.5
        else:
            lat = round(max(base_cpu + jitter + bs * 4.2, 2.0), 1)
            pre = 8.0
        total   = lat + pre
        thru    = round((1000 / total) * bs, 1)
        mem     = round(model_mb + bs * 1.8, 1)
        results.append({"batchSize": bs, "latencyMs": lat, "throughputRps": thru, "memoryMb": mem})
        time.sleep(0.03)
    return results


# ─── Metrics Helpers ───────────────────────────────────────────────────────────

def _metrics_summary() -> dict:
    with _metrics_lock:
        data = list(_inferences)
    if not data:
        return {"totalInferences": 0, "avgLatencyMs": 0, "avgThroughputRps": 0,
                "cpuInferences": 0, "gpuInferences": 0}
    cpu_items = [d for d in data if d["computeMode"] == "cpu"]
    gpu_items = [d for d in data if d["computeMode"] == "gpu"]
    return {
        "totalInferences":   len(data),
        "avgLatencyMs":      round(sum(d["latencyMs"] for d in data) / len(data), 1),
        "avgThroughputRps":  round(sum(d["throughputRps"] for d in data) / len(data), 1),
        "cpuInferences":     len(cpu_items),
        "gpuInferences":     len(gpu_items),
    }


def _model_breakdown() -> list:
    with _metrics_lock:
        data = list(_inferences)
    models = {}
    for d in data:
        m = d["model"]
        if m not in models:
            models[m] = {"count": 0, "latSum": 0.0, "confSum": 0.0}
        models[m]["count"]   += 1
        models[m]["latSum"]  += d["latencyMs"]
        models[m]["confSum"] += d["confidence"]
    return [
        {
            "model":        m,
            "count":        v["count"],
            "avgLatencyMs": round(v["latSum"] / v["count"], 1),
            "avgConfidence": round(v["confSum"] / v["count"], 4),
        }
        for m, v in models.items()
    ]


def _batch_performance() -> list:
    with _metrics_lock:
        data = list(_inferences)
    batches = {}
    for d in data:
        bs = d["batchSize"]
        if bs not in batches:
            batches[bs] = {"latSum": 0.0, "thrSum": 0.0, "count": 0}
        batches[bs]["latSum"]  += d["latencyMs"]
        batches[bs]["thrSum"]  += d["throughputRps"]
        batches[bs]["count"]   += 1
    return [
        {
            "batchSize":       bs,
            "avgLatencyMs":    round(v["latSum"] / v["count"], 1),
            "avgThroughputRps": round(v["thrSum"] / v["count"], 1),
        }
        for bs, v in sorted(batches.items())
    ]


def _comparison_data() -> dict:
    with _metrics_lock:
        data = list(_inferences)
    models_list = ["resnet50", "mobilenet", "random_forest"]
    lat_comp = []
    for m in models_list:
        cpu = [d for d in data if d["model"] == m and d["computeMode"] == "cpu"]
        gpu = [d for d in data if d["model"] == m and d["computeMode"] == "gpu"]
        if cpu and gpu:
            cpu_lat = sum(d["latencyMs"] for d in cpu) / len(cpu)
            gpu_lat = sum(d["latencyMs"] for d in gpu) / len(gpu)
            lat_comp.append({
                "model":       m,
                "cpuLatencyMs": round(cpu_lat, 1),
                "gpuLatencyMs": round(gpu_lat, 1),
                "improvement":  round((cpu_lat - gpu_lat) / cpu_lat * 100, 1),
            })

    thr_comp = []
    for bs in [1, 2, 4, 8, 16, 32]:
        cpu = [d for d in data if d["batchSize"] == bs and d["computeMode"] == "cpu"]
        gpu = [d for d in data if d["batchSize"] == bs and d["computeMode"] == "gpu"]
        if cpu or gpu:
            thr_comp.append({
                "batchSize":    bs,
                "cpuThroughput": round(sum(d["throughputRps"] for d in cpu) / len(cpu), 1) if cpu else 0,
                "gpuThroughput": round(sum(d["throughputRps"] for d in gpu) / len(gpu), 1) if gpu else 0,
            })

    cpu_items = [d for d in data if d["computeMode"] == "cpu"]
    gpu_items = [d for d in data if d["computeMode"] == "gpu"]
    if cpu_items and gpu_items:
        cpu_avg = sum(d["latencyMs"] for d in cpu_items) / len(cpu_items)
        gpu_avg = sum(d["latencyMs"] for d in gpu_items) / len(gpu_items)
        speedup = round(cpu_avg / gpu_avg, 2)
    else:
        speedup = 0

    return {"latencyComparison": lat_comp, "throughputComparison": thr_comp, "speedupFactor": speedup}


def _system_info() -> dict:
    import platform, os
    try:
        import psutil
        cpu_count  = psutil.cpu_count()
        total_mem  = round(psutil.virtual_memory().total / 1024 / 1024)
    except ImportError:
        cpu_count  = os.cpu_count() or 1
        total_mem  = 0
    return {
        "pythonVersion": platform.python_version(),
        "platform":      platform.system(),
        "cpuCores":      cpu_count,
        "totalMemoryMb": total_mem,
        "gpuAvailable":  False,
        "uptime":        round(time.time() - _start_time),
    }


_start_time = time.time()

# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def metrics_page():
    return render_template("metrics.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


# ─── API: Upload ──────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}
    if f.mimetype not in allowed:
        return jsonify({"error": "Only image files allowed"}), 400

    file_id  = str(uuid.uuid4())
    ext      = os.path.splitext(f.filename)[1] or ".jpg"
    filename = f"{file_id}{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)

    # Clean up uploads older than 1 hour
    _cleanup_uploads()

    return jsonify({
        "fileId":     file_id,
        "filename":   f.filename,
        "fileSize":   os.path.getsize(save_path),
        "dimensions": {"width": 224, "height": 224},
        "uploadedAt": datetime.utcnow().isoformat() + "Z",
    })


def _cleanup_uploads():
    now = time.time()
    folder = app.config["UPLOAD_FOLDER"]
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > 3600:
                os.remove(fpath)
        except Exception:
            pass


# ─── API: Predict ─────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    file_id      = data.get("fileId", "")
    model        = data.get("model", "resnet50")
    compute_mode = data.get("computeMode", "cpu")
    batch_size   = int(data.get("batchSize", 1))
    precision    = data.get("precision", "fp32")

    valid_models  = {"resnet50", "mobilenet", "random_forest"}
    valid_modes   = {"cpu", "gpu"}
    valid_prec    = {"fp32", "fp16"}
    valid_batches = {1, 2, 4, 8, 16, 32}

    if not file_id:
        return jsonify({"error": "fileId is required"}), 400
    if model not in valid_models:
        return jsonify({"error": f"model must be one of {sorted(valid_models)}"}), 400
    if compute_mode not in valid_modes:
        return jsonify({"error": "computeMode must be 'cpu' or 'gpu'"}), 400
    if batch_size not in valid_batches:
        return jsonify({"error": f"batchSize must be one of {sorted(valid_batches)}"}), 400
    if precision not in valid_prec:
        return jsonify({"error": "precision must be 'fp32' or 'fp16'"}), 400

    result = run_inference(file_id, model, compute_mode, batch_size, precision)

    # ── Hybrid Enhancement ────────────────────────────────────────────────────
    enhancement = maybe_enhance(
        uploads_folder=app.config["UPLOAD_FOLDER"],
        file_id=file_id,
        original_prediction=result["prediction"],
        original_confidence=result["confidence"],
        top_predictions=result["topPredictions"],
    )
    if enhancement["enhancementUsed"]:
        extra_ms = enhancement["extraPostprocMs"]
        result["prediction"]    = enhancement["prediction"]
        result["confidence"]    = enhancement["confidence"]
        result["topPredictions"] = enhancement["topPredictions"]
        # Fold extra latency into postprocessing to preserve pipeline metrics
        result["timings"]["postprocessingMs"] = round(
            result["timings"]["postprocessingMs"] + extra_ms, 1
        )
        result["timings"]["totalMs"] = round(result["timings"]["totalMs"] + extra_ms, 1)
        result["timings"]["throughputRps"] = round(
            (1000 / result["timings"]["totalMs"]) * batch_size, 1
        )
        # Update last pipeline stage to reflect adjusted time
        if result["pipelineStages"]:
            result["pipelineStages"][-1]["durationMs"] = round(
                result["pipelineStages"][-1]["durationMs"] + extra_ms, 1
            )
        result["_enhancementUsed"] = True
        result["_originalPrediction"] = enhancement["originalPrediction"]
        result["_originalConfidence"] = enhancement["originalConfidence"]
    else:
        result["_enhancementUsed"] = False

    # ── Debug store ───────────────────────────────────────────────────────────
    _debug_store[f"{file_id}:{model}"] = {
        "fileId":              file_id,
        "model":               model,
        "enhancementUsed":     result["_enhancementUsed"],
        "originalPrediction":  result.get("_originalPrediction", result["prediction"]),
        "originalConfidence":  result.get("_originalConfidence", result["confidence"]),
        "enhancedPrediction":  result["prediction"] if result["_enhancementUsed"] else None,
        "enhancedConfidence":  result["confidence"] if result["_enhancementUsed"] else None,
        "confidenceDelta":     round(
            result["confidence"] - result.get("_originalConfidence", result["confidence"]), 4
        ),
        "timestamp":           result["timestamp"],
    }

    return jsonify(result)


# ─── API: Batch Benchmark ─────────────────────────────────────────────────────

@app.route("/api/batch", methods=["POST"])
def api_batch():
    data = request.get_json(force=True) or {}
    model        = data.get("model", "resnet50")
    batch_sizes  = data.get("batchSizes", [1, 2, 4, 8, 16, 32])
    compute_mode = data.get("computeMode", "cpu")
    precision    = data.get("precision", "fp32")

    if not isinstance(batch_sizes, list) or not batch_sizes:
        return jsonify({"error": "batchSizes must be a non-empty list"}), 400

    results = run_batch_benchmark(model, batch_sizes, compute_mode, precision)
    best    = max(results, key=lambda r: r["throughputRps"])
    summary = {
        "bestBatchSize":  best["batchSize"],
        "peakThroughput": best["throughputRps"],
        "minLatency":     min(r["latencyMs"] for r in results),
        "avgMemoryMb":    round(sum(r["memoryMb"] for r in results) / len(results), 1),
    }
    return jsonify({"model": model, "computeMode": compute_mode,
                    "precision": precision, "results": results, "summary": summary})


# ─── API: Metrics ─────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    return jsonify({
        **_metrics_summary(),
        "modelBreakdown":  _model_breakdown(),
        "batchPerformance": _batch_performance(),
        "systemInfo":      _system_info(),
    })


@app.route("/api/metrics/compare")
def api_metrics_compare():
    return jsonify(_comparison_data())


@app.route("/api/metrics/history")
def api_metrics_history():
    with _metrics_lock:
        data = list(reversed(_inferences))[:100]
    return jsonify({"items": data, "total": len(_inferences)})


@app.route("/api/metrics/clear", methods=["DELETE"])
def api_metrics_clear():
    with _metrics_lock:
        _inferences.clear()
    return jsonify({"success": True, "message": "All metrics cleared"})


# ─── Debug: Prediction Detail ─────────────────────────────────────────────────

_debug_store: dict = {}  # fileId → {model, original, enhanced}

@app.route("/debug/prediction", methods=["GET"])
def debug_prediction():
    file_id = request.args.get("fileId", "")
    model   = request.args.get("model", "")
    if not file_id:
        return jsonify({"error": "fileId query parameter required"}), 400
    key = f"{file_id}:{model}"
    entry = _debug_store.get(key)
    if not entry:
        return jsonify({"error": "No debug data for this fileId/model combination"}), 404
    return jsonify(entry)


# ─── Health ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "uptime": round(time.time() - _start_time)})


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
