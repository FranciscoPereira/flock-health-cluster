"""
Flask API server: bridges the federated ML pipeline with the Google Maps frontend.
Endpoints:
  GET  /                  → Serve index.html
  GET  /api/status        → Current node statuses + anomaly time-series
  POST /api/run-pipeline  → Trigger full federated train + detect cycle
  GET  /api/alerts        → Alert log (all fired alerts)
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, render_template, request

# Ensure project root is on the path so sibling modules resolve
sys.path.insert(0, os.path.dirname(__file__))

from data_gen import generate_synthetic_data
from federated_train import federated_training_loop
from detect import run_anomaly_detection

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Node metadata: lat/lng for Google Maps markers
# ---------------------------------------------------------------------------
NODE_META = {
    "node_a": {
        "label": "Node A – Wales",
        "region": "Wales",
        "lat": 51.4816,
        "lng": -3.1791,
        "color": "#00c853",  # green default
    },
    "node_b": {
        "label": "Node B – England",
        "region": "England",
        "lat": 51.5074,
        "lng": -0.1278,
        "color": "#00c853",
    },
    "node_c": {
        "label": "Node C – Scotland",
        "region": "Scotland",
        "lat": 55.9533,
        "lng": -3.1883,
        "color": "#00c853",
    },
}

# In-memory cache so the page can poll without re-running the pipeline
_cache: dict = {
    "status": "IDLE",      # IDLE | RUNNING | TRAINED
    "trained_at": None,
    "node_results": {},
    "alerts": [],
}


def _build_node_payload() -> list[dict]:
    """Convert raw detection results into a JSON-safe list for the frontend."""
    nodes = []
    for key, meta in NODE_META.items():
        result = _cache["node_results"].get(key, {})
        is_alert = result.get("is_alert", False)
        errors = result.get("errors", [])
        window_starts = result.get("window_starts", [])
        anomaly_count = result.get("anomaly_count", 0)
        threshold = result.get("threshold", 0)

        nodes.append({
            **meta,
            "key": key,
            "is_alert": is_alert,
            "color": "#e53935" if is_alert else "#00c853",
            "anomaly_count": anomaly_count,
            "threshold": round(threshold, 4),
            "time_series": [
                {"day": int(d), "error": round(float(e), 4)}
                for d, e in zip(window_starts, errors)
            ],
            "anomalies": result.get("anomalies", []),
        })
    return nodes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    return render_template("index.html", google_maps_api_key=api_key)


@app.route("/api/status")
def api_status():
    return jsonify({
        "pipeline_status": _cache["status"],
        "trained_at": _cache["trained_at"],
        "nodes": _build_node_payload(),
        "alert_count": len(_cache["alerts"]),
    })


@app.route("/api/alerts")
def api_alerts():
    return jsonify(_cache["alerts"])


@app.route("/api/run-pipeline", methods=["POST"])
def api_run_pipeline():
    """Trigger the full federated learning + anomaly detection pipeline."""
    if _cache["status"] == "RUNNING":
        return jsonify({"error": "Pipeline already running"}), 409

    _cache["status"] = "RUNNING"
    _cache["alerts"] = []
    _cache["node_results"] = {}

    try:
        # 1. Synthetic data
        generate_synthetic_data("data")

        # 2. Federated training
        Path("models").mkdir(exist_ok=True)
        global_weights = federated_training_loop(rounds=2)
        with open("models/global_model.pt", "wb") as f:
            f.write(global_weights)

        # 3. Anomaly detection
        detection = run_anomaly_detection(global_weights)

        # 4. Cache results (convert numpy types to plain Python)
        for key, res in detection["node_results"].items():
            _cache["node_results"][key] = {
                **res,
                "errors": [float(e) for e in res["errors"]],
                "window_starts": [int(d) for d in res["window_starts"]],
                "anomalies": [
                    {k: (float(v) if isinstance(v, float) else v)
                     for k, v in a.items()}
                    for a in res["anomalies"]
                ],
            }

        _cache["alerts"] = detection["alerts"]
        _cache["trained_at"] = datetime.now().isoformat()
        _cache["status"] = "TRAINED"

        return jsonify({"ok": True, "alert_count": len(_cache["alerts"])})

    except Exception as exc:
        _cache["status"] = "ERROR"
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Starting NHS Federated Disease Detection API…")
    print("Open http://localhost:5050 in your browser")
    app.run(host="0.0.0.0", port=5050, debug=False)
