"""
Anomaly detection: Each node uses global federated model to detect outbreaks locally.
Reconstruction error > 3-sigma = CLUSTER DETECTED
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from autoencoder import RespiratoryAutoencoder, deserialize_model, get_device
from datetime import datetime


class AnomalyDetector:
    """Detects anomalies using federated model on local data."""

    def __init__(self, node_name: str, csv_path: str, global_model_weights: bytes, device=None):
        self.node_name = node_name
        self.csv_path = csv_path
        self.device = device or get_device()
        self._load_data()
        self._load_model(global_model_weights)

    def _load_data(self):
        """Load local test data."""
        df = pd.read_csv(self.csv_path)
        self.visits = df["respiratory_visits"].values.astype(np.float32)
        self.dates = df["day"].values

    def _load_model(self, global_model_weights: bytes):
        """Load global federated model."""
        self.model = deserialize_model(global_model_weights)
        self.model.to(self.device)
        self.model.eval()

    def compute_reconstruction_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute reconstruction error for each 7-day window.
        ERROR > 3-sigma = ANOMALY

        For valid anomaly detection, use pre-outbreak baseline (days 1-44)
        to compute threshold, then detect anomalies in all windows.
        """
        windows = []
        window_starts = []

        # Create 7-day windows
        for i in range(len(self.visits) - 6):
            window = self.visits[i:i+7]
            # Normalize
            window_norm = (window - self.visits.min()) / (self.visits.max() - self.visits.min() + 1e-6)
            windows.append(window_norm)
            window_starts.append(self.dates[i])

        windows = np.array(windows)
        window_starts = np.array(window_starts)

        # Compute errors
        errors = []
        with torch.no_grad():
            for window in windows:
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                reconstructed = self.model(x)
                error = torch.mean((x - reconstructed) ** 2).item()
                errors.append(error)

        return np.array(errors), window_starts

    def detect_anomalies(self, sigma_threshold: float = 2.5) -> dict:
        """
        Detect anomalies using 3-sigma rule.
        Use pre-outbreak baseline (days 1-44) to compute threshold.
        """
        errors, window_starts = self.compute_reconstruction_errors()

        # Use pre-outbreak windows (day 1-44 = window indices 0-36) for baseline
        baseline_cutoff = 37  # Last window before outbreak (day 44 = window 37)
        baseline_errors = errors[:baseline_cutoff]

        mean_error = np.mean(baseline_errors)
        std_error = np.std(baseline_errors)
        threshold = mean_error + sigma_threshold * std_error

        anomalies = []
        for i, error in enumerate(errors):
            if error > threshold:
                anomalies.append({
                    "window_day": int(window_starts[i]),
                    "error": error,
                    "threshold": threshold,
                    "z_score": (error - mean_error) / (std_error + 1e-6)
                })

        return {
            "node": self.node_name,
            "mean_error": mean_error,
            "std_error": std_error,
            "threshold": threshold,
            "anomalies": anomalies,
            "is_alert": len(anomalies) > 0,
            "anomaly_count": len(anomalies),
            "errors": errors,
            "window_starts": window_starts
        }


def run_anomaly_detection(global_model_weights: bytes) -> dict:
    """
    Run anomaly detection on all 3 nodes independently.
    Each node computes on its own data using global model.
    """
    device = get_device()
    print(f"\n{'='*70}")
    print("ANOMALY DETECTION: Each node scans locally for outbreaks")
    print(f"{'='*70}\n")

    node_configs = [
        ("node_a", "data/node_a.csv"),
        ("node_b", "data/node_b.csv"),
        ("node_c", "data/node_c.csv"),
    ]

    results = {}
    alerts = []

    for node_name, csv_path in node_configs:
        print(f"\n→ {node_name.upper()}: Scanning for anomalies (using global federated model)...")
        detector = AnomalyDetector(node_name, csv_path, global_model_weights, device)
        result = detector.detect_anomalies(sigma_threshold=3.0)
        results[node_name] = result

        # Status
        status = "🔴 ALERT" if result["is_alert"] else "🟢 NORMAL"
        print(f"  Status: {status}")
        print(f"  Anomalies detected: {result['anomaly_count']}")
        print(f"  Threshold: {result['threshold']:.4f}")

        if result["is_alert"]:
            for anom in result["anomalies"]:
                alert_msg = {
                    "timestamp": datetime.now().isoformat(),
                    "node": node_name,
                    "day": anom["window_day"],
                    "error": anom["error"],
                    "z_score": anom["z_score"],
                    "severity": "CRITICAL" if anom["z_score"] > 5 else "HIGH"
                }
                alerts.append(alert_msg)
                print(f"    └─ DAY {anom['window_day']:02d}: Error {anom['error']:.4f} (Z-score: {anom['z_score']:.2f})")

    print(f"\n{'='*70}")
    print("✓ ANOMALY DETECTION COMPLETE")
    print(f"{'='*70}\n")

    return {
        "node_results": results,
        "alerts": alerts,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Load model and run detection
    with open("models/global_model.pt", "rb") as f:
        global_weights = f.read()

    detection_results = run_anomaly_detection(global_weights)
    print(f"\n📊 Total alerts: {len(detection_results['alerts'])}")
