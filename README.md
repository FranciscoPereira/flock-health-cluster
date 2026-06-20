# 🏥 Federated Disease Cluster Detection - NHS Hackathon MVP

**Objective:** 3 simulated NHS nodes collaboratively train an anomaly detection model via **Flock.io's federated learning**, then each node locally detects a respiratory disease spike **without sharing raw patient data**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Federated Learning Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Node A (Wales)    Node B (England)    Node C (Scotland)       │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
│  │ Local Data │    │ Local Data │    │ Local Data │           │
│  │ (60 days)  │    │ + OUTBREAK │    │ (60 days)  │           │
│  └──────┬─────┘    └──────┬─────┘    └──────┬─────┘           │
│         │                 │                 │                  │
│         └─────────────────┼─────────────────┘                  │
│                           │                                    │
│                  ┌────────▼────────┐                           │
│                  │ Federated Train  │                          │
│                  │ (FedAvg)         │                          │
│                  │ Round 1, 2, ...  │                          │
│                  └────────┬────────┘                           │
│                           │                                    │
│              Only model weights shared                         │
│              Raw data STAYS LOCAL                              │
│                           │                                    │
│         ┌─────────────────┼─────────────────┐                 │
│         │                 │                 │                 │
│  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐         │
│  │ Detect: A   │   │ Detect: B   │   │ Detect: C   │         │
│  │ Normal ✓    │   │ ALERT ✗     │   │ Normal ✓    │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.10+**
- **Flock.io SDK** - Federated learning orchestration
- **PyTorch** - Autoencoder model
- **Flask** - REST API server for the web dashboard
- **Streamlit** - Alternative interactive dashboard
- **Pandas/NumPy** - Data manipulation
- **Leaflet.js / Google Maps** - Interactive map

## Running the Website Locally

The main interface is a Flask web app with a Google Maps/Leaflet dashboard. Follow these steps to run it on your machine.

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/FranciscoPereira/flock-health-cluster.git
cd flock-health-cluster
```

### 2. Install the Flock.io SDK

```bash
pip install git+https://github.com/FLock-io/v1-sdk.git
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add a Google Maps API key

The dashboard works out of the box using a free Leaflet/OpenStreetMap dark tile. If you have a Google Maps API key and want to use Google Maps instead:

```bash
export GOOGLE_MAPS_API_KEY=your_key_here
```

Get a free key at [console.cloud.google.com](https://console.cloud.google.com) — enable the **Maps JavaScript API**.

### 5. Start the web server

```bash
python3 api.py
```

You should see:

```
Starting NHS Federated Disease Detection API…
Open http://localhost:5050 in your browser
```

### 6. Open the dashboard

Go to **http://localhost:5050** in your browser.

### 7. Run the federated pipeline

Click the **▶ RUN FEDERATED PIPELINE** button. The dashboard will:

1. Generate synthetic respiratory ED visit data for 3 NHS nodes
2. Train a federated autoencoder across all 3 nodes (data stays local)
3. Run anomaly detection on each node independently
4. Update the map, charts, and alert log in real time

Expected result: **Node B – England** turns red with outbreak alerts on days 45–52. Nodes A and C remain green.

### Alternative: Streamlit dashboard

If you prefer Streamlit over the Flask web app:

```bash
streamlit run streamlit_app.py
```

Then click **Run Full Pipeline** in the sidebar.

### Alternative: Command-line only

```bash
./run_demo.sh
```

---

## Files

1. **api.py** - Flask REST API server (serves the web dashboard)
   - `GET /` — main dashboard page
   - `GET /api/status` — node statuses + anomaly time-series (JSON)
   - `POST /api/run-pipeline` — trigger federated train + detect
   - `GET /api/alerts` — full alert log (JSON)

2. **templates/index.html** - Google Maps / Leaflet dashboard
   - Dark HUD layout with live-updating map, charts, and alert log

3. **static/js/app.js** - Frontend logic
   - Map rendering, Chart.js anomaly graphs, 4-second API polling

4. **static/css/styles.css** - HUD dark theme

5. **data_gen.py** - Generate 3 synthetic CSVs (60 days respiratory ED visits)
   - Node A: Normal pattern
   - Node B: Normal + outbreak spike (days 45-52, 2x visits)
   - Node C: Normal pattern
   - **CRITICAL:** Data stays local on each node

2. **autoencoder.py** - PyTorch autoencoder for anomaly detection
   - Input: 7-day sliding windows of visit counts
   - Learns normal patterns → Anomalies = high reconstruction error

3. **federated_train.py** - Federated learning orchestration
   - Each node trains locally on its data
   - Only model weights exported (no data leakage)
   - FedAvg aggregation across nodes
   - 2 training rounds

4. **detect.py** - Local anomaly detection
   - Each node loads global federated model
   - Computes reconstruction error on test windows
   - Flags if error > 3-sigma threshold

5. **streamlit_app.py** - HUD-style dashboard
   - Regional status map (3 regions, color-coded)
   - Anomaly score line charts per node
   - Alert log (timestamp, node, severity)
   - System status display

6. **requirements.txt** - Dependencies

## Quick Start (TL;DR)

```bash
git clone https://github.com/FranciscoPereira/flock-health-cluster.git
cd flock-health-cluster
pip install git+https://github.com/FLock-io/v1-sdk.git
pip install -r requirements.txt
python3 api.py
# → open http://localhost:5050
```

## How Flock.io is Used

### The Flock.io SDK contract

The Flock.io v1 SDK defines three abstract methods that every federated model must implement:

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `train(parameters)` | Global model weights as `bytes` | Updated local weights as `bytes` | Node trains on its own data |
| `evaluate(parameters)` | Model weights as `bytes` | Accuracy as `float` | Node scores the current global model |
| `aggregate(parameters_list)` | List of local weight `bytes` | Aggregated global weights as `bytes` | Coordinator averages all node updates |

In a production Flock.io deployment, `FlockSDK` wraps your model in a Flask microservice. The Flock platform then orchestrates the round by posting to each node's `/call` endpoint:

```
POST /call  {"method": "train",     "parameters": "<base64 weights>"}
POST /call  {"method": "evaluate",  "parameters": "<base64 weights>"}
POST /call  {"method": "aggregate", "parameters_list": ["<b64>", "<b64>", "<b64>"]}
```

### How this project maps to the SDK

This MVP implements the same three-method contract directly in Python, simulating the network orchestration layer in-process so the demo runs on a single machine without requiring Flock.io credentials.

**`NodeLocalTrainer`** → implements `train()`

```python
# federated_train.py
def train_locally(self, model, epochs=3) -> RespiratoryAutoencoder:
    # Trains on LOCAL CSV only — mirrors FlockModel.train(parameters) -> bytes
    for window in self.data_tensor:
        loss = criterion(model(window), window)
        loss.backward()
        optimizer.step()
    return model  # weights serialised to bytes by get_model_weights()
```

**`FederatedAggregator`** → implements `aggregate()`

```python
# federated_train.py
def aggregate_weights(weights_list: list[bytes]) -> bytes:
    # FedAvg: average every parameter tensor across all nodes
    # Mirrors FlockModel.aggregate(parameters_list) -> bytes
    for key in avg_state.keys():
        avg_state[key] = sum(m[key] for m in models) / len(models)
    return serialize_model(global_model)
```

**`AnomalyDetector`** → implements `evaluate()`

```python
# detect.py
def detect_anomalies(self) -> dict:
    # Loads global model, scores each window, flags error > 2.5σ
    # Mirrors FlockModel.evaluate(parameters) -> float (here: anomaly score)
    errors = [mse(model(window), window) for window in all_windows]
```

### Weight serialisation (matches Flock.io wire format)

Flock.io transmits model weights as base64-encoded bytes over HTTP. This project uses the same bytes representation end-to-end:

```python
# autoencoder.py
def serialize_model(model) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)   # identical to Flock.io's encoding
    return buffer.getvalue()

def deserialize_model(data: bytes) -> RespiratoryAutoencoder:
    model.load_state_dict(torch.load(io.BytesIO(data)))
    return model
```

### Federated round lifecycle

Each federated round follows the exact sequence the Flock.io platform would orchestrate remotely:

```
Round N
  ├─ Broadcast global weights to all nodes
  ├─ Node A: train(global_weights) → local_weights_A   [local data only]
  ├─ Node B: train(global_weights) → local_weights_B   [local data only]
  ├─ Node C: train(global_weights) → local_weights_C   [local data only]
  └─ Aggregator: aggregate([A, B, C]) → new global_weights
```

### Upgrading to a real Flock.io deployment

To deploy each NHS node as a real Flock.io participant, wrap the model in `FlockSDK`:

```python
from flock_sdk import FlockSDK, FlockModel

class NHSRespiratoryModel(FlockModel):
    def init_dataset(self, dataset_path):
        # load node's local CSV
        ...
    def train(self, parameters: bytes) -> bytes:
        # same logic as NodeLocalTrainer.train_locally()
        ...
    def evaluate(self, parameters: bytes) -> float:
        # same logic as AnomalyDetector.detect_anomalies()
        ...
    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        # same logic as FederatedAggregator.aggregate_weights()
        ...

if __name__ == "__main__":
    model = NHSRespiratoryModel()
    sdk = FlockSDK(model)
    sdk.run()   # exposes POST /call on 0.0.0.0:5000
```

Each node then registers its `/call` URL with the Flock.io platform, and the platform drives all rounds automatically — no code changes needed beyond the `FlockModel` wrapper.

---

## Key Properties

### 🔐 Privacy-Preserving
- **Raw patient data never leaves any node**
- Only model weights are shared
- Each node independently computes anomaly detection

### 🤝 Federated Learning
- **FedAvg:** Average model weights from 3 nodes
- No central data repository
- Privacy-by-design architecture

### 🚨 Anomaly Detection
- **Threshold:** Reconstruction error > 3-sigma
- **Node B Alert:** Outbreak detected on days 45-52 (injected spike)
- **Nodes A & C:** Normal operation

### 📊 Audit Trail
- Timestamp, node ID, severity for each alert
- Full reconstruction error progression
- Z-score statistics for context

## Demo Output

```
======================================================================
FEDERATED LEARNING: 3 NHS Nodes (Data stays LOCAL)
======================================================================

--- FEDERATED ROUND 1/2 ---
→ node_a: Training locally...
    └─ node_a: Loaded 53 windows from data/node_a.csv
    └─ Training complete | Final loss: 0.0245
    └─ node_a: Weights exported (no data leaked)
...

======================================================================
ANOMALY DETECTION: Each node scans locally for outbreaks
======================================================================

→ NODE_A: Scanning for anomalies (using global federated model)...
  Status: 🟢 NORMAL
  Anomalies detected: 0

→ NODE_B: Scanning for anomalies (using global federated model)...
  Status: 🔴 ALERT
  Anomalies detected: 7
    └─ DAY 45: Error 0.1234 (Z-score: 3.45)
    └─ DAY 46: Error 0.1289 (Z-score: 3.67)
    ...

→ NODE_C: Scanning for anomalies (using global federated model)...
  Status: 🟢 NORMAL
  Anomalies detected: 0

======================================================================
✓ ANOMALY DETECTION COMPLETE
======================================================================
```

## Environment Variables

If using real Flock.io infrastructure:

```bash
export FLOCK_API_KEY=your_key
export FLOCK_API_URL=https://api.flock.io
```

For this MVP, Flock.io credentials are simulated locally.

## Future Enhancements

- [ ] Connect to real Flock.io API for multi-institution deployments
- [ ] Add differential privacy to model weights
- [ ] Support for other anomaly detection models (LSTM, Transformer)
- [ ] Real-time streaming data ingestion
- [ ] Multi-node horizontal scaling

## References

- [Flock.io Documentation](https://github.com/FLock-io/v1-sdk)
- [Federated Learning (McMahan et al.)](https://arxiv.org/abs/1602.05629)
- [Privacy-Preserving ML in Healthcare](https://arxiv.org/abs/2007.02282)
