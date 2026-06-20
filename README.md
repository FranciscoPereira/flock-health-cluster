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
