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
- **Streamlit** - Interactive dashboard
- **Pandas/NumPy** - Data manipulation

## Files

1. **data_gen.py** - Generate 3 synthetic CSVs (60 days respiratory ED visits)
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

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Demo

```bash
# Generate data, train, detect, show results
python federated_train.py && python detect.py
```

### 3. Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Click "Run Full Pipeline" to see:
- Synthetic data generation
- Federated training across 3 nodes
- Anomaly detection results
- Regional status map
- Alert log

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
